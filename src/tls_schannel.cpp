// The TLS engine on SChannel, Windows' own. See tls.hpp for the shape.
//
// SSPI was always buffer-driven -- `InitializeSecurityContext` and
// `AcceptSecurityContext` take the bytes that arrived and hand back the bytes
// to send, and `DecryptMessage` and `EncryptMessage` work in place on a
// buffer -- so the engine is this file's bookkeeping around those four calls:
// `in_` is ciphertext received and not yet consumed, `out_` ciphertext to
// send, `plain_` decrypted text not yet read. Anything SChannel did not use
// comes back as a SECBUFFER_EXTRA and is kept for the next call.
//
// **Certificates are checked here, not by SChannel.** A client asks for
// manual validation and builds the chain itself with CryptoAPI, because that
// is the only way to honour `Config::ca_pem` the way the other backends do: a
// certificate is trusted when its chain ends at one of *those* roots, and the
// system's store is consulted only when no `ca` was given. It also means a
// failure is classified from the chain's own status bits -- untrusted, expired,
// misnamed -- into the kinds tls.hpp promises, rather than from whichever
// SEC_E_ code SChannel would have chosen.
//
// **An identity's key is persisted, then deleted.** SChannel does its
// cryptography in LSASS, which cannot reach a key that exists only in this
// process, so a PKCS#12 bundle imported for a server (or a client offering a
// certificate) has to go into the user's key store. The import is kept for
// the life of the VM -- one per distinct bundle, shared by every connection
// that uses it -- and the key is deleted from the store when the VM exits.
// A VM that is killed cannot do that, so each key is also written down, in
// `%LOCALAPPDATA%\dream\tls-keys`, under the name of the process that made
// it; the first import in any later VM deletes the keys of processes that
// are gone, and their entries with them. A key outlives a killed VM only
// until the next VM to use an identity.
//
// TLS 1.3 needs `SCH_CREDENTIALS`, which Windows understands from 10 1809;
// on anything older the credentials fall back to `SCHANNEL_CRED` and TLS 1.2.

#include "tls.hpp"

#define SECURITY_WIN32
#include "windows.hpp"

#include <wincrypt.h>
#include <ncrypt.h>
// `SCH_CREDENTIALS` is declared only for a program that says it knows the
// newer structure; it needs UNICODE_STRING from subauth.h.
#define SCHANNEL_USE_BLACKLISTS
#include <subauth.h>
#include <schannel.h>
#include <security.h>
#include <sspi.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <cstring>
#include <map>
#include <mutex>

// The SSL policy's "ignore" flags, which live in wininet.h -- a header for an
// HTTP client this file has no other use for.
#ifndef SECURITY_FLAG_IGNORE_UNKNOWN_CA
#define SECURITY_FLAG_IGNORE_UNKNOWN_CA 0x00000100
#endif
#ifndef SECURITY_FLAG_IGNORE_REVOCATION
#define SECURITY_FLAG_IGNORE_REVOCATION 0x00000080
#endif
#ifndef SECURITY_FLAG_IGNORE_CERT_DATE_INVALID
#define SECURITY_FLAG_IGNORE_CERT_DATE_INVALID 0x00002000
#endif

namespace dream::tls {

namespace {

std::string status_text(SECURITY_STATUS s) {
    char hex[16];
    std::snprintf(hex, sizeof hex, "0x%08lx", static_cast<unsigned long>(s));
    return windows::error(DWORD(s)) + " (" + hex + ")";
}

/// The certificates in a PEM text, in a store of their own.
HCERTSTORE pem_store(const std::string& pem, int* count) {
    HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, 0, CERT_STORE_CREATE_NEW_FLAG, nullptr);
    *count = 0;
    if (!store) return nullptr;
    for (const std::string& der : pem_certificates(pem)) {
        if (CertAddEncodedCertificateToStore(store, X509_ASN_ENCODING,
                                             reinterpret_cast<const BYTE*>(der.data()), DWORD(der.size()),
                                             CERT_STORE_ADD_ALWAYS, nullptr)) {
            ++*count;
        }
    }
    return store;
}

/// The revocation lists in a PEM text, in a store of their own.
HCERTSTORE crl_store(const std::string& pem, int* count) {
    HCERTSTORE store = CertOpenStore(CERT_STORE_PROV_MEMORY, 0, 0, CERT_STORE_CREATE_NEW_FLAG, nullptr);
    *count = 0;
    if (!store) return nullptr;
    for (const std::string& der : pem_blocks(pem, "X509 CRL")) {
        if (CertAddEncodedCRLToStore(store, X509_ASN_ENCODING, reinterpret_cast<const BYTE*>(der.data()),
                                     DWORD(der.size()), CERT_STORE_ADD_ALWAYS, nullptr)) {
            ++*count;
        }
    }
    return store;
}

/// Delete a persisted key, named as CERT_KEY_PROV_INFO_PROP_ID names it: by
/// provider and container, CNG when the provider type is 0 and CryptoAPI
/// otherwise.
void delete_key(const std::wstring& provider, const std::wstring& container, DWORD type, DWORD flags) {
    if (type == 0) {
        NCRYPT_PROV_HANDLE prov = 0;
        if (NCryptOpenStorageProvider(&prov, provider.c_str(), 0) != ERROR_SUCCESS) return;
        NCRYPT_KEY_HANDLE key = 0;
        if (NCryptOpenKey(prov, &key, container.c_str(), 0, flags & NCRYPT_MACHINE_KEY_FLAG) == ERROR_SUCCESS) {
            NCryptDeleteKey(key, 0);  // frees the handle too
        }
        NCryptFreeObject(prov);
    } else {
        HCRYPTPROV prov = 0;
        CryptAcquireContextW(&prov, container.c_str(), provider.c_str(), type,
                             CRYPT_DELETEKEYSET | (flags & CRYPT_MACHINE_KEYSET));
    }
}

// --- the journal of persisted keys --------------------------------------------
//
// One file per key, named `PID-START-N.key` -- the process, when it started
// (so a pid Windows has since handed to someone else is not mistaken for the
// owner), and a counter -- holding four lines: the provider type, the flags,
// the provider and the container, as UTF-8.

/// The journal's directory, made if it is not there; empty if it cannot be.
std::wstring journal_dir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    std::wstring base;
    if (n > 0 && n < MAX_PATH) {
        base.assign(buf, n);
    } else {
        n = GetTempPathW(MAX_PATH, buf);
        if (n == 0 || n >= MAX_PATH) return {};
        base.assign(buf, n);
    }
    if (base.back() != L'\\' && base.back() != L'/') base += L'\\';
    std::wstring dir = base + L"dream";
    CreateDirectoryW(dir.c_str(), nullptr);
    dir += L"\\tls-keys";
    CreateDirectoryW(dir.c_str(), nullptr);
    DWORD attrs = GetFileAttributesW(dir.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_DIRECTORY)) return {};
    return dir;
}

/// When a process started, as a FILETIME's 64 bits; 0 for a process that has
/// exited or cannot be asked.
unsigned long long started(DWORD pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return 0;
    FILETIME created{}, exited{}, kernel{}, user{};
    DWORD code = 0;
    unsigned long long t = 0;
    if (GetProcessTimes(h, &created, &exited, &kernel, &user) && GetExitCodeProcess(h, &code)
        && code == STILL_ACTIVE) {
        t = (static_cast<unsigned long long>(created.dwHighDateTime) << 32) | created.dwLowDateTime;
    }
    CloseHandle(h);
    return t;
}

bool read_file(const std::wstring& path, std::string* out) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    char buf[4096];
    DWORD got = 0;
    out->clear();
    while (ReadFile(h, buf, sizeof buf, &got, nullptr) && got > 0) out->append(buf, got);
    CloseHandle(h);
    return true;
}

bool write_file(const std::wstring& path, const std::string& text) {
    HANDLE h = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, 0, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD put = 0;
    bool ok = WriteFile(h, text.data(), DWORD(text.size()), &put, nullptr) && put == text.size();
    CloseHandle(h);
    return ok;
}

/// Delete the keys, and the entries, of every process in the journal that is
/// no longer running. Once per VM, at its first import.
void sweep_journal(const std::wstring& dir) {
    WIN32_FIND_DATAW found;
    HANDLE find = FindFirstFileW((dir + L"\\*.key").c_str(), &found);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        unsigned long pid = 0;
        unsigned long long start = 0;
        if (std::swscanf(found.cFileName, L"%lu-%llu-", &pid, &start) != 2) continue;
        if (start != 0 && started(DWORD(pid)) == start) continue;  // its owner is alive
        std::wstring path = dir + L"\\" + found.cFileName;
        std::string text;
        if (!read_file(path, &text)) continue;
        std::vector<std::string> lines;
        size_t at = 0;
        for (size_t nl; (nl = text.find('\n', at)) != std::string::npos; at = nl + 1) {
            lines.push_back(text.substr(at, nl - at));
        }
        if (lines.size() == 4) {
            delete_key(windows::wide(lines[2]), windows::wide(lines[3]), DWORD(std::strtoul(lines[0].c_str(), nullptr, 10)),
                       DWORD(std::strtoul(lines[1].c_str(), nullptr, 10)));
        }
        DeleteFileW(path.c_str());
    } while (FindNextFileW(find, &found));
    FindClose(find);
}

/// The journal, swept of the dead the first time it is asked for -- which is
/// before this VM's first import, so that nothing it imports is mistaken for
/// a dead process's key that happens to share a container's name.
const std::wstring& journal() {
    static const std::wstring dir = [] {
        std::wstring d = journal_dir();
        if (!d.empty()) sweep_journal(d);
        return d;
    }();
    return dir;
}

/// Write down a key this VM persisted; answers the entry's path, or empty
/// when there is nowhere to write it (the key is still deleted at exit).
std::wstring journal_key(const CRYPT_KEY_PROV_INFO& info) {
    const std::wstring& dir = journal();
    if (dir.empty()) return {};
    static std::atomic<unsigned> counter{0};
    DWORD pid = GetCurrentProcessId();
    wchar_t name[96];
    std::swprintf(name, 96, L"\\%lu-%llu-%u.key", static_cast<unsigned long>(pid), started(pid), counter++);
    std::wstring path = dir + name;
    std::string text = std::to_string(info.dwProvType) + "\n" + std::to_string(info.dwFlags) + "\n"
                       + windows::utf8(info.pwszProvName ? info.pwszProvName : L"") + "\n"
                       + windows::utf8(info.pwszContainerName ? info.pwszContainerName : L"") + "\n";
    return write_file(path, text) ? path : std::wstring{};
}

/// The key provider property of a certificate, in a buffer of its own.
std::vector<BYTE> key_prov_info(PCCERT_CONTEXT cert) {
    DWORD size = 0;
    if (!CertGetCertificateContextProperty(cert, CERT_KEY_PROV_INFO_PROP_ID, nullptr, &size)) return {};
    std::vector<BYTE> buf(size);
    if (!CertGetCertificateContextProperty(cert, CERT_KEY_PROV_INFO_PROP_ID, buf.data(), &size)) return {};
    return buf;
}

/// A PKCS#12 bundle imported: the store it went into, the certificate whose
/// private key came with it, and the journal entry for that key.
struct Identity {
    HCERTSTORE store = nullptr;
    PCCERT_CONTEXT cert = nullptr;
    std::wstring journal;

    ~Identity() {
        if (cert) {
            forget_key();
            CertFreeCertificateContext(cert);
        }
        if (store) CertCloseStore(store, 0);
        if (!journal.empty()) DeleteFileW(journal.c_str());
    }

    /// Delete the key the import persisted. See the head of this file.
    void forget_key() {
        std::vector<BYTE> buf = key_prov_info(cert);
        if (buf.empty()) return;
        auto* info = reinterpret_cast<CRYPT_KEY_PROV_INFO*>(buf.data());
        delete_key(info->pwszProvName ? info->pwszProvName : L"",
                   info->pwszContainerName ? info->pwszContainerName : L"", info->dwProvType, info->dwFlags);
    }
};

std::shared_ptr<Identity> import_identity(const std::string& p12, const std::string& password, Failure* why) {
    CRYPT_DATA_BLOB blob{DWORD(p12.size()), reinterpret_cast<BYTE*>(const_cast<char*>(p12.data()))};
    if (!PFXIsPFXBlob(&blob)) {
        *why = {"tls_config", "`identity` is not a PKCS#12 bundle"};
        return nullptr;
    }
    std::wstring wpassword = windows::wide(password);
    journal();
    auto id = std::make_shared<Identity>();
    id->store = PFXImportCertStore(&blob, wpassword.c_str(), CRYPT_USER_KEYSET);
    if (!id->store) {
        *why = {"tls_config", "cannot open the PKCS#12 identity (is the password right?): " + windows::error()};
        return nullptr;
    }
    // The certificate whose key came with it, found by the key's provider
    // property rather than CERT_FIND_HAS_PRIVATE_KEY, which is Windows 8's.
    // That key is the one written down; a bundle carries one.
    PCCERT_CONTEXT c = nullptr;
    while ((c = CertEnumCertificatesInStore(id->store, c)) != nullptr) {
        std::vector<BYTE> info = key_prov_info(c);
        if (info.empty()) continue;
        if (!id->cert) {
            id->cert = CertDuplicateCertificateContext(c);
            id->journal = journal_key(*reinterpret_cast<CRYPT_KEY_PROV_INFO*>(info.data()));
        }
    }
    if (!id->cert) {
        *why = {"tls_config", "the PKCS#12 identity holds no certificate with its private key"};
        return nullptr;
    }
    return id;
}

/// Credentials: what `AcquireCredentialsHandle` made, and the identity they
/// present, which must outlive them.
struct Credentials {
    CredHandle handle{};
    std::shared_ptr<Identity> identity;
    ~Credentials() { FreeCredentialsHandle(&handle); }
};

std::shared_ptr<Credentials> acquire(bool server, std::shared_ptr<Identity> identity, Failure* why) {
    auto creds = std::make_shared<Credentials>();
    creds->identity = std::move(identity);
    PCCERT_CONTEXT certs[1] = {creds->identity ? creds->identity->cert : nullptr};
    DWORD count = creds->identity ? 1 : 0;
    DWORD flags = SCH_USE_STRONG_CRYPTO;
    if (!server) flags |= SCH_CRED_MANUAL_CRED_VALIDATION | SCH_CRED_NO_DEFAULT_CREDS;
    ULONG use = server ? SECPKG_CRED_INBOUND : SECPKG_CRED_OUTBOUND;

    // TLS 1.2 and 1.3, nothing older.
    TLS_PARAMETERS params{};
    params.grbitDisabledProtocols = ~DWORD(server ? (SP_PROT_TLS1_2_SERVER | SP_PROT_TLS1_3_SERVER)
                                                  : (SP_PROT_TLS1_2_CLIENT | SP_PROT_TLS1_3_CLIENT));
    SCH_CREDENTIALS modern{};
    modern.dwVersion = SCH_CREDENTIALS_VERSION;
    modern.cCreds = count;
    modern.paCred = count ? certs : nullptr;
    modern.dwFlags = flags;
    modern.cTlsParameters = 1;
    modern.pTlsParameters = &params;
    SECURITY_STATUS s = AcquireCredentialsHandleW(nullptr, const_cast<SEC_WCHAR*>(UNISP_NAME_W), use, nullptr,
                                                  &modern, nullptr, nullptr, &creds->handle, nullptr);
    if (s != SEC_E_OK) {
        SCHANNEL_CRED legacy{};
        legacy.dwVersion = SCHANNEL_CRED_VERSION;
        legacy.cCreds = count;
        legacy.paCred = count ? certs : nullptr;
        legacy.grbitEnabledProtocols = server ? SP_PROT_TLS1_2_SERVER : SP_PROT_TLS1_2_CLIENT;
        legacy.dwFlags = flags;
        s = AcquireCredentialsHandleW(nullptr, const_cast<SEC_WCHAR*>(UNISP_NAME_W), use, nullptr,
                                      &legacy, nullptr, nullptr, &creds->handle, nullptr);
    }
    if (s != SEC_E_OK) {
        // Nothing to free: the handle was never made.
        creds->handle = CredHandle{};
        *why = {"tls_config", "SChannel refused the credentials: " + status_text(s)};
        return nullptr;
    }
    return creds;
}

/// Credentials for `config`, shared by every connection that would make the
/// same ones -- above all so a bundle is imported, and its key persisted, once.
std::shared_ptr<Credentials> credentials(const Config& config, Failure* why) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<Credentials>> cache;
    std::string key = std::string(config.server ? "s" : "c") + '\0' + config.identity_password + '\0'
                      + config.identity_p12;
    std::lock_guard<std::mutex> g(mutex);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    std::shared_ptr<Identity> identity;
    if (!config.identity_p12.empty()) {
        identity = import_identity(config.identity_p12, config.identity_password, why);
        if (!identity) return nullptr;
    }
    std::shared_ptr<Credentials> creds = acquire(config.server, identity, why);
    if (creds) cache[key] = creds;
    return creds;
}

/// The bytes of an IP address written as text, or empty when it is not one.
std::string address_bytes(const std::string& host) {
    unsigned char buf[16];
    if (InetPtonA(AF_INET, host.c_str(), buf) == 1) return std::string(reinterpret_cast<char*>(buf), 4);
    if (InetPtonA(AF_INET6, host.c_str(), buf) == 1) return std::string(reinterpret_cast<char*>(buf), 16);
    return {};
}

/// Whether the certificate names the address `ip` among its alternative names.
bool names_address(PCCERT_CONTEXT cert, const std::string& ip) {
    PCERT_EXTENSION ext = CertFindExtension(szOID_SUBJECT_ALT_NAME2, cert->pCertInfo->cExtension,
                                            cert->pCertInfo->rgExtension);
    if (!ext) return false;
    CERT_ALT_NAME_INFO* names = nullptr;
    DWORD size = 0;
    if (!CryptDecodeObjectEx(X509_ASN_ENCODING, X509_ALTERNATE_NAME, ext->Value.pbData, ext->Value.cbData,
                             CRYPT_DECODE_ALLOC_FLAG, nullptr, &names, &size)) {
        return false;
    }
    bool found = false;
    for (DWORD i = 0; i < names->cAltEntry && !found; ++i) {
        const CERT_ALT_NAME_ENTRY& e = names->rgAltEntry[i];
        found = e.dwAltNameChoice == CERT_ALT_NAME_IP_ADDRESS && e.IPAddress.cbData == ip.size()
                && std::memcmp(e.IPAddress.pbData, ip.data(), ip.size()) == 0;
    }
    LocalFree(names);
    return found;
}

std::string subject_of(PCCERT_CONTEXT cert) {
    DWORD flags = CERT_X500_NAME_STR | CERT_NAME_STR_REVERSE_FLAG;
    CERT_NAME_BLOB* name = &cert->pCertInfo->Subject;
    DWORD n = CertNameToStrW(X509_ASN_ENCODING, name, flags, nullptr, 0);
    if (n <= 1) return {};
    std::wstring text(n, L'\0');
    CertNameToStrW(X509_ASN_ENCODING, name, flags, text.data(), n);
    text.resize(n - 1);
    return windows::utf8(text);
}

class SChannelEngine : public Engine {
public:
    bool init(const Config& config, Failure* why) {
        server_ = config.server;
        verify_ = config.verify;
        check_name_ = config.check_name;
        host_ = config.host;
        whost_ = windows::wide(config.host);
        if (!config.ca_pem.empty()) {
            int count = 0;
            ca_ = pem_store(config.ca_pem, &count);
            if (count == 0) {
                *why = {"tls_config", "`ca` holds no PEM certificate"};
                return false;
            }
        }
        if (!config.crl_pem.empty()) {
            int count = 0;
            crl_ = crl_store(config.crl_pem, &count);
            if (count == 0) {
                *why = {"tls_config", "`crl` holds no PEM certificate revocation list"};
                return false;
            }
        }
        if (server_ && verify_ && !ca_) verify_ = false;  // nothing to check a client against
        creds_ = credentials(config, why);
        if (!creds_) return false;

        // The ALPN list as SSPI wants it: a SEC_APPLICATION_PROTOCOLS of one
        // list, built by hand because it is a header followed by bytes.
        std::string wire;
        for (const std::string& n : config.alpn) {
            if (n.empty() || n.size() > 255) continue;
            wire.push_back(char(n.size()));
            wire += n;
        }
        if (!wire.empty()) {
            uint32_t lists = uint32_t(4 + 2 + wire.size());
            uint32_t ext = SecApplicationProtocolNegotiationExt_ALPN;
            uint16_t size = uint16_t(wire.size());
            alpn_.append(reinterpret_cast<const char*>(&lists), 4);
            alpn_.append(reinterpret_cast<const char*>(&ext), 4);
            alpn_.append(reinterpret_cast<const char*>(&size), 2);
            alpn_ += wire;
        }
        return true;
    }

    ~SChannelEngine() override {
        if (have_ctx_) DeleteSecurityContext(&ctx_);
        if (ca_) CertCloseStore(ca_, 0);
        if (crl_) CertCloseStore(crl_, 0);
    }

    void feed(const char* data, size_t len) override { in_.append(data, len); }
    void feed_eof() override { eof_ = true; }
    std::string take_output() override {
        std::string out;
        out.swap(out_);
        return out;
    }

    Status handshake() override {
        for (;;) {
            if (server_ && !have_ctx_ && in_.empty()) return Status::WantRead;
            if (!server_ && have_ctx_ && in_.empty()) return Status::WantRead;
            SECURITY_STATUS s = step(true);
            if (s == SEC_E_INCOMPLETE_MESSAGE) return Status::WantRead;
            if (s == SEC_I_INCOMPLETE_CREDENTIALS) {
                // The server asked for a certificate and this client has none
                // to offer: carry on without, and let the server decide.
                continue;
            }
            if (s == SEC_I_CONTINUE_NEEDED) {
                if (in_.empty()) return Status::WantRead;
                continue;
            }
            if (s != SEC_E_OK) return refused(s);
            if (QueryContextAttributesW(&ctx_, SECPKG_ATTR_STREAM_SIZES, &sizes_) != SEC_E_OK) {
                return fail("handshake_failed", "SChannel did not report its record sizes");
            }
            Status checked = check_peer();
            if (checked != Status::Ok) {
                // SChannel thinks the handshake succeeded -- the check was
                // ours, made after it -- so the peer has to be told, or it
                // waits for a conversation that is never coming.
                alert(failure_.kind);
                return checked;
            }
            done_ = true;
            return Status::Ok;
        }
    }

    Status read(char* out, size_t cap, size_t* got) override {
        *got = 0;
        while (plain_.empty()) {
            if (closed_) return Status::Closed;
            if (renegotiating_) {
                SECURITY_STATUS s = step(false);
                if (s == SEC_E_INCOMPLETE_MESSAGE || (s == SEC_I_CONTINUE_NEEDED && in_.empty())) {
                    if (eof_) return Status::Closed;
                    return Status::WantRead;
                }
                if (s != SEC_E_OK && s != SEC_I_CONTINUE_NEEDED) return fail("tls_error", status_text(s));
                if (s == SEC_E_OK) renegotiating_ = false;
                continue;
            }
            if (in_.empty()) {
                if (!eof_ && last_decrypt_ != SEC_E_OK) failure_.message = "the last record decrypted as " + status_text(last_decrypt_);
                return eof_ ? Status::Closed : Status::WantRead;
            }
            SecBuffer b[4] = {
                {ULONG(in_.size()), SECBUFFER_DATA, in_.data()},
                {0, SECBUFFER_EMPTY, nullptr},
                {0, SECBUFFER_EMPTY, nullptr},
                {0, SECBUFFER_EMPTY, nullptr},
            };
            SecBufferDesc desc{SECBUFFER_VERSION, 4, b};
            SECURITY_STATUS s = DecryptMessage(&ctx_, &desc, 0, nullptr);
            if (s == SEC_E_INCOMPLETE_MESSAGE) return eof_ ? Status::Closed : Status::WantRead;
            last_decrypt_ = s;
            if (s != SEC_E_OK && s != SEC_I_RENEGOTIATE && s != SEC_I_CONTEXT_EXPIRED) {
                return fail("tls_error", "cannot decrypt a record: " + status_text(s));
            }
            // The plaintext and what is left over both point into `in_`, so
            // they are copied out before it changes. Plaintext only from a
            // record that had some: a close_notify answers SEC_I_CONTEXT_EXPIRED
            // and leaves the alert's own bytes in a DATA buffer, which are not
            // the application's.
            std::string extra;
            bool has_data = s == SEC_E_OK || s == SEC_I_RENEGOTIATE;
            for (SecBuffer& x : b) {
                if (has_data && x.BufferType == SECBUFFER_DATA && x.cbBuffer) {
                    plain_.append(static_cast<char*>(x.pvBuffer), x.cbBuffer);
                }
                if (x.BufferType == SECBUFFER_EXTRA && x.cbBuffer) {
                    extra.assign(in_.data() + (in_.size() - x.cbBuffer), x.cbBuffer);
                }
            }
            in_.swap(extra);
            if (s == SEC_I_CONTEXT_EXPIRED) closed_ = true;
            // A TLS 1.3 ticket or key update: the handshake functions take it.
            if (s == SEC_I_RENEGOTIATE) renegotiating_ = true;
        }
        size_t n = std::min(cap, plain_.size());
        std::memcpy(out, plain_.data(), n);
        plain_.erase(0, n);
        *got = n;
        return Status::Ok;
    }

    Status write(const char* data, size_t len) override {
        if (!done_) return fail("tls_error", "the handshake has not finished");
        while (len > 0) {
            size_t chunk = std::min<size_t>(len, sizes_.cbMaximumMessage);
            std::string record(sizes_.cbHeader + chunk + sizes_.cbTrailer, '\0');
            std::memcpy(record.data() + sizes_.cbHeader, data, chunk);
            SecBuffer b[4] = {
                {sizes_.cbHeader, SECBUFFER_STREAM_HEADER, record.data()},
                {ULONG(chunk), SECBUFFER_DATA, record.data() + sizes_.cbHeader},
                {sizes_.cbTrailer, SECBUFFER_STREAM_TRAILER, record.data() + sizes_.cbHeader + chunk},
                {0, SECBUFFER_EMPTY, nullptr},
            };
            SecBufferDesc desc{SECBUFFER_VERSION, 4, b};
            SECURITY_STATUS s = EncryptMessage(&ctx_, 0, &desc, 0);
            if (s != SEC_E_OK) return fail("tls_error", "cannot encrypt a record: " + status_text(s));
            out_.append(record.data(), b[0].cbBuffer + b[1].cbBuffer + b[2].cbBuffer);
            data += chunk;
            len -= chunk;
        }
        return Status::Ok;
    }

    void close() override {
        if (!have_ctx_ || !done_) return;
        DWORD type = SCHANNEL_SHUTDOWN;
        SecBuffer b{sizeof type, SECBUFFER_TOKEN, &type};
        SecBufferDesc desc{SECBUFFER_VERSION, 1, &b};
        SECURITY_STATUS s = ApplyControlToken(&ctx_, &desc);
        if (s != SEC_E_OK) {
            failure_.message = "SCHANNEL_SHUTDOWN: " + status_text(s);
            return;
        }
        in_.clear();
        s = step(false);  // the close_notify, into `out_`
        if (s != SEC_E_OK) failure_.message = "after SCHANNEL_SHUTDOWN: " + status_text(s);
        done_ = false;
    }

    Info info() const override {
        Info i;
        auto* ctx = const_cast<CtxtHandle*>(&ctx_);
        SecPkgContext_ConnectionInfo conn{};
        if (QueryContextAttributesW(ctx, SECPKG_ATTR_CONNECTION_INFO, &conn) == SEC_E_OK) {
            if (conn.dwProtocol & (SP_PROT_TLS1_3_CLIENT | SP_PROT_TLS1_3_SERVER)) i.version = "TLSv1.3";
            else if (conn.dwProtocol & (SP_PROT_TLS1_2_CLIENT | SP_PROT_TLS1_2_SERVER)) i.version = "TLSv1.2";
            else i.version = "TLS";
        }
        SecPkgContext_CipherInfo cipher{};
        cipher.dwVersion = SECPKGCONTEXT_CIPHERINFO_V1;
        if (QueryContextAttributesW(ctx, SECPKG_ATTR_CIPHER_INFO, &cipher) == SEC_E_OK) {
            i.cipher = windows::utf8(cipher.szCipherSuite);
        }
        SecPkgContext_ApplicationProtocol alpn{};
        if (QueryContextAttributesW(ctx, SECPKG_ATTR_APPLICATION_PROTOCOL, &alpn) == SEC_E_OK
            && alpn.ProtoNegoStatus == SecApplicationProtocolNegotiationStatus_Success) {
            i.alpn.assign(reinterpret_cast<const char*>(alpn.ProtocolId), alpn.ProtocolIdSize);
        }
        PCCERT_CONTEXT peer = nullptr;
        if (QueryContextAttributesW(ctx, SECPKG_ATTR_REMOTE_CERT_CONTEXT, &peer) == SEC_E_OK && peer) {
            i.peer = subject_of(peer);
            CertFreeCertificateContext(peer);
        }
        return i;
    }

private:
    /// One call of the handshake function for this side, with `in_` as its
    /// input. What it produces goes to `out_`; what it did not consume stays.
    SECURITY_STATUS step(bool handshaking) {
        bool first = !have_ctx_;
        SecBuffer in[3] = {
            {ULONG(in_.size()), SECBUFFER_TOKEN, in_.empty() ? nullptr : in_.data()},
            {0, SECBUFFER_EMPTY, nullptr},
            {ULONG(alpn_.size()), SECBUFFER_APPLICATION_PROTOCOLS, alpn_.data()},
        };
        ULONG in_count = (first && !alpn_.empty()) ? 3 : 2;
        SecBufferDesc in_desc{SECBUFFER_VERSION, in_count, in};
        // With nothing to hand over -- the call after SCHANNEL_SHUTDOWN, which
        // only has a close_notify to give back -- a client passes no input at
        // all, as Microsoft's shutdown sequence does, and a server one empty
        // buffer, since AcceptSecurityContext requires a descriptor.
        SecBuffer nothing{0, SECBUFFER_EMPTY, nullptr};
        SecBufferDesc empty_desc{SECBUFFER_VERSION, 1, &nothing};
        bool no_input = !first && in_.empty();
        SecBufferDesc alpn_only{SECBUFFER_VERSION, 1, &in[2]};
        SecBuffer out[2] = {{0, SECBUFFER_TOKEN, nullptr}, {0, SECBUFFER_ALERT, nullptr}};
        SecBufferDesc out_desc{SECBUFFER_VERSION, 2, out};
        ULONG attrs = 0;
        SECURITY_STATUS s;
        if (server_) {
            ULONG flags = ASC_REQ_SEQUENCE_DETECT | ASC_REQ_REPLAY_DETECT | ASC_REQ_CONFIDENTIALITY
                          | ASC_REQ_ALLOCATE_MEMORY | ASC_REQ_STREAM | ASC_REQ_EXTENDED_ERROR;
            if (verify_) flags |= ASC_REQ_MUTUAL_AUTH;
            s = AcceptSecurityContext(&creds_->handle, first ? nullptr : &ctx_, no_input ? &empty_desc : &in_desc,
                                      flags, 0, &ctx_, &out_desc, &attrs, nullptr);
        } else {
            ULONG flags = ISC_REQ_SEQUENCE_DETECT | ISC_REQ_REPLAY_DETECT | ISC_REQ_CONFIDENTIALITY
                          | ISC_REQ_ALLOCATE_MEMORY | ISC_REQ_STREAM | ISC_REQ_EXTENDED_ERROR
                          | ISC_REQ_MANUAL_CRED_VALIDATION;
            if (creds_->identity) flags |= ISC_REQ_USE_SUPPLIED_CREDS;
            SEC_WCHAR* target = whost_.empty() ? nullptr : whost_.data();
            SecBufferDesc* input = first ? (alpn_.empty() ? nullptr : &alpn_only)
                                         : (no_input ? nullptr : &in_desc);
            s = InitializeSecurityContextW(&creds_->handle, first ? nullptr : &ctx_, target, flags, 0, 0, input,
                                           0, &ctx_, &out_desc, &attrs, nullptr);
        }
        if (s == SEC_E_OK || s == SEC_I_CONTINUE_NEEDED || s == SEC_I_INCOMPLETE_CREDENTIALS
            || (FAILED(s) && s != SEC_E_INCOMPLETE_MESSAGE && !first)) {
            have_ctx_ = true;
        }
        for (SecBuffer& o : out) {
            if (o.pvBuffer) {
                if (o.cbBuffer) out_.append(static_cast<char*>(o.pvBuffer), o.cbBuffer);
                FreeContextBuffer(o.pvBuffer);
            }
        }
        if (s == SEC_E_INCOMPLETE_MESSAGE || s == SEC_I_INCOMPLETE_CREDENTIALS) return s;
        // What the call consumed is gone; anything it left is an EXTRA.
        if (!first || server_) {
            if (in[1].BufferType == SECBUFFER_EXTRA && in[1].cbBuffer) {
                in_.erase(0, in_.size() - in[1].cbBuffer);
            } else if (!FAILED(s)) {
                in_.clear();
            }
        }
        (void)handshaking;
        return s;
    }

    Status refused(SECURITY_STATUS s) {
        switch (s) {
            case SEC_E_UNTRUSTED_ROOT:
                return fail("certificate_untrusted", "the certificate is not trusted: " + status_text(s));
            case SEC_E_CERT_EXPIRED:
                return fail("certificate_expired", "the certificate is outside its validity period: " + status_text(s));
            case SEC_E_WRONG_PRINCIPAL:
                return fail("hostname_mismatch", "the certificate is not for this host: " + status_text(s));
            default:
                return fail("handshake_failed", status_text(s));
        }
    }

    /// Send the fatal alert that says why the handshake is over, as OpenSSL
    /// does for a check it makes itself.
    void alert(const std::string& kind) {
        DWORD number = kind == "certificate_expired"   ? TLS1_ALERT_CERTIFICATE_EXPIRED
                       : kind == "certificate_revoked" ? TLS1_ALERT_CERTIFICATE_REVOKED
                       : kind == "hostname_mismatch"   ? TLS1_ALERT_BAD_CERTIFICATE
                       : kind == "certificate_untrusted" ? TLS1_ALERT_UNKNOWN_CA
                                                         : TLS1_ALERT_ACCESS_DENIED;
        SCHANNEL_ALERT_TOKEN token{SCHANNEL_ALERT, TLS1_ALERT_FATAL, number};
        SecBuffer b{sizeof token, SECBUFFER_TOKEN, &token};
        SecBufferDesc desc{SECBUFFER_VERSION, 1, &b};
        if (ApplyControlToken(&ctx_, &desc) != SEC_E_OK) return;
        in_.clear();
        step(false);  // the alert record, into `out_`
    }

    /// The certificate the peer presented, checked as tls.hpp says: chained to
    /// `ca_` if there is one and to the system's roots if not, within its
    /// dates, for the right purpose, and -- for a server -- for `host_`.
    Status check_peer() {
        bool client_checking = !server_ && verify_;
        bool server_checking = server_ && verify_;
        if (!client_checking && !server_checking) return Status::Ok;
        PCCERT_CONTEXT cert = nullptr;
        if (QueryContextAttributesW(&ctx_, SECPKG_ATTR_REMOTE_CERT_CONTEXT, &cert) != SEC_E_OK || !cert) {
            return server_ ? fail("handshake_failed", "the client sent no certificate")
                           : fail("certificate_untrusted", "the server sent no certificate");
        }
        Status result = check_chain(cert);
        CertFreeCertificateContext(cert);
        return result;
    }

    Status check_chain(PCCERT_CONTEXT cert) {
        // The peer's own chain, and the roots given, where chain-building can
        // find them.
        HCERTSTORE pool = CertOpenStore(CERT_STORE_PROV_COLLECTION, 0, 0, 0, nullptr);
        if (cert->hCertStore) CertAddStoreToCollection(pool, cert->hCertStore, 0, 0);
        if (ca_) CertAddStoreToCollection(pool, ca_, 0, 0);
        if (crl_) CertAddStoreToCollection(pool, crl_, 0, 0);

        LPSTR usage = const_cast<LPSTR>(server_ ? szOID_PKIX_KP_CLIENT_AUTH : szOID_PKIX_KP_SERVER_AUTH);
        CERT_CHAIN_PARA para{};
        para.cbSize = sizeof para;
        para.RequestedUsage.dwType = USAGE_MATCH_TYPE_AND;
        para.RequestedUsage.Usage.cUsageIdentifier = 1;
        para.RequestedUsage.Usage.rgpszUsageIdentifier = &usage;
        PCCERT_CHAIN_CONTEXT chain = nullptr;
        // Revocation from what Windows already has -- a stapled OCSP response
        // SChannel cached, the system's own cache -- and never from the
        // network, as tls.hpp says. The lists given as `crl` are read by
        // `listed`.
        DWORD flags = CERT_CHAIN_REVOCATION_CHECK_CHAIN_EXCLUDE_ROOT | CERT_CHAIN_REVOCATION_CHECK_CACHE_ONLY;
        BOOL built = CertGetCertificateChain(nullptr, cert, nullptr, pool, &para, flags, nullptr, &chain);
        CertCloseStore(pool, 0);
        if (!built || !chain || chain->cChain == 0) {
            if (chain) CertFreeCertificateChain(chain);
            return fail("certificate_untrusted", "cannot build the certificate's chain: " + windows::error());
        }

        DWORD errors = chain->TrustStatus.dwErrorStatus;
        // A certificate nothing here speaks for is accepted: revoked is an
        // answer, and "no list to ask" is not one.
        errors &= ~DWORD(CERT_TRUST_REVOCATION_STATUS_UNKNOWN | CERT_TRUST_IS_OFFLINE_REVOCATION);
        bool trusted_root = true;
        if (ca_) {
            // Exclusive: the chain must end at one of the given roots, whatever
            // the system thinks of it.
            const CERT_SIMPLE_CHAIN* simple = chain->rgpChain[0];
            PCCERT_CONTEXT root = simple->rgpElement[simple->cElement - 1]->pCertContext;
            PCCERT_CONTEXT match = CertFindCertificateInStore(ca_, X509_ASN_ENCODING, 0, CERT_FIND_EXISTING,
                                                              root, nullptr);
            trusted_root = match != nullptr;
            if (match) CertFreeCertificateContext(match);
            errors &= ~DWORD(CERT_TRUST_IS_UNTRUSTED_ROOT | CERT_TRUST_IS_PARTIAL_CHAIN);
        }
        Status result = Status::Ok;
        if (!trusted_root) {
            result = fail("certificate_untrusted", "the certificate does not chain to a trusted root");
        } else if ((errors & CERT_TRUST_IS_REVOKED) || listed(chain->rgpChain[0])) {
            result = fail("certificate_revoked", "the certificate has been revoked");
        } else if (errors & CERT_TRUST_IS_NOT_TIME_VALID) {
            result = fail("certificate_expired", "the certificate is outside its validity period");
        } else if (errors != 0) {
            char hex[16];
            std::snprintf(hex, sizeof hex, "0x%08lx", static_cast<unsigned long>(errors));
            result = fail("certificate_untrusted", std::string("the certificate is not trusted (chain status ") + hex + ")");
        } else if (!server_ && check_name_ && !host_.empty()) {
            result = check_name(chain);
        }
        CertFreeCertificateChain(chain);
        return result;
    }

    /// Whether one of the `crl` lists names a certificate of the chain, each
    /// list read only for the certificates its issuer signed and only when
    /// that issuer's signature on it holds. Done here rather than left to the
    /// chain engine, whose revocation provider may or may not look for lists
    /// in the stores it is handed -- Wine's does, Windows' did not -- and a
    /// list that is honoured on one machine and not another is worse than
    /// none.
    bool listed(const CERT_SIMPLE_CHAIN* simple) {
        if (!crl_) return false;
        for (DWORD i = 0; i + 1 < simple->cElement; ++i) {
            PCCERT_CONTEXT cert = simple->rgpElement[i]->pCertContext;
            PCCERT_CONTEXT issuer = simple->rgpElement[i + 1]->pCertContext;
            PCCRL_CONTEXT crl = nullptr;
            while ((crl = CertEnumCRLsInStore(crl_, crl)) != nullptr) {
                if (!CertCompareCertificateName(X509_ASN_ENCODING, &crl->pCrlInfo->Issuer,
                                                &cert->pCertInfo->Issuer)) {
                    continue;
                }
                if (!CryptVerifyCertificateSignatureEx(0, X509_ASN_ENCODING, CRYPT_VERIFY_CERT_SIGN_SUBJECT_CRL,
                                                       const_cast<CRL_CONTEXT*>(crl),
                                                       CRYPT_VERIFY_CERT_SIGN_ISSUER_CERT,
                                                       const_cast<CERT_CONTEXT*>(issuer), 0, nullptr)) {
                    continue;
                }
                PCRL_ENTRY entry = nullptr;
                if (CertFindCertificateInCRL(cert, crl, 0, nullptr, &entry) && entry) {
                    CertFreeCRLContext(crl);
                    return true;
                }
            }
        }
        return false;
    }

    /// The server's name, by the SSL policy for a name and by the certificate's
    /// address entries for an address. Trust, time and revocation were judged
    /// already, so the policy is told to ignore them and can only say something
    /// about the name.
    Status check_name(PCCERT_CHAIN_CONTEXT chain) {
        PCCERT_CONTEXT leaf = chain->rgpChain[0]->rgpElement[0]->pCertContext;
        std::string ip = address_bytes(host_);
        if (!ip.empty()) {
            return names_address(leaf, ip) ? Status::Ok
                                           : fail("hostname_mismatch", "the certificate is not for " + host_);
        }
        SSL_EXTRA_CERT_CHAIN_POLICY_PARA ssl{};
        ssl.cbSize = sizeof ssl;
        ssl.dwAuthType = AUTHTYPE_SERVER;
        ssl.fdwChecks = SECURITY_FLAG_IGNORE_UNKNOWN_CA | SECURITY_FLAG_IGNORE_CERT_DATE_INVALID
                        | SECURITY_FLAG_IGNORE_REVOCATION;
        ssl.pwszServerName = whost_.data();
        CERT_CHAIN_POLICY_PARA policy{};
        policy.cbSize = sizeof policy;
        policy.dwFlags = CERT_CHAIN_POLICY_IGNORE_ALL_REV_UNKNOWN_FLAGS;
        policy.pvExtraPolicyPara = &ssl;
        CERT_CHAIN_POLICY_STATUS status{};
        status.cbSize = sizeof status;
        if (!CertVerifyCertificateChainPolicy(CERT_CHAIN_POLICY_SSL, chain, &policy, &status)) {
            return fail("certificate_untrusted", "cannot check the certificate's name: " + windows::error());
        }
        if (status.dwError == 0) return Status::Ok;
        if (status.dwError == DWORD(CERT_E_CN_NO_MATCH)) {
            return fail("hostname_mismatch", "the certificate is not for " + host_);
        }
        return fail("certificate_untrusted", "the certificate is not trusted: " + windows::error(status.dwError));
    }

    bool server_ = false;
    bool verify_ = true;
    bool check_name_ = true;
    std::string host_;
    std::wstring whost_;
    HCERTSTORE ca_ = nullptr;
    HCERTSTORE crl_ = nullptr;
    std::shared_ptr<Credentials> creds_;
    std::string alpn_;
    CtxtHandle ctx_{};
    bool have_ctx_ = false;
    bool done_ = false;
    bool eof_ = false;
    bool closed_ = false;
    bool renegotiating_ = false;
    SECURITY_STATUS last_decrypt_ = SEC_E_OK;
    SecPkgContext_StreamSizes sizes_{};
    std::string in_;
    std::string out_;
    std::string plain_;
};

}  // namespace

std::unique_ptr<Engine> make_engine(const Config& config, Failure* why) {
    auto engine = std::make_unique<SChannelEngine>();
    if (!engine->init(config, why)) return nullptr;
    return engine;
}

std::string backend() { return "SChannel"; }

}  // namespace dream::tls

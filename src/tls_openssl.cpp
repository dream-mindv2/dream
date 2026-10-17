// The TLS engine on OpenSSL 3, for Linux, macOS and Android. See tls.hpp for the shape.
//
// The SSL object reads its records from one memory BIO and writes them to
// another, so it never sees a descriptor: `feed` is a `BIO_write` into the
// first and `take_output` a `BIO_read` out of the second. Until `feed_eof`
// the input BIO answers "retry" when it is empty, which OpenSSL reports as
// SSL_ERROR_WANT_READ -- the `WantRead` io.cpp parks the process on.
//
// A context (SSL_CTX) carries what is expensive to set up -- the trust store,
// the identity -- and is shared where a connection asks for nothing of its own:
// every default client shares one, so the system's CA bundle is read once per
// VM rather than once per connection.

#include "tls.hpp"

#include <cstring>
#include <filesystem>
#include <mutex>

#include <openssl/err.h>
#include <openssl/ocsp.h>
#include <openssl/pem.h>
#include <openssl/pkcs12.h>
#include <openssl/ssl.h>
#include <openssl/x509v3.h>

namespace dream::tls {

namespace {

/// The most recent error on this thread's queue, as text, or `fallback`.
std::string last_error(const char* fallback) {
    unsigned long code = ERR_peek_last_error();
    if (code == 0) return fallback;
    char buf[256];
    ERR_error_string_n(code, buf, sizeof buf);
    return buf;
}

using CtxPtr = std::shared_ptr<SSL_CTX>;

CtxPtr wrap(SSL_CTX* ctx) { return CtxPtr(ctx, [](SSL_CTX* c) { SSL_CTX_free(c); }); }

/// ALPN names in the protocol's wire form: each a length byte and the name.
std::string alpn_wire(const std::vector<std::string>& names) {
    std::string out;
    for (const std::string& n : names) {
        if (n.empty() || n.size() > 255) continue;
        out.push_back(char(n.size()));
        out += n;
    }
    return out;
}

/// The server's choice: its own preference order, the first the client offers.
int select_alpn(SSL*, const unsigned char** out, unsigned char* outlen,
                const unsigned char* in, unsigned int inlen, void* arg) {
    const std::string& ours = *static_cast<const std::string*>(arg);
    unsigned char* chosen = nullptr;
    if (SSL_select_next_proto(&chosen, outlen,
                              reinterpret_cast<const unsigned char*>(ours.data()),
                              unsigned(ours.size()), in, inlen) != OPENSSL_NPN_NEGOTIATED) {
        return SSL_TLSEXT_ERR_NOACK;
    }
    *out = chosen;
    return SSL_TLSEXT_ERR_OK;
}

/// Trust exactly the certificates in `pem`.
bool load_ca(SSL_CTX* ctx, const std::string& pem, Failure* why) {
    X509_STORE* store = X509_STORE_new();
    BIO* bio = BIO_new_mem_buf(pem.data(), int(pem.size()));
    int count = 0;
    while (X509* cert = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr)) {
        X509_STORE_add_cert(store, cert);
        X509_free(cert);
        ++count;
    }
    BIO_free(bio);
    ERR_clear_error();  // the read that ended the loop
    if (count == 0) {
        X509_STORE_free(store);
        *why = {"tls_config", "`ca` holds no PEM certificate"};
        return false;
    }
    SSL_CTX_set_cert_store(ctx, store);
    return true;
}

/// This side's certificate, chain and key, from a PKCS#12 bundle.
bool load_identity(SSL_CTX* ctx, const std::string& p12, const std::string& password, Failure* why) {
    BIO* bio = BIO_new_mem_buf(p12.data(), int(p12.size()));
    PKCS12* bundle = d2i_PKCS12_bio(bio, nullptr);
    BIO_free(bio);
    if (!bundle) {
        *why = {"tls_config", "`identity` is not a PKCS#12 bundle: " + last_error("unreadable")};
        return false;
    }
    EVP_PKEY* key = nullptr;
    X509* cert = nullptr;
    STACK_OF(X509)* chain = nullptr;
    bool ok = PKCS12_parse(bundle, password.c_str(), &key, &cert, &chain) == 1;
    PKCS12_free(bundle);
    if (!ok || !key || !cert) {
        *why = {"tls_config", "cannot open the PKCS#12 identity (is the password right?): "
                              + last_error("no key or certificate in it")};
        EVP_PKEY_free(key);
        X509_free(cert);
        sk_X509_pop_free(chain, X509_free);
        return false;
    }
    ok = SSL_CTX_use_certificate(ctx, cert) == 1 && SSL_CTX_use_PrivateKey(ctx, key) == 1
         && SSL_CTX_check_private_key(ctx) == 1;
    for (int i = 0; ok && chain && i < sk_X509_num(chain); ++i) {
        ok = SSL_CTX_add1_chain_cert(ctx, sk_X509_value(chain, i)) == 1;
    }
    EVP_PKEY_free(key);
    X509_free(cert);
    sk_X509_pop_free(chain, X509_free);
    if (!ok) *why = {"tls_config", "cannot use the PKCS#12 identity: " + last_error("rejected")};
    return ok;
}

/// Revocation lists, into the context's store, and the store told to consult
/// them for the certificate being checked.
bool load_crls(SSL_CTX* ctx, const std::string& pem, Failure* why) {
    X509_STORE* store = SSL_CTX_get_cert_store(ctx);
    BIO* bio = BIO_new_mem_buf(pem.data(), int(pem.size()));
    int count = 0;
    while (X509_CRL* crl = PEM_read_bio_X509_CRL(bio, nullptr, nullptr, nullptr)) {
        X509_STORE_add_crl(store, crl);
        X509_CRL_free(crl);
        ++count;
    }
    BIO_free(bio);
    ERR_clear_error();
    if (count == 0) {
        *why = {"tls_config", "`crl` holds no PEM certificate revocation list"};
        return false;
    }
    X509_STORE_set_flags(store, X509_V_FLAG_CRL_CHECK);
    return true;
}

/// Verification that does not fail for want of a revocation list. With CRL
/// checking on, OpenSSL refuses a certificate whose issuer has no list in the
/// store, or whose list has lapsed; SChannel calls the same certificate's
/// status unknown and accepts it. The kinder answer is the one both can give,
/// and it is what tls.hpp promises: a certificate is refused for being listed,
/// not for not being.
int tolerant_verify(int ok, X509_STORE_CTX* store) {
    if (ok) return 1;
    switch (X509_STORE_CTX_get_error(store)) {
        case X509_V_ERR_UNABLE_TO_GET_CRL:
        case X509_V_ERR_UNABLE_TO_GET_CRL_ISSUER:
        case X509_V_ERR_CRL_HAS_EXPIRED:
        case X509_V_ERR_CRL_NOT_YET_VALID:
            X509_STORE_CTX_set_error(store, X509_V_OK);
            return 1;
        default:
            return 0;
    }
}

int ocsp_status(SSL* ssl, void*);

// Android's CA filenames use legacy subject hashes. Read the PEM files rather
// than asking OpenSSL 3's hashed-directory lookup to interpret those names.
// Explicit :ca still replaces these roots in context() below.
int default_roots(SSL_CTX* ctx) {
#if defined(__ANDROID__) && !defined(DREAM_TERMUX)
    std::error_code ec;
    const char* path = "/apex/com.android.conscrypt/cacerts";
    if (!std::filesystem::is_directory(path, ec))
        path = "/system/etc/security/cacerts";
    std::filesystem::directory_iterator it(path, ec), end;
    if (ec) return 0;
    unsigned loaded = 0;
    for (; it != end; it.increment(ec)) {
        if (ec) return 0;
        BIO* bio = BIO_new_file(it->path().c_str(), "r");
        if (!bio) { ERR_clear_error(); continue; }
        X509* cert = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);
        if (cert) {
            if (X509_STORE_add_cert(SSL_CTX_get_cert_store(ctx), cert) == 1) ++loaded;
            X509_free(cert);
        }
        ERR_clear_error();
    }
    return !ec && loaded > 0;
#else
    return SSL_CTX_set_default_verify_paths(ctx);
#endif
}

/// A context for `config`. The plainest client configuration is made once.
CtxPtr context(const Config& config, std::string* alpn_holder, Failure* why) {
    bool plain = !config.server && config.ca_pem.empty() && config.identity_p12.empty() && config.crl_pem.empty();
    if (plain) {
        static std::once_flag once;
        static CtxPtr shared;
        static Failure shared_why;
        std::call_once(once, [] {
            SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
            if (!ctx) { shared_why = {"tls_error", last_error("SSL_CTX_new failed")}; return; }
            SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
            SSL_CTX_set_tlsext_status_cb(ctx, ocsp_status);
            if (default_roots(ctx) != 1) {
                shared_why = {"tls_config", "cannot load the system's trusted certificates: "
                                            + last_error("no default paths")};
                SSL_CTX_free(ctx);
                return;
            }
            shared = wrap(ctx);
        });
        if (!shared) *why = shared_why;
        return shared;
    }

    SSL_CTX* raw = SSL_CTX_new(config.server ? TLS_server_method() : TLS_client_method());
    if (!raw) {
        *why = {"tls_error", last_error("SSL_CTX_new failed")};
        return nullptr;
    }
    CtxPtr ctx = wrap(raw);
    SSL_CTX_set_min_proto_version(raw, TLS1_2_VERSION);
    if (!config.ca_pem.empty()) {
        if (!load_ca(raw, config.ca_pem, why)) return nullptr;
    } else if (!config.server && default_roots(raw) != 1) {
        *why = {"tls_config", "cannot load the system's trusted certificates: " + last_error("no default paths")};
        return nullptr;
    }
    if (!config.crl_pem.empty() && !load_crls(raw, config.crl_pem, why)) return nullptr;
    if (!config.server) SSL_CTX_set_tlsext_status_cb(raw, ocsp_status);
    if (!config.identity_p12.empty()
        && !load_identity(raw, config.identity_p12, config.identity_password, why)) {
        return nullptr;
    }
    if (config.server && !config.alpn.empty()) {
        *alpn_holder = alpn_wire(config.alpn);
        SSL_CTX_set_alpn_select_cb(raw, select_alpn, alpn_holder);
    }
    return ctx;
}

class OpenSslEngine : public Engine {
public:
    bool init(const Config& config, Failure* why) {
        server_ = config.server;
        // The ALPN list the server's callback reads lives here, beside the
        // context that points at it, so it outlives every handshake made with it.
        alpn_ = std::make_shared<std::string>();
        ctx_ = context(config, alpn_.get(), why);
        if (!ctx_) return false;
        ssl_ = SSL_new(ctx_.get());
        if (!ssl_) {
            *why = {"tls_error", last_error("SSL_new failed")};
            return false;
        }
        SSL_set_app_data(ssl_, this);
        in_ = BIO_new(BIO_s_mem());
        out_ = BIO_new(BIO_s_mem());
        // Empty means "not yet", not "never": a read of an empty input BIO is
        // a retry until `feed_eof` says the network has ended.
        BIO_set_mem_eof_return(in_, -1);
        SSL_set_bio(ssl_, in_, out_);

        if (server_) {
            SSL_set_accept_state(ssl_);
            if (config.verify && !config.ca_pem.empty()) {
                SSL_set_verify(ssl_, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, tolerant_verify);
            }
            return true;
        }

        SSL_set_connect_state(ssl_);
        if (!config.host.empty()) {
            X509_VERIFY_PARAM* param = SSL_get0_param(ssl_);
            // An address is checked against the certificate's IP entries and
            // is never sent as SNI, which is for names only.
            bool address = X509_VERIFY_PARAM_set1_ip_asc(param, config.host.c_str()) == 1;
            ERR_clear_error();
            if (address && !config.check_name) X509_VERIFY_PARAM_set1_ip(param, nullptr, 0);
            if (!address) {
                SSL_set_tlsext_host_name(ssl_, config.host.c_str());
                if (config.check_name && SSL_set1_host(ssl_, config.host.c_str()) != 1) {
                    *why = {"tls_config", "`" + config.host + "` is not a host name"};
                    return false;
                }
            }
        }
        SSL_set_verify(ssl_, config.verify ? SSL_VERIFY_PEER : SSL_VERIFY_NONE, tolerant_verify);
        // Ask the server to staple what its CA's OCSP responder says of its
        // certificate; `ocsp_status` reads the answer.
        if (config.verify) SSL_set_tlsext_status_type(ssl_, TLSEXT_STATUSTYPE_ocsp);
        if (!config.alpn.empty()) {
            std::string wire = alpn_wire(config.alpn);
            SSL_set_alpn_protos(ssl_, reinterpret_cast<const unsigned char*>(wire.data()),
                                unsigned(wire.size()));
        }
        return true;
    }

    ~OpenSslEngine() override {
        if (ssl_) SSL_free(ssl_);  // frees both BIOs
    }

    void feed(const char* data, size_t len) override {
        if (len) BIO_write(in_, data, int(len));
    }

    void feed_eof() override { BIO_set_mem_eof_return(in_, 0); }

    std::string take_output() override {
        std::string out;
        char buf[16384];
        int n;
        while ((n = BIO_read(out_, buf, sizeof buf)) > 0) out.append(buf, size_t(n));
        return out;
    }

    Status handshake() override {
        ERR_clear_error();
        int r = SSL_do_handshake(ssl_);
        if (r == 1) return Status::Ok;
        return failed(r, true);
    }

    Status read(char* out, size_t cap, size_t* got) override {
        *got = 0;
        ERR_clear_error();
        size_t n = 0;
        if (SSL_read_ex(ssl_, out, cap, &n) == 1) {
            *got = n;
            return Status::Ok;
        }
        return failed(0, false);
    }

    Status write(const char* data, size_t len) override {
        ERR_clear_error();
        size_t n = 0;
        if (len == 0 || SSL_write_ex(ssl_, data, len, &n) == 1) return Status::Ok;
        return failed(0, false);
    }

    void close() override {
        ERR_clear_error();
        SSL_shutdown(ssl_);
        ERR_clear_error();
    }

    Info info() const override {
        Info i;
        i.version = SSL_get_version(ssl_);
        if (const char* c = SSL_get_cipher_name(ssl_)) i.cipher = c;
        const unsigned char* alpn = nullptr;
        unsigned alpn_len = 0;
        SSL_get0_alpn_selected(ssl_, &alpn, &alpn_len);
        if (alpn) i.alpn.assign(reinterpret_cast<const char*>(alpn), alpn_len);
        if (X509* peer = SSL_get1_peer_certificate(ssl_)) {
            BIO* bio = BIO_new(BIO_s_mem());
            X509_NAME_print_ex(bio, X509_get_subject_name(peer), 0, XN_FLAG_RFC2253);
            char* text = nullptr;
            long len = BIO_get_mem_data(bio, &text);
            if (text && len > 0) i.peer.assign(text, size_t(len));
            BIO_free(bio);
            X509_free(peer);
        }
        return i;
    }

private:
    /// What a failed call means. During the handshake a failed verification is
    /// reported as what it was, rather than as the alert it caused.
    Status failed(int r, bool handshaking) {
        int err = SSL_get_error(ssl_, r);
        switch (err) {
            case SSL_ERROR_WANT_READ:
                return Status::WantRead;
            case SSL_ERROR_ZERO_RETURN:
                return Status::Closed;
            case SSL_ERROR_SYSCALL:
            case SSL_ERROR_SSL: {
                if (handshaking && revoked_) {
                    return fail("certificate_revoked", "the server's own OCSP response says its certificate is revoked");
                }
                long verify = SSL_get_verify_result(ssl_);
                if (handshaking && verify != X509_V_OK) return verification(verify);
                unsigned long code = ERR_peek_last_error();
                // The peer went away without a close_notify. After the
                // handshake that is how a great many servers end a response,
                // and treating it as the end of the stream is what every
                // client does; during the handshake it is a failure.
                if (!handshaking && (code == 0 || ERR_GET_REASON(code) == SSL_R_UNEXPECTED_EOF_WHILE_READING)) {
                    return Status::Closed;
                }
                if (handshaking && code == 0) {
                    return fail("handshake_failed", "the peer closed the connection during the handshake");
                }
                return fail(handshaking ? "handshake_failed" : "tls_error", last_error("TLS failed"));
            }
            default:
                return fail("tls_error", "unexpected TLS state " + std::to_string(err));
        }
    }

    Status verification(long code) {
        std::string why = X509_verify_cert_error_string(code);
        switch (code) {
            case X509_V_ERR_HOSTNAME_MISMATCH:
            case X509_V_ERR_IP_ADDRESS_MISMATCH:
                return fail("hostname_mismatch", "the certificate is not for this host: " + why);
            case X509_V_ERR_CERT_REVOKED:
                return fail("certificate_revoked", "the certificate has been revoked: " + why);
            case X509_V_ERR_CERT_HAS_EXPIRED:
            case X509_V_ERR_CERT_NOT_YET_VALID:
                return fail("certificate_expired", "the certificate is outside its validity period: " + why);
            default:
                return fail("certificate_untrusted", "the certificate is not trusted: " + why);
        }
    }

    bool server_ = false;
    CtxPtr ctx_;

public:
    /// Set by `ocsp_status` when the stapled response says revoked.
    bool revoked_ = false;

private:
    std::shared_ptr<std::string> alpn_;
    SSL* ssl_ = nullptr;
    BIO* in_ = nullptr;
    BIO* out_ = nullptr;
};

/// What the server stapled, if it stapled anything. A response that is not
/// one -- unparseable, unsigned by anyone the store trusts, about another
/// certificate -- says nothing and is ignored, as an absent one is: only a
/// trustworthy "revoked" refuses the handshake. See `Config::crl_pem`.
int ocsp_status(SSL* ssl, void*) {
    auto* engine = static_cast<OpenSslEngine*>(SSL_get_app_data(ssl));
    const unsigned char* p = nullptr;
    long len = SSL_get_tlsext_status_ocsp_resp(ssl, &p);
    if (!p || len <= 0 || !engine) return 1;
    OCSP_RESPONSE* resp = d2i_OCSP_RESPONSE(nullptr, &p, len);
    if (!resp) {
        ERR_clear_error();
        return 1;
    }
    int verdict = 1;
    STACK_OF(X509)* chain = SSL_get0_verified_chain(ssl);
    if (OCSP_response_status(resp) == OCSP_RESPONSE_STATUS_SUCCESSFUL && chain && sk_X509_num(chain) >= 2) {
        if (OCSP_BASICRESP* basic = OCSP_response_get1_basic(resp)) {
            X509_STORE* store = SSL_CTX_get_cert_store(SSL_get_SSL_CTX(ssl));
            if (OCSP_basic_verify(basic, chain, store, 0) > 0) {
                OCSP_CERTID* id = OCSP_cert_to_id(nullptr, sk_X509_value(chain, 0), sk_X509_value(chain, 1));
                int status = 0, reason = 0;
                ASN1_GENERALIZEDTIME *revoked_at = nullptr, *this_update = nullptr, *next_update = nullptr;
                if (id && OCSP_resp_find_status(basic, id, &status, &reason, &revoked_at, &this_update, &next_update) == 1
                    && status == V_OCSP_CERTSTATUS_REVOKED) {
                    engine->revoked_ = true;
                    verdict = 0;
                }
                OCSP_CERTID_free(id);
            }
            OCSP_BASICRESP_free(basic);
        }
    }
    OCSP_RESPONSE_free(resp);
    ERR_clear_error();
    return verdict;
}

}  // namespace

std::unique_ptr<Engine> make_engine(const Config& config, Failure* why) {
    auto engine = std::make_unique<OpenSslEngine>();
    if (!engine->init(config, why)) return nullptr;
    return engine;
}

std::string backend() { return OpenSSL_version(OPENSSL_VERSION); }

}  // namespace dream::tls

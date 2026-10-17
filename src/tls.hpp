// TLS, as an engine that never touches a socket.
//
// A Dream process is not a thread, so nothing here may block: the socket
// belongs to io.cpp, which already knows how to try an operation, park the
// process on the poller when the descriptor is not ready, and try again when it
// is. TLS slots in *between* that loop and the bytes: an engine takes the
// ciphertext io.cpp read from the network (`feed`), hands back the ciphertext
// io.cpp must write (`take_output`), and in between turns plaintext into
// records and records into plaintext. When it needs more of the network than it
// has been fed, it says `WantRead`, and io.cpp parks on the socket exactly as a
// plain `read!` would.
//
// That shape is the whole portability argument. Every TLS library worth using
// can be driven from memory -- OpenSSL through memory BIOs, SChannel through
// the buffers `InitializeSecurityContext` already takes -- so the hard part of
// a TLS socket (non-blocking IO, parking, a close that races a reader) is
// written once, in io.cpp, and a backend is only the cryptography. Linux and
// macOS use OpenSSL, linked at build time; Windows uses SChannel, its own. A
// Dream program cannot tell them apart: the configuration it gives is the
// subset every backend can honour, and a failure is reported as one of the
// kinds in `Failure`, whatever the library called it.
//
// An engine is not thread-safe. io.cpp holds the session's mutex around every
// call, because two processes may be on two workers using one handle.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace dream::tls {

/// What a program asked for, in the terms every backend shares.
struct Config {
    /// A server answers a handshake; a client starts one.
    bool server = false;
    /// The name a client expects the server's certificate to carry, and the
    /// name it asks for (SNI). Empty skips both, which is only meaningful with
    /// `verify` off.
    std::string host;
    /// Client: check the server's certificate chain and name. Server: require
    /// a client certificate and check it against `ca_pem`.
    bool verify = true;
    /// Client: whether checking includes `host`. Off is a chain that must be
    /// trusted for a server that may be any name -- libpq's `verify-ca`.
    bool check_name = true;
    /// PEM certificates to trust *instead of* the system's store. Exclusive
    /// rather than additional, because that is the one meaning every backend
    /// can give it, and it is what makes a program's trust the same on every
    /// machine it runs on.
    std::string ca_pem;
    /// PEM certificate revocation lists. A certificate one of them lists is
    /// refused. Revocation is also learned from an OCSP response the server
    /// staples -- OpenSSL's client asks for one, and SChannel's chain check
    /// reads whatever Windows has cached, a staple included. Neither is fetched: with no list
    /// and no staple a certificate's status is unknown and it is accepted --
    /// what SChannel does without the network, and what keeps a handshake
    /// from blocking a worker on a download.
    std::string crl_pem;
    /// A PKCS#12 bundle: this side's certificate, its chain, and its private
    /// key. Required for a server; a client offers it when asked. PKCS#12
    /// rather than PEM because it is the format all the backends can import.
    std::string identity_p12;
    std::string identity_password;
    /// ALPN protocol names, in preference order.
    std::vector<std::string> alpn;
};

enum class Status {
    Ok,
    /// More ciphertext is needed than has been fed.
    WantRead,
    /// The peer closed the TLS session cleanly, or the connection ended.
    Closed,
    /// See `failure()`.
    Error,
};

/// Why something failed. `kind` is the atom a Dream program matches on.
struct Failure {
    /// `certificate_untrusted`, `certificate_expired`, `certificate_revoked`,
    /// `hostname_mismatch`, `handshake_failed`, `tls_config` or `tls_error`.
    std::string kind;
    std::string message;
};

/// What a finished handshake agreed on.
struct Info {
    std::string version;
    std::string cipher;
    std::string alpn;
    /// The peer certificate's subject, or empty when there was none.
    std::string peer;
};

class Engine {
public:
    virtual ~Engine() = default;

    /// Ciphertext from the network.
    virtual void feed(const char* data, size_t len) = 0;
    /// The network has nothing more to give.
    virtual void feed_eof() = 0;
    /// Ciphertext for the network, taken: the engine forgets it.
    virtual std::string take_output() = 0;

    /// Drive the handshake as far as what has been fed allows.
    virtual Status handshake() = 0;
    /// Up to `cap` bytes of plaintext. `Ok` with `*got > 0`.
    virtual Status read(char* out, size_t cap, size_t* got) = 0;
    /// Encrypt all of `data`; the records go to `take_output`.
    virtual Status write(const char* data, size_t len) = 0;
    /// Queue a close_notify.
    virtual void close() = 0;

    virtual Info info() const = 0;
    const Failure& failure() const { return failure_; }

protected:
    Status fail(const char* kind, std::string message) {
        failure_.kind = kind;
        failure_.message = std::move(message);
        return Status::Error;
    }
    Failure failure_;
};

/// An engine for `config`, or null with `*why` filled in.
std::unique_ptr<Engine> make_engine(const Config& config, Failure* why);

/// Which library this VM was built with, and its version.
std::string backend();

/// The DER bytes of each certificate in a PEM text, in order. Shared by the
/// backends whose library does not read PEM itself.
std::vector<std::string> pem_certificates(const std::string& pem);
/// The same for each block labelled `label` -- "X509 CRL" for revocation lists.
std::vector<std::string> pem_blocks(const std::string& pem, const std::string& label);

}  // namespace dream::tls

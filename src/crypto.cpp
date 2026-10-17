// `std.crypto`: the cryptography a program needs, done by the machine.
//
// Dream has no bitwise operators and no fixed-width words, so a hash written
// in it computes each 32-bit AND as a walk over nibble tables -- which is what
// `std.sql.pg.crypto` did for MD5, and why a SCRAM login spent its time in
// four thousand rounds of PBKDF2 each built from two native digests and a
// thirty-two byte xor in Dream. Every member here is one call where that was
// thousands of reductions, and every one is pure except `random_bytes!` and
// `digest_file!`, which read the world.
//
// **Bytes in, bytes out.** A digest, a MAC and a derived key are raw bytes in
// a string, because that is what the next step wants -- an HMAC key, a PBKDF2
// salt, a frame on the wire. `hex` and `base64` are here to spell them, since
// a digest is the first thing anyone wants printed; `unhex` and `unbase64`
// answer `()` for text that is not what they read, as `num.parse_int` does.
//
// **The algorithm is an atom** -- `:md5`, `:sha1`, `:sha256`, `:sha384`,
// `:sha512` -- wherever one is a parameter (`hmac`, `pbkdf2`, `hkdf`), and the
// five digests are also members of their own, which is what a call site that
// knows its hash wants to write.
//
// **No yielding.** A native runs to completion on its worker, so a PBKDF2 of a
// million rounds holds that worker for the second or so it takes. The caller
// chose the count, and a login is the only thing that asks for one.

#include "crypto.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include "windows.hpp"
#include <bcrypt.h>
#elif defined(__APPLE__)
#include <cstdlib>
#else
#include <sys/random.h>
#endif

#include "builtins.hpp"
#include "digest.hpp"
#include "interp.hpp"
#include "process.hpp"

namespace dream {
namespace {

NativeResult crypto_fail(Process& p, const char* kind, const std::string& message) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom(kind), message));
}

Value bytes_value(Process& p, const void* data, size_t len) {
    return p.heap().make_string(static_cast<const char*>(data), uint32_t(len));
}

/// Which hash an atom names, or false.
bool hash_alg(Process& p, Value v, HashAlg* out) {
    v = resolve(v);
    if (!is_atom(v)) return false;
    const std::string& name = p.runtime().atom_name(uint32_t(imm_payload(v)));
    if (name == "sha256") *out = HashAlg::Sha256;
    else if (name == "sha512") *out = HashAlg::Sha512;
    else if (name == "sha384") *out = HashAlg::Sha384;
    else if (name == "sha1") *out = HashAlg::Sha1;
    else if (name == "md5") *out = HashAlg::Md5;
    else return false;
    return true;
}

NativeResult bad_alg(Process& p, const char* fn) {
    return crypto_fail(p, "type_error", std::string(fn) +
                                            " needs a hash: :md5, :sha1, :sha256, :sha384 or :sha512");
}

/// A non-negative integer argument, as a size.
bool size_arg(Value v, uint64_t* out) {
    v = resolve(v);
    if (!is_fixnum(v) || fixnum_value(v) < 0) return false;
    *out = uint64_t(fixnum_value(v));
    return true;
}

// --- digests ---------------------------------------------------------------------

/// The five digest members are this one function, told which hash by the
/// `user` word of the native that called it.
NativeResult crypto_digest(Process& p, Value callee, Value* args, uint32_t) {
    const auto alg = HashAlg(static_cast<NativeObj*>(as_obj(callee))->user);
    Bytes b;
    if (!string_bytes(args[0], &b)) return crypto_fail(p, "type_error", "a digest needs a string");
    Hasher h(alg);
    h.update(b.data, size_t(b.len));
    uint8_t out[kMaxDigest];
    h.finish(out);
    return NativeResult::ok(bytes_value(p, out, h.digest_size()));
}

/// `digest alg s`: the same, with the hash chosen by the caller.
NativeResult crypto_digest_by(Process& p, Value, Value* args, uint32_t) {
    HashAlg alg = HashAlg::Sha256;
    if (!hash_alg(p, args[0], &alg)) return bad_alg(p, "digest");
    Bytes b;
    if (!string_bytes(args[1], &b)) return crypto_fail(p, "type_error", "digest needs a string");
    Hasher h(alg);
    h.update(b.data, size_t(b.len));
    uint8_t out[kMaxDigest];
    h.finish(out);
    return NativeResult::ok(bytes_value(p, out, h.digest_size()));
}

/// `digest_file! alg path`: the digest of a file's contents, read here a block
/// at a time. The file never has to fit on the Dream heap, which is the point:
/// `ship` writes the MD5 of every file into a .deb, the largest of them a
/// 100 MB library, and reading one into a string to hand to `digest` held it
/// whole -- and the collector's headroom over it, four times its size in all.
NativeResult crypto_digest_file(Process& p, Value, Value* args, uint32_t) {
    HashAlg alg = HashAlg::Sha256;
    if (!hash_alg(p, args[0], &alg)) return bad_alg(p, "digest_file!");
    Bytes b;
    if (!string_bytes(args[1], &b)) return crypto_fail(p, "type_error", "digest_file! needs a path");
    const std::string path(b.data, size_t(b.len));
    const std::filesystem::path where(std::u8string(path.begin(), path.end()));
    std::error_code ec;
    if (!std::filesystem::is_regular_file(where, ec)) {
        return crypto_fail(p, std::filesystem::exists(where, ec) ? "wrong_kind" : "not_found",
                           "digest_file!: " + path + " is not a file");
    }
    std::ifstream in(where, std::ios::binary);
    if (!in) return crypto_fail(p, "permission_denied", "digest_file!: cannot open " + path);
    Hasher h(alg);
    std::vector<char> buf(1 << 16);
    while (in) {
        in.read(buf.data(), std::streamsize(buf.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) h.update(buf.data(), size_t(got));
    }
    if (in.bad()) return crypto_fail(p, "io_error", "digest_file!: reading " + path + " failed");
    uint8_t out[kMaxDigest];
    h.finish(out);
    return NativeResult::ok(bytes_value(p, out, h.digest_size()));
}

// --- MACs and key derivation -----------------------------------------------------------

/// `hmac alg key message` (RFC 2104).
NativeResult crypto_hmac(Process& p, Value, Value* args, uint32_t) {
    HashAlg alg = HashAlg::Sha256;
    if (!hash_alg(p, args[0], &alg)) return bad_alg(p, "hmac");
    Bytes key, msg;
    if (!string_bytes(args[1], &key) || !string_bytes(args[2], &msg)) {
        return crypto_fail(p, "type_error", "hmac needs a key and a message, both strings");
    }
    Hmac mac(alg, reinterpret_cast<const uint8_t*>(key.data), size_t(key.len));
    uint8_t out[kMaxDigest];
    mac.mac(reinterpret_cast<const uint8_t*>(msg.data), size_t(msg.len), out);
    return NativeResult::ok(bytes_value(p, out, mac.digest_size()));
}

/// `pbkdf2 alg password salt iterations length` (RFC 8018), HMAC as the PRF.
///
/// The password's two padded blocks are hashed once, in `Hmac`, and every one
/// of the rounds after that is two compressions of a single block each.
NativeResult crypto_pbkdf2(Process& p, Value, Value* args, uint32_t) {
    HashAlg alg = HashAlg::Sha256;
    if (!hash_alg(p, args[0], &alg)) return bad_alg(p, "pbkdf2");
    Bytes password, salt;
    if (!string_bytes(args[1], &password) || !string_bytes(args[2], &salt)) {
        return crypto_fail(p, "type_error", "pbkdf2 needs a password and a salt, both strings");
    }
    uint64_t rounds = 0, length = 0;
    if (!size_arg(args[3], &rounds) || rounds == 0) {
        return crypto_fail(p, "type_error", "pbkdf2 needs a positive iteration count");
    }
    if (!size_arg(args[4], &length) || length == 0 || length > (uint64_t(1) << 24)) {
        return crypto_fail(p, "type_error", "pbkdf2 needs a key length between 1 and 2^24 bytes");
    }
    Hmac mac(alg, reinterpret_cast<const uint8_t*>(password.data), size_t(password.len));
    const size_t hlen = mac.digest_size();
    std::vector<uint8_t> out(length);
    std::vector<uint8_t> first(size_t(salt.len) + 4);
    if (salt.len) std::memcpy(first.data(), salt.data, size_t(salt.len));
    uint8_t u[kMaxDigest], t[kMaxDigest];
    for (uint32_t block = 1; size_t(block - 1) * hlen < length; ++block) {
        uint8_t* index = first.data() + salt.len;
        index[0] = uint8_t(block >> 24); index[1] = uint8_t(block >> 16);
        index[2] = uint8_t(block >> 8); index[3] = uint8_t(block);
        mac.mac(first.data(), first.size(), u);
        std::memcpy(t, u, hlen);
        for (uint64_t r = 1; r < rounds; ++r) {
            mac.mac(u, hlen, u);
            for (size_t i = 0; i < hlen; ++i) t[i] ^= u[i];
        }
        const size_t at = size_t(block - 1) * hlen;
        const size_t take = length - at < hlen ? size_t(length - at) : hlen;
        std::memcpy(out.data() + at, t, take);
    }
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

/// `hkdf alg key salt info length` (RFC 5869): extract, then expand.
NativeResult crypto_hkdf(Process& p, Value, Value* args, uint32_t) {
    HashAlg alg = HashAlg::Sha256;
    if (!hash_alg(p, args[0], &alg)) return bad_alg(p, "hkdf");
    Bytes ikm, salt, info;
    if (!string_bytes(args[1], &ikm) || !string_bytes(args[2], &salt) ||
        !string_bytes(args[3], &info)) {
        return crypto_fail(p, "type_error", "hkdf needs a key, a salt and an info, all strings");
    }
    uint64_t length = 0;
    const size_t hlen = Hasher(alg).digest_size();
    if (!size_arg(args[4], &length) || length == 0 || length > 255 * hlen) {
        return crypto_fail(p, "type_error",
                           "hkdf needs a length between 1 and 255 times the digest size");
    }
    // An empty salt is a block of zeros the digest's length, which as an HMAC
    // key is the same thing as no key at all.
    uint8_t prk[kMaxDigest];
    Hmac(alg, reinterpret_cast<const uint8_t*>(salt.data), size_t(salt.len))
        .mac(reinterpret_cast<const uint8_t*>(ikm.data), size_t(ikm.len), prk);
    Hmac expand(alg, prk, hlen);
    std::vector<uint8_t> out;
    out.reserve(length);
    std::vector<uint8_t> step;
    uint8_t t[kMaxDigest];
    size_t tlen = 0;
    for (uint8_t counter = 1; out.size() < length; ++counter) {
        step.assign(t, t + tlen);
        step.insert(step.end(), info.data, info.data + info.len);
        step.push_back(counter);
        expand.mac(step.data(), step.size(), t);
        tlen = hlen;
        const size_t take = length - out.size() < hlen ? size_t(length - out.size()) : hlen;
        out.insert(out.end(), t, t + take);
    }
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

// --- comparison and combination -----------------------------------------------------

/// Whether two strings hold the same bytes, in time that depends only on their
/// lengths. `==` stops at the first byte that differs, which tells someone
/// timing a MAC check how much of their forgery was right.
NativeResult crypto_equal(Process& p, Value, Value* args, uint32_t) {
    Bytes a, b;
    if (!string_bytes(args[0], &a) || !string_bytes(args[1], &b)) {
        return crypto_fail(p, "type_error", "equal needs two strings");
    }
    if (a.len != b.len) return NativeResult::ok(FALSE_V);
    volatile uint8_t diff = 0;
    for (uint64_t i = 0; i < a.len; ++i) diff = uint8_t(diff | (uint8_t(a.data[i]) ^ uint8_t(b.data[i])));
    return NativeResult::ok(make_bool(diff == 0));
}

/// Two strings of one length, exclusive-or'd byte by byte.
NativeResult crypto_xor(Process& p, Value, Value* args, uint32_t) {
    Bytes a, b;
    if (!string_bytes(args[0], &a) || !string_bytes(args[1], &b)) {
        return crypto_fail(p, "type_error", "xor needs two strings");
    }
    if (a.len != b.len) return crypto_fail(p, "type_error", "xor needs two strings of one length");
    Value out = p.heap().make_string(nullptr, uint32_t(a.len));
    char* d = static_cast<StrObj*>(as_obj(out))->data();
    for (uint64_t i = 0; i < a.len; ++i) d[i] = char(uint8_t(a.data[i]) ^ uint8_t(b.data[i]));
    return NativeResult::ok(out);
}

// --- spelling bytes ----------------------------------------------------------------------

NativeResult crypto_hex(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return crypto_fail(p, "type_error", "hex needs a string");
    static const char digits[] = "0123456789abcdef";
    Value out = p.heap().make_string(nullptr, uint32_t(b.len * 2));
    char* d = static_cast<StrObj*>(as_obj(out))->data();
    for (uint64_t i = 0; i < b.len; ++i) {
        d[2 * i] = digits[uint8_t(b.data[i]) >> 4];
        d[2 * i + 1] = digits[uint8_t(b.data[i]) & 15];
    }
    return NativeResult::ok(out);
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

NativeResult crypto_unhex(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return crypto_fail(p, "type_error", "unhex needs a string");
    if (b.len % 2) return NativeResult::ok(UNIT);
    std::string out(size_t(b.len / 2), '\0');
    for (size_t i = 0; i < out.size(); ++i) {
        int hi = hex_digit(b.data[2 * i]), lo = hex_digit(b.data[2 * i + 1]);
        if (hi < 0 || lo < 0) return NativeResult::ok(UNIT);
        out[i] = char(hi << 4 | lo);
    }
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

constexpr char kBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

NativeResult crypto_base64(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return crypto_fail(p, "type_error", "base64 needs a string");
    std::string out;
    out.reserve(size_t((b.len + 2) / 3 * 4));
    const auto* s = reinterpret_cast<const uint8_t*>(b.data);
    uint64_t i = 0;
    for (; i + 3 <= b.len; i += 3) {
        uint32_t w = uint32_t(s[i]) << 16 | uint32_t(s[i + 1]) << 8 | s[i + 2];
        out += kBase64[w >> 18];
        out += kBase64[(w >> 12) & 63];
        out += kBase64[(w >> 6) & 63];
        out += kBase64[w & 63];
    }
    if (b.len - i == 1) {
        uint32_t w = uint32_t(s[i]) << 16;
        out += kBase64[w >> 18];
        out += kBase64[(w >> 12) & 63];
        out += "==";
    } else if (b.len - i == 2) {
        uint32_t w = uint32_t(s[i]) << 16 | uint32_t(s[i + 1]) << 8;
        out += kBase64[w >> 18];
        out += kBase64[(w >> 12) & 63];
        out += kBase64[(w >> 6) & 63];
        out += '=';
    }
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

int base64_digit(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/// Padded base64, the RFC 4648 alphabet. Anything else -- a length that is not
/// a multiple of four, a stray character, padding in the middle -- is `()`.
NativeResult crypto_unbase64(Process& p, Value, Value* args, uint32_t) {
    Bytes b;
    if (!string_bytes(args[0], &b)) return crypto_fail(p, "type_error", "unbase64 needs a string");
    if (b.len % 4) return NativeResult::ok(UNIT);
    std::string out;
    out.reserve(size_t(b.len / 4 * 3));
    for (uint64_t i = 0; i < b.len; i += 4) {
        const bool last = i + 4 == b.len;
        int pad = 0;
        uint32_t w = 0;
        for (int j = 0; j < 4; ++j) {
            char c = b.data[i + j];
            int d;
            if (c == '=' && last && j >= 2 && (j == 3 || b.data[i + 3] == '=')) {
                d = 0;
                ++pad;
            } else {
                d = base64_digit(c);
                if (d < 0 || pad) return NativeResult::ok(UNIT);
            }
            w = w << 6 | uint32_t(d);
        }
        out += char(w >> 16);
        if (pad < 2) out += char((w >> 8) & 255);
        if (pad < 1) out += char(w & 255);
    }
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

// --- randomness ------------------------------------------------------------------------------

/// `n` bytes from the operating system's generator: the one source a key, a
/// nonce or a salt should come from.
NativeResult crypto_random_bytes(Process& p, Value, Value* args, uint32_t) {
    uint64_t n = 0;
    if (!size_arg(args[0], &n) || n > (uint64_t(1) << 24)) {
        return crypto_fail(p, "type_error", "random_bytes! needs a count between 0 and 2^24");
    }
    std::string out(size_t(n), '\0');
    auto* buf = reinterpret_cast<unsigned char*>(out.data());
#ifdef _WIN32
    if (n && BCryptGenRandom(nullptr, buf, ULONG(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        return crypto_fail(p, "io_error", "the system's random number generator failed");
    }
#elif defined(__APPLE__)
    arc4random_buf(buf, size_t(n));
#else
    for (size_t got = 0; got < n;) {
        ssize_t r = getrandom(buf + got, size_t(n) - got, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return crypto_fail(p, "io_error", std::string("getrandom: ") + std::strerror(errno));
        }
        got += size_t(r);
    }
#endif
    return NativeResult::ok(bytes_value(p, out.data(), out.size()));
}

}  // namespace

ModuleDef make_crypto_module() {
    return ModuleDef{"std.crypto",
                     {
                         {"md5", 1, 0b1, crypto_digest, uint64_t(HashAlg::Md5)},
                         {"sha1", 1, 0b1, crypto_digest, uint64_t(HashAlg::Sha1)},
                         {"sha256", 1, 0b1, crypto_digest, uint64_t(HashAlg::Sha256)},
                         {"sha384", 1, 0b1, crypto_digest, uint64_t(HashAlg::Sha384)},
                         {"sha512", 1, 0b1, crypto_digest, uint64_t(HashAlg::Sha512)},
                         {"digest", 2, 0b11, crypto_digest_by},
                         {"digest_file!", 2, 0b11, crypto_digest_file},
                         {"hmac", 3, 0b111, crypto_hmac},
                         {"pbkdf2", 5, 0b11111, crypto_pbkdf2},
                         {"hkdf", 5, 0b11111, crypto_hkdf},
                         {"equal", 2, 0b11, crypto_equal},
                         {"xor", 2, 0b11, crypto_xor},
                         {"hex", 1, 0b1, crypto_hex},
                         {"unhex", 1, 0b1, crypto_unhex},
                         {"base64", 1, 0b1, crypto_base64},
                         {"unbase64", 1, 0b1, crypto_unbase64},
                         {"random_bytes!", 1, 0b1, crypto_random_bytes},
                     }};
}

}  // namespace dream

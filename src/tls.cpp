// What every TLS backend shares. See tls.hpp.

#include "tls.hpp"

namespace dream::tls {

namespace {

int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

/// The bytes of a base64 body, skipping whitespace and stopping at padding.
std::string base64_decode(const std::string& text, size_t from, size_t to) {
    std::string out;
    unsigned bits = 0;
    int count = 0;
    for (size_t i = from; i < to; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '=') break;
        int v = base64_value(c);
        if (v < 0) continue;
        bits = (bits << 6) | unsigned(v);
        count += 6;
        if (count >= 8) {
            count -= 8;
            out.push_back(char((bits >> count) & 0xff));
        }
    }
    return out;
}

}  // namespace

std::vector<std::string> pem_certificates(const std::string& pem) { return pem_blocks(pem, "CERTIFICATE"); }

std::vector<std::string> pem_blocks(const std::string& pem, const std::string& label) {
    const std::string begin = "-----BEGIN " + label + "-----";
    const std::string end = "-----END " + label + "-----";
    std::vector<std::string> out;
    size_t at = 0;
    for (;;) {
        size_t b = pem.find(begin, at);
        if (b == std::string::npos) break;
        size_t body = b + begin.size();
        size_t e = pem.find(end, body);
        if (e == std::string::npos) break;
        out.push_back(base64_decode(pem, body, e));
        at = e + end.size();
    }
    return out;
}

}  // namespace dream::tls

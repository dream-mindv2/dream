// `std.regex`'s matcher, as a native: `vm.regex_run code subject from mode`.
//
// `std.regex` parses a pattern and compiles it to a small program in Dream,
// which is where the language is good -- the parser is `std.parse`, the
// errors are its errors, and `comp regex.compile ".."` does all of it while
// the program is built. Running that program over text is the opposite kind
// of work: a few comparisons per character, millions of times, and written in
// Dream it ran at about 100 KB a second. Here it is a loop over bytes -- the
// walk the machine can do, done by the machine, as `wire_encode` is.
//
// **Two engines, chosen per pattern and per question.**
//
// A *lazy DFA* answers "is there a match" and "where does it end" at a table
// lookup per character. Its states are ordered sets of the program's
// threads, built the first time the scan needs one and kept, so a pattern
// costs only the states the text actually visits. It runs over *classes* of
// characters -- the program is cut into the ranges it cannot tell apart, so
// `[a-z]+@` has four columns, not a million -- and it keeps Pike's
// leftmost-first order inside each state: a thread reaching `MATCH` cuts off
// the threads after it, and the unanchored restart stops once something has
// matched. So the last position the forward DFA is in a matching state is
// exactly where the leftmost-first match ends. Where it *starts* comes from a
// second DFA over the reversed pattern (which `std.regex` compiles and sends
// along), run backwards from that end: the furthest back it matches is the
// leftmost start, since nothing starts before the leftmost match.
//
// A *Pike VM* (Russ Cox, "Regular Expression Matching: the Virtual Machine
// Approach") answers what a DFA cannot: the captures of groups, and patterns
// with assertions (`^`, `$`, `\b`), whose truth depends on the characters on
// both sides of a position. It runs the threads in lockstep in a sparse set
// with a row of captures each, allocating nothing per step. When the DFA has
// already found a match it runs only over that match, anchored at its start.
//
// Both are linear in the text. Neither backtracks, so no pattern is slow.
//
// **Most text is skipped, not matched.** When a search is in its starting
// state the next place a match could begin is found with `memchr` -- on the
// first byte of the literal every match starts with, or a scan for a byte in
// the set of possible first bytes -- and everything before it is passed over.
//
// **Engines are cached.** The program arrives as bytes (the layout is below)
// and a decoded engine, with whatever DFA states it has built, is kept per
// worker thread keyed by those bytes, so a pattern used in a loop is decoded
// once and its DFA warms up across calls. The cache is small and is emptied
// rather than evicted from when it fills; a native runs to completion on its
// worker, so no lock is needed and no Dream value is held.
//
// The program, little-endian:
//
//     "DRX1"  u32 groups  u32 ninsts  ninsts x (u8 op, i32 a, i32 b)
//     u32 nclasses  nclasses x (u8 negated, u32 nranges, nranges x (u32 lo, u32 hi))
//     u32 prefix_len  prefix bytes   u8 has_starts  [32 bytes: a bitmap of first bytes]
//     u32 reverse_len  [reverse_len bytes: the reversed program, itself "DRX1" ..]
//
// The modes: 0 `is_match` (a bool), 1 the first match at or after `from` (an
// array of capture offsets, or unit), 2 every match (a list of those arrays),
// 3 how many matches there are, 4 the match starting exactly at `from`.
// Offsets are bytes; an unset capture is -1.

#include "regex.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "builtins.hpp"
#include "interp.hpp"
#include "process.hpp"

namespace dream {
namespace {

enum Op : uint8_t { CHAR = 0, ANY = 1, ANY_NL = 2, CLASS = 3, SPLIT = 4, JMP = 5, SAVE = 6, ASSERT = 7, MATCH = 8 };
enum Assert : int32_t { TEXT_START = 0, TEXT_END = 1, LINE_START = 2, LINE_END = 3, WORD = 4, NOT_WORD = 5 };

struct Inst {
    uint8_t op;
    int32_t a, b;
};

struct Class {
    bool negated = false;
    uint8_t ascii[16] = {};  // bit c % 8 of byte c / 8: whether ASCII character c is in the ranges
    std::vector<std::pair<uint32_t, uint32_t>> ranges;

    bool contains(uint32_t c) const {
        bool in = false;
        if (c < 128) {
            in = (ascii[c >> 3] >> (c & 7)) & 1;
        } else {
            // Sorted and disjoint, so halving finds the one range that could hold it.
            size_t lo = 0, hi = ranges.size();
            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                if (c < ranges[mid].first) hi = mid;
                else if (c > ranges[mid].second) lo = mid + 1;
                else { in = true; break; }
            }
        }
        return in != negated;
    }
};

struct Prog {
    uint32_t groups = 0;
    std::vector<Inst> insts;
    std::vector<Class> classes;
    std::string prefix;
    bool has_starts = false;
    uint8_t starts[32] = {};
    bool has_asserts = false;

    bool consumes(const Inst& in, uint32_t c) const {
        switch (in.op) {
            case CHAR: return c == uint32_t(in.a);
            case ANY: return c != '\n';
            case ANY_NL: return true;
            case CLASS: return classes[size_t(in.a)].contains(c);
            default: return false;
        }
    }
};

struct Cursor {
    const uint8_t* p;
    size_t n, at = 0;
    bool ok = true;
    bool need(size_t k) { if (!ok || k > n || at > n - k) ok = false; return ok; }
    uint8_t u8() { if (!need(1)) return 0; return p[at++]; }
    uint32_t u32() {
        if (!need(4)) return 0;
        uint32_t v = uint32_t(p[at]) | uint32_t(p[at + 1]) << 8 | uint32_t(p[at + 2]) << 16 | uint32_t(p[at + 3]) << 24;
        at += 4;
        return v;
    }
};

/// One program from `c`, or false when the bytes are not one: a jump out of
/// range, a class that does not exist, a capture slot past the groups.
/// Checked here so that nothing below needs to check anything.
bool decode_prog(Cursor& c, Prog* prog) {
    if (!c.need(4) || std::memcmp(c.p + c.at, "DRX1", 4) != 0) return false;
    c.at += 4;
    prog->groups = c.u32();
    uint32_t ninsts = c.u32();
    if (!c.ok || ninsts == 0 || ninsts > (1u << 24) || prog->groups == 0 || prog->groups > (1u << 16)) return false;
    if (!c.need(size_t(ninsts) * 9)) return false;
    prog->insts.resize(ninsts);
    for (auto& in : prog->insts) {
        in.op = c.u8();
        in.a = int32_t(c.u32());
        in.b = int32_t(c.u32());
    }
    uint32_t nclasses = c.u32();
    if (!c.ok || nclasses > (1u << 20)) return false;
    prog->classes.resize(nclasses);
    for (auto& cl : prog->classes) {
        cl.negated = c.u8() != 0;
        uint32_t nr = c.u32();
        if (!c.ok || !c.need(size_t(nr) * 8)) return false;
        cl.ranges.resize(nr);
        for (auto& r : cl.ranges) {
            r.first = c.u32();
            r.second = c.u32();
            if (r.second < r.first) return false;
            for (uint32_t ch = r.first; ch <= r.second && ch < 128; ++ch) cl.ascii[ch >> 3] |= uint8_t(1u << (ch & 7));
        }
    }
    uint32_t plen = c.u32();
    if (!c.need(plen)) return false;
    prog->prefix.assign(reinterpret_cast<const char*>(c.p + c.at), plen);
    c.at += plen;
    prog->has_starts = c.u8() != 0;
    if (prog->has_starts) {
        if (!c.need(32)) return false;
        std::memcpy(prog->starts, c.p + c.at, 32);
        c.at += 32;
    }
    if (!c.ok) return false;
    for (const auto& in : prog->insts) {
        switch (in.op) {
            case CHAR: case ANY: case ANY_NL: case MATCH: break;
            case CLASS: if (in.a < 0 || uint32_t(in.a) >= nclasses) return false; break;
            case SPLIT: if (in.b < 0 || uint32_t(in.b) >= ninsts) return false; [[fallthrough]];
            case JMP: if (in.a < 0 || uint32_t(in.a) >= ninsts) return false; break;
            case SAVE: if (in.a < 0 || uint32_t(in.a) >= 2 * prog->groups) return false; break;
            case ASSERT:
                if (in.a < TEXT_START || in.a > NOT_WORD) return false;
                prog->has_asserts = true;
                break;
            default: return false;
        }
    }
    // Stepping past the last instruction is only possible from one that falls
    // through, so the program must end in a jump or a match.
    uint8_t last = prog->insts.back().op;
    return last == MATCH || last == JMP;
}

/// The character at `at`: its code point, and its width in bytes. A malformed
/// byte reads as U+FFFD one byte wide, as `std.parse`'s `char_at` reads it.
inline uint32_t decode_char(const uint8_t* s, size_t n, size_t at, size_t* width) {
    uint8_t b0 = s[at];
    if (b0 < 0x80) { *width = 1; return b0; }
    auto cont = [&](size_t k) -> int { return at + k < n && (s[at + k] & 0xC0) == 0x80 ? s[at + k] & 0x3F : -1; };
    if (b0 >= 0xC2 && b0 < 0xE0) {
        int c1 = cont(1);
        if (c1 >= 0) { *width = 2; return uint32_t(b0 & 0x1F) << 6 | uint32_t(c1); }
    } else if (b0 >= 0xE0 && b0 < 0xF0) {
        int c1 = cont(1), c2 = cont(2);
        if (c1 >= 0 && c2 >= 0) {
            uint32_t cp = uint32_t(b0 & 0x0F) << 12 | uint32_t(c1) << 6 | uint32_t(c2);
            if (cp >= 0x800 && (cp < 0xD800 || cp > 0xDFFF)) { *width = 3; return cp; }
        }
    } else if (b0 >= 0xF0 && b0 < 0xF5) {
        int c1 = cont(1), c2 = cont(2), c3 = cont(3);
        if (c1 >= 0 && c2 >= 0 && c3 >= 0) {
            uint32_t cp = uint32_t(b0 & 0x07) << 18 | uint32_t(c1) << 12 | uint32_t(c2) << 6 | uint32_t(c3);
            if (cp >= 0x10000 && cp <= 0x10FFFF) { *width = 4; return cp; }
        }
    }
    *width = 1;
    return 0xFFFD;
}

/// The character that ends at `at`, read backwards, or false when the bytes
/// before `at` do not split the way reading forwards would -- which happens
/// only in text that is not UTF-8, and sends the caller to the engine that
/// reads forwards.
inline bool decode_before(const uint8_t* s, size_t n, size_t at, uint32_t* cp, size_t* width) {
    uint8_t last = s[at - 1];
    if (last < 0x80) { *cp = last; *width = 1; return true; }
    size_t k = at - 1;
    size_t back = 0;
    while (back < 3 && k > 0 && (s[k] & 0xC0) == 0x80) { --k; ++back; }
    size_t w = 0;
    uint32_t c = decode_char(s, n, k, &w);
    if (k + w != at) return false;
    *cp = c;
    *width = w;
    return true;
}

inline bool word_byte(const uint8_t* s, size_t n, int64_t i) {
    if (i < 0 || size_t(i) >= n) return false;
    uint8_t b = s[i];
    return (b >= '0' && b <= '9') || (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b == '_';
}

/// The next offset at or after `pos` where a match of `prog` could begin, or
/// past the end when there is none.
size_t skip(const Prog& prog, const uint8_t* s, size_t n, size_t pos) {
    if (!prog.prefix.empty()) {
        const std::string& pre = prog.prefix;
        uint8_t first = uint8_t(pre[0]);
        while (pos + pre.size() <= n) {
            const void* hit = std::memchr(s + pos, first, n - pos - pre.size() + 1);
            if (!hit) return n + 1;
            size_t at = size_t(static_cast<const uint8_t*>(hit) - s);
            if (std::memcmp(s + at, pre.data(), pre.size()) == 0) return at;
            pos = at + 1;
        }
        return n + 1;
    }
    if (prog.has_starts) {
        while (pos < n && !((prog.starts[s[pos] >> 3] >> (s[pos] & 7)) & 1)) ++pos;
        return pos < n ? pos : n + 1;
    }
    return pos;
}

// --- the lazy DFA ---------------------------------------------------------------------

/// A DFA built as it is used. `cut` is leftmost-first (the forward search):
/// a `MATCH` thread ends a state's list, and a state that matched stops
/// restarting the search. Without it (the reverse search) every thread
/// survives, which finds the longest match -- the furthest-back start.
class Dfa {
public:
    static constexpr int32_t UNKNOWN = -1;
    static constexpr int32_t GAVE_UP = -2;
    /// Past this many states the DFA stops growing and the search falls back
    /// to the Pike VM: a pattern whose DFA is exponential (`(a|b)*a(a|b){20}`)
    /// costs a bounded table and a slower answer, never a blow-up.
    static constexpr size_t MAX_STATES = 4096;

    Dfa(const Prog& prog, bool cut) : prog_(prog), cut_(cut) {
        // The class boundaries: every place where some instruction's answer
        // can change as the code point rises.
        std::vector<uint32_t> b{0};
        for (const auto& in : prog.insts) {
            if (in.op == CHAR) { b.push_back(uint32_t(in.a)); b.push_back(uint32_t(in.a) + 1); }
            else if (in.op == ANY) { b.push_back('\n'); b.push_back('\n' + 1); }
            else if (in.op == CLASS) {
                for (const auto& r : prog.classes[size_t(in.a)].ranges) { b.push_back(r.first); b.push_back(r.second + 1); }
            }
        }
        std::sort(b.begin(), b.end());
        b.erase(std::unique(b.begin(), b.end()), b.end());
        while (!b.empty() && b.back() > 0x10FFFF) b.pop_back();
        bounds_ = b;
        for (uint32_t c = 0; c < 128; ++c) ascii_class_[c] = uint16_t(slow_class(c));
        visited_.assign(prog.insts.size(), 0);
    }

    int class_of(uint32_t c) const { return c < 128 ? ascii_class_[c] : slow_class(c); }

    /// The state the search starts in: the program's start, restarting at
    /// every position (`seeded`) unless anchored.
    int32_t start(bool seeded) {
        int32_t& slot = seeded ? start_seeded_ : start_anchored_;
        if (slot == UNKNOWN) {
            std::vector<uint32_t> pcs;
            begin_closure();
            closure(0, pcs);
            int32_t id = intern(std::move(pcs), seeded);
            (seeded ? start_seeded_ : start_anchored_) = id;
            return id;
        }
        return slot;
    }

    bool is_match(int32_t s) const { return states_[size_t(s)].matched; }
    bool is_dead(int32_t s) const { return states_[size_t(s)].pcs.empty() && !states_[size_t(s)].seeded; }

    int32_t next(int32_t s, int cls) {
        size_t slot = size_t(s) * bounds_.size() + size_t(cls);
        int32_t t = trans_[slot];
        if (t != UNKNOWN) return t;
        int32_t computed = step(s, cls);
        if (computed != GAVE_UP) trans_[slot] = computed;
        return computed;
    }

private:
    struct State {
        std::vector<uint32_t> pcs;
        bool seeded;
        bool matched;
    };

    int slow_class(uint32_t c) const {
        auto it = std::upper_bound(bounds_.begin(), bounds_.end(), c);
        return int(it - bounds_.begin()) - 1;
    }

    void begin_closure() {
        if (++generation_ == 0) { std::fill(visited_.begin(), visited_.end(), 0); generation_ = 1; }
    }

    /// The threads reachable from `pc0` without a character, appended to
    /// `out` in priority order. Assertions never reach a DFA.
    void closure(uint32_t pc0, std::vector<uint32_t>& out) {
        stack_.clear();
        stack_.push_back(pc0);
        while (!stack_.empty()) {
            uint32_t pc = stack_.back();
            stack_.pop_back();
            if (visited_[pc] == generation_) continue;
            visited_[pc] = generation_;
            const Inst& in = prog_.insts[pc];
            switch (in.op) {
                case JMP: stack_.push_back(uint32_t(in.a)); break;
                case SPLIT: stack_.push_back(uint32_t(in.b)); stack_.push_back(uint32_t(in.a)); break;
                case SAVE: stack_.push_back(pc + 1); break;
                default: out.push_back(pc); break;
            }
        }
    }

    int32_t step(int32_t s, int cls) {
        uint32_t rep = bounds_[size_t(cls)];
        std::vector<uint32_t> pcs;
        begin_closure();
        bool seeded = states_[size_t(s)].seeded && !(cut_ && states_[size_t(s)].matched);
        // Copied: `intern` below may move `states_`.
        std::vector<uint32_t> current = states_[size_t(s)].pcs;
        for (uint32_t pc : current) {
            const Inst& in = prog_.insts[pc];
            if (in.op == MATCH) {
                if (cut_) break;
                continue;
            }
            if (prog_.consumes(in, rep)) closure(pc + 1, pcs);
        }
        if (seeded) closure(0, pcs);
        return intern(std::move(pcs), seeded);
    }

    int32_t intern(std::vector<uint32_t>&& pcs, bool seeded) {
        key_.assign(reinterpret_cast<const char*>(pcs.data()), pcs.size() * sizeof(uint32_t));
        key_.push_back(seeded ? '\1' : '\0');
        auto found = index_.find(key_);
        if (found != index_.end()) return found->second;
        if (states_.size() >= MAX_STATES) return GAVE_UP;
        bool matched = false;
        for (uint32_t pc : pcs) {
            if (prog_.insts[pc].op == MATCH) { matched = true; break; }
        }
        int32_t id = int32_t(states_.size());
        states_.push_back(State{std::move(pcs), seeded, matched});
        trans_.resize(states_.size() * bounds_.size(), UNKNOWN);
        index_.emplace(key_, id);
        return id;
    }

    const Prog& prog_;
    bool cut_;
    std::vector<uint32_t> bounds_;
    uint16_t ascii_class_[128] = {};
    std::vector<State> states_;
    std::vector<int32_t> trans_;
    std::unordered_map<std::string, int32_t> index_;
    std::vector<uint32_t> visited_;
    uint32_t generation_ = 0;
    std::vector<uint32_t> stack_;
    std::string key_;
    int32_t start_seeded_ = UNKNOWN, start_anchored_ = UNKNOWN;
};

// --- the Pike VM -----------------------------------------------------------------------

/// A set of program counters with a row of captures each, in the order they
/// were added -- which is their priority.
struct Threads {
    std::vector<uint32_t> dense, sparse;
    std::vector<int64_t> caps;
    size_t size = 0, ncap = 0;

    void init(size_t ninsts, size_t ncaps) {
        dense.assign(ninsts, 0);
        sparse.assign(ninsts, 0);
        caps.assign(ninsts * ncaps, -1);
        ncap = ncaps;
        size = 0;
    }
    bool has(uint32_t pc) const { uint32_t i = sparse[pc]; return i < size && dense[i] == pc; }
    size_t insert(uint32_t pc) { sparse[pc] = uint32_t(size); dense[size] = pc; return size++; }
    int64_t* row(size_t i) { return caps.data() + i * ncap; }
};

class Pike {
public:
    explicit Pike(const Prog& prog) : prog_(prog), ncap_(2 * size_t(prog.groups)) {
        a_.init(prog.insts.size(), ncap_);
        b_.init(prog.insts.size(), ncap_);
        work_.assign(ncap_, -1);
        best_.assign(ncap_, -1);
    }

    /// The first match at or after `from` (only at `from` when anchored), its
    /// captures in `best()`. `want_caps` false stops at the first thread to
    /// reach `MATCH`, whatever its priority: enough to say there is a match.
    bool search(const uint8_t* s, size_t n, size_t from, bool anchored, bool want_caps) {
        s_ = s;
        n_ = n;
        Threads* clist = &a_;
        Threads* nlist = &b_;
        clist->size = 0;
        bool matched = false;
        size_t pos = from;
        for (;;) {
            if (clist->size == 0 && !matched) {
                if (anchored) {
                    if (pos != from) break;
                } else {
                    size_t next = skip(prog_, s_, n_, pos);
                    if (next > n_) break;
                    pos = next;
                }
            }
            if (!matched && (!anchored || pos == from)) {
                std::fill(work_.begin(), work_.end(), -1);
                add(*clist, 0, pos);
            }
            size_t width = 0;
            bool at_end = pos >= n_;
            uint32_t ch = at_end ? 0 : decode_char(s_, n_, pos, &width);
            nlist->size = 0;
            for (size_t i = 0; i < clist->size; ++i) {
                uint32_t pc = clist->dense[i];
                const Inst& in = prog_.insts[pc];
                if (in.op == MATCH) {
                    std::memcpy(best_.data(), clist->row(i), ncap_ * sizeof(int64_t));
                    matched = true;
                    if (!want_caps) return true;
                    break;  // every thread after this one has lower priority
                }
                if (!at_end && prog_.consumes(in, ch)) {
                    std::memcpy(work_.data(), clist->row(i), ncap_ * sizeof(int64_t));
                    add(*nlist, pc + 1, pos + width);
                }
            }
            std::swap(clist, nlist);
            if (at_end) break;
            pos += width;
            if (clist->size == 0 && matched) break;
        }
        return matched;
    }

    const std::vector<int64_t>& best() const { return best_; }

private:
    bool holds(int32_t kind, size_t pos) const {
        switch (kind) {
            case TEXT_START: return pos == 0;
            case TEXT_END: return pos >= n_;
            case LINE_START: return pos == 0 || s_[pos - 1] == '\n';
            case LINE_END: return pos >= n_ || s_[pos] == '\n';
            case WORD: return word_byte(s_, n_, int64_t(pos) - 1) != word_byte(s_, n_, int64_t(pos));
            case NOT_WORD: return word_byte(s_, n_, int64_t(pos) - 1) == word_byte(s_, n_, int64_t(pos));
        }
        return false;
    }

    /// Follow everything from `pc0` that needs no character, adding the
    /// threads that wait on one to `list` with the captures in `work_`. An
    /// explicit stack rather than recursion: `a{0,1000}` is a chain of a
    /// thousand splits, and a worker thread's stack is not the place for it.
    void add(Threads& list, uint32_t pc0, size_t pos) {
        stack_.clear();
        stack_.push_back({pc0, -1, 0});
        while (!stack_.empty()) {
            Job job = stack_.back();
            stack_.pop_back();
            if (job.restore_slot >= 0) { work_[size_t(job.restore_slot)] = job.restore_value; continue; }
            uint32_t pc = job.pc;
            if (list.has(pc)) continue;
            size_t at = list.insert(pc);
            const Inst& in = prog_.insts[pc];
            switch (in.op) {
                case JMP: stack_.push_back({uint32_t(in.a), -1, 0}); break;
                case SPLIT:
                    stack_.push_back({uint32_t(in.b), -1, 0});
                    stack_.push_back({uint32_t(in.a), -1, 0});
                    break;
                case SAVE:
                    stack_.push_back({0, in.a, work_[size_t(in.a)]});
                    work_[size_t(in.a)] = int64_t(pos);
                    stack_.push_back({pc + 1, -1, 0});
                    break;
                case ASSERT:
                    if (holds(in.a, pos)) stack_.push_back({pc + 1, -1, 0});
                    break;
                default:
                    std::memcpy(list.row(at), work_.data(), ncap_ * sizeof(int64_t));
                    break;
            }
        }
    }

    struct Job { uint32_t pc; int32_t restore_slot; int64_t restore_value; };
    const Prog& prog_;
    const uint8_t* s_ = nullptr;
    size_t n_ = 0;
    size_t ncap_;
    Threads a_, b_;
    std::vector<int64_t> work_, best_;
    std::vector<Job> stack_;
};

// --- an engine: both, and the choice between them -----------------------------------------

class Engine {
public:
    Prog fwd, rev;

    void ready() {
        // A DFA cannot see the characters on both sides of a position, so a
        // pattern with an assertion is the Pike VM's alone.
        use_dfa_ = !fwd.has_asserts && !rev.insts.empty() && !rev.has_asserts;
        pike_ = std::make_unique<Pike>(fwd);
        if (use_dfa_) {
            forward_ = std::make_unique<Dfa>(fwd, true);
            reverse_ = std::make_unique<Dfa>(rev, false);
        }
    }

    /// The first match at or after `from`, its captures in `caps`.
    /// `want_caps` false is `is_match`; `need_start` false is `count`, which
    /// needs only where each match ends (and leaves `caps[0]` at `from`).
    bool search(const uint8_t* s, size_t n, size_t from, bool anchored, bool want_caps, bool need_start,
                std::vector<int64_t>& caps) {
        if (use_dfa_) {
            int64_t end = forward_search(s, n, from, anchored, !want_caps);
            if (end == -1) return false;
            if (end >= 0 && !want_caps) return true;
            if (end >= 0) {
                size_t e = size_t(end);
                // With no assertions, a pattern that can match empty matches
                // empty where the search began, so an end there is an empty
                // match there; any other end has its start before it.
                int64_t start;
                if (anchored || e == from || !need_start) start = int64_t(anchored || !need_start ? from : e);
                else start = reverse_search(s, n, e, from);
                if (start >= 0) {
                    if (fwd.groups == 1 || !need_start) {
                        caps.assign(2, -1);
                        caps[0] = start;
                        caps[1] = int64_t(e);
                        return true;
                    }
                    // Groups: the Pike VM, anchored at the start, over the match.
                    if (pike_->search(s, n, size_t(start), true, true)) { caps = pike_->best(); return true; }
                }
            }
            // The DFA gave up (too many states, or text that is not UTF-8
            // read backwards): the Pike VM answers this one.
        }
        if (!pike_->search(s, n, from, anchored, want_caps)) return false;
        caps = pike_->best();
        return true;
    }

private:
    /// Where the leftmost-first match ends: an offset, -1 for none, -2 when
    /// the DFA gave up.
    int64_t forward_search(const uint8_t* s, size_t n, size_t from, bool anchored, bool earliest) {
        Dfa& d = *forward_;
        int32_t start = d.start(!anchored);
        if (start == Dfa::GAVE_UP) return -2;
        int32_t st = start;
        int64_t last = -1;
        size_t pos = from;
        for (;;) {
            if (st == start && !anchored && !d.is_match(st)) {
                size_t next = skip(fwd, s, n, pos);
                if (next > n) return last;
                pos = next;
            }
            if (d.is_match(st)) {
                last = int64_t(pos);
                if (earliest) return last;
            }
            if (pos >= n || d.is_dead(st)) return last;
            uint32_t c;
            size_t w;
            uint8_t b = s[pos];
            if (b < 0x80) { c = b; w = 1; } else { c = decode_char(s, n, pos, &w); }
            st = d.next(st, d.class_of(c));
            if (st == Dfa::GAVE_UP) return -2;
            pos += w;
        }
    }

    /// The furthest back, not before `floor`, that the reversed pattern
    /// matches ending at `end`: the leftmost start. -2 when it gave up.
    int64_t reverse_search(const uint8_t* s, size_t n, size_t end, size_t floor) {
        Dfa& d = *reverse_;
        int32_t st = d.start(false);
        if (st == Dfa::GAVE_UP) return -2;
        int64_t found = -2;
        size_t pos = end;
        for (;;) {
            if (d.is_match(st)) found = int64_t(pos);
            if (pos <= floor || d.is_dead(st)) return found;
            uint32_t c;
            size_t w;
            if (!decode_before(s, n, pos, &c, &w)) return -2;
            st = d.next(st, d.class_of(c));
            if (st == Dfa::GAVE_UP) return -2;
            pos -= w;
        }
    }

    bool use_dfa_ = false;
    std::unique_ptr<Pike> pike_;
    std::unique_ptr<Dfa> forward_, reverse_;
};

/// The engine for these bytes: from this worker's cache, or decoded now and
/// kept. `nullptr` when the bytes are not a program.
Engine* engine_for(const Bytes& code) {
    thread_local std::unordered_map<std::string, std::unique_ptr<Engine>> cache;
    std::string key(code.data, size_t(code.len));
    auto found = cache.find(key);
    if (found != cache.end()) return found->second.get();
    auto e = std::make_unique<Engine>();
    Cursor c{reinterpret_cast<const uint8_t*>(code.data), size_t(code.len)};
    if (!decode_prog(c, &e->fwd)) return nullptr;
    uint32_t rlen = c.u32();
    if (!c.ok || !c.need(rlen)) return nullptr;
    if (rlen > 0) {
        Cursor rc{c.p + c.at, rlen};
        if (!decode_prog(rc, &e->rev)) return nullptr;
    }
    e->ready();
    if (cache.size() >= 64) cache.clear();
    Engine* raw = e.get();
    cache.emplace(std::move(key), std::move(e));
    return raw;
}

Value captures_value(Process& p, const std::vector<int64_t>& caps) {
    Value arr = p.heap().make_array(uint32_t(caps.size()));
    auto* a = static_cast<ArrayObj*>(as_obj(arr));
    for (size_t i = 0; i < caps.size(); ++i) a->items()[i] = make_integer(p, caps[i]);
    return arr;
}

NativeResult regex_fail(Process& p, const std::string& message) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom("type_error"), message));
}

}  // namespace

NativeResult vm_regex_run(Process& p, Value, Value* args, uint32_t) {
    Bytes code, subject;
    if (!string_bytes(args[0], &code)) return regex_fail(p, "regex_run needs a compiled program as a string");
    if (!string_bytes(args[1], &subject)) return regex_fail(p, "regex_run needs a string to search");
    Value from_v = resolve(args[2]), mode_v = resolve(args[3]);
    if (!is_fixnum(from_v) || !is_fixnum(mode_v)) return regex_fail(p, "regex_run needs an offset and a mode");
    int64_t from = fixnum_value(from_v), mode = fixnum_value(mode_v);
    if (mode < 0 || mode > 4) return regex_fail(p, "regex_run's mode is 0 to 4");
    Engine* engine = engine_for(code);
    if (!engine) return regex_fail(p, "regex_run was given bytes that are not a compiled pattern");
    const auto* s = reinterpret_cast<const uint8_t*>(subject.data);
    size_t n = size_t(subject.len);
    if (from < 0 || uint64_t(from) > n) {
        switch (mode) {
            case 0: return NativeResult::ok(make_bool(false));
            case 2: return NativeResult::ok(NIL);
            case 3: return NativeResult::ok(make_fixnum(0));
            default: return NativeResult::ok(UNIT);
        }
    }
    std::vector<int64_t> caps;
    switch (mode) {
        case 0: return NativeResult::ok(make_bool(engine->search(s, n, size_t(from), false, false, false, caps)));
        case 1:
        case 4:
            if (!engine->search(s, n, size_t(from), mode == 4, true, true, caps)) return NativeResult::ok(UNIT);
            return NativeResult::ok(captures_value(p, caps));
        default: {
            // Every match, left to right. After an empty match the next search
            // starts a character on, so `a*` over "baa" finds "", "aa", "".
            std::vector<std::vector<int64_t>> found;
            int64_t count = 0;
            size_t at = size_t(from);
            while (at <= n && engine->search(s, n, at, false, true, mode == 2, caps)) {
                size_t start = size_t(caps[0]), stop = size_t(caps[1]);
                if (mode == 2) found.push_back(caps);
                ++count;
                if (stop > start) at = stop;
                else if (stop < n) { size_t w = 0; decode_char(s, n, stop, &w); at = stop + w; }
                else break;
            }
            if (mode == 3) return NativeResult::ok(make_integer(p, count));
            Value list = NIL;
            for (size_t i = found.size(); i-- > 0;) list = p.heap().make_cons(captures_value(p, found[i]), list);
            return NativeResult::ok(list);
        }
    }
}

}  // namespace dream

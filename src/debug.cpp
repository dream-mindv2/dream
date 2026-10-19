// The debugger. See debug.hpp for what it is for and what it answers.

#include "debug.hpp"

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "bigint.hpp"
#include "builtins.hpp"
#include "interp.hpp"
#include "process.hpp"

namespace dream {

namespace {

/// How a stopped session goes on.
enum class Going { Run, Step, Next, Finish };

struct Debugger {
    std::mutex mutex;
    bool on = false;
    /// Commands still to run, from the script.
    std::deque<std::string> pending;
    /// The terminal, when there is one to ask once the script runs out.
    std::FILE* tty = nullptr;
    std::set<std::string> breaks, traces, watches;
    /// Commands to run whenever a function of that name stops.
    std::map<std::string, std::vector<std::string>> on_stop;
    Going going = Going::Run;
    /// Where `next` and `finish` were asked: the process and how deep its
    /// continuations were.
    uint64_t at_pid = 0;
    size_t at_depth = 0;
    /// Calls traced and not yet answered: the process, the depth the answer
    /// comes back below, and the function.
    struct Open {
        uint64_t pid;
        size_t depth;
        uint32_t func;
    };
    std::vector<Open> open;
    /// Each function's body node, by image, to know a call is being entered.
    const Image* bodies_of = nullptr;
    std::unordered_map<uint32_t, uint32_t> body_func;
    /// What a debug build named each local (`LOCL`), by function and slot.
    std::unordered_map<uint64_t, std::string> local_name;
    /// `break FILE:LINE`s, and the functions they came to in this image: the
    /// one written last at or before the line, in a file whose path ends so.
    std::vector<std::pair<std::string, uint32_t>> line_breaks;
    std::set<uint32_t> break_funcs;
};

Debugger& dbg() {
    static Debugger d;
    return d;
}

std::string name_of(const Image& img, uint32_t fi) {
    if (fi >= img.func_count()) return "?";
    StringRef s = img.str(img.func(fi).name);
    return std::string(s.data, s.len);
}

std::string where_of(const Image& img, uint32_t fi) {
    if (fi >= img.func_count()) return "";
    const FuncRec& f = img.func(fi);
    if (f.source == 0 || f.source - 1 >= img.string_count()) return "";
    StringRef s = img.str(f.source - 1);
    std::string path(s.data, s.len);
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos) path = path.substr(slash + 1);
    return " (" + path + ":" + std::to_string(f.line) + ")";
}

/// The function a frame, or a flat suspension's closure, belongs to.
uint32_t func_of(Value v) {
    if (is_obj(v, ObjType::Closure)) return static_cast<ClosureObj*>(as_obj(v))->func;
    if (!is_obj(v, ObjType::Frame)) return UINT32_MAX;
    Value c = static_cast<FrameObj*>(as_obj(v))->closure;
    return is_obj(c, ObjType::Closure) ? static_cast<ClosureObj*>(as_obj(c))->func : UINT32_MAX;
}

std::string callee_name(Value callee) {
    if (is_builtin(callee)) return builtin_def(uint32_t(imm_payload(callee))).name;
    if (is_obj(callee, ObjType::Native)) {
        Bytes b;
        if (string_bytes(static_cast<NativeObj*>(as_obj(callee))->name, &b)) return std::string(b.data, b.len);
    }
    return "?";
}

/// A value as far as it is evaluated, `depth` levels into what it holds.
/// Forces nothing.
void show(Process& p, Value v, int depth, std::string* out) {
    v = resolve(v);
    const Image& img = *p.code;
    if (is_fixnum(v)) {
        *out += std::to_string(fixnum_value(v));
        return;
    }
    if (is_imm(v)) {
        switch (imm_kind(v)) {
            case IMM_UNIT: *out += "()"; return;
            case IMM_BOOL: *out += truthy(v) ? "true" : "false"; return;
            case IMM_CHAR: *out += "'" + std::string(1, char(imm_payload(v) & 0x7f)) + "'"; return;
            case IMM_ATOM: *out += ":" + p.runtime().atom_name(uint32_t(imm_payload(v))); return;
            case IMM_NIL: *out += "[]"; return;
            case IMM_BUILTIN: *out += "<builtin " + std::string(builtin_def(uint32_t(imm_payload(v))).name) + ">"; return;
            default: *out += "<value>"; return;
        }
    }
    if (!is_ptr(v)) {
        *out += "<value>";
        return;
    }
    Obj* o = as_obj(v);
    const bool deep = depth > 0;
    switch (o->type) {
        case ObjType::Float: {
            char b[32];
            std::snprintf(b, sizeof b, "%.17g", static_cast<FloatObj*>(o)->value);
            *out += b;
            return;
        }
        case ObjType::BigInt: bigint::append_decimal(v, out); return;
        case ObjType::Str: case ObjType::BigStr: {
            Bytes b;
            string_bytes(v, &b);
            const size_t n = b.len > 60 ? 60 : size_t(b.len);
            *out += "\"" + std::string(b.data, n) + (b.len > 60 ? "...\"" : "\"");
            return;
        }
        case ObjType::Cons: {
            if (!deep) { *out += "[..]"; return; }
            *out += "[";
            Value cur = v;
            for (int i = 0; is_obj(cur, ObjType::Cons); ++i) {
                if (i) *out += ", ";
                if (i == 8) { *out += ".."; break; }
                show(p, static_cast<ConsObj*>(as_obj(cur))->head, depth - 1, out);
                cur = resolve(static_cast<ConsObj*>(as_obj(cur))->tail);
                if (is_ptr(cur) && obj_type(as_obj(cur)) != ObjType::Cons) {
                    *out += ", ..";
                    show(p, cur, 0, out);
                    break;
                }
            }
            *out += "]";
            return;
        }
        case ObjType::Array: case ObjType::Record: {
            const bool rec = o->type == ObjType::Record;
            const uint32_t len = rec ? static_cast<RecordObj*>(o)->len : static_cast<ArrayObj*>(o)->len;
            Value* items = rec ? static_cast<RecordObj*>(o)->items() : static_cast<ArrayObj*>(o)->items();
            if (rec) {
                std::string name = p.runtime().atom_name(uint32_t(imm_payload(static_cast<RecordObj*>(o)->id())));
                const size_t dot = name.find_last_of('.');
                *out += (dot == std::string::npos ? name : name.substr(dot + 1)) + "(";
            } else {
                *out += "#[";
            }
            if (!deep && len) *out += "..";
            for (uint32_t i = 0; deep && i < len; ++i) {
                if (i) *out += ", ";
                if (i == 8) { *out += ".."; break; }
                show(p, items[i], depth - 1, out);
            }
            *out += rec ? ")" : "]";
            return;
        }
        case ObjType::Map: case ObjType::MapLeaf: {
            if (!deep) { *out += "%{..}"; return; }
            std::vector<std::pair<Value, Value>> es;
            map_collect(v, es);
            *out += "%{";
            for (size_t i = 0; i < es.size(); ++i) {
                if (i) *out += ", ";
                if (i == 6) { *out += ".. " + std::to_string(es.size()) + " entries"; break; }
                show(p, es[i].first, depth - 1, out);
                *out += " => ";
                show(p, es[i].second, depth - 1, out);
            }
            *out += "}";
            return;
        }
        case ObjType::Trunk: {
            const uint32_t n = trunk_length(v);
            *out += "trunk[";
            for (uint32_t i = 0; deep && i < n; ++i) {
                if (i) *out += ", ";
                if (i == 8) { *out += ".. " + std::to_string(n) + " in all"; break; }
                show(p, trunk_element(v, i), depth - 1, out);
            }
            if (!deep && n) *out += std::to_string(n) + " elements";
            *out += "]";
            return;
        }
        case ObjType::Closure:
            *out += "<fn " + name_of(img, static_cast<ClosureObj*>(o)->func) + ">";
            return;
        case ObjType::Pap:
            *out += "<partial ";
            show(p, static_cast<PapObj*>(o)->fn, 0, out);
            *out += ">";
            return;
        case ObjType::Thunk: {
            auto* t = static_cast<ThunkObj*>(o);
            if (!thunk_flat(t) && t->node == HOLE_NODE) { *out += "<not bound yet>"; return; }
            const uint32_t fi = func_of(t->frame);
            *out += "<lazy" + (fi == UINT32_MAX ? std::string() : " in " + name_of(img, fi)) + ">";
            return;
        }
        case ObjType::Blackhole: *out += "<being evaluated>"; return;
        case ObjType::ErrorBox: {
            auto* e = static_cast<ErrorObj*>(o);
            *out += "<error ";
            show(p, e->kind, 0, out);
            *out += " ";
            show(p, e->payload, depth - 1, out);
            *out += ">";
            return;
        }
        case ObjType::Pid: *out += "<process " + std::to_string(static_cast<PidObj*>(o)->id) + ">"; return;
        default: *out += std::string("<") + obj_type_name(o->type) + ">"; return;
    }
}

std::string shown(Process& p, Value v, int depth = 2) {
    std::string s;
    show(p, v, depth, &s);
    return s;
}

std::string path_of(const Image& img, uint32_t fi) {
    const FuncRec& f = img.func(fi);
    if (f.source == 0 || f.source - 1 >= img.string_count()) return "";
    StringRef s = img.str(f.source - 1);
    return std::string(s.data, s.len);
}

/// The functions `break FILE:LINE` stops in.
void place_breaks(Debugger& d, const Image& img) {
    d.break_funcs.clear();
    for (const auto& [file, line] : d.line_breaks) {
        uint32_t best = UINT32_MAX, best_line = 0;
        for (uint32_t fi = 0; fi < img.func_count(); ++fi) {
            const std::string path = path_of(img, fi);
            const FuncRec& f = img.func(fi);
            if (path.size() < file.size() || path.compare(path.size() - file.size(), file.size(), file) != 0) continue;
            if (f.line <= line && f.line >= best_line) { best = fi; best_line = f.line; }
        }
        if (best == UINT32_MAX) std::fprintf(stderr, "  no function at %s:%u\n", file.c_str(), line);
        else d.break_funcs.insert(best);
    }
}

void index_bodies(Debugger& d, const Image& img) {
    if (d.bodies_of == &img) return;
    d.bodies_of = &img;
    d.body_func.clear();
    for (uint32_t fi = 0; fi < img.func_count(); ++fi) d.body_func.emplace(img.func(fi).body, fi);
    d.local_name.clear();
    for (uint32_t i = 0; i < img.local_count(); ++i) {
        const LocalRec& r = img.local(i);
        if (r.name >= img.string_count()) continue;
        StringRef s = img.str(r.name);
        d.local_name[(uint64_t(r.func) << 32) | r.slot] = std::string(s.data, s.len);
    }
    place_breaks(d, img);
}

/// What a debug build named slot `slot` of function `fi`, or "".
std::string slot_name(Debugger& d, const Image& img, uint32_t fi, uint32_t slot) {
    index_bodies(d, img);
    auto it = d.local_name.find((uint64_t(fi) << 32) | slot);
    return it == d.local_name.end() ? std::string() : it->second;
}

/// The function a call is entering at this step, or UINT32_MAX.
uint32_t entering(Debugger& d, Process& p) {
    index_bodies(d, *p.code);
    auto it = d.body_func.find(p.node);
    return it == d.body_func.end() ? UINT32_MAX : it->second;
}

std::string args_of(Debugger& d, Process& p, uint32_t fi) {
    const Image& img = *p.code;
    std::string s;
    if (!is_obj(p.frame, ObjType::Frame) || fi >= img.func_count()) return s;
    auto* f = static_cast<FrameObj*>(as_obj(p.frame));
    const uint32_t arity = img.func(fi).arity;
    for (uint32_t i = 0; i < arity && i < f->nslots; ++i) {
        s += i ? ", " : "";
        const std::string name = slot_name(d, img, fi, i);
        if (!name.empty()) s += name + " = ";
        s += f->slots()[i] == NIL_SLOT ? "<unbound>" : shown(p, f->slots()[i], 1);
    }
    return s;
}

void backtrace(Process& p, size_t limit) {
    const Image& img = *p.code;
    uint32_t last = UINT32_MAX;
    size_t n = 0;
    const uint32_t here = func_of(p.frame);
    if (here != UINT32_MAX) {
        std::fprintf(stderr, "  #0 %s%s\n", name_of(img, here).c_str(), where_of(img, here).c_str());
        last = here;
        ++n;
    }
    for (size_t i = p.conts.size(); i > 0 && n < limit; --i) {
        const Cont& c = p.conts.begin()[i - 1];
        uint32_t fi = UINT32_MAX;
        const char* how = "";
        switch (c.kind) {
            case ContKind::UpdateThunk: {
                Obj* o = as_obj(c.v1);
                if (obj_type(o) == ObjType::Thunk || obj_type(o) == ObjType::Blackhole) {
                    fi = func_of(static_cast<ThunkObj*>(o)->frame);
                    how = "forcing a value made in ";
                }
                break;
            }
            case ContKind::IfBranch: case ContKind::BinRight: case ContKind::LogicRight:
            case ContKind::BlockNext: case ContKind::BindThen: case ContKind::Catch: case ContKind::Finalize:
            case ContKind::NativeArgs: case ContKind::ClosureArgs: case ContKind::IndexKey: case ContKind::IndexApply:
            case ContKind::GetWalk: case ContKind::SetWalk: case ContKind::SwitchOn: case ContKind::SwitchKey:
            case ContKind::EnterRetry: case ContKind::SetManyStep:
                fi = func_of(c.v1);
                break;
            default: break;
        }
        if (fi == UINT32_MAX || (fi == last && !*how)) continue;
        std::fprintf(stderr, "  #%zu %s%s%s\n", n, how, name_of(img, fi).c_str(), where_of(img, fi).c_str());
        last = fi;
        ++n;
    }
}

void locals(Debugger& d, Process& p, int depth) {
    if (!is_obj(p.frame, ObjType::Frame)) {
        std::fprintf(stderr, "  no frame\n");
        return;
    }
    auto* f = static_cast<FrameObj*>(as_obj(p.frame));
    const uint32_t fi = func_of(p.frame);
    for (uint32_t i = 0; i < f->nslots; ++i) {
        if (f->slots()[i] == NIL_SLOT) continue;
        const std::string name = slot_name(d, *p.code, fi, i);
        std::fprintf(stderr, "  s%u%s = %s\n", i, name.empty() ? "" : (" " + name).c_str(),
                     shown(p, f->slots()[i], depth).c_str());
    }
    if (is_obj(f->closure, ObjType::Closure)) {
        auto* c = static_cast<ClosureObj*>(as_obj(f->closure));
        for (uint32_t i = 0; i < c->ncaps; ++i)
            std::fprintf(stderr, "  c%u = %s\n", i, shown(p, c->caps()[i], depth).c_str());
    }
}

void print_slot(Debugger& d, Process& p, const std::string& asked, int depth) {
    if (asked.empty() || !is_obj(p.frame, ObjType::Frame)) {
        std::fprintf(stderr, "  print NAME, sN or cN\n");
        return;
    }
    auto* f = static_cast<FrameObj*>(as_obj(p.frame));
    // A name a debug build gave a slot of this function, else `sN`/`cN`.
    std::string which = asked;
    const uint32_t fi = func_of(p.frame);
    for (uint32_t i = 0; i < f->nslots; ++i) {
        if (slot_name(d, *p.code, fi, i) == asked) {
            which = "s" + std::to_string(i);
            break;
        }
    }
    if (which.size() < 2) {
        std::fprintf(stderr, "  no %s here\n", asked.c_str());
        return;
    }
    const uint32_t i = uint32_t(std::strtoul(which.c_str() + 1, nullptr, 10));
    if (which[0] == 's' && i < f->nslots) {
        std::fprintf(stderr, "  %s = %s\n", asked.c_str(),
                     f->slots()[i] == NIL_SLOT ? "<unbound>" : shown(p, f->slots()[i], depth).c_str());
    } else if (which[0] == 'c' && is_obj(f->closure, ObjType::Closure) &&
               i < static_cast<ClosureObj*>(as_obj(f->closure))->ncaps) {
        std::fprintf(stderr, "  %s = %s\n", which.c_str(),
                     shown(p, static_cast<ClosureObj*>(as_obj(f->closure))->caps()[i], depth).c_str());
    } else {
        std::fprintf(stderr, "  no %s here\n", which.c_str());
    }
}

/// The next command: the script's, then the terminal's, or "continue" when
/// there is neither.
std::string next_command(Debugger& d) {
    if (!d.pending.empty()) {
        std::string c = d.pending.front();
        d.pending.pop_front();
        return c;
    }
    if (!d.tty) return "continue";
    std::fprintf(stderr, "(dream) ");
    std::fflush(stderr);
    char line[1024];
    if (!std::fgets(line, sizeof line, d.tty)) return "continue";
    std::string s(line);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

/// Split a script into commands at `;` and new lines.
void queue_script(Debugger& d, const std::string& script, std::deque<std::string>* into) {
    std::string cur;
    for (char ch : script) {
        if (ch == ';' || ch == '\n') {
            if (!words(cur).empty()) into->push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    if (!words(cur).empty()) into->push_back(cur);
    (void)d;
}

/// One command. True when it lets the program go on.
bool command(Debugger& d, Process& p, const std::string& line) {
    std::vector<std::string> w = words(line);
    if (w.empty()) return false;
    const std::string& c = w[0];
    const std::string arg = w.size() > 1 ? w[1] : "";
    const int depth = w.size() > 2 ? std::atoi(w[2].c_str()) : 2;
    if (c == "c" || c == "continue") { d.going = Going::Run; return true; }
    if (c == "s" || c == "step") { d.going = Going::Step; return true; }
    if (c == "n" || c == "next") {
        d.going = Going::Next;
        d.at_pid = p.id();
        d.at_depth = p.conts.size();
        return true;
    }
    if (c == "finish") {
        d.going = Going::Finish;
        d.at_pid = p.id();
        d.at_depth = p.conts.size();
        return true;
    }
    if (c == "q" || c == "quit") std::exit(0);
    if (c == "bt" || c == "backtrace") { backtrace(p, arg.empty() ? 12 : size_t(std::atoi(arg.c_str()))); return false; }
    if (c == "locals" || c == "frame") { locals(d, p, arg.empty() ? 1 : std::atoi(arg.c_str())); return false; }
    if (c == "p" || c == "print") { print_slot(d, p, arg, depth); return false; }
    if (c == "node") {
        const Node& n = p.code->node(p.node);
        std::fprintf(stderr, "  node %u: %s a=%u b=%u c=%u\n", p.node, op_name(Op(n.op)), n.a, n.b, n.c);
        return false;
    }
    if (c == "b" || c == "break") {
        // `FILE:LINE`, or a function's name.
        const size_t colon = arg.rfind(':');
        if (colon != std::string::npos && colon + 1 < arg.size() && std::isdigit(uint8_t(arg[colon + 1]))) {
            d.line_breaks.push_back({arg.substr(0, colon), uint32_t(std::stoul(arg.substr(colon + 1)))});
            place_breaks(d, *p.code);
        } else {
            d.breaks.insert(arg);
        }
        return false;
    }
    if (c == "trace") { d.traces.insert(arg); return false; }
    if (c == "watch") { d.watches.insert(arg); return false; }
    if (c == "delete") { d.breaks.erase(arg); d.traces.erase(arg); d.watches.erase(arg); d.on_stop.erase(arg); return false; }
    if (c == "on" && w.size() > 2) {
        // The rest of the line, up to the end of the script's command: what
        // `on` runs is everything after its name, `,`-separated.
        const std::string rest = line.substr(line.find(arg) + arg.size());
        std::vector<std::string>& cmds = d.on_stop[arg];
        cmds.clear();
        std::string cur;
        for (char ch : rest + ",") {
            if (ch == ',') {
                if (!words(cur).empty()) cmds.push_back(cur);
                cur.clear();
            } else {
                cur += ch;
            }
        }
        d.breaks.insert(arg);
        return false;
    }
    std::fprintf(stderr, "  commands: break trace watch on delete continue step next finish bt locals print node quit\n");
    return false;
}

/// Stopped: say where, run what was asked for here, then ask.
void session(Debugger& d, Process& p, const char* why) {
    const Image& img = *p.code;
    const uint32_t fi = func_of(p.frame);
    std::fprintf(stderr, "-- %s: %s%s, process %" PRIu64 "\n", why, name_of(img, fi).c_str(),
                 where_of(img, fi).c_str(), p.id());
    auto on = d.on_stop.find(name_of(img, fi));
    if (on != d.on_stop.end()) {
        for (const std::string& c : on->second) {
            if (command(d, p, c)) return;
        }
        return;
    }
    for (;;) {
        if (command(d, p, next_command(d))) return;
    }
}

}  // namespace

bool g_debugging = false;

void debug_start(const std::string& script) {
    Debugger& d = dbg();
    d.on = true;
    g_debugging = true;
    // Every command waits for the first step, which stops: what the script
    // sets up is in place before the program does anything, and whatever it
    // says after that runs there too.
    queue_script(d, script, &d.pending);
    d.tty = std::fopen("/dev/tty", "r");
    d.going = Going::Step;
}

void debug_eval(Process& p) {
    Debugger& d = dbg();
    std::lock_guard<std::mutex> g(d.mutex);
    const uint32_t fi = entering(d, p);
    const std::string name = fi == UINT32_MAX ? std::string() : name_of(*p.code, fi);
    if (fi != UINT32_MAX && d.traces.count(name)) {
        std::fprintf(stderr, "-> %s(%s)\n", name.c_str(), args_of(d, p, fi).c_str());
        d.open.push_back({p.id(), p.conts.size(), fi});
    }
    const char* why = nullptr;
    if (d.going == Going::Step) why = "step";
    else if (d.going == Going::Next && p.id() == d.at_pid && p.conts.size() <= d.at_depth) why = "next";
    else if (fi != UINT32_MAX && (d.breaks.count(name) || d.break_funcs.count(fi))) why = "break";
    if (why) session(d, p, why);
}

void debug_return(Process& p) {
    Debugger& d = dbg();
    if (d.open.empty() && d.going != Going::Finish) return;
    std::lock_guard<std::mutex> g(d.mutex);
    for (size_t i = d.open.size(); i > 0; --i) {
        const Debugger::Open o = d.open[i - 1];
        if (o.pid != p.id() || p.conts.size() >= o.depth) continue;
        std::fprintf(stderr, "<- %s = %s\n", name_of(*p.code, o.func).c_str(), shown(p, p.result).c_str());
        d.open.erase(d.open.begin() + long(i - 1));
    }
    if (d.going == Going::Finish && p.id() == d.at_pid && p.conts.size() < d.at_depth) {
        std::fprintf(stderr, "   answered %s\n", shown(p, p.result).c_str());
        d.going = Going::Step;
    }
}

void debug_native(Process& p, Value callee, const Value* args, uint32_t argc, const NativeResult& r) {
    Debugger& d = dbg();
    if (d.watches.empty()) return;
    std::lock_guard<std::mutex> g(d.mutex);
    const std::string name = callee_name(callee);
    if (!d.watches.count(name)) return;
    std::string s = name + "(";
    for (uint32_t i = 0; i < argc; ++i) s += (i ? ", " : "") + shown(p, args[i], 1);
    s += ")";
    switch (r.outcome) {
        case NativeOutcome::Value: case NativeOutcome::Enter: s += " = " + shown(p, r.value); break;
        case NativeOutcome::Raise: s += " raised " + shown(p, r.value); break;
        default: s += " waits"; break;
    }
    std::fprintf(stderr, "   %s\n", s.c_str());
}

}  // namespace dream

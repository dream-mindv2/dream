// LLVM IR text in, an object file out: the one thing the AOT compiler cannot do
// in Dream without writing an instruction selector and an optimizer of its own.
//
// It is here rather than in a program of its own so that building a native
// executable needs nothing beyond the toolchain that was already installed: the
// VM carries LLVM for its JIT, and no clang or llc has to be found. It shares no
// state with the JIT, and the AOT's output owes nothing to the VM -- this is a
// service the compiler calls, not a part of what it compiles.
//
// See dreams/docs/aot.md.

#include "jit.hpp"
#include "digest.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cctype>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Config/llvm-config.h"

#ifdef DREAM_AOT_CROSS
extern "C" {
void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();
void LLVMInitializeAArch64TargetInfo();
void LLVMInitializeAArch64Target();
void LLVMInitializeAArch64TargetMC();
void LLVMInitializeAArch64AsmPrinter();
void LLVMInitializeAArch64AsmParser();
}
#endif

namespace dream {

namespace {

void initialize_targets() {
    static std::once_flag once;
    std::call_once(once, [] {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        llvm::InitializeNativeTargetAsmParser();
#ifdef DREAM_AOT_CROSS
        // The other machines `dreams --aot --target` writes for (`DREAM_AOT_CROSS` in CMake).
        LLVMInitializeX86TargetInfo();
        LLVMInitializeX86Target();
        LLVMInitializeX86TargetMC();
        LLVMInitializeX86AsmPrinter();
        LLVMInitializeX86AsmParser();
        LLVMInitializeAArch64TargetInfo();
        LLVMInitializeAArch64Target();
        LLVMInitializeAArch64TargetMC();
        LLVMInitializeAArch64AsmPrinter();
        LLVMInitializeAArch64AsmParser();
#endif
    });
}

/// The machine `module` says it is for, at `opt_level`, the module told its layout.
std::unique_ptr<llvm::TargetMachine> machine_for(llvm::Module& module, int opt_level, std::string* error) {
    // The module's own triple wins, so that the compiler is what says what it
    // generated for; only the host's target is linked in today, and asking for
    // another is an answer from the registry rather than a crash.
    llvm::Triple triple(module.getTargetTriple().str().empty()
                            ? llvm::Triple(llvm::sys::getDefaultTargetTriple())
                            : module.getTargetTriple());
    std::string lookup;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget("", triple, lookup);
    if (!target) {
        *error = lookup;
        return nullptr;
    }
    const auto level = opt_level <= 0   ? llvm::CodeGenOptLevel::None
                       : opt_level == 1 ? llvm::CodeGenOptLevel::Less
                                        : llvm::CodeGenOptLevel::Default;
    // Static and not position-independent: the executable writer lays out a
    // fixed address space and resolves the relocations itself, and PIC would only
    // add a GOT for it to build.
    // A section for each function and each datum, so that the executable writer can leave out
    // what nothing reaches: every program carries the whole runtime and library, compiled once.
    llvm::TargetOptions options;
    options.FunctionSections = true;
    options.DataSections = true;
    // x86-64-v2 rather than the first x86-64: POPCNT, SSE4.2 and CMPXCHG16B, which every x86-64
    // made since 2009 has, and which the platforms the toolchain is built for require already
    // (AlmaLinux 9). A map's slot is a popcount of its bitmap, and on the first x86-64 that was
    // a dozen instructions on every step down a map.
    const char* cpu = triple.getArch() == llvm::Triple::x86_64 ? "x86-64-v2" : "generic";
    std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
        triple, cpu, "", options, llvm::Reloc::Static, llvm::CodeModel::Small, level));
    if (!machine) {
        *error = "no target machine for " + triple.str();
        return nullptr;
    }
    module.setTargetTriple(triple);
    module.setDataLayout(machine->createDataLayout());
    return machine;
}

/// `module` optimized at `opt_level` and written as an object.
bool compiled(llvm::Module& module, llvm::TargetMachine& machine, int opt_level, std::string* object,
              std::string* error) {
    if (opt_level > 0) {
        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;
        llvm::PassBuilder builder(&machine);
        builder.registerModuleAnalyses(mam);
        builder.registerCGSCCAnalyses(cgam);
        builder.registerFunctionAnalyses(fam);
        builder.registerLoopAnalyses(lam);
        builder.crossRegisterProxies(lam, fam, cgam, mam);
        const auto o = opt_level == 1 ? llvm::OptimizationLevel::O1 : llvm::OptimizationLevel::O2;
        builder.buildPerModuleDefaultPipeline(o).run(module, mam);
    }
    llvm::SmallVector<char, 0> bytes;
    llvm::raw_svector_ostream stream(bytes);
    llvm::legacy::PassManager emit;
    if (machine.addPassesToEmitFile(emit, stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        *error = "this target cannot write an object file";
        return false;
    }
    emit.run(module);
    object->assign(bytes.data(), bytes.size());
    return true;
}

/// `ir` parsed and checked, in `context`.
std::unique_ptr<llvm::Module> parsed(std::string_view ir, llvm::LLVMContext& context, std::string* error) {
    llvm::SMDiagnostic diagnostic;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(ir.data(), ir.size()), "aot",
                                                   /*RequiresNullTerminator=*/false);
    std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
    if (!module) {
        *error = "line " + std::to_string(diagnostic.getLineNo()) + ": " + diagnostic.getMessage().str();
        return nullptr;
    }
    std::string problems;
    llvm::raw_string_ostream problem_stream(problems);
    if (llvm::verifyModule(*module, &problem_stream)) {
        *error = problems;
        return nullptr;
    }
    return module;
}

}  // namespace

bool aot_object(std::string_view ir, int opt_level, std::string* object, std::string* error) {
    initialize_targets();
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module = parsed(ir, context, error);
    if (!module) return false;
    std::unique_ptr<llvm::TargetMachine> machine = machine_for(*module, opt_level, error);
    if (!machine) return false;
    return compiled(*module, *machine, opt_level, object, error);
}

// A program the size of a compiler is hundreds of thousands of functions, and LLVM optimizes a
// module on the thread that asked: one module of `dreams` was twenty minutes on one core of
// twenty-four, and more memory than the machine could spare. So the text is cut into parts at its
// definitions, and each part is read, optimized and compiled on a thread of its own, in a context
// of its own. Cut as text, not as a module: `SplitModule` made each part a clone of the whole, every
// one of half a million declarations included, and splitting dre into twenty-four took twice as
// long as compiling it in one. What is lost is only an optimization across two parts: nothing is
// inlined from one into another.
//
// What the AOT writes is simple at the top level, and this reads only that: a `define` runs to its
// closing `}`, everything else is a line. The target, the declarations and the payload's external
// symbol go in every part, `module asm` in the first; a function or a datum goes in the part with
// the least so far, an alias with what it names. Every definition is made `hidden` rather than
// `internal` or `private`, so that the others can name it, and each part declares what it names
// and does not define: a function as its definition says, a datum as a byte, which is all a
// reference to one needs.
//
// Parts kept. What stands between `; dream: cached` and `; dream: end cached` -- the runtime, the
// same text for every program (`machine.to_ir`) -- goes in parts of its own, cut by its size
// alone, which take nothing from outside it, declare in a fixed order and carry only the header
// lines they use, so that each part is the same text in every program. Such a part, once compiled,
// is kept in `DREAMS_AOT_CACHE` (`~/.cache/dream/aot` by default, `0` for none) under the SHA-256
// of its text, LLVM's version and the level, and read back the next time instead.

namespace {

struct Top {
    std::string_view text;
    std::string name;  // what it defines, or empty
    enum Kind { Header, Asm, Function, Datum, Alias } kind;
    std::string aliasee;
    bool cached = false;  // between `; dream: cached` and `; dream: end cached`
};

bool starts(std::string_view s, std::string_view p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }

bool name_char(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '$'; }

std::string name_after_at(std::string_view s, size_t at) {
    size_t e = at + 1;
    while (e < s.size() && name_char(s[e])) ++e;
    return std::string(s.substr(at + 1, e - at - 1));
}

/// The text's top-level items, in order.
std::vector<Top> tops_of(std::string_view ir) {
    std::vector<Top> out;
    size_t i = 0;
    bool cached = false;
    auto push = [&](Top t) {
        t.cached = cached;
        out.push_back(std::move(t));
    };
    while (i < ir.size()) {
        size_t eol = ir.find('\n', i);
        if (eol == std::string_view::npos) eol = ir.size();
        std::string_view line = ir.substr(i, eol - i);
        if (starts(line, "define ")) {
            // To the line that is only `}`.
            size_t end = ir.find("\n}\n", eol);
            end = end == std::string_view::npos ? ir.size() : end + 3;
            const size_t at = line.find('@');
            push({ir.substr(i, end - i), at == std::string_view::npos ? "" : name_after_at(line, at), Top::Function, ""});
            i = end;
            continue;
        }
        const size_t next = eol < ir.size() ? eol + 1 : eol;
        std::string_view item = ir.substr(i, next - i);
        if (line == "; dream: cached") {
            cached = true;
        } else if (line == "; dream: end cached") {
            cached = false;
        } else if (line.empty() || line[0] == ';') {
            // nothing
        } else if (starts(line, "module asm")) {
            push({item, "", Top::Asm, ""});
        } else if (line[0] == '@' && line.find(" = external ") == std::string_view::npos) {
            const std::string name = name_after_at(line, 0);
            if (line.find(" alias ") != std::string_view::npos) {
                const size_t at = line.rfind('@');
                push({item, name, Top::Alias, name_after_at(line, at)});
            } else {
                push({item, name, Top::Datum, ""});
            }
        } else {
            push({item, "", Top::Header, ""});
        }
        i = next;
    }
    return out;
}

/// `text` with the linkage of the definition it starts with made `hidden`.
std::string hidden(std::string_view text) {
    std::string out(text);
    const size_t eol = out.find('\n');
    const size_t limit = eol == std::string::npos ? out.size() : eol;
    for (const char* linkage : {" internal ", " private "}) {
        const size_t at = out.find(linkage);
        if (at != std::string::npos && at < limit) {
            out.replace(at, std::strlen(linkage), " hidden ");
            break;
        }
    }
    return out;
}

/// A function's declaration, from its definition's first line.
std::string declaration_of(std::string_view define) {
    std::string_view line = define.substr(0, define.find('\n'));
    const size_t at = line.find('@');
    const size_t open = line.find('(', at);
    const size_t close = line.find(')', open);
    // What comes between `define` and the name, but its linkage.
    std::string head(line.substr(7, at - 7));
    for (const char* linkage : {"internal ", "private ", "dso_local ", "hidden "}) {
        const size_t k = head.find(linkage);
        if (k != std::string::npos) head.erase(k, std::strlen(linkage));
    }
    std::string params;
    std::string_view list = line.substr(open + 1, close - open - 1);
    size_t p = 0;
    while (p < list.size()) {
        size_t comma = list.find(',', p);
        if (comma == std::string_view::npos) comma = list.size();
        std::string_view one = list.substr(p, comma - p);
        while (!one.empty() && one.front() == ' ') one.remove_prefix(1);
        const size_t pct = one.rfind(" %");
        if (pct != std::string_view::npos) one = one.substr(0, pct);
        if (!params.empty()) params += ", ";
        params += std::string(one);
        p = comma + 1;
    }
    return "declare hidden " + head + "@" + std::string(line.substr(at + 1, open - at - 1)) + "(" + params + ")\n";
}

/// Every name `text` refers to with `@`, outside quoted strings.
void names_in(std::string_view text, std::unordered_set<std::string>* out) {
    bool quoted = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '"') quoted = !quoted;
        else if (!quoted && c == '@') {
            std::string name = name_after_at(text, i);
            if (!name.empty()) out->insert(std::move(name));
        }
    }
}

/// `text` with every name `renames` holds replaced by what it says, outside quoted strings.
std::string renamed(std::string_view text, const std::unordered_map<std::string, std::string>& renames) {
    if (renames.empty()) return std::string(text);
    std::string out;
    out.reserve(text.size());
    bool quoted = false;
    for (size_t i = 0; i < text.size();) {
        const char c = text[i];
        if (c == '"') quoted = !quoted;
        if (!quoted && c == '@') {
            std::string name = name_after_at(text, i);
            auto it = renames.find(name);
            out += '@';
            out += it == renames.end() ? name : it->second;
            i += 1 + name.size();
            continue;
        }
        out += c;
        ++i;
    }
    return out;
}

struct Part {
    std::string text;
    bool cached;  // the runtime's: the same for every program, and kept (`aot_objects`)
};

std::vector<std::string> sorted(const std::unordered_set<std::string>& names) {
    std::vector<std::string> out(names.begin(), names.end());
    std::sort(out.begin(), out.end());
    return out;
}

/// The text cut into `units` parts, each standing alone, and the runtime's section into parts of
/// its own.
//
// An alias is not written: whatever names one names what it is an alias of. The AOT writes a
// continuation it has written before as an alias of the first (`finish` in machine.dr), and in
// `dreams` that was 107,000 aliases of a few dozen functions, all in the part with them; LLVM's
// GlobalOpt took three minutes over that part alone, and the rest of its work a second.
std::vector<Part> parts_of(std::string_view ir, int units) {
    std::vector<Top> tops = tops_of(ir);
    std::unordered_map<std::string, std::string> renames;
    for (const Top& t : tops) {
        if (t.kind == Top::Alias) renames[t.name] = t.aliasee;
    }
    for (auto& [name, target] : renames) {
        // An alias of an alias is one of what that one is of.
        for (auto it = renames.find(target); it != renames.end() && it->second != target; it = renames.find(target)) {
            target = it->second;
        }
    }
    std::unordered_map<std::string, size_t> defined_at;  // name -> index in tops
    for (size_t i = 0; i < tops.size(); ++i) {
        if (!tops[i].name.empty()) defined_at[tops[i].name] = i;
    }
    std::string header;
    std::unordered_set<std::string> declared;
    for (const Top& t : tops) {
        if (t.kind == Top::Header) {
            header += std::string(t.text);
            names_in(t.text, &declared);
        }
    }
    // Where each definition goes: the lightest part, an alias with what it names. The runtime's
    // section has parts of its own, as many as its size says, so that they are the same text in
    // every program; the program's go in `units`.
    size_t cached_bytes = 0, own_bytes = 0;
    for (const Top& t : tops) {
        if (t.kind == Top::Function || t.kind == Top::Datum) (t.cached ? cached_bytes : own_bytes) += t.text.size();
    }
    // The parts asked for are for what the text is in all; the program's own share is what
    // they are for, a part for each half megabyte of it, and the kept ones are counted apart.
    units = std::max(1, std::min(units, int(1 + own_bytes / 524288)));
    const int kept = cached_bytes == 0 ? 0 : int(1 + cached_bytes / 524288);
    const int all = units + kept;
    std::vector<size_t> weight(size_t(all), 0);
    std::vector<int> unit_of(tops.size(), -1);
    for (size_t i = 0; i < tops.size(); ++i) {
        if (tops[i].kind == Top::Function || tops[i].kind == Top::Datum) {
            auto from = weight.begin() + (tops[i].cached ? units : 0);
            auto to = tops[i].cached ? weight.end() : weight.begin() + units;
            const size_t u = size_t(std::min_element(from, to) - weight.begin());
            unit_of[i] = int(u);
            weight[u] += tops[i].text.size();
        } else if (tops[i].kind == Top::Asm) {
            unit_of[i] = 0;
        }
    }

    std::vector<std::string> bodies(static_cast<size_t>(all));
    std::vector<std::unordered_set<std::string>> defines(static_cast<size_t>(all));
    for (size_t i = 0; i < tops.size(); ++i) {
        if (unit_of[i] < 0) continue;
        const size_t u = size_t(unit_of[i]);
        bodies[u] += tops[i].kind == Top::Asm ? std::string(tops[i].text) : renamed(hidden(tops[i].text), renames);
        if (!tops[i].name.empty()) defines[u].insert(tops[i].name);
    }
    // A small function of the runtime or its library that a part calls and another defines goes
    // in as `available_externally`, so LLVM can inline it where it is called and writes it nowhere:
    // its tests of a kind and its constants were calls once the parts were apart, and a compile
    // spent a sixth of its time in them, twice the time it took in one module.
    auto inlinable = [&](const std::string& name, const Top& t) {
        return t.kind == Top::Function && t.text.size() <= 4096 && name.size() > 1 &&
               (name[0] == 'r' || name[0] == 'l') && std::isdigit(static_cast<unsigned char>(name[1]));
    };
    // A runtime part's text is the same in every program only if nothing about it is the
    // program's: it copies in nothing from outside its section, its declarations and copies come
    // in an order of their own, and of the header it has only the lines it uses.
    std::vector<Part> parts;
    for (size_t u = 0; u < size_t(all); ++u) {
        const bool cached = int(u) >= units;
        std::unordered_set<std::string> used;
        names_in(bodies[u], &used);
        std::string copies;
        std::vector<std::string> todo = sorted(used);
        std::reverse(todo.begin(), todo.end());
        std::unordered_set<std::string> copied;
        while (!todo.empty()) {
            std::string name = std::move(todo.back());
            todo.pop_back();
            if (defines[u].count(name) || copied.count(name)) continue;
            auto it = defined_at.find(name);
            if (it == defined_at.end() || !inlinable(name, tops[it->second])) continue;
            if (cached && !tops[it->second].cached) continue;
            copied.insert(name);
            std::string copy = renamed(hidden(tops[it->second].text), renames);
            const size_t at = copy.find("define hidden ");
            if (at != std::string::npos) copy.replace(at, 14, "define available_externally ");
            std::unordered_set<std::string> more;
            names_in(copy, &more);
            std::vector<std::string> fresh;
            for (const std::string& m : sorted(more)) {
                if (!used.count(m)) {
                    used.insert(m);
                    fresh.push_back(m);
                }
            }
            for (auto f = fresh.rbegin(); f != fresh.rend(); ++f) todo.push_back(*f);
            copies += copy;
        }
        std::string declarations;
        for (const std::string& name : sorted(used)) {
            if (defines[u].count(name) || declared.count(name) || copied.count(name)) continue;
            auto it = defined_at.find(name);
            if (it == defined_at.end()) continue;  // the verifier says what is missing
            const Top& t = tops[it->second];
            if (t.kind == Top::Function) {
                declarations += declaration_of(t.text);
            } else if (t.kind == Top::Alias) {
                auto target = defined_at.find(t.aliasee);
                if (target != defined_at.end() && tops[target->second].kind == Top::Function) {
                    std::string d = declaration_of(tops[target->second].text);
                    const size_t at = d.find('@');
                    d.replace(at + 1, d.find('(', at) - at - 1, name);
                    declarations += d;
                }
            } else {
                declarations += "@" + name + " = external hidden global i8, align 8\n";
            }
        }
        std::string head = header;
        if (cached) {
            // Only the header's lines this part names, and those naming nothing.
            head.clear();
            for (const Top& t : tops) {
                if (t.kind != Top::Header) continue;
                std::unordered_set<std::string> names;
                names_in(t.text, &names);
                bool needed = names.empty();
                for (const std::string& n : names) needed = needed || used.count(n);
                if (needed) head += std::string(t.text);
            }
        }
        parts.push_back({head + declarations + copies + bodies[u], cached});
    }
    return parts;
}

/// Where compiled runtime parts are kept: `DREAMS_AOT_CACHE`, which is off when empty or `0`,
/// or `$XDG_CACHE_HOME/dream/aot`, or `~/.cache/dream/aot`.
std::string cache_dir() {
    if (const char* set = std::getenv("DREAMS_AOT_CACHE")) {
        const std::string dir(set);
        return dir == "0" ? "" : dir;
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) return std::string(xdg) + "/dream/aot";
    if (const char* home = std::getenv("HOME"); home && *home) return std::string(home) + "/.cache/dream/aot";
    return "";
}

/// A part's key: its text, as LLVM at this version and level would compile it.
std::string part_key(const std::string& text, int opt_level) {
    Sha256 sha;
    std::string salt = "dream-aot-part-3";
    salt += '\0';
    salt += LLVM_VERSION_STRING;
    salt += '\0';
    salt += std::to_string(opt_level);
    salt += '\0';
    sha.update(reinterpret_cast<const uint8_t*>(salt.data()), salt.size());
    sha.update(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return sha.hex();
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream all;
    all << in.rdbuf();
    *out = all.str();
    return !out->empty();
}

/// `bytes` at `path`, written beside it first and moved there, so that a reader never sees half.
void write_file(const std::string& dir, const std::string& path, const std::string& bytes) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string tmp = path + ".tmp" + std::to_string(::getpid()) + "-" +
                            std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    {
        std::ofstream out(tmp, std::ios::binary);
        if (!out) return;
        out.write(bytes.data(), std::streamsize(bytes.size()));
        if (!out) return;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

}  // namespace

bool aot_objects(std::string_view ir, int opt_level, int units, std::vector<std::string>* objects, size_t* kept,
                 std::string* error) {
    objects->clear();
    initialize_targets();
    if (units < 1) units = 1;
    std::vector<Part> parts = parts_of(ir, units);
    const std::string dir = cache_dir();
    // The kept parts come last (`parts_of`): how many, so that what links them can keep what it
    // reads of their objects too.
    *kept = size_t(std::count_if(parts.begin(), parts.end(), [](const Part& part) { return part.cached; }));
    // `DREAMS_AOT_KEEP_PARTS=DIR`: each part's text as `DIR/part-N.ll`, for `opt` to say where a
    // slow one spends its time.
    if (const char* keep = std::getenv("DREAMS_AOT_KEEP_PARTS")) {
        for (size_t i = 0; i < parts.size(); ++i) {
            const std::string path = std::string(keep) + "/part-" + std::to_string(i) + ".ll";
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(parts[i].text.data(), 1, parts[i].text.size(), f);
                std::fclose(f);
            }
        }
    }
    objects->assign(parts.size(), std::string());
    std::vector<std::string> errors(parts.size());
    std::atomic<size_t> next{0};
    // `DREAMS_AOT_TRACE`: each part's size, its largest function and how long it took, for when
    // one part is the build's critical path.
    const bool trace = std::getenv("DREAMS_AOT_TRACE") != nullptr;
    auto work = [&] {
        for (size_t i = next++; i < parts.size(); i = next++) {
            const auto started = std::chrono::steady_clock::now();
            const std::string& text = parts[i].text;
            const size_t bytes = text.size();
            // A runtime part compiled before is read back: the same text, level and LLVM made it.
            std::string path;
            if (parts[i].cached && !dir.empty()) {
                path = dir + "/" + part_key(text, opt_level) + ".o";
                if (read_file(path, &(*objects)[i])) {
                    if (trace) std::fprintf(stderr, "aot part %zu: %zu bytes, kept\n", i, bytes);
                    continue;
                }
            }
            std::string largest;
            size_t largest_size = 0;
            if (trace) {
                for (size_t at = text.find("\ndefine "); at != std::string::npos; ) {
                    const size_t end = text.find("\n}\n", at);
                    const size_t size = (end == std::string::npos ? text.size() : end) - at;
                    if (size > largest_size) {
                        largest_size = size;
                        const size_t name = text.find('@', at);
                        largest = text.substr(name, text.find('(', name) - name);
                    }
                    at = end == std::string::npos ? end : text.find("\ndefine ", end);
                }
            }
            llvm::LLVMContext context;
            std::unique_ptr<llvm::Module> module = parsed(text, context, &errors[i]);
            if (!module) {
                errors[i] = "part " + std::to_string(i) + ", " + errors[i];
                continue;
            }
            std::unique_ptr<llvm::TargetMachine> machine = machine_for(*module, opt_level, &errors[i]);
            if (!machine) continue;
            if (compiled(*module, *machine, opt_level, &(*objects)[i], &errors[i]) && !path.empty()) {
                write_file(dir, path, (*objects)[i]);
            }
            std::string().swap(parts[i].text);
            if (trace) {
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                std::fprintf(stderr, "aot part %zu: %zu bytes, largest %s (%zu bytes), %.1f s\n", i, bytes,
                             largest.c_str(), largest_size, seconds);
            }
        }
    };
    // Each thread holds a part as LLVM optimizes it, and a part of a large program is a large
    // module: more of them at once than this is memory for no time.
    const unsigned cores = std::min(12u, std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < std::min<unsigned>(cores, unsigned(parts.size())); ++t) threads.emplace_back(work);
    for (auto& t : threads) t.join();
    for (const std::string& e : errors) {
        if (!e.empty()) {
            *error = e;
            return false;
        }
    }
    return true;
}

}  // namespace dream

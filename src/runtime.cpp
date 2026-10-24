#include "runtime.hpp"

#include "crypto.hpp"
#include "io.hpp"
#include "os.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

#include "builtins.hpp"
#include "tensor.hpp"
#include "jit.hpp"
#include "process.hpp"
#include "scheduler.hpp"

namespace dream {

Runtime::Runtime(bool owns_host_services)
    : wk_(std::make_unique<WellKnownAtoms>()), owns_host_services_(owns_host_services) {
    // Intern the atoms the runtime itself names, before any image atoms, so
    // their indices are stable no matter what is loaded.
    wk_->error = intern_atom("error");
    wk_->divide_by_zero = intern_atom("divide_by_zero");
    wk_->type_error = intern_atom("type_error");
    wk_->not_a_function = intern_atom("not_a_function");
    wk_->no_such_member = intern_atom("no_such_member");
    wk_->loop = intern_atom("loop");
    wk_->stack_overflow = intern_atom("stack_overflow");
    wk_->out_of_memory = intern_atom("out_of_memory");
    wk_->killed = intern_atom("killed");
    wk_->normal = intern_atom("normal");
    wk_->timeout = intern_atom("timeout");
    wk_->ok = intern_atom("ok");
    wk_->out_of_bounds = intern_atom("out_of_bounds");
    wk_->no_such_key = intern_atom("no_such_key");
    // The names `type_of` answers with, in `dream_type` order. Interned once
    // here rather than at every call: a fold that asks the type of its
    // accumulator asks once per element.
    static const char* const kTypeNames[] = {
        "integer", "float",  "char",   "bool",   "unit",    "string",
        "atom",    "list",   "array",  "map",    "pure_fn", "impure_fn",
        "module",  "error",  "process", "unknown", "bigstr",  "tensor",
        "record",  "trunk",
    };
    static_assert(std::size(kTypeNames) == DREAM_TYPE_TRUNK + 1,
                  "every surface type needs a name");
    for (size_t i = 0; i < std::size(kTypeNames); ++i) {
        wk_->types[i] = intern_atom(kTypeNames[i]);
    }
    wk_->result_ok = intern_atom("std.prelude.Result.ok");
    wk_->result_error = intern_atom("std.prelude.Result.error");

    register_module(make_math_module());
    register_module(make_tensor_module());
    register_module(make_vm_module());
#ifndef DREAM_BROWSER
    register_module(make_ffi_module());
    register_module(make_io_module());
    register_module(make_net_module());
    register_module(make_tls_module());
    register_module(make_crypto_module());
    register_module(make_os_module());
    // The poller thread has to exist before any process can wait on a
    // descriptor, and it costs nothing when nothing does IO.
    io_init();
#endif
}

void Runtime::enable_profile(size_t top) {
    profiling_ = true;
    profile_top_ = top;
    if (const char* pid = std::getenv("DREAM_PROFILE_PID")) profile_pid_ = std::strtoull(pid, nullptr, 10);
    // Sized from the image, so a function index is a slot and no lookup is
    // needed on the hot path. An image loaded after this call is not profiled,
    // which cannot happen: the image is loaded before anything runs.
    std::vector<std::atomic<uint64_t>> fresh(image_ ? image_->func_count() : 0);
    profile_.swap(fresh);
    std::vector<std::atomic<uint64_t>> fresh_alloc(image_ ? image_->func_count() : 0);
    alloc_.swap(fresh_alloc);
}

/// A function as a profile names it. A lambda or a thunk has no name of its
/// own -- they are all `<lambda>` and `<thunk>` -- so it carries its index,
/// which `dreams --ir` maps back to where it was written. Without it the
/// largest row of a compile's profile was a sum of hundreds of unrelated
/// functions.
///
/// It is also named after the function it was written in. Lowering numbers a
/// function's lambdas, thunks and fused loops right after the function
/// itself, so the nearest named function before it is the one it came from
/// -- `<fused list>#2647 in map_modules` says where to look without `--ir`,
/// whose numbering a whole-program lowering need not share with a build in
/// parts.
static std::string profile_name(const Image& img, uint32_t fi) {
    auto name_of = [&img](uint32_t i) {
        StringRef name = img.str(img.func(i).name);
        return std::string(name.data, name.len);
    };
    std::string out = name_of(fi);
    if (out.empty() || out[0] != '<') return out;
    out += "#" + std::to_string(fi);
    for (uint32_t i = fi; i-- > 0;) {
        std::string outer = name_of(i);
        if (!outer.empty() && outer[0] != '<') return out + " in " + outer;
    }
    return out;
}

std::string Runtime::profile_function_name(uint32_t fi) const {
    return image_ ? dream::profile_name(*image_, fi) : std::string();
}

void Runtime::print_profile() const {
    auto profile_name = [this](uint32_t fi) { return dream::profile_name(*image_, fi); };
    const size_t top = profile_top_;
    if (!profiling_ || !image_) return;
    std::vector<std::pair<uint64_t, uint32_t>> rows;
    rows.reserve(profile_.size());
    uint64_t total = 0;
    for (uint32_t i = 0; i < profile_.size(); ++i) {
        uint64_t n = profile_[i].load(std::memory_order_relaxed);
        total += n;
        if (n) rows.push_back({n, i});
    }
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "; profile: %llu reductions in %zu functions\n",
                 static_cast<unsigned long long>(total), rows.size());
    for (size_t i = 0; i < rows.size() && i < top; ++i) {
        double pct = total ? 100.0 * double(rows[i].first) / double(total) : 0.0;
        std::fprintf(stderr, ";  %5.1f%%  %12llu  %s\n", pct,
                     static_cast<unsigned long long>(rows[i].first),
                     profile_name(rows[i].second).c_str());
    }

    // And the same by bytes. A function can be cheap in reductions and
    // expensive in garbage -- one allocation in a loop the machine walks in a
    // single step -- so the two orders are worth reading side by side.
    rows.clear();
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < alloc_.size(); ++i) {
        uint64_t n = alloc_[i].load(std::memory_order_relaxed);
        bytes += n;
        if (n) rows.push_back({n, i});
    }
    if (!bytes) return;
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::fprintf(stderr, "; allocated: %llu bytes in %zu functions\n",
                 static_cast<unsigned long long>(bytes), rows.size());
    for (size_t i = 0; i < rows.size() && i < top; ++i) {
        double pct = 100.0 * double(rows[i].first) / double(bytes);
        std::fprintf(stderr, ";  %5.1f%%  %12llu  %s\n", pct,
                     static_cast<unsigned long long>(rows[i].first),
                     profile_name(rows[i].second).c_str());
    }
}

void dbg_dump_thunks();

namespace {
int64_t wall_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
/// The process's CPU, every thread of it: `std::clock` is that on every
/// POSIX system, which is where this line is read.
int64_t cpu_now_ns() { return int64_t(double(std::clock()) * 1e9 / CLOCKS_PER_SEC); }
}  // namespace

void Runtime::mark_run_start() {
    run_wall_ns_ = wall_now_ns();
    run_cpu_ns_ = cpu_now_ns();
}

void Runtime::print_stats() const {
    dbg_dump_thunks();
    if (!stats_) return;
    print_trunk_stats();
    uint64_t reductions = scheduler_ ? scheduler_->total_reductions() : 0;
    uint64_t major = 0, minor = 0, promoted = 0, allocated = 0;
    uint64_t minor_ns = 0, major_ns = 0, rounds = 0;
    uint64_t concurrent = 0, concurrent_ns = 0;
    size_t peak = 0, held = 0, atpeak = 0;
    // Every process, not just the root: a program that does its work in
    // children -- which is what green processes are for -- would otherwise
    // report a heap that never did anything.
    //
    // A counter belongs to the worker running its process, and this may be
    // called from `os.exit!` while other workers are still running theirs, so
    // a number here can be one collection out of date. That is the same
    // bargain `std.vm.process_info!` makes for the same reason: a measurement
    // is worth having slightly stale, and worth nothing if taking it needs an
    // atomic on the allocation path.
    for (const auto& p : all_processes()) {
        const Heap& h = p->heap();
        major += h.major_collections();
        minor += h.minor_collections();
        promoted += h.bytes_promoted();
        minor_ns += h.nanos_minor();
        major_ns += h.nanos_major();
        rounds += h.parallel_rounds();
        concurrent += h.concurrent_marks();
        concurrent_ns += h.nanos_concurrent();
        allocated += h.bytes_total();
        peak += h.bytes_peak();
        held += h.bytes_peak_held();
        atpeak += h.bytes_alloc_at_peak();
    }
    std::fprintf(stderr,
                 "; %llu reductions, %llu major + %llu minor collections\n"
                 "; %llu bytes allocated, %llu promoted, %zu live at each heap's peak\n"\
                 "; %zu bytes held from the OS at each heap's peak (%zu of it allocated)\n"
                 "; %.0f ms stopped in collection (%.0f major, %.0f minor),"
                 " %llu rounds divided across threads\n"
                 "; %llu majors with their marking overlapped,"
                 " %.0f ms of it off the critical path\n",
                 static_cast<unsigned long long>(reductions),
                 static_cast<unsigned long long>(major),
                 static_cast<unsigned long long>(minor),
                 static_cast<unsigned long long>(allocated),
                 static_cast<unsigned long long>(promoted), peak, held, atpeak,
                 double(major_ns + minor_ns) / 1e6, double(major_ns) / 1e6,
                 double(minor_ns) / 1e6, static_cast<unsigned long long>(rounds),
                 static_cast<unsigned long long>(concurrent), double(concurrent_ns) / 1e6);
    // What the program cost, and nothing before it. The CPU is the whole
    // process's -- compile threads and parallel marking included -- because
    // that is what the machine paid.
    if (run_wall_ns_) {
        std::fprintf(stderr, "; run: %.0f ms wall, %.0f ms CPU\n",
                     double(wall_now_ns() - run_wall_ns_) / 1e6,
                     double(cpu_now_ns() - run_cpu_ns_) / 1e6);
    }

    // Where the bytes went, by object kind. A lazy language's garbage is mostly
    // thunks and frames, and the share of each is what says whether the next
    // thing worth writing is a strictness analysis or something else entirely.
    std::array<uint64_t, 32> kinds{};
    for (const auto& p : all_processes()) {
        const auto& t = p->heap().bytes_by_type();
        for (size_t i = 0; i < kinds.size(); ++i) kinds[i] += t[i];
    }
    std::vector<std::pair<uint64_t, size_t>> rows;
    uint64_t all = 0;
    for (size_t i = 0; i < kinds.size(); ++i) {
        all += kinds[i];
        if (kinds[i]) rows.push_back({kinds[i], i});
    }
    if (!all) return;
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::string line = "; allocated by kind:";
    for (size_t i = 0; i < rows.size() && i < 8; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof buf, " %s %.0f%%",
                      obj_type_name(ObjType(rows[i].second)),
                      100.0 * double(rows[i].first) / double(all));
        line += buf;
    }
    std::fprintf(stderr, "%s\n", line.c_str());

    // And of what *survived*, what is it? The line above is what the program
    // made; this is what it is still holding, which is the different question
    // and the one that decides whether a large program compiles at all. A
    // compile's peak is not garbage it has yet to collect -- `bytes_peak` and
    // the held figure above already say that -- so the only thing left to ask
    // is which objects they are.
    //
    // The moment is the largest *major* collection, which is the largest live
    // set anything here can vouch for: a minor never looks at old space, so
    // its survivors include everything old space happens to be carrying. It
    // is printed with its own total for exactly that reason -- it is usually
    // smaller than the peak on the line above, and reading one as the other
    // would overstate what is known.
    std::array<uint64_t, 64> live_kinds{};
    size_t live_peak = 0;
    for (const auto& p : all_processes()) {
        const auto& t = p->heap().live_by_type_at_peak();
        for (size_t i = 0; i < live_kinds.size(); ++i) live_kinds[i] += t[i];
        live_peak += p->heap().bytes_live_at_major_peak();
    }
    if (live_peak) {
        std::vector<std::pair<uint64_t, size_t>> live_rows;
        for (size_t i = 0; i < 32; ++i)
            if (live_kinds[i]) live_rows.push_back({live_kinds[i], i});
        std::sort(live_rows.begin(), live_rows.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });
        char head[96];
        std::snprintf(head, sizeof head, "; %zu live at the largest major, by kind:",
                      live_peak);
        std::string live_line = head;
        for (size_t i = 0; i < live_rows.size() && i < 8; ++i) {
            char buf[64];
            // The average size comes with the share, because a kind's bytes
            // and its object count answer different questions and the gap
            // between them is where a representation problem shows up.
            const uint64_t n = live_kinds[32 + live_rows[i].second];
            std::snprintf(buf, sizeof buf, " %s %.0f%% (%llu at %llu B)",
                          obj_type_name(ObjType(live_rows[i].second)),
                          100.0 * double(live_rows[i].first) / double(live_peak),
                          static_cast<unsigned long long>(n),
                          static_cast<unsigned long long>(n ? live_rows[i].first / n : 0));
            live_line += buf;
        }
        std::fprintf(stderr, "%s\n", live_line.c_str());
    }

    // How much of the run the JIT took over. It is printed here rather than
    // beside the "workers, jit on" line in the CLI for the reason the rest of
    // this function exists: every tool in this repository ends with `os.exit!`,
    // which never returns to `main`, so a number only `main` prints is never
    // printed for the runs worth measuring. Whether a change to the tier
    // *widened* it is exactly the question a self-compile is asked.
    if (jit_) {
        std::fprintf(stderr, "; jit policy %s\n", jit_policy_name(jit_->policy()));
        std::fprintf(stderr, "; %llu functions compiled\n",
                     static_cast<unsigned long long>(jit_->compiled_count()));
    }
}

ImageSession::ImageSession() = default;
/// Out of line because `runtime` is a `unique_ptr` to a type that is still
/// incomplete where the struct is declared -- `ImageSession` is named by
/// `Runtime` and names it back.
ImageSession::~ImageSession() = default;

Runtime::~Runtime() {
    // A session owns a whole runtime, so leaving one open leaks an image, a
    // heap and an atom table. Nothing else closes them: the compiler closes
    // the sessions it opened, but a compile that raises does not get that
    // far, and there is no path out of a process that does not come here.
    {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        sessions_.clear();
    }
    // Stop the poller before the scheduler it wakes into can go away, and
    // close whatever descriptors the program left open.
    // A `comp!` or macro VM must leave its caller's handles and jobs alive.
    if (owns_host_services_) {
        io_shutdown();
        os_shutdown();
    }
}

uint64_t Runtime::open_session(std::unique_ptr<ImageSession> session) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    uint64_t handle = next_session_++;
    sessions_.emplace(handle, std::move(session));
    return handle;
}

ImageSession* Runtime::session(uint64_t handle) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = sessions_.find(handle);
    return it == sessions_.end() ? nullptr : it->second.get();
}

bool Runtime::close_session(uint64_t handle) {
    std::unique_ptr<ImageSession> held;
    {
        std::lock_guard<std::mutex> lock(sessions_mutex_);
        auto it = sessions_.find(handle);
        if (it == sessions_.end()) return false;
        // Moved out and freed with the lock dropped: tearing down a runtime
        // stops a scheduler, and holding a mutex across that is how a
        // deadlock is written.
        held = std::move(it->second);
        sessions_.erase(it);
    }
    return true;
}

uint64_t Runtime::library_session(const std::string& key) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    auto it = library_sessions_.find(key);
    return it == library_sessions_.end() ? 0 : it->second;
}

uint64_t Runtime::note_library_session(const std::string& key, uint64_t handle) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    return library_sessions_.emplace(key, handle).first->second;
}

uint64_t Runtime::add_image_binding(ImageBinding binding) {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    std::string key = std::to_string(binding.session) + ":" + std::to_string(binding.func);
    auto it = image_binding_ids_.find(key);
    if (it != image_binding_ids_.end()) return it->second;
    image_bindings_.push_back(std::move(binding));
    uint64_t id = image_bindings_.size();
    image_binding_ids_.emplace(key, id);
    return id;
}

bool Runtime::image_binding(uint64_t id, ImageBinding* out) const {
    std::lock_guard<std::mutex> lock(sessions_mutex_);
    if (id == 0 || id > image_bindings_.size()) return false;
    *out = image_bindings_[id - 1];
    return true;
}

bool Runtime::load_image_file(const std::string& path, std::string& error) {
    auto img = std::make_unique<Image>();
    if (!img->load_file(path, error)) return false;
    image_ = std::move(img);
    size_caches();
    intern_image_atoms();

    // Best-effort: keep the source around so runtime errors can quote a line.
    std::ifstream in(image_->source_name().str());
    if (in) {
        std::ostringstream ss;
        ss << in.rdbuf();
        source_text_ = ss.str();
    }
    return true;
}

bool Runtime::load_image_bytes(const uint8_t* data, size_t size, std::string& error) {
    auto img = std::make_unique<Image>();
    if (!img->load_bytes(data, size, error)) return false;
    image_ = std::move(img);
    size_caches();
    intern_image_atoms();
    return true;
}

/// The lookup caches are sized from the image, so a cache slot is an index and
/// the fast path has nothing to search. Both are cleared by being rebuilt: an
/// image replaced at run time is not something the runtime supports, but a
/// second `load_image_*` on a fresh Runtime must not inherit a stale table.
void Runtime::size_caches() {
    std::vector<std::atomic<const ModuleDef*>> imports(image_ ? image_->import_count() : 0);
    import_defs_.swap(imports);
    std::vector<std::atomic<uint64_t>> fields(image_ ? image_->node_count() : 0);
    field_cache_.swap(fields);
}

/// Intern every atom the image names, and remember the answer per index.
///
/// The map is the point: after this, evaluating a `:name` constant is one
/// array read. Doing it the other way -- interning at each use -- meant a
/// `std::string`, a mutex and a hash on a path that runs once per reduction.
void Runtime::intern_image_atoms() {
    const uint32_t n = image_->atom_count();
    image_atom_ids_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        StringRef name = image_->atom_name(i);
        image_atom_ids_[i] = intern_atom(std::string_view(name.data, name.len));
    }
}

void Runtime::register_module(ModuleDef module) {
    // Number this module's members into the runtime-wide sequence, so a
    // process can cache the function value for each in a flat array.
    module.member_base = native_member_count_;
    native_member_count_ += uint32_t(module.members.size());
    modules_.push_back(std::move(module));
}

const ModuleDef* Runtime::module_for_import(uint32_t import_index) {
    if (import_index < import_defs_.size()) {
        const ModuleDef* hit = import_defs_[import_index].load(std::memory_order_relaxed);
        if (hit) return hit;
    }
    if (!image_ || import_index >= image_->import_count()) return nullptr;
    StringRef path = image_->str(image_->import(import_index).path);
    const ModuleDef* found = nullptr;
    for (const auto& m : modules_) {
        if (m.name.size() == path.len && std::memcmp(m.name.data(), path.data, path.len) == 0) {
            found = &m;
            break;
        }
    }
    // A confined runtime has no foreign libraries to offer (see `confine`):
    // the import resolves to nothing, and its first call says the module is
    // not provided, as for any module this runtime lacks.
    if (found && confined_ && found->name == "std.ffi") found = nullptr;
    if (found && import_index < import_defs_.size()) {
        import_defs_[import_index].store(found, std::memory_order_relaxed);
    }
    return found;
}

const ModuleDef* Runtime::find_module(const std::string& name) const {
    if (confined_ && name == "std.ffi") return nullptr;
    for (const auto& m : modules_) {
        if (m.name == name) return &m;
    }
    return nullptr;
}

uint32_t Runtime::intern_atom(std::string_view name) {
    std::lock_guard<std::mutex> g(atoms_mutex_);
    // Transparent lookup: no `std::string` is built unless the name is new.
    auto it = atom_ids_.find(name);
    if (it != atom_ids_.end()) return it->second;
    uint32_t id = uint32_t(atom_names_.size());
    atom_names_.emplace_back(name);
    atom_ids_.emplace(atom_names_.back(), id);
    return id;
}

const std::string& Runtime::atom_name(uint32_t index) const {
    static const std::string unknown = "<unknown-atom>";
    std::lock_guard<std::mutex> g(atoms_mutex_);
    if (index >= atom_names_.size()) return unknown;
    return atom_names_[index];
}

bool Runtime::find_atom(std::string_view name, uint32_t* out) const {
    std::lock_guard<std::mutex> g(atoms_mutex_);
    // Transparent lookup again: asking whether a name is an atom must not be
    // the one thing that builds a `std::string` out of it.
    auto it = atom_ids_.find(name);
    if (it == atom_ids_.end()) return false;
    *out = it->second;
    return true;
}

uint32_t Runtime::atom_count() const {
    std::lock_guard<std::mutex> g(atoms_mutex_);
    return uint32_t(atom_names_.size());
}

std::shared_ptr<Process> Runtime::spawn_process() {
    std::unique_lock<std::shared_mutex> g(processes_mutex_);
    uint64_t id = next_pid_++;
    auto p = std::make_shared<Process>(*this, id);
    processes_.emplace(id, p);
    return p;
}

std::shared_ptr<Process> Runtime::find_process(uint64_t id) const {
    std::shared_lock<std::shared_mutex> g(processes_mutex_);
    auto it = processes_.find(id);
    return it == processes_.end() ? nullptr : it->second;
}

void Runtime::retire_process(uint64_t id) {
    std::unique_lock<std::shared_mutex> g(processes_mutex_);
    processes_.erase(id);
}

size_t Runtime::live_process_count() const {
    std::shared_lock<std::shared_mutex> g(processes_mutex_);
    size_t n = 0;
    for (const auto& [id, p] : processes_) {
        if (!p->is_done()) ++n;
    }
    return n;
}

std::vector<std::shared_ptr<Process>> Runtime::all_processes() const {
    std::shared_lock<std::shared_mutex> g(processes_mutex_);
    std::vector<std::shared_ptr<Process>> out;
    out.reserve(processes_.size());
    for (auto& [id, p] : processes_) out.push_back(p);
    std::sort(out.begin(), out.end(),
              [](const std::shared_ptr<Process>& a, const std::shared_ptr<Process>& b) {
                  return a->id() < b->id();
              });
    return out;
}

const WellKnownAtoms& well_known(Runtime& rt) { return rt.well_known_atoms(); }

}  // namespace dream

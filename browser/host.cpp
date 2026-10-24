// The VM as a Web Worker runs it: one image, one scheduler pumped a slice at a
// time from the worker's event loop, and a bridge of two natives.
//
// The bridge is the whole of what the page gives a program, and it carries
// only text: `std.browser.send!` hands the host a JSON message and
// `next!` waits for one. What the messages mean -- render this tree, fetch
// that URL, call this registered JavaScript function -- is std.graphics's
// protocol and the worker's, written once on each side, so a new browser
// service is a case in two switch statements and never a native.
//
// Everything else an ordinary program reaches is a module here as on the
// command line: `std.crypto` itself, and `std.io`/`std.os` as services.cpp
// says. Nothing is refused when the image loads: a module a browser cannot
// have answers `:unsupported` (or "not provided by this runtime") when it is
// called, so importing a module that could open a file is not the same as
// opening one, and a frontend can share code with its backend.
#include <emscripten.h>
#include <deque>
#include <exception>
#include <memory>
#include <string>
#include "builtins.hpp"
#include "crypto.hpp"
#include "interp.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"
#include "services.hpp"

using namespace dream;
namespace {
constexpr size_t max_message = 1024 * 1024;
constexpr size_t max_pending = 4096;
struct Browser {
    std::vector<uint8_t> image;
    Runtime rt;
    std::unique_ptr<Scheduler> scheduler;
    std::shared_ptr<Process> root;
    std::deque<std::string> events;
    uint64_t reader = 0;
};
std::unique_ptr<Browser> browser;

EM_JS(void, emit, (const char* bytes, size_t size), {
    Module.onMessage?.(UTF8ToString(bytes, size));
});
EM_JS(void, report_error, (const char* bytes), {
    Module.onError?.(UTF8ToString(bytes));
});

NativeResult failure(Process& p, const char* message) {
    return NativeResult::raise(raise_error(p, p.runtime().intern_atom("graphics_error"), message));
}
NativeResult send_message(Process& p, Value, Value* args, uint32_t) {
    Bytes bytes;
    if (!string_bytes(args[0], &bytes) || bytes.len > max_message)
        return failure(p, "graphics message exceeds limit or is not a string");
    emit(bytes.data, bytes.len);
    return NativeResult::ok(UNIT);
}
/// One reader, so the order the page sent events in is the order a program
/// sees them. std.graphics gives it a process of its own, which forwards each
/// event to whoever asked; a second reader is a mistake worth stopping on.
NativeResult next_event(Process& p, Value, Value*, uint32_t) {
    if (browser->reader && browser->reader != p.id())
        return failure(p, "only one process may consume graphics events");
    browser->reader = p.id();
    if (!browser->events.empty()) {
        auto text = std::move(browser->events.front());
        browser->events.pop_front();
        return NativeResult::ok(p.heap().make_string(text.data(), uint32_t(text.size())));
    }
    p.park_requested = true;
    p.wait_reason.store(WaitReason::Message);
    return NativeResult::block();
}
}

extern "C" {
int dream_browser_load(const uint8_t* bytes, size_t size) {
    if (browser || size == 0 || size > 64 * 1024 * 1024) return 0;
    auto next = std::make_unique<Browser>();
    next->rt.register_module(make_crypto_module());
    next->rt.register_module(make_browser_io_module());
    next->rt.register_module(make_browser_os_module());
    next->rt.register_module(ModuleDef{"std.browser", {
        {"send!", 1, 1, send_message}, {"next!", 1, 1, next_event}}});
    std::string error;
    next->image.assign(bytes, bytes + size);
    if (!next->rt.load_image_bytes(next->image.data(), next->image.size(), error)) {
        report_error(error.c_str());
        return 0;
    }
    const auto& img = next->rt.image();
    if (img.entry() == NO_NODE) {
        report_error("the image has no entry point");
        return 0;
    }
    browser_reset_exit();
    next->scheduler = std::make_unique<Scheduler>(next->rt, 1);
    next->root = next->scheduler->create_process();
    prime_entry(*next->root, img, img.entry());
    next->scheduler->enqueue(next->root);
    browser = std::move(next);
    return 1;
}
// 0: finished, 1: runnable, 2: waiting for the host or a timer, -1: failed.
//
// The program is finished when `main!` is, as on the command line once the
// root has an answer: a process still parked on the bridge, or a ticker the
// program started, does not keep a page's program alive after it returned.
//
// Failures are the command line's too. The root's ends the run at once; any
// other process's is only *uncaught* if nothing joins it before the end -- a
// `proc.try_map!` joins its failures after they happen -- so those are asked
// for once, when the run is over, and make it a failure then.
int dream_browser_tick() {
    if (!browser) return -1;
    bool live;
    // Nothing C++ throws may cross into the page's JavaScript, where it is an
    // opaque pointer; it is a failure of the program, said in words.
    try { live = browser->scheduler->pump(1); }
    catch (const std::exception& e) {
        report_error((std::string("the VM failed: ") + e.what()).c_str());
        return -1;
    }
    const auto& root = *browser->root;
    if (root.is_done() && root.failed) {
        report_error(("uncaught error: " + describe_failure(*browser->root, root.exit_value)).c_str());
        return -1;
    }
    const bool over = browser_exit_requested(nullptr) || root.is_done() || !live;
    if (!over) return browser->scheduler->queued() ? 1 : 2;
    std::string uncaught;
    for (const auto& f : browser->scheduler->take_failures())
        uncaught += (uncaught.empty() ? "" : "\n") + ("uncaught error in " + f);
    if (!uncaught.empty()) {
        report_error(uncaught.c_str());
        return -1;
    }
    return 0;
}
int dream_browser_delay() {
    if (!browser) return -1;
    if (browser->scheduler->queued()) return 0;
    auto delay = browser->scheduler->next_timer_delay_ms();
    return int(std::min<int64_t>(delay, 2147483647));
}
/// The code `os.exit!` was given, or 0 for a program that returned.
int dream_browser_exit_code() {
    int code = 0;
    return browser_exit_requested(&code) ? code : 0;
}
int dream_browser_event(const char* bytes, size_t size) {
    if (!browser || size > max_message || browser->events.size() >= max_pending) return 0;
    browser->events.emplace_back(bytes, size);
    if (browser->reader) browser->scheduler->wake(browser->reader);
    return 1;
}
void dream_browser_dispose() { browser.reset(); }
}

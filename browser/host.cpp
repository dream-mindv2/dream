#include <emscripten.h>
#include <deque>
#include <memory>
#include <string>
#include "builtins.hpp"
#include "interp.hpp"
#include "runtime.hpp"
#include "scheduler.hpp"

using namespace dream;
namespace {
constexpr size_t max_message = 1024 * 1024;
constexpr size_t max_pending = 256;
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
    next->rt.register_module(ModuleDef{"dream.graphics.bridge", {
        {"send!", 1, 1, send_message}, {"next!", 1, 1, next_event}}});
    std::string error;
    next->image.assign(bytes, bytes + size);
    if (!next->rt.load_image_bytes(next->image.data(), next->image.size(), error)) {
        report_error(error.c_str());
        return 0;
    }
    // Reject ambient native services at load time, including unused imports.
    const auto& img = next->rt.image();
    for (uint32_t i = 0; i < img.import_count(); ++i) {
        auto path = img.str(img.import(i).path).str();
        if (path != "std.math" && path != "std.vm" && path != "std.tensor" &&
            path != "dream.graphics.bridge") {
            error = "native module unavailable in browser: " + path;
            report_error(error.c_str());
            return 0;
        }
    }
    if (img.entry() == NO_NODE) return 0;
    next->scheduler = std::make_unique<Scheduler>(next->rt, 1);
    next->root = next->scheduler->create_process();
    prime_entry(*next->root, img, img.entry());
    next->scheduler->enqueue(next->root);
    browser = std::move(next);
    return 1;
}
// 0: finished, 1: runnable, 2: waiting for the host or a timer, -1: failed.
int dream_browser_tick() {
    if (!browser) return -1;
    bool live = browser->scheduler->pump(1);
    auto errors = browser->scheduler->take_failures();
    if (!errors.empty()) {
        report_error(errors.front().c_str());
        return -1;
    }
    return live ? (browser->scheduler->queued() ? 1 : 2) : 0;
}
int dream_browser_delay() {
    if (!browser) return -1;
    if (browser->scheduler->queued()) return 0;
    auto delay = browser->scheduler->next_timer_delay_ms();
    return int(std::min<int64_t>(delay, 2147483647));
}
int dream_browser_event(const char* bytes, size_t size) {
    if (!browser || size > max_message || browser->events.size() >= max_pending) return 0;
    browser->events.emplace_back(bytes, size);
    if (browser->reader) browser->scheduler->wake(browser->reader);
    return 1;
}
void dream_browser_dispose() { browser.reset(); }
}

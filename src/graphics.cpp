// Optional Linux desktop services for std.graphics. Application state, views,
// updates, command permissions and schemas remain in Dream.
//
// This process never runs JavaScript of its own. What the page sends arrives
// as the JSCValue WebKit hands a script message handler, and what goes back is
// a JSON string quoted here (`quote`) and parsed by the page. It used to keep a
// JSCContext for that, and on WebKitGTK 2.54 a context made in the UI process
// before the view stops the view's first load from ever starting.
#include "graphics.hpp"
#include "image.hpp"
#include <webkit2/webkit2.h>
#include <jsc/jsc.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace dream {
namespace {
constexpr size_t max_message = 1024 * 1024;
constexpr size_t max_asset = 64 * 1024 * 1024;
constexpr size_t max_assets = 4096;
constexpr size_t max_total = 512 * 1024 * 1024;
constexpr size_t max_queue_bytes = 8 * 1024 * 1024;
constexpr const char* origin = "dream://localhost/";
constexpr const char* page_uri = "dream://localhost/index.html";
/// What an application without a `csp.txt` gets, and the floor build.py's own
/// policy starts from.
constexpr const char* default_csp = "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; "
    "worker-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self' data:; media-src 'self' blob:; base-uri 'none'; form-action 'none'";
bool ours(const char* uri) { return uri && std::strncmp(uri, origin, std::strlen(origin)) == 0; }

template<class T> using Object = std::unique_ptr<T, decltype(&g_object_unref)>;
template<class T> Object<T> hold(T* value) { return Object<T>(value, g_object_unref); }
using Value = Object<JSCValue>;
Value property(JSCValue* value, const char* key) {
    return hold(value && jsc_value_is_object(value) ? jsc_value_object_get_property(value, key) : nullptr);
}
bool string_value(JSCValue* value, std::string& out) {
    if (!value || !jsc_value_is_string(value)) return false;
    GBytes* bytes = jsc_value_to_string_as_bytes(value);
    gsize size = 0;
    const auto* data = static_cast<const char*>(g_bytes_get_data(bytes, &size));
    out.assign(data ? data : "", size);
    g_bytes_unref(bytes);
    return true;
}
bool string_property(JSCValue* value, const char* key, std::string& out) {
    auto field = property(value, key);
    return string_value(field.get(), out);
}
/// `text` as a JSON string, which is a JavaScript string literal too.
std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 0x20) { char esc[8]; std::snprintf(esc, sizeof esc, "\\u%04x", c); out += esc; }
        else out += char(c);
    }
    return out + "\"";
}

const char* media_type(const std::string& name) {
    static const std::pair<const char*, const char*> types[] = {
        // No `charset` on a script or a stylesheet: both are UTF-8 unless they
        // say otherwise, and WebKitGTK never finishes loading a module whose
        // custom-scheme response names one.
        {".html", "text/html; charset=utf-8"}, {".mjs", "text/javascript"},
        {".js", "text/javascript"}, {".css", "text/css"},
        {".json", "application/json"}, {".wasm", "application/wasm"}, {".svg", "image/svg+xml"},
        {".png", "image/png"}, {".jpg", "image/jpeg"}, {".jpeg", "image/jpeg"}, {".gif", "image/gif"},
        {".webp", "image/webp"}, {".avif", "image/avif"}, {".ico", "image/x-icon"},
        {".woff2", "font/woff2"}, {".woff", "font/woff"}, {".ttf", "font/ttf"}, {".otf", "font/otf"},
        {".mp3", "audio/mpeg"}, {".ogg", "audio/ogg"}, {".wav", "audio/wav"}, {".mp4", "video/mp4"},
        {".webm", "video/webm"}, {".txt", "text/plain; charset=utf-8"},
    };
    for (auto [ext, type] : types) if (name.ends_with(ext)) return type;
    return "application/octet-stream";
}

void trace_load(WebKitWebView* view, WebKitLoadEvent event, gpointer) {
    static const char* const names[] = {"started", "redirected", "committed", "finished"};
    std::fprintf(stderr, "graphics: load %s %s\n", names[event], webkit_web_view_get_uri(view));
}

struct Host {
    std::map<std::string, std::string> assets;
    std::string csp = default_csp;
    Object<WebKitWebContext> context = hold(webkit_web_context_new_ephemeral());
    Object<WebKitUserContentManager> manager = hold(webkit_user_content_manager_new());
    Object<GCancellable> cancel = hold(g_cancellable_new());
    Object<GSubprocess> backend = hold<GSubprocess>(nullptr);
    GtkWidget* window = nullptr;
    WebKitWebView* view = nullptr;
    std::deque<std::string> outgoing;
    std::string incoming;
    size_t written = 0, queued_bytes = 0;
    bool done = false, script_pending = false;
    int status = 0;
    guint pump_source = 0, close_source = 0;

    ~Host() {
        if (pump_source) g_source_remove(pump_source);
        if (close_source) g_source_remove(close_source);
        g_cancellable_cancel(cancel.get());
        // An evaluation callback owns no state beyond this host. Drain its
        // cancellation before destroying the window or returning to the caller.
        while (script_pending) g_main_context_iteration(nullptr, TRUE);
        if (backend) {
            finish_backend();
            g_subprocess_force_exit(backend.get());
            g_subprocess_wait(backend.get(), nullptr, nullptr);
        }
        if (window) gtk_widget_destroy(window);
    }
    /// What the frontend asked last -- a `closing!` saving its work -- still
    /// reaches the backend: the queue is written out, the pipe is closed so
    /// `serve!` sees its end and returns, and the backend has two seconds to
    /// do so before it is stopped.
    void finish_backend() {
        if (status != 0) return;  // a failed application is not owed a goodbye
        auto* input = g_subprocess_get_stdin_pipe(backend.get());
        for (bool first = true; !outgoing.empty(); first = false) {
            const auto& m = outgoing.front();
            const size_t from = first ? written : 0;
            if (!g_output_stream_write_all(input, m.data() + from, m.size() - from, nullptr, nullptr, nullptr)) break;
            outgoing.pop_front();
        }
        g_output_stream_close(input, nullptr, nullptr);
        // Heap-held: if the backend outlives the wait, the callback comes after
        // this frame has gone, and frees it then.
        auto* exited = new bool(false);
        g_subprocess_wait_async(backend.get(), nullptr, [](GObject*, GAsyncResult*, gpointer data) {
            auto* flag = static_cast<bool*>(data);
            if (*flag) delete flag; else *flag = true;
        }, exited);
        const gint64 deadline = g_get_monotonic_time() + 2 * G_USEC_PER_SEC;
        while (!*exited && g_get_monotonic_time() < deadline) {
            if (!g_main_context_iteration(nullptr, FALSE)) g_usleep(5000);
        }
        if (*exited) delete exited; else *exited = true;
    }
    void stop(int code, const std::string& reason = {}) {
        if (done) return;
        done = true;
        status = code;
        if (!reason.empty()) std::fprintf(stderr, "dream graphics: %s\n", reason.c_str());
        if (gtk_main_level()) gtk_main_quit();
    }
    void deliver(const std::string& text) { deliver(std::vector<std::string>{text}); }
    void deliver(const std::vector<std::string>& frames) {
        // Quote each entire JSON frame as one string argument. Never
        // interpolate application text as executable JavaScript, even after
        // validation. A frame that is not JSON fails in `JSON.parse`, which
        // stops the application below, as an invalid frame should.
        //
        // Every frame that has arrived goes in one evaluation: a backend
        // streaming events -- a terminal's output -- sends many a tick, and
        // one round trip into the page each held them to a hundred a second.
        std::string script;
        for (const auto& text : frames) script += "window.__dreamReceive(JSON.parse(" + quote(text) + "));";
        script_pending = true;
        webkit_web_view_evaluate_javascript(view, script.c_str(), script.size(), nullptr, nullptr,
            cancel.get(), [](GObject* object, GAsyncResult* result, gpointer data) {
                auto& host = *static_cast<Host*>(data);
                GError* error = nullptr;
                auto answer = hold(webkit_web_view_evaluate_javascript_finish(WEBKIT_WEB_VIEW(object), result, &error));
                host.script_pending = false;
                if (error) {
                    if (!host.done) host.stop(1, error->message);
                    g_error_free(error);
                }
            }, this);
    }
    /// The backend's answer, made here: there is no backend to ask.
    void unavailable(const std::string& reply) {
        deliver("{\"name\":" + quote(reply) + ",\"key\":\"\",\"payload\":" +
                quote("{\"ok\":false,\"error\":\"backend_unavailable\"}") + "}");
    }
    void pump() {
        if (!backend) return;
        auto* output = G_POLLABLE_OUTPUT_STREAM(g_subprocess_get_stdin_pipe(backend.get()));
        if (!outgoing.empty()) {
            const auto& message = outgoing.front();
            GError* error = nullptr;
            const gssize n = g_pollable_output_stream_write_nonblocking(output,
                message.data() + written, message.size() - written, nullptr, &error);
            if (n > 0) {
                written += size_t(n);
                if (written == message.size()) {
                    queued_bytes -= message.size();
                    outgoing.pop_front();
                    written = 0;
                }
            } else if (!error || !g_error_matches(error, G_IO_ERROR, G_IO_ERROR_WOULD_BLOCK)) {
                stop(1, "backend pipe closed");
            }
            g_clear_error(&error);
        }
        // Apply backpressure while a reply is entering the frontend. No worker
        // thread or unbounded main-loop queue is needed for the child pipes.
        if (done || script_pending) return;
        auto* input = G_POLLABLE_INPUT_STREAM(g_subprocess_get_stdout_pipe(backend.get()));
        // Up to 4 MiB of frames a tick, delivered together.
        std::vector<std::string> frames;
        size_t taken = 0;
        auto send = [&] { if (!frames.empty() && !done) deliver(frames); };
        for (size_t budget = 0; budget < 4 * max_message; budget += 65536) {
            for (auto newline = incoming.find('\n'); newline != std::string::npos; newline = incoming.find('\n')) {
                std::string frame = incoming.substr(0, newline);
                incoming.erase(0, newline + 1);
                if (frame.size() > max_message || frame.find('\0') != std::string::npos) {
                    stop(1, "invalid backend response");
                    return;
                }
                taken += frame.size();
                frames.push_back(std::move(frame));
            }
            if (incoming.size() > max_message) { stop(1, "backend frame exceeds limit"); return; }
            if (taken >= 4 * max_message) break;
            char bytes[65536];
            GError* error = nullptr;
            const gssize n = g_pollable_input_stream_read_nonblocking(input, bytes, sizeof bytes, nullptr, &error);
            if (n > 0) incoming.append(bytes, size_t(n));
            else {
                const bool would_block = error && g_error_matches(error, G_IO_ERROR, G_IO_ERROR_WOULD_BLOCK);
                g_clear_error(&error);
                send();
                if (!would_block) stop(1, "backend stopped or sent an incomplete frame");
                return;
            }
        }
        send();
    }
    void message(JSCValue* value) {
        // Any page of the application's own -- a route it pushed is still it.
        if (done || !ours(webkit_web_view_get_uri(view))) return;
        std::string type;
        if (!string_property(value, "type", type)) { stop(1, "invalid graphics message"); return; }
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fprintf(stderr, "graphics IPC: %s\n", type.c_str());
        if (type == "close") {
            // The program finished, with the status `os.exit!` gave it or 0.
            auto code = property(value, "code");
            const double n = code && jsc_value_is_number(code.get()) ? jsc_value_to_double(code.get()) : 0;
            stop(n >= 0 && n <= 255 ? int(n) : 1);
            return;
        }
        if (type == "log") {
            // The frontend's standard output and error, line by line.
            std::string stream, line;
            if (string_property(value, "stream", stream) && string_property(value, "message", line) &&
                line.size() <= max_message) {
                std::fprintf(stream == "stderr" ? stderr : stdout, "%s\n", line.c_str());
                std::fflush(stream == "stderr" ? stderr : stdout);
            }
            return;
        }
        if (type == "failure") {
            std::string reason;
            string_property(value, "message", reason);
            stop(1, reason.empty() ? "frontend failed" : reason);
            return;
        }
        if (type == "open") {
            auto options = property(value, "options");
            std::string title;
            auto width = property(options.get(), "width"), height = property(options.get(), "height");
            auto dimension = [](JSCValue* v) {
                if (!v || !jsc_value_is_number(v)) return 0;
                const double n = jsc_value_to_double(v);
                return n >= 1 && n <= 16384 && std::floor(n) == n ? int(n) : 0;
            };
            const int w = dimension(width.get()), h = dimension(height.get());
            if (!string_property(options.get(), "title", title) || title.size() > 4096 ||
                title.find('\0') != std::string::npos || !w || !h) { stop(1, "invalid window options"); return; }
            gtk_window_set_title(GTK_WINDOW(window), title.c_str());
            gtk_window_resize(GTK_WINDOW(window), w, h);
            return;
        }
        std::string reply, command, payload;
        if (type != "invoke" || !string_property(value, "reply", reply) || reply.empty() || reply.size() > 256 ||
            !string_property(value, "command", command) || command.empty() || command.size() > 256 ||
            !string_property(value, "payload", payload) || payload.size() > max_message) {
            stop(1, "invalid command request");
            return;
        }
        if (!backend) { unavailable(reply); return; }
        // The backend is sent exactly the envelope it validates, rewritten
        // here rather than passed on, so nothing else the page put in it does.
        const std::string frame = "{\"type\":\"invoke\",\"reply\":" + quote(reply) + ",\"command\":" + quote(command) +
            ",\"payload\":" + quote(payload) + "}\n";
        if (outgoing.size() >= 64 || queued_bytes + frame.size() > max_queue_bytes) {
            stop(1, "backend request queue is full");
            return;
        }
        outgoing.push_back(frame);
        queued_bytes += frame.size();
    }
    void serve(WebKitURISchemeRequest* request) {
        const char* uri = webkit_uri_scheme_request_get_uri(request);
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fprintf(stderr, "graphics asset: %s\n", uri);
        const char* method = webkit_uri_scheme_request_get_http_method(request);
        // A path with no extension is a route of the application's, which a
        // reload asks for by name: the page is the answer, as a static host
        // serving a single-page application would give.
        std::string wanted = uri ? uri : "";
        if (auto q = wanted.find_first_of("?#"); q != std::string::npos) wanted.erase(q);
        const auto slash = wanted.rfind('/');
        if (ours(wanted.c_str()) && wanted.find('.', slash == std::string::npos ? 0 : slash) == std::string::npos)
            wanted = page_uri;
        const auto found = assets.find(wanted);
        const bool ok = found != assets.end() && method && !std::strcmp(method, "GET");
        const std::string bytes = ok ? found->second : "";
        GBytes* data = g_bytes_new(bytes.data(), bytes.size());
        auto stream = hold(g_memory_input_stream_new_from_bytes(data));
        g_bytes_unref(data);
        auto response = hold(webkit_uri_scheme_response_new(stream.get(), bytes.size()));
        webkit_uri_scheme_response_set_status(response.get(), ok ? 200 : 404, nullptr);
        const char* mime = media_type(wanted);
        webkit_uri_scheme_response_set_content_type(response.get(), mime);
        SoupMessageHeaders* headers = soup_message_headers_new(SOUP_MESSAGE_HEADERS_RESPONSE);
        soup_message_headers_append(headers, "Content-Security-Policy", csp.c_str());
        // The type again, as a header: these headers replace the response's,
        // and under `nosniff` a script that has lost its type is refused.
        soup_message_headers_append(headers, "Content-Type", mime);
        soup_message_headers_append(headers, "X-Content-Type-Options", "nosniff");
        soup_message_headers_append(headers, "Cache-Control", "no-store");
        // The response takes the headers (transfer full): unreferencing them
        // here as well freed them under it.
        webkit_uri_scheme_response_set_http_headers(response.get(), headers);
        webkit_uri_scheme_request_finish_with_response(request, response.get());
    }
    /// One file of the page, served as `name`.
    void add_asset(const std::string& name, std::string bytes, size_t& total) {
        if (assets.size() >= max_assets) throw std::runtime_error("too many assets");
        if (bytes.size() > max_asset || total + bytes.size() > max_total)
            throw std::runtime_error("asset exceeds size limit: " + name);
        total += bytes.size();
        assets.emplace(origin + name, std::move(bytes));
    }
    /// What every page needs, and the policy, once the assets are in.
    void finish_assets() {
        for (const char* name : {"index.html", "main.mjs", "dom.mjs", "worker.mjs",
                                "dream-browser.mjs", "dream-browser.wasm", "frontend.dream"})
            if (!assets.count(std::string(origin) + name)) throw std::runtime_error(std::string("missing asset: ") + name);
        // The policy build.py wrote into the page, sent as a header too, so the
        // two never disagree about what the page may reach.
        if (auto found = assets.find(std::string(origin) + "csp.txt"); found != assets.end()) {
            std::string policy = found->second;
            while (!policy.empty() && (policy.back() == '\n' || policy.back() == '\r')) policy.pop_back();
            if (!policy.empty() && policy.find_first_of("\r\n") == std::string::npos) csp = policy;
        }
    }
    /// Every regular file under `assets/`, read now: the application's own
    /// directory is what it ships, and nothing outside it -- a symbolic link
    /// included -- is served. Read once, so a rebuild while the window is open
    /// changes nothing under it.
    void load_assets(const std::filesystem::path& dir) {
        size_t total = 0;
        for (auto it = std::filesystem::recursive_directory_iterator(dir); it != std::filesystem::end(it); ++it) {
            std::error_code ec;
            if (!std::filesystem::is_regular_file(it->symlink_status(ec)) || ec) continue;
            std::ifstream file(it->path(), std::ios::binary | std::ios::ate);
            const auto size = file.tellg();
            if (!file || size < 0 || uint64_t(size) > max_asset)
                throw std::runtime_error("asset exceeds size limit: " + it->path().string());
            std::string bytes(size_t(size), '\0');
            file.seekg(0);
            if (!file.read(bytes.data(), size)) throw std::runtime_error("cannot read asset: " + it->path().string());
            add_asset(std::filesystem::relative(it->path(), dir).generic_string(), std::move(bytes), total);
        }
        finish_assets();
    }
    /// The page an image carries: each payload named `graphics/PATH` is the
    /// file at PATH. Copied out, as a directory's are read, so the map holds
    /// what it serves whatever happens to the file the image was mapped from.
    void load_assets(const Image& image) {
        size_t total = 0;
        constexpr std::string_view prefix = "graphics/";
        for (uint32_t i = 0; i < image.data_count(); ++i) {
            const std::string_view name = image.data_name(i);
            if (!name.starts_with(prefix) || name.size() == prefix.size()) continue;
            if (image.data_length(i) > max_asset) throw std::runtime_error("asset exceeds size limit: " + std::string(name));
            add_asset(std::string(name.substr(prefix.size())),
                      std::string(image.data_bytes(i), size_t(image.data_length(i))), total);
        }
        finish_assets();
    }
    /// The backend: this VM again, running `argv` with its standard input and
    /// output as the pipes the page's commands go over.
    void start_backend(const std::vector<std::string>& argv) {
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: starting backend\n", stderr);
        std::vector<const char*> args;
        for (const auto& a : argv) args.push_back(a.c_str());
        args.push_back(nullptr);
        GError* error = nullptr;
        backend = hold(g_subprocess_newv(args.data(),
            GSubprocessFlags(G_SUBPROCESS_FLAGS_STDIN_PIPE | G_SUBPROCESS_FLAGS_STDOUT_PIPE), &error));
        if (!backend) {
            const std::string reason = error->message;
            g_error_free(error);
            throw std::runtime_error(reason);
        }
    }
    void start(const std::filesystem::path& application) {
        load_assets(application / "assets");
        // A backend is optional: without one, a command is answered
        // `backend_unavailable`, as it is in a browser tab.
        if (std::filesystem::exists(application / "backend.dream"))
            start_backend({std::filesystem::canonical("/proc/self/exe").string(),
                           std::filesystem::canonical(application / "backend.dream").string()});
        open_view();
    }
    /// An application image: its page from its payloads, and its program as
    /// the backend when `app.json` says it is one rather than the frontend.
    /// `--headless` is what stops the backend opening a window of its own.
    void start(const Image& image, const std::string& path, const std::vector<std::string>& program_args) {
        load_assets(image);
        const auto manifest = assets.find(std::string(origin) + "app.json");
        if (manifest != assets.end() && manifest->second.find("\"backend\":true") != std::string::npos) {
            std::vector<std::string> argv = {std::filesystem::canonical("/proc/self/exe").string(), "--headless",
                                             std::filesystem::canonical(path).string()};
            argv.insert(argv.end(), program_args.begin(), program_args.end());
            start_backend(argv);
        }
        open_view();
    }
    void open_view() {
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: creating webview\n", stderr);
        auto* security = webkit_web_context_get_security_manager(context.get());
        webkit_security_manager_register_uri_scheme_as_secure(security, "dream");
        // Secure and CORS-enabled, but not *local*: a local scheme is treated
        // like file://, which the application needs no part of, and under it
        // WebKitGTK refuses a module's imports without an
        // Access-Control-Allow-Origin header -- main.mjs loaded and dom.mjs
        // was never asked for.
        webkit_security_manager_register_uri_scheme_as_cors_enabled(security, "dream");
        webkit_web_context_register_uri_scheme(context.get(), "dream", [](WebKitURISchemeRequest* request, gpointer data) {
            static_cast<Host*>(data)->serve(request);
        }, this, nullptr);
        g_signal_connect(manager.get(), "script-message-received::dream", G_CALLBACK(+[](WebKitUserContentManager*, WebKitJavascriptResult* result, gpointer data) {
            static_cast<Host*>(data)->message(webkit_javascript_result_get_js_value(result));
        }), this);
        webkit_user_content_manager_register_script_message_handler(manager.get(), "dream");
        const char* allow[] = { "dream://localhost/*", nullptr };
        auto* bridge = webkit_user_script_new(
            "Object.defineProperty(globalThis,'ipc',{value:Object.freeze({postMessage:message=>window.webkit.messageHandlers.dream.postMessage(message)})});",
            WEBKIT_USER_CONTENT_INJECT_TOP_FRAME, WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START, allow, nullptr);
        webkit_user_content_manager_add_script(manager.get(), bridge);
        webkit_user_script_unref(bridge);
        window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_title(GTK_WINDOW(window), "Dream");
        gtk_window_set_default_size(GTK_WINDOW(window), 960, 640);
        view = WEBKIT_WEB_VIEW(g_object_new(WEBKIT_TYPE_WEB_VIEW, "web-context", context.get(),
            "user-content-manager", manager.get(), nullptr));
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: webview created\n", stderr);
        if (std::getenv("DREAM_GRAPHICS_TRACE"))
            g_object_set(webkit_web_view_get_settings(view), "enable-write-console-messages-to-stdout", TRUE, nullptr);
        gtk_container_add(GTK_CONTAINER(window), GTK_WIDGET(view));
        // The webview shows the application and nothing else. A link to the
        // web that the person clicked opens in their browser, as it would from
        // any desktop program; every other navigation away is refused.
        g_signal_connect(view, "decide-policy", G_CALLBACK(+[](WebKitWebView*, WebKitPolicyDecision* decision, WebKitPolicyDecisionType type, gpointer data) -> gboolean {
            if (type != WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION && type != WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION) return FALSE;
            auto* action = webkit_navigation_policy_decision_get_navigation_action(WEBKIT_NAVIGATION_POLICY_DECISION(decision));
            const char* uri = webkit_uri_request_get_uri(webkit_navigation_action_get_request(action));
            if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fprintf(stderr, "graphics: navigation %d %s\n", int(type), uri ? uri : "");
            if (type == WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION && ours(uri)) return FALSE;
            if (uri && webkit_navigation_action_get_navigation_type(action) == WEBKIT_NAVIGATION_TYPE_LINK_CLICKED &&
                webkit_navigation_action_is_user_gesture(action) &&
                (!std::strncmp(uri, "https://", 8) || !std::strncmp(uri, "http://", 7)))
                gtk_show_uri_on_window(GTK_WINDOW(static_cast<Host*>(data)->window), uri, GDK_CURRENT_TIME, nullptr);
            webkit_policy_decision_ignore(decision);
            return TRUE;
        }), this);
        if (std::getenv("DREAM_GRAPHICS_TRACE"))
            g_signal_connect(view, "load-changed", G_CALLBACK(trace_load), nullptr);
        g_signal_connect(view, "load-failed", G_CALLBACK(+[](WebKitWebView*, WebKitLoadEvent, const char*, GError* error, gpointer data) -> gboolean {
            static_cast<Host*>(data)->stop(1, error->message);
            return TRUE;
        }), this);
        g_signal_connect(view, "web-process-terminated", G_CALLBACK(+[](WebKitWebView*, WebKitWebProcessTerminationReason, gpointer data) {
            static_cast<Host*>(data)->stop(1, "webview process terminated");
        }), this);
        g_signal_connect(window, "delete-event", G_CALLBACK(+[](GtkWidget*, GdkEvent*, gpointer data) -> gboolean {
            auto& host = *static_cast<Host*>(data);
            if (!host.close_source) {
                webkit_web_view_evaluate_javascript(host.view, "window.__dreamReceive?.({name:'close',key:'',value:null});", -1,
                    nullptr, nullptr, nullptr, nullptr, nullptr);
                host.close_source = g_timeout_add(2000, [](gpointer data) -> gboolean {
                    auto& host = *static_cast<Host*>(data);
                    host.close_source = 0;
                    host.stop(0);
                    return G_SOURCE_REMOVE;
                }, data);
            }
            return TRUE;
        }), this);
        pump_source = g_timeout_add(10, [](gpointer data) -> gboolean {
            auto& host = *static_cast<Host*>(data);
            host.pump();
            return G_SOURCE_CONTINUE;
        }, this);
        webkit_web_view_load_uri(view, page_uri);
        gtk_widget_show_all(window);
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: window shown\n", stderr);
    }
};
}

namespace {
/// WebKitGTK hands the compositor its frames as DMA-BUFs, and on NVIDIA's
/// driver under Wayland the compositor refuses them: GDK reports "Error 71
/// (Protocol error) dispatching to Wayland display" and the window is gone
/// before it is drawn. Shared memory is what WebKit falls back to when told,
/// and it is told only there, and only when the person has not said
/// otherwise. Before GTK starts, so the web process WebKit spawns inherits it.
void prepare_display() {
    if (std::getenv("WAYLAND_DISPLAY") && !std::getenv("WEBKIT_DISABLE_DMABUF_RENDERER") &&
        std::filesystem::exists("/proc/driver/nvidia/version"))
        setenv("WEBKIT_DISABLE_DMABUF_RENDERER", "1", 0);
}
}

int run_graphics(const std::string& application) {
    if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: initializing display\n", stderr);
    prepare_display();
    if ((!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) || !gtk_init_check(nullptr, nullptr)) {
        std::fputs("dream: graphics_unavailable: no display\n", stderr);
        return 1;
    }
    try {
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: constructing host\n", stderr);
        Host host;
        if (std::getenv("DREAM_GRAPHICS_TRACE")) std::fputs("graphics: loading application\n", stderr);
        // An application image works here too, so `--graphics` takes what
        // `mind build` makes as well as a directory laid out by hand.
        if (std::filesystem::is_regular_file(application)) {
            Image image;
            std::string error;
            if (!image.load_file(application, error)) throw std::runtime_error(error);
            host.start(image, application, {});
        } else {
            host.start(std::filesystem::canonical(application));
        }
        if (!host.done) gtk_main();
        return host.status;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dream graphics: %s\n", error.what());
        return 1;
    }
}

int run_graphics_image(const Image& image, const std::string& path, const std::vector<std::string>& args) {
    prepare_display();
    if ((!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY")) || !gtk_init_check(nullptr, nullptr)) {
        std::fputs("dream: graphics_unavailable: no display (run it with `--headless` for its backend alone)\n", stderr);
        return 1;
    }
    try {
        Host host;
        host.start(image, path, args);
        if (!host.done) gtk_main();
        return host.status;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dream graphics: %s\n", error.what());
        return 1;
    }
}
}

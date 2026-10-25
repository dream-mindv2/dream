#pragma once
// GTK 3 and WebKitGTK, as much of them as graphics.cpp uses, found at run time.
//
// The VM a release ships is self-contained: it needs the C library and
// nothing else, so one binary starts on every Linux of its architecture. A
// window needs GTK and WebKit, which are neither small nor everywhere, and a
// VM linked against them would not start at all where they are missing --
// on a server, in a container -- though it has a window to open on almost
// no run. So they are not linked. The first time an application opens,
// `load_graphics_api` finds WebKitGTK the way `std.ffi` finds a library
// (`find_system_library`) and looks up every function below in it: GTK,
// GLib, GIO, JavaScriptCore and libsoup are all its dependencies, so one
// handle answers for all of them, and it answers with the libsoup that
// WebKit was built against. Where it is not found, an application says
// `graphics_unavailable` and what was tried, and nothing else changes.
//
// API 4.1 is preferred and 4.0 accepted: they differ in the libsoup they
// use (3 and 2.4), and everything here is spelled the same in both. Nothing
// is taken from their headers -- a build needs none of them installed -- so
// what is declared here is the stable C ABI: opaque structures, enum values
// as GTK 3 and WebKitGTK 2.40+ define them, and the few macros graphics.cpp
// uses as the casts and calls they expand to.
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>

namespace dream::gtk {

using gboolean = int;
using gint = int;
using guint = unsigned int;
using gulong = unsigned long;
using guint32 = uint32_t;
using gint64 = int64_t;
using gsize = size_t;
using gssize = ssize_t;
using gchar = char;
using gpointer = void*;
using gconstpointer = const void*;
using GType = gsize;
using GQuark = guint32;
constexpr gboolean TRUE = 1;
constexpr gboolean FALSE = 0;

struct GError {
    GQuark domain;
    gint code;
    gchar* message;
};

#define DREAM_GTK_OPAQUE(X)                                                                        \
    X(GObject) X(GAsyncResult) X(GBytes) X(GCancellable) X(GMainContext) X(GInputStream)          \
    X(GOutputStream) X(GPollableInputStream) X(GPollableOutputStream) X(GSubprocess) X(GClosure) \
    X(GtkWidget) X(GtkWindow) X(GtkContainer) X(GdkEvent) X(JSCValue) X(SoupMessageHeaders)      \
    X(WebKitWebView) X(WebKitWebContext) X(WebKitSettings) X(WebKitSecurityManager)              \
    X(WebKitUserContentManager) X(WebKitUserScript) X(WebKitJavascriptResult)                    \
    X(WebKitURISchemeRequest) X(WebKitURISchemeResponse) X(WebKitURIRequest)                     \
    X(WebKitPolicyDecision) X(WebKitNavigationPolicyDecision) X(WebKitNavigationAction)
#define DREAM_GTK_DECLARE(T) struct T;
DREAM_GTK_OPAQUE(DREAM_GTK_DECLARE)
#undef DREAM_GTK_DECLARE

using GCallback = void (*)();
using GSourceFunc = gboolean (*)(gpointer);
using GDestroyNotify = void (*)(gpointer);
using GClosureNotify = void (*)(gpointer, GClosure*);
using GAsyncReadyCallback = void (*)(GObject*, GAsyncResult*, gpointer);
using WebKitURISchemeRequestCallback = void (*)(WebKitURISchemeRequest*, gpointer);

// Enumerations, by value.
using GSubprocessFlags = int;
using GtkWindowType = int;
using SoupMessageHeadersType = int;
using WebKitLoadEvent = int;
using WebKitPolicyDecisionType = int;
using WebKitNavigationType = int;
using WebKitWebProcessTerminationReason = int;
using WebKitUserContentInjectedFrames = int;
using WebKitUserScriptInjectionTime = int;
constexpr GSubprocessFlags G_SUBPROCESS_FLAGS_STDIN_PIPE = 1 << 0;
constexpr GSubprocessFlags G_SUBPROCESS_FLAGS_STDOUT_PIPE = 1 << 2;
constexpr gint G_IO_ERROR_WOULD_BLOCK = 27;
constexpr gboolean G_SOURCE_REMOVE = FALSE;
constexpr gboolean G_SOURCE_CONTINUE = TRUE;
constexpr gint64 G_USEC_PER_SEC = 1000000;
constexpr GtkWindowType GTK_WINDOW_TOPLEVEL = 0;
constexpr guint32 GDK_CURRENT_TIME = 0;
constexpr SoupMessageHeadersType SOUP_MESSAGE_HEADERS_RESPONSE = 1;
constexpr WebKitPolicyDecisionType WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION = 0;
constexpr WebKitPolicyDecisionType WEBKIT_POLICY_DECISION_TYPE_NEW_WINDOW_ACTION = 1;
constexpr WebKitNavigationType WEBKIT_NAVIGATION_TYPE_LINK_CLICKED = 0;
constexpr WebKitUserContentInjectedFrames WEBKIT_USER_CONTENT_INJECT_TOP_FRAME = 1;
constexpr WebKitUserScriptInjectionTime WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START = 0;

// Every function graphics.cpp calls: the library's name for it, what it
// answers, and what it takes. Each becomes a pointer of that name, so the
// host is written as it would be against the headers.
#define DREAM_GTK_FUNCTIONS(X)                                                                                        \
    X(void, g_object_unref, (gpointer))                                                                               \
    X(gpointer, g_object_new, (GType, const gchar*, ...))                                                             \
    X(void, g_object_set, (gpointer, const gchar*, ...))                                                              \
    X(gulong, g_signal_connect_data, (gpointer, const gchar*, GCallback, gpointer, GClosureNotify, int))             \
    X(guint, g_timeout_add, (guint, GSourceFunc, gpointer))                                                           \
    X(gboolean, g_source_remove, (guint))                                                                             \
    X(gboolean, g_main_context_iteration, (GMainContext*, gboolean))                                                  \
    X(gint64, g_get_monotonic_time, ())                                                                               \
    X(void, g_usleep, (gulong))                                                                                       \
    X(gboolean, g_error_matches, (const GError*, GQuark, gint))                                                       \
    X(void, g_error_free, (GError*))                                                                                  \
    X(void, g_clear_error, (GError**))                                                                                \
    X(GQuark, g_io_error_quark, ())                                                                                   \
    X(GBytes*, g_bytes_new, (gconstpointer, gsize))                                                                   \
    X(void, g_bytes_unref, (GBytes*))                                                                                 \
    X(gconstpointer, g_bytes_get_data, (GBytes*, gsize*))                                                             \
    X(GCancellable*, g_cancellable_new, ())                                                                           \
    X(void, g_cancellable_cancel, (GCancellable*))                                                                    \
    X(GInputStream*, g_memory_input_stream_new_from_bytes, (GBytes*))                                                 \
    X(GSubprocess*, g_subprocess_newv, (const gchar* const*, GSubprocessFlags, GError**))                             \
    X(GOutputStream*, g_subprocess_get_stdin_pipe, (GSubprocess*))                                                    \
    X(GInputStream*, g_subprocess_get_stdout_pipe, (GSubprocess*))                                                    \
    X(void, g_subprocess_force_exit, (GSubprocess*))                                                                  \
    X(gboolean, g_subprocess_wait, (GSubprocess*, GCancellable*, GError**))                                           \
    X(void, g_subprocess_wait_async, (GSubprocess*, GCancellable*, GAsyncReadyCallback, gpointer))                    \
    X(gssize, g_pollable_output_stream_write_nonblocking,                                                             \
      (GPollableOutputStream*, const void*, gsize, GCancellable*, GError**))                                          \
    X(gssize, g_pollable_input_stream_read_nonblocking, (GPollableInputStream*, void*, gsize, GCancellable*, GError**)) \
    X(gboolean, g_output_stream_write_all, (GOutputStream*, const void*, gsize, gsize*, GCancellable*, GError**))     \
    X(gboolean, g_output_stream_close, (GOutputStream*, GCancellable*, GError**))                                     \
    X(gboolean, gtk_init_check, (int*, char***))                                                                      \
    X(void, gtk_main, ())                                                                                             \
    X(void, gtk_main_quit, ())                                                                                        \
    X(guint, gtk_main_level, ())                                                                                      \
    X(GtkWidget*, gtk_window_new, (GtkWindowType))                                                                    \
    X(void, gtk_window_set_title, (GtkWindow*, const gchar*))                                                         \
    X(void, gtk_window_set_default_size, (GtkWindow*, gint, gint))                                                    \
    X(void, gtk_window_resize, (GtkWindow*, gint, gint))                                                              \
    X(void, gtk_widget_show_all, (GtkWidget*))                                                                        \
    X(void, gtk_widget_destroy, (GtkWidget*))                                                                         \
    X(void, gtk_container_add, (GtkContainer*, GtkWidget*))                                                           \
    X(gboolean, gtk_show_uri_on_window, (GtkWindow*, const char*, guint32, GError**))                                 \
    X(gboolean, jsc_value_is_object, (JSCValue*))                                                                     \
    X(gboolean, jsc_value_is_string, (JSCValue*))                                                                     \
    X(gboolean, jsc_value_is_number, (JSCValue*))                                                                     \
    X(JSCValue*, jsc_value_object_get_property, (JSCValue*, const char*))                                             \
    X(GBytes*, jsc_value_to_string_as_bytes, (JSCValue*))                                                             \
    X(double, jsc_value_to_double, (JSCValue*))                                                                       \
    X(SoupMessageHeaders*, soup_message_headers_new, (SoupMessageHeadersType))                                        \
    X(void, soup_message_headers_append, (SoupMessageHeaders*, const char*, const char*))                             \
    X(GType, webkit_web_view_get_type, ())                                                                            \
    X(const gchar*, webkit_web_view_get_uri, (WebKitWebView*))                                                        \
    X(void, webkit_web_view_evaluate_javascript,                                                                      \
      (WebKitWebView*, const char*, gssize, const char*, const char*, GCancellable*, GAsyncReadyCallback, gpointer))  \
    X(JSCValue*, webkit_web_view_evaluate_javascript_finish, (WebKitWebView*, GAsyncResult*, GError**))               \
    X(void, webkit_web_view_load_uri, (WebKitWebView*, const gchar*))                                                 \
    X(WebKitSettings*, webkit_web_view_get_settings, (WebKitWebView*))                                                \
    X(WebKitWebContext*, webkit_web_context_new_ephemeral, ())                                                        \
    X(void, webkit_web_context_register_uri_scheme,                                                                   \
      (WebKitWebContext*, const gchar*, WebKitURISchemeRequestCallback, gpointer, GDestroyNotify))                    \
    X(WebKitSecurityManager*, webkit_web_context_get_security_manager, (WebKitWebContext*))                           \
    X(void, webkit_security_manager_register_uri_scheme_as_secure, (WebKitSecurityManager*, const gchar*))            \
    X(void, webkit_security_manager_register_uri_scheme_as_cors_enabled, (WebKitSecurityManager*, const gchar*))      \
    X(WebKitUserContentManager*, webkit_user_content_manager_new, ())                                                 \
    X(gboolean, webkit_user_content_manager_register_script_message_handler, (WebKitUserContentManager*, const gchar*)) \
    X(void, webkit_user_content_manager_add_script, (WebKitUserContentManager*, WebKitUserScript*))                   \
    X(WebKitUserScript*, webkit_user_script_new,                                                                      \
      (const gchar*, WebKitUserContentInjectedFrames, WebKitUserScriptInjectionTime, const gchar* const*,             \
       const gchar* const*))                                                                                          \
    X(void, webkit_user_script_unref, (WebKitUserScript*))                                                            \
    X(JSCValue*, webkit_javascript_result_get_js_value, (WebKitJavascriptResult*))                                    \
    X(const gchar*, webkit_uri_scheme_request_get_uri, (WebKitURISchemeRequest*))                                     \
    X(const gchar*, webkit_uri_scheme_request_get_http_method, (WebKitURISchemeRequest*))                             \
    X(void, webkit_uri_scheme_request_finish_with_response, (WebKitURISchemeRequest*, WebKitURISchemeResponse*))      \
    X(WebKitURISchemeResponse*, webkit_uri_scheme_response_new, (GInputStream*, gint64))                              \
    X(void, webkit_uri_scheme_response_set_status, (WebKitURISchemeResponse*, guint, const gchar*))                   \
    X(void, webkit_uri_scheme_response_set_content_type, (WebKitURISchemeResponse*, const gchar*))                    \
    X(void, webkit_uri_scheme_response_set_http_headers, (WebKitURISchemeResponse*, SoupMessageHeaders*))             \
    X(const gchar*, webkit_uri_request_get_uri, (WebKitURIRequest*))                                                  \
    X(void, webkit_policy_decision_ignore, (WebKitPolicyDecision*))                                                   \
    X(WebKitNavigationAction*, webkit_navigation_policy_decision_get_navigation_action,                               \
      (WebKitNavigationPolicyDecision*))                                                                              \
    X(WebKitURIRequest*, webkit_navigation_action_get_request, (WebKitNavigationAction*))                             \
    X(WebKitNavigationType, webkit_navigation_action_get_navigation_type, (WebKitNavigationAction*))                  \
    X(gboolean, webkit_navigation_action_is_user_gesture, (WebKitNavigationAction*))

#define DREAM_GTK_POINTER(R, name, params) inline R(*name) params = nullptr;
DREAM_GTK_FUNCTIONS(DREAM_GTK_POINTER)
#undef DREAM_GTK_POINTER

// The macros graphics.cpp uses, as what they expand to.
#define G_CALLBACK(f) reinterpret_cast<::dream::gtk::GCallback>(f)
#define g_signal_connect(instance, signal, handler, data) \
    ::dream::gtk::g_signal_connect_data((instance), (signal), (handler), (data), nullptr, 0)
#define G_IO_ERROR (::dream::gtk::g_io_error_quark())
#define WEBKIT_TYPE_WEB_VIEW (::dream::gtk::webkit_web_view_get_type())
#define GTK_WINDOW(p) reinterpret_cast<::dream::gtk::GtkWindow*>(p)
#define GTK_WIDGET(p) reinterpret_cast<::dream::gtk::GtkWidget*>(p)
#define GTK_CONTAINER(p) reinterpret_cast<::dream::gtk::GtkContainer*>(p)
#define WEBKIT_WEB_VIEW(p) reinterpret_cast<::dream::gtk::WebKitWebView*>(p)
#define WEBKIT_NAVIGATION_POLICY_DECISION(p) reinterpret_cast<::dream::gtk::WebKitNavigationPolicyDecision*>(p)
#define G_POLLABLE_INPUT_STREAM(p) reinterpret_cast<::dream::gtk::GPollableInputStream*>(p)
#define G_POLLABLE_OUTPUT_STREAM(p) reinterpret_cast<::dream::gtk::GPollableOutputStream*>(p)

/// Find WebKitGTK and every function above in it, once. False, with every
/// place that was tried or the function that was missing, when it cannot.
bool load_graphics_api(std::string* why);
/// The WebKitGTK file `load_graphics_api` would use, or "" when none is
/// found. It is opened, which starts nothing: GTK is not initialised.
std::string webkit_found();

}  // namespace dream::gtk

#pragma once
#include <cstdio>
#include <string>
#include <vector>

namespace dream {
class Image;

/// The payload that makes an image an application: `mind build` carries the
/// page's files as `graphics/` and each one's path.
constexpr const char* graphics_page_payload = "graphics/index.html";

/// An application directory laid out by hand -- `assets/` and an optional
/// `backend.dream` -- or an application image, opened in a window. On
/// desktop Linux, with GTK and WebKitGTK found at run time (graphics_api.hpp);
/// elsewhere, and where they are missing, `graphics_unavailable`.
int run_graphics(const std::string& application);
/// The application `image` carries, opened in a window: the page from its
/// `graphics/` payloads, and the image itself run `--headless` as the
/// backend, given `args`.
int run_graphics_image(const Image& image, const std::string& path, const std::vector<std::string>& args);
/// The WebKitGTK a window would use, as a file, or "" when there is none
/// here (or no windows on this platform).
std::string graphics_library();
}

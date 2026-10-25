#pragma once
#include <cstdio>
#include <string>
#include <vector>

namespace dream {
class Image;

/// The payload that makes an image an application: `mind build` carries the
/// page's files as `graphics/` and each one's path.
constexpr const char* graphics_page_payload = "graphics/index.html";

#ifdef DREAM_HAVE_GRAPHICS
int run_graphics(const std::string& application);
/// The application `image` carries, opened in a window: the page from its
/// `graphics/` payloads, and the image itself run `--headless` as the
/// backend, given `args`.
int run_graphics_image(const Image& image, const std::string& path, const std::vector<std::string>& args);
#else
inline int run_graphics(const std::string&) {
    std::fputs("dream: graphics_unavailable: this VM was built without graphics support\n", stderr);
    return 1;
}
inline int run_graphics_image(const Image&, const std::string&, const std::vector<std::string>&) {
    std::fputs("dream: graphics_unavailable: this image is a graphics application, and this VM was built "
               "without graphics support; run it with `--headless` for its backend alone\n", stderr);
    return 1;
}
#endif
}

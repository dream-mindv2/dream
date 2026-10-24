#pragma once
#include <cstdio>
#include <string>

namespace dream {
#ifdef DREAM_HAVE_GRAPHICS
int run_graphics(const std::string& application);
#else
inline int run_graphics(const std::string&) {
    std::fputs("dream: graphics_unavailable: this VM was built without graphics support\n", stderr);
    return 1;
}
#endif
}

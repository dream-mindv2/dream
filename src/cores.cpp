#include "cores.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace dream {

namespace {

#if defined(__linux__)
/// The quota a cgroup puts on this process, as a whole number of cores rounded
/// up, or 0 when there is none (or it cannot be read). Only the cgroup this
/// process is in is asked, through the mount a container sees as its own; a
/// quota set higher up the hierarchy is what the container runtime already
/// reflects there.
unsigned cgroup_quota() {
    // v2: "max 100000" or "150000 100000".
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {
        char quota[32] = {0};
        long long period = 0;
        int got = std::fscanf(f, "%31s %lld", quota, &period);
        std::fclose(f);
        if (got == 2 && std::strcmp(quota, "max") != 0 && period > 0) {
            long long q = std::atoll(quota);
            if (q > 0) return unsigned(std::max<long long>(1, (q + period - 1) / period));
        }
        return 0;
    }
    // v1: a quota of -1 is none.
    long long q = -1, period = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(f, "%lld", &q) != 1) q = -1;
        std::fclose(f);
    }
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
        if (std::fscanf(f, "%lld", &period) != 1) period = 0;
        std::fclose(f);
    }
    if (q > 0 && period > 0) return unsigned(std::max<long long>(1, (q + period - 1) / period));
    return 0;
}
#endif

unsigned compute() {
    if (const char* env = std::getenv("DREAM_CORES")) {
        long long n = std::atoll(env);
        if (n > 0) return unsigned(std::min<long long>(n, 1024));
    }
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) n = 1;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        int c = CPU_COUNT(&set);
        if (c > 0) n = std::min(n, unsigned(c));
    }
    if (unsigned q = cgroup_quota()) n = std::min(n, q);
#endif
    return std::max(n, 1u);
}

}  // namespace

unsigned usable_cores() {
    static const unsigned n = compute();
    return n;
}

}  // namespace dream

// How many cores this process may actually run on.
//
// `std::thread::hardware_concurrency()` answers how many the *machine* has,
// which is not the question. A process pinned with `taskset`, started by a
// container runtime with `--cpus=2`, or run as a CI job on a slice of a big
// host is told the host's count -- measured here, a VM pinned to one core of
// four still asked for four workers and four collector threads, and a
// self-compile took 15.6 s where the same VM told `-j 1` took 11.7. Every
// thread past the cores that exist is a thread that only takes turns with the
// ones doing the work, and the collector's helpers make it worse: a parallel
// mark that waits on a helper the kernel has not scheduled is slower than a
// serial one.
//
// So the answer is the smallest of three limits: the machine, the affinity
// mask, and the CFS quota a cgroup sets (v2's `cpu.max`, or v1's
// `cpu.cfs_quota_us` over `cpu.cfs_period_us`), rounded up -- a quota of 1.5
// cores still lets two threads make progress. `DREAM_CORES` overrides all
// three, which is how to ask what a smaller machine would do without owning
// one. Computed once; the answer is at least 1.

#pragma once

namespace dream {

unsigned usable_cores();

}  // namespace dream

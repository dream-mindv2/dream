#!/usr/bin/env python3
"""The critical path of a run recorded with DREAM_TIMELINE=FILE.

    DREAM_TIMELINE=/tmp/t.txt dream prog.dream ...
    dream/tools/timeline.py /tmp/t.txt [--chrome out.json] [--segments N]

A parallel program's wall clock is its longest chain of dependent work. The
VM records three things (scheduler.cpp, "DREAM_TIMELINE"): each slice a
process ran, each spawn, and each join that had to wait. The chain is found
by walking back from the root process's exit: through the process's own
slices, across a join wait to the end of the process it waited on, and from a
process's first slice to the moment its parent spawned it. What is on the
chain is either work (a slice), a wait for a worker (runnable, not running),
or a wait nothing recorded explains (a message, a timer, IO).

With `--profile N` and DREAM_PROFILE_PID=PID as well, every 1024th step of
that process is sampled, and each stretch of it on the path lists the
functions it was in -- run with `--no-jit`, since a compiled body is not
sampled. The root is the one to ask: its stretches are the serial part.

Each process is named by what it was spawned to run -- the function its work
closure or suspension belongs to -- so the summary says which kind of part
the chain runs through, and the segments say in which order.
"""
import argparse
import bisect
import collections
import json
import sys


def read(path):
    slices = collections.defaultdict(list)   # pid -> [(t0, t1, end, worker)]
    spawned = {}                              # child -> (parent, t, label)
    joins = collections.defaultdict(list)     # pid -> [(t, target)]
    copies = collections.defaultdict(list)    # pid -> [(t0, t1)]: work or results copied between heaps
    gc = collections.Counter()                # (pid, t0) -> microseconds of that slice collecting
    samples = collections.defaultdict(list)   # pid -> [(t, function)], with --profile
    owner = None
    with open(path) as f:
        for line in f:
            kind, end, worker, sched, pid, other, t0, t1, label = line.rstrip("\n").split(" ", 8)
            sched, pid, other, t0, t1 = int(sched), int(pid), int(other), int(t0), int(t1)
            if owner is None:
                owner = sched
            owner = min(owner, sched)
            if sched != owner:
                continue
            if kind == "s":
                slices[pid].append((t0, t1, end, int(worker)))
                gc[(pid, t0)] = other
            elif kind == "c":
                spawned[other] = (pid, t1, "" if label == "-" else label)
                copies[pid].append((t0, t1))
            elif kind == "x":
                copies[pid].append((t0, t1))
            elif kind == "r":
                samples[pid].append((t0, label))
            elif kind == "j":
                joins[pid].append((t0, other))
    for v in slices.values():
        v.sort()
    for v in joins.values():
        v.sort()
    return slices, spawned, joins, copies, gc, samples


def label_of(pid, spawned):
    return spawned[pid][2] or "?" if pid in spawned else "root"


def critical_path(slices, spawned, joins):
    finish = {pid: s[-1][1] for pid, s in slices.items() if s and s[-1][2] == "h"}
    root = min(pid for pid in slices if pid not in spawned)
    path = []          # (kind, pid, t0, t1), newest first
    pid = root
    t = slices[root][-1][1]
    while True:
        s = slices[pid]
        i = bisect.bisect_left([x[0] for x in s], t) - 1
        jumped = False
        while not jumped:
            t0, t1, _, _ = s[i]
            path.append(("work", pid, t0, min(t1, t)))
            t = t0
            if i == 0:
                break
            p0, p1, pend, _ = s[i - 1]
            if pend == "p":
                # Parked: was it a join, and did what it waited on end in the gap?
                waited = [tg for (jt, tg) in joins.get(pid, []) if p0 <= jt <= p1]
                done = [(finish[tg], tg) for tg in waited if tg in finish and finish[tg] <= t0 + 50]
                if done:
                    ft, tg = max(done)
                    path.append(("wake", pid, ft, t0))
                    pid, t = tg, ft
                    jumped = True
                    continue
                path.append(("wait", pid, p1, t0))
            elif t0 > p1:
                path.append(("queued", pid, p1, t0))
            t = p1
            i -= 1
        if jumped:
            continue
        # The process's first slice: before it, its parent spawned it.
        if pid not in spawned:
            break
        parent, st, _ = spawned[pid]
        path.append(("queued", pid, st, t))
        pid, t = parent, st
    path.reverse()
    return root, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--chrome", help="also write a Chrome/Perfetto trace here")
    ap.add_argument("--segments", type=int, default=40, help="how many of the longest path segments to list")
    a = ap.parse_args()
    slices, spawned, joins, copies, gc, samples = read(a.file)
    if not slices:
        sys.exit("no slices recorded")
    root, path = critical_path(slices, spawned, joins)
    wall = slices[root][-1][1] - min(s[0][0] for s in slices.values())
    cpu = sum(t1 - t0 for s in slices.values() for (t0, t1, _, _) in s)
    print(f"wall {wall / 1000:.0f} ms, cpu in slices {cpu / 1000:.0f} ms, {len(slices)} processes")

    by_kind = collections.Counter()
    by_label = collections.Counter()
    for kind, pid, t0, t1 in path:
        by_kind[kind] += t1 - t0
        if kind == "work":
            by_label[label_of(pid, spawned)] += t1 - t0
    # Of the work on the path, how much was collecting and how much copying
    # between heaps: a slice's collection is charged whole when the path
    # covers any of it, a copy when it lies inside a covered stretch.
    on_gc = on_copy = 0
    covered = collections.defaultdict(list)
    for kind, pid, t0, t1 in path:
        if kind == "work":
            covered[pid].append((t0, t1))
    for pid, spans in covered.items():
        starts = {x[0]: x for x in slices[pid]}
        for t0, t1 in spans:
            for (c0, c1) in copies.get(pid, []):
                if t0 <= c0 and c1 <= t1:
                    on_copy += c1 - c0
        for t0, _ in spans:
            if t0 in starts:
                on_gc += gc[(pid, t0)]
    by_kind["work: collecting"] = on_gc
    by_kind["work: copying between heaps"] = on_copy
    by_kind["work"] -= on_gc + on_copy
    total = sum(by_kind.values())
    print(f"critical path {total / 1000:.0f} ms: " + ", ".join(
        f"{k} {v / 1000:.0f}" for k, v in by_kind.most_common()))
    print("\nwork on the path, by what the process was spawned to run:")
    for lab, v in by_label.most_common(25):
        print(f"  {v / 1000:8.1f} ms  {lab}")

    # Consecutive entries of one process merge into one segment.
    segs = []
    for kind, pid, t0, t1 in path:
        if segs and segs[-1][0] == pid and kind == "work":
            segs[-1][2] = t1
            segs[-1][3] += t1 - t0
        elif kind == "work":
            segs.append([pid, t0, t1, t1 - t0])
    print(f"\nthe path in order ({len(segs)} segments; the {a.segments} longest marked *):")
    longest = set(id(s) for s in sorted(segs, key=lambda s: -s[3])[:a.segments])
    for s in segs:
        if id(s) in longest or len(segs) <= a.segments:
            print(f"  {s[1] / 1000:8.1f} .. {s[2] / 1000:8.1f} ms  work {s[3] / 1000:7.1f}  pid {s[0]:<6} {label_of(s[0], spawned)}")
            # What a profiled process was doing in this stretch: run with
            # `--profile N` and DREAM_PROFILE_PID, it samples every 1024th step.
            inside = collections.Counter(f for (t, f) in samples.get(s[0], []) if s[1] <= t <= s[2])
            n = sum(inside.values())
            if n:
                print("        " + ", ".join(f"{f} {v * 100 // n}%" for f, v in inside.most_common(6)))

    if a.chrome:
        ev = []
        for pid, s in slices.items():
            for t0, t1, end, worker in s:
                ev.append({"name": label_of(pid, spawned) + f" #{pid}", "ph": "X", "ts": t0, "dur": max(t1 - t0, 1),
                           "pid": 1, "tid": worker})
        for kind, pid, t0, t1 in path:
            if kind == "work":
                ev.append({"name": label_of(pid, spawned) + f" #{pid}", "ph": "X", "ts": t0, "dur": max(t1 - t0, 1),
                           "pid": 2, "tid": 0})
        ev.append({"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "workers"}})
        ev.append({"name": "process_name", "ph": "M", "pid": 2, "args": {"name": "critical path"}})
        with open(a.chrome, "w") as f:
            json.dump({"traceEvents": ev}, f)


if __name__ == "__main__":
    main()

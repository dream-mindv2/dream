#!/usr/bin/env python3
"""Compare JIT selection policies with interleaved, output-checked runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time


POLICIES = ('calls', 'density', 'calls_times_size', 'work',
            'calls_per_size_squared', 'adaptive')
WORKLOADS = ('fib', 'sum', 'collatz', 'pi', 'strbuild', 'mapfilter', 'bytescan')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vm', type=Path, default=root / 'build-dream/bin/dream')
    parser.add_argument('--compiler', type=Path, default=root / 'build/dreams.dream')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--policies', nargs='+', choices=POLICIES + ('calls_per_size', 'work_per_size'),
                        default=POLICIES)
    parser.add_argument('--workloads', nargs='+', choices=('self',) + WORKLOADS,
                        default=('self',) + WORKLOADS)
    parser.add_argument('--cpu', type=int, help='restrict children to one CPU')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.runs < 3:
        parser.error('use at least three interleaved runs')
    if args.cpu is not None and not hasattr(os, 'sched_setaffinity'):
        parser.error('--cpu needs sched_setaffinity')
    vm, compiler = args.vm.resolve(), args.compiler.resolve()
    # Diagnostics such as heap verification must not accidentally become part
    # of a timing comparison. Keep only the policy under test from DREAM_JIT_*.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('DREAM_JIT_') and k not in
           ('DREAM_VERIFY_HEAP', 'DREAM_GC_TRACE', 'DREAM_NURSERY_MAX')}

    def run(command, policy=None):
        settings = dict(env)

        def affinity():
            if args.cpu is not None:
                os.sched_setaffinity(0, {args.cpu})

        start = time.perf_counter()
        if policy:
            command = command[:1] + ['--jit-policy', policy] + command[1:]
        result = subprocess.run(command, cwd=root, env=settings, capture_output=True,
                                timeout=180, preexec_fn=affinity if args.cpu is not None else None)
        elapsed = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'exit {result.returncode}: {command}\n'
                               f'{result.stdout.decode()}{result.stderr.decode()}')
        return elapsed, result.stdout

    report = dict(vm=str(vm), compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
                  cpu=args.cpu, runs=args.runs, seconds={}, medians={})
    with tempfile.TemporaryDirectory(prefix='dream-jit-selection-') as temp:
        image = Path(temp) / 'benchmark.dream'
        dest = Path(temp) / 'self.dream'
        if any(w in WORKLOADS for w in args.workloads):
            run([str(vm), '--no-jit', str(compiler), '-L', 'mind',
                 'benchmark/benchmark/src/main.dr', '-o', str(image)])
        for workload in args.workloads:
            samples = {p: [] for p in args.policies}
            reference = None
            # Rotate the order each round so a warm filesystem or thermal drift
            # does not consistently favour one policy. Run children serially.
            for round_index in range(args.runs):
                order = list(args.policies)
                shift = round_index % len(order)
                order = order[shift:] + order[:shift]
                for policy in order:
                    if workload == 'self':
                        command = [str(vm), str(compiler), '-L', 'mind', '-L', '.',
                                   '-o', str(dest), 'dreams/src/main.dr']
                    else:
                        command = [str(vm), str(image), workload]
                    elapsed, output = run(command, policy)
                    if workload == 'self':
                        output = dest.read_bytes()
                        if output != compiler.read_bytes():
                            raise RuntimeError('self-compile differs from the compiler image')
                    if reference is None:
                        reference = output
                    elif output != reference:
                        raise RuntimeError(f'{policy} changes {workload} output')
                    samples[policy].append(elapsed)
                    print(f'{workload} {policy} {elapsed:.4f} s', flush=True)
            report['seconds'][workload] = samples
            report['medians'][workload] = {p: statistics.median(s) for p, s in samples.items()}
    serialized = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(serialized)
    print(serialized)


if __name__ == '__main__':
    main()

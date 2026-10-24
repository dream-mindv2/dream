#!/usr/bin/env python3
"""Whether ordinary Dream runs in a browser: every single-file example, compiled
once and run on the browser VM under Node, must print exactly what is recorded
beside it -- the same bytes the native VM prints.

    compat.py --vm VM --compiler dreams.dream --runtime build-browser --examples examples

A page has no sockets, so an example that serves or connects over the network
is expected to stop, and to stop with the one error that says so: `std.net` is
not provided by this runtime. Anything else from one of those is a failure too.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

# Examples that open sockets, and so cannot run in a page.
NEEDS_NETWORK = {'15_remote', '16_registry', '21_http'}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--vm', type=Path, required=True)
parser.add_argument('--compiler', type=Path, required=True)
parser.add_argument('--runtime', type=Path, required=True, help='browser VM build directory')
parser.add_argument('--examples', type=Path, required=True)
parser.add_argument('-L', dest='libs', action='append', default=[])
args = parser.parse_args()
here = Path(__file__).resolve().parent
runner = here / 'run.mjs'
runtime = (args.runtime / 'dream-browser.mjs').resolve()
libs = [x for lib in args.libs for x in ('-L', lib)]

failures = []
with tempfile.TemporaryDirectory(prefix='dream-browser-compat-') as temp:
    for source in sorted(args.examples.glob('[0-9]*.dr')):
        name = source.stem
        image = Path(temp) / f'{name}.dream'
        built = subprocess.run([str(args.vm), str(args.compiler), str(source), *libs, '-o', str(image)],
                               capture_output=True, text=True, timeout=600)
        if built.returncode:
            failures.append(f'{name}: does not compile\n{built.stdout}{built.stderr}')
            continue
        try:
            ran = subprocess.run(['node', str(runner), str(runtime), str(image)], capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            failures.append(f'{name}: timed out on the browser VM')
            continue
        if name in NEEDS_NETWORK:
            if ran.returncode and 'module `std.net` is not provided by this runtime' in ran.stderr:
                print(f'  {name}: needs the network, and says so')
            else:
                failures.append(f'{name}: expected the std.net refusal, got status {ran.returncode}\n{ran.stderr[-2000:]}')
            continue
        expected = source.with_suffix('.expected').read_text()
        if ran.returncode or ran.stdout != expected:
            failures.append(f'{name}: status {ran.returncode}, output differs\n{ran.stderr[-2000:]}')
        else:
            print(f'  {name}: identical')
for failure in failures:
    print(f'FAIL {failure}', file=sys.stderr)
if failures:
    sys.exit(1)
print('Browser compatibility: every example runs on the browser VM as it does natively')

#!/usr/bin/env python3
"""Console behavior through pipes/files, with interpreter and JIT parity."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--dream', default=str(ROOT / 'build-dream/bin/dream'))
p.add_argument('--compiler', default=str(ROOT / 'build/dreams.dream'))
a = p.parse_args()

with tempfile.TemporaryDirectory(prefix='dream-console-') as directory:
    root = Path(directory)
    number = 0

    def compile_program(body):
        global number
        number += 1
        source = root / f'{number}.dr'
        image = source.with_suffix('.dream')
        source.write_text('import std.console;\nimport std.io;\nimport std.error;\n'
                          'import std.streams;\nimport std.str;\n' + body)
        r = subprocess.run([a.dream, a.compiler, '-L', str(ROOT / 'mind'), str(source), '-o', str(image)],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        return image

    def check(body, stdout, stderr='', stdin='', env=None):
        image = compile_program(body)
        for flags in (['--no-jit'], ['--jit-threshold', '1']):
            r = subprocess.run([a.dream, *flags, str(image)], input=stdin, capture_output=True,
                               text=True, timeout=30, env={**os.environ, **(env or {})})
            assert r.returncode == 0, (body, r.stdout, r.stderr)
            assert r.stdout == stdout, (body, repr(r.stdout), repr(stdout))
            assert r.stderr == stderr, (body, repr(r.stderr), repr(stderr))
        return image

    check('''let main! = {
        console.write! "雪"
        console.print! [1, 2]
        console.line! ()
        console.printf! "{{{}}} = {}" ["answer", 42]
        console.error_write! "warning: "
        console.error! "é"
        console.print! (try! { console.printf! "{} {}" [1] } catch e { error.kind e })
    };''', '雪[1, 2]\n()\n{answer} = 42\n:bad_argument\n', 'warning: é\n')

    check('''let main! = {
        console.print! (console.prompt! "Name? ")
        console.print! (console.confirm! "Continue?" false)
        console.print! (console.prompt_int! "Count? ")
        console.print! (console.confirm! "EOF?" true)
    };''', '雪\ntrue\n42\n()\n',
          'Name? Continue? [y/N] Please answer yes or no.\nContinue? [y/N] '
          'Count? Please enter an integer.\nCount? EOF? [Y/n] ',
          '雪\r\nmaybe\n YES \nnot a number\n42\n')
    check('''let main! = {
        console.print! (console.read_line! ())
        console.print! (console.read_line! ())
        console.print! (console.read_line! ())
    };''', '\nlast\n()\n', stdin='\nlast')
    check('let main! = console.print! (console.confirm! "Proceed?" false);',
          'false\n', 'Proceed? [y/N] ', '\n')

    input_file, output_file = root / 'input.txt', root / 'output.txt'
    input_file.write_text('abcd\n')
    check(f'''let main! = {{
        let h = io.open! {json.dumps(str(input_file))} :read;
        console.print! (try! {{ console.read_line_from! 3 h }} catch e {{ error.kind e }})
        io.close! h
        console.print! (try! {{ console.read_line_from! 0 h }} catch e {{ error.kind e }})
        console.print! (try! {{ console.write_to! h "closed" }} catch e {{ error.kind e }})
    }};''', ':line_too_long\n:bad_argument\n:io_closed\n')

    # A real short writer, including writes through UTF-8 code points, checks
    # both progress and that offsets count bytes rather than characters.
    check(f'''let main! = {{
        let h = io.open! {json.dumps(str(output_file))} :write;
        let small! text = io.write! h (str.slice 0 2 text);
        streams.write_all_with! small! "雪-é-done"
        io.close! h
        console.print! (try! {{ streams.write_all_with! (fn _ -> 0) "x" }} catch e {{ error.kind e }})
        console.print! (try! {{ streams.write_all_with! (fn _ -> 2) "x" }} catch e {{ error.kind e }})
        console.print! (streams.write_all_with! (fn _ -> 1 / 0) "")
    }};''', ':io_error\n:io_error\n:done\n')
    assert output_file.read_text() == '雪-é-done'

    check('''let main! = {
        let logger = console.logger! () |> console.Logger.set_prefix "worker"
                                      |> console.Logger.set_color :off;
        console.log_with! logger :debug (1 / 0)
        console.logf! :debug "{}" []
        console.log_with! logger :info "one\ntwo"
        console.log_with! (console.Logger.set_minimum :off logger) :error (1 / 0)
        console.warn! "careful"
        console.logf! :error "failed: {}" [42]
        console.styled! [:bold, :green] "plain in a pipe"
        console.print! (console.paint! :always (io.stdout! ()) [:red] "red")
        console.print! (io.is_terminal! (io.stdout! ()))
    };''', 'plain in a pipe\n\x1b[31mred\x1b[0m\nfalse\n',
          '[INFO] worker: one\n[INFO] worker: two\n[WARN] careful\n[ERROR] failed: 42\n', env={'TERM': 'xterm'})

    input_file.write_text('injected\n')
    check(f'''let main! = {{
        let input = io.open! {json.dumps(str(input_file))} :read;
        let out = io.open! {json.dumps(str(output_file))} :write;
        let session = console.Session.new input out;
        console.print! (console.prompt_with! session "Question: ")
        console.print! (io.is_open! input && io.is_open! out)
        io.close! input
        io.close! out
    }};''', 'injected\ntrue\n')
    assert output_file.read_text() == 'Question: '

    if os.name == 'posix':
        import errno
        import pty
        image = compile_program('let main! = console.styled! [:green] "tty";')
        for changes, expected in (({}, b'\x1b[32mtty\x1b[0m\r\n'),
                                  ({'NO_COLOR': '1'}, b'tty\r\n'),
                                  ({'TERM': 'dumb'}, b'tty\r\n')):
            master, slave = pty.openpty()
            try:
                r = subprocess.run([a.dream, str(image)], stdin=subprocess.DEVNULL,
                                   stdout=slave, stderr=subprocess.PIPE, timeout=15,
                                   env={**os.environ, 'TERM': 'xterm', 'NO_COLOR': '', **changes})
                assert r.returncode == 0, r.stderr
                os.close(slave)
                slave = None
                data = b''
                while True:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break
                        raise
                    if not chunk:
                        break
                    data += chunk
                assert data == expected, (data, expected)
            finally:
                os.close(master)
                if slave is not None:
                    os.close(slave)

    # Prompt bytes must arrive before the process receives any input.
    image = compile_program('let main! = console.print! (console.prompt! "Ready? ");')
    child = subprocess.Popen([a.dream, str(image)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        import queue
        import threading
        prompt = queue.Queue()
        reader = threading.Thread(target=lambda: prompt.put(child.stderr.read(7)), daemon=True)
        reader.start()
        assert prompt.get(timeout=10) == b'Ready? ', 'prompt was buffered while waiting for input'
        reader.join(timeout=1)
        out, err = child.communicate(b'yes\n', timeout=10)
        assert child.returncode == 0 and out == b'yes\n' and err == b'', (out, err)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

print('console: output, formatting, prompts, EOF, limits, colors, logging and short writes passed')

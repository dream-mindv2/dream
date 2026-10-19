// The debugger: `dream --debug [COMMANDS] IMAGE`.
//
// A lazy program's failure is rarely where it was written, and neither is its
// slowness, and the questions that find either are the same few: where is the
// machine, what has this call been handed, what does this native answer, how
// often is this function entered and with what. They used to be answered by a
// probe written into the VM for the occasion and taken out again. This is
// those probes kept, behind commands:
//
//   break NAME        stop when a function of that name is entered
//   trace NAME        print each call of it, its arguments and what it answers
//   watch NATIVE      print each call of a native or builtin and its answer
//   on NAME CMD; ..   commands to run whenever NAME stops, then go on
//   step, next, finish, continue, bt [N], locals, print sN|cN [DEPTH],
//   node, delete NAME, quit
//
// Commands come from the script given to `--debug`, separated by `;` or new
// lines, and then from the terminal when there is one -- so the same
// session can be typed, or written down and run by a tool. Everything is
// printed to stderr, and nothing is forced: a value not yet evaluated is
// shown as `<lazy>`, with the function it was made in, because a debugger
// that evaluated what it showed would change what the program does.
//
// A debugged run is interpreted. Compiled code keeps its frames in registers
// and calls nothing back per step, so a breakpoint in it would never fire.

#pragma once

#include <string>

#include "runtime.hpp"
#include "value.hpp"

namespace dream {

class Process;

/// Whether `--debug` was given: set once, before anything runs, and read
/// inline because a native call reads it every time.
extern bool g_debugging;
inline bool debug_on() { return g_debugging; }
/// Turn the debugger on, with the commands to start with.
void debug_start(const std::string& script);
/// Before each step the machine evaluates.
void debug_eval(Process& p);
/// After each step that returned a value.
void debug_return(Process& p);
/// After a native or builtin has answered.
void debug_native(Process& p, Value callee, const Value* args, uint32_t argc, const NativeResult& r);

}  // namespace dream

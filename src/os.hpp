// The operating system: arguments, environment, directories, subprocesses.
//
// See os.cpp for why `exec!` uses a helper thread rather than the IO poller.
#pragma once

#include <string>
#include <vector>

#include "runtime.hpp"

namespace dream {

// --- the installation ----------------------------------------------------------
//
// Where installed images live, and the libraries that ship beside them. This
// is written once, here, because it was written six times -- the command line,
// `mind` twice, the compiler twice and the language server -- and the copies
// had drifted: the compiler's never fell back to `~/.mindv2`. The command line
// calls these, a running program asks through `os.install_dirs!`, and
// `std.install` is the Dream side of the same answer.

/// The home directory, or `""` when the environment does not say.
std::string home_dir();

/// A leading `~` as the home directory. `~other`, another user's home, is left
/// alone: guessing where that is would be worse than not answering.
std::string expand_home(const std::string& path);

/// A list of directories joined by the platform's separator, as `PATH` is,
/// split, each with its `~` expanded and the empty ones dropped.
std::vector<std::string> split_dir_list(const std::string& list);

/// `$MINDV2_PATH`; or else the installation this VM came with, `lib/dream`
/// beside the directory its binary is in (`/usr/bin/dream` has
/// `/usr/lib/dream`); or else `~/.mindv2` when that directory exists; or
/// `""` when there is no installation at all.
std::string install_path();

/// The second of those: `lib/dream` beside the running binary's directory,
/// links resolved, or `""` when there is none.
std::string own_installation();

/// The same, split: every directory an installation keeps things in.
std::vector<std::string> install_dirs();

/// Whether `path` names a file (and not a directory).
bool is_file(const std::string& path);

/// The image an installation calls `name` -- `DIR/name` or `DIR/name.dream`
/// for the first `DIR` of `dirs` that has one -- or `""`.
std::string installed_image(const std::string& name, const std::vector<std::string>& dirs);

/// Wait for any subprocess still running. Called when the runtime goes away.
void os_shutdown();

ModuleDef make_os_module();

}  // namespace dream

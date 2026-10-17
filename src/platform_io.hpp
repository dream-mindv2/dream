#pragma once

#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include "windows.hpp"
#include <io.h>
#include <direct.h>
#include <algorithm>
#include <climits>
#include <mutex>
#include <unordered_map>

#ifndef S_ISREG
#define S_ISREG(mode) (((mode) & _S_IFMT) == _S_IFREG)
#define S_ISDIR(mode) (((mode) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC _O_NOINHERIT
#endif
#ifndef SHUT_WR
#define SHUT_WR SD_SEND
#endif
namespace dream::sys {
using FileStat = struct _stat64;
using Offset = int64_t;
using Count = int64_t;
using SockLen = int;

// SOCKET is pointer-sized on Windows, and is not a CRT file descriptor. Keep
// it in a separate table, giving the VM small integer tokens for diagnostics.
inline std::mutex socket_mutex;
inline std::unordered_map<int, SOCKET> sockets;
inline int next_socket = 1 << 24;
inline SOCKET socket_of(int fd) {
    std::lock_guard lock(socket_mutex);
    auto it = sockets.find(fd);
    return it == sockets.end() ? INVALID_SOCKET : it->second;
}
inline int socket_error() {
    switch (WSAGetLastError()) {
    case WSAEWOULDBLOCK: errno = EAGAIN; break;
    case WSAEINPROGRESS: case WSAEALREADY: errno = EINPROGRESS; break;
    case WSAECONNREFUSED: errno = ECONNREFUSED; break;
    case WSAECONNRESET: case WSAECONNABORTED: errno = ECONNRESET; break;
    case WSAEADDRINUSE: errno = EADDRINUSE; break;
    case WSAETIMEDOUT: errno = ETIMEDOUT; break;
    case WSAEINTR: errno = EINTR; break;
    case WSAEACCES: errno = EACCES; break;
    default: errno = EIO; break;
    }
    return -1;
}
inline int keep_socket(SOCKET socket) {
    if (socket == INVALID_SOCKET) return socket_error();
    if (!SetHandleInformation(reinterpret_cast<HANDLE>(socket), HANDLE_FLAG_INHERIT, 0)) {
        closesocket(socket); errno = EIO; return -1;
    }
    std::lock_guard lock(socket_mutex);
    if (next_socket == INT_MAX) { closesocket(socket); errno = EMFILE; return -1; }
    int fd = next_socket++;
    sockets.emplace(fd, socket);
    return fd;
}
inline bool init() {
    static const bool ready = [] { WSADATA data{}; return WSAStartup(MAKEWORD(2, 2), &data) == 0; }();
    for (int fd = 0; fd < 3; ++fd) if (_get_osfhandle(fd) != -1) _setmode(fd, _O_BINARY);
    return ready;
}
inline int close(int fd) {
    SOCKET socket = INVALID_SOCKET;
    {
        std::lock_guard lock(socket_mutex);
        auto it = sockets.find(fd);
        if (it != sockets.end()) { socket = it->second; sockets.erase(it); }
    }
    return socket == INVALID_SOCKET ? ::_close(fd) : ::closesocket(socket);
}
inline int open(const char* path, int flags, int mode) {
    return _wopen(windows::wide(path).c_str(), flags | _O_BINARY | _O_NOINHERIT, mode);
}
inline int stat(const char* path, FileStat* st) { return _wstat64(windows::wide(path).c_str(), st); }
inline int fstat(int fd, FileStat* st) { return _fstat64(fd, st); }
inline int fsync(int fd) { return _commit(fd); }
inline bool is_terminal(int fd) {
    return socket_of(fd) == INVALID_SOCKET && ::_isatty(fd) != 0;
}
inline bool enable_ansi(int fd) {
    if (!is_terminal(fd)) return false;
    HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    DWORD mode = 0;
    return GetConsoleMode(handle, &mode) &&
           SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}
inline Offset lseek(int fd, Offset offset, int origin) { return _lseeki64(fd, offset, origin); }
inline int mkdir(const char* path, int) { return _wmkdir(windows::wide(path).c_str()); }
inline int remove(const char* path) {
    auto wide = windows::wide(path);
    FileStat st{};
    if (_wstat64(wide.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) return _wrmdir(wide.c_str());
    return _wremove(wide.c_str());
}
inline int rename(const char* from, const char* to) {
    if (MoveFileExW(windows::wide(from).c_str(), windows::wide(to).c_str(), MOVEFILE_REPLACE_EXISTING)) return 0;
    errno = GetLastError() == ERROR_FILE_NOT_FOUND ? ENOENT : EACCES;
    return -1;
}
inline Count read(int fd, void* data, size_t size) {
    SOCKET socket = socket_of(fd);
    int n = int(std::min(size, size_t(INT_MAX)));
    if (socket != INVALID_SOCKET) {
        int result = ::recv(socket, static_cast<char*>(data), n, 0);
        return result == SOCKET_ERROR ? socket_error() : result;
    }
    HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (GetFileType(handle) == FILE_TYPE_PIPE) {
        DWORD available = 0;
        if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) {
            if (GetLastError() == ERROR_BROKEN_PIPE) return 0;
            errno = EIO; return -1;
        }
        if (!available) { errno = EAGAIN; return -1; }
        n = int(std::min(DWORD(n), available));
    }
    return _read(fd, data, n);
}
inline Count write(int fd, const void* data, size_t size) {
    SOCKET socket = socket_of(fd);
    int n = int(std::min(size, size_t(INT_MAX)));
    if (socket == INVALID_SOCKET) return _write(fd, data, n);
    int result = ::send(socket, static_cast<const char*>(data), n, 0);
    return result == SOCKET_ERROR ? socket_error() : result;
}
inline bool nonblocking(int fd) {
    SOCKET socket = socket_of(fd);
    if (socket == INVALID_SOCKET) return true;
    u_long on = 1;
    return ::ioctlsocket(socket, FIONBIO, &on) == 0;
}
inline bool ready(int fd, bool writable) {
    SOCKET socket = socket_of(fd);
    if (socket != INVALID_SOCKET) {
        WSAPOLLFD item{socket, short(writable ? POLLWRNORM : POLLRDNORM), 0};
        return WSAPoll(&item, 1, 0) != 0;
    }
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (writable || GetFileType(h) != FILE_TYPE_PIPE) return true;
    DWORD available = 0;
    return !PeekNamedPipe(h, nullptr, 0, nullptr, &available, nullptr) || available != 0;
}
inline int socket(int family, int type, int protocol) { return keep_socket(::socket(family, type, protocol)); }
inline int accept(int fd, sockaddr* addr, SockLen* len) { return keep_socket(::accept(socket_of(fd), addr, len)); }
inline int bind(int fd, const sockaddr* addr, SockLen len) {
    return ::bind(socket_of(fd), addr, len) == SOCKET_ERROR ? socket_error() : 0;
}
inline int listen(int fd, int count) { return ::listen(socket_of(fd), count) == SOCKET_ERROR ? socket_error() : 0; }
inline int connect(int fd, const sockaddr* addr, SockLen len) {
    if (::connect(socket_of(fd), addr, len) != SOCKET_ERROR) return 0;
    socket_error();
    if (errno == EAGAIN) errno = EINPROGRESS;
    return -1;
}
inline int setsockopt(int fd, int level, int name, const void* data, SockLen len) {
    return ::setsockopt(socket_of(fd), level, name, static_cast<const char*>(data), len);
}
inline int getsockopt(int fd, int level, int name, void* data, SockLen* len) {
    if (::getsockopt(socket_of(fd), level, name, static_cast<char*>(data), len) == SOCKET_ERROR) return socket_error();
    if (level == SOL_SOCKET && name == SO_ERROR && *static_cast<int*>(data)) {
        WSASetLastError(*static_cast<int*>(data)); socket_error(); *static_cast<int*>(data) = errno;
    }
    return 0;
}
inline int getpeername(int fd, sockaddr* addr, SockLen* len) {
    return ::getpeername(socket_of(fd), addr, len) == SOCKET_ERROR ? socket_error() : 0;
}
inline int getsockname(int fd, sockaddr* addr, SockLen* len) {
    return ::getsockname(socket_of(fd), addr, len) == SOCKET_ERROR ? socket_error() : 0;
}
inline int shutdown(int fd, int how) { return ::shutdown(socket_of(fd), how); }
}
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <poll.h>
namespace dream::sys {
using FileStat = struct stat;
using Offset = off_t;
using Count = ssize_t;
using SockLen = socklen_t;
using ::open; using ::close; using ::stat; using ::fstat; using ::fsync;
using ::lseek; using ::mkdir; using ::remove; using ::rename;
using ::read; using ::write; using ::bind; using ::listen;
using ::connect; using ::setsockopt; using ::getsockopt;
using ::getpeername; using ::getsockname; using ::shutdown;
inline bool init() { return true; }
inline bool is_terminal(int fd) { return ::isatty(fd) != 0; }
inline bool enable_ansi(int fd) { return is_terminal(fd); }
inline bool nonblocking(int fd) {
    int flags = ::fcntl(fd, F_GETFL, 0);
    return flags >= 0 && ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}
inline int socket(int family, int type, int protocol) {
    int fd = ::socket(family, type, protocol);
    if (fd < 0) return fd;
    if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) { int error = errno; ::close(fd); errno = error; return -1; }
#ifdef SO_NOSIGPIPE
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
    return fd;
}
inline int accept(int listener, sockaddr* addr, SockLen* len) {
    int fd = ::accept(listener, addr, len);
    if (fd < 0) return fd;
    if (::fcntl(fd, F_SETFD, FD_CLOEXEC) < 0) { int error = errno; ::close(fd); errno = error; return -1; }
#ifdef SO_NOSIGPIPE
    int on = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
    return fd;
}
inline bool ready(int fd, bool writable) {
    pollfd item{fd, short(writable ? POLLOUT : POLLIN), 0};
    return ::poll(&item, 1, 0) > 0;
}
}
#endif

#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>

namespace fusionrt::http {

#ifdef _WIN32
using SocketHandle = SOCKET;
using SocketLength = int;
constexpr SocketHandle invalid_socket = INVALID_SOCKET;
inline void close_socket(SocketHandle fd) { closesocket(fd); }

struct SocketRuntime {
    SocketRuntime() {
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) throw std::runtime_error("WSAStartup failed");
    }
    ~SocketRuntime() { WSACleanup(); }
};
#else
using SocketHandle = int;
using SocketLength = socklen_t;
constexpr SocketHandle invalid_socket = -1;
inline void close_socket(SocketHandle fd) { close(fd); }
struct SocketRuntime {};
#endif

class Socket {
public:
    explicit Socket(SocketHandle fd) : fd_(fd) {}
    ~Socket() { if (fd_ != invalid_socket) close_socket(fd_); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept : fd_(other.release()) {}

    SocketHandle get() const { return fd_; }
    SocketHandle release() {
        auto fd = fd_;
        fd_ = invalid_socket;
        return fd;
    }
private:
    SocketHandle fd_;
};

inline bool set_nonblocking(SocketHandle fd) {
#ifdef _WIN32
    u_long yes = 1;
    return ioctlsocket(fd, FIONBIO, &yes) == 0;
#else
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
#endif
}

inline bool would_block() {
#ifdef _WIN32
    const auto error = WSAGetLastError();
    return error == WSAEWOULDBLOCK || error == WSAEINTR;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
#endif
}

inline int socket_ready(SocketHandle fd, bool write, int ms) {
#ifdef _WIN32
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    timeval timeout{ms / 1000, (ms % 1000) * 1000};
    return select(0, write ? nullptr : &fds, write ? &fds : nullptr, nullptr, &timeout);
#else
    pollfd p{fd, static_cast<short>(write ? POLLOUT : POLLIN), 0};
    return poll(&p, 1, ms);
#endif
}

using Clock = std::chrono::steady_clock;
using Write = std::function<bool(std::string_view)>;

inline bool await_socket(SocketHandle fd, bool write, Clock::time_point deadline, std::stop_token stop) {
    while (!stop.stop_requested()) {
        const auto remaining = deadline - Clock::now();
        if (remaining <= Clock::duration::zero()) return false;
        const auto wait = std::chrono::ceil<std::chrono::milliseconds>(remaining);
        const int ready = socket_ready(fd, write, static_cast<int>(std::min(wait.count(), decltype(wait.count()){100})));
        if (ready > 0) return true;
        if (ready < 0 && !would_block()) return false;
    }
    return false;
}

inline int receive(SocketHandle fd, char* buffer, int size, Clock::time_point deadline, std::stop_token stop) {
    while (await_socket(fd, false, deadline, stop)) {
        const auto n = recv(fd, buffer, size, 0);
        if (n >= 0 || !would_block()) return static_cast<int>(n);
    }
    return -1;
}

inline bool send_all(SocketHandle fd, std::string_view bytes, Clock::time_point deadline, std::stop_token stop) {
    while (!bytes.empty()) {
        if (!await_socket(fd, true, deadline, stop)) return false;
        const int size = static_cast<int>(std::min<size_t>(bytes.size(), 64 * 1024));
        const auto n = send(fd, bytes.data(), size, 0);
        if (n < 0 && would_block()) continue;
        if (n <= 0) return false;
        bytes.remove_prefix(static_cast<size_t>(n));
    }
    return true;
}

inline bool disconnected(SocketHandle fd) {
    const int ready = socket_ready(fd, false, 0);
    if (ready <= 0) return ready < 0 && !would_block();
    char c;
    const auto n = recv(fd, &c, 1, MSG_PEEK);
    return n == 0 || (n < 0 && !would_block());
}

class ResponsePipe {
public:
    std::atomic_bool cancelled{false};

    bool write(std::string_view bytes) {
        std::lock_guard lock(mutex_);
        if (cancelled.load(std::memory_order_relaxed)) return false;
        constexpr size_t max_buffered = 16 * 1024 * 1024;
        if (bytes.size() > max_buffered - buffered_) {
            cancelled.store(true, std::memory_order_relaxed);
            return false;
        }
        bytes_.emplace_back(bytes);
        buffered_ += bytes.size();
        ready_.notify_one();
        return true;
    }

    void finish() {
        std::lock_guard lock(mutex_);
        done_ = true;
        ready_.notify_one();
    }

    void flush(SocketHandle fd, std::stop_token stop) {
        using namespace std::chrono_literals;
        auto deadline = Clock::time_point::max();
        while (!stop.stop_requested() && !cancelled.load(std::memory_order_relaxed) && !disconnected(fd)) {
            std::deque<std::string> batch;
            bool done;
            {
                std::unique_lock lock(mutex_);
                ready_.wait_for(lock, 10ms, [&] { return done_ || !bytes_.empty(); });
                batch.swap(bytes_);
                buffered_ = 0;
                done = done_;
            }
            if (!batch.empty() && deadline == Clock::time_point::max()) deadline = Clock::now() + 5min;
            bool sent = true;
            for (const auto& chunk : batch) {
                if (!(sent = send_all(fd, chunk, deadline, stop))) break;
            }
            if (!sent || Clock::now() >= deadline) break;
            if (done) return;
        }
        cancelled.store(true, std::memory_order_relaxed);
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> bytes_;
    size_t buffered_ = 0;
    bool done_ = false;
};

class InferenceQueue {
public:
    using Job = std::function<void()>;

    bool try_push(Job job) {
        std::lock_guard lock(mutex_);
        constexpr size_t max_pending = 8;
        if (jobs_.size() >= max_pending) return false;
        jobs_.push_back(std::move(job));
        ready_.notify_one();
        return true;
    }

    Job take(std::stop_token stop) {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, stop, [&] { return !jobs_.empty(); });
        if (stop.stop_requested()) return {};
        auto job = std::move(jobs_.front());
        jobs_.pop_front();
        return job;
    }

private:
    std::mutex mutex_;
    std::condition_variable_any ready_;
    std::deque<Job> jobs_;
};

}  // namespace fusionrt::http

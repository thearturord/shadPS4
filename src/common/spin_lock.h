// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <thread>

namespace Common {

/**
 * SpinLock class
 * a lock similar to mutex that forces a thread to spin wait instead calling the
 * supervisor. Should be used on short sequences of code.
 */
class SpinLock {
public:
    SpinLock() = default;

    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;

    void lock();
    void unlock();
    [[nodiscard]] bool try_lock();

private:
    void LockSlow();

    std::atomic_flag lck = ATOMIC_FLAG_INIT;
    /// Holder of the lock, for the hang check in LockSlow.
    std::atomic<std::thread::id> owner{};
};

} // namespace Common

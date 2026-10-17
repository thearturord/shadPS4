// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

// Explicit readbacks, step 1 (documents/tlg/native-explicit-readbacks.md): only measures. Records which memory the engine's copy and
// counter read-back records write for the CPU to read (their destinations), which ranges the CPU maps through `AsyncMap`, and which
// CPU read faults hit those ranges, the pages they share with other data, or neither. Debug server endpoint /readbacks.
namespace Core::NativeReadbacks {

/// A copy or counter read-back record writes [address, address + size): `usage` is the Usage of the destination resource (0 default,
/// 1 immutable, 2 dynamic, 3 staging, ~0 unknown), `kind` the record kind (9 counter, 10 copy), `caller` the Ghidra address of the call.
void NoteDestination(VAddr address, u64 size, u32 usage, u32 kind, u64 caller);

/// The engine mapped [address, address + size) for the CPU (`AsyncMap`), from the Ghidra address `caller`.
void NoteMap(VAddr address, u64 size, u64 caller, u32 first_dword = 0);

/// The CPU read a page the GPU wrote and the emulator is about to download it.
void NoteReadFault(VAddr address, u64 size);

/// Explicit staging readbacks are on (bit 33554432 of `native_sdk_strip`; needs the entry list, 8388608, and `readback_dma_async`).
bool ExplicitEnabled();

/// An explicit readback was done (a copy into a STAGING resource: downloaded in order, destination not marked GPU modified), or refused
/// (`reason`: 0 the copy took another path first, 1 larger than the download limit).
void NoteExplicit(bool done, u64 bytes, u32 reason);

/// The buffer cache registers how to wait for the explicit downloads in flight over a range: returns 0 nothing in flight, 1 waited, 2 timed out.
using LandingGate = std::function<u32(VAddr address, u64 size, u32 timeout_ms)>;
void SetLandingGate(LandingGate gate);

/// The `AsyncMap` gate: if [address, address + size) is a staging destination of an explicit readback and a download of it is still in flight,
/// waits for it to land (at most 20 ms), so the engine never gets a pointer to bytes that are not there yet.
void GateMap(VAddr address, u64 size);

/// A landing (the priority thread writing downloaded bytes to guest memory) begins / ends.
void LandingBegin(VAddr address, u64 size);
void LandingEnd(VAddr address, u64 size);

/// The worker function that reads the staging ranges (FUN_01c4c570) is entered / left: the window in which the CPU reads them.
void ReaderEnter();
void ReaderExit();

nlohmann::json Report();

} // namespace Core::NativeReadbacks

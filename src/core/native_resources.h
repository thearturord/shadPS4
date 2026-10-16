// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <nlohmann/json_fwd.hpp>
#include "common/types.h"

// What the native resource model knows about the game's GPU memory: the engine allocates every buffer and texture from two
// heaps of its own (garlic: write combined GPU memory, onion: CPU coherent memory). The census scans those heaps once a
// second and attributes the backend's buffer requests, uploads, image refreshes and CPU write faults to the allocation they
// fall in. It answers which resources the GPU reads, how big they are, who rewrites them and how often (debug server
// endpoint /resources, `?start=1` turns it on).
namespace Core::NativeResources {

/// How a buffer request was answered.
enum class BufferPath : u8 {
    StreamCopy,  ///< read-only, small: copied into the stream buffer
    StreamReuse, ///< the same copy as earlier in the command buffer
    Memo,        ///< large or written, answered from the per-command-buffer memo
    Arena,       ///< large or written, looked up in the arenas (`uploaded` bytes were copied from guest memory)
};

extern bool census_enabled;

/// The engine created a buffer (called after JsBuffer creation, the object is complete).
void NoteBufferObject(u64 object);
/// The engine is initializing a texture (called before the initialization; the object is read on the next scan).
void NoteTextureObject(u64 object);
/// The engine destroys a buffer.
void NoteBufferDestroyed(u64 object);

void NoteBufferRequest(VAddr address, u32 size, bool written, BufferPath path, u64 uploaded);
/// An image was (re)uploaded from guest memory.
void NoteImageUpload(VAddr address, u64 size, bool from_gpu_memory);
/// The CPU wrote tracked memory.
void NoteCpuWrite(VAddr address, u64 size);

/// An engine buffer with Usage IMMUTABLE: the engine writes its contents once, at creation, and never again. Once the backend
/// has synchronized the whole buffer with the GPU, requests for it need no tracker query and no copy. A CPU write fault on the
/// range (the page protection stays as the safety net) or the destruction of the buffer drops that state.
struct ImmutableBuffer {
    u64 begin{};
    u64 end{};
    u64 object{};
    std::atomic<bool> alive{true};
    std::atomic<bool> synced{false};
    std::atomic<u32> generation{0}; ///< bumped by every CPU write to the range
    std::atomic<u64> hash{0};       ///< hash of the guest memory at the first check after the synchronization (0: not taken yet)
    std::atomic<u32> requests{0};
};

/// True when native_sdk_strip bit 65536 is set.
bool ImmutableResidencyEnabled();
/// The live immutable buffer that contains [address, address + size), or null.
ImmutableBuffer* FindImmutable(VAddr address, u32 size);
/// The backend synchronized the whole buffer; `generation` is the one read before it started.
void ImmutableSynced(ImmutableBuffer& buffer, u32 generation);
/// Every 4096th request: true when the guest memory no longer matches what was synchronized (the state is dropped).
bool ImmutableCheck(ImmutableBuffer& buffer);
/// Counters of the residency: hits answered without a copy, whole buffer synchronizations, drops by CPU writes.
void NoteImmutableHit();
void NoteImmutableSync();

/// Image to image copies of the engine's CopyResource: done natively, or refused (reason: 0 no source image, 1 no destination
/// image, 2 the two images do not have the same layout).
void NoteImageCopy(bool done, u32 reason);
/// True when native_sdk_strip bit 131072 is set.
bool ImageCopyEnabled();

void Start();
nlohmann::json Report();

} // namespace Core::NativeResources

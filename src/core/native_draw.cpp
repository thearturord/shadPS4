// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <set>
#include <bitset>
#include <bit>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <tuple>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/logging/log.h"
#include "core/debug_server.h"
#include "core/guest_hooks.h"
#include "core/native_draw.h"
#include "core/native_resources.h"
#include "video_core/native_draw_tag.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_fixed_state.h"
#include "video_core/readback_policy.h"
#include "video_core/untracked_ring.h"
#include "core/memory.h"
#include "core/address_space.h"

namespace Core::NativeDraw {

namespace {

// ---- reading the engine's objects -----------------------------------------------------------------------------
// Offsets: documents/tlg/spike/js_ctx_state.h (JsCtx / JsObj). Everything here reads guest memory directly; the
// context and the objects it points to are guest heap memory owned by the game.
template <typename T>
inline T Rd(u64 base, u64 offset) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
    return value;
}

thread_local bool guard_active;
thread_local bool bad_object_seen;
inline u64 Obj(u64 pointer) {
    if (pointer != 0 && !(pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0)) {
        bad_object_seen = true;
        return 0;
    }
    return pointer;
}

/// Runs a record builder with a guard: the engine's objects are read without knowing which entries are valid, and a stale
/// entry that looks like a pointer must not take the game down. A fault ends the build (no record, the packet path).
using GuardedThunk = u64 (*)(void*);
#ifdef _WIN32
__declspec(noinline) u64 RunGuarded(GuardedThunk thunk, void* argument, bool* faulted) {
    guard_active = true;
    u64 result = 0;
    __try {
        result = thunk(argument);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *faulted = true;
    }
    guard_active = false;
    return result;
}
#else
u64 RunGuarded(GuardedThunk thunk, void* argument, bool*) {
    return thunk(argument);
}
#endif
std::atomic<u64> guarded_faults{};
template <typename F>
u64 Guarded(F&& body) {
    bool faulted = false;
    const u64 result = RunGuarded([](void* argument) -> u64 { return (*static_cast<F*>(argument))(); }, &body, &faulted);
    if (faulted) {
        if (guarded_faults.fetch_add(1) < 8) {
            LOG_WARNING(Core, "native record builder: an engine object held a stale pointer, the draw takes the packet path");
        }
        return 0;
    }
    return result;
}

namespace Ctx {
constexpr u64 CurrentIndexBuffer = 0x58;
constexpr u64 InputLayout = 0x68;
constexpr u64 VertexBufferSlots = 0x70;
constexpr u64 VertexStrides = 0x170;
constexpr u64 VertexOffsets = 0x1f0;
constexpr u64 Topology = 0x4480;
constexpr u64 RenderTargetCount = 0x4484;
constexpr u64 RenderTargets = 0x4488;
constexpr u64 DepthStencilView = 0x44d0;
constexpr u64 Viewport = 0x44ec;
constexpr u64 PixelShader = 0x4690;
constexpr u64 VertexShader = 0x4698;
constexpr u64 RasterizerState = 0x46c8;
constexpr u64 DepthStencilState = 0x46e0;
constexpr u64 StencilRef = 0x46e8;
constexpr u64 BlendState = 0x46f0;
constexpr u64 BlendFactor = 0x46f8;
constexpr u64 InstanceCount = 0x4724;
constexpr u64 ConstantBufferTable(u64 s) { return 0x280 + s * 0x80; }
constexpr u64 ShaderResourceTable(u64 s) { return 0x580 + s * 0x400; }
constexpr u64 SamplerTable(u64 s) { return 0x1d80 + s * 0x80; }
} // namespace Ctx

namespace Shader {
// The stage block starts at shader + 0x30; its used-slot masks are at +0xb0 (constant buffers), +0xb4 (samplers),
// +0xc0/+0xc8 (shader resources, 128 bits).
constexpr u64 CbMask = 0x30 + 0xb0;
constexpr u64 SamplerMask = 0x30 + 0xb4;
constexpr u64 SrvMaskLo = 0x30 + 0xc0;
constexpr u64 SrvMaskHi = 0x30 + 0xc8;
} // namespace Shader

namespace Layout {
constexpr u64 ElementCount = 0x430;
constexpr u64 Elements = 0x38;
constexpr u64 ElementStride = 0x20;
} // namespace Layout

// ---- hashing -----------------------------------------------------------------------------------------------------
struct Hasher {
    u64 h = 0xcbf29ce484222325ull;
    void Add(u64 v) {
        h ^= v;
        h *= 0x100000001b3ull;
        h ^= h >> 29;
    }
};

enum Group : u32 {
    Shaders,
    InputObjects, // layout, index buffer, topology
    Targets,      // render target views, depth view
    States,       // blend, rasterizer, depth-stencil, viewport, factors
    ConstantBuffers,
    ShaderResources,
    Samplers,
    VertexBuffers, // vertex buffer objects, strides, offsets of the slots the layout reads
    Args,          // draw arguments
    NumGroups,
};
constexpr const char* GroupNames[NumGroups] = {"shaders",         "layout_ib_topology", "targets",
                                               "fixed_function",  "constant_buffers",   "shader_resources",
                                               "samplers",        "vertex_buffers",     "draw_args"};

struct Signature {
    std::array<u64, NumGroups> group{};
    u64 pipeline_key = 0; ///< vertex shader, pixel shader, layout, topology (not the index buffer)
    bool valid = false;
};

void HashSlots(Hasher& h, u64 ctx, u64 table, u64 mask_lo, u64 mask_hi) {
    for (u32 slot = 0; slot < 128; ++slot) {
        const u64 bit = slot < 64 ? (mask_lo >> slot) & 1 : (mask_hi >> (slot - 64)) & 1;
        if (bit) {
            h.Add(Rd<u64>(ctx, table + slot * 8));
        }
    }
}

Signature Compute(u64 ctx, u64 a1, u64 a2, u64 a3, u32 kind) {
    Signature s;
    const u64 vs = Rd<u64>(ctx, Ctx::VertexShader);
    const u64 ps = Rd<u64>(ctx, Ctx::PixelShader);
    const u64 layout = Rd<u64>(ctx, Ctx::InputLayout);
    {
        Hasher h;
        h.Add(vs);
        h.Add(ps);
        s.group[Shaders] = h.h;
    }
    {
        Hasher h;
        h.Add(layout);
        h.Add(Rd<u64>(ctx, Ctx::CurrentIndexBuffer));
        h.Add(Rd<u32>(ctx, Ctx::Topology));
        s.group[InputObjects] = h.h;
    }
    {
        Hasher h;
        const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
        h.Add(count);
        for (u32 i = 0; i < 7; ++i) {
            h.Add(Rd<u64>(ctx, Ctx::RenderTargets + i * 8));
        }
        h.Add(Rd<u64>(ctx, Ctx::DepthStencilView));
        s.group[Targets] = h.h;
    }
    {
        Hasher h;
        h.Add(Rd<u64>(ctx, Ctx::BlendState));
        h.Add(Rd<u64>(ctx, Ctx::RasterizerState));
        h.Add(Rd<u64>(ctx, Ctx::DepthStencilState));
        h.Add(Rd<u8>(ctx, Ctx::StencilRef));
        h.Add(Rd<u64>(ctx, Ctx::BlendFactor));
        h.Add(Rd<u64>(ctx, Ctx::BlendFactor + 8));
        for (u32 i = 0; i < 6; ++i) {
            h.Add(Rd<u32>(ctx, Ctx::Viewport + i * 4));
        }
        s.group[States] = h.h;
    }
    {
        Hasher cb, srv, smp;
        const std::array<u64, 2> shaders{vs, ps};
        for (u64 stage = 0; stage < 2; ++stage) {
            const u64 shader = shaders[stage];
            if (!shader) {
                continue;
            }
            const u32 cb_mask = Rd<u32>(shader, Shader::CbMask) & 0xffff;
            for (u32 slot = 0; slot < 16; ++slot) {
                if ((cb_mask >> slot) & 1) {
                    cb.Add(Obj(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8)));
                }
            }
            HashSlots(srv, ctx, Ctx::ShaderResourceTable(stage), Rd<u64>(shader, Shader::SrvMaskLo),
                      Rd<u64>(shader, Shader::SrvMaskHi));
            const u32 smp_mask = Rd<u32>(shader, Shader::SamplerMask) & 0xffff;
            for (u32 slot = 0; slot < 16; ++slot) {
                if ((smp_mask >> slot) & 1) {
                    smp.Add(Obj(Rd<u64>(ctx, Ctx::SamplerTable(stage) + slot * 8)));
                }
            }
            cb.Add(0xffff0000 + stage);
            srv.Add(0xffff0000 + stage);
            smp.Add(0xffff0000 + stage);
        }
        s.group[ConstantBuffers] = cb.h;
        s.group[ShaderResources] = srv.h;
        s.group[Samplers] = smp.h;
    }
    {
        Hasher h;
        if (layout) {
            const u32 n = Rd<u32>(layout, Layout::ElementCount);
            for (u32 i = 0; i < n && i < 32; ++i) {
                const u32 slot = Rd<u32>(layout, Layout::Elements + i * Layout::ElementStride + 8);
                if (slot < 16) {
                    h.Add(Rd<u64>(ctx, Ctx::VertexBufferSlots + slot * 8));
                    h.Add(Rd<u32>(ctx, Ctx::VertexStrides + slot * 4));
                    h.Add(Rd<u32>(ctx, Ctx::VertexOffsets + slot * 4));
                }
            }
        }
        s.group[VertexBuffers] = h.h;
    }
    {
        Hasher h;
        h.Add(kind);
        h.Add(a1);
        h.Add(a2);
        h.Add(a3);
        h.Add(Rd<u32>(ctx, Ctx::InstanceCount));
        s.group[Args] = h.h;
    }
    {
        Hasher h;
        h.Add(vs);
        h.Add(ps);
        h.Add(layout);
        h.Add(Rd<u32>(ctx, Ctx::Topology));
        s.pipeline_key = h.h;
    }
    s.valid = true;
    return s;
}

// ---- statistics --------------------------------------------------------------------------------------------------
enum Class : u32 { ZPass, GBuffer, Forward, Other, NumClasses };
constexpr const char* ClassNames[NumClasses] = {"z_prepass", "g_buffer", "forward", "other"};

struct Counters {
    std::atomic<u64> draws{};
    std::array<std::atomic<u64>, NumGroups> changed{};
    std::atomic<u64> state_changed{};     // any of shaders, layout/ib/topology, targets, fixed function
    std::atomic<u64> resources_changed{}; // any of constant buffers, shader resources, samplers, vertex buffers
    std::atomic<u64> nothing_changed{};   // everything but the draw arguments identical to the previous draw
    std::atomic<u64> only_args_and_vb{};  // state and cb/srv/samplers identical, only vertex buffers / args differ
};
std::array<Counters, NumClasses> counters;
std::atomic<VAddr> image_base{0};
std::atomic<u64> last_publish_ns{0};

// Ghidra address ranges of the three passes (documents/tlg/Legacy/native-phase1-results-2026-10-15.md).
Class Classify(u64 ret) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    if (base == 0 || ret < base) {
        return Other;
    }
    const u64 g = ret - base + 0x1000000;
    if (g >= 0x1c74a00 && g < 0x1c76710) {
        return ZPass;
    }
    if (g >= 0x1c76710 && g < 0x1c779a0) {
        return GBuffer;
    }
    if (g >= 0x1c779a0 && g < 0x1c78f00) {
        return Forward;
    }
    return Other;
}

u64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Where the constant buffers (vertex and pixel stage) of sampled z-prepass and G-buffer draws live: 64 MB region of the
// buffer's memory (JsBuffer + 0x30) and size class (D3D11 ByteWidth at +0x180).
struct CbStats {
    std::mutex mutex;
    std::map<u64, u64> region[2]; // [pass] 64 MB region index -> count
    std::array<u64, 6> size_class[2]{};
    u64 total[2]{};
    u64 draws_sampled[2]{};
    std::map<u64, u64> distinct_objects[2]; // object -> count in the window (capped)
};
CbStats cb_stats;
thread_local u32 sample_counter = 0;

void Publish() {
    nlohmann::json out = nlohmann::json::object();
    for (u32 c = 0; c < NumClasses; ++c) {
        auto& k = counters[c];
        const u64 draws = k.draws.exchange(0);
        if (draws == 0) {
            continue;
        }
        nlohmann::json entry = nlohmann::json::object();
        entry["draws"] = draws;
        const auto pct = [&](std::atomic<u64>& counter) {
            return 100.0 * static_cast<double>(counter.exchange(0)) / static_cast<double>(draws);
        };
        std::array<double, NumGroups> p{};
        nlohmann::json changed = nlohmann::json::object();
        for (u32 g = 0; g < NumGroups; ++g) {
            p[g] = pct(k.changed[g]);
            changed[GroupNames[g]] = p[g];
        }
        entry["percent_of_draws_where_this_differs_from_the_previous_draw"] = changed;
        entry["percent_any_state_changed"] = pct(k.state_changed);
        entry["percent_any_resource_changed"] = pct(k.resources_changed);
        entry["percent_only_vertex_buffers_and_args_differ"] = pct(k.only_args_and_vb);
        entry["percent_everything_but_args_identical"] = pct(k.nothing_changed);
        out[ClassNames[c]] = std::move(entry);
        LOG_INFO(Core,
                 "native draw probe {}: {} draws; differs from the previous draw: shaders {:.1f}% layout/ib {:.1f}% "
                 "targets {:.1f}% states {:.1f}% cb {:.1f}% srv {:.1f}% samplers {:.1f}% vb {:.1f}%",
                 ClassNames[c], draws, p[Shaders], p[InputObjects], p[Targets], p[States], p[ConstantBuffers],
                 p[ShaderResources], p[Samplers], p[VertexBuffers]);
    }
    {
        std::scoped_lock lk{cb_stats.mutex};
        nlohmann::json cbs = nlohmann::json::object();
        for (u32 pass = 0; pass < 2; ++pass) {
            nlohmann::json e = nlohmann::json::object();
            e["draws_sampled"] = cb_stats.draws_sampled[pass];
            e["constant_buffer_uses"] = cb_stats.total[pass];
            e["distinct_buffers"] = cb_stats.distinct_objects[pass].size();
            nlohmann::json regions = nlohmann::json::object();
            for (const auto& [region, count] : cb_stats.region[pass]) {
                regions[fmt::format("{:#x}", region << 26)] = count;
            }
            e["by_64mb_region"] = regions;
            e["by_size_le_64_256_1k_4k_64k_more"] = cb_stats.size_class[pass];
            cbs[pass == 0 ? "z_prepass" : "g_buffer"] = std::move(e);
            LOG_INFO(Core,
                     "native draw probe constant buffers {}: sampled draws {}, uses {}, distinct buffers {}, regions {}, "
                     "size classes (<=64,<=256,<=1k,<=4k,<=64k,more) {} {} {} {} {} {}",
                     pass == 0 ? "z_prepass" : "g_buffer", cb_stats.draws_sampled[pass], cb_stats.total[pass],
                     cb_stats.distinct_objects[pass].size(), cb_stats.region[pass].size(),
                     cb_stats.size_class[pass][0], cb_stats.size_class[pass][1], cb_stats.size_class[pass][2],
                     cb_stats.size_class[pass][3], cb_stats.size_class[pass][4], cb_stats.size_class[pass][5]);
            for (const auto& [region, count] : cb_stats.region[pass]) {
                LOG_INFO(Core, "native draw probe constant buffers {}: region {:#x} uses {}",
                         pass == 0 ? "z_prepass" : "g_buffer", region << 26, count);
            }
            cb_stats.region[pass].clear();
            cb_stats.size_class[pass] = {};
            cb_stats.total[pass] = 0;
            cb_stats.draws_sampled[pass] = 0;
            cb_stats.distinct_objects[pass].clear();
        }
        out["constant_buffers"] = std::move(cbs);
    }
    DebugServer::Publish("native_draw_probe", std::move(out));
}

thread_local Signature previous;


void SampleConstantBuffers(u64 ctx, Class cls) {
    if (cls != ZPass && cls != GBuffer) {
        return;
    }
    if ((++sample_counter & 15) != 0) {
        return;
    }
    const u32 pass = cls == ZPass ? 0 : 1;
    const std::array<u64, 2> shaders{Rd<u64>(ctx, Ctx::VertexShader), Rd<u64>(ctx, Ctx::PixelShader)};
    std::scoped_lock lk{cb_stats.mutex};
    cb_stats.draws_sampled[pass]++;
    for (u64 stage = 0; stage < 2; ++stage) {
        const u64 shader = shaders[stage];
        if (!shader) {
            continue;
        }
        const u32 mask = Rd<u32>(shader, Shader::CbMask) & 0xffff;
        for (u32 slot = 0; slot < 16; ++slot) {
            if (!((mask >> slot) & 1)) {
                continue;
            }
            const u64 obj = Obj(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8));
            if (!obj) {
                continue;
            }
            const u64 address = Rd<u64>(obj, 0x30);
            const u32 bytes = Rd<u32>(obj, 0x180);
            cb_stats.total[pass]++;
            cb_stats.region[pass][address >> 26]++;
            const u32 cls_index = bytes <= 64 ? 0 : bytes <= 256 ? 1 : bytes <= 1024 ? 2 : bytes <= 4096 ? 3 : bytes <= 65536 ? 4 : 5;
            cb_stats.size_class[pass][cls_index]++;
            if (cb_stats.distinct_objects[pass].size() < 100000) {
                cb_stats.distinct_objects[pass][obj]++;
            }
        }
    }
}

u64 Probe(u64 ctx, u64 a1, u64 a2, u64 a3, u64 ret, u32 kind) {
    if (ctx == 0) {
        return 0;
    }
    const Class cls = Classify(ret);
    const Signature now = Compute(ctx, a1, a2, a3, kind);
    auto& k = counters[cls];
    k.draws.fetch_add(1, std::memory_order_relaxed);
    if (previous.valid) {
        bool state = false;
        for (u32 g = 0; g < NumGroups; ++g) {
            if (now.group[g] != previous.group[g]) {
                k.changed[g].fetch_add(1, std::memory_order_relaxed);
            }
        }
        for (u32 g : {Shaders, InputObjects, Targets, States}) {
            state |= now.group[g] != previous.group[g];
        }
        bool resources = false;
        for (u32 g : {ConstantBuffers, ShaderResources, Samplers, VertexBuffers}) {
            resources |= now.group[g] != previous.group[g];
        }
        const bool cb_srv_smp = now.group[ConstantBuffers] != previous.group[ConstantBuffers] ||
                                now.group[ShaderResources] != previous.group[ShaderResources] ||
                                now.group[Samplers] != previous.group[Samplers];
        if (state) {
            k.state_changed.fetch_add(1, std::memory_order_relaxed);
        }
        if (resources) {
            k.resources_changed.fetch_add(1, std::memory_order_relaxed);
        }
        if (!state && !cb_srv_smp) {
            k.only_args_and_vb.fetch_add(1, std::memory_order_relaxed);
        }
        if (!state && !resources) {
            k.nothing_changed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    previous = now;
    SampleConstantBuffers(ctx, cls);

    const u64 t = NowNs();
    u64 last = last_publish_ns.load(std::memory_order_relaxed);
    if (t - last > 5'000'000'000ull && last_publish_ns.compare_exchange_strong(last, t)) {
        if (last != 0) {
            Publish();
        }
    }
    return 0;
}

// ---- writing the tag into the game's draw command buffer ------------------------------------------------------------
// The context's command writer is at ctx + 0x47c0: +0x08 end, +0x10 current position, +0x18 function that makes room
// (writer, dwords, argument) -> bool, +0x20 its argument. This mirrors how the game reserves space for its own packets.
template <typename T>
inline void Wr(u64 base, u64 offset, T value) {
    std::memcpy(reinterpret_cast<void*>(base + offset), &value, sizeof(T));
}

bool StripPackets(u64 begin, u64 end);

/// Synchronization packets that connect the GPU to the CPU or to another queue become tags of the same size, in place: a no-op
/// whose second dword is a magic and whose other dwords are the packet's. The command processor rebuilds the packet from the tag
/// (RestoreSyncTag) and runs the same code, at the same place of the stream, so the timeline and the fence kinds do not change.
///  - graphics end-of-pipe write (6 dwords: fence, frame counter, frame interrupt): magic NativeFenceMagic; the event control
///    dword is not used by anything and is dropped;
///  - release memory (7 dwords, async compute labels): ReleaseMagic; the same;
///  - wait on a memory label (7 dwords): WaitMagicBase + the control dword (function, memory space, engine; only when it fits a byte);
///  - CPU style write of one to four dwords to a label (WRITE_DATA, destination select 2 or 5): WriteMagicBase + the destination select.
/// Returns the kind (SyncEop, SyncRelease, SyncWait) or 0 when the packet is not one of them.
constexpr u32 ReleaseMagic = 0x4E44524Cu;  // "NDRL"
constexpr u32 WaitMagicBase = 0x4E445700u; // "NDW" + control byte
constexpr u32 WriteMagicBase = 0x4E445800u; // "NDX" + destination select
enum SyncKind : u32 { SyncNone, SyncEop, SyncRelease, SyncWait, SyncWrite };

u32 ConvertSyncPacket(u64 at) {
    const u32 header = Rd<u32>(at, 0);
    if ((header >> 30) != 3) {
        return SyncNone;
    }
    const u32 opcode = (header >> 8) & 0xff;
    const u32 count = (header >> 16) & 0x3fff;
    if (opcode == 0x47 && header == 0xC0044700u) {
        Wr<u32>(at, 4, VideoCore::NativeFenceMagic);
        Wr<u32>(at, 0, 0xC0041000u);
        return SyncEop;
    }
    if (opcode == 0x49 && count == 5) {
        Wr<u32>(at, 4, ReleaseMagic);
        Wr<u32>(at, 0, 0xC0051000u);
        return SyncRelease;
    }
    if (opcode == 0x3c && count == 5) {
        const u32 control = Rd<u32>(at, 4);
        if ((control & ~0xffu) == 0 && (control & 0x10) != 0) {
            Wr<u32>(at, 4, WaitMagicBase | control);
            Wr<u32>(at, 0, 0xC0051000u);
            return SyncWait;
        }
    }
    if (opcode == 0x37 && count >= 3 && count <= 6) {
        const u32 control = Rd<u32>(at, 4);
        const u32 destination = (control >> 8) & 0xf;
        if ((control & 0x10000) == 0 && (destination == 2 || destination == 5)) {
            Wr<u32>(at, 4, WriteMagicBase | destination);
            Wr<u32>(at, 0, 0xC0001000u | count << 16);
            return SyncWrite;
        }
    }
    return SyncNone;
}

void CountSyncConverted(u32 kind, std::atomic<u64>& place) {
    Live().sync_converted[kind].fetch_add(1, std::memory_order_relaxed);
    place.fetch_add(1, std::memory_order_relaxed);
}

/// Synchronization the game writes between its commands that only orders the GPU against itself. The raster thread runs the
/// operations in order and the rasterizer inserts the barriers its resources need, so:
///  - a scratch barrier unit (a label inside the command buffer, an end of shader event that writes 1 to it, a wait for it
///    and a cache acquire; the idiom of FUN_0247e510 and its inlined copies) orders a dispatch against later work and the label
///    is read by nothing else;
///  - a cache acquire is ignored by the command processor;
///  - an event write other than the stream out flush and the Z pass dump is ignored by it.
/// Fences with game visible labels, waits on them, CPU writes and flips stay. Returns the number of packets taken out.
u32 SweepSyncPackets(u64 begin, u64 end) {
    u32 removed = 0;
    const u32 policy_bits = VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed);
    for (u64 at = begin; at < end;) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            break;
        }
        const u32 opcode = (header >> 8) & 0xff;
        const u64 size = (((header >> 16) & 0x3fff) + 2) * 4ull;
        if (at + size > end) {
            break;
        }
        const auto next_at = [&](u64 from) {
            const u32 h = Rd<u32>(from, 0);
            return (h >> 30) == 3 ? from + (((h >> 16) & 0x3fff) + 2) * 4ull : 0ull;
        };
        const auto opcode_at = [&](u64 from) { return (Rd<u32>(from, 0) >> 8) & 0xff; };
        if (opcode == 0x10 && Rd<u32>(at, 4) == 0x68753000u) {
            // label NOP, EOS (0x48), WAIT_REG_MEM (0x3c), ACQUIRE_MEM (0x58)
            const u64 eos = at + size;
            const u64 wait = eos < end ? next_at(eos) : 0;
            const u64 acquire = wait != 0 && wait < end ? next_at(wait) : 0;
            const u64 after = acquire != 0 && acquire < end ? next_at(acquire) : 0;
            if (wait != 0 && acquire != 0 && after != 0 && after <= end && opcode_at(eos) == 0x48 && opcode_at(wait) == 0x3c &&
                opcode_at(acquire) == 0x58) {
                const u64 eos_address = static_cast<u64>(Rd<u32>(eos, 8) & 0xfffffffcu) | static_cast<u64>(Rd<u32>(eos, 12) & 0xff) << 32;
                const u64 wait_address = static_cast<u64>(Rd<u32>(wait, 8) & 0xfffffffcu) | static_cast<u64>(Rd<u32>(wait, 12) & 0xffff) << 32;
                if (eos_address == wait_address && eos_address >= at + 8 && eos_address + 8 <= at + size && Rd<u32>(eos, 16) == 1 &&
                    StripPackets(at, after)) {
                    Live().sync_units_stripped.fetch_add(1, std::memory_order_relaxed);
                    removed += 4;
                    at = after;
                    continue;
                }
            }
        } else if (opcode == 0x47) {
            if ((policy_bits & 2097152) != 0) {
                if (const u32 kind = ConvertSyncPacket(at)) {
                    CountSyncConverted(kind, Live().sync_converted_sweep);
                }
            }
        } else if (opcode == 0x49 || opcode == 0x3c || opcode == 0x37) {
            if ((policy_bits & 2097152) != 0) {
                if (const u32 kind = ConvertSyncPacket(at)) {
                    CountSyncConverted(kind, Live().sync_converted_sweep);
                }
            }
        } else if (opcode == 0x58) {
            if (StripPackets(at, at + size)) {
                Live().sync_acquires_stripped.fetch_add(1, std::memory_order_relaxed);
                ++removed;
            }
        } else if (opcode == 0x46) {
            const u32 control = Rd<u32>(at, 4);
            const u32 type = control & 0x3f;
            const u32 index = (control >> 8) & 0xf;
            if (type != 31 && index != 1 && StripPackets(at, at + size)) {
                Live().sync_events_stripped.fetch_add(1, std::memory_order_relaxed);
                ++removed;
            }
        }
        at += size;
    }
    return removed;
}

bool SweepEnabled() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
           (policy.native_sdk_strip.load(std::memory_order_relaxed) & 524288) != 0;
}

bool WriteDwordsAt(u64 writer, const u32* words, u32 count) {
    // the game's own synchronization between the last tag written to this command buffer and this one
    struct Sweep {
        u64 writer{};
        u64 base{};
        u64 last{};
    };
    static thread_local std::array<Sweep, 6> sweeps;
    if (SweepEnabled()) {
        const u64 cursor = Rd<u64>(writer, 0x10);
        const u64 base = Rd<u64>(writer, 0x0);
        Sweep* entry = nullptr;
        for (auto& s : sweeps) {
            if (s.writer == writer) {
                entry = &s;
            }
        }
        if (entry == nullptr) {
            entry = &sweeps[(writer >> 4) % sweeps.size()];
            *entry = {writer, base, 0};
        }
        if (entry->base == base && entry->last != 0 && entry->last < cursor && cursor - entry->last < (1u << 16)) {
            SweepSyncPackets(entry->last, cursor);
        }
        entry->base = base;
        entry->last = cursor + count * sizeof(u32);
    }
    u64 cur = Rd<u64>(writer, 0x10);
    const u64 end = Rd<u64>(writer, 0x08);
    if (cur == 0 || cur > end) {
        return false;
    }
    if (((end - cur) >> 2) < count) {
        using MakeRoom = u8(PS4_SYSV_ABI*)(u64 writer, u32 dwords, u64 argument);
        const auto make_room = reinterpret_cast<MakeRoom>(Rd<u64>(writer, 0x18));
        if (!make_room || !make_room(writer, count, Rd<u64>(writer, 0x20))) {
            return false;
        }
        cur = Rd<u64>(writer, 0x10);
    }
    std::memcpy(reinterpret_cast<void*>(cur), words, count * sizeof(u32));
    Wr<u64>(writer, 0x10, cur + count * sizeof(u32));
    if (SubmitShadowEnabled()) {
        for (u32 i = 0; i + 1 < count;) {
            const u32 header = words[i];
            if ((header >> 30) != 3) {
                ++i;
                continue;
            }
            const u32 size = ((header >> 16) & 0x3fff) + 2;
            if (((header >> 8) & 0xff) == 0x10 && size >= 2 + VideoCore::NativeTagPayloadDwords - 1 && words[i + 1] == VideoCore::NativeTagMagic &&
                i + size <= count) {
                u64 tag_seq;
                std::memcpy(&tag_seq, &words[i + 9], sizeof(tag_seq));
                NoteAnchor(cur + i * sizeof(u32), tag_seq, (words[i + 2] >> 8) & 0xff);
            }
            i += size;
        }
    }
    return true;
}

bool WriteDwords(u64 ctx, const u32* words, u32 count) {
    return WriteDwordsAt(ctx + 0x47c0, words, count);
}

// ---- the native front end: the register state the game's objects describe -------------------------------------------
// PrepareDraw copies register words that the state objects hold into the command buffer (documents/tlg/spike/
// WP7_prepare_draw_notes.md). This builds the same words, per draw, straight from the objects. In shadow mode they are
// compared with the register file the rasterizer got from the command packets.
constexpr u32 ContextBase = 0xA000;
constexpr u32 UconfigBase = 0xC000;
constexpr u32 ShBase = 0x2C00;
constexpr u32 MaxSynthRegs = 256;

struct SynthReg {
    u32 index; ///< index in the register file (dword offset)
    u32 value;
};

struct NativeRecord {
    std::atomic<u64> seq{0};
    u32 count{};
    u64 vs_gnm{}; ///< Gnm shader binaries of the draw (for the shader input dump)
    u64 ps_gnm{};
    u64 dbg_vs{};
    u64 dbg_fetch{};
    u64 dbg_layout{};
    u32 kind{};
    u32 a1{};
    u32 a2{};
    u64 caller{}; ///< Ghidra address of the draw call's return address
    std::array<u32, 8> cb_extent{};
    u32 cb_extent_mask{};
    u32 db_extent{};
    u64 args_addr{};      ///< indirect draws: the argument buffer
    u64 extra_addr{};     ///< copy records: the source address
    u32 vertex_sgpr{};    ///< indirect draws: user data registers that receive the base vertex / start instance
    u32 instance_sgpr{};
    u32 rt_info{};
    u32 flags{}; ///< RecordFlag bits: why this draw cannot run natively
    u32 reason{}; ///< Reason bits of the special pipeline flag (diagnostics, see Reason)
    /// Tables the user data points to (extended user data, descriptor tables): built in `address`, compared by
    /// content with what the register `reg` points to in the packet path's state.
    struct TableCheck {
        u32 reg;
        u32 kind;
        u32 dwords;
        u64 address;
        u32 entry_dwords; ///< size of one table entry (0 = compare the table as a whole)
        std::array<u64, 2> slot_mask; ///< entries the shader reads (bit per slot); the others hold stale data
        u32 hi_flags{};               ///< bits the register after the address carries besides it (buffer descriptor flags)
    };
    u32 check_count{};
    std::array<TableCheck, 8> checks{};
    u64 fill_destination{}; ///< a dispatch of the device's fill shader: the pattern fill it performs
    u32 fill_bytes{};
    u32 fill_dwords{};
    std::array<u32, 4> fill_pattern{};
    u64 copy_destination{}; ///< a dispatch of the device's buffer copy shader (CsCopyBufferDwords): what it copies
    u64 copy_source{};
    u32 copy_bytes{};
    u32 fixed_stale{}; ///< FixedStateStale groups the draw does not write (the hardware keeps what it holds)
    const Vulkan::FixedState* fixed{}; ///< draw records: the fixed-function state built from the engine's objects (null: from the registers)
    u32 unsupported_inputs{}; ///< input usage types the builder does not know
    u32 unsupported_mask{};   ///< those types, bit per type
    std::array<SynthReg, MaxSynthRegs> regs{};
};

/// Why a record is flagged as a special pipeline (diagnostics only).
enum Reason : u32 {
    ReasonVsVariant = 1,        ///< vertex shader has a variant object at +0x420
    ReasonCtx4680 = 2,
    ReasonCtx4688 = 4,
    ReasonDispatchDraw = 8,     ///< ctx + 0x130c8 == 0x200
    ReasonIndirectView = 16,    ///< indirect draw without index buffer view / arguments
    ReasonIndexType = 32,       ///< index type of the context is not the usual one
    ReasonUnsupportedInput = 64,
    ReasonTableOverflow = 128,
    ReasonBadObject = 256,
    ReasonNoFetchObject = 512,
};

enum RecordFlag : u32 {
    NoVertexShader = 1,
    NoFetchObject = 2,
    OddRenderTarget = 4,
    SpecialPipeline = 8,
};

constexpr u32 RecordSlots = 16384;
std::unique_ptr<NativeRecord[]> records;
std::once_flag records_once;
std::atomic<u64> next_seq{1};
std::atomic<u64> consumed_seq{0};

/// Draws are executed from records only for call sites whose records matched what the game's own packets produced.
/// Call sites that set state around the render layer (own shaders, direct packets) never match and stay on the
/// packet path.
struct CallerTrust {
    std::atomic<u64> caller{0};
    std::atomic<u32> total{0};
    std::atomic<u32> bad{0};
    std::atomic<u32> since_audit{0};
};
constexpr u32 TrustSlots = 512;
std::array<CallerTrust, TrustSlots> caller_trust;

CallerTrust* TrustFor(u64 caller) {
    const u32 start = static_cast<u32>((caller * 0x9E3779B97F4A7C15ull) >> 40) % TrustSlots;
    for (u32 i = 0; i < TrustSlots; ++i) {
        auto& entry = caller_trust[(start + i) % TrustSlots];
        u64 current = entry.caller.load(std::memory_order_acquire);
        if (current == caller) {
            return &entry;
        }
        if (current == 0 && entry.caller.compare_exchange_strong(current, caller, std::memory_order_acq_rel)) {
            return &entry;
        }
        if (entry.caller.load(std::memory_order_acquire) == caller) {
            return &entry;
        }
    }
    return nullptr;
}

void NoteConsumed(u64 seq) {
    u64 seen = consumed_seq.load(std::memory_order_relaxed);
    while (seq > seen && !consumed_seq.compare_exchange_weak(seen, seq, std::memory_order_relaxed)) {
    }
}

struct DeviceAddresses {
    u64 device{}; ///< JsDevice (global pointer in the eboot)
    u64 topology_table{};
    u64 depth_stencil_table{};
};

DeviceAddresses ResolveDevice() {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    DeviceAddresses d;
    if (base == 0) {
        return d;
    }
    const auto rt = [&](u64 ghidra) { return base + ghidra - 0x1000000; };
    d.device = Rd<u64>(rt(0x6b03558), 0);
    d.topology_table = rt(0x25e0810);
    d.depth_stencil_table = rt(0x25e07e0);
    return d;
}

/// Descriptor tables the user data of a record points to live as long as the record: each record slot owns an arena
/// (the slot is reused only after the record is consumed, and a consumer that finds the slot reused drops the record).
constexpr u32 TableArenaDwords = 4096;
constexpr u32 RecordSlotCount = RecordSlots;
std::unique_ptr<u32[]> table_arena;
std::once_flag table_arena_once;
thread_local u64 table_seq;
thread_local u32 table_used;
thread_local bool table_overflow;

NativeRecord& Slot(u64 seq);

void BeginTables(u64 seq) {
    Slot(seq).copy_bytes = 0;
    Slot(seq).fill_dwords = 0;
    table_seq = seq;
    table_used = 0;
    table_overflow = false;
    bad_object_seen = false;
}

u32* AllocTable(u32 dwords) {
    std::call_once(table_arena_once, [] {
        table_arena = std::make_unique_for_overwrite<u32[]>(static_cast<size_t>(RecordSlotCount) * TableArenaDwords);
    });
    const u32 size = (dwords + 3) & ~3u;
    if (table_used + size > TableArenaDwords) {
        // does not fit: the record is marked unsupported and its tables go to a scratch buffer nobody reads
        table_overflow = true;
        thread_local std::vector<u32> spill;
        spill.assign(size, 0);
        return spill.data();
    }
    u32* table = table_arena.get() + (table_seq % RecordSlotCount) * TableArenaDwords + table_used;
    table_used += size;
    return table;
}

// Descriptors of the objects bound in the context's tables (api stage 0 = vertex, 1 = pixel).
/// Writes up to `max_dwords` of the descriptor of shader resource slot `slot`; returns the descriptor size in dwords.
u32 ResourceDescriptor(u64 ctx, u32 stage, u32 slot, u32* out, u32 max_dwords) {
    const u64 view = Obj(Rd<u64>(ctx, Ctx::ShaderResourceTable(stage) + slot * 8));
    u32 size = 0;
    if (view) {
        const u32 kind = Rd<u32>(view, 0x44);
        const bool buffer = kind == 1 || kind == 0xb;
        size = buffer ? 4 : 8;
        const u64 src = view + (buffer ? 0x78 : 0x58);
        for (u32 i = 0; i < size && i < max_dwords; ++i) {
            out[i] = Rd<u32>(src, i * 4);
        }
    }
    for (u32 i = size; i < max_dwords; ++i) {
        out[i] = 0;
    }
    return size;
}

/// Unordered access view `slot` of the compute stage (V# for buffers, T# for textures).
u32 UavDescriptor(u64 ctx, u32 slot, u32* out, u32 max_dwords) {
    const u64 view = slot < 8 ? Rd<u64>(ctx, 0x2098 + slot * 8) : 0;
    u32 size = 0;
    if (view) {
        const bool buffer = Rd<u32>(view, 0x44) == 1;
        size = buffer ? 4 : 8;
        const u64 src = view + (buffer ? 0x74 : 0x54);
        for (u32 i = 0; i < size && i < max_dwords; ++i) {
            out[i] = Rd<u32>(src, i * 4);
        }
    }
    for (u32 i = size; i < max_dwords; ++i) {
        out[i] = 0;
    }
    return size;
}

void ConstBufferDescriptor(u64 ctx, u32 stage, u32 slot, u32* out) {
    const u64 buffer = Obj(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8));
    for (u32 i = 0; i < 4; ++i) {
        out[i] = buffer ? Rd<u32>(buffer, 0x60 + i * 4) : 0;
    }
}

void SamplerDescriptor(u64 ctx, u32 stage, u32 slot, u32* out) {
    const u64 sampler = Obj(Rd<u64>(ctx, Ctx::SamplerTable(stage) + slot * 8));
    for (u32 i = 0; i < 4; ++i) {
        out[i] = sampler ? Rd<u32>(sampler, 0x64 + i * 4) : 0;
    }
}

/// Which dwords of the tables behind user data pointers a shader reads: the scalar loads (`s_load_*`) from the SGPR
/// pair that holds the pointer. The constant update engine writes only these entries of a table.
struct ShaderLoads {
    std::array<std::bitset<256>, 16> loaded; ///< per SGPR (first of the pair): dword offsets read with immediate offsets
    std::bitset<16> dynamic;                 ///< loads with an SGPR offset: anything may be read
};

std::mutex shader_loads_mutex;
std::unordered_map<u64, std::unique_ptr<ShaderLoads>> shader_loads_cache;

const ShaderLoads& ScanShaderLoads(u64 gnm) {
    std::scoped_lock lk{shader_loads_mutex};
    auto& entry = shader_loads_cache[gnm];
    if (entry) {
        return *entry;
    }
    entry = std::make_unique<ShaderLoads>();
    ShaderLoads& loads = *entry;
    const u64 code = static_cast<u64>(Rd<u32>(gnm + 8, 0)) << 8;
    constexpr u32 SignatureWord = 0x5362724f; // "OrbS" of the "OrbShdr" footer after the last instruction
    for (u32 pc = 0; pc < 0x8000;) {
        const u32 w = Rd<u32>(code, pc * 4);
        if (w == SignatureWord) {
            break;
        }
        u32 size = 1;
        if ((w >> 23) == 0x17F || (w >> 23) == 0x17E || (w >> 23) == 0x17D) { // SOPP, SOPC, SOP1
            const bool literal = (w & 0xff) == 255 || ((w >> 23) == 0x17E && ((w >> 8) & 0xff) == 255);
            size += literal;
        } else if ((w >> 28) == 0xB) { // SOPK
        } else if ((w >> 30) == 2) {   // SOP2
            size += (w & 0xff) == 255 || ((w >> 8) & 0xff) == 255;
        } else if ((w >> 27) == 0x18) { // SMRD
            const u32 opcode = (w >> 22) & 0x1f;
            if (opcode <= 4) {
                const u32 sgpr = ((w >> 9) & 0x3f) * 2;
                // an x16 load reads a texture descriptor with its fmask words: only the first 8 dwords are the slot
                const u32 dwords = opcode == 4 ? 8u : 1u << opcode;
                if ((w >> 8) & 1) {
                    for (u32 k = 0; k < dwords && sgpr < 16; ++k) {
                        if ((w & 0xff) + k < 256) {
                            loads.loaded[sgpr].set((w & 0xff) + k);
                        }
                    }
                } else if (sgpr < 16) {
                    loads.dynamic.set(sgpr);
                }
            }
        } else if ((w >> 25) == 0x3F || (w >> 25) == 0x3E) { // VOP1, VOPC
            size += (w & 0x1ff) == 255;
        } else if ((w >> 31) == 0) { // VOP2
            const u32 opcode = (w >> 25) & 0x3f;
            size += (w & 0x1ff) == 255 || opcode == 32 || opcode == 33;
        } else {
            const u32 prefix = w >> 26;
            if (prefix == 0x34 || prefix == 0x36 || prefix == 0x37 || prefix == 0x38 || prefix == 0x3A ||
                prefix == 0x3C || prefix == 0x3E) { // VOP3, DS, FLAT, MUBUF, MTBUF, MIMG, EXP
                size = 2;
            }
        }
        pc += size;
    }
    return loads;
}

/// Slots of a table the game's constant update engine fills for this shader: the engine's own masks in the shader's
/// stage block (shader resources 128 bits at +0xc0/+0xc8, constant buffers +0xb0, samplers +0xb4, RW resources +0xd0).
/// The other slots hold whatever the packet memory had before.
std::array<u64, 2> EngineUsedSlots(u64 block, u32 type) {
    switch (type) {
    case 19:
        return {Rd<u64>(block, 0xc0), Rd<u64>(block, 0xc8)};
    case 22:
        return {Rd<u32>(block, 0xb0) & 0xffffu, 0};
    case 21:
        return {Rd<u32>(block, 0xb4) & 0xffffu, 0};
    default:
        return {Rd<u32>(block, 0xd0) & 0xffu, 0};
    }
}

/// Entries of a table (`entry_dwords` per slot) the shader reads through the pointer in SGPR `sgpr`.
std::array<u64, 2> UsedSlots(const ShaderLoads& loads, u32 sgpr, u32 entry_dwords, u32 slots) {
    std::array<u64, 2> mask{};
    const bool unknown = loads.dynamic.test(sgpr) || loads.loaded[sgpr].none();
    for (u32 s = 0; s < slots && s < 128; ++s) {
        bool used = unknown;
        for (u32 k = 0; k < entry_dwords && !used; ++k) {
            const u32 dword = s * entry_dwords + k;
            used = dword < 256 && loads.loaded[sgpr].test(dword);
        }
        if (used) {
            mask[s >> 6] |= u64(1) << (s & 63);
        }
    }
    return mask;
}

thread_local u32 synth_flags;
thread_local u32 synth_reason;
thread_local u32 inline_instances; ///< instance count the game set before an inlined draw (0 = the draw functions' reset to 1)
thread_local std::array<u32, 8> synth_cb_extent;
thread_local u32 synth_cb_extent_mask;
thread_local u32 synth_db_extent;
thread_local u64 synth_fetch_address;
thread_local u64 synth_vs_gnm;
thread_local u32 synth_rt_info; ///< count | view mask << 8 | empty block mask << 16
thread_local u64 dbg_vs;
thread_local u64 dbg_fetch;

u32 Synthesize(u64 ctx, SynthReg* out, u64* vs_gnm_out, u64* ps_gnm_out);

/// User data of one stage (SGPR values the constant update engine would have written): descriptors of the bound
/// objects copied by the shader's input usage table. `sgpr_base` = index of the stage's first user data register.
void SynthesizeUserData(u64 ctx, u32 stage, u64 shader_block, u64 gnm, u64 inputs_offset, u32 sgpr_base, SynthReg* out,
                        u32& n, NativeRecord& record) {
    if (!gnm) {
        return;
    }
    const u32 count = Rd<u8>(gnm, 3);
    std::array<u32, 16> sgpr{};
    std::array<bool, 16> sgpr_set{};
    std::array<u32, 128> eud{};
    u32 eud_dwords = 0;
    struct PendingPtr {
        u32 sgpr;
        u32 kind;
        u32* table;
        u32 dwords;
        u32 entry;
        std::array<u64, 2> mask;
    };
    std::array<PendingPtr, 8> pending{};
    u32 pending_count = 0;
    bool have_eud_ptr = false;
    u32 eud_ptr_sgpr = 0;
    const auto put = [&](u32 start, const u32* words, u32 size) {
        for (u32 i = 0; i < size; ++i) {
            const u32 reg = start + i;
            if (reg < 16) {
                sgpr[reg] = words[i];
                sgpr_set[reg] = true;
            } else if (reg - 16 < eud.size()) {
                eud[reg - 16] = words[i];
                eud_dwords = std::max(eud_dwords, reg - 16 + 1);
            }
        }
    };
    for (u32 i = 0; i < count && i < 40; ++i) {
        const u64 e = gnm + inputs_offset + i * 4;
        const u32 type = Rd<u8>(e, 0);
        const u32 slot = Rd<u8>(e, 1);
        const u32 start = Rd<u8>(e, 2);
        const u32 flags = Rd<u8>(e, 3);
        std::array<u32, 8> words{};
        switch (type) {
        case 0: { // immediate resource (V# of 4 dwords or T# of 8)
            const u32 size = (flags & 1) ? 8 : 4;
            ResourceDescriptor(ctx, stage, slot, words.data(), size);
            put(start, words.data(), size);
            break;
        }
        case 1: // immediate sampler
            SamplerDescriptor(ctx, stage, slot, words.data());
            put(start, words.data(), 4);
            break;
        case 2: // immediate constant buffer
            ConstBufferDescriptor(ctx, stage, slot, words.data());
            put(start, words.data(), 4);
            break;
        case 18: // pointer to the fetch shader
            words[0] = static_cast<u32>(synth_fetch_address);
            words[1] = static_cast<u32>(synth_fetch_address >> 32);
            put(start, words.data(), 2);
            if (synth_fetch_address == 0) {
                ++record.unsupported_inputs;
            }
            break;
        case 4: { // immediate RW resource (compute)
            const u32 size = (flags & 1) ? 8 : 4;
            UavDescriptor(ctx, slot, words.data(), size);
            put(start, words.data(), size);
            break;
        }
        case 26: { // pointer to the global internal resource table (CUE field)
            const u64 table = Rd<u64>(ctx, 0x12f30);
            words[0] = static_cast<u32>(table);
            words[1] = static_cast<u32>(table >> 32);
            put(start, words.data(), 2);
            break;
        }
        case 23: { // pointer to the vertex buffer table: one V# per element of the input layout
            const u64 layout = stage == 0 ? Rd<u64>(ctx, Ctx::InputLayout) : 0;
            const u32 elements = layout ? Rd<u32>(layout, Layout::ElementCount) : 0;
            if (layout == 0 || elements == 0 || elements > 32 || pending_count >= pending.size() || start >= 15) {
                ++record.unsupported_inputs;
                record.unsupported_mask |= 1u << 23;
                break;
            }
            const VAddr image = image_base.load(std::memory_order_relaxed);
            const u64 format_table = image + 0x6b035a0 - 0x1000000;
            const u64 empty_descriptor = image + 0x25380a0 - 0x1000000;
            u32* table = AllocTable(elements * 4);
            for (u32 e = 0; e < elements; ++e) {
                const u64 element = layout + Layout::Elements + e * Layout::ElementStride;
                const u32 vertex_slot = Rd<u32>(element, 8);
                const u64 buffer = Rd<u64>(ctx, Ctx::VertexBufferSlots + vertex_slot * 8);
                u32* v = table + e * 4;
                if (buffer == 0 || vertex_slot >= 32) {
                    for (u32 k = 0; k < 4; ++k) {
                        v[k] = Rd<u32>(empty_descriptor, k * 4);
                    }
                    continue;
                }
                const u32 stride = Rd<u32>(ctx, Ctx::VertexStrides + vertex_slot * 4);
                const u64 address = u64(Rd<u32>(ctx, Ctx::VertexOffsets + vertex_slot * 4)) + Rd<u32>(element, 12) +
                                    Rd<u64>(buffer, 0x30);
                const u32 format = Rd<u32>(format_table, Rd<u32>(element, 4) * 4);
                u32 extra = 0;
                if ((format & 0x800) < 0x800 && ((format & 0xff) - 1) < 0xe) {
                    const u32 shifted = (format & 0xf) << 15;
                    if ((format & 0xf00) == 0x700) {
                        extra = shifted;
                    }
                    if ((format & 0xfe) != 6) {
                        extra = shifted;
                    }
                }
                v[0] = static_cast<u32>(address);
                v[1] = (stride & 0x3fff) << 16 | static_cast<u32>(address >> 32) & 0xfff;
                v[2] = stride != 0 ? Rd<u32>(buffer, 0x180) / stride : 0;
                v[3] = (format >> 12 & 0xfff) | extra | (format & 0x700) << 4 | 0x20000000;
            }
            const u64 address = reinterpret_cast<u64>(table);
            sgpr[start] = static_cast<u32>(address);
            sgpr[start + 1] = static_cast<u32>(address >> 32);
            sgpr_set[start] = sgpr_set[start + 1] = false;
            pending[pending_count++] = {sgpr_base + start, type, table, elements * 4, 4, {~0ull, ~0ull}};
            break;
        }
        case 27: // pointer to the extended user data
            have_eud_ptr = true;
            eud_ptr_sgpr = start;
            break;
        case 19:   // pointer to the resource table (32 bytes per slot)
        case 25:   // pointer to the RW resource table (compute, 32 bytes per slot)
        case 21:   // pointer to the sampler table (16 bytes per slot)
        case 22: { // pointer to the constant buffer table (16 bytes per slot)
            if (pending_count >= pending.size()) {
                ++record.unsupported_inputs;
                break;
            }
            u32 slots = 0;
            for (u32 s = 0; s < (type == 19 ? 128u : type == 25 ? 8u : 16u); ++s) {
                const u64 obj = type == 19 ? Obj(Rd<u64>(ctx, Ctx::ShaderResourceTable(stage) + s * 8))
                                : type == 25 ? Obj(Rd<u64>(ctx, 0x2098 + s * 8))
                                : type == 22 ? Obj(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + s * 8))
                                             : Obj(Rd<u64>(ctx, Ctx::SamplerTable(stage) + s * 8));
                if (obj) {
                    slots = s + 1;
                }
            }
            slots = std::max(slots, 1u);
            const u32 entry = type == 19 || type == 25 ? 8 : 4;
            u32* table = AllocTable(slots * entry);
            for (u32 s = 0; s < slots; ++s) {
                if (type == 19) {
                    ResourceDescriptor(ctx, stage, s, table + s * 8, 8);
                } else if (type == 25) {
                    UavDescriptor(ctx, s, table + s * 8, 8);
                } else if (type == 22) {
                    ConstBufferDescriptor(ctx, stage, s, table + s * 4);
                } else {
                    SamplerDescriptor(ctx, stage, s, table + s * 4);
                }
            }
            const u64 address = reinterpret_cast<u64>(table);
            if (start < 15) {
                sgpr[start] = static_cast<u32>(address);
                sgpr[start + 1] = static_cast<u32>(address >> 32);
                sgpr_set[start] = sgpr_set[start + 1] = false; // compared by content, not by value
                pending[pending_count++] = {sgpr_base + start, type, table, slots * entry, entry,
                                            EngineUsedSlots(shader_block, type)};
            }
            break;
        }
        default:
            ++record.unsupported_inputs;
            record.unsupported_mask |= 1u << (type & 31);
            break;
        }
    }
    if (have_eud_ptr && eud_ptr_sgpr < 15) {
        u32* table = AllocTable(std::max(eud_dwords, 1u));
        for (u32 i = 0; i < eud_dwords; ++i) {
            table[i] = eud[i];
        }
        pending[pending_count++] = {sgpr_base + eud_ptr_sgpr, 27, table, eud_dwords, 0, {}};
    }
    for (u32 r = 0; r < 16; ++r) {
        if (sgpr_set[r]) {
            out[n++] = {sgpr_base + r, sgpr[r]};
        }
    }
    for (u32 p = 0; p < pending_count && record.check_count < record.checks.size(); ++p) {
        record.checks[record.check_count++] = {pending[p].sgpr,
                                               pending[p].kind,
                                               pending[p].dwords,
                                               reinterpret_cast<u64>(pending[p].table),
                                               pending[p].entry,
                                               pending[p].mask};
    }
}

u32 Synthesize(u64 ctx, SynthReg* out, u64* vs_gnm_out, u64* ps_gnm_out) {
    const DeviceAddresses dev = ResolveDevice();
    if (dev.device == 0) {
        return 0;
    }
    u32 n = 0;
    const auto context_reg = [&](u32 offset, u32 value) { out[n++] = {ContextBase + offset, value}; };
    const auto uconfig_reg = [&](u32 offset, u32 value) { out[n++] = {UconfigBase + offset, value}; };
    const auto sh_reg = [&](u32 offset, u32 value) { out[n++] = {ShBase + offset, value}; };

    // blend state: CB_BLENDi_CONTROL, CB_BLEND_FACTORS..., CB_TARGET_MASK, blend constant
    {
        u64 blend = Rd<u64>(ctx, Ctx::BlendState);
        if (blend == 0) {
            blend = Rd<u64>(dev.device, 0x30280);
        }
        if (blend) {
            for (u32 i = 0; i < 8; ++i) {
                context_reg(0x1e0 + i, Rd<u32>(blend, 0x134 + i * 4) & 0xffff1fff);
            }
            context_reg(0x2dc, Rd<u32>(blend, 0x154));
            u32 mask = Rd<u32>(blend, 0x158);
            if (Rd<u64>(ctx, 0x130e0) != 0) {
                mask |= 0xf0000000u;
            }
            context_reg(0x8e, mask);
        }
        for (u32 i = 0; i < 4; ++i) {
            context_reg(0x105 + i, Rd<u32>(ctx, Ctx::BlendFactor + i * 4));
        }
    }
    // rasterizer state: PA_SU_SC_MODE_CNTL, PA_CL_CLIP_CNTL
    {
        u64 raster = Rd<u64>(ctx, Ctx::RasterizerState);
        if (raster == 0) {
            raster = Rd<u64>(dev.device, 0x30278);
        }
        // PrepareDraw writes the rasterizer registers only while the state is marked dirty, and PA_SU_SC_MODE_CNTL only when
        // its cached copy differs: a register file that holds something else (reset by the frame's default state) keeps it.
        if (raster && Rd<u8>(ctx, 0x46d8) != 0) {
            if (Rd<u32>(ctx, 0x46d0) != Rd<u32>(raster, 0x4c)) {
                context_reg(0x205, Rd<u32>(raster, 0x4c) & 0x393fff);
            }
            context_reg(0x204, Rd<u32>(raster, 0x50) & 0xf7fe03f);
        }
    }
    // depth-stencil state: DB_DEPTH_CONTROL, DB_STENCILREFMASK(_BF), DB_RENDER_CONTROL-like word
    {
        u64 ds = Rd<u64>(ctx, Ctx::DepthStencilState);
        if (ds == 0) {
            ds = Rd<u64>(dev.device, 0x30288);
        }
        if (ds) {
            u32 control = Rd<u32>(ds, 0x60);
            u32 stencil_write_mask = Rd<u8>(ds, 0x66);
            const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView);
            if (dsv) {
                const u32 flags = Rd<u32>(dsv, 0x48);
                if (flags & 1) {
                    const u32 index = Rd<u32>(ds, 0x38);
                    control = (control & 0xffffff8bu) | ((Rd<u32>(dev.depth_stencil_table, index * 4) & 7u) << 4);
                }
                if (flags & 2) {
                    stencil_write_mask = 0;
                }
            }
            context_reg(0x200, control & 0xc07007ffu);
            const u32 refmask = (((u32(Rd<u8>(ds, 0x65)) << 8) | (u32(Rd<u8>(ds, 0x67)) << 24)) +
                                 u32(Rd<u8>(ctx, Ctx::StencilRef))) |
                                (stencil_write_mask << 16);
            context_reg(0x10c, refmask);
            context_reg(0x10d, refmask);
            context_reg(0x10b, Rd<u32>(ds, 0x68) & 0xffffff);
        }
    }
    // primitive type (VGT_PRIMITIVE_TYPE)
    {
        const u32 topology = Rd<u32>(ctx, Ctx::Topology);
        const u32 value = topology < 10 ? (Rd<u32>(dev.topology_table, topology * 4) & 0x3f) : 9u;
        uconfig_reg(0x242, value);
    }
    // render targets: CB_COLORi_* (15 registers per target from 0x318; a bound target writes 14 words of the block at
    // view + 0x54, an unbound one clears CB_COLORi_INFO). Target 7 is the object at ctx + 0x130e0.
    {
        const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
        synth_rt_info = count & 0xff;
        for (u32 slot = 0; slot < 7; ++slot) {
            if (const u64 view = Rd<u64>(ctx, Ctx::RenderTargets + slot * 8); view != 0) {
                synth_rt_info |= 1u << (8 + slot);
                if (Rd<u32>(view + 0x54, 0) == 0) {
                    synth_rt_info |= 1u << (16 + slot);
                }
            }
        }
        const auto emit_target = [&](u32 slot, u64 view) {
            const u32 base = 0x318 + 0xf * slot;
            if (view) {
                const u64 block = view + 0x54;
                if (Rd<u32>(block, 0) == 0 && slot < 7) {
                    synth_flags |= OddRenderTarget;
                }
                if (Rd<u32>(block, 0) != 0) {
                    for (u32 k = 0; k < 14; ++k) {
                        context_reg(base + k, Rd<u32>(block, k * 4));
                    }
                    synth_cb_extent[slot] = Rd<u32>(block, 15 * 4);
                    synth_cb_extent_mask |= 1u << slot;
                    return;
                }
            }
            context_reg(base + 4, 0);
        };
        for (u32 slot = 0; slot < 7; ++slot) {
            const u64 view = slot < count ? Rd<u64>(ctx, Ctx::RenderTargets + slot * 8) : 0;
            emit_target(slot, view);
        }
        emit_target(7, Rd<u64>(ctx, 0x130e0));
    }
    // depth target: DB_Z_INFO.. DB_DEPTH_SLICE and friends from the block at view + 0x58; clear values from the resource
    {
        const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView);
        synth_db_extent = dsv ? Rd<u32>(dsv + 0x58, 12 * 4) : 0;
        bool valid = false;
        if (dsv) {
            const u64 block = dsv + 0x58;
            valid = Rd<u32>(block, 8) != 0 || Rd<u32>(block, 12) != 0;
            if (valid) {
                for (u32 k = 0; k < 8; ++k) {
                    context_reg(0x10 + k, Rd<u32>(block, k * 4));
                }
                context_reg(0xf, Rd<u32>(block, 11 * 4));
                context_reg(0x2, Rd<u32>(block, 8 * 4));
                context_reg(0x5, Rd<u32>(block, 9 * 4));
                context_reg(0x2af, Rd<u32>(block, 10 * 4));
            }
            if (const u64 resource = Rd<u64>(dsv, 0x30); resource && Rd<u64>(resource, 0x48) != 0) {
                context_reg(0xb, Rd<u32>(resource, 0x158));
                context_reg(0xa, Rd<u8>(resource, 0x15c));
                context_reg(0x2b0, Rd<u32>(resource, 0x178));
                context_reg(0x2b1, Rd<u32>(resource, 0x17c));
            }
        }
        if (!valid) {
            context_reg(0x10, 0);
            context_reg(0x11, 0);
        }
    }
    // viewport (Gnmx setupScreenViewport): scale / offset, z range, screen scissor and guard band
    {
        const float x = Rd<float>(ctx, Ctx::Viewport + 0);
        const float y = Rd<float>(ctx, Ctx::Viewport + 4);
        const float w = Rd<float>(ctx, Ctx::Viewport + 8);
        const float h = Rd<float>(ctx, Ctx::Viewport + 12);
        const float z_near = Rd<float>(ctx, Ctx::Viewport + 16);
        const float z_far = Rd<float>(ctx, Ctx::Viewport + 20);
        const auto to_u32 = [](float f) { return static_cast<u32>(static_cast<s64>(f)); };
        const u32 left = to_u32(x);
        const u32 top = to_u32(y);
        const u32 right = to_u32(x + w);
        const u32 bottom = to_u32(y + h);
        const s32 height = static_cast<s32>(bottom) - static_cast<s32>(top);
        const float x_scale = static_cast<float>(static_cast<s32>(right - left)) * 0.5f;
        const float height_f = static_cast<float>(height);
        const float y_scale = height_f * -0.5f;
        const float x_offset = static_cast<float>(left) + x_scale;
        const float y_offset = static_cast<float>(top) + height_f * 0.5f;
        const auto bits = [](float f) { return std::bit_cast<u32>(f); };
        context_reg(0xb4, 0);
        context_reg(0xb5, 0x3f800000);
        context_reg(0x10f, bits(x_scale));
        context_reg(0x110, bits(x_offset));
        context_reg(0x111, bits(y_scale));
        context_reg(0x112, bits(y_offset));
        context_reg(0x113, bits(z_far - z_near));
        context_reg(0x114, bits(z_near));
        context_reg(0x206, 0x43f);
        context_reg(0xc, (top << 16) | (left & 0xffff));
        context_reg(0xd, (static_cast<u32>(bottom) << 16) | (right & 0xffff));
        const auto grid = [](float offset) {
            const s32 v = static_cast<s32>(std::floor(offset * 0.0625f + 0.5f));
            return v > 0x1fb ? 0x1fcu : (static_cast<u32>(v) & 0xfffffffcu);
        };
        const u32 qx = grid(x_offset);
        const u32 qy = grid(y_offset);
        context_reg(0x8d, ((qy & 0x1fc) << 16) | (qx & 0x1fc));
        const float limit = std::bit_cast<float>(0x46fffffeu);
        const auto guard = [&](float offset, float abs_scale, u32 q) {
            const float q16 = static_cast<float>(static_cast<s32>(q << 4));
            const float t0 = (offset - abs_scale) - q16;
            const float t1 = (limit - abs_scale) - offset;
            const float a = 32768.0f + t0;
            const float b = q16 + t1;
            return (a < b ? a : b) / abs_scale;
        };
        context_reg(0x2fa, bits(guard(y_offset, std::fabs(y_scale), qy)));
        context_reg(0x2fb, 0x3f800000);
        context_reg(0x2fc, bits(guard(x_offset, std::fabs(x_scale), qx)));
        context_reg(0x2fd, 0x3f800000);
    }
    // vertex stage (sceGnmSetVsShader): program address, resource words, outputs, and the fetch shader address in the
    // first two user data registers; the fetch shader object is cached next to the vertex shader's stage block
    {
        const u64 vs = Rd<u64>(ctx, Ctx::VertexShader);
        const u64 gnm = vs ? Rd<u64>(vs, 0x120) : 0;
        *vs_gnm_out = gnm;
        synth_vs_gnm = gnm;
        if (!gnm) {
            synth_flags |= NoVertexShader;
        }
        if (gnm) {
            const u64 regs = gnm + 8;
            // The fetch shader object belongs to the input layout; without a layout the game binds the shader with
            // modifier 0 and no fetch address.
            const bool has_layout = Rd<u64>(ctx, Ctx::InputLayout) != 0;
            const u64 fetch_obj = has_layout ? Rd<u64>(vs + 0x30, 0xe0) : 0;
            if (Rd<u64>(vs, 0x420) != 0 || Rd<u64>(ctx, 0x4680) != 0 || Rd<u64>(ctx, 0x4688) != 0 ||
                Rd<u32>(ctx, 0x130c8) == 0x200) {
                synth_flags |= SpecialPipeline;
                synth_reason |= (Rd<u64>(vs, 0x420) != 0 ? ReasonVsVariant : 0u) | (Rd<u64>(ctx, 0x4680) != 0 ? ReasonCtx4680 : 0u) |
                                (Rd<u64>(ctx, 0x4688) != 0 ? ReasonCtx4688 : 0u) | (Rd<u32>(ctx, 0x130c8) == 0x200 ? ReasonDispatchDraw : 0u);
            }
            dbg_vs = vs;
            dbg_fetch = fetch_obj;
            synth_fetch_address = fetch_obj ? Rd<u64>(fetch_obj, 0x30) : 0;
            const u32 modifier = fetch_obj ? Rd<u32>(fetch_obj, 0x40) : 0;
            const u64 fetch_addr = fetch_obj ? Rd<u64>(fetch_obj, 0x30) : 0;
            const u32 r2 = Rd<u32>(regs, 8);
            bool takes_fetch_shader = false;
            for (u32 k = 0; k < Rd<u8>(gnm, 3) && k < 40; ++k) {
                takes_fetch_shader |= Rd<u8>(gnm + 0x28 + k * 4, 0) == 18;
            }
            // Shaders that take the fetch shader pointer cannot run natively until that object exists (it is created by
            // their first draw).
            if (takes_fetch_shader && fetch_obj == 0) {
                synth_flags |= NoFetchObject;
                synth_reason |= ReasonNoFetchObject;
            }
            // PrepareDraw binds the vertex shader only when (shader, modifier, fetch address) differ from the cached
            // ones, and a bind merges the modifier into the first resource word when it is not zero. A register file
            // that holds something else (a reset by the frame's default state) keeps it.
            const bool bind = Rd<u64>(ctx, 0x4440) != gnm || Rd<u32>(ctx, 0x4448) != modifier ||
                              Rd<u64>(ctx, 0x4450) != fetch_addr;
            if (bind) {
                sh_reg(0x48, Rd<u32>(regs, 0));
                sh_reg(0x49, 0);
                sh_reg(0x4a, modifier != 0 ? (r2 & 0xfcfffc3fu) | modifier : r2);
                sh_reg(0x4b, Rd<u32>(regs, 12));
                context_reg(0x207, Rd<u32>(regs, 24));
                context_reg(0x1b1, Rd<u32>(regs, 16));
                context_reg(0x1c3, Rd<u32>(regs, 20));
            }
            if (fetch_obj && (Rd<u32>(ctx, 0x4710) != Rd<u32>(fetch_obj, 0x48) || Rd<u32>(ctx, 0x4714) != Rd<u32>(fetch_obj, 0x4c))) {
                context_reg(0x2a8, Rd<u32>(fetch_obj, 0x48));
                context_reg(0x2a9, Rd<u32>(fetch_obj, 0x4c));
            }
        }
    }
    // pixel stage (sceGnmSetPsShader350)
    {
        const u64 ps = Rd<u64>(ctx, Ctx::PixelShader);
        const u64 gnm = ps ? Rd<u64>(ps, 0x120) : 0;
        *ps_gnm_out = gnm;
        if (gnm) {
            const u64 regs = gnm + 8;
            sh_reg(8, Rd<u32>(regs, 0));
            sh_reg(9, 0);
            sh_reg(10, Rd<u32>(regs, 8));
            sh_reg(11, Rd<u32>(regs, 12));
            context_reg(0x1c4, Rd<u32>(regs, 16));
            context_reg(0x1c5, Rd<u32>(regs, 20));
            context_reg(0x1b3, Rd<u32>(regs, 24));
            context_reg(0x1b4, Rd<u32>(regs, 28));
            context_reg(0x1b6, Rd<u32>(regs, 32));
            context_reg(0x1b8, Rd<u32>(regs, 36));
            context_reg(0x203, Rd<u32>(regs, 40));
            context_reg(0x8f, Rd<u32>(regs, 44));
            // SPI_PS_INPUT_CNTL_n: the interpolant table that follows the shader's input usage table
            const u64 interpolants = gnm + 0x3c + u64(Rd<u8>(gnm, 3)) * 4;
            // each entry is matched with the vertex shader's output semantics (same rule as the constant update engine)
            const u64 vertex = synth_vs_gnm;
            const u64 vertex_semantics =
                vertex + u64(Rd<u8>(vertex, 3)) * 4 + 0x28 + u64(Rd<u8>(vertex, 0x24)) * 4;
            const u32 vertex_semantic_count = Rd<u8>(vertex, 0x25);
            for (u32 k = 0; vertex != 0 && k < Rd<u8>(gnm, 0x38) && k < 32; ++k) {
                const u32 entry = Rd<u16>(interpolants, k * 2);
                bool found = false;
                u32 offset = 0;
                for (u32 j = 0; j < vertex_semantic_count && !found; ++j) {
                    if (Rd<u8>(vertex_semantics, j * 2) == (entry & 0xff)) {
                        found = true;
                        offset = Rd<u8>(vertex_semantics, j * 2 + 1) & 0x1f;
                    }
                }
                u32 value = 0x20;
                if (found) {
                    value = ((entry >> 7) & 0x20) | offset;
                    value |= (((entry >> 12) | (entry >> 10)) & 1) << 10;
                }
                context_reg(0x191 + k, ((entry >> 8) & 3) << 8 | value);
            }
        } else {
            sh_reg(8, 0);
            sh_reg(9, 0);
            context_reg(0x203, 0);
            context_reg(0x8f, 0xf);
        }
        const bool flag = (ps != 0 && Rd<u8>(ps, 0x107) != 0) || Rd<u8>(ctx, 0x130e8) != 0;
        context_reg(0x293, flag ? 0x6030000u : 0x6020000u);
    }
    return n;
}

// ---- native fixed-function state --------------------------------------------------------------------------------------
// The engine's blend, rasterizer and depth-stencil state objects keep the D3D11-style description they were created
// from (JsBlendState / JsRasterizerState / JsDepthStencilState, description at object + 0x30). The fixed-function state
// of a draw is built from those descriptions in Vulkan terms; nothing of it passes through register bits. States are
// interned by the inputs they were built from: a draw whose objects did not change gets the same pointer.
namespace Fixed {

constexpr std::array<vk::BlendFactor, 19> BlendFactors{
    vk::BlendFactor::eZero,
    vk::BlendFactor::eOne,
    vk::BlendFactor::eSrcColor,
    vk::BlendFactor::eOneMinusSrcColor,
    vk::BlendFactor::eSrcAlpha,
    vk::BlendFactor::eOneMinusSrcAlpha,
    vk::BlendFactor::eDstAlpha,
    vk::BlendFactor::eOneMinusDstAlpha,
    vk::BlendFactor::eDstColor,
    vk::BlendFactor::eOneMinusDstColor,
    vk::BlendFactor::eSrcAlphaSaturate,
    vk::BlendFactor::eConstantColor,
    vk::BlendFactor::eOneMinusConstantColor,
    vk::BlendFactor::eConstantAlpha,
    vk::BlendFactor::eOneMinusConstantAlpha,
    vk::BlendFactor::eSrc1Color,
    vk::BlendFactor::eOneMinusSrc1Color,
    vk::BlendFactor::eSrc1Alpha,
    vk::BlendFactor::eOneMinusSrc1Alpha,
};
constexpr std::array<vk::BlendOp, 5> BlendOps{vk::BlendOp::eAdd, vk::BlendOp::eSubtract, vk::BlendOp::eReverseSubtract,
                                              vk::BlendOp::eMin, vk::BlendOp::eMax};
constexpr std::array<vk::StencilOp, 8> StencilOps{
    vk::StencilOp::eKeep,
    vk::StencilOp::eZero,
    vk::StencilOp::eReplace,
    vk::StencilOp::eIncrementAndClamp,
    vk::StencilOp::eDecrementAndClamp,
    vk::StencilOp::eInvert,
    vk::StencilOp::eIncrementAndWrap,
    vk::StencilOp::eDecrementAndWrap,
};
/// The engine's comparison functions are 1 based (never .. always).
constexpr std::array<vk::CompareOp, 9> CompareOps{
    vk::CompareOp::eNever,        vk::CompareOp::eNever,   vk::CompareOp::eLess,
    vk::CompareOp::eEqual,        vk::CompareOp::eLessOrEqual, vk::CompareOp::eGreater,
    vk::CompareOp::eNotEqual,     vk::CompareOp::eGreaterOrEqual, vk::CompareOp::eAlways,
};

template <typename T, size_t N>
bool Pick(const std::array<T, N>& table, u32 index, T& out, u32 first = 0) {
    if (index < first || index >= N) {
        return false;
    }
    out = table[index];
    return true;
}

constexpr u32 KeyWords = 24;
struct Entry {
    std::array<u64, KeyWords> key{};
    std::unique_ptr<Vulkan::FixedState> state;
};
std::mutex mutex;
std::unordered_multimap<u64, std::unique_ptr<Entry>> states;

/// True when the fixed-function state is built from the objects (bit 15 of native_sdk_strip).
bool Enabled() {
    return (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 32768) != 0;
}

/// The fixed-function state of the draw the context describes, or null when an object is not what the builder knows.
/// `stale` receives the groups the draw's own state writes leave to the hardware: the engine writes the rasterizer
/// registers only while its state is dirty and PA_SU_SC_MODE_CNTL only when its cached copy differs.
const Vulkan::FixedState* Build(u64 ctx, const DeviceAddresses& dev, u32& stale) {
    stale = 0;
    u64 blend = Rd<u64>(ctx, Ctx::BlendState);
    u64 raster = Rd<u64>(ctx, Ctx::RasterizerState);
    u64 ds = Rd<u64>(ctx, Ctx::DepthStencilState);
    if (blend == 0) {
        blend = Rd<u64>(dev.device, 0x30280);
    }
    if (raster == 0) {
        raster = Rd<u64>(dev.device, 0x30278);
    }
    if (ds == 0) {
        ds = Rd<u64>(dev.device, 0x30288);
    }
    if (blend == 0 || raster == 0 || ds == 0 || dev.topology_table == 0) {
        return nullptr;
    }
    if (Rd<u8>(ctx, 0x46d8) == 0) {
        stale = Vulkan::StaleSurface | Vulkan::StaleClip;
    } else if (Rd<u32>(ctx, 0x46d0) == Rd<u32>(raster, 0x4c)) {
        stale = Vulkan::StaleSurface;
    }
    const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView);
    const u32 dsv_flags = dsv ? Rd<u32>(dsv, 0x48) : 0;
    const u64 dsv_block = dsv ? dsv + 0x58 : 0;
    const u32 topology = Rd<u32>(ctx, Ctx::Topology);
    const u64 ps = Rd<u64>(ctx, Ctx::PixelShader);
    const u64 ps_gnm = ps ? Rd<u64>(ps, 0x120) : 0;
    const u32 ps_mask = ps_gnm ? Rd<u32>(ps_gnm + 8, 44) : 0xfu;
    const bool target7 = Rd<u64>(ctx, 0x130e0) != 0;
    const u32 stencil_ref = Rd<u8>(ctx, Ctx::StencilRef);

    std::array<u64, KeyWords> key{};
    key[0] = blend;
    key[1] = Rd<u64>(blend, 0x10);
    key[2] = raster;
    key[3] = Rd<u64>(raster, 0x10);
    key[4] = ds;
    key[5] = Rd<u64>(ds, 0x10);
    key[6] = stencil_ref | u64(topology) << 8 | u64(target7) << 40 | u64(dsv != 0) << 41;
    key[7] = dsv_flags | u64(ps_mask) << 32;
    key[8] = dsv_block ? Rd<u64>(dsv_block, 0) : 0;
    key[9] = dsv_block ? Rd<u64>(dsv_block, 8) : 0;
    key[10] = dsv_block ? Rd<u32>(dsv_block, 8 * 4) | u64(Rd<u32>(dsv_block, 12 * 4)) << 32 : 0;
    for (u32 i = 0; i < 4; ++i) {
        key[11 + i] = Rd<u32>(ctx, Ctx::BlendFactor + i * 4);
    }
    for (u32 i = 0; i < 6; ++i) {
        key[15 + i] = Rd<u32>(ctx, Ctx::Viewport + i * 4);
    }
    u64 hash = 0xcbf29ce484222325ull;
    for (const u64 word : key) {
        hash = (hash ^ word) * 0x100000001b3ull;
    }
    std::lock_guard lock{mutex};
    for (auto [it, end] = states.equal_range(hash); it != end; ++it) {
        if (it->second->key == key) {
            return it->second->state.get();
        }
    }
    if (states.size() >= 16384) {
        return nullptr;
    }

    Vulkan::FixedState s = Vulkan::DefaultFixedState();
    // what the driver's default state block sets once and the engine never changes: colour writes on, no logic op,
    // lines one pixel wide
    s.color_enabled = true;
    s.logic_op_enabled = false;
    s.logic_op = vk::LogicOp::eCopy;
    s.line_width = 1.0f;
    // blend: D3D11_BLEND_DESC (alpha to coverage byte at +0x30, independent blend byte at +0x31, then a render target
    // description of 32 bytes per target from +0x34)
    const bool independent = Rd<u8>(blend, 0x31) != 0;
    for (u32 i = 0; i < AmdGpu::NUM_COLOR_BUFFERS; ++i) {
        const u64 rt = blend + 0x34 + u64(independent ? i : 0) * 0x20;
        auto& out = s.blend[i];
        out = {};
        if (Rd<u8>(rt, 0) != 0) {
            out.enable = 1;
            if (!Pick(BlendFactors, Rd<u32>(rt, 4), out.src_color) || !Pick(BlendFactors, Rd<u32>(rt, 8), out.dst_color) ||
                !Pick(BlendOps, Rd<u32>(rt, 12), out.color_op) || !Pick(BlendFactors, Rd<u32>(rt, 16), out.src_alpha) ||
                !Pick(BlendFactors, Rd<u32>(rt, 20), out.dst_alpha) || !Pick(BlendOps, Rd<u32>(rt, 24), out.alpha_op)) {
                return nullptr;
            }
        }
        s.target_write_mask[i] = target7 && i == 7 ? 0xf : static_cast<u8>(Rd<u8>(rt, 0x1c) & 0xf);
        if ((ps_mask >> (i * 4) & 8) == 0) {
            s.alpha_not_exported |= 1u << i;
        } else {
            s.alpha_not_exported &= ~(1u << i);
        }
    }
    for (u32 i = 0; i < 4; ++i) {
        s.blend_constants[i] = Rd<float>(ctx, Ctx::BlendFactor + i * 4);
    }
    // rasterizer: D3D11_RASTERIZER_DESC (fill, cull, front counter clockwise, bias values, depth clip)
    {
        const u32 fill = Rd<u32>(raster, 0x30);
        const u32 cull = Rd<u32>(raster, 0x34);
        const bool counter_clockwise = Rd<u8>(raster, 0x38) != 0;
        s.polygon_mode = fill == 2 ? vk::PolygonMode::eLine : vk::PolygonMode::eFill;
        s.front_face = counter_clockwise ? vk::FrontFace::eCounterClockwise : vk::FrontFace::eClockwise;
        s.depth_clip = Rd<u8>(raster, 0x48) != 0;
        const vk::CullModeFlags cull_mode = cull == 2   ? vk::CullModeFlagBits::eFront
                                            : cull == 3 ? vk::CullModeFlagBits::eBack
                                                        : vk::CullModeFlagBits::eNone;
        const u32 prim_code = topology < 10 ? (Rd<u32>(dev.topology_table, topology * 4) & 0x3f) : 9u;
        const auto prim = static_cast<AmdGpu::PrimitiveType>(prim_code);
        if (prim == AmdGpu::PrimitiveType::None) {
            return nullptr;
        }
        Vulkan::SetFixedStateTopology(s, prim, cull_mode);
        s.clip_disabled = prim == AmdGpu::PrimitiveType::RectList;
        s.user_clip_plane_mask = 0;
        s.degamma = false;
        s.viewport_empty = false;
    }
    // depth-stencil: D3D11_DEPTH_STENCIL_DESC as dwords (depth enable, write mask, function; stencil enable / read mask /
    // write mask packed in dword 3; front operations and function at 4..7, back at 8..11)
    {
        const u64 d = ds + 0x30;
        s.depth_test = Rd<u32>(d, 0) != 0;
        s.depth_write = (Rd<u32>(d, 4) & 1) != 0;
        if ((dsv_flags & 1) != 0) {
            // a depth view bound for read-only use writes no depth and replaces the comparison with the one the engine
            // keeps for it
            s.depth_write = false;
            const u32 index = Rd<u32>(d, 8);
            if (dev.depth_stencil_table == 0) {
                return nullptr;
            }
            s.depth_compare = Vulkan::LiverpoolToVK::CompareOp(
                static_cast<AmdGpu::CompareFunc>(Rd<u32>(dev.depth_stencil_table, index * 4) & 7));
        } else if (!Pick(CompareOps, Rd<u32>(d, 8), s.depth_compare, 1)) {
            return nullptr;
        }
        const u32 stencil = Rd<u32>(d, 12);
        s.stencil_test = (stencil & 1) != 0;
        auto& front = s.stencil_front_ops;
        auto& back = s.stencil_back_ops;
        if (!Pick(StencilOps, Rd<u32>(d, 16), front.fail_op) || !Pick(StencilOps, Rd<u32>(d, 20), front.depth_fail_op) ||
            !Pick(StencilOps, Rd<u32>(d, 24), front.pass_op) || !Pick(CompareOps, Rd<u32>(d, 28), front.compare_op, 1) ||
            !Pick(StencilOps, Rd<u32>(d, 32), back.fail_op) || !Pick(StencilOps, Rd<u32>(d, 36), back.depth_fail_op) ||
            !Pick(StencilOps, Rd<u32>(d, 40), back.pass_op) || !Pick(CompareOps, Rd<u32>(d, 44), back.compare_op, 1)) {
            return nullptr;
        }
        s.stencil_reference = {stencil_ref, stencil_ref};
        const u32 write_mask = (dsv_flags & 2) != 0 ? 0u : ((stencil >> 16) & 0xff);
        s.stencil_write_mask = {write_mask, write_mask};
        s.stencil_compare_mask = {(stencil >> 8) & 0xff, (stencil >> 8) & 0xff};
        // the targets of the depth view: format bits of its depth / stencil info words and a non-zero base address
        const bool block = dsv_block != 0 && (Rd<u32>(dsv_block, 8 * 4) != 0 || Rd<u32>(dsv_block, 12 * 4) != 0);
        s.depth_target = block && Rd<u32>(dsv_block, 2 * 4) != 0 && (Rd<u32>(dsv_block, 0) & 3) != 0;
        s.stencil_target = block && Rd<u32>(dsv_block, 3 * 4) != 0 && (Rd<u32>(dsv_block, 4) & 1) != 0;
    }
    // viewport (the context's rectangle and depth range) and the scissor the pipeline derives from it
    {
        const float x = Rd<float>(ctx, Ctx::Viewport + 0);
        const float y = Rd<float>(ctx, Ctx::Viewport + 4);
        const float w = Rd<float>(ctx, Ctx::Viewport + 8);
        const float h = Rd<float>(ctx, Ctx::Viewport + 12);
        const float z_near = Rd<float>(ctx, Ctx::Viewport + 16);
        const float z_far = Rd<float>(ctx, Ctx::Viewport + 20);
        const auto to_u32 = [](float f) { return static_cast<u32>(static_cast<s64>(f)); };
        const u32 left = to_u32(x);
        const u32 top = to_u32(y);
        const u32 right = to_u32(x + w);
        const u32 bottom = to_u32(y + h);
        const float height = static_cast<float>(static_cast<s32>(bottom) - static_cast<s32>(top));
        s.viewport_count = 1;
        const float x_scale = static_cast<float>(static_cast<s32>(right - left)) * 0.5f;
        s.transform = {
            .x_offset = static_cast<float>(left) + x_scale,
            .x_scale = x_scale,
            .y_offset = static_cast<float>(top) + height * 0.5f,
            .y_scale = height * -0.5f,
        };
        s.viewports[0] = vk::Viewport{
            .x = static_cast<float>(left),
            .y = static_cast<float>(top) + height,
            .width = static_cast<float>(static_cast<s32>(right - left)),
            .height = -height,
            .minDepth = z_near,
            .maxDepth = z_near + (z_far - z_near),
        };
        AmdGpu::Scissor scissor{};
        scissor.top_left_x = static_cast<s16>(left);
        scissor.top_left_y = static_cast<s16>(top);
        scissor.bottom_right_x = static_cast<s16>(right);
        scissor.bottom_right_y = static_cast<s16>(bottom);
        s.scissors[0] = vk::Rect2D{.offset = {scissor.top_left_x, scissor.top_left_y},
                                   .extent = {scissor.GetWidth(), scissor.GetHeight()}};
    }

    auto entry = std::make_unique<Entry>();
    entry->key = key;
    entry->state = std::make_unique<Vulkan::FixedState>(s);
    const Vulkan::FixedState* result = entry->state.get();
    states.emplace(hash, std::move(entry));
    return result;
}

} // namespace Fixed

NativeRecord& Slot(u64 seq) {
    std::call_once(records_once, [] { records = std::make_unique<NativeRecord[]>(RecordSlots); });
    return records[seq % RecordSlots];
}

/// Guest thread: builds the record of a draw. Returns its number.
u64 MakeRecord(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3, u64 caller) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    synth_flags = 0;
    synth_reason = 0;
    synth_cb_extent_mask = 0;
    slot.count = Synthesize(ctx, slot.regs.data(), &slot.vs_gnm, &slot.ps_gnm);
    slot.fixed = nullptr;
    if (kind <= 2 && Fixed::Enabled()) {
        slot.fixed = Fixed::Build(ctx, ResolveDevice(), slot.fixed_stale);
    }
    slot.cb_extent = synth_cb_extent;
    slot.cb_extent_mask = synth_cb_extent_mask;
    slot.db_extent = synth_db_extent;
    slot.flags = synth_flags | (table_overflow || bad_object_seen ? SpecialPipeline : 0u);
    slot.rt_info = synth_rt_info;
    slot.kind = kind;
    slot.caller = caller;
    slot.a1 = static_cast<u32>(a1);
    slot.a2 = static_cast<u32>(a2);
    slot.dbg_vs = dbg_vs;
    slot.dbg_fetch = dbg_fetch;
    slot.dbg_layout = Rd<u64>(ctx, 0x68);
    {
        u32 n = slot.count;
        const u64 vertex_shader = Rd<u64>(ctx, Ctx::VertexShader);
        const u64 pixel_shader = Rd<u64>(ctx, Ctx::PixelShader);
        SynthesizeUserData(ctx, 0, vertex_shader + 0x30, slot.vs_gnm, 0x28, ShBase + 0x4c, slot.regs.data(), n, slot);
        SynthesizeUserData(ctx, 1, pixel_shader + 0x30, slot.ps_gnm, 0x3c, ShBase + 0x0c, slot.regs.data(), n, slot);
        // draw arguments: VGT_NUM_INDICES / VGT_NUM_INSTANCES, then for indexed draws the index buffer view
        // (IT_INDEX_BASE), the base vertex and the draw packet's maximum size
        if (kind != 2) {
            // an inlined draw (the count is only known to the draw packet) keeps the instance count the game set
            const bool unknown_count = a1 == Core::NativeDraw::UnknownDrawCount;
            if (!unknown_count) {
                slot.regs[n++] = {UconfigBase + 0x24c, static_cast<u32>(a1)};
            }
            slot.regs[n++] = {UconfigBase + 0x24d, inline_instances != 0 ? inline_instances : 1u}; // DrawIndexedCommon and DrawAuto reset the instance count
        }
        if (kind == 2) {
            // indirect indexed draw: the argument buffer, the index buffer view and the registers that receive the
            // base vertex / start instance (nibbles of a byte in the vertex shader)
            // ctx + 0x4acc (the stage configuration) is what PrepareDraw sets next: 0 for a vertex / pixel pipeline,
            // which is the case when no tessellation or geometry stage is bound
            if (const u64 view = Rd<u64>(ctx, 0x58); view != 0 && a1 != 0 && slot.vs_gnm != 0 && Rd<u64>(ctx, 0x4680) == 0 &&
                                                       Rd<u64>(ctx, 0x4688) == 0) {
                slot.regs[n++] = {ContextBase + 0x1fa, Rd<u32>(view, 0x60) & 0xfffffffeu};
                slot.regs[n++] = {ContextBase + 0x1f9, Rd<u32>(view, 0x64) & 0xff};
                slot.regs[n++] = {ContextBase + 0x102, Rd<u32>(ctx, 0x64)};
                slot.regs[n++] = {ContextBase + 0x29e, Rd<u32>(view, 0x68)};
                const u64 args = Rd<u64>(a1, 0x30);
                slot.args_addr = (args & 0xfffffff8ull) | ((args >> 32) & 0xff) << 32;
                const u32 locations = Rd<u8>(slot.vs_gnm, 0x27);
                constexpr u32 VertexStageSgprBase = 0x4c;
                slot.vertex_sgpr = (locations & 0xf) == 0 ? 0 : (locations & 0xf) + VertexStageSgprBase;
                slot.instance_sgpr = (locations >> 4) == 0 ? 0 : (locations >> 4) + VertexStageSgprBase;
                if (Rd<u32>(ctx, 0x60) != 0x39) {
                    slot.flags |= SpecialPipeline;
                    synth_reason |= ReasonIndexType;
                }
            } else {
                slot.flags |= SpecialPipeline;
                synth_reason |= ReasonIndirectView | (Rd<u64>(ctx, 0x58) == 0 ? 0x1000u : 0u) | (a1 == 0 ? 0x2000u : 0u) |
                                (slot.vs_gnm == 0 ? 0x4000u : 0u) | (Rd<u64>(ctx, 0x4680) != 0 || Rd<u64>(ctx, 0x4688) != 0 ? 0x8000u : 0u);
            }
        }
        if (kind == 0) {
            if (const u64 view = Rd<u64>(ctx, 0x58); view != 0) {
                slot.regs[n++] = {ContextBase + 0x1fa, Rd<u32>(view, 0x60) & 0xfffffffeu};
                slot.regs[n++] = {ContextBase + 0x1f9, Rd<u32>(view, 0x64) & 0xff};
                slot.regs[n++] = {ContextBase + 0x102, Rd<u32>(ctx, 0x64)};
                if (Rd<u32>(ctx, 0x60) != 0x39) {
                    slot.flags |= SpecialPipeline;
                    synth_reason |= ReasonIndexType;
                }
            }
            slot.regs[n++] = {ContextBase + 0x29e, static_cast<u32>(a1)};
        }
        if (slot.unsupported_inputs != 0) {
            slot.flags |= SpecialPipeline;
            synth_reason |= ReasonUnsupportedInput;
        }
        slot.count = n;
        slot.reason = synth_reason | (table_overflow ? ReasonTableOverflow : 0u) | (bad_object_seen ? ReasonBadObject : 0u);
    }
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

/// Guest thread: the record of a dispatch from the device context's compute stage (the shader at ctx + 0x4670,
/// resource tables of API stage 5, unordered access views at ctx + 0x2098): the user data registers and the
/// program registers the compute constant update engine writes, and the dispatch arguments.
u64 MakeComputeRecord(u64 ctx, u64 x, u64 y, u64 z, u64 caller) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    synth_flags = 0;
    synth_fetch_address = 0;
    slot.kind = 3;
    slot.caller = caller;
    slot.a1 = static_cast<u32>(x);
    slot.a2 = static_cast<u32>(y);
    slot.rt_info = 0;
    slot.dbg_vs = slot.dbg_fetch = slot.dbg_layout = 0;
    slot.ps_gnm = 0;
    const u64 shader = Rd<u64>(ctx, 0x4670);
    const u64 gnm = shader ? Rd<u64>(shader, 0x120) : 0;
    slot.vs_gnm = gnm;
    u32 n = 0;
    if (gnm == 0) {
        synth_flags |= NoVertexShader;
    } else {
        const u64 regs = gnm + 8;
        const auto cs_reg = [&](u32 dword, u32 value) { slot.regs[n++] = {Core::NativeDraw::ComputeRegisterBase + dword, value}; };
        cs_reg(0x00, (u32(Rd<u8>(ctx, 0x13015)) * 8 & 0x18) + 1); // dispatch initiator
        cs_reg(0x01, static_cast<u32>(x));
        cs_reg(0x02, static_cast<u32>(y));
        cs_reg(0x03, static_cast<u32>(z));
        cs_reg(0x07, Rd<u32>(regs, 16)); // COMPUTE_NUM_THREAD_X / Y / Z
        cs_reg(0x08, Rd<u32>(regs, 20));
        cs_reg(0x09, Rd<u32>(regs, 24));
        cs_reg(0x0c, Rd<u32>(regs, 0)); // COMPUTE_PGM_LO / HI: the shader bind
        cs_reg(0x0d, 0);
        cs_reg(0x12, Rd<u32>(regs, 8)); // COMPUTE_PGM_RSRC1 / RSRC2
        cs_reg(0x13, Rd<u32>(regs, 12));
        SynthesizeUserData(ctx, 5, shader + 0x30, gnm, 0x28, Core::NativeDraw::ComputeRegisterBase + 0x40, slot.regs.data(),
                           n, slot);
    }
    slot.flags = (table_overflow || bad_object_seen ? SpecialPipeline : 0u) | synth_flags | (slot.unsupported_inputs != 0 ? SpecialPipeline : 0u);
    slot.count = n;
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

/// Guest thread: the record of a dispatch of the game's async compute context (JsComputeContext, the SDK's lightweight
/// compute constant update engine at +0xa0). The engine objects (shader at +0x1d10, samplers +0x2178, constant buffers
/// +0x21f8, shader resource views +0x1d18, unordered access views +0x2118) are described to the shader through user
/// data: small descriptors go straight into the user data registers, the others into a table (the engine's staging
/// copy at +0x138 holds its content) that a register points to. Returns 0 when this dispatch needs the game's own path.
thread_local u32 asc_fail; ///< source line of the check that made MakeAscRecord give up (diagnostics)

/// Append / consume counters the dispatch initializes before it runs: the game writes the value into the GDS counter with a
/// DMA packet; with execution a GDS fill record does.
struct CounterInit {
    u32 index;
    u32 value;
};
thread_local std::array<CounterInit, 8> asc_counter_inits;
thread_local u32 asc_counter_init_count;

u64 MakeAscRecord(u64 jcc, u64 x, u64 y, u64 z, u64 caller) {
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    asc_fail = 0;
    asc_counter_init_count = 0;
    const u64 cs_obj = Rd<u64>(jcc, 0x1d10);
    const u64 gnm = valid(cs_obj) ? Rd<u64>(cs_obj, 0x120) : 0;
    if (!valid(cs_obj) || !valid(gnm) || Rd<u8>(jcc, 0x1c7c) == 0) {
        asc_fail = __LINE__;
        return 0;
    }
    const u64 offs = cs_obj + 0x130;
    const u32 table_dwords = Rd<u16>(offs, 0);
    if (table_dwords > 0x600) {
        asc_fail = __LINE__;
        return 0;
    }
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    slot.kind = 3;
    slot.caller = caller;
    slot.a1 = static_cast<u32>(x);
    slot.a2 = static_cast<u32>(y);
    slot.rt_info = 0;
    slot.dbg_vs = slot.dbg_fetch = slot.dbg_layout = 0;
    slot.ps_gnm = 0;
    slot.vs_gnm = gnm;
    u32 n = 0;
    const auto cs_reg = [&](u32 dword, u32 value) { slot.regs[n++] = {Core::NativeDraw::ComputeRegisterBase + dword, value}; };
    const u64 shader_regs = gnm + 8;
    cs_reg(0x00, ((Rd<u8>(gnm, 0x24) & 3u) << 3) + 1); // dispatch initiator
    cs_reg(0x01, static_cast<u32>(x));
    cs_reg(0x02, static_cast<u32>(y));
    cs_reg(0x03, static_cast<u32>(z));
    cs_reg(0x07, Rd<u32>(shader_regs, 16)); // COMPUTE_NUM_THREAD_X / Y / Z
    cs_reg(0x08, Rd<u32>(shader_regs, 20));
    cs_reg(0x09, Rd<u32>(shader_regs, 24));
    cs_reg(0x0c, Rd<u32>(shader_regs, 0)); // COMPUTE_PGM_LO / HI
    cs_reg(0x0d, 0);
    cs_reg(0x12, Rd<u32>(shader_regs, 8)); // COMPUTE_PGM_RSRC1 / RSRC2
    cs_reg(0x13, Rd<u32>(shader_regs, 12));
    // the descriptor table: the engine's staging copy with this dispatch's descriptors placed in it
    u32* table = AllocTable(std::max(table_dwords, 4u));
    std::memcpy(table, reinterpret_cast<const void*>(jcc + 0x138), table_dwords * sizeof(u32));
    bool ok = true;
    const auto place = [&](u16 offset, const u32* source, u32 dwords) {
        if (offset == 0xffff) {
            return;
        }
        const u32 index = offset & 0x3fff;
        if ((offset & 0x8000) != 0) {
            if (index + dwords > 16) { // only 16 user data registers exist
                ok = false; asc_fail = __LINE__;
                return;
            }
            for (u32 k = 0; k < dwords; ++k) {
                cs_reg(0x40 + index + k, source[k]);
            }
        } else if (index + dwords <= table_dwords) {
            std::memcpy(table + index, source, dwords * sizeof(u32));
        } else {
            ok = false; asc_fail = __LINE__;
        }
    };
    std::array<u32, 8> descriptor;
    const auto read_descriptor = [&](u64 address, u32 dwords) {
        for (u32 k = 0; k < dwords; ++k) {
            descriptor[k] = Rd<u32>(address, k * 4);
        }
    };
    for (u32 mask = Rd<u32>(cs_obj, 0xe4), index = 0; mask != 0 && ok; mask >>= 1, ++index) { // samplers
        if ((mask & 1) != 0) {
            const u64 sampler = Rd<u64>(jcc, 0x2178 + index * 8);
            if (!valid(sampler)) {
                ok = false; asc_fail = __LINE__;
                break;
            }
            read_descriptor(sampler + 0x64, 4);
            place(Rd<u16>(offs, 0x142 + index * 2), descriptor.data(), 4);
        }
    }
    for (u32 mask = Rd<u32>(cs_obj, 0xe0), index = 0; mask != 0 && ok; mask >>= 1, ++index) { // constant buffers
        if ((mask & 1) != 0) {
            const u64 buffer = Rd<u64>(jcc, 0x21f8 + index * 8);
            if (!valid(buffer)) {
                ok = false; asc_fail = __LINE__;
                break;
            }
            read_descriptor(buffer + 0x60, 4);
            place(Rd<u16>(offs, 0x162 + index * 2), descriptor.data(), 4);
        }
    }
    for (u32 mask = Rd<u32>(cs_obj, 0xf0), index = 0; mask != 0 && ok; mask >>= 1, ++index) { // shader resource views
        if ((mask & 1) != 0) {
            const u64 view = Rd<u64>(jcc, 0x1d18 + index * 8);
            if (!valid(view)) {
                ok = false; asc_fail = __LINE__;
                break;
            }
            const u32 kind = Rd<u32>(view, 0x44);
            if (kind == 0xb || kind == 1) { // buffer: the engine rewrites two fields of its V# first
                descriptor[0] = Rd<u32>(view, 0x78);
                descriptor[1] = (Rd<u32>(view, 0x7c) & 0xffff0fffu) | 0xb000u;
                descriptor[2] = Rd<u32>(view, 0x80);
                descriptor[3] = (Rd<u32>(view, 0x84) & 0xc7ffffffu) | 0x18000000u;
                place(Rd<u16>(offs, 0x22 + index * 2), descriptor.data(), 4);
            } else {
                read_descriptor(view + 0x58, 8);
                place(Rd<u16>(offs, 0x22 + index * 2), descriptor.data(), 8);
            }
        }
    }
    u32 counter_range = Rd<u32>(jcc, 0x1940);
    for (u32 mask = Rd<u32>(cs_obj, 0x100), index = 0; mask != 0 && ok; mask >>= 1, ++index) { // unordered access views
        if ((mask & 1) != 0) {
            const u64 view = Rd<u64>(jcc, 0x2118 + index * 8);
            if (!valid(view)) {
                ok = false; asc_fail = __LINE__;
                break;
            }
            if (Rd<u32>(view, 0x44) == 1) {
                read_descriptor(view + 0x74, 4);
                place(Rd<u16>(offs, 0x122 + index * 2), descriptor.data(), 4);
                if ((Rd<u8>(view, 0x50) & 2) != 0) {
                    if (const u32 init = Rd<u32>(jcc, 0x2158 + index * 4); init != 0xffffffffu) {
                        if (asc_counter_init_count >= asc_counter_inits.size()) {
                            ok = false; asc_fail = __LINE__;
                            break;
                        }
                        asc_counter_inits[asc_counter_init_count++] = {index, init};
                    }
                    if (Rd<u8>(offs, 0xd) != 0xff) {
                        counter_range = Rd<u32>(jcc, 0x22f8) << 16 | 0x10;
                    }
                }
            } else {
                read_descriptor(view + 0x54, 8);
                place(Rd<u16>(offs, 0x122 + index * 2), descriptor.data(), 8);
            }
        }
    }
    if (!ok || table_overflow) {
        if (asc_fail == 0) {
            asc_fail = __LINE__;
        }
        return 0;
    }
    // what the flush of the constant update engine puts into user data: pointers into the table and two immediates
    const u64 table_address = reinterpret_cast<u64>(table);
    u32 checks = 0;
    const auto pointer = [&](u32 start, u64 address, u32 dwords) {
        if (start == 0xff) {
            return;
        }
        if (start + 2 > 16 || checks >= slot.checks.size()) { // outside the 16 user data registers: not a layout we know
            ok = false; asc_fail = __LINE__;
            return;
        }
        slot.checks[checks] = {Core::NativeDraw::ComputeRegisterBase + 0x40 + start, 20 + checks, dwords, address, 0, {~0ull, ~0ull}};
        ++checks;
    };
    pointer(Rd<u8>(offs, 7), table_address, table_dwords);
    static constexpr std::array<u32, 4> table_offsets{0x16, 0x1a, 0x1c, 0x1e};
    for (u32 k = 0; k < 4; ++k) {
        const u32 offset = Rd<u16>(offs, table_offsets[k]);
        pointer(Rd<u8>(offs, 8 + k), table_address + offset * 4ull, offset < table_dwords ? table_dwords - offset : 0);
    }
    for (const u32 start : {Rd<u8>(offs, 0xc), Rd<u8>(offs, 0xd), Rd<u8>(offs, 0xe)}) {
        if (start != 0xff && start + 2 > 16) {
            ok = false; asc_fail = __LINE__;
        }
    }
    if (!ok) {
        asc_fail = __LINE__;
        return 0;
    }
    if (const u32 start = Rd<u8>(offs, 0xc); start != 0xff) {
        const u64 global_table = Rd<u64>(jcc, 0x118);
        cs_reg(0x40 + start, static_cast<u32>(global_table));
        cs_reg(0x40 + start + 1, static_cast<u32>(global_table >> 32));
    }
    if (const u32 start = Rd<u8>(offs, 0xd); start != 0xff) {
        cs_reg(0x40 + start, counter_range);
    }
    if (const u32 start = Rd<u8>(offs, 0xe); start != 0xff) {
        cs_reg(0x40 + start, Rd<u32>(jcc, 0x1944));
    }
    slot.check_count = checks;
    slot.flags = 0;
    slot.count = n;
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

/// Guest thread: the record of a CsCopyBufferDwords dispatch. The state is fully given by the arguments and the device's
/// copy shader: the source buffer is the shader resource of slot 0, the destination the RW resource of slot 0, both
/// described by V# words built from the addresses. No program packets are written by the game's path for this one
/// beyond the shader bind, so the record carries the program registers too.
u64 MakeCopyRecord(u64 ctx, u64 destination, u64 source, u64 bytes, u64 caller, bool async_context = false) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    synth_flags = 0;
    slot.kind = 3;
    slot.caller = caller;
    slot.a1 = (static_cast<u32>(bytes) + 3) >> 2;
    slot.a2 = 1;
    if (!async_context) {
        slot.copy_destination = destination;
        slot.copy_source = source;
        slot.copy_bytes = static_cast<u32>(bytes);
    }
    slot.rt_info = 0;
    slot.dbg_vs = slot.dbg_fetch = slot.dbg_layout = 0;
    slot.ps_gnm = 0;
    const VAddr image = image_base.load(std::memory_order_relaxed);
    const u64 device = image ? Rd<u64>(image + 0x6b03540 - 0x1000000, 0) : 0;
    const u64 gnm = device ? Rd<u64>(device, 0x120) : 0;
    slot.vs_gnm = gnm;
    u32 n = 0;
    if (gnm == 0) {
        synth_flags |= NoVertexShader;
    } else {
        const u64 regs = gnm + 8;
        const u32 words = (static_cast<u32>(bytes) + 3) >> 2;
        const auto cs_reg = [&](u32 dword, u32 value) {
            slot.regs[n++] = {Core::NativeDraw::ComputeRegisterBase + dword, value};
        };
        cs_reg(0x00, async_context ? ((Rd<u8>(gnm, 0x24) & 3u) << 3) + 1 : (u32(Rd<u8>(ctx, 0x13015)) * 8 & 0x18) + 1);
        cs_reg(0x01, (words + 0x3f) >> 6);
        cs_reg(0x02, 1);
        cs_reg(0x03, 1);
        cs_reg(0x07, Rd<u32>(regs, 16));
        cs_reg(0x08, Rd<u32>(regs, 20));
        cs_reg(0x09, Rd<u32>(regs, 24));
        cs_reg(0x0c, Rd<u32>(regs, 0));
        cs_reg(0x0d, 0);
        cs_reg(0x12, Rd<u32>(regs, 8));
        cs_reg(0x13, Rd<u32>(regs, 12));
        // the async context describes the source like the destination (read-write flags)
        const std::array<u32, 4> source_descriptor{static_cast<u32>(source),
                                                   static_cast<u32>(source >> 32) & 0xfff | (async_context ? 0x47000u : 0x40000u), words,
                                                   async_context ? 0x18024004u : 0x20024004u};
        const std::array<u32, 4> destination_descriptor{static_cast<u32>(destination),
                                                        static_cast<u32>(destination >> 32) & 0xfff | 0x47000, words,
                                                        0x18024004};
        std::array<u32, 16> sgpr{};
        for (u32 i = 0; i < Rd<u8>(gnm, 3) && i < 40; ++i) {
            const u64 e = gnm + 0x28 + i * 4;
            const u32 type = Rd<u8>(e, 0);
            const u32 slot_index = Rd<u8>(e, 1);
            const u32 start = Rd<u8>(e, 2);
            if (start > 12) {
                ++slot.unsupported_inputs;
            } else if ((type == 0 || type == 4) && slot_index == 0) {
                const auto& descriptor = type == 0 ? source_descriptor : destination_descriptor;
                for (u32 k = 0; k < 4; ++k) {
                    sgpr[start + k] = descriptor[k];
                }
                for (u32 k = 0; k < 4; ++k) {
                    cs_reg(0x40 + start + k, descriptor[k]);
                }
            } else if (type == 26) {
                const u64 table = Rd<u64>(ctx, async_context ? 0x118 : 0x12f30);
                cs_reg(0x40 + start, static_cast<u32>(table));
                cs_reg(0x40 + start + 1, static_cast<u32>(table >> 32));
            } else {
                ++slot.unsupported_inputs;
            }
        }
    }
    slot.flags = (table_overflow || bad_object_seen ? SpecialPipeline : 0u) | synth_flags | (slot.unsupported_inputs != 0 ? SpecialPipeline : 0u);
    slot.count = n;
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

/// Guest thread: the record of a JsDeviceContext::CopyResource of a two dimensional texture (a compute dispatch of the
/// device's copy shader for the destination's format: shader resource = the source subresource, constant buffer = the
/// source size at the subresource's mip, RW resource = the destination subresource). Returns 0 when the arguments are not
/// the plain case.
u64 MakeCopyResourceRecord(u64 ctx, u64 destination, u32 destination_sub, u64 source, u32 source_sub, u64 caller) {
    const VAddr image = image_base.load(std::memory_order_relaxed);
    if (image == 0) {
        return 0;
    }
    const auto texture_descriptor = [&](u64 resource, u32 subresource, bool writable, std::array<u32, 8>& t) {
        for (u32 k = 0; k < 8; ++k) {
            t[k] = Rd<u32>(resource, 0x70 + k * 4);
        }
        const u32 mips = t[3] > 0xdfffffffu ? 1u : ((t[3] >> 16) & 0xf) + 1;
        if (((t[3] >> 28) | 1) != 0xf) {
            const u32 mip = (subresource % mips) & 0xf;
            t[3] = (mip << 16) | (mip << 12) | (t[3] & 0xfff00fffu);
        }
        const u32 slice = subresource / mips;
        t[5] = ((slice & 0x1fff) << 13) | (slice & 0x1fff) | (t[5] & 0xfc000000u);
        if (writable) {
            t[3] &= ~(1u << 26);
            t[1] = (t[1] & 0x3fffff3fu) | 0xc0000040u;
        }
    };
    std::array<u32, 8> srv{}, uav{};
    texture_descriptor(destination, destination_sub, true, uav);
    texture_descriptor(source, source_sub, false, srv);
    const u32 source_mips = Rd<u32>(source, 0x188);
    if (source_mips == 0) {
        return 0;
    }
    const u32 mip_shift = (source_sub % source_mips) & 0x1f;
    const u32 width = Rd<u32>(source, 0x180) >> mip_shift;
    const u32 height = Rd<u32>(source, 0x184) >> mip_shift;
    const u32 format_bits = (uav[1] >> 20) & 0x3f;
    const u32 class_entry = format_bits < 0x3d ? Rd<u32>(image + 0x25e0520 - 0x1000000, format_bits * 4) : 0;
    const u64 shader_index = format_bits < 0x3d ? static_cast<u64>(static_cast<s64>(static_cast<s32>(class_entry) + 3)) : 3;
    if (shader_index > 12) {
        return 0;
    }
    const u64 device = Rd<u64>(image + 0x6b034e0 - 0x1000000, shader_index * 8);
    const u64 gnm = device ? Rd<u64>(device, 0x120) : 0;
    if (gnm == 0 || width == 0 || height == 0) {
        return 0;
    }
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    synth_flags = 0;
    slot.kind = 3;
    slot.caller = caller;
    slot.a1 = (width + 7) >> 3;
    slot.a2 = (height + 7) >> 3;
    slot.rt_info = 0;
    slot.dbg_vs = slot.dbg_fetch = slot.dbg_layout = 0;
    slot.ps_gnm = 0;
    slot.vs_gnm = gnm;
    u32 n = 0;
    const auto cs_reg = [&](u32 dword, u32 value) { slot.regs[n++] = {Core::NativeDraw::ComputeRegisterBase + dword, value}; };
    const u64 program = gnm + 8;
    cs_reg(0x00, (u32(Rd<u8>(ctx, 0x13015)) * 8 & 0x18) + 1);
    cs_reg(0x01, slot.a1);
    cs_reg(0x02, slot.a2);
    cs_reg(0x03, 1);
    cs_reg(0x07, Rd<u32>(program, 16));
    cs_reg(0x08, Rd<u32>(program, 20));
    cs_reg(0x09, Rd<u32>(program, 24));
    cs_reg(0x0c, Rd<u32>(program, 0));
    cs_reg(0x0d, 0);
    cs_reg(0x12, Rd<u32>(program, 8));
    cs_reg(0x13, Rd<u32>(program, 12));
    u32* constants = AllocTable(8);
    constants[0] = width;
    constants[4] = height;
    constants[1] = constants[2] = constants[3] = constants[5] = constants[6] = constants[7] = 0;
    const u64 constants_address = reinterpret_cast<u64>(constants);
    const std::array<u32, 2> constant_descriptor_tail{2u, 0x20077facu};
    std::array<u32, 16> eud{};
    u32 eud_dwords = 0;
    bool have_eud_pointer = false;
    u32 eud_pointer_sgpr = 0;
    u32 checks = 0;
    const auto put = [&](u32 start, const u32* words, u32 size) {
        for (u32 k = 0; k < size; ++k) {
            const u32 reg = start + k;
            if (reg < 16) {
                cs_reg(0x40 + reg, words[k]);
            } else if (reg - 16 < eud.size()) {
                eud[reg - 16] = words[k];
                eud_dwords = std::max(eud_dwords, reg - 16 + 1);
            } else {
                ++slot.unsupported_inputs;
            }
        }
    };
    for (u32 i = 0; i < Rd<u8>(gnm, 3) && i < 40; ++i) {
        const u64 e = gnm + 0x28 + i * 4;
        const u32 type = Rd<u8>(e, 0);
        const u32 slot_index = Rd<u8>(e, 1);
        const u32 start = Rd<u8>(e, 2);
        const u32 size = (Rd<u8>(e, 3) & 1) ? 8 : 4;
        if (type == 27) {
            have_eud_pointer = true;
            eud_pointer_sgpr = start;
        } else if (slot_index != 0 || (type != 0 && type != 2 && type != 4) || (type != 2 && size != 8)) {
            ++slot.unsupported_inputs;
        } else if (type == 0) {
            put(start, srv.data(), 8);
        } else if (type == 4) {
            put(start, uav.data(), 8);
        } else if (start + 4 > 16 || checks >= slot.checks.size()) {
            ++slot.unsupported_inputs;
        } else {
            // the constant buffer descriptor holds the address of the constants: compared by content
            slot.checks[checks++] = {Core::NativeDraw::ComputeRegisterBase + 0x40 + start, 30, 8, constants_address, 0, {~0ull, ~0ull}, 0x100000};
            put(start + 2, constant_descriptor_tail.data(), 2);
        }
    }
    if (have_eud_pointer) {
        if (eud_pointer_sgpr + 2 > 16 || checks >= slot.checks.size() || eud_dwords == 0) {
            ++slot.unsupported_inputs;
        } else {
            u32* table = AllocTable(eud_dwords);
            std::memcpy(table, eud.data(), eud_dwords * sizeof(u32));
            slot.checks[checks++] = {Core::NativeDraw::ComputeRegisterBase + 0x40 + eud_pointer_sgpr, 27, eud_dwords,
                                     reinterpret_cast<u64>(table), 0, {~0ull, ~0ull}};
        }
    } else if (eud_dwords != 0) {
        ++slot.unsupported_inputs;
    }
    slot.check_count = checks;
    slot.count = n;
    slot.flags = (table_overflow || bad_object_seen ? SpecialPipeline : 0u) | synth_flags | (slot.unsupported_inputs != 0 ? SpecialPipeline : 0u);
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

/// Guest thread: the record of FUN_02485c00 with a source of several different dwords: the device's copy-with-wrap shader
/// (destination[i] = source[i mod size]; the variant for a power-of-two size is a mask) with the destination as read-write
/// buffer, the source as buffer and a constant buffer holding the element count and the size mask. `writer` is the draw
/// context's command buffer structure (context + 0x47c0).
u64 MakeFillPatternRecord(u64 writer, u64 destination, u32 count, u64 source, u32 size, u64 caller) {
    const u64 ctx = writer - 0x47c0;
    const VAddr image = image_base.load(std::memory_order_relaxed);
    if (image == 0 || count == 0 || size == 0) {
        return 0;
    }
    const bool power_of_two = ((size - 1) & size) == 0;
    const u64 device = Rd<u64>(image + (power_of_two ? 0x6b03550 : 0x6b03548) - 0x1000000, 0);
    const u64 gnm = device ? Rd<u64>(device, 0x120) : 0;
    if (gnm == 0) {
        return 0;
    }
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    slot.unsupported_mask = 0;
    synth_flags = 0;
    slot.kind = 3;
    slot.caller = caller;
    slot.a1 = (count + 0x3f) >> 6;
    slot.a2 = 1;
    if (size <= 4) {
        slot.fill_destination = destination;
        slot.fill_bytes = count * 4;
        slot.fill_dwords = size;
        for (u32 k = 0; k < size; ++k) {
            slot.fill_pattern[k] = Rd<u32>(source, k * 4);
        }
    }
    slot.rt_info = 0;
    slot.dbg_vs = slot.dbg_fetch = slot.dbg_layout = 0;
    slot.ps_gnm = 0;
    slot.vs_gnm = gnm;
    u32 n = 0;
    const auto cs_reg = [&](u32 dword, u32 value) { slot.regs[n++] = {Core::NativeDraw::ComputeRegisterBase + dword, value}; };
    const u64 program = gnm + 8;
    cs_reg(0x00, (u32(Rd<u8>(ctx, 0x13015)) * 8 & 0x18) + 1);
    cs_reg(0x01, slot.a1);
    cs_reg(0x02, 1);
    cs_reg(0x03, 1);
    cs_reg(0x07, Rd<u32>(program, 16));
    cs_reg(0x08, Rd<u32>(program, 20));
    cs_reg(0x09, Rd<u32>(program, 24));
    cs_reg(0x0c, Rd<u32>(program, 0));
    cs_reg(0x0d, 0);
    cs_reg(0x12, Rd<u32>(program, 8));
    cs_reg(0x13, Rd<u32>(program, 12));
    const std::array<u32, 4> uav{static_cast<u32>(destination), static_cast<u32>(destination >> 32) & 0xfff | 0x47000, count, 0x18024004u};
    const std::array<u32, 4> srv{static_cast<u32>(source), static_cast<u32>(source >> 32) & 0xfff | 0x40000, size, 0x20024004u};
    u32* constants = AllocTable(4);
    constants[0] = count;
    constants[1] = size - (power_of_two ? 1u : 0u);
    constants[2] = constants[3] = 0;
    const u64 constants_address = reinterpret_cast<u64>(constants);
    const std::array<u32, 2> constant_descriptor_tail{1u, 0x20077facu};
    std::array<u32, 16> eud{};
    u32 eud_dwords = 0;
    bool have_eud_pointer = false;
    u32 eud_pointer_sgpr = 0;
    u32 checks = 0;
    const auto put = [&](u32 start, const u32* words, u32 words_count) {
        for (u32 k = 0; k < words_count; ++k) {
            const u32 reg = start + k;
            if (reg < 16) {
                cs_reg(0x40 + reg, words[k]);
            } else if (reg - 16 < eud.size()) {
                eud[reg - 16] = words[k];
                eud_dwords = std::max(eud_dwords, reg - 16 + 1);
            } else {
                ++slot.unsupported_inputs;
            }
        }
    };
    for (u32 i = 0; i < Rd<u8>(gnm, 3) && i < 40; ++i) {
        const u64 e = gnm + 0x28 + i * 4;
        const u32 type = Rd<u8>(e, 0);
        const u32 slot_index = Rd<u8>(e, 1);
        const u32 start = Rd<u8>(e, 2);
        const bool big = (Rd<u8>(e, 3) & 1) != 0;
        if (type == 27) {
            have_eud_pointer = true;
            eud_pointer_sgpr = start;
        } else if (type == 26) {
            const u64 table = Rd<u64>(ctx, 0x12f30);
            const std::array<u32, 2> words{static_cast<u32>(table), static_cast<u32>(table >> 32)};
            put(start, words.data(), 2);
        } else if (slot_index != 0 || big || (type != 0 && type != 2 && type != 4)) {
            ++slot.unsupported_inputs;
        } else if (type == 0) {
            put(start, srv.data(), 4);
        } else if (type == 4) {
            put(start, uav.data(), 4);
        } else if (start + 4 > 16 || checks >= slot.checks.size()) {
            ++slot.unsupported_inputs;
        } else {
            // the constant buffer descriptor holds the address of the constants: compared by content
            slot.checks[checks++] = {Core::NativeDraw::ComputeRegisterBase + 0x40 + start, 31, 2, constants_address, 0, {~0ull, ~0ull}, 0x100000};
            put(start + 2, constant_descriptor_tail.data(), 2);
        }
    }
    if (have_eud_pointer) {
        if (eud_pointer_sgpr + 2 > 16 || checks >= slot.checks.size() || eud_dwords == 0) {
            ++slot.unsupported_inputs;
        } else {
            u32* table = AllocTable(eud_dwords);
            std::memcpy(table, eud.data(), eud_dwords * sizeof(u32));
            slot.checks[checks++] = {Core::NativeDraw::ComputeRegisterBase + 0x40 + eud_pointer_sgpr, 27, eud_dwords,
                                     reinterpret_cast<u64>(table), 0, {~0ull, ~0ull}};
        }
    } else if (eud_dwords != 0) {
        ++slot.unsupported_inputs;
    }
    slot.check_count = checks;
    slot.count = n;
    slot.flags = (table_overflow || bad_object_seen ? SpecialPipeline : 0u) | synth_flags | (slot.unsupported_inputs != 0 ? SpecialPipeline : 0u);
    slot.seq.store(seq, std::memory_order_release);
    return seq;
}

struct VerifyStats {
    std::mutex mutex;
    u64 draws_checked{};
    u64 draws_missing{};
    u64 registers_compared{};
    u64 mismatches{};
    u64 draws_with_mismatch{};
    u64 draws_flagged{};
    u64 compute_checked{};
    u64 compute_differ{};
    u64 compute_flagged{};
    u64 draws_clean_mismatch{};
    std::array<u64, 4> flag_counts{};
    std::unordered_map<u64, u64> index_type_pairs;
    std::unordered_map<u32, u64> rt_odd_info;
    std::map<std::tuple<u64, u32, u32, u32>, u64> flagged_causes; ///< caller, flags, reason bits, unsupported input types -> draws
    std::map<std::pair<u64, bool>, u64> callers; ///< draw call return address, differs
    struct PerReg {
        u64 count{};
        u32 expected{};
        u32 actual{};
    };
    std::unordered_map<u32, PerReg> by_register;
    std::unordered_map<u32, PerReg> by_register_cs; ///< differences of dispatch records
    std::unordered_map<u32, u64> uncovered; ///< registers that changed between draws without being in the record
    std::unordered_map<u32, std::pair<u32, u32>> uncovered_values; ///< last value and the one before
    u64 last_log_ns{};
} verify_stats;

}  // namespace

void VerifyRecordImpl(u64 seq, const u32* reg_array, const AmdGpu::DrawRegs* mapping);

void VerifyRecord(u64 seq, const AmdGpu::DrawRegs& state) {
    static thread_local std::unique_ptr<u32[]> dump;
    if (!dump) {
        dump = std::make_unique<u32[]>(AmdGpu::Regs::NumRegs);
    }
    state.Dump(dump.get());
    VerifyRecordImpl(seq, dump.get(), &state);
}

void VerifyRecord(u64 seq, const u32* reg_array) {
    VerifyRecordImpl(seq, reg_array, nullptr);
}

void VerifyRecordImpl(u64 seq, const u32* reg_array, const AmdGpu::DrawRegs* mapping) {
    NoteConsumed(seq);
    auto& slot = Slot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        std::scoped_lock lk{verify_stats.mutex};
        ++verify_stats.draws_missing;
        return;
    }
    // copy: the guest thread may reuse the slot while this is compared
    std::array<SynthReg, MaxSynthRegs> regs = slot.regs;
    u32 count = slot.count;
    const u64 vs_gnm = slot.vs_gnm;
    const u64 ps_gnm = slot.ps_gnm;
    const u32 flags = slot.flags;
    const u32 reason = slot.reason;
    const u32 unsupported_types = slot.unsupported_mask;
    const u32 record_kind = slot.kind;
    const u64 caller = slot.caller;
    const u32 rt_info = slot.rt_info;
    const u64 dbg_vs = slot.dbg_vs;
    const u64 dbg_fetch = slot.dbg_fetch;
    const u64 dbg_layout = slot.dbg_layout;
    u32 check_count = slot.check_count;
    auto checks = slot.checks;
    const u32 unsupported = slot.unsupported_inputs;
    const u32 unsupported_mask = slot.unsupported_mask;
    if (mapping != nullptr) {
        // the draw state holds only what the backend reads: the other registers cannot be compared
        u32 kept = 0;
        for (u32 i = 0; i < count; ++i) {
            if (AmdGpu::DrawRegs::IsMapped(regs[i].index)) {
                regs[kept++] = regs[i];
            }
        }
        count = kept;
        kept = 0;
        for (u32 i = 0; i < check_count; ++i) {
            if (AmdGpu::DrawRegs::IsMapped(checks[i].reg) && AmdGpu::DrawRegs::IsMapped(checks[i].reg + 1)) {
                checks[kept++] = checks[i];
            }
        }
        check_count = kept;
    }
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        std::scoped_lock lk{verify_stats.mutex};
        ++verify_stats.draws_missing;
        return;
    }
    std::scoped_lock lk{verify_stats.mutex};
    auto& s = verify_stats;
    auto& by_reg = record_kind == 3 ? s.by_register_cs : s.by_register;
    ++s.draws_checked;
    bool any = false;
    bool critical = false;
    {
        static std::map<u64, u32> caller_dumps;
        u32& dumped_for_caller = caller_dumps[caller];
        if (dumped_for_caller < (flags == 0 ? 12u : 2u)) {
            std::string diff;
            u32 differing = 0;
            for (u32 i = 0; i < count; ++i) {
                if (reg_array[regs[i].index] != regs[i].value && regs[i].index != ShBase + 0x4a) {
                    ++differing;
                    if (differing <= 30) {
                        diff += fmt::format(" {:#x}:{:x}/{:x}", regs[i].index, regs[i].value, reg_array[regs[i].index]);
                    }
                }
            }
            if (differing > 0 && count > 0) {
                ++dumped_for_caller;
                const VAddr base = image_base.load(std::memory_order_relaxed);
                LOG_INFO(Core, "native draw diff: caller {:#x} kind {} count {} actual count {} flags {} regs differ {} (built/actual):{}",
                         base ? caller - base + 0x1000000 : caller, slot.kind, slot.a1,
                         record_kind == 3 ? 0u : reg_array[UconfigBase + 0x24c], flags, differing, diff);
            }
        }
    }
    for (u32 i = 0; i < count; ++i) {
        ++s.registers_compared;
        const u32 actual = reg_array[regs[i].index];
        if (actual != regs[i].value) {
            critical |= regs[i].index != ShBase + 0x4a;
            if (regs[i].index == ShBase + 0x4c || regs[i].index == ShBase + 0x4a) {
                static u32 fetch_dumps = 0;
                static std::map<u64, u32> fetch_dumped_vs;
                if (fetch_dumps < 200 && dbg_vs != 0 && fetch_dumped_vs[dbg_vs]++ < 2) {
                    ++fetch_dumps;
                    const auto rd = [](u64 base, u64 off) { return base ? Rd<u64>(base, off) : u64{0}; };
                    const u64 alt = rd(dbg_vs + 0x330, 0xe0);
                    LOG_INFO(Core,
                             "native fetch diff: layout {:#x} caller {:#x} reg {:#x} vs {:#x} variant420 {:#x} fetch_obj {:#x} built {:#x} actual {:#x}/{:#x} "
                             "obj30 {:#x} 38 {:#x} 40 {:#x} 48 {:#x} | variant330 obj {:#x}: 30 {:#x} 40 {:#x}",
                             dbg_layout, caller, regs[i].index, dbg_vs, rd(dbg_vs, 0x420), dbg_fetch, regs[i].value, actual, reg_array[ShBase + 0x4d],
                             rd(dbg_fetch, 0x30), rd(dbg_fetch, 0x38), rd(dbg_fetch, 0x40), rd(dbg_fetch, 0x48), alt,
                             rd(alt, 0x30), rd(alt, 0x40));
                }
            }
            if (regs[i].index == ContextBase + 0x29f) {
                ++s.index_type_pairs[(u64(regs[i].value) << 32) | actual];
            }
            ++s.mismatches;
            any = true;
            auto& r = by_reg[regs[i].index];
            ++r.count;
            r.expected = regs[i].value;
            r.actual = actual;
        }
    }
    // tables the user data points to: compare their content
    for (u32 c = 0; c < check_count; ++c) {
        ++s.registers_compared;
        const auto& check = checks[c];
        const u64 ptr = ((u64(reg_array[check.reg + 1]) << 32) | reg_array[check.reg]) & 0x0000ffffffffffffull;
        bool same = false;
        u32 first_actual = 0;
        if (ptr != 0 && Core::Memory::Instance()->IsValidMapping(ptr, check.dwords * 4)) {
            if (check.entry_dwords == 0) {
                same = std::memcmp(reinterpret_cast<const void*>(check.address), reinterpret_cast<const void*>(ptr),
                                   check.dwords * 4) == 0;
            } else {
                same = true;
                for (u32 slot = 0; slot < check.dwords / check.entry_dwords && same; ++slot) {
                    if (slot < 128 && ((check.slot_mask[slot >> 6] >> (slot & 63)) & 1) != 0) {
                        same = std::memcmp(reinterpret_cast<const u32*>(check.address) + slot * check.entry_dwords,
                                           reinterpret_cast<const void*>(ptr + slot * check.entry_dwords * 4),
                                           check.entry_dwords * 4) == 0;
                    }
                }
            }
            first_actual = Rd<u32>(ptr, 0);
        }
        if (!same) {
            critical = true;
            static u32 table_dumps = 0;
            static std::map<std::pair<u64, u32>, u32> table_dumped;
            if (ptr != 0 && table_dumps < 300 && table_dumped[{caller, check.reg}]++ < 2 && Core::Memory::Instance()->IsValidMapping(ptr, check.dwords * 4)) {
                ++table_dumps;
                std::string text;
                for (u32 k = 0; k < check.dwords && k < 48; ++k) {
                    const u32 b = Rd<u32>(check.address, k * 4);
                    const u32 a = Rd<u32>(ptr, k * 4);
                    text += b == a ? fmt::format(" {:x}", a) : fmt::format(" [{}:{:x}/{:x}]", k, b, a);
                }
                const VAddr image = image_base.load(std::memory_order_relaxed);
                LOG_INFO(Core, "native table diff: caller {:#x} kind {} reg {:#x} dwords {} used {:#x}/{:#x} (built/actual):{}",
                         image ? caller - image + 0x1000000 : caller, check.kind, check.reg, check.dwords, check.slot_mask[0],
                         check.slot_mask[1], text);
            }
            ++s.mismatches;
            any = true;
            auto& r = by_reg[0x90000000u + check.kind * 0x100 + (check.reg & 0xff)];
            ++r.count;
            r.expected = Rd<u32>(check.address, 0);
            r.actual = first_actual;
        }
    }
    if (unsupported != 0) {
        auto& r = by_reg[0x91000000u];
        r.count += unsupported;
        for (u32 bit = 0; bit < 32; ++bit) {
            if ((unsupported_mask >> bit) & 1) {
                ++by_reg[0x91000100u + bit].count;
            }
        }
    }
    if (record_kind != 3) {
        constexpr u32 ContextFirst = 0xA000, ContextCount = 0x400, ShFirst = 0x2C00, ShCount = 0x280, UconfigFirst = 0xC200,
                      UconfigCount = 0x80;
        constexpr u32 Total = ContextCount + ShCount + UconfigCount;
        static std::array<u32, Total> previous{};
        static std::array<u64, Total> covered_in{};
        static u64 stamp = 0;
        static bool have_previous = false;
        ++stamp;
        const auto slot_of = [](u32 index) -> u32 {
            if (index >= ContextFirst && index < ContextFirst + ContextCount) {
                return index - ContextFirst;
            }
            if (index >= ShFirst && index < ShFirst + ShCount) {
                return ContextCount + index - ShFirst;
            }
            if (index >= UconfigFirst && index < UconfigFirst + UconfigCount) {
                return ContextCount + ShCount + index - UconfigFirst;
            }
            return Total;
        };
        for (u32 i = 0; i < count; ++i) {
            if (const u32 slot_index = slot_of(regs[i].index); slot_index < Total) {
                covered_in[slot_index] = stamp;
            }
        }
        for (u32 c = 0; c < check_count; ++c) {
            for (u32 k = 0; k < 2; ++k) {
                if (const u32 slot_index = slot_of(checks[c].reg + k); slot_index < Total) {
                    covered_in[slot_index] = stamp;
                }
            }
        }
        for (u32 slot_index = 0; slot_index < Total; ++slot_index) {
            const u32 index = slot_index < ContextCount ? ContextFirst + slot_index
                              : slot_index < ContextCount + ShCount ? ShFirst + slot_index - ContextCount
                                                                    : UconfigFirst + slot_index - ContextCount - ShCount;
            const u32 value = reg_array[index];
            if (have_previous && flags == 0 && value != previous[slot_index] && covered_in[slot_index] != stamp) {
                ++s.uncovered[index];
                if (index == ShFirst + 0x0c || index == ShFirst + 0x54 || index == ShFirst + 0x58) {
                    static std::array<u32, 3> sgpr_dumps{};
                    u32& dumped_count = sgpr_dumps[index == ShFirst + 0x0c ? 0 : index == ShFirst + 0x54 ? 1 : 2];
                    if (dumped_count < 8) {
                        ++dumped_count;
                        const u64 gnm = index == ShFirst + 0x0c ? ps_gnm : vs_gnm;
                        std::string inputs, ud;
                        for (u32 k = 0; gnm && k < Rd<u8>(gnm, 3) && k < 40; ++k) {
                            const u64 e = gnm + (index == ShFirst + 0x0c ? 0x3c : 0x28) + k * 4;
                            inputs += fmt::format("[{} s{} r{} f{}]", Rd<u8>(e, 0), Rd<u8>(e, 1), Rd<u8>(e, 2), Rd<u8>(e, 3));
                        }
                        const u32 base = index == ShFirst + 0x0c ? ShFirst + 0x0c : ShFirst + 0x4c;
                        for (u32 k = 0; k < 16; ++k) {
                            ud += fmt::format(" {:x}", reg_array[base + k]);
                        }
                        LOG_INFO(Core, "native uncovered user data {:#x}: shader {:#x} inputs {} | actual{}", index, gnm,
                                 inputs, ud);
                    }
                }
                s.uncovered_values[index] = {value, previous[slot_index]};
            }
            previous[slot_index] = value;
        }
        have_previous = true;
    }
    s.draws_with_mismatch += any;
    s.draws_flagged += flags != 0;
    if (record_kind == 3) {
        ++s.compute_checked;
        s.compute_differ += any;
        s.compute_flagged += flags != 0;
    }
    ++s.callers[{caller, critical}];
    if (auto* trust = TrustFor(caller)) {
        trust->total.fetch_add(1, std::memory_order_relaxed);
        // a record the builder itself flagged (a shader whose fetch object does not exist yet, ...) never executes; its
        // difference says nothing about the call site
        trust->bad.fetch_add(critical && flags == 0 ? 1 : 0, std::memory_order_relaxed);
        if (trust->total.load(std::memory_order_relaxed) >= 4096) {
            trust->total.store(trust->total.load() / 2, std::memory_order_relaxed);
            trust->bad.store(trust->bad.load() / 2, std::memory_order_relaxed);
        }
    }
    if (flags & OddRenderTarget) {
        ++s.rt_odd_info[rt_info];
    }
    if (flags != 0 && record_kind != 3) {
        ++s.flagged_causes[{caller, flags, reason, unsupported_types}];
    }
    for (u32 bit = 0; bit < 4; ++bit) {
        s.flag_counts[bit] += (flags >> bit) & 1;
    }
    s.draws_clean_mismatch += any && flags == 0;
    if (any || true) {
        // the first draws of each distinct vertex/pixel shader pair: input table and the actual user data
        static std::unordered_map<u64, u32> dumped;
        const u64 pair = vs_gnm * 31 + ps_gnm;
        if (dumped.size() < 60 && vs_gnm && dumped.emplace(pair, 0).second) {
            const auto dump_inputs = [&](u64 gnm, u64 inputs_offset) {
                std::string text;
                const u32 n_inputs = Rd<u8>(gnm, 3);
                for (u32 i = 0; i < n_inputs && i < 40; ++i) {
                    const u64 e = gnm + inputs_offset + i * 4;
                    text += fmt::format("[{} s{} r{} f{}]", Rd<u8>(e, 0), Rd<u8>(e, 1), Rd<u8>(e, 2), Rd<u8>(e, 3));
                }
                return text;
            };
            std::string vs_ud, ps_ud;
            for (u32 i = 0; i < 16; ++i) {
                vs_ud += fmt::format(" {:x}", reg_array[ShBase + 0x4c + i]);
                ps_ud += fmt::format(" {:x}", reg_array[ShBase + 0x0c + i]);
            }
            std::string ps_words, ps_inputs_cntl;
            for (u32 i = 0; ps_gnm && i < 40; ++i) {
                ps_words += fmt::format(" {:x}", Rd<u32>(ps_gnm, i * 4));
            }
            for (u32 i = 0; i < 12; ++i) {
                ps_inputs_cntl += fmt::format(" {:x}", reg_array[ContextBase + 0x191 + i]);
            }
            LOG_INFO(Core, "native ps words: gnm {:#x}:{} | actual 0x191..:{}", ps_gnm, ps_words, ps_inputs_cntl);
            LOG_INFO(Core, "native shader dump: vs {:#x} inputs {} | actual vs user data{} | ps {:#x} inputs {} | actual ps user data{}",
                     vs_gnm, dump_inputs(vs_gnm, 0x28), vs_ud, ps_gnm, ps_gnm ? dump_inputs(ps_gnm, 0x3c) : std::string{}, ps_ud);
        }
    }

    const u64 now = NowNs();
    if (now - s.last_log_ns > 5'000'000'000ull) {
        if (s.last_log_ns != 0) {
            std::vector<std::pair<u32, VerifyStats::PerReg>> top(s.by_register.begin(), s.by_register.end());
            std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
            LOG_INFO(Core,
                     "native front end check: {} draws checked, {} without record, {} registers compared, {} differ "
                     "({} draws with a difference)",
                     s.draws_checked, s.draws_missing, s.registers_compared, s.mismatches, s.draws_with_mismatch);
            LOG_INFO(Core, "native front end check: dispatches: {} checked, {} differ, {} cannot run natively",
                     s.compute_checked, s.compute_differ, s.compute_flagged);
            LOG_INFO(Core,
                     "native front end check: {} draws cannot run natively (no vertex shader {}, no fetch object {}, odd "
                     "render target {}, special pipeline {}); {} of the others differ",
                     s.draws_flagged, s.flag_counts[0], s.flag_counts[1], s.flag_counts[2], s.flag_counts[3],
                     s.draws_clean_mismatch);
            for (size_t k = 0; k < top.size() && k < 12; ++k) {
                LOG_INFO(Core, "native front end check: register {:#x} differs {} times (last: built {:#x}, actual {:#x})",
                         top[k].first, top[k].second.count, top[k].second.expected, top[k].second.actual);
            }
            {
                std::vector<std::pair<u32, VerifyStats::PerReg>> top_cs(s.by_register_cs.begin(), s.by_register_cs.end());
                std::ranges::sort(top_cs, [](const auto& x, const auto& y) { return x.second.count > y.second.count; });
                for (size_t k = 0; k < top_cs.size() && k < 6; ++k) {
                    LOG_INFO(Core, "native front end check: dispatch register {:#x} differs {} times (last: built {:#x}, actual {:#x})",
                             top_cs[k].first, top_cs[k].second.count, top_cs[k].second.expected, top_cs[k].second.actual);
                }
            }
            for (const auto& [pair, count] : s.index_type_pairs) {
                LOG_INFO(Core, "native front end check: index type built {:#x} actual {:#x}: {} draws", pair >> 32,
                         pair & 0xffffffff, count);
            }
            {
                std::vector<std::pair<u64, std::tuple<u64, u32, u32, u32>>> causes;
                for (const auto& [key, count] : s.flagged_causes) {
                    causes.emplace_back(count, key);
                }
                std::ranges::sort(causes, [](const auto& x, const auto& y) { return x.first > y.first; });
                const VAddr image = image_base.load(std::memory_order_relaxed);
                for (size_t k = 0; k < causes.size() && k < 14; ++k) {
                    const auto& [caller_address, cause_flags, cause_reason, cause_types] = causes[k].second;
                    LOG_INFO(Core, "native front end check: cannot run natively: caller {:#x} flags {:#x} reason {:#x} unsupported types {:#x}: {} draws",
                             image ? caller_address - image + 0x1000000 : caller_address, cause_flags, cause_reason, cause_types, causes[k].first);
                }
                s.flagged_causes.clear();
            }
            for (const auto& [info, count] : s.rt_odd_info) {
                LOG_INFO(Core, "native front end check: odd render targets: count {} views {:#x} empty blocks {:#x}: {} draws",
                         info & 0xff, (info >> 8) & 0xff, (info >> 16) & 0xff, count);
            }
            {
                const VAddr base = image_base.load(std::memory_order_relaxed);
                std::vector<std::pair<u64, u64>> bad;
                for (const auto& [key, count] : s.callers) {
                    if (key.second) {
                        const u64 total = count + s.callers[{key.first, false}];
                        bad.push_back({key.first, count * 1000000 + total});
                    }
                }
                std::ranges::sort(bad, [](const auto& x, const auto& y) { return x.second > y.second; });
                for (size_t k = 0; k < bad.size() && k < 6; ++k) {
                    LOG_INFO(Core, "native front end check: draws called from {:#x} differ {} of {}",
                             base ? bad[k].first - base + 0x1000000 : bad[k].first, bad[k].second / 1000000,
                             bad[k].second % 1000000);
                }
            }
            std::vector<std::pair<u32, u64>> gaps(s.uncovered.begin(), s.uncovered.end());
            std::ranges::sort(gaps, [](const auto& x, const auto& y) { return x.second > y.second; });
            std::string gap_text;
            for (size_t k = 0; k < gaps.size() && k < 90; ++k) {
                const auto v = s.uncovered_values[gaps[k].first];
                gap_text += fmt::format("\n  {:#x}: {} draws (last {:#x} was {:#x})", gaps[k].first, gaps[k].second,
                                        v.first, v.second);
            }
            LOG_INFO(Core, "native front end gaps ({} registers changed outside the record, register:draws):{}",
                     gaps.size(), gap_text);
        }
        s.draws_checked = s.draws_missing = s.registers_compared = s.mismatches = s.draws_with_mismatch = 0;
        s.draws_flagged = s.draws_clean_mismatch = 0;
        s.compute_checked = s.compute_differ = s.compute_flagged = 0;
        s.flag_counts = {};
        s.by_register.clear();
        s.by_register_cs.clear();
        s.uncovered.clear();
        s.index_type_pairs.clear();
        s.callers.clear();
        s.rt_odd_info.clear();
        s.uncovered_values.clear();
        s.last_log_ns = now;
    }
}

void VerifyComputeRecord(u64 seq, const u32* cs_dwords) {
    VerifyRecord(seq, cs_dwords - ComputeRegisterBase);
}

namespace {

/// Appends the register writes of `packets` to `out`; false when something else is in them. `strict_nops`: padding must be
/// plain (no markers, tags or flips).
bool ParseStateSequence(const u32* packets, u32 dwords, StateTemplate& out, bool strict_nops, bool log) {
    for (u32 at = 0; at < dwords;) {
        const u32 header = packets[at];
        if ((header >> 30) != 3) {
            return false;
        }
        const u32 opcode = (header >> 8) & 0xff;
        const u32 size = ((header >> 16) & 0x3fff) + 2;
        if (at + size > dwords) {
            if (opcode == 0x10 && !strict_nops) {
                return true; // padding that runs to the end of the buffer
            }
            if (log) {
                LOG_INFO(Core, "hardware state sequence: packet {:#x} at dword {} of {} runs past the table", opcode, at, dwords);
            }
            return false;
        }
        switch (opcode) {
        case 0x10: // NOP
            if (strict_nops && size > 2 && ((packets[at + 1] >> 16) == 0x6875 || packets[at + 1] == VideoCore::NativeTagMagic)) {
                return false;
            }
            break;
        case 0x28: // CONTEXT_CONTROL
        case 0x58: // ACQUIRE_MEM
            break;
        case 0x12:
            out.writes.push_back({StateWrite::ClearState, 0, 0});
            break;
        case 0x2f:
            out.writes.push_back({StateWrite::NumInstances, 0, packets[at + 1]});
            break;
        case 0x2a:
            out.writes.push_back({StateWrite::IndexType, 0, packets[at + 1]});
            break;
        case 0x69:
        case 0x76:
        case 0x79: {
            const u32 offset = packets[at + 1];
            const u8 kind = opcode == 0x69 ? StateWrite::Context : opcode == 0x76 ? StateWrite::Sh : StateWrite::Uconfig;
            for (u32 k = 0; k + 2 < size; ++k) {
                // colour and depth target registers carry a trailing extent hint the packet handler reads
                if (kind == StateWrite::Context && ((offset + k >= 0x318 && offset + k < 0x3a0) || (offset + k >= 0x10 && offset + k < 0x20))) {
                    if (log) {
                        LOG_INFO(Core, "hardware state sequence: target register {:#x} is written", offset + k);
                    }
                    return false;
                }
                out.writes.push_back({kind, offset + k, packets[at + 2 + k]});
            }
            break;
        }
        default:
            if (log) {
                LOG_INFO(Core, "hardware state sequence: unsupported packet {:#x} at dword {}", opcode, at);
            }
            return false;
        }
        at += size;
    }
    return true;
}

} // namespace

const StateTemplate* HardwareStateTemplate(const u32* prefix, u32 prefix_dwords, const u32* sequence, u32 sequence_dwords) {
    static std::mutex mutex;
    static std::map<const u32*, std::unique_ptr<StateTemplate>> templates;
    std::lock_guard lock{mutex};
    if (const auto it = templates.find(sequence); it != templates.end()) {
        return it->second.get();
    }
    auto result = std::make_unique<StateTemplate>();
    const bool supported = ParseStateSequence(prefix, prefix_dwords, *result, false, true) &&
                           ParseStateSequence(sequence, sequence_dwords, *result, false, true);
    if (!supported) {
        LOG_INFO(Core, "hardware state sequence {} is not a plain register sequence: it stays packets", static_cast<const void*>(sequence));
    }
    const StateTemplate* out = supported ? result.get() : nullptr;
    templates[sequence] = supported ? std::move(result) : nullptr;
    return out;
}

const StateTemplate* IndirectStateTemplate(const u32* packets, u32 dwords) {
    struct Entry {
        std::unique_ptr<StateTemplate> state;
        u32 dwords{};
    };
    static std::mutex mutex;
    static std::unordered_map<u64, Entry> templates;
    u64 hash = 0xcbf29ce484222325ull ^ dwords;
    for (u32 i = 0; i < dwords; ++i) {
        hash = (hash ^ packets[i]) * 0x100000001b3ull;
    }
    std::lock_guard lock{mutex};
    if (const auto it = templates.find(hash); it != templates.end()) {
        return it->second.dwords == dwords ? it->second.state.get() : nullptr;
    }
    if (templates.size() > 256) {
        return nullptr; // the buffers are not constant: stop spending time on them
    }
    auto result = std::make_unique<StateTemplate>();
    const bool supported = ParseStateSequence(packets, dwords, *result, true, false);
    if (supported) {
        LOG_INFO(Core, "indirect buffer of {} dwords is a constant state sequence: {} register writes", dwords, result->writes.size());
    }
    Entry entry;
    entry.dwords = dwords;
    if (supported) {
        entry.state = std::move(result);
    }
    const StateTemplate* out = entry.state.get();
    templates.emplace(hash, std::move(entry));
    return out;
}

u32 RecordDrawCount(u64 seq) {
    auto& slot = Slot(seq);
    const u32 count = slot.a1;
    return slot.seq.load(std::memory_order_acquire) == seq ? count : 0xffffffffu;
}

u64 RecordCaller(u64 seq) {
    auto& slot = Slot(seq);
    return slot.seq.load(std::memory_order_acquire) == seq ? slot.caller : 0;
}

bool TakeExecDraw(u64 seq, ExecDraw& out) {
    NoteConsumed(seq);
    auto& slot = Slot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        return false;
    }
    u32 n = 0;
    for (u32 i = 0; i < slot.count && n < ExecDraw::MaxRegisters - 2 * slot.check_count; ++i) {
        out.regs[n++] = {slot.regs[i].index, slot.regs[i].value};
    }
    for (u32 c = 0; c < slot.check_count; ++c) {
        const auto& check = slot.checks[c];
        out.regs[n++] = {check.reg, static_cast<u32>(check.address)};
        out.regs[n++] = {check.reg + 1, static_cast<u32>(check.address >> 32) | check.hi_flags};
    }
    out.count = n;
    out.kind = slot.kind;
    out.start_index = slot.a2;
    out.a1 = slot.a1;
    out.args_addr = slot.args_addr;
    out.extra_addr = slot.extra_addr;
    out.offset = slot.a2;
    out.vertex_sgpr = slot.vertex_sgpr;
    out.cb_extent = slot.cb_extent;
    out.cb_extent_mask = slot.cb_extent_mask;
    out.db_extent = slot.db_extent;
    out.has_extents = slot.kind != 3;
    out.cb_extents_only = slot.kind == 4;
    out.instance_sgpr = slot.instance_sgpr;
    // a fast clear eliminate is not an ordinary draw: it keeps the register based classification
    if (slot.kind == 3 && slot.fill_dwords != 0) {
        out.fill_destination = slot.fill_destination;
        out.fill_bytes = slot.fill_bytes;
        out.fill_dwords = slot.fill_dwords;
        out.fill_pattern = slot.fill_pattern;
    }
    if (slot.kind == 3 && slot.copy_bytes != 0) {
        out.copy_destination = slot.copy_destination;
        out.copy_source = slot.copy_source;
        out.copy_bytes = slot.copy_bytes;
    }
    out.fixed = slot.kind == 4 ? nullptr : slot.fixed;
    out.fixed_stale = slot.fixed_stale;
    return slot.seq.load(std::memory_order_acquire) == seq;
}

namespace {

std::atomic<u64> tags_written{};
std::atomic<u64> tags_skipped_class{};
std::atomic<u64> tags_failed{};
std::atomic<u64> last_tag_log_ns{0};

constexpr u32 ExecFlag = 1u << 16; ///< tag meta: the draw is executed from its record, the game wrote no packets for it
/// Every Nth draw of a call site takes the packet path (0 = never).
u32 AuditPeriod() {
    return VideoCore::ReadbackPolicy::Get().native_audit_every.load(std::memory_order_relaxed);
}

struct ExecContextState {
    u64 ctx{};
    u64 target_signature{}; ///< render targets as the game's own path last bound them
    u64 target_pointers{};  ///< view pointers whose bind side effects were applied
    u32 draws{};
};

thread_local std::array<ExecContextState, 4> exec_states;
thread_local u32 exec_state_next;

ExecContextState& ExecState(u64 ctx) {
    for (auto& state : exec_states) {
        if (state.ctx == ctx) {
            return state;
        }
    }
    auto& state = exec_states[exec_state_next++ % exec_states.size()];
    state = {};
    state.ctx = ctx;
    return state;
}

/// Hash of everything PrepareDraw writes for the render and depth targets.
u64 TargetSignature(u64 ctx) {
    Hasher h;
    const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
    h.Add(count);
    const auto add_view = [&](u64 view) {
        h.Add(view);
        if (view) {
            for (u32 k = 0; k < 14; ++k) {
                h.Add(Rd<u32>(view + 0x54, k * 4));
            }
        }
    };
    for (u32 slot = 0; slot < 7; ++slot) {
        add_view(slot < count ? Rd<u64>(ctx, Ctx::RenderTargets + slot * 8) : 0);
    }
    add_view(Rd<u64>(ctx, 0x130e0));
    const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView);
    h.Add(dsv);
    if (dsv) {
        for (u32 k = 0; k < 12; ++k) {
            h.Add(Rd<u32>(dsv + 0x58, k * 4));
        }
        h.Add(Rd<u64>(dsv, 0x30));
    }
    return h.h | 1;
}

/// The game's own path runs PrepareDraw: remember the targets it bound.
u64 PS4_SYSV_ABI PrepareNote(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx != 0 && VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0) {
        ExecState(ctx).target_signature = TargetSignature(ctx);
    }
    return 0;
}

std::atomic<u64> exec_drawn{};
std::atomic<u64> asc_drawn{}, asc_packet_path{}, asc_no_record{};
std::atomic<u64> compute_drawn{};
std::atomic<u64> exec_fallback{};

/// Why draws stayed on the game's packet path (per 5 s window, logged with the tag counters).
enum Fallback : u32 { FbFlags, FbPostDraw, FbRing, FbTargets, FbTrustNew, FbTrustBad, FbAudit, FbCount };
std::array<std::atomic<u64>, FbCount> fallback_counts;
constexpr const char* FallbackNames[FbCount] = {"record flags", "post-draw flag", "record ring", "targets changed",
                                                "call site not yet verified", "call site differs", "audit"};

bool ExecEligible(u64 ctx, u64 seq, u64 caller) {
    const auto no = [](Fallback why) {
        fallback_counts[why].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (Slot(seq).flags != 0) {
        return no(FbFlags);
    }
    if ((Rd<u8>(ctx, 0x13009) & 1) != 0 || (Rd<u8>(ctx, 0x48de) & 1) != 0) {
        return no(FbPostDraw);
    }
    if (seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
        return no(FbRing);
    }
    CallerTrust* trust = TrustFor(caller);
    if (trust == nullptr) {
        return no(FbTrustNew);
    }
    const u32 total = trust->total.load(std::memory_order_relaxed);
    if (total < 16) {
        return no(FbTrustNew);
    }
    if (trust->bad.load(std::memory_order_relaxed) * 500 >= total) {
        static std::mutex log_mutex;
        static std::map<u64, std::array<u32, 3>> seen; ///< caller -> draws refused, total, bad
        static u64 last_log_ms = 0;
        std::lock_guard lock{log_mutex};
        auto& entry = seen[caller - ImageBase() + 0x1000000];
        ++entry[0];
        entry[1] = total;
        entry[2] = trust->bad.load(std::memory_order_relaxed);
        const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - last_log_ms >= 5000) {
            last_log_ms = now;
            for (const auto& [site, values] : seen) {
                LOG_INFO(Core, "untrusted draw call site {:#x}: {} draws refused, trust total {} bad {}", site, values[0], values[1], values[2]);
            }
            seen.clear();
        }
        return no(FbTrustBad);
    }
    if (const u32 period = AuditPeriod();
        period != 0 && (trust->since_audit.fetch_add(1, std::memory_order_relaxed) + 1) % period == 0) {
        return no(FbAudit);
    }
    return true;
}

/// A render target that was cleared with a fast clear and is now sampled needs its clear eliminated. The game does it with
/// a full-screen quad in eliminate mode; the command processor only needs the target's registers, the eliminate mode of
/// CB_COLOR_CONTROL and the extent hint, which come from the resource's own render target description (+0xb0, hint +0xec).
std::atomic<u64> track_native{}, track_eliminated{}, track_game_second{}, track_game_depth{};

/// The game's resource tracking function (FUN_02479db0); with the tracking hook installed, its original entry.
void(PS4_SYSV_ABI* GameTrackFunction())(u64, u64) {
    u64 entry = Core::GuestHooks::OriginalEntry("ResTransition");
    if (entry == 0) {
        entry = image_base.load(std::memory_order_relaxed) + 0x2479db0 - 0x1000000;
    }
    return reinterpret_cast<void(PS4_SYSV_ABI*)(u64, u64)>(entry);
}

bool EmitFastClearEliminate(u64 ctx, u64 resource) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    BeginTables(seq);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.kind = 4;
    slot.flags = 0;
    slot.a1 = 3;
    slot.caller = 0;
    u32 n = 0;
    for (u32 k = 0; k < 14; ++k) {
        slot.regs[n++] = {ContextBase + 0x318 + k, Rd<u32>(resource, 0xb0 + k * 4)};
    }
    slot.regs[n++] = {ContextBase + 0x202, 0xcc0020u}; // CB_COLOR_CONTROL: eliminate fast clear
    slot.count = n;
    slot.cb_extent = {};
    slot.cb_extent[0] = Rd<u32>(resource, 0xec);
    slot.cb_extent_mask = 1;
    slot.seq.store(seq, std::memory_order_release);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8);
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (4u << 8) | ExecFlag;
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    return WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1);
}

/// Native replacement of the game's per-resource tracking for the common cases: usage flags are cleared (the cache
/// flush has no host effect) and a render target in state 1 / 2 that was cleared with a fast clear gets its clear
/// eliminated, then state 3. Resources with depth transitions or the second resource (+0x138) keep the game's function.
void TrackResource(u64 ctx, u64 resource, bool native, void(PS4_SYSV_ABI* game_track)(u64, u64)) {
    if (native && Rd<u64>(resource, 0x138) != 0) {
        track_game_second.fetch_add(1, std::memory_order_relaxed);
        Live().track_game_second.fetch_add(1, std::memory_order_relaxed);
    } else if (native && (Rd<u32>(resource, 0x144) - 1u) < 2) {
        track_game_depth.fetch_add(1, std::memory_order_relaxed);
        Live().track_game_depth.fetch_add(1, std::memory_order_relaxed);
    }
    if (!native || Rd<u64>(resource, 0x138) != 0 || (Rd<u32>(resource, 0x144) - 1u) < 2) {
        game_track(ctx, resource);
        return;
    }
    track_native.fetch_add(1, std::memory_order_relaxed);
    Live().track_native.fetch_add(1, std::memory_order_relaxed);
    if (Rd<u32>(resource, 0x124) == 1) {
        return;
    }
    if (Rd<u32>(resource, 0x134) != 0) {
        Wr<u32>(resource, 0x134, 0);
    }
    const u32 state = Rd<u32>(resource, 0x14c);
    if (state == 1 || state == 2) {
        if ((Rd<u8>(resource, 0xc1) & 0x20) != 0 && Rd<u32>(resource, 0xcc) != 0 && !EmitFastClearEliminate(ctx, resource)) {
            game_track(ctx, resource);
            return;
        }
        track_eliminated.fetch_add(1, std::memory_order_relaxed);
        Live().track_fce.fetch_add(1, std::memory_order_relaxed);
        Wr<u32>(resource, 0x14c, 3);
    }
}

/// The resource tracking at the start of PrepareDraw: for every shader resource view the vertex and pixel shader
/// use, the game's own tracking function runs (it may write barrier packets or draw internal full-screen quads).
void RunResourceTracking(u64 ctx) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    const bool native = VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0;
    const auto track = GameTrackFunction();
    // engine objects live in the game's heap; anything else is a stale table entry (the game would fault on it too)
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    const auto run_stage = [&](u64 mask_lo, u64 mask_hi, u64 table, u32 stage) {
        for (u32 slot = 0; slot < 128; ++slot) {
            const u64 mask = slot < 64 ? mask_lo : mask_hi;
            if (((mask >> (slot & 63)) & 1) == 0) {
                continue;
            }
            const u64 view = Rd<u64>(ctx, table + slot * 8);
            if (view == 0) {
                continue;
            }
            const u64 resource = valid(view) ? Rd<u64>(view, 0x30) : 0;
            if (!valid(resource)) {
                static std::atomic<u32> reports{0};
                if (reports.fetch_add(1) < 6) {
                    LOG_WARNING(Core, "native tracking: stage {} slot {} has view {:#x} with resource {:#x}: skipped", stage,
                                slot, view, resource);
                }
                continue;
            }
            TrackResource(ctx, resource, native, track);
        }
    };
    if (const u64 vs = Rd<u64>(ctx, Ctx::VertexShader); valid(vs) && Rd<u8>(vs + 0x30, 0xd6) == 0) {
        run_stage(Rd<u64>(vs + 0x30, 0xc0), Rd<u64>(vs + 0x30, 0xc8), Ctx::ShaderResourceTable(0), 0);
    }
    if (const u64 ps = Rd<u64>(ctx, Ctx::PixelShader); valid(ps) && Rd<u8>(ps, 0x106) == 0) {
        run_stage(Rd<u64>(ps, 0xf0), Rd<u64>(ps, 0xf8), Ctx::ShaderResourceTable(1), 1);
    }
}

/// The resource tracking of PrepareDispatch (0x2479b70) for the compute shader's views, done natively.
void RunComputeTracking(u64 ctx) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    const bool native = VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0;
    const auto track = GameTrackFunction();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    const u64 cs = Rd<u64>(ctx, 0x4670);
    const auto track_view = [&](u64 view) {
        if (view != 0 && valid(view)) {
            if (const u64 resource = Rd<u64>(view, 0x30); valid(resource)) {
                TrackResource(ctx, resource, native, track);
            }
        }
    };
    for (u32 bits = Rd<u32>(cs, 0x100), slot = 0; bits != 0; bits >>= 1, ++slot) {
        if ((bits & 1) != 0) {
            track_view(Rd<u64>(ctx, 0x2098 + slot * 8));
        }
    }
    for (u32 slot = 0; slot < 128; ++slot) {
        if (((Rd<u64>(cs, slot < 64 ? 0xf0 : 0xf8) >> (slot & 63)) & 1) != 0) {
            track_view(Rd<u64>(ctx, 0x1980 + slot * 8));
        }
    }
}

/// What PrepareDispatch and the compute part of the constant update engine flush (FUN_0247dad0) leave in the engine's
/// objects besides command packets: the last-use stamps of the shader, its samplers and views, and the append / consume
/// counter range. The descriptor tables of the flush are rebuilt by the next flush that runs (nothing reads them here).
void ComputeSideEffects(u64 ctx) {
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    const u64 cs = Rd<u64>(ctx, 0x4670);
    const u32 stamp = Rd<u32>(ctx, 0x47ac);
    Wr<u32>(cs, 0x2c, stamp);
    const u32 samplers = Rd<u32>(cs, 0xb0) & 0xffff;
    for (u32 slot = 0; slot < 16; ++slot) {
        if (((samplers >> slot) & 1) != 0) {
            if (const u64 sampler = Rd<u64>(ctx, 5 * 0x80 + 0x280 + slot * 8); valid(sampler)) {
                Wr<u32>(sampler, 0x2c, stamp);
            }
        }
    }
    for (u32 slot = 0; slot < 128; ++slot) {
        if (((Rd<u64>(cs, slot < 64 ? 0xc0 : 0xc8) >> (slot & 63)) & 1) != 0) {
            if (const u64 view = Rd<u64>(ctx, 0x1980 + slot * 8); valid(view)) {
                Wr<u32>(view, 0x2c, stamp);
                if (const u64 resource = Rd<u64>(view, 0x30); valid(resource)) {
                    Wr<u32>(resource, 0x2c, stamp);
                }
            }
        }
    }
    for (u32 bits = Rd<u32>(cs, 0xd0), slot = 0; bits != 0; bits >>= 1, ++slot) {
        if ((bits & 1) != 0) {
            if (const u64 view = Rd<u64>(ctx, 0x2098 + slot * 8); valid(view)) {
                Wr<u32>(view, 0x2c, stamp);
                if (Rd<u32>(view, 0x44) == 1) {
                    Wr<u32>(view, 0x84, slot);
                    Wr<u32>(ctx, 0x4f18, Rd<u32>(ctx, 0x130b8) << 16 | 0x10);
                }
            }
        }
    }
}

/// Cheap identity of the bound render and depth views (the flags and stamps of a target bind depend on these only).
u64 TargetPointers(u64 ctx) {
    Hasher h;
    const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
    h.Add(count);
    for (u32 slot = 0; slot < 7 && slot < count; ++slot) {
        h.Add(Rd<u64>(ctx, Ctx::RenderTargets + slot * 8));
    }
    h.Add(Rd<u64>(ctx, 0x130e0));
    h.Add(Rd<u64>(ctx, Ctx::DepthStencilView));
    return h.h | 1;
}

/// The refresh PrepareDraw runs when the targets are marked dirty: every bound render target view updates its register
/// block through its virtual function (+0x18), the depth view through FUN_024755f0.
void RefreshTargets(u64 ctx) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (Rd<u8>(ctx, 0x44c8) != 0) {
        const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
        for (u32 slot = 0; slot < count && slot < 7; ++slot) {
            const u64 view = Rd<u64>(ctx, Ctx::RenderTargets + slot * 8);
            if (valid(view) && valid(Rd<u64>(view, 0))) {
                if (const u64 function = Rd<u64>(Rd<u64>(view, 0), 0x18); function != 0) {
                    reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(function)(view);
                }
            }
        }
    }
    if (Rd<u8>(ctx, 0x44d8) != 0) {
        if (const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView); valid(dsv)) {
            reinterpret_cast<void(PS4_SYSV_ABI*)(u64)>(base + 0x24755f0 - 0x1000000)(dsv);
        }
    }
}

/// What the game's target bind leaves in the resources and views: usage flags and the last-use stamp.
void ApplyTargetSideEffects(u64 ctx) {
    const u32 stamp = Rd<u32>(ctx, 0x47ac);
    const u32 count = Rd<u32>(ctx, Ctx::RenderTargetCount);
    for (u32 slot = 0; slot < count && slot < 7; ++slot) {
        const u64 view = Rd<u64>(ctx, Ctx::RenderTargets + slot * 8);
        if (view == 0) {
            continue;
        }
        if (const u64 resource = Rd<u64>(view, 0x30); resource != 0) {
            if (Rd<u32>(resource, 0x14c) == 1) {
                Wr<u32>(resource, 0x14c, 2);
            }
            Wr<u32>(resource, 0x134, Rd<u32>(resource, 0x134) | (0x40u << slot));
        }
        Wr<u32>(view, 0x2c, stamp);
    }
    if (const u64 dsv = Rd<u64>(ctx, Ctx::DepthStencilView); dsv != 0) {
        if (const u64 resource = Rd<u64>(dsv, 0x30); resource != 0) {
            Wr<u8>(resource, 0x135, Rd<u8>(resource, 0x135) | 0x40);
        }
        Wr<u32>(dsv, 0x2c, stamp);
    }
}

/// What the game's draw functions and PrepareDraw leave in the engine's objects besides command packets.
void ExecSideEffects(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3) {
    if (kind == 2) {
        Wr<u32>(ctx, 0x4720, 0);
        const u32 stamp = Rd<u32>(ctx, 0x47ac);
        for (const u64 object : {Rd<u64>(ctx, Ctx::VertexShader), Rd<u64>(ctx, Ctx::PixelShader),
                                 Rd<u64>(ctx, Ctx::BlendState), Rd<u64>(ctx, Ctx::RasterizerState),
                                 Rd<u64>(ctx, Ctx::DepthStencilState)}) {
            if (object != 0) {
                Wr<u32>(object, 0x2c, stamp);
            }
        }
        return;
    }
    if (Rd<u32>(ctx, 0x4724) != 1) {
        Wr<u32>(ctx, 0x4724, 1);
    }
    Wr<u32>(ctx, 0x479c, static_cast<u32>(kind == 0 ? a3 : a2));
    Wr<u32>(ctx, 0x47a0, 0);
    Wr<u32>(ctx, 0x30, Rd<u32>(ctx, 0x30) + 1);
    Wr<u32>(ctx, 0x38, Rd<u32>(ctx, 0x38) + static_cast<u32>(a1));
    const u32 stamp = Rd<u32>(ctx, 0x47ac);
    for (const u64 object : {Rd<u64>(ctx, Ctx::VertexShader), Rd<u64>(ctx, Ctx::PixelShader),
                             Rd<u64>(ctx, Ctx::BlendState), Rd<u64>(ctx, Ctx::RasterizerState),
                             Rd<u64>(ctx, Ctx::DepthStencilState)}) {
        if (object != 0) {
            Wr<u32>(object, 0x2c, stamp);
        }
    }
    if (const u64 layout = Rd<u64>(ctx, Ctx::InputLayout); layout != 0) {
        Wr<u32>(layout, 0x2c, stamp);
        const u32 elements = Rd<u32>(layout, Layout::ElementCount);
        for (u32 e = 0; e < elements && e < 32; ++e) {
            const u32 vertex_slot = Rd<u32>(layout + Layout::Elements + e * Layout::ElementStride, 8);
            if (const u64 buffer = vertex_slot < 32 ? Rd<u64>(ctx, Ctx::VertexBufferSlots + vertex_slot * 8) : 0; buffer != 0) {
                Wr<u32>(buffer, 0x2c, stamp);
            }
        }
    }
}

/// Set when the PrepareDraw hook skipped PrepareDraw of an inlined draw: the constant update engine flush that the
/// game's code runs right after it is skipped too (the record carries the user data).
thread_local bool skip_cue_flush = false;

/// What PrepareDraw itself leaves in the engine's objects (the draw function around it does its own counters).
void PrepareDrawSideEffects(u64 ctx) {
    Wr<u32>(ctx, 0x47a0, 0);
    const u32 stamp = Rd<u32>(ctx, 0x47ac);
    for (const u64 object : {Rd<u64>(ctx, Ctx::VertexShader), Rd<u64>(ctx, Ctx::PixelShader), Rd<u64>(ctx, Ctx::BlendState),
                             Rd<u64>(ctx, Ctx::RasterizerState), Rd<u64>(ctx, Ctx::DepthStencilState)}) {
        if (object != 0) {
            Wr<u32>(object, 0x2c, stamp);
        }
    }
    if (const u64 layout = Rd<u64>(ctx, Ctx::InputLayout); layout != 0) {
        Wr<u32>(layout, 0x2c, stamp);
        const u32 elements = Rd<u32>(layout, Layout::ElementCount);
        for (u32 e = 0; e < elements && e < 32; ++e) {
            const u32 vertex_slot = Rd<u32>(layout + Layout::Elements + e * Layout::ElementStride, 8);
            if (const u64 buffer = vertex_slot < 32 ? Rd<u64>(ctx, Ctx::VertexBufferSlots + vertex_slot * 8) : 0; buffer != 0) {
                Wr<u32>(buffer, 0x2c, stamp);
            }
        }
    }
}

/// `inline_draw`: the draw is an inlined copy of a draw function; PrepareDraw is what is hooked. With execution the hook
/// skips PrepareDraw and makes the caller skip the draw (it does the counters the caller would have done).
u64 Tag(u64 ctx, u64 a1, u64 a2, u64 a3, u64 ret, u32 kind, bool track, bool inline_draw = false) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || ctx == 0) {
        return 0;
    }
    const Class cls = Classify(ret);
    if (((policy.native_draw_classes.load(std::memory_order_relaxed) >> cls) & 1) == 0) {
        tags_skipped_class.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    // The game's resource tracking runs first (as it does at the start of PrepareDraw): it may update view descriptors
    // and write barrier packets or draw internal quads, which then come before the tag.
    if (policy.native_synth_verify.load(std::memory_order_relaxed) != 0) {
        RefreshTargets(ctx);
    }
    if (track && policy.native_synth_verify.load(std::memory_order_relaxed) != 0) {
        const u64 vertex_shader = Rd<u64>(ctx, Ctx::VertexShader);
        if (vertex_shader != 0 && Rd<u64>(vertex_shader, 0x420) == 0 && Rd<u64>(ctx, 0x4680) == 0 &&
            Rd<u64>(ctx, 0x4688) == 0 && Rd<u32>(ctx, 0x130c8) != 0x200) {
            RunResourceTracking(ctx);
        }
    }
    {
        static std::atomic<u32> forward_dumps{0};
        const VAddr base = image_base.load(std::memory_order_relaxed);
        if (base != 0 && ret - base + 0x1000000 == 0x1c7819d && forward_dumps.fetch_add(1) < 8) {
            const u64 ps = Rd<u64>(ctx, Ctx::PixelShader);
            std::string text = fmt::format("ps {:#x} flags 106={:#x} 107={:#x} 108={:#x} masks {:#x}/{:#x}:", ps,
                                           Rd<u8>(ps, 0x106), Rd<u8>(ps, 0x107), Rd<u32>(ps, 0x108), Rd<u64>(ps, 0xf0),
                                           Rd<u64>(ps, 0xf8));
            for (u32 slot = 0; slot < 16; ++slot) {
                const u64 view = Rd<u64>(ctx, Ctx::ShaderResourceTable(1) + slot * 8);
                if (view) {
                    text += fmt::format(" [{}: view {:#x} kind {} T# {:x} {:x} res {:#x}]", slot, view, Rd<u32>(view, 0x44),
                                        Rd<u32>(view, 0x58), Rd<u32>(view, 0x5c), Rd<u64>(view, 0x30));
                }
            }
            const u64 ps_gnm_dump = Rd<u64>(ps, 0x120);
            for (u32 k = 0; ps_gnm_dump && k < Rd<u8>(ps_gnm_dump, 3) && k < 40; ++k) {
                const u64 e = ps_gnm_dump + 0x3c + k * 4;
                text += fmt::format(" <{} s{} r{} f{}>", Rd<u8>(e, 0), Rd<u8>(e, 1), Rd<u8>(e, 2), Rd<u8>(e, 3));
            }
            LOG_INFO(Core, "native forward dump: {}", text);
        }
    }
    const Signature sig = Compute(ctx, a1, a2, a3, kind);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    const u64 k1 = sig.pipeline_key;
    Hasher h2, h3;
    h2.Add(sig.group[Targets]);
    h2.Add(sig.group[States]);
    h3.Add(sig.group[ShaderResources]);
    h3.Add(sig.group[Samplers]);
    const u64 k2 = h2.h;
    const u64 k3 = h3.h;
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(cls) | (kind << 8);
    std::memcpy(&words[3], &k1, sizeof(u64));
    std::memcpy(&words[5], &k2, sizeof(u64));
    std::memcpy(&words[7], &k3, sizeof(u64));
    const u64 seq = policy.native_synth_verify.load(std::memory_order_relaxed) != 0 ? Guarded([&] { return MakeRecord(ctx, kind, a1, a2, a3, ret); }) : 0;
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    const bool exec = seq != 0 && policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligible(ctx, seq, ret);
    if (exec) {
        words[2] |= ExecFlag;
    } else {
        static std::mutex mutex;
        static std::map<std::tuple<u64, u32, u32>, u32> not_executed; ///< caller, kind, record flags -> draws
        static u64 last_ms = 0;
        std::lock_guard lock{mutex};
        ++not_executed[{ret - ImageBase() + 0x1000000, kind, seq != 0 ? Slot(seq).flags : 0xffffu}];
        const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - last_ms >= 5000) {
            last_ms = now;
            for (const auto& [key, count] : not_executed) {
                LOG_INFO(Core, "draws left on the packet path per 5 s: caller {:#x} kind {} record flags {:#x}: {} draws", std::get<0>(key),
                         std::get<1>(key), std::get<2>(key), count);
            }
            not_executed.clear();
        }
    }
    if (WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_written.fetch_add(1, std::memory_order_relaxed);
        Live().gfx_tags.fetch_add(1, std::memory_order_relaxed);
        if (exec) {
            if (auto& state = ExecState(ctx); state.target_pointers != TargetPointers(ctx)) {
                state.target_pointers = TargetPointers(ctx);
                ApplyTargetSideEffects(ctx);
            }
            exec_drawn.fetch_add(1, std::memory_order_relaxed);
            Live().gfx_exec.fetch_add(1, std::memory_order_relaxed);
            if (inline_draw) {
                PrepareDrawSideEffects(ctx);
                // what the inlined caller does after a successful PrepareDraw and draw packet
                Wr<u32>(ctx, 0x30, Rd<u32>(ctx, 0x30) + 1);
                Wr<u32>(ctx, 0x38, Rd<u32>(ctx, 0x38) + Rd<u32>(ctx, 0x4724) * static_cast<u32>(a1));
                return 0x8000000000000000ull; // PrepareDraw returns false: the caller does not draw
            }
            ExecSideEffects(ctx, kind, a1, a2, a3);
            return 1;
        }
    } else {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
    }
    const u64 t = NowNs();
    u64 last = last_tag_log_ns.load(std::memory_order_relaxed);
    if (t - last > 5'000'000'000ull && last_tag_log_ns.compare_exchange_strong(last, t) && last != 0) {
        LOG_INFO(Core, "native draw tags: {} written, {} failed, {} skipped (class not enabled), {} drawn from records",
                 tags_written.exchange(0), tags_failed.exchange(0), tags_skipped_class.exchange(0),
                 exec_drawn.exchange(0) + compute_drawn.exchange(0));
        std::string why;
        for (u32 k = 0; k < FbCount; ++k) {
            why += fmt::format(" {}: {};", FallbackNames[k], fallback_counts[k].exchange(0));
        }
        LOG_INFO(Core, "native draws on the packet path:{}", why);
        for (const auto& entry : caller_trust) {
            const u64 caller = entry.caller.load(std::memory_order_relaxed);
            const u32 total = entry.total.load(std::memory_order_relaxed);
            const u32 bad = entry.bad.load(std::memory_order_relaxed);
            if (caller != 0 && bad != 0 && bad * 500 >= total) {
                LOG_INFO(Core, "native call site {:#x} stays on the packet path: {} of {} records differ", caller - ImageBase() + 0x1000000, bad,
                         total);
            }
        }
        LOG_INFO(Core, "async compute dispatches: {} from records, {} on the packet path, {} without a record",
                 asc_drawn.exchange(0), asc_packet_path.exchange(0), asc_no_record.exchange(0));
        LOG_INFO(Core, "native tracking: {} resources, {} fast clear eliminates; game function: {} second resource, {} depth state",
                 track_native.exchange(0), track_eliminated.exchange(0), track_game_second.exchange(0),
                 track_game_depth.exchange(0));
    }
    return 0;
}

// The frame allocator (AllocFrameGarlic(allocator, size, align)) hands out memory from one ring: base at
// allocator + 0x18, size in the low half of the first qword. With native_draw_mode 3 and readbacks off the ring is taken out of page tracking.
std::atomic<bool> ring_checked{false};

u64 PS4_SYSV_ABI RingDiscover(u64 allocator, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ring_checked.load(std::memory_order_relaxed)) {
        return 0;
    }
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) < 3 ||
        VideoCore::ReadbackPolicy::ReadbacksMode() != 0 || allocator == 0) {
        return 0;
    }
    if (ring_checked.exchange(true)) {
        return 0;
    }
    const u64 base = Rd<u64>(allocator, 0x18);
    const u64 size = Rd<u64>(allocator, 0x0) & 0xffffffffull;
    if (base == 0 || (base & 0xfff) != 0 || (size & 0xfff) != 0 || size < (1ull << 20) ||
        size > (1ull << 30)) {
        LOG_WARNING(Core, "native draw: frame allocator ring not recognised (base {:#x}, size {:#x})",
                    base, size);
        return 0;
    }
    VideoCore::UntrackedRing::Get().Set(base, size);
    // Pages the tracker already protected become writable; nothing protects them again.
    Core::Memory::Instance()->GetAddressSpace().Protect(base, size, Core::MemoryPermission::ReadWrite);
    LOG_INFO(Core, "native draw: upload ring {:#x} + {:#x} is no longer page tracked", base, size);
    return 0;
}

u64 PS4_SYSV_ABI TagIndexed(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Tag(ctx, a1, a2, a3, ret, 0, (a4 & 0xff) != 0);
}
bool ExecEligibleCompute(u64 ctx, u64 seq, u64 caller) {
    if (Slot(seq).flags != 0 || (Rd<u8>(ctx, 0x13009) & 1) != 0) {
        return false;
    }
    if (seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
        return false;
    }
    CallerTrust* trust = TrustFor(caller);
    if (trust == nullptr) {
        return false;
    }
    const u32 total = trust->total.load(std::memory_order_relaxed);
    if (total < 16 || trust->bad.load(std::memory_order_relaxed) * 500 >= total) {
        return false;
    }
    const u32 period = AuditPeriod();
    return period == 0 || (trust->since_audit.fetch_add(1, std::memory_order_relaxed) + 1) % period != 0;
}

/// A dispatch wrapper (x, y, z thread groups): PrepareDispatch, the constant update engine flush and the dispatch packet.
/// With execution the game's PrepareDispatch still runs (it tracks the bound resources and writes the program
/// packets); the flush and the dispatch packet are replaced by the tag.
u64 TagCompute(u64 ctx, u64 x, u64 y, u64 z, u64 ret, bool exec_capable) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || ctx == 0 || x == 0 ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_draw_classes.load(std::memory_order_relaxed) >> Other) & 1) == 0) {
        return 0;
    }
    // The resource tracking runs natively first; the descriptor flush and the program packets of PrepareDispatch are the
    // game's own and only run (in the wrapper) when the dispatch is not executed from its record
    if (Rd<u64>(ctx, 0x4670) != 0) {
        RunComputeTracking(ctx);
    }
    const u64 seq = Guarded([&] { return MakeComputeRecord(ctx, x, y, z, ret); });
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (3u << 8);
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    const bool exec = exec_capable && policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
                      ExecEligibleCompute(ctx, seq, ret);
    if (exec) {
        words[2] |= ExecFlag;
    }
    if (!WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
    if (!exec) {
        return 0;
    }
    Live().compute_exec.fetch_add(1, std::memory_order_relaxed);
    ComputeSideEffects(ctx);
    Wr<u32>(ctx, 0x34, Rd<u32>(ctx, 0x34) + 1);
    Wr<u32>(ctx, 0x3c, Rd<u32>(ctx, 0x3c) + static_cast<u32>(x) * static_cast<u32>(y) * static_cast<u32>(z));
    compute_drawn.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// JsDeviceContext::DispatchIndirect(context, argument buffer, offset): PrepareDispatch, the set-base packet with the
/// argument buffer and the indirect dispatch packet. The record is the compute record without the group counts (they are
/// read from the argument buffer when the dispatch runs); while the call site is not verified it is tagged like a direct
/// dispatch and compared with the game's packets.
u64 PS4_SYSV_ABI TagDispatchIndirect(u64 ctx, u64 buffer, u64 offset, u64 a3, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || ctx == 0 || !valid(buffer) ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_draw_classes.load(std::memory_order_relaxed) >> Other) & 1) == 0 ||
        ((policy.native_sdk_strip.load(std::memory_order_relaxed) >> 11) & 1) == 0 || Rd<u64>(ctx, 0x4670) == 0) {
        return 0;
    }
    RunComputeTracking(ctx);
    const u64 seq = Guarded([&] { return MakeComputeRecord(ctx, 1, 1, 1, ret); });
    auto& slot = Slot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        return 0;
    }
    u32 kept = 0;
    for (u32 i = 0; i < slot.count; ++i) {
        const u32 index = slot.regs[i].index;
        if (index < Core::NativeDraw::ComputeRegisterBase + 1 || index > Core::NativeDraw::ComputeRegisterBase + 3) {
            slot.regs[kept++] = slot.regs[i]; // the group counts come from the argument buffer
        }
    }
    slot.count = kept;
    const u64 address = (static_cast<u64>(Rd<u32>(buffer, 0x30)) & 0xfffffff8ull) | (static_cast<u64>(Rd<u32>(buffer, 0x34)) & 0xff) << 32;
    const bool exec = policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligibleCompute(ctx, seq, ret);
    slot.kind = exec ? 11 : 3;
    slot.args_addr = address;
    slot.a2 = static_cast<u32>(offset);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | ((exec ? 11u : 3u) << 8) | (exec ? ExecFlag : 0u);
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    if (!WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
    if (!exec) {
        return 0;
    }
    Live().compute_exec.fetch_add(1, std::memory_order_relaxed);
    ComputeSideEffects(ctx);
    Wr<u32>(ctx, 0x34, Rd<u32>(ctx, 0x34) + 1);
    compute_drawn.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// CsCopyBufferDwords(writer, destination, source, bytes): the game's own code builds the descriptors and binds the copy
/// shader; with execution the record replaces that and the dispatch packet.
u64 TagCsCopy(u64 writer, u64 destination, u64 source, u64 bytes, u64 ret) {
    const u64 ctx = writer - 0x47c0;
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || writer == 0 ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_draw_classes.load(std::memory_order_relaxed) >> Other) & 1) == 0) {
        return 0;
    }
    if (bytes >= (8u << 20)) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 12) {
            LOG_INFO(Core, "CsCopyBufferDwords of {} bytes: caller {:#x} destination {:#x} source {:#x}", bytes, ret - ImageBase() + 0x1000000,
                     destination, source);
        }
    }
    const u64 seq = Guarded([&] { return MakeCopyRecord(ctx, destination, source, bytes, ret); });
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (3u << 8);
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    const bool exec = policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
                      (Rd<u8>(ctx, 0x13009) & 1) == 0 && ExecEligibleCompute(ctx, seq, ret);
    if (exec) {
        words[2] |= ExecFlag;
    }
    if (!WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
    if (!exec) {
        return 0;
    }
    Live().compute_exec.fetch_add(1, std::memory_order_relaxed);
    compute_drawn.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

bool ExecEligibleAsc(u64 seq, u64 caller);

/// FUN_02473340(async compute context, destination, source, bytes): the async context's buffer copy, the device's copy shader
/// with both buffers described as read-write buffers. While the record of the call site is not verified the tag precedes the
/// game's packets; then the function is replaced by the tag.
u64 PublishAscMemoryCopy(u64 jcc, u64 destination, u64 source, u64 to, u64 from, u64 bytes, u64 ret);

u64 PS4_SYSV_ABI TagAscCsCopy(u64 jcc, u64 destination, u64 source, u64 bytes_arg, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const u64 bytes = bytes_arg & 0xffffffff;
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || jcc == 0 || Rd<u8>(jcc, 0x1c7c) == 0 || bytes == 0 ||
        bytes > (1ull << 30) || destination == 0 || source == 0 ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_sdk_strip.load(std::memory_order_relaxed) >> 6) & 1) == 0) {
        return 0;
    }
    const auto in_guest_memory = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull; };
    if ((policy.native_sdk_strip.load(std::memory_order_relaxed) & 131072) != 0 && policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
        in_guest_memory(destination) && in_guest_memory(source) && (bytes & 3) == 0) {
        // the device's buffer copy is a copy of memory: the command processor copies it (two images of the same layout as images)
        return PublishAscMemoryCopy(jcc, 0, 0, destination, source, bytes, ret);
    }
    const u64 seq = Guarded([&] { return MakeCopyRecord(jcc, destination, source, bytes, ret, true); });
    if (seq == 0 || Slot(seq).flags != 0) {
        return 0;
    }
    const bool exec = policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligibleAsc(seq, ret);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (3u << 8) | (exec ? ExecFlag : 0u);
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    if (!WriteDwordsAt(jcc + 0x60, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    if (!exec) {
        asc_packet_path.fetch_add(1, std::memory_order_relaxed);
        Live().asc_packet.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    asc_drawn.fetch_add(1, std::memory_order_relaxed);
    Live().asc_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

using GuestFunction6 = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64);
bool StripPackets(u64 begin, u64 end);

/// FUN_02471530(async compute context, counter buffer, unordered access view): hands an append / consume counter back to its
/// buffer. A counter without a GDS slot or with a pending initial value is a CPU write of that value; otherwise the GPU
/// counter in GDS is copied to the buffer's memory (a DMA packet and a scratch label in the game's function): a GDS read-back
/// record (kind 9).
u64 PS4_SYSV_ABI TagAscCounterReadback(u64 jcc, u64 buffer, u64 view, u64 a3, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (jcc == 0 || !valid(buffer) || !valid(view) || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 512) == 0) {
        return 0;
    }
    const u64 memory = ((static_cast<u64>(Rd<u32>(buffer, 0x64)) & 0xfff) << 32) + Rd<u32>(buffer, 0x60);
    if (memory == 0 || (memory & 3) != 0) {
        return 0;
    }
    const u32 slot = Rd<u32>(view, 0x84);
    u32 pending = 0xffffffffu;
    if (slot != 0xffffffffu) {
        if (slot >= 8) {
            return 0;
        }
        pending = Rd<u32>(jcc, 0x2158 + slot * 4);
    }
    if (slot != 0xffffffffu && pending == 0xffffffffu) {
        // the GPU counter: GDS to memory, in stream order
        const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
        if (seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
            return 0;
        }
        auto& record = Slot(seq);
        record.seq.store(0, std::memory_order_release);
        record.check_count = 0;
        record.count = 0;
        record.kind = 9;
        record.flags = 0;
        record.caller = ret;
        record.a1 = 1;
        record.a2 = static_cast<u32>(Rd<u32>(jcc, 0x22f8) + slot * 4ull); // GDS byte offset
        record.args_addr = memory;
        record.seq.store(seq, std::memory_order_release);
        u32 words[2 + VideoCore::NativeTagPayloadDwords];
        words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
        words[1] = VideoCore::NativeTagMagic;
        words[2] = static_cast<u32>(Other) | (9u << 8) | ExecFlag;
        std::memset(&words[3], 0, 3 * sizeof(u64));
        std::memcpy(&words[9], &seq, sizeof(u64));
        words[11] = 0;
        words[12] = 0;
        if (!WriteDwordsAt(jcc + 0x60, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
            return 0;
        }
        tags_written.fetch_add(1, std::memory_order_relaxed);
    } else {
        // CPU write of the counter's value
        u32 value = pending;
        if (slot == 0xffffffffu) {
            value = Rd<u32>(view, 0x88) != 0xffffffffu ? Rd<u32>(view, 0x88) : 0;
        }
        Wr<u32>(memory, 0, value);
    }
    Wr<u32>(buffer, 0x2c, Rd<u32>(jcc, 0x22fc));
    Live().counter_readback.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// FUN_024717c0(async compute context, destination resource, source resource): AsyncCopyResource. The game copies the
/// resource memory with DMA packets (a buffer's bytes, a texture's width x height bytes) and orders it with a scratch label;
/// with execution it is a memory copy record (kind 10). Copies of more than one DMA chunk keep the game's function.
/// Writes the tag of a memory copy record (kind 10: destination, source, bytes) for the async compute context `jcc`.
/// `destination` / `source` are the engine resources to stamp as used (0 when the copy was given plain addresses).
u64 PublishAscMemoryCopy(u64 jcc, u64 destination, u64 source, u64 to, u64 from, u64 bytes, u64 ret) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    if (seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
        return 0;
    }
    auto& record = Slot(seq);
    record.seq.store(0, std::memory_order_release);
    record.check_count = 0;
    record.count = 0;
    record.kind = 10;
    record.flags = 0;
    record.caller = ret;
    record.a1 = static_cast<u32>(bytes);
    record.args_addr = to;
    record.extra_addr = from;
    record.seq.store(seq, std::memory_order_release);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (10u << 8) | ExecFlag;
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    if (!WriteDwordsAt(jcc + 0x60, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    if (destination != 0) {
        Wr<u32>(destination, 0x2c, Rd<u32>(jcc, 0x22fc));
        Wr<u32>(source, 0x2c, Rd<u32>(jcc, 0x22fc));
    }
    Live().async_copy_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

u64 PS4_SYSV_ABI TagAscCopyResource(u64 jcc, u64 destination, u64 source, u64 a3, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (jcc == 0 || !valid(destination) || !valid(source) || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 1024) == 0) {
        return 0;
    }
    const VAddr image = image_base.load(std::memory_order_relaxed);
    if (image == 0) {
        return 0;
    }
    const auto texture_base = reinterpret_cast<u32(PS4_SYSV_ABI*)(u64)>(image + 0x246b850 - 0x1000000);
    const auto address_of = [&](u64 resource) -> u64 {
        if (Rd<u32>(resource, 0x124) == 1) {
            return static_cast<u64>(Rd<u32>(resource, 0x60)) | (static_cast<u64>(Rd<u32>(resource, 0x64)) & 0xfff) << 32;
        }
        return static_cast<u64>(texture_base(resource + 0x70)) << 8;
    };
    u64 bytes;
    if (Rd<u32>(source, 0x124) == 1) {
        const u32 stride = (Rd<u32>(source, 0x64) >> 16) & 0x3fff;
        bytes = static_cast<u64>(stride != 0 ? stride : 1) * Rd<u32>(source, 0x68);
    } else {
        bytes = (static_cast<u64>((Rd<u32>(source, 0x78) >> 14) & 0x3fff) + 1) * (((Rd<u32>(source, 0x80) >> 13) & 0x3fff) + 1);
    }
    bytes = (bytes + 3) & ~3ull;
    const u64 to = address_of(destination);
    const u64 from = address_of(source);
    if (bytes == 0 || bytes >= 0x1ffffc || to == 0 || from == 0) {
        return 0;
    }
    return PublishAscMemoryCopy(jcc, destination, source, to, from, bytes, ret);
}

/// FUN_02480d60(draw context): a CPU visible fence. The function takes eight bytes from the frame ring, stores 0x11 in them, writes
/// an end-of-pipe packet that stores 0x10 there, and returns the address the game polls. The game's function runs (it owns the
/// label); the packet it wrote becomes a fence tag (ConvertSyncPacket).
u64 PS4_SYSV_ABI TagFenceSignal(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static std::atomic<u64> original_entry{0};
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (ctx == 0 || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 2097152) == 0) {
        return 0;
    }
    u64 original = original_entry.load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry("FenceEop");
        original_entry.store(original, std::memory_order_relaxed);
    }
    if (original == 0) {
        return 0;
    }
    const u64 before = Rd<u64>(ctx, 0x47d0);
    const u64 label = reinterpret_cast<GuestFunction6>(original)(ctx, a1, a2, a3, a4, a5);
    const u64 after = Rd<u64>(ctx, 0x47d0);
    if (after == before + 6 * sizeof(u32) && (before & 3) == 0 && ConvertSyncPacket(before) == SyncEop) {
        CountSyncConverted(SyncEop, Live().sync_converted_hook);
    } else {
        Live().fence_packets_kept.fetch_add(1, std::memory_order_relaxed);
    }
    return label | (1ull << 63);
}

/// FUN_024826c0(draw context): the reset at the start of a frame. It sets the engine fields, has the driver write the hardware
/// default state (a template tag with padding) and then writes a few context registers and the index type itself. Those
/// register packets after the padding become one state template tag.
u64 PS4_SYSV_ABI TagDeviceBegin(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static std::atomic<u64> original_entry{0};
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (ctx == 0 || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 16384) == 0) {
        return 0;
    }
    u64 original = original_entry.load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry("DeviceBegin");
        original_entry.store(original, std::memory_order_relaxed);
    }
    if (original == 0) {
        return 0;
    }
    reinterpret_cast<GuestFunction6>(original)(ctx, a1, a2, a3, a4, a5);
    const u64 begin = Rd<u64>(ctx, 0x47c0);
    const u64 end = Rd<u64>(ctx, 0x47d0);
    if (end <= begin || end - begin > 0x4000 || (begin & 3) != 0) {
        return 1;
    }
    // the driver's block: a template tag followed by one large padding packet; everything after it is the reset's own
    u64 tail = 0;
    for (u64 at = begin; at < end;) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            return 1;
        }
        const u32 size = (((header >> 16) & 0x3fff) + 2) * 4;
        at += size;
        if (((header >> 8) & 0xff) == 0x10 && size >= 128 * 4 && Rd<u32>(at - size, 4) != VideoCore::NativeTagMagic) {
            tail = at;
            break;
        }
    }
    constexpr u32 TagDwords = 2 + VideoCore::NativeTagPayloadDwords - 1;
    if (tail == 0 || tail >= end || (end - tail) / 4 < TagDwords + 2) {
        return 1;
    }
    const u32 tail_dwords = static_cast<u32>((end - tail) / 4);
    const StateTemplate* state = IndirectStateTemplate(reinterpret_cast<const u32*>(tail), tail_dwords);
    if (state == nullptr) {
        return 1;
    }
    u32 words[TagDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8);
    words[1] = VideoCore::NativeTagMagic;
    words[2] = (12u << 8) | ExecFlag;
    std::memset(&words[3], 0, (TagDwords - 3) * sizeof(u32));
    const u64 pointer = reinterpret_cast<u64>(state);
    std::memcpy(&words[3], &pointer, sizeof(pointer));
    for (u32 k = 0; k < TagDwords; ++k) {
        Wr<u32>(tail, k * 4, words[k]);
    }
    NoteAnchor(tail, 0, 12);
    const u32 rest = tail_dwords - TagDwords;
    Wr<u32>(tail + TagDwords * 4, 0, 0xC0000000u | ((rest - 2) << 16) | (0x10u << 8));
    Wr<u32>(tail + TagDwords * 4, 4, 0);
    Live().device_begin_stripped.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// JsComputeContext / JsDeviceContext::SetComputeResourceLimits (FUN_024720f0, FUN_02480ed0): one COMPUTE_RESOURCE_LIMITS
/// register write; nothing of the host reads it.
u64 PS4_SYSV_ABI SkipResourceLimits(u64 context, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return context != 0 && policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
                   (policy.native_sdk_strip.load(std::memory_order_relaxed) & 256) != 0
               ? 1
               : 0;
}

/// FUN_0246e8e0(async compute context): the start of a command buffer. The game's function resets the context and writes the
/// cache acquire, the hardware state block and the scratch ring size; those packets are replaced by one NOP.
u64 PS4_SYSV_ABI TagAscBegin(u64 jcc, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    static std::atomic<u64> original_entry{0};
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (jcc == 0 || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 256) == 0) {
        return 0;
    }
    u64 original = original_entry.load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry("AscBegin");
        original_entry.store(original, std::memory_order_relaxed);
    }
    if (original == 0) {
        return 0;
    }
    reinterpret_cast<GuestFunction6>(original)(jcc, a1, a2, a3, a4, a5);
    const u64 begin = Rd<u64>(jcc, 0x60);
    const u64 end = Rd<u64>(jcc, 0x70);
    if (end > begin && end - begin < 0x2000 && (begin & 3) == 0 && StripPackets(begin, end)) {
        Live().asc_begin_stripped.fetch_add(1, std::memory_order_relaxed);
    }
    return 1;
}

/// JsDeviceContext::CopyResource(context, destination, destination subresource, source, source subresource) for two
/// dimensional textures: the tracking of both resources, the switch to compute mode and one dispatch of the copy shader.
/// While the record of the call site is not yet verified the tag is written in front of the game's own packets (they are
/// compared with it); then the whole function is replaced by the tag.
u64 PS4_SYSV_ABI TagCopyResource(u64 ctx, u64 destination, u64 destination_sub, u64 source, u64 source_sub, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_sdk_strip.load(std::memory_order_relaxed) >> 4) & 1) == 0 || ctx == 0 || !valid(destination) ||
        !valid(source) || destination == source) {
        return 0;
    }
    const u32 format = Rd<u32>(destination, 0x190);
    if (Rd<u32>(destination, 0x124) != 3 || Rd<u32>(source, 0x124) != 3 || Rd<u32>(destination, 0x19c) == 1 ||
        (Rd<u8>(destination, 0x1a0) & 0x40) != 0 || format != Rd<u32>(source, 0x190) || format > 0x65 ||
        ((Rd<u32>(destination, 0x194) | Rd<u32>(source, 0x194)) > 1) || Rd<u32>(source, 0x188) == 0) {
        return 0; // the game's function reports the error or handles another resource kind
    }
    if (Rd<u32>(destination, 0x180) >= 1900) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 12) {
            LOG_INFO(Core, "CopyResource of a large texture: caller {:#x} destination {:#x} ({}x{} format {} mips {} sub {}) source {:#x} (sub {})",
                     ret - ImageBase() + 0x1000000, Rd<u64>(destination, 0x30), Rd<u32>(destination, 0x180), Rd<u32>(destination, 0x184), format,
                     Rd<u32>(destination, 0x188), destination_sub, Rd<u64>(source, 0x30), source_sub);
        }
    }
    const u32 current_mode = Rd<u32>(ctx, 0x278);
    if ((current_mode != 0 && current_mode != 0x8000) || (Rd<u8>(ctx, 0x13009) & 1) != 0) {
        return 0;
    }
    const u64 seq = Guarded([&] {
        return MakeCopyResourceRecord(ctx, destination, static_cast<u32>(destination_sub), source, static_cast<u32>(source_sub), ret);
    });
    if (seq == 0) {
        return 0;
    }
    const bool exec = policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligibleCompute(ctx, seq, ret);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (3u << 8) | (exec ? ExecFlag : 0u);
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    if (!exec) {
        if (!WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
            tags_failed.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        tags_written.fetch_add(1, std::memory_order_relaxed);
        Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto game_track = GameTrackFunction();
    TrackResource(ctx, destination, true, game_track);
    TrackResource(ctx, source, true, game_track);
    if (Rd<u32>(ctx, 0x278) != 0x8000) {
        Wr<u32>(ctx, 0x4acc, 0x8000);
        Wr<u8>(ctx, 0x47b4, 1);
        if (Rd<u8>(ctx, 0x47a4) != 0 || Rd<u32>(ctx, 0x130dc) != Rd<u32>(ctx, 0x130d8)) {
            Wr<u32>(ctx, 0x130dc, Rd<u32>(ctx, 0x130d8));
            Wr<u8>(ctx, 0x47a4, 0);
        }
        Wr<u32>(ctx, 0x278, 0x8000);
    }
    if (!WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        Wr<u64>(ctx, 0x20f8, 0);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
    Live().compute_exec.fetch_add(1, std::memory_order_relaxed);
    compute_drawn.fetch_add(1, std::memory_order_relaxed);
    Wr<u64>(ctx, 0x20f8, 0);
    return 1;
}

/// JsDeviceContext::ExCsDepthCopy(context, destination, source): the depth surface and its HTILE surface copied as raw
/// dwords with the device's copy shader, two dispatches (the buffer copy of CsCopyBufferDwords), with event and scratch
/// label fences in between. With execution: two copy records, the fields the function sets at its end, nothing else.
u64 PS4_SYSV_ABI TagExCsDepthCopy(u64 ctx, u64 destination, u64 source, u64 a3, u64 a4, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const auto valid = [](u64 pointer) { return pointer >= 0x200000000ull && pointer < 0x400000000ull && (pointer & 7) == 0; };
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_sdk_strip.load(std::memory_order_relaxed) >> 3) & 1) == 0 || ctx == 0 || !valid(destination) ||
        !valid(source) || destination == source || (Rd<u8>(ctx, 0x13009) & 1) != 0) {
        return 0;
    }
    const std::array<u64, 2> surfaces{0x30, 0x48};
    std::array<u64, 2> seqs{};
    for (u32 i = 0; i < 2; ++i) {
        const u64 bytes = Rd<u64>(destination, surfaces[i] + 8);
        const u64 to = Rd<u64>(destination, surfaces[i]);
        const u64 from = Rd<u64>(source, surfaces[i]);
        if (bytes == 0 || bytes > (1ull << 30) || to == 0 || from == 0) {
            return 0;
        }
        seqs[i] = Guarded([&] { return MakeCopyRecord(ctx, to, from, bytes, ret); });
        if (seqs[i] == 0 || Slot(seqs[i]).flags != 0 || seqs[i] - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
            return 0;
        }
    }
    constexpr u32 TagDwords = 2 + VideoCore::NativeTagPayloadDwords - 1;
    u32 words[2 * TagDwords];
    for (u32 i = 0; i < 2; ++i) {
        u32* tag = &words[i * TagDwords];
        tag[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8);
        tag[1] = VideoCore::NativeTagMagic;
        tag[2] = static_cast<u32>(Other) | (3u << 8) | ExecFlag;
        std::memset(&tag[3], 0, 3 * sizeof(u64));
        std::memcpy(&tag[9], &seqs[i], sizeof(u64));
        tag[11] = 0;
        tag[12] = 0;
    }
    if (!WriteDwords(ctx, words, 2 * TagDwords)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(2, std::memory_order_relaxed);
    Live().compute_tags.fetch_add(2, std::memory_order_relaxed);
    Live().compute_exec.fetch_add(2, std::memory_order_relaxed);
    compute_drawn.fetch_add(2, std::memory_order_relaxed);
    // what the function leaves behind: the compute mode, three cleared engine caches and the destination's state copied
    // from the source
    Wr<u64>(ctx, 0x20f8, 0);
    Wr<u64>(ctx, 0x2478, 0);
    Wr<u64>(ctx, 0x43f8, 0);
    Wr<u32>(ctx, 0x278, 0x8000);
    Wr<u32>(destination, 0x144, Rd<u32>(source, 0x144));
    Wr<u32>(destination, 0x158, Rd<u32>(source, 0x158));
    Wr<u8>(destination, 0x15c, Rd<u8>(source, 0x15c));
    Wr<u32>(destination, 0x178, Rd<u32>(source, 0x178));
    Wr<u32>(destination, 0x17c, Rd<u32>(source, 0x17c));
    return 1;
}

u64 PS4_SYSV_ABI TagCsCopyHandler(u64 writer, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return TagCsCopy(writer, a1, a2, a3 & 0xffffffff, ret);
}
bool ExecEligibleAsc(u64 seq, u64 caller) {
    if (Slot(seq).flags != 0 || seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
        return false;
    }
    CallerTrust* trust = TrustFor(caller);
    if (trust == nullptr) {
        return false;
    }
    const u32 total = trust->total.load(std::memory_order_relaxed);
    if (total < 16 || trust->bad.load(std::memory_order_relaxed) * 500 >= total) {
        return false;
    }
    const u32 period = AuditPeriod();
    return period == 0 || (trust->since_audit.fetch_add(1, std::memory_order_relaxed) + 1) % period != 0;
}

/// What the game's PrepareDispatch of the compute context and the constant update engine flush leave in the engine's
/// objects besides command packets: last-use stamps, the two fields of a buffer view the engine rewrites, the append /
/// consume counter range and the constant buffer cache.
void AscSideEffects(u64 jcc, u64 x, u64 y, u64 z) {
    const u64 cs_obj = Rd<u64>(jcc, 0x1d10);
    const u64 offs = cs_obj + 0x130;
    const u32 stamp = Rd<u32>(jcc, 0x22fc);
    Wr<u32>(cs_obj, 0x2c, stamp);
    for (u32 mask = Rd<u32>(cs_obj, 0xe0), index = 0; mask != 0; mask >>= 1, ++index) {
        if ((mask & 1) != 0) {
            const u64 buffer = Rd<u64>(jcc, 0x21f8 + index * 8);
            Wr<u64>(jcc, 0x2278 + index * 8, Rd<u64>(buffer, 0x160));
            Wr<u32>(buffer, 0x2c, stamp);
        }
    }
    for (u32 mask = Rd<u32>(cs_obj, 0xf0), index = 0; mask != 0; mask >>= 1, ++index) {
        if ((mask & 1) != 0) {
            const u64 view = Rd<u64>(jcc, 0x1d18 + index * 8);
            if (const u32 kind = Rd<u32>(view, 0x44); kind == 0xb || kind == 1) {
                Wr<u32>(view, 0x84, (Rd<u32>(view, 0x84) & 0xc7ffffffu) | 0x18000000u);
                Wr<u32>(view, 0x7c, (Rd<u32>(view, 0x7c) & 0xffff0fffu) | 0xb000u);
            }
            Wr<u32>(view, 0x2c, stamp);
            Wr<u32>(Rd<u64>(view, 0x30), 0x2c, stamp);
        }
    }
    for (u32 mask = Rd<u32>(cs_obj, 0x100), index = 0; mask != 0; mask >>= 1, ++index) {
        if ((mask & 1) != 0) {
            const u64 view = Rd<u64>(jcc, 0x2118 + index * 8);
            if (Rd<u32>(view, 0x44) == 1 && (Rd<u8>(view, 0x50) & 2) != 0) {
                Wr<u32>(jcc, 0x2158 + index * 4, 0xffffffffu);
                Wr<u32>(view, 0x84, index);
                if (Rd<u8>(offs, 0xd) != 0xff) {
                    Wr<u32>(jcc, 0x1940, Rd<u32>(jcc, 0x22f8) << 16 | 0x10);
                }
            }
            Wr<u32>(Rd<u64>(view, 0x30), 0x2c, stamp);
        }
    }
    Wr<u32>(jcc, 0x34, Rd<u32>(jcc, 0x34) + 1);
    Wr<u32>(jcc, 0x3c, Rd<u32>(jcc, 0x3c) + static_cast<u32>(x) * static_cast<u32>(y) * static_cast<u32>(z));
}

/// Dispatch of the async compute context: PrepareDispatch, the constant update engine flush and the dispatch packet.
/// With execution the record replaces all of it (the packets are only written when the game's own path runs).
u64 TagAsc(u64 jcc, u64 x, u64 y, u64 z, u64 ret, bool allow_exec = true) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || jcc == 0 || x == 0 || y == 0 || z == 0 ||
        policy.native_synth_verify.load(std::memory_order_relaxed) == 0 ||
        ((policy.native_draw_classes.load(std::memory_order_relaxed) >> Other) & 1) == 0) {
        return 0;
    }
    const u64 seq = Guarded([&] { return MakeAscRecord(jcc, x, y, z, ret); });
    if (seq == 0) {
        {
            static std::mutex mutex;
            static std::map<std::pair<u64, u32>, u64> reasons; ///< caller, source line -> dispatches
            static u64 last_report_ms = 0;
            std::lock_guard lock{mutex};
            ++reasons[{ret - ImageBase() + 0x1000000, asc_fail}];
            const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - last_report_ms >= 5000) {
                last_report_ms = now;
                for (const auto& [key, count] : reasons) {
                    LOG_INFO(Core, "async dispatch without record per 5 s: caller {:#x} check at line {}: {} dispatches", key.first, key.second, count);
                }
                reasons.clear();
            }
        }
        asc_no_record.fetch_add(1, std::memory_order_relaxed);
        Live().asc_no_record.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    constexpr u32 TagDwords = 2 + VideoCore::NativeTagPayloadDwords - 1;
    std::array<u32, TagDwords * 9> words;
    const auto write_tag = [&](u32* tag, u32 kind, u64 tag_seq) {
        tag[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
        tag[1] = VideoCore::NativeTagMagic;
        tag[2] = static_cast<u32>(Other) | (kind << 8);
        std::memset(&tag[3], 0, 3 * sizeof(u64));
        std::memcpy(&tag[9], &tag_seq, sizeof(u64));
        tag[11] = 0;
        tag[12] = 0;
    };
    const bool exec = allow_exec && policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligibleAsc(seq, ret);
    u32 word_count = 0;
    if (exec) {
        // counters the game initializes with DMA packets: GDS fill records, in front of the dispatch
        const u32 counter_base = Rd<u32>(jcc, 0x22f8);
        for (u32 k = 0; k < asc_counter_init_count; ++k) {
            const u64 fill_seq = next_seq.fetch_add(1, std::memory_order_relaxed);
            auto& fill = Slot(fill_seq);
            fill.seq.store(0, std::memory_order_release);
            fill.check_count = 0;
            fill.count = 0;
            fill.kind = 8;
            fill.flags = 0;
            fill.caller = ret;
            fill.a1 = 1;
            fill.a2 = asc_counter_inits[k].value;
            fill.args_addr = counter_base + asc_counter_inits[k].index * 4ull;
            fill.seq.store(fill_seq, std::memory_order_release);
            write_tag(&words[word_count], 8, fill_seq);
            words[word_count + 2] |= ExecFlag;
            word_count += TagDwords;
        }
    }
    write_tag(&words[word_count], 3, seq);
    if (exec) {
        words[word_count + 2] |= ExecFlag;
    }
    word_count += TagDwords;
    if (!WriteDwordsAt(jcc + 0x60, words.data(), word_count)) {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    if (!exec) {
        asc_packet_path.fetch_add(1, std::memory_order_relaxed);
        Live().asc_packet.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    {
        static std::atomic<u32> exec_log{0};
        if (const u32 n = exec_log.fetch_add(1); n < 1500 && (n < 40 || ret - ImageBase() + 0x1000000 != 0x2471253)) {
            const u64 cs_obj = Rd<u64>(jcc, 0x1d10);
            LOG_INFO(Core, "asc exec #{}: site {:#x} jcc {:#x} shader {:#x} groups {}x{}x{} masks s{:#x} cb{:#x} srv{:#x} uav{:#x}", n,
                     ret - ImageBase() + 0x1000000, jcc, Rd<u64>(cs_obj, 0x120), x, y, z, Rd<u32>(cs_obj, 0xe4), Rd<u32>(cs_obj, 0xe0),
                     Rd<u32>(cs_obj, 0xf0), Rd<u32>(cs_obj, 0x100));
        }
    }
    AscSideEffects(jcc, x, y, z);
    asc_drawn.fetch_add(1, std::memory_order_relaxed);
    Live().asc_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// True when the `size` dwords at `source` are all the same (a fill with a pattern of equal dwords is a fill with one value).
bool SourceDwordsEqual(u64 source, u32 size) {
    if (source < 0x200000 || size == 0 || size > 64) {
        return false;
    }
    const u32 first = Rd<u32>(source, 0);
    for (u32 k = 1; k < size; ++k) {
        if (Rd<u32>(source, k * 4) != first) {
            return false;
        }
    }
    return true;
}

/// FUN_02485c00(context, destination, dwords, source, source dwords): the SDK's compute copy with wrap, a shader that
/// writes destination[i] = source[i mod size]. Every call of the game has a source of one dword: a buffer fill. With
/// execution it is a fill operation of the command processor (record kind 6); any other size keeps the game's path.
u64 PS4_SYSV_ABI TagFill(u64 ctx, u64 destination, u64 count_arg, u64 source, u64 size_arg, u64 a5, u64 ret) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    const u32 count = static_cast<u32>(count_arg);
    const u32 size = static_cast<u32>(size_arg);
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        ctx == 0 || size == 0 || size > 64 || count == 0 || count > (64u << 20) / 4 || source == 0 || destination == 0 ||
        (Rd<u8>(ctx, 0xe849) & 1) != 0 || (destination & 3) != 0 || !SourceDwordsEqual(source, size)) {
        if (ctx != 0 && size >= 2 && size <= 64 && count != 0 && count <= (64u << 20) / 4 && source != 0 && destination != 0 &&
            (destination & 3) == 0 && (Rd<u8>(ctx, 0xe849) & 1) == 0 && policy.native_draw_mode.load(std::memory_order_relaxed) != 0 &&
            policy.native_synth_verify.load(std::memory_order_relaxed) != 0 &&
            ((policy.native_sdk_strip.load(std::memory_order_relaxed) >> 12) & 1) != 0) {
            const u64 writer = ctx;
            if (const u64 seq = Guarded([&] { return MakeFillPatternRecord(writer, destination, count, source, size, ret); });
                seq != 0) {
                u32 words[2 + VideoCore::NativeTagPayloadDwords];
                words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
                words[1] = VideoCore::NativeTagMagic;
                words[2] = static_cast<u32>(Other) | (3u << 8);
                std::memset(&words[3], 0, 3 * sizeof(u64));
                std::memcpy(&words[9], &seq, sizeof(u64));
                words[11] = 0;
                words[12] = 0;
                const bool exec = policy.native_draw_exec.load(std::memory_order_relaxed) != 0 && ExecEligibleCompute(writer - 0x47c0, seq, ret);
                if (exec) {
                    words[2] |= ExecFlag;
                }
                if (WriteDwordsAt(writer, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
                    tags_written.fetch_add(1, std::memory_order_relaxed);
                    Live().compute_tags.fetch_add(1, std::memory_order_relaxed);
                    if (exec) {
                        Live().compute_exec.fetch_add(1, std::memory_order_relaxed);
                        compute_drawn.fetch_add(1, std::memory_order_relaxed);
                        return 1;
                    }
                    return 0;
                }
            }
        }
        if (ctx != 0) {
            static std::mutex mutex;
            static std::map<std::tuple<u32, u32, u32, u64>, u32> shapes; ///< count, size, flag, caller -> calls
            static u64 last_ms = 0;
            std::lock_guard lock{mutex};
            ++shapes[{count, size, static_cast<u32>(Rd<u8>(ctx, 0xe849) & 1), ret - ImageBase() + 0x1000000}];
            if (source >= 0x200000 && size >= 2) {
                static std::atomic<u32> value_logs{0};
                if (value_logs.fetch_add(1) < 6) {
                    LOG_INFO(Core, "fill/copy source of {} dwords: {:#x} {:#x}", size, Rd<u32>(source, 0), Rd<u32>(source, 4));
                }
            }
            const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            if (now - last_ms >= 5000) {
                last_ms = now;
                for (const auto& [key, calls] : shapes) {
                    LOG_INFO(Core, "fill/copy on the packet path per 5 s: dwords {} source dwords {} post-draw flag {} caller {:#x}: {} calls",
                             std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), calls);
                }
                shapes.clear();
            }
        }
        return 0;
    }
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.count = 0;
    slot.kind = 6;
    slot.flags = 0;
    slot.caller = ret;
    slot.a1 = count;
    slot.a2 = Rd<u32>(source, 0); // the value the shader would read
    slot.args_addr = destination;
    slot.seq.store(seq, std::memory_order_release);
    u32 words[2 + VideoCore::NativeTagPayloadDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8); // PM4 NOP
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (6u << 8) | ExecFlag;
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    if (!WriteDwordsAt(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        return 0;
    }
    tags_written.fetch_add(1, std::memory_order_relaxed);
    Live().fill_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// FUN_0247b920(context, mode): the draw context's mode switch between graphics (0) and compute (0x8000) work. The game's
/// function writes a barrier (an end-of-shader event, a wait on a scratch label, a cache flush), three context registers that
/// keep their default value (0) in these two modes, and sets the engine fields below. The raster thread runs in order, so with
/// execution only the fields are set. The other modes (dispatch draw, tessellation and geometry stage sets) are never used by
/// the game; the call then runs the game's own function.
u64 PS4_SYSV_ABI TagModeSwitch(u64 ctx, u64 mode, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx == 0 || VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    const u32 current = Rd<u32>(ctx, 0x278);
    const u32 wanted = static_cast<u32>(mode);
    if (current == wanted) {
        return 1; // the game's function returns at once
    }
    if ((current != 0 && current != 0x8000) || (wanted != 0 && wanted != 0x8000)) {
        static std::mutex mutex;
        static std::map<std::tuple<u32, u32, u64>, u32> refused; ///< current, wanted, caller -> calls
        static u64 last_ms = 0;
        std::lock_guard lock{mutex};
        ++refused[{current, wanted, a5 == 0 ? ret - ImageBase() + 0x1000000 : 0}];
        const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - last_ms >= 5000) {
            last_ms = now;
            for (const auto& [key, calls] : refused) {
                LOG_INFO(Core, "mode switch left to the game per 5 s: from {:#x} to {:#x} caller {:#x}: {} calls", std::get<0>(key), std::get<1>(key),
                         std::get<2>(key), calls);
            }
            refused.clear();
        }
        return 0;
    }
    if (Rd<u8>(ctx, 0x47a4) != 0 || Rd<u32>(ctx, 0x130dc) != Rd<u32>(ctx, 0x130d8)) {
        Wr<u32>(ctx, 0x130dc, Rd<u32>(ctx, 0x130d8));
        Wr<u8>(ctx, 0x47a4, 0);
    }
    Wr<u32>(ctx, 0x4acc, wanted);
    Wr<u8>(ctx, 0x47b4, 1);
    Wr<u32>(ctx, 0x278, wanted);
    Live().mode_switch_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// FUN_024712d0(async compute context): a barrier inside the queue. It writes a scratch label, releases it at the end of
/// the pipe, waits for it and rewrites it; the label is local to the function. The raster thread runs the queue's work in
/// order, so with execution the barrier is not needed.
u64 PS4_SYSV_ABI TagAscBarrier(u64 jcc, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (jcc == 0 || VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    Live().asc_barrier_exec.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

/// SDK helpers that write packets of their own into the draw context's command buffer: the handler runs the game's
/// function and looks at what it wrote (cursor before and after).
struct EmitStat {
    const char* name;
    std::atomic<u64> calls{0}, dwords{0}, packets{0}, draws{0}, split{0}, stripped{0};
};
EmitStat emit_stats[6] = {{"ResTransition"}, {"RenderTargetEliminate"}, {"ClearQuad"}, {"CopyResource"}, {"ExCsDepthCopy"}, {"HtileClear"}};

void ReportEmitStats() {
    static std::atomic<u64> last{0};
    const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    u64 expected = last.load();
    if (now - expected < 5000 || !last.compare_exchange_strong(expected, now)) {
        return;
    }
    for (auto& stat : emit_stats) {
        if (const u64 calls = stat.calls.exchange(0)) {
            LOG_INFO(Core, "sdk helper packets per 5 s: {}: {} calls, {} dwords, {} packets, {} draws, {} left a buffer, {} stripped",
                     stat.name, calls, stat.dwords.exchange(0), stat.packets.exchange(0), stat.draws.exchange(0),
                     stat.split.exchange(0), stat.stripped.exchange(0));
        }
    }
}


/// Replaces the packets a helper wrote by NOP packets, packet by packet merged into runs: the memory stays as it was (tables
/// and labels the engine still points to are untouched), only the first two dwords of a run change. Our own tag packets stay.
bool StripPackets(u64 begin, u64 end) {
    u64 run_start = 0;
    const auto flush = [&](u64 run_end) {
        if (run_start == 0) {
            return;
        }
        const u64 dwords = (run_end - run_start) / 4;
        if (dwords >= 2 && dwords - 2 < 0x4000) {
            Wr<u32>(run_start, 0, 0xC0000000u | (static_cast<u32>(dwords - 2) << 16) | (0x10u << 8));
            if (dwords >= 2) {
                Wr<u32>(run_start, 4, 0);
            }
        }
        run_start = 0;
    };
    for (u64 at = begin; at < end;) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            return false;
        }
        const u64 size = (((header >> 16) & 0x3fff) + 2) * 4ull;
        const bool is_tag = ((header >> 8) & 0xff) == 0x10 && ((header >> 16) & 0x3fff) != 0 &&
                            Rd<u32>(at, 4) == static_cast<u32>(VideoCore::NativeTagMagic);
        if (is_tag) {
            flush(at);
        } else if (run_start == 0) {
            run_start = at;
        }
        at += size;
    }
    flush(end);
    return true;
}

/// `dcb` is the address of the command buffer structure (end at +8, cursor at +0x10) the function writes to.
u64 RunAndMeasure(EmitStat& stat, const char* hook_name, u64 dcb, u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5,
                  u32 strip_bit = 0, u64 strip_from = 0) {
    static std::atomic<u64> originals[8]{};
    const u64 index = &stat - emit_stats;
    u64 original = originals[index].load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry(hook_name);
        originals[index].store(original, std::memory_order_relaxed);
    }
    if (original == 0 || dcb == 0) {
        return 0;
    }
    const u64 end_before = Rd<u64>(dcb, 8);
    const u64 cursor_before = Rd<u64>(dcb, 0x10);
    reinterpret_cast<GuestFunction6>(original)(a0, a1, a2, a3, a4, a5);
    stat.calls.fetch_add(1, std::memory_order_relaxed);
    const u64 end_after = Rd<u64>(dcb, 8);
    const u64 cursor_after = Rd<u64>(dcb, 0x10);
    if (end_after != end_before || cursor_after < cursor_before || cursor_after - cursor_before > (1u << 20)) {
        stat.split.fetch_add(1, std::memory_order_relaxed);
    } else {
        stat.dwords.fetch_add((cursor_after - cursor_before) / 4, std::memory_order_relaxed);
        u64 draws = 0, packets = 0, at = cursor_before;
        while (at < cursor_after) {
            const u32 header = Rd<u32>(at, 0);
            if ((header >> 30) != 3) {
                break;
            }
            ++packets;
            const u32 opcode = (header >> 8) & 0xff;
            draws += (opcode == 0x2d || opcode == 0x27 || opcode == 0x2c || opcode == 0x15);
            at += ((header >> 16) & 0x3fff) * 4 + 8;
        }
        stat.packets.fetch_add(packets, std::memory_order_relaxed);
        stat.draws.fetch_add(draws, std::memory_order_relaxed);
        if (strip_bit != 0 && (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & strip_bit) != 0 &&
            VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0 && packets != 0 &&
            at == cursor_after && StripPackets(strip_from != 0 ? strip_from : cursor_before, cursor_after)) {
            stat.stripped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ReportEmitStats();
    return 1;
}

/// FUN_02479db0(context, resource): the game's per-resource tracking before a resource is read or written (called from
/// draws, dispatches and copies). With execution it is the native tracking; resources with a depth transition or a second
/// resource keep the game's function (whose depth quads are stripped, see MeasureTargetEliminate).
u64 PS4_SYSV_ABI TagTrackResource(u64 ctx, u64 resource, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx == 0 || resource < 0x200000000ull || resource >= 0x400000000ull || (resource & 7) != 0 ||
        VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 2) == 0) {
        return 0;
    }
    const auto game = GameTrackFunction();
    if (reinterpret_cast<u64>(game) == image_base.load(std::memory_order_relaxed) + 0x2479db0 - 0x1000000) {
        return 0; // the original entry is not available
    }
    TrackResource(ctx, resource, true, game);
    return 1;
}
u64 PS4_SYSV_ABI MeasureTargetEliminate(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return ctx ? RunAndMeasure(emit_stats[1], "Fn247c680", ctx + 0x47c0, ctx, a1, a2, a3, a4, a5, 1) : 0;
}
/// Where the depth / stencil clear of the render layer (FUN_024860c0) began writing packets: the register setup it writes
/// before the SDK helpers it calls (clear quad, HTILE clear) is part of the clear.
struct ClearPrelude {
    u64 dcb{};
    u64 begin{};
};
thread_local ClearPrelude clear_prelude;

u64 PS4_SYSV_ABI NoteClearDsv(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx != 0) {
        clear_prelude = {ctx + 0x47c0, Rd<u64>(ctx, 0x47d0)};
    }
    return 0;
}

/// The start of the clear's own packets for a helper that begins writing at `begin` in `dcb` (== `begin` when the helper is not
/// part of a clear or the packets in between are anything but register state). `values` receives the last value written to
/// the clear registers (render control, stencil / depth clear value, depth control) in the prelude.
u64 TakeClearPrelude(u64 dcb, u64 begin, std::array<std::pair<u32, u32>, 4>* values = nullptr) {
    const ClearPrelude prelude = std::exchange(clear_prelude, {});
    if (prelude.dcb != dcb || prelude.begin == 0 || prelude.begin > begin || begin - prelude.begin > 4096 || (prelude.begin & 3) != 0) {
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1) < 8) {
            LOG_INFO(Core, "clear prelude not used: noted dcb {:#x} begin {:#x}, helper dcb {:#x} begin {:#x}", prelude.dcb, prelude.begin, dcb, begin);
        }
        return begin;
    }
    std::array<std::pair<u32, u32>, 4> found{{{0x0, ~0u}, {0xa, ~0u}, {0xb, ~0u}, {0x200, ~0u}}};
    for (u64 at = prelude.begin; at < begin;) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            return begin;
        }
        const u32 opcode = (header >> 8) & 0xff;
        const u32 size = (((header >> 16) & 0x3fff) + 2) * 4;
        if (opcode == 0x69) {
            const u32 offset = Rd<u32>(at, 4);
            for (u32 k = 0; k + 2 < size / 4; ++k) {
                for (auto& entry : found) {
                    if (entry.first == offset + k) {
                        entry.second = Rd<u32>(at, 8 + k * 4);
                    }
                }
            }
        } else if (opcode == 0x10) {
            if (size > 8 && (Rd<u32>(at, 4) == VideoCore::NativeTagMagic)) {
                return begin; // one of our own records
            }
        } else if (opcode != 0x2f && opcode != 0x28 && opcode != 0x58 && opcode != 0x46) {
            static std::atomic<u32> logged{0};
            if (logged.fetch_add(1) < 8) {
                LOG_INFO(Core, "clear prelude has packet {:#x} ({} dwords) at +{} dwords: not folded", opcode, size / 4, (at - prelude.begin) / 4);
            }
            return begin;
        }
        at += size;
    }
    if (values != nullptr) {
        *values = found;
    }
    return prelude.begin;
}

/// The SDK's depth / stencil clear quad (FUN_024879c0(command buffer, depth target)). The game's function runs; its
/// packets (state, embedded vertex shader, the quad) are replaced by one depth clear record carrying the target's
/// registers, except for the two register writes that end the function (color target mask, depth render control).
bool ConvertClearQuad(u64 target, u64 start, u64 begin, u64 end, u64 caller, const std::array<std::pair<u32, u32>, 4>& prelude_values) {
    constexpr u32 TagDwords = 2 + VideoCore::NativeTagPayloadDwords - 1;
    const u64 total = (end - start) / 4;
    if (total < TagDwords + 6 || (Rd<u32>(target, 0) & 3) == 0) {
        return false;
    }
    u64 tag_end_is_boundary = 0;
    u64 last_two[2] = {0, 0};
    bool target_registers = false;
    u64 at = start;
    while (at < end) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            return false;
        }
        if (at == start + TagDwords * 4) {
            tag_end_is_boundary = 1;
        }
        if (header == 0xc0086900u && Rd<u32>(at, 4) == 0x10 && at >= begin) {
            target_registers = true;
        }
        last_two[0] = last_two[1];
        last_two[1] = at;
        at += (((header >> 16) & 0x3fff) + 2) * 4ull;
    }
    (void)tag_end_is_boundary;
    if (at != end || !target_registers || last_two[0] == 0 ||
        Rd<u32>(last_two[0], 0) != 0xc0016900u || Rd<u32>(last_two[0], 4) != 0x8e || Rd<u32>(last_two[0], 8) != 0xf ||
        Rd<u32>(last_two[1], 0) != 0xc0016900u || Rd<u32>(last_two[1], 4) != 0 || last_two[0] + 12 != last_two[1] ||
        last_two[1] + 12 != end || last_two[0] < start + TagDwords * 4) {
        return false;
    }
    // the two tail writes (color target mask, depth render control) are applied by the record itself
    const u64 middle = (end - start) / 4 - TagDwords;
    if (middle == 1) {
        return false;
    }
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    if (seq - consumed_seq.load(std::memory_order_relaxed) > RecordSlots - 4096) {
        return false;
    }
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.kind = 7;
    slot.flags = 0;
    slot.a1 = 3;
    slot.caller = caller;
    u32 n = 0;
    for (u32 k = 0; k < 8; ++k) {
        slot.regs[n++] = {ContextBase + 0x10 + k, Rd<u32>(target, k * 4)};
    }
    slot.regs[n++] = {ContextBase + 0xf, Rd<u32>(target, 0x2c)};
    slot.regs[n++] = {ContextBase + 0x2, Rd<u32>(target, 0x20)};
    slot.regs[n++] = {ContextBase + 0x5, Rd<u32>(target, 0x24)};
    slot.regs[n++] = {ContextBase + 0x2af, Rd<u32>(target, 0x28)};
    for (const auto& [reg, value] : prelude_values) {
        if (value != ~0u) {
            slot.regs[n++] = {ContextBase + reg, value}; // what the clear's setup wrote for the clear
        }
    }
    slot.count = n;
    slot.cb_extent = {};
    slot.cb_extent_mask = 0;
    slot.db_extent = Rd<u32>(target, 0x30);
    slot.seq.store(seq, std::memory_order_release);
    u32 words[TagDwords];
    words[0] = 0xC0000000u | ((VideoCore::NativeTagPayloadDwords - 1) << 16) | (0x10u << 8);
    words[1] = VideoCore::NativeTagMagic;
    words[2] = static_cast<u32>(Other) | (7u << 8) | ExecFlag;
    std::memset(&words[3], 0, 3 * sizeof(u64));
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
    for (u32 k = 0; k < TagDwords; ++k) {
        Wr<u32>(start, k * 4, words[k]);
    }
    NoteAnchor(start, seq, 7);
    if (middle >= 2) {
        const u64 run = start + TagDwords * 4;
        Wr<u32>(run, 0, 0xC0000000u | (static_cast<u32>(middle - 2) << 16) | (0x10u << 8));
        Wr<u32>(run, 4, 0);
    }
    return true;
}

u64 PS4_SYSV_ABI MeasureClearQuad(u64 dcb, u64 target, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (dcb == 0) {
        return 0;
    }
    const u64 begin = Rd<u64>(dcb, 0x10);
    const u64 end_marker = Rd<u64>(dcb, 8);
    std::array<std::pair<u32, u32>, 4> prelude_values{{{0x0, ~0u}, {0xa, ~0u}, {0xb, ~0u}, {0x200, ~0u}}};
    const u64 start = TakeClearPrelude(dcb, begin, &prelude_values);
    const u64 result = RunAndMeasure(emit_stats[2], "ClearQuad", dcb, dcb, target, a2, a3, a4, a5);
    const u64 end = Rd<u64>(dcb, 0x10);
    if (result != 0 && Rd<u64>(dcb, 8) == end_marker && end > begin && target >= 0x200000000ull && target < 0x400000000ull &&
        (VideoCore::ReadbackPolicy::Get().native_sdk_strip.load(std::memory_order_relaxed) & 32) != 0 &&
        VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0 &&
        (ConvertClearQuad(target, start, begin, end, ret, prelude_values) ||
         (start != begin && ConvertClearQuad(target, begin, begin, end, ret, {{{0x0, ~0u}, {0xa, ~0u}, {0xb, ~0u}, {0x200, ~0u}}})))) {
        emit_stats[2].stripped.fetch_add(1, std::memory_order_relaxed);
        Live().clear_exec.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

/// FUN_02487e40(command buffer, depth target, value): the clear of a depth target's HTILE surface. The fill itself is a native
/// record (FUN_02485c00); the event and the scratch label fence around it are of no use to the command processor.
u64 PS4_SYSV_ABI MeasureHtileClear(u64 dcb, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (dcb == 0) {
        return 0;
    }
    const u64 begin = Rd<u64>(dcb, 0x10);
    const u64 start = TakeClearPrelude(dcb, begin);
    return RunAndMeasure(emit_stats[5], "HtileClear", dcb, dcb, a1, a2, a3, a4, a5, 4, start);
}
u64 PS4_SYSV_ABI MeasureCopyResource(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx == 0) {
        return 0;
    }
    static std::mutex mutex;
    static std::map<std::tuple<u32, u32, u64>, u32> shapes; ///< dimension, format, caller -> calls
    static std::atomic<u64> last_report{0};
    if (a1 >= 0x200000000ull && a1 < 0x400000000ull) {
        std::lock_guard lock{mutex};
        const u32 dimension = Rd<u32>(a1, 0x124);
        ++shapes[{dimension, dimension == 2 ? Rd<u32>(a1, 0x18c) : (dimension == 3 || dimension == 4) ? Rd<u32>(a1, 0x190) : 0,
                  ret - ImageBase() + 0x1000000}];
        const u64 now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now - last_report.load() >= 5000) {
            last_report.store(now);
            for (const auto& [key, calls] : shapes) {
                LOG_INFO(Core, "CopyResource per 5 s: dimension {} format {} caller {:#x}: {} calls", std::get<0>(key), std::get<1>(key),
                         std::get<2>(key), calls);
            }
            shapes.clear();
        }
    }
    return RunAndMeasure(emit_stats[3], "CopyResource", ctx + 0x47c0, ctx, a1, a2, a3, a4, a5);
}
u64 PS4_SYSV_ABI MeasureExCsDepthCopy(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return ctx ? RunAndMeasure(emit_stats[4], "ExCsDepthCopy", ctx + 0x47c0, ctx, a1, a2, a3, a4, a5) : 0;
}

u64 PS4_SYSV_ABI TagAscDispatch(u64 jcc, u64 x, u64 y, u64 z, u64 a4, u64 a5, u64 ret) {
    return TagAsc(jcc, x & 0xffffffff, y & 0xffffffff, z & 0xffffffff, ret);
}
u64 PS4_SYSV_ABI TagAscDispatchRecordOnly(u64 jcc, u64 x, u64 y, u64 z, u64 a4, u64 a5, u64 ret) {
    return TagAsc(jcc, x & 0xffffffff, y & 0xffffffff, z & 0xffffffff, ret, false);
}

u64 PS4_SYSV_ABI TagDispatch(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return TagCompute(ctx, a1, a2, a3, ret, true);
}
/// The fenced dispatch wrapper adds an in-stream scratch label that the dispatch writes at its end and a wait on it: a
/// command processor stall that only orders the dispatch against later commands. The raster thread runs everything in
/// order, so with execution the wait and its label are not needed (game-visible fences are other packets).
u64 PS4_SYSV_ABI TagDispatchFenced(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return TagCompute(ctx, a1, a2, a3, ret, true);
}
u64 PS4_SYSV_ABI TagIndirect(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Tag(ctx, a1, a2, a3, ret, 2, true);
}
/// PrepareDraw(ctx, 0, 1): its callers are the draw functions (hooked at their own entry) and one inlined copy of the auto
/// draw in a render pass function, which is where this hook acts.
u64 PS4_SYSV_ABI PrepareDrawHook(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret, u64 caller_rsp) {
    const VAddr base = image_base.load(std::memory_order_relaxed);
    if (ctx != 0 && base != 0 && ret - base + 0x1000000 == 0x1c727a7 &&
        VideoCore::ReadbackPolicy::Get().native_draw_exec.load(std::memory_order_relaxed) != 0) {
        // the vertex count sits in the caller's frame ([rsp + 0x108], stored just before the call)
        const u64 count = Rd<u32>(caller_rsp, 0x108);
        if (count != 0 && count < 0x1000000) {
            inline_instances = Rd<u32>(ctx, 0x4724);
            const u64 result = Tag(ctx, count, 0, 0, ret, 1, true, true);
            inline_instances = 0;
            if (result != 0) {
                return result;
            }
        }
    }
    return PrepareNote(ctx, a1, a2, a3, a4, a5, ret);
}

/// The constant update engine's flush before a draw packet (CUE + 0x0 as argument).
u64 PS4_SYSV_ABI CueFlushHook(u64 cue, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (skip_cue_flush) {
        skip_cue_flush = false;
        return 1;
    }
    return 0;
}

u64 PS4_SYSV_ABI TagAuto(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Tag(ctx, a1, a2, a3, ret, 1, true);
}

u64 PS4_SYSV_ABI ProbeIndexed(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 0);
}
u64 PS4_SYSV_ABI ProbeAuto(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 1);
}
u64 PS4_SYSV_ABI ProbeIndirect(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Probe(ctx, a1, a2, a3, ret, 2);
}

} // namespace

/// JsDevice::CreateBuffer(device, description, initial data, out): the buffer is complete when the original returns.
u64 PS4_SYSV_ABI ResourceBufferCreated(u64 device, u64 description, u64 initial, u64 out, u64 a4, u64 a5, u64 ret) {
    static std::atomic<u64> original_entry{0};
    u64 original = original_entry.load(std::memory_order_relaxed);
    if (original == 0) {
        original = Core::GuestHooks::OriginalEntry("ResourceBufferCreate");
        original_entry.store(original, std::memory_order_relaxed);
    }
    if (original == 0) {
        return 0;
    }
    reinterpret_cast<GuestFunction6>(original)(device, description, initial, out, a4, a5);
    if (out != 0) {
        NativeResources::NoteBufferObject(Guarded([&] { return Rd<u64>(out, 0); }));
    }
    return 1;
}

/// JsTexture initialization (this, description, initial data): the object is read later.
u64 PS4_SYSV_ABI ResourceTextureInit(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    NativeResources::NoteTextureObject(object);
    return 0;
}

u64 PS4_SYSV_ABI ResourceBufferDestroyed(u64 object, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    NativeResources::NoteBufferDestroyed(object);
    return 0;
}

/// FUN_0247e510(device context): the scratch barrier unit as a function (see SweepSyncPackets): nothing to write.
u64 PS4_SYSV_ABI SkipWaitComputeDone(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    if (ctx == 0 || !SweepEnabled()) {
        return 0;
    }
    Live().barrier_calls_skipped.fetch_add(1, std::memory_order_relaxed);
    return 1;
}

u64 FindHandler(const std::string& name) {
    if (name == "skip_wait_compute_done") {
        return reinterpret_cast<u64>(&SkipWaitComputeDone);
    }
    if (name == "resource_buffer_created") {
        return reinterpret_cast<u64>(&ResourceBufferCreated);
    }
    if (name == "resource_texture_init") {
        return reinterpret_cast<u64>(&ResourceTextureInit);
    }
    if (name == "resource_buffer_destroyed") {
        return reinterpret_cast<u64>(&ResourceBufferDestroyed);
    }
    if (name == "draw_probe_indexed") {
        return reinterpret_cast<u64>(&ProbeIndexed);
    }
    if (name == "draw_probe_auto") {
        return reinterpret_cast<u64>(&ProbeAuto);
    }
    if (name == "tag_cs_copy") {
        return reinterpret_cast<u64>(&TagCsCopyHandler);
    }
    if (name == "tag_mode_switch") {
        return reinterpret_cast<u64>(&TagModeSwitch);
    }
    if (name == "tag_asc_barrier") {
        return reinterpret_cast<u64>(&TagAscBarrier);
    }
    if (name == "tag_track_resource") {
        return reinterpret_cast<u64>(&TagTrackResource);
    }
    if (name == "measure_target_eliminate") {
        return reinterpret_cast<u64>(&MeasureTargetEliminate);
    }
    if (name == "measure_clear_quad") {
        return reinterpret_cast<u64>(&MeasureClearQuad);
    }
    if (name == "measure_copy_resource") {
        return reinterpret_cast<u64>(&MeasureCopyResource);
    }
    if (name == "measure_excs_depth_copy") {
        return reinterpret_cast<u64>(&MeasureExCsDepthCopy);
    }
    if (name == "tag_dispatch_indirect") {
        return reinterpret_cast<u64>(&TagDispatchIndirect);
    }
    if (name == "tag_asc_copy_resource") {
        return reinterpret_cast<u64>(&TagAscCopyResource);
    }
    if (name == "tag_asc_counter_readback") {
        return reinterpret_cast<u64>(&TagAscCounterReadback);
    }
    if (name == "tag_device_begin") {
        return reinterpret_cast<u64>(&TagDeviceBegin);
    }
    if (name == "skip_resource_limits") {
        return reinterpret_cast<u64>(&SkipResourceLimits);
    }
    if (name == "tag_asc_begin") {
        return reinterpret_cast<u64>(&TagAscBegin);
    }
    if (name == "note_clear_dsv") {
        return reinterpret_cast<u64>(&NoteClearDsv);
    }
    if (name == "tag_asc_cs_copy") {
        return reinterpret_cast<u64>(&TagAscCsCopy);
    }
    if (name == "tag_copy_resource") {
        return reinterpret_cast<u64>(&TagCopyResource);
    }
    if (name == "tag_excs_depth_copy") {
        return reinterpret_cast<u64>(&TagExCsDepthCopy);
    }
    if (name == "measure_htile_clear") {
        return reinterpret_cast<u64>(&MeasureHtileClear);
    }
    if (name == "tag_fill") {
        return reinterpret_cast<u64>(&TagFill);
    }
    if (name == "tag_prepare_draw") {
        return reinterpret_cast<u64>(&PrepareDrawHook);
    }
    if (name == "tag_cue_flush") {
        return reinterpret_cast<u64>(&CueFlushHook);
    }
    if (name == "tag_asc_dispatch_record_only") {
        return reinterpret_cast<u64>(&TagAscDispatchRecordOnly);
    }
    if (name == "tag_asc_dispatch") {
        return reinterpret_cast<u64>(&TagAscDispatch);
    }
    if (name == "tag_dispatch") {
        return reinterpret_cast<u64>(&TagDispatch);
    }
    if (name == "tag_dispatch_fenced") {
        return reinterpret_cast<u64>(&TagDispatchFenced);
    }
    if (name == "tag_indirect") {
        return reinterpret_cast<u64>(&TagIndirect);
    }
    if (name == "prepare_note") {
        return reinterpret_cast<u64>(&PrepareNote);
    }
    if (name == "draw_probe_indirect") {
        return reinterpret_cast<u64>(&ProbeIndirect);
    }
    if (name == "ring_discover") {
        return reinterpret_cast<u64>(&RingDiscover);
    }
    if (name == "tag_indexed") {
        return reinterpret_cast<u64>(&TagIndexed);
    }
    if (name == "tag_auto") {
        return reinterpret_cast<u64>(&TagAuto);
    }
    if (name == "tag_fence") {
        return reinterpret_cast<u64>(&TagFenceSignal);
    }
    return 0;
}

LiveCounters& Live() {
    static LiveCounters counters;
    return counters;
}

// ---- direct submission, step 1: the shadow index (documents/tlg/native-direct-submission.md) ---------------------------------
namespace {
struct Anchor {
    std::atomic<u64> address{0};
    std::atomic<u64> seq{0};
    std::atomic<u32> kind{0};
};
constexpr u32 AnchorSlots = 16384;
std::unique_ptr<Anchor[]> anchors;
std::once_flag anchors_once;
std::atomic<u64> anchor_next{0};
std::array<std::atomic<u64>, 256> shadow_other{};
std::array<std::atomic<u64>, 64> shadow_extra_kind{};

Anchor* AnchorRing() {
    std::call_once(anchors_once, [] { anchors = std::make_unique<Anchor[]>(AnchorSlots); });
    return anchors.get();
}

/// True when a native tag (of record `seq`, 0: any) is at `address`.
bool IsTagAt(u64 address, u64 seq) {
    const u32 header = Rd<u32>(address, 0);
    if ((header >> 30) != 3 || ((header >> 8) & 0xff) != 0x10 || ((header >> 16) & 0x3fff) + 1 < VideoCore::NativeTagPayloadDwords ||
        Rd<u32>(address, 4) != VideoCore::NativeTagMagic) {
        return false;
    }
    return seq == 0 || Rd<u64>(address, 9 * sizeof(u32)) == seq;
}
} // namespace

namespace {
void PrepareBuffer(u64 begin, u64 end, u32 depth) {
    for (u64 at = begin; at < end;) {
        const u32 header = Rd<u32>(at, 0);
        if ((header >> 30) != 3) {
            break;
        }
        const u64 size = (((header >> 16) & 0x3fff) + 2) * 4ull;
        if (at + size > end) {
            break;
        }
        const u32 opcode = (header >> 8) & 0xff;
        if (opcode == 0x3f && size == 16 && depth < 2) {
            // an indirect buffer: the game's command buffer behind a ring window or a call (async queues, small state buffers)
            const u64 target = Rd<u32>(at, 4) | static_cast<u64>(Rd<u32>(at, 8) & 0xffff) << 32;
            const u64 dwords = Rd<u32>(at, 12) & 0xfffff;
            if (target >= 0x200000000ull && target < 0x1000000000ull && (target & 3) == 0 && dwords != 0 && dwords <= (1u << 20)) {
                PrepareBuffer(target, target + dwords * sizeof(u32), depth + 1);
            }
        } else if (opcode == 0x47 || opcode == 0x49 || opcode == 0x3c || opcode == 0x37) {
            if (const u32 kind = ConvertSyncPacket(at)) {
                CountSyncConverted(kind, Live().sync_converted_submit);
            }
        }
        at += size;
    }
}
} // namespace

void PrepareSubmittedBuffer(const u32* dcb, u32 dwords) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (dcb == nullptr || policy.native_draw_exec.load(std::memory_order_relaxed) == 0 ||
        (policy.native_sdk_strip.load(std::memory_order_relaxed) & 2097152) == 0) {
        return;
    }
    PrepareBuffer(reinterpret_cast<u64>(dcb), reinterpret_cast<u64>(dcb) + dwords * sizeof(u32), 0);
}

u32 RestoreSyncTag(const u32* tag, std::array<u32, 8>& packet, bool count_execution) {
    const u32 count = (tag[0] >> 16) & 0x3fff;
    const u32 magic = tag[1];
    u32 kind = SyncNone;
    u32 opcode = 0;
    if (magic == VideoCore::NativeFenceMagic && count == 4) {
        packet = {0xC0044700u, 0, tag[2], tag[3], tag[4], tag[5]};
        kind = SyncEop;
        opcode = 0x47;
    } else if (magic == ReleaseMagic && count == 5) {
        packet = {0xC0054902u, 0, tag[2], tag[3], tag[4], tag[5], tag[6]};
        kind = SyncRelease;
        opcode = 0x49;
    } else if ((magic & ~0xffu) == WaitMagicBase && count == 5) {
        packet = {0xC0053C00u, magic & 0xff, tag[2], tag[3], tag[4], tag[5], tag[6]};
        kind = SyncWait;
        opcode = 0x3c;
    }
    if (kind == SyncNone && (magic & ~0xffu) == WriteMagicBase && count >= 3 && count <= 6) {
        packet = {0xC0003700u | count << 16, (magic & 0xf) << 8};
        for (u32 i = 2; i < count + 2; ++i) {
            packet[i] = tag[i];
        }
        kind = SyncWrite;
        opcode = 0x37;
    }
    if (kind != SyncNone && count_execution) {
        Live().sync_exec[kind].fetch_add(1, std::memory_order_relaxed);
    }
    return opcode;
}

bool SubmitShadowEnabled() {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    return policy.native_draw_exec.load(std::memory_order_relaxed) != 0 &&
           (policy.native_sdk_strip.load(std::memory_order_relaxed) & 1048576) != 0;
}

void NoteAnchor(u64 address, u64 seq, u32 kind) {
    if (!SubmitShadowEnabled()) {
        return;
    }
    auto& slot = AnchorRing()[anchor_next.fetch_add(1, std::memory_order_relaxed) % AnchorSlots];
    slot.address.store(0, std::memory_order_release);
    slot.seq.store(seq, std::memory_order_relaxed);
    slot.kind.store(kind, std::memory_order_relaxed);
    slot.address.store(address, std::memory_order_release);
    Live().shadow_anchors.fetch_add(1, std::memory_order_relaxed);
}

u64 ShadowExtraByKind(u32 kind) {
    return kind < 64 ? shadow_extra_kind[kind].load(std::memory_order_relaxed) : 0;
}

u64 ShadowNonEntryCount(u32 opcode) {
    return opcode < 256 ? shadow_other[opcode].load(std::memory_order_relaxed) : 0;
}

void SubmitShadow::Begin(u32 queue_id, const u32* span, u32 dwords) {
    active = span != nullptr && dwords != 0 && SubmitShadowEnabled();
    if (!active) {
        return;
    }
    queue = queue_id;
    begin = reinterpret_cast<u64>(span);
    end = begin + dwords * sizeof(u32);
    Anchor* ring = AnchorRing();
    for (u32 i = 0; i < AnchorSlots; ++i) {
        const u64 address = ring[i].address.load(std::memory_order_acquire);
        if (address >= begin && address + VideoCore::NativeTagPayloadDwords * sizeof(u32) <= end && IsTagAt(address, ring[i].seq.load(std::memory_order_relaxed))) {
            expected.push_back(address);
        }
    }
    std::ranges::sort(expected);
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    seen.assign(expected.size(), 0);
}

void SubmitShadow::Tag(const void* header) {
    if (!active) {
        return;
    }
    const u64 address = reinterpret_cast<u64>(header);
    const auto it = std::lower_bound(expected.begin(), expected.end(), address);
    if (it != expected.end() && *it == address) {
        seen[it - expected.begin()] = 1;
        Live().shadow_matched.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Live().shadow_extra.fetch_add(1, std::memory_order_relaxed);
    const u32 kind = (Rd<u32>(address, 8) >> 8) & 0xff;
    shadow_extra_kind[kind & 63].fetch_add(1, std::memory_order_relaxed);
    static std::atomic<u32> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
        LOG_INFO(Core, "submission shadow: the walk found a tag (kind {}, seq {}) at {:#x} (queue {}, span {:#x}..{:#x}, offset {:#x}) that the index has not",
                 kind, Rd<u64>(address, 9 * sizeof(u32)), address, queue, begin, end, address - begin);
    }
    ++extra;
}

void SubmitShadow::Packet(u32 opcode, bool inert) {
    if (active && !inert) {
        ++other[opcode & 255];
    }
}

void SubmitShadow::Indirect(u64 address, u32 dwords) {
    if (active) {
        nested.emplace_back(address, address + dwords * sizeof(u32));
    }
}

void SubmitShadow::Skipped(u64 address, u32 dwords) {
    if (active) {
        skipped.emplace_back(address, address + dwords * sizeof(u32));
    }
}

void SubmitShadow::End() {
    if (!active) {
        return;
    }
    active = false;
    auto& live = Live();
    live.shadow_spans.fetch_add(1, std::memory_order_relaxed);
    u32 missing = 0;
    u64 first_missing = 0;
    for (size_t i = 0; i < seen.size(); ++i) {
        if (seen[i] == 0) {
            const bool in_child = std::ranges::any_of(nested, [&](const auto& range) { return expected[i] >= range.first && expected[i] < range.second; });
            const bool in_payload = std::ranges::any_of(skipped, [&](const auto& range) { return expected[i] >= range.first && expected[i] < range.second; });
            if (in_child) {
                live.shadow_nested.fetch_add(1, std::memory_order_relaxed);
            } else if (in_payload) {
                live.shadow_stale.fetch_add(1, std::memory_order_relaxed);
            } else if (missing++ == 0) {
                first_missing = expected[i];
            }
        }
    }
    if (missing != 0) {
        live.shadow_missing.fetch_add(missing, std::memory_order_relaxed);
        static std::atomic<u32> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 20) {
            std::string around;
            for (int k = -10; k < 3; ++k) {
                around += fmt::format(" {:08x}", Rd<u32>(first_missing, k * 4));
            }
            LOG_INFO(Core,
                     "submission shadow: {} indexed tags of the span {:#x}..{:#x} (queue {}) were not found by the walk, first at {:#x} (offset {:#x}, "
                     "kind {}, seq {}, still a tag {}), dwords from -10:{}",
                     missing, begin, end, queue, first_missing, first_missing - begin, (Rd<u32>(first_missing, 8) >> 8) & 0xff,
                     Rd<u64>(first_missing, 9 * sizeof(u32)), IsTagAt(first_missing, 0), around);
        }
    }
    u32 other_total = 0;
    std::string signature;
    for (u32 opcode = 0; opcode < 256; ++opcode) {
        if (other[opcode] != 0) {
            other_total += other[opcode];
            shadow_other[opcode].fetch_add(other[opcode], std::memory_order_relaxed);
            signature += fmt::format(" {:#x}:{}", opcode, other[opcode]);
        }
    }
    if (other_total == 0) {
        live.shadow_pure.fetch_add(1, std::memory_order_relaxed);
    } else {
        live.shadow_mixed.fetch_add(1, std::memory_order_relaxed);
        // one line per distinct set of opcodes (the counts change from frame to frame)
        static std::mutex signatures_mutex;
        static std::set<std::string> signatures;
        std::string key = fmt::format("q{}", queue);
        for (u32 opcode = 0; opcode < 256; ++opcode) {
            if (other[opcode] != 0) {
                key += fmt::format(" {:#x}", opcode);
            }
        }
        std::scoped_lock lock{signatures_mutex};
        if (signatures.size() < 48 && signatures.insert(key).second) {
            LOG_INFO(Core, "submission shadow: mixed span, queue {}, {} dwords, {} tags, packets that are not entries:{}", queue,
                     (end - begin) / sizeof(u32), expected.size(), signature);
        }
    }
    nested.clear();
    skipped.clear();
    expected.clear();
    expected.shrink_to_fit();
    seen.clear();
    other.fill(0);
}

std::string DescribeEntry(const void* header) {
    const u64 address = reinterpret_cast<u64>(header);
    const u32 meta = Rd<u32>(address, 8);
    const u32 kind = (meta >> 8) & 0xff;
    const u64 seq = Rd<u64>(address, 9 * sizeof(u32));
    const char* mode = (meta & VideoCore::NativeTagExecFlag) == 0       ? "packets-follow"
                       : (meta & VideoCore::NativeTagFollowFlag) != 0 ? "at-draw-packet"
                                                                      : "record";
    const char* name = "UNKNOWN";
    switch (kind) {
    case 0: name = "DRAW_INDEXED"; break;
    case 1: name = "DRAW_AUTO"; break;
    case 2: name = "DRAW_INDIRECT"; break;
    case 3: name = "DISPATCH"; break;
    case 4: name = "FAST_CLEAR_ELIMINATE"; break;
    case 6: name = "FILL"; break;
    case 7: name = "DEPTH_CLEAR"; break;
    case 8: name = "GDS_FILL"; break;
    case 9: name = "GDS_READBACK"; break;
    case 10: name = "COPY"; break;
    case 11: name = "DISPATCH_INDIRECT"; break;
    case 12: name = "STATE_TEMPLATE"; break;
    default: break;
    }
    std::string text = fmt::format("ENTRY {} seq={} {} class={}", name, seq, mode, meta & 0xff);
    if (kind == 12) {
        if (const u64 pointer = Rd<u64>(address, 12); pointer != 0) {
            text += fmt::format(" writes={}", reinterpret_cast<const StateTemplate*>(pointer)->writes.size());
        }
        return text;
    }
    if (seq == 0) {
        return text + " no-record";
    }
    const auto& slot = Slot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        return text + " record-gone";
    }
    const auto find = [&](u32 index, u32& value) {
        for (u32 i = 0; i < slot.count; ++i) {
            if (slot.regs[i].index == index) {
                value = slot.regs[i].value;
                return true;
            }
        }
        return false;
    };
    const VAddr image = image_base.load(std::memory_order_relaxed);
    text += fmt::format(" caller={:#x}", image != 0 && slot.caller >= image ? slot.caller - image + 0x1000000 : slot.caller);
    switch (kind) {
    case 0:
    case 1:
    case 2: {
        if (kind == 2) {
            text += fmt::format(" args={:#x}+{}", slot.args_addr, slot.a2);
        } else if (slot.a1 == UnknownDrawCount) {
            text += fmt::format(" count=? first={}", slot.a2);
        } else {
            text += fmt::format(" count={} first={}", slot.a1, slot.a2);
        }
        text += fmt::format(" vs={:#x} ps={:#x} keys={:#x}/{:#x}/{:#x}", slot.vs_gnm, slot.ps_gnm, Rd<u64>(address, 12), Rd<u64>(address, 20),
                            Rd<u64>(address, 28));
        for (u32 target = 0; target < 8; ++target) {
            if (u32 base; find(ContextBase + 0x318 + 0xf * target, base) && base != 0) {
                text += fmt::format(" rt{}={:#x}", target, u64{base} << 8);
            }
        }
        if (u32 base; find(ContextBase + 0x14, base) && base != 0) {
            text += fmt::format(" z={:#x}", u64{base} << 8);
        }
        text += fmt::format(" fixed={} special={:#x}", slot.fixed != nullptr, slot.flags);
        break;
    }
    case 3:
    case 11: {
        if (u32 x, y, z; find(ComputeRegisterBase + 1, x) && find(ComputeRegisterBase + 2, y) && find(ComputeRegisterBase + 3, z)) {
            text += fmt::format(" groups={}x{}x{}", x, y, z);
        }
        if (kind == 11) {
            text += fmt::format(" args={:#x}+{}", slot.args_addr, slot.a2);
        }
        if (slot.copy_bytes != 0) {
            text += fmt::format(" as-copy dst={:#x} src={:#x} bytes={}", slot.copy_destination, slot.copy_source, slot.copy_bytes);
        } else if (slot.fill_dwords != 0) {
            text += fmt::format(" as-fill dst={:#x} bytes={} pattern-dwords={} first={:#x}", slot.fill_destination, slot.fill_bytes, slot.fill_dwords,
                                slot.fill_pattern[0]);
        }
        break;
    }
    case 4:
        break;
    case 6:
    case 8:
        text += fmt::format(" dst={:#x}{} dwords={} value={:#x}", slot.args_addr, kind == 8 ? "(gds)" : "", slot.a1, slot.a2);
        break;
    case 9:
        text += fmt::format(" dst={:#x} gds={:#x} dwords={}", slot.args_addr, slot.a2, slot.a1);
        break;
    case 10:
        text += fmt::format(" dst={:#x} src={:#x} bytes={}", slot.args_addr, slot.extra_addr, slot.a1);
        break;
    case 7:
        if (u32 base; find(ContextBase + 0x14, base)) {
            text += fmt::format(" z={:#x}", u64{base} << 8);
        }
        break;
    default:
        break;
    }
    return text;
}

u64 EngineDevice() {
    return ResolveDevice().device;
}

bool IsGuarded() {
    return guard_active;
}

VAddr ImageBase() {
    return image_base.load(std::memory_order_relaxed);
}

void SetImageBase(VAddr base) {
    image_base.store(base, std::memory_order_relaxed);
}

} // namespace Core::NativeDraw

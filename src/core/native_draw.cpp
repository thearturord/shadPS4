// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "core/debug_server.h"
#include "core/native_draw.h"
#include "video_core/native_draw_tag.h"
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
                    cb.Add(Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8));
                }
            }
            HashSlots(srv, ctx, Ctx::ShaderResourceTable(stage), Rd<u64>(shader, Shader::SrvMaskLo),
                      Rd<u64>(shader, Shader::SrvMaskHi));
            const u32 smp_mask = Rd<u32>(shader, Shader::SamplerMask) & 0xffff;
            for (u32 slot = 0; slot < 16; ++slot) {
                if ((smp_mask >> slot) & 1) {
                    smp.Add(Rd<u64>(ctx, Ctx::SamplerTable(stage) + slot * 8));
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

// Ghidra address ranges of the three passes (documents/tlg/native-phase1-results-2026-10-15.md).
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
            const u64 obj = Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8);
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

bool WriteDwords(u64 ctx, const u32* words, u32 count) {
    const u64 writer = ctx + 0x47c0;
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
    return true;
}

std::atomic<u64> tags_written{};
std::atomic<u64> tags_skipped_class{};
std::atomic<u64> tags_failed{};
std::atomic<u64> last_tag_log_ns{0};

u64 Tag(u64 ctx, u64 a1, u64 a2, u64 a3, u64 ret, u32 kind) {
    const auto& policy = VideoCore::ReadbackPolicy::Get();
    if (policy.native_draw_mode.load(std::memory_order_relaxed) == 0 || ctx == 0) {
        return 0;
    }
    const Class cls = Classify(ret);
    if (((policy.native_draw_classes.load(std::memory_order_relaxed) >> cls) & 1) == 0) {
        tags_skipped_class.fetch_add(1, std::memory_order_relaxed);
        return 0;
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
    words[9] = 0;
    words[10] = 0;
    if (WriteDwords(ctx, words, 2 + VideoCore::NativeTagPayloadDwords - 1)) {
        tags_written.fetch_add(1, std::memory_order_relaxed);
    } else {
        tags_failed.fetch_add(1, std::memory_order_relaxed);
    }
    const u64 t = NowNs();
    u64 last = last_tag_log_ns.load(std::memory_order_relaxed);
    if (t - last > 5'000'000'000ull && last_tag_log_ns.compare_exchange_strong(last, t) && last != 0) {
        LOG_INFO(Core, "native draw tags: {} written, {} failed, {} skipped (class not enabled)",
                 tags_written.exchange(0), tags_failed.exchange(0), tags_skipped_class.exchange(0));
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
    return Tag(ctx, a1, a2, a3, ret, 0);
}
u64 PS4_SYSV_ABI TagAuto(u64 ctx, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, u64 ret) {
    return Tag(ctx, a1, a2, a3, ret, 1);
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

u64 FindHandler(const std::string& name) {
    if (name == "draw_probe_indexed") {
        return reinterpret_cast<u64>(&ProbeIndexed);
    }
    if (name == "draw_probe_auto") {
        return reinterpret_cast<u64>(&ProbeAuto);
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
    return 0;
}

void SetImageBase(VAddr base) {
    image_base.store(base, std::memory_order_relaxed);
}

} // namespace Core::NativeDraw

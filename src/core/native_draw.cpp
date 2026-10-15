// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bitset>
#include <bit>
#include <cmath>
#include <atomic>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <string>
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
    u32 rt_info{};
    u32 flags{}; ///< RecordFlag bits: why this draw cannot run natively
    /// Tables the user data points to (extended user data, descriptor tables): built in `address`, compared by
    /// content with what the register `reg` points to in the packet path's state.
    struct TableCheck {
        u32 reg;
        u32 kind;
        u32 dwords;
        u64 address;
        u32 entry_dwords; ///< size of one table entry (0 = compare the table as a whole)
        std::array<u64, 2> slot_mask; ///< entries the shader reads (bit per slot); the others hold stale data
    };
    u32 check_count{};
    std::array<TableCheck, 8> checks{};
    u32 unsupported_inputs{}; ///< input usage types the builder does not know
    std::array<SynthReg, MaxSynthRegs> regs{};
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

constexpr u32 TableRingDwords = 4u << 20; // 16 MB of scratch descriptor tables
std::unique_ptr<u32[]> table_ring;
std::once_flag table_ring_once;
std::atomic<u64> table_cursor{0};

u32* AllocTable(u32 dwords) {
    std::call_once(table_ring_once, [] { table_ring = std::make_unique<u32[]>(TableRingDwords); });
    u64 start = table_cursor.fetch_add(dwords, std::memory_order_relaxed) % TableRingDwords;
    if (start + dwords > TableRingDwords) {
        table_cursor.fetch_add(TableRingDwords - start, std::memory_order_relaxed);
        start = 0;
    }
    return table_ring.get() + start;
}

// Descriptors of the objects bound in the context's tables (api stage 0 = vertex, 1 = pixel).
/// Writes up to `max_dwords` of the descriptor of shader resource slot `slot`; returns the descriptor size in dwords.
u32 ResourceDescriptor(u64 ctx, u32 stage, u32 slot, u32* out, u32 max_dwords) {
    const u64 view = Rd<u64>(ctx, Ctx::ShaderResourceTable(stage) + slot * 8);
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

void ConstBufferDescriptor(u64 ctx, u32 stage, u32 slot, u32* out) {
    const u64 buffer = Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + slot * 8);
    for (u32 i = 0; i < 4; ++i) {
        out[i] = buffer ? Rd<u32>(buffer, 0x60 + i * 4) : 0;
    }
}

void SamplerDescriptor(u64 ctx, u32 stage, u32 slot, u32* out) {
    const u64 sampler = Rd<u64>(ctx, Ctx::SamplerTable(stage) + slot * 8);
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
                const u32 dwords = 1u << opcode;
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
thread_local u32 synth_rt_info; ///< count | view mask << 8 | empty block mask << 16
thread_local u64 dbg_vs;
thread_local u64 dbg_fetch;

u32 Synthesize(u64 ctx, SynthReg* out, u64* vs_gnm_out, u64* ps_gnm_out);

/// User data of one stage (SGPR values the constant update engine would have written): descriptors of the bound
/// objects copied by the shader's input usage table. `sgpr_base` = index of the stage's first user data register.
void SynthesizeUserData(u64 ctx, u32 stage, u64 gnm, u64 inputs_offset, u32 sgpr_base, SynthReg* out, u32& n,
                        NativeRecord& record) {
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
        case 27: // pointer to the extended user data
            have_eud_ptr = true;
            eud_ptr_sgpr = start;
            break;
        case 19:   // pointer to the resource table (32 bytes per slot)
        case 21:   // pointer to the sampler table (16 bytes per slot)
        case 22: { // pointer to the constant buffer table (16 bytes per slot)
            if (pending_count >= pending.size()) {
                ++record.unsupported_inputs;
                break;
            }
            u32 slots = 0;
            for (u32 s = 0; s < (type == 19 ? 128u : 16u); ++s) {
                const u64 obj = type == 19 ? Rd<u64>(ctx, Ctx::ShaderResourceTable(stage) + s * 8)
                                : type == 22 ? Rd<u64>(ctx, Ctx::ConstantBufferTable(stage) + s * 8)
                                             : Rd<u64>(ctx, Ctx::SamplerTable(stage) + s * 8);
                if (obj) {
                    slots = s + 1;
                }
            }
            slots = std::max(slots, 1u);
            const u32 entry = type == 19 ? 8 : 4;
            u32* table = AllocTable(slots * entry);
            for (u32 s = 0; s < slots; ++s) {
                if (type == 19) {
                    ResourceDescriptor(ctx, stage, s, table + s * 8, 8);
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
                                            UsedSlots(ScanShaderLoads(gnm), start, entry, slots)};
            }
            break;
        }
        default:
            ++record.unsupported_inputs;
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
        if (raster) {
            context_reg(0x205, Rd<u32>(raster, 0x4c) & 0x393fff);
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
        if (!gnm) {
            synth_flags |= NoVertexShader;
        }
        if (gnm) {
            const u64 regs = gnm + 8;
            const u64 fetch_obj = Rd<u64>(vs + 0x30, 0xe0);
            if (Rd<u64>(vs, 0x420) != 0 || Rd<u64>(ctx, 0x4680) != 0 || Rd<u64>(ctx, 0x4688) != 0 ||
                Rd<u32>(ctx, 0x130c8) == 0x200) {
                synth_flags |= SpecialPipeline;
            }
            dbg_vs = vs;
            dbg_fetch = fetch_obj;
            const u32 modifier = fetch_obj ? Rd<u32>(fetch_obj, 0x40) : 0;
            const u64 fetch_addr = fetch_obj ? Rd<u64>(fetch_obj, 0x30) : 0;
            const u32 r2 = Rd<u32>(regs, 8);
            sh_reg(0x48, Rd<u32>(regs, 0));
            sh_reg(0x49, 0);
            // the modifier of the fetch shader object goes into the shader's first resource word; a vertex shader whose
            // fetch object does not exist yet gets it created during this draw
            sh_reg(0x4a, fetch_obj ? (r2 & 0xfcfffc3fu) | modifier : r2);
            sh_reg(0x4b, Rd<u32>(regs, 12));
            context_reg(0x207, Rd<u32>(regs, 24));
            context_reg(0x1b1, Rd<u32>(regs, 16));
            context_reg(0x1c3, Rd<u32>(regs, 20));
            bool user_data_in_first_sgprs = false;
            for (u32 k = 0; k < Rd<u8>(gnm, 3) && k < 40; ++k) {
                user_data_in_first_sgprs |= Rd<u8>(gnm + 0x28 + k * 4, 2) < 2;
            }
            if (fetch_addr && !user_data_in_first_sgprs) {
                sh_reg(0x4c, static_cast<u32>(fetch_addr));
                sh_reg(0x4d, static_cast<u32>(fetch_addr >> 32));
            }
            if (fetch_obj) {
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
            for (u32 k = 0; k < (Rd<u32>(regs, 32) & 0x3f) && k < 32; ++k) {
                context_reg(0x191 + k, Rd<u16>(interpolants, k * 2));
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

NativeRecord& Slot(u64 seq) {
    std::call_once(records_once, [] { records = std::make_unique<NativeRecord[]>(RecordSlots); });
    return records[seq % RecordSlots];
}

/// Guest thread: builds the record of a draw. Returns its number.
u64 MakeRecord(u64 ctx, u32 kind, u64 a1, u64 a2, u64 a3) {
    const u64 seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    auto& slot = Slot(seq);
    slot.seq.store(0, std::memory_order_release);
    slot.check_count = 0;
    slot.unsupported_inputs = 0;
    synth_flags = 0;
    slot.count = Synthesize(ctx, slot.regs.data(), &slot.vs_gnm, &slot.ps_gnm);
    slot.flags = synth_flags;
    slot.rt_info = synth_rt_info;
    slot.dbg_vs = dbg_vs;
    slot.dbg_fetch = dbg_fetch;
    {
        u32 n = slot.count;
        SynthesizeUserData(ctx, 0, slot.vs_gnm, 0x28, ShBase + 0x4c, slot.regs.data(), n, slot);
        SynthesizeUserData(ctx, 1, slot.ps_gnm, 0x3c, ShBase + 0x0c, slot.regs.data(), n, slot);
        // draw arguments: VGT_NUM_INDICES / VGT_NUM_INSTANCES, then for indexed draws the index buffer view
        // (IT_INDEX_BASE), the base vertex and the draw packet's maximum size
        slot.regs[n++] = {UconfigBase + 0x24c, static_cast<u32>(a1)};
        slot.regs[n++] = {UconfigBase + 0x24d, Rd<u32>(ctx, 0x4724)};
        if (kind == 0) {
            if (const u64 view = Rd<u64>(ctx, 0x58); view != 0) {
                slot.regs[n++] = {ContextBase + 0x1fa, Rd<u32>(view, 0x60) & 0xfffffffeu};
                slot.regs[n++] = {ContextBase + 0x1f9, Rd<u32>(view, 0x64) & 0xff};
                slot.regs[n++] = {ContextBase + 0x102, Rd<u32>(ctx, 0x64)};
                if (Rd<u32>(ctx, 0x60) != 0x39) {
                    slot.flags |= SpecialPipeline;
                }
            }
            slot.regs[n++] = {ContextBase + 0x29e, static_cast<u32>(a1)};
        }
        slot.count = n;
    }
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
    u64 draws_clean_mismatch{};
    std::array<u64, 4> flag_counts{};
    std::unordered_map<u64, u64> index_type_pairs;
    std::unordered_map<u32, u64> rt_odd_info;
    struct PerReg {
        u64 count{};
        u32 expected{};
        u32 actual{};
    };
    std::unordered_map<u32, PerReg> by_register;
    std::unordered_map<u32, u64> uncovered; ///< registers that changed between draws without being in the record
    std::unordered_map<u32, std::pair<u32, u32>> uncovered_values; ///< last value and the one before
    u64 last_log_ns{};
} verify_stats;

}  // namespace

void VerifyRecord(u64 seq, const u32* reg_array) {
    auto& slot = Slot(seq);
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        std::scoped_lock lk{verify_stats.mutex};
        ++verify_stats.draws_missing;
        return;
    }
    // copy: the guest thread may reuse the slot while this is compared
    std::array<SynthReg, MaxSynthRegs> regs = slot.regs;
    const u32 count = slot.count;
    const u64 vs_gnm = slot.vs_gnm;
    const u64 ps_gnm = slot.ps_gnm;
    const u32 flags = slot.flags;
    const u32 rt_info = slot.rt_info;
    const u64 dbg_vs = slot.dbg_vs;
    const u64 dbg_fetch = slot.dbg_fetch;
    const u32 check_count = slot.check_count;
    const auto checks = slot.checks;
    const u32 unsupported = slot.unsupported_inputs;
    if (slot.seq.load(std::memory_order_acquire) != seq) {
        std::scoped_lock lk{verify_stats.mutex};
        ++verify_stats.draws_missing;
        return;
    }
    std::scoped_lock lk{verify_stats.mutex};
    auto& s = verify_stats;
    ++s.draws_checked;
    bool any = false;
    for (u32 i = 0; i < count; ++i) {
        ++s.registers_compared;
        const u32 actual = reg_array[regs[i].index];
        if (actual != regs[i].value) {
            if (regs[i].index == ShBase + 0x4c || regs[i].index == ShBase + 0x4a) {
                static u32 fetch_dumps = 0;
                if (fetch_dumps < 40 && dbg_vs != 0) {
                    ++fetch_dumps;
                    const auto rd = [](u64 base, u64 off) { return base ? Rd<u64>(base, off) : u64{0}; };
                    const u64 alt = rd(dbg_vs + 0x330, 0xe0);
                    LOG_INFO(Core,
                             "native fetch diff: reg {:#x} vs {:#x} variant420 {:#x} fetch_obj {:#x} built {:#x} actual {:#x}/{:#x} "
                             "obj30 {:#x} 38 {:#x} 40 {:#x} 48 {:#x} | variant330 obj {:#x}: 30 {:#x} 40 {:#x}",
                             regs[i].index, dbg_vs, rd(dbg_vs, 0x420), dbg_fetch, regs[i].value, actual, reg_array[ShBase + 0x4d],
                             rd(dbg_fetch, 0x30), rd(dbg_fetch, 0x38), rd(dbg_fetch, 0x40), rd(dbg_fetch, 0x48), alt,
                             rd(alt, 0x30), rd(alt, 0x40));
                }
            }
            if (regs[i].index == ContextBase + 0x29f) {
                ++s.index_type_pairs[(u64(regs[i].value) << 32) | actual];
            }
            ++s.mismatches;
            any = true;
            auto& r = s.by_register[regs[i].index];
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
            static u32 table_dumps = 0;
            if (ptr != 0 && table_dumps < 80 && Core::Memory::Instance()->IsValidMapping(ptr, check.dwords * 4)) {
                ++table_dumps;
                std::string text;
                for (u32 k = 0; k < check.dwords && k < 48; ++k) {
                    const u32 b = Rd<u32>(check.address, k * 4);
                    const u32 a = Rd<u32>(ptr, k * 4);
                    text += b == a ? fmt::format(" {:x}", a) : fmt::format(" [{}:{:x}/{:x}]", k, b, a);
                }
                LOG_INFO(Core, "native table diff: kind {} reg {:#x} dwords {} (built/actual):{}", check.kind, check.reg,
                         check.dwords, text);
            }
            ++s.mismatches;
            any = true;
            auto& r = s.by_register[0x90000000u + check.kind * 0x100 + (check.reg & 0xff)];
            ++r.count;
            r.expected = Rd<u32>(check.address, 0);
            r.actual = first_actual;
        }
    }
    if (unsupported != 0) {
        auto& r = s.by_register[0x91000000u];
        r.count += unsupported;
    }
    {
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
    if (flags & OddRenderTarget) {
        ++s.rt_odd_info[rt_info];
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
            LOG_INFO(Core,
                     "native front end check: {} draws cannot run natively (no vertex shader {}, no fetch object {}, odd "
                     "render target {}, special pipeline {}); {} of the others differ",
                     s.draws_flagged, s.flag_counts[0], s.flag_counts[1], s.flag_counts[2], s.flag_counts[3],
                     s.draws_clean_mismatch);
            for (size_t k = 0; k < top.size() && k < 12; ++k) {
                LOG_INFO(Core, "native front end check: register {:#x} differs {} times (last: built {:#x}, actual {:#x})",
                         top[k].first, top[k].second.count, top[k].second.expected, top[k].second.actual);
            }
            for (const auto& [pair, count] : s.index_type_pairs) {
                LOG_INFO(Core, "native front end check: index type built {:#x} actual {:#x}: {} draws", pair >> 32,
                         pair & 0xffffffff, count);
            }
            for (const auto& [info, count] : s.rt_odd_info) {
                LOG_INFO(Core, "native front end check: odd render targets: count {} views {:#x} empty blocks {:#x}: {} draws",
                         info & 0xff, (info >> 8) & 0xff, (info >> 16) & 0xff, count);
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
        s.flag_counts = {};
        s.by_register.clear();
        s.uncovered.clear();
        s.index_type_pairs.clear();
        s.rt_odd_info.clear();
        s.uncovered_values.clear();
        s.last_log_ns = now;
    }
}

namespace {

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
    const u64 seq = policy.native_synth_verify.load(std::memory_order_relaxed) != 0 ? MakeRecord(ctx, kind, a1, a2, a3) : 0;
    std::memcpy(&words[9], &seq, sizeof(u64));
    words[11] = 0;
    words[12] = 0;
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

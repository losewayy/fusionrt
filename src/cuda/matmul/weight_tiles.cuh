#pragma once

#include "../cuda_common.cuh"

#include <cuda_bf16.h>

namespace fusionrt::cuda_ops {

namespace {

template <typename W>
inline constexpr bool TileQ4_144 = std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ4_KA>;

template <typename W>
struct TileQ4Trait {
    static_assert(TileQ4_144<W> && sizeof(W) == 9 * 16);
    static_assert(offsetof(BlockQ4_A, qs) % 16 == 0 && offsetof(BlockQ4_KA, q) % 16 == 0);
    static constexpr bool symmetric = std::is_same_v<W, BlockQ4_A>;

    __device__ static __forceinline__ const uint4* nibbles(const W& w, int sub) {
        if constexpr (symmetric) return reinterpret_cast<const uint4*>(w.qs) + sub;
        else return reinterpret_cast<const uint4*>(w.q[sub]);
    }

    __device__ static __forceinline__ uint4 load_nibbles(const W& w, int sub) {
        return __ldg(nibbles(w, sub));
    }

    __device__ static __forceinline__ float2 scales(const W& w, int sub) {
        if constexpr (symmetric) {
            const float d = half_bits_to_f32(__ldg(&w.d[sub]));
            return make_float2(d, 8.0f * d);
        } else {
            const float2 base = __half22float2(*reinterpret_cast<const half2*>(&w.d));
            const unsigned factor = unpack_q4_ka_scales(w, sub);
            return make_float2(base.x * (factor & 63), base.y * (factor >> 6));
        }
    }
};

template <typename W>
    requires TileQ4_144<W>
__device__ __forceinline__ float repacked_k_value(const W& w, int i) {
    using Tile = TileQ4Trait<W>;
    const int sub = i / 32;
    const int q = (reinterpret_cast<const uint8_t*>(Tile::nibbles(w, sub))[i % 16] >> ((i / 16 % 2) * 4)) & 15;
    const float2 sc = Tile::scales(w, sub);
    if constexpr (Tile::symmetric) return sc.x * (q - 8);
    else return sc.x * q - sc.y;
}

__device__ __forceinline__ float repacked_k_value(const BlockQ6_A& b, int i) {
    const int sub = i / 32;
    const int half = i / 16 % 2;
    const int lo = (b.ql[sub][i % 16] >> (half * 4)) & 15;
    const int hi = (b.qh[sub / 2][i % 16] >> (sub % 2 * 4 + half * 2)) & 3;
    return half_bits_to_f32(b.d) * b.scale[i / 16] * ((lo | (hi << 4)) - 32);
}

// Weight kind enum for unified WMMA prefill kernel:
// 0=F16, 1=BF16, 2=Q8_0, 3=Q4_0, 4=Q4_1, 5=Q4_K, 6=Q6_K, 7=Q5_K, 8=Q4_KA, 9=Q6_A
template <int WK>
__device__ __forceinline__ float dequant_weight_elem(const void* w, int n, int k0, int c, int K) {
    if constexpr (WK == 0) {
        return __half2float(__ldg(reinterpret_cast<const half*>(w) + static_cast<int64_t>(n) * K + k0 + c));
    } else if constexpr (WK == 1) {
        return __bfloat162float(__ldg(reinterpret_cast<const __nv_bfloat16*>(w) + static_cast<int64_t>(n) * K + k0 + c));
    } else if constexpr (WK == 2) {
        int nb = K >> 5;
        const BlockQ8_0& blk = reinterpret_cast<const BlockQ8_0*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(__ldg(&blk.qs[c]));
    } else if constexpr (WK == 3) {
        int nb = K >> 5;
        const BlockQ4_0& blk = reinterpret_cast<const BlockQ4_0*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        uint8_t byte = __ldg(&blk.qs[c & 15]);
        int q = (c < 16) ? (static_cast<int>(byte & 0x0Fu) - 8) : (static_cast<int>(byte >> 4) - 8);
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(q);
    } else if constexpr (WK == 4) {
        int nb = K >> 5;
        const BlockQ4_1& blk = reinterpret_cast<const BlockQ4_1*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 5)];
        uint8_t byte = __ldg(&blk.qs[c & 15]);
        int q = (c < 16) ? static_cast<int>(byte & 0x0Fu) : static_cast<int>(byte >> 4);
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(q) + half_bits_to_f32(__ldg(&blk.m));
    } else if constexpr (WK == 5 || WK == 7) {
        using W = std::conditional_t<WK == 5, BlockQ4_K, BlockQ5_K>;
        int nb = K >> 8;
        const W& blk = reinterpret_cast<const W*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 8)];
        int sub = k0 & 255;
        int grp = sub >> 6;
        int hi = (sub >> 5) & 1;
        int j = grp * 2 + hi;
        uint8_t sc = (j < 4) ? (blk.scales[j] & 63u) : static_cast<uint8_t>((blk.scales[j + 4] & 0xFu) | ((blk.scales[j - 4] >> 6) << 4));
        uint8_t mn = (j < 4) ? (blk.scales[j + 4] & 63u) : static_cast<uint8_t>((blk.scales[j + 4] >> 4) | ((blk.scales[j] >> 6) << 4));
        uint8_t qb = __ldg(&blk.qs[grp * 32 + c]);
        int q = hi ? (qb >> 4) : (qb & 0xFu);
        if constexpr (WK == 7) q |= ((__ldg(&blk.qh[c]) >> j) & 1) << 4;
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(sc) * static_cast<float>(q)
             - half_bits_to_f32(__ldg(&blk.dmin)) * static_cast<float>(mn);
    } else if constexpr (WK == 8 || WK == 9) {
        using W = std::conditional_t<WK == 8, BlockQ4_KA, BlockQ6_A>;
        const int64_t block = static_cast<int64_t>(n) * (K / 256) + k0 / 256;
        return repacked_k_value(static_cast<const W*>(w)[block], (k0 + c) % 256);
    } else {
        int nb = K >> 8;
        const BlockQ6_K& blk = reinterpret_cast<const BlockQ6_K*>(w)[static_cast<int64_t>(n) * nb + (k0 >> 8)];
        int sub = k0 & 255;
        int half_idx = sub >> 7;
        int q_part = (sub >> 5) & 3;
        const uint8_t* ql = blk.ql + half_idx * 64 + ((q_part & 1) ? 32 : 0);
        const uint8_t* qh = blk.qh + half_idx * 32;
        const int8_t* sc = blk.scales + half_idx * 8;
        uint8_t qlv = __ldg(&ql[c]);
        uint8_t qhv = __ldg(&qh[c]);
        int lo4 = (q_part >= 2) ? (qlv >> 4) : (qlv & 0xFu);
        int hi2 = (qhv >> (q_part * 2)) & 3u;
        int q = (lo4 | (hi2 << 4)) - 32;
        int scv = static_cast<int>(__ldg(&sc[(c >> 4) + q_part * 2]));
        return half_bits_to_f32(__ldg(&blk.d)) * static_cast<float>(scv) * static_cast<float>(q);
    }
}

constexpr unsigned kNibbleMask = 0x0f0f0f0fu;

__device__ __forceinline__ int dot_q8_words(uint4 w, uint4 x) {
    int dot = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), 0);
    dot = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), dot);
    dot = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), dot);
    return __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), dot);
}

__device__ __forceinline__ unsigned unpack_q4_words(unsigned v, int shift) {
    return (v >> shift) & kNibbleMask;
}

__device__ __forceinline__ uint4 unpack_q4_words(uint4 v, int shift) {
    return make_uint4(unpack_q4_words(v.x, shift), unpack_q4_words(v.y, shift),
                      unpack_q4_words(v.z, shift), unpack_q4_words(v.w, shift));
}

__device__ __forceinline__ uint4 load_q6_words(const uint8_t* p) {
    // The 210-byte block stride can leave words two bytes off alignment.
    auto address = reinterpret_cast<uintptr_t>(p);
    auto* aligned = reinterpret_cast<const unsigned*>(address & ~uintptr_t{3});
    unsigned shift = static_cast<unsigned>(address & 2) * 8;
    unsigned a = __ldg(aligned);
    unsigned b = __ldg(aligned + 1);
    unsigned c = __ldg(aligned + 2);
    unsigned d = __ldg(aligned + 3);
    unsigned e = __ldg(aligned + 4);
    return make_uint4(__funnelshift_r(a, b, shift), __funnelshift_r(b, c, shift),
                      __funnelshift_r(c, d, shift), __funnelshift_r(d, e, shift));
}

__device__ __forceinline__ uint4 unpack_q6_words(const uint8_t* ql, const uint8_t* qh, int part) {
    uint4 lo = unpack_q4_words(load_q6_words(ql), (part >> 1) * 4);
    uint4 hi = load_q6_words(qh);
    int shift = part * 2;
    constexpr unsigned mask = 0x03030303u;
    return make_uint4(lo.x | (((hi.x >> shift) & mask) << 4),
                      lo.y | (((hi.y >> shift) & mask) << 4),
                      lo.z | (((hi.z >> shift) & mask) << 4),
                      lo.w | (((hi.w >> shift) & mask) << 4));
}

__device__ __forceinline__ unsigned unpack_q6_a(unsigned low, unsigned high, int half, int parity) {
    const unsigned lo = (low >> (half * 4)) & kNibbleMask;
    const unsigned hi = (high >> (parity * 4 + half * 2)) & 0x03030303u;
    return lo | (hi << 4);
}

__device__ __forceinline__ uint4 unpack_q6_a(uint4 low, uint4 high, int half, int parity) {
    return make_uint4(unpack_q6_a(low.x, high.x, half, parity), unpack_q6_a(low.y, high.y, half, parity),
                      unpack_q6_a(low.z, high.z, half, parity), unpack_q6_a(low.w, high.w, half, parity));
}

template <typename W>
__device__ __forceinline__ float dot_q8_1(const W& w, int sub, uint4 lo, uint4 hi, float4 x) {
    if constexpr (TileQ4_144<W>) {
        using Tile = TileQ4Trait<W>;
        const uint4 qs = Tile::load_nibbles(w, sub);
        const int dot = dot_q8_words(unpack_q4_words(qs, 0), lo) + dot_q8_words(unpack_q4_words(qs, 4), hi);
        const float2 scales = Tile::scales(w, sub);
        if constexpr (Tile::symmetric) return scales.x * (x.x * dot - 8.0f * x.w);
        else return scales.x * x.x * dot - scales.y * x.w;
    } else if constexpr (std::is_same_v<W, BlockQ8_A>) {
        const auto* qs = reinterpret_cast<const uint4*>(&w.qs[8 * sub]);
        int dot = dot_q8_words(__ldg(qs), lo) + dot_q8_words(__ldg(qs + 1), hi);
        return half_bits_to_f32(__ldg(&w.d[sub])) * x.x * dot;
    } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
        const uint4 low = __ldg(reinterpret_cast<const uint4*>(w.ql[sub]));
        const uint4 high = __ldg(reinterpret_cast<const uint4*>(w.qh[sub / 2]));
        int dot_lo = dot_q8_words(unpack_q6_a(low, high, 0, sub % 2), lo);
        int dot_hi = dot_q8_words(unpack_q6_a(low, high, 1, sub % 2), hi);
        return half_bits_to_f32(w.d) *
               (w.scale[2 * sub] * (x.x * dot_lo - 32.0f * x.y) +
                w.scale[2 * sub + 1] * (x.x * dot_hi - 32.0f * x.z));
    } else if constexpr (std::is_same_v<W, BlockQ6_K>) {
        int part = sub & 3;
        const uint8_t* ql = w.ql + (sub >> 2) * 64 + (part & 1) * 32;
        const uint8_t* qh = w.qh + (sub >> 2) * 32;
        int dot_lo = dot_q8_words(unpack_q6_words(ql, qh, part), lo);
        int dot_hi = dot_q8_words(unpack_q6_words(ql + 16, qh + 16, part), hi);
        return half_bits_to_f32(__ldg(&w.d)) *
               (__ldg(&w.scales[2 * sub]) * (x.x * dot_lo - 32.0f * x.y) +
                __ldg(&w.scales[2 * sub + 1]) * (x.x * dot_hi - 32.0f * x.z));
    } else {
        const auto* qs = reinterpret_cast<const uint4*>(w.qs + (sub >> 1) * 32);
        uint4 q_lo = unpack_q4_words(__ldg(qs), (sub & 1) * 4);
        uint4 q_hi = unpack_q4_words(__ldg(qs + 1), (sub & 1) * 4);
        if constexpr (std::is_same_v<W, BlockQ5_K>) {
            uint4 h_lo = __ldg(reinterpret_cast<const uint4*>(w.qh));
            uint4 h_hi = __ldg(reinterpret_cast<const uint4*>(w.qh) + 1);
            constexpr unsigned mask = 0x01010101u;
            q_lo.x |= ((h_lo.x >> sub) & mask) << 4;
            q_lo.y |= ((h_lo.y >> sub) & mask) << 4;
            q_lo.z |= ((h_lo.z >> sub) & mask) << 4;
            q_lo.w |= ((h_lo.w >> sub) & mask) << 4;
            q_hi.x |= ((h_hi.x >> sub) & mask) << 4;
            q_hi.y |= ((h_hi.y >> sub) & mask) << 4;
            q_hi.z |= ((h_hi.z >> sub) & mask) << 4;
            q_hi.w |= ((h_hi.w >> sub) & mask) << 4;
        }
        int dot = dot_q8_words(q_lo, lo) + dot_q8_words(q_hi, hi);
        int sc, mn;
        if (sub < 4) {
            sc = w.scales[sub] & 63;
            mn = w.scales[sub + 4] & 63;
        } else {
            sc = (w.scales[sub + 4] & 15) | ((w.scales[sub - 4] >> 6) << 4);
            mn = (w.scales[sub + 4] >> 4) | ((w.scales[sub] >> 6) << 4);
        }
        return half_bits_to_f32(w.d) * sc * x.x * dot - half_bits_to_f32(w.dmin) * mn * x.w;
    }
}

}  // namespace

namespace matmul {

void launch_gemm_tiled_f32(const float* x, const float* w, float* y,
                           int M, int N, int K, cudaStream_t stream);

template <int WK, bool HAS_NORM>
void launch_wmma(const float* x, const float* norm_w, const void* w, float* y,
                 int M, int N, int K, float eps, cudaStream_t stream);

void launch_mmq_q4_a_and_q8_a(const BlockQ8_1* x, const BlockQ4_A* w4, const BlockQ8_A* w8,
                              float* y4, float* y8, int M, int N4, int N8, int K, cudaStream_t stream);

}  // namespace matmul

}  // namespace fusionrt::cuda_ops

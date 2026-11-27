#include "weight_tiles.cuh"
#include "fusionrt/weight_codec.h"

#include <mma.h>

#include <algorithm>
#include <iterator>
#include <utility>

namespace fusionrt::cuda_ops {

namespace {

__global__ void k_linear_gemm_tiled_f32(const float* __restrict__ x, const float* __restrict__ w,
                                        float* __restrict__ y, int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float sx[TILE][TILE];
    __shared__ float sw[TILE][TILE];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;  // n
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        int kx = t + threadIdx.x;
        int kw = t + threadIdx.y;
        sx[threadIdx.y][threadIdx.x] = (row < M && kx < K) ? x[row * K + kx] : 0.0f;
        sw[threadIdx.y][threadIdx.x] = (col < N && kw < K) ? w[static_cast<int64_t>(col) * K + kw] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            acc += sx[threadIdx.y][k] * sw[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        y[row * N + col] = acc;
    }
}

template <int WK, bool HAS_NORM>
__global__ void k_gemm_wmma(const float* __restrict__ x,
                            const float* __restrict__ norm_w,
                            const void* __restrict__ w,
                            float* __restrict__ y,
                            int M, int N, int K, float eps) {
    // Tile 32(M) x 64(N) x 32(K) using 4 warps (128 threads), 2x2 warp grid of 16x32 each
    __shared__ half sA[32][40];
    __shared__ half sB[64][40];
    __shared__ float sC[4][256];
    __shared__ float red_buf[32];
    __shared__ float inv_rms_sh[32];

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int bm = blockIdx.y * 32;
    int bn = blockIdx.x * 64;

    if constexpr (HAS_NORM) {
        int rows_here = min(32, M - bm);
        for (int lm = 0; lm < rows_here; ++lm) {
            const float* xm = x + static_cast<int64_t>(bm + lm) * K;
            float sum_sq = 0.0f;
            int k4 = K >> 2;
            const float4* xm4 = reinterpret_cast<const float4*>(xm);
            for (int i = tid; i < k4; i += 128) {
                float4 v = __ldg(xm4 + i);
                sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
            for (int k = (k4 << 2) + tid; k < K; k += 128) {
                float v = __ldg(xm + k);
                sum_sq += v * v;
            }
            float tot = block_reduce_sum(sum_sq, red_buf);
            if (tid == 0) inv_rms_sh[lm] = rsqrtf(tot / static_cast<float>(K) + eps);
        }
        __syncthreads();
    }

    int warp_m = wid >> 1;
    int warp_n = (wid & 1) * 2;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2];
    nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
    nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);

    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int idx = tid; idx < 32 * 32; idx += 128) {
            int lm = idx >> 5;
            int c = idx & 31;
            int gm = bm + lm;
            int gk = k0 + c;
            float v = (gm < M && gk < K) ? __ldg(&x[static_cast<int64_t>(gm) * K + gk]) : 0.0f;
            if constexpr (HAS_NORM) {
                if (gm < M && gk < K) v *= inv_rms_sh[lm] * __ldg(&norm_w[gk]);
            }
            sA[lm][c] = __float2half_rn(v);
        }
        for (int idx = tid; idx < 64 * 32; idx += 128) {
            int ln = idx >> 5;
            int c = idx & 31;
            int gn = bn + ln;
            float wv = (gn < N && (k0 + c) < K) ? dequant_weight_elem<WK>(w, gn, k0, c, K) : 0.0f;
            sB[ln][c] = __float2half_rn(wv);
        }
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 32; ks += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::load_matrix_sync(a_frag, &sA[warp_m * 16][ks], 40);
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> b_frag;
                nvcuda::wmma::load_matrix_sync(b_frag, &sB[(warp_n + j) * 16][ks], 40);
                nvcuda::wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
            }
        }
        __syncthreads();
    }

    int lane = tid & 31;
    #pragma unroll
    for (int j = 0; j < 2; ++j) {
        nvcuda::wmma::store_matrix_sync(sC[wid], c_frag[j], 16, nvcuda::wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 256; idx += 32) {
            int lm = warp_m * 16 + (idx >> 4);
            int ln = (warp_n + j) * 16 + (idx & 15);
            int gm = bm + lm;
            int gn = bn + ln;
            if (gm < M && gn < N) {
                y[static_cast<int64_t>(gm) * N + gn] = sC[wid][idx];
            }
        }
        __syncwarp();
    }
}

struct MmqWeight {
    unsigned lo, hi;
    float scale, other;
};

template <class WC>
struct MmqCodecBlock {
    using Codec = WC;
    uint8_t packed[WC::kBlockBytes];
};

template <class W>
constexpr bool mmq_codec = requires { typename W::Codec; };

template <typename W>
constexpr bool mmq_packed_q4 =
    std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ4_K> || std::is_same_v<W, BlockQ4_KA>;

template <typename W>
constexpr bool mmq_async = mmq_packed_q4<W> || std::is_same_v<W, BlockQ6_A> || std::is_same_v<W, BlockQ5_K>;

template <typename W>
constexpr bool mmq_has_min =
    std::is_same_v<W, BlockQ4_K> || std::is_same_v<W, BlockQ4_KA> || std::is_same_v<W, BlockQ5_K>;

template <typename W>
constexpr bool mmq_split_scale = std::is_same_v<W, BlockQ6_K> || std::is_same_v<W, BlockQ6_A>;

template <typename W>
__device__ __forceinline__ float2 mmq_k_scales(const W& w, int sub) {
    const unsigned scale_min = unpack_k_scale_min(w.scales, sub);
    return make_float2(half_bits_to_f32(w.d) * (scale_min & 255),
                      -half_bits_to_f32(w.dmin) * (scale_min >> 8));
}

__device__ __forceinline__ unsigned load_word_unaligned(const uint8_t* p) {
    auto address = reinterpret_cast<uintptr_t>(p);
    auto* aligned = reinterpret_cast<const unsigned*>(address & ~uintptr_t{3});
    return __funnelshift_r(__ldg(aligned), __ldg(aligned + 1), (address & 2) * 8);
}

__device__ __forceinline__ unsigned mmq_signed_q4(unsigned q) {
    return ((q & kNibbleMask) + 0x78787878u) ^ 0x80808080u;
}

__device__ __forceinline__ MmqWeight load_raw_q6_mmq_weight(const BlockQ6_K& w, int sub, int word) {
    MmqWeight v;
    int part = sub & 3;
    auto* ql = w.ql + (sub >> 2) * 64 + (part & 1) * 32 + 4 * word;
    auto* qh = w.qh + (sub >> 2) * 32 + 4 * word;
    unsigned lo = load_word_unaligned(ql);
    unsigned hi = load_word_unaligned(ql + 16);
    unsigned h0 = load_word_unaligned(qh);
    unsigned h1 = load_word_unaligned(qh + 16);
    v.lo = ((((lo >> ((part >> 1) * 4)) & kNibbleMask) |
             (((h0 >> (part * 2)) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
    v.hi = ((((hi >> ((part >> 1) * 4)) & kNibbleMask) |
             (((h1 >> (part * 2)) & 0x03030303u) << 4)) + 0x60606060u) ^ 0x80808080u;
    float d = half_bits_to_f32(__ldg(&w.d));
    v.scale = d * __ldg(w.scales + 2 * sub);
    v.other = d * __ldg(w.scales + 2 * sub + 1);
    return v;
}

__device__ __forceinline__ int4 mma_s8(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                       unsigned b0, unsigned b1) {
    int4 c;
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
    return c;
}

__device__ __forceinline__ int4 mma_s8_k16(unsigned a0, unsigned a1, unsigned b0) {
    int4 c;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w)
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
    return c;
}

template <int Rows, int BK, int Words, bool Other>
struct MmqTile {
    // One extra 16-byte segment disjoins the eight rows of an ldmatrix across banks.
    alignas(16) unsigned qs[Rows][BK * Words + 4];
    float scale[BK][Rows + 32 / BK];
    float other[Other ? BK : 1][Other ? Rows + 32 / BK : 1];
};

template <int Bytes>
__device__ __forceinline__ void mmq_copy_async(void* dst, const void* src) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" :: "r"(smem), "l"(src), "n"(Bytes) : "memory");
}

template <typename W, int BM, int BN, int BK>
__device__ __forceinline__ void load_mmq_tile(
    const BlockQ8_1* x, const W* w,
    MmqTile<BM, BK, 8, mmq_has_min<W>>& xs,
    MmqTile<BN, BK, mmq_packed_q4<W> ? 4 : 8, mmq_has_min<W> || mmq_split_scale<W>>& ws,
    int m0, int n0, int M, int N, int K, int kb) {
    constexpr bool async = mmq_async<W>;
    int slot = threadIdx.x % BK;
    int xstride = K / 32;
    const int xsub = mmq_codec<W> ? min(kb + slot, xstride - 1) : kb + slot;
    int sub = (kb + slot) % 8;
    for (int m = threadIdx.x / BK; m < BM; m += blockDim.x / BK) {
        const auto* src = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(min(m0 + m, M - 1)) * xstride + xsub);
        auto* dst = reinterpret_cast<uint4*>(xs.qs[m] + slot * 8);
        if constexpr (async) {
            mmq_copy_async<16>(dst, src);
            mmq_copy_async<16>(dst + 1, src + 1);
            mmq_copy_async<4>(&xs.scale[slot][m], &src[2].x);
        } else {
            dst[0] = __ldg(src);
            dst[1] = __ldg(src + 1);
            xs.scale[slot][m] = !mmq_codec<W> || kb + slot < xstride ?
                                __uint_as_float(__ldg(&src[2].x)) : 0.0f;
        }
        if constexpr (mmq_has_min<W>) {
            if constexpr (async) mmq_copy_async<4>(&xs.other[slot][m], &src[2].w);
            else xs.other[slot][m] = __uint_as_float(__ldg(&src[2].w));
        }
    }
    for (int n = threadIdx.x / BK; n < BN; n += blockDim.x / BK) {
        constexpr int block_elems = [] {
            if constexpr (mmq_codec<W>) return W::Codec::kBlockSize;
            else return 256;
        }();
        const W& src = w[static_cast<int64_t>(min(n0 + n, N - 1)) * (K / block_elems) + xsub / (block_elems / 32)];
        constexpr int words = mmq_packed_q4<W> ? 4 : 8;
        auto* dst = reinterpret_cast<uint4*>(ws.qs[n] + slot * words);
        if constexpr (mmq_codec<W>) {
            using Codec = typename W::Codec;
            const int part = xsub % (block_elems / 32);
            dst[0] = make_uint4(Codec::int8_word(src.packed, part, 0), Codec::int8_word(src.packed, part, 1),
                                Codec::int8_word(src.packed, part, 2), Codec::int8_word(src.packed, part, 3));
            dst[1] = make_uint4(Codec::int8_word(src.packed, part, 4), Codec::int8_word(src.packed, part, 5),
                                Codec::int8_word(src.packed, part, 6), Codec::int8_word(src.packed, part, 7));
            ws.scale[slot][n] = Codec::int8_scale(src.packed, part);
        } else if constexpr (TileQ4_144<W>) {
            using Tile = TileQ4Trait<W>;
            mmq_copy_async<16>(dst, Tile::nibbles(src, sub));
            if constexpr (Tile::symmetric) {
                const auto* scale = reinterpret_cast<const unsigned*>(src.d) + sub / 2;
                mmq_copy_async<4>(&ws.scale[slot][n], scale);
            } else {
                const float2 scales = Tile::scales(src, sub);
                ws.scale[slot][n] = scales.x;
                ws.other[slot][n] = -scales.y;
            }
        } else if constexpr (std::is_same_v<W, BlockQ4_K>) {
            // Adjacent slots stage the two aligned halves of a packed Q4_K pair.
            mmq_copy_async<16>(dst, reinterpret_cast<const uint4*>(src.qs) + sub);
            const float2 scales = mmq_k_scales(src, sub);
            ws.scale[slot][n] = scales.x;
            ws.other[slot][n] = scales.y;
        } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
            mmq_copy_async<16>(dst, src.ql[sub]);
            mmq_copy_async<16>(dst + 1, src.qh[sub / 2]);
            const float d = half_bits_to_f32(src.d);
            ws.scale[slot][n] = d * src.scale[2 * sub];
            ws.other[slot][n] = d * src.scale[2 * sub + 1];
        } else if constexpr (std::is_same_v<W, BlockQ5_K>) {
            // Adjacent slots stage the low/high halves consumed by each 32-element dot.
            mmq_copy_async<16>(dst, reinterpret_cast<const uint4*>(src.qs) + sub);
            mmq_copy_async<16>(dst + 1, reinterpret_cast<const uint4*>(src.qh) + sub % 2);
            const float2 scales = mmq_k_scales(src, sub);
            ws.scale[slot][n] = scales.x;
            ws.other[slot][n] = scales.y;
        } else if constexpr (std::is_same_v<W, BlockQ8_A>) {
            auto* ptr = reinterpret_cast<const uint4*>(src.qs) + 2 * sub;
            dst[0] = __ldg(ptr);
            dst[1] = __ldg(ptr + 1);
            ws.scale[slot][n] = half_bits_to_f32(__ldg(src.d + sub));
        } else {
            MmqWeight a = load_raw_q6_mmq_weight(src, sub, 0);
            MmqWeight b = load_raw_q6_mmq_weight(src, sub, 1);
            MmqWeight c = load_raw_q6_mmq_weight(src, sub, 2);
            MmqWeight d = load_raw_q6_mmq_weight(src, sub, 3);
            dst[0] = make_uint4(a.lo, b.lo, c.lo, d.lo);
            dst[1] = make_uint4(a.hi, b.hi, c.hi, d.hi);
            ws.scale[slot][n] = a.scale;
            ws.other[slot][n] = a.other;
        }
    }
    if constexpr (async) asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ uint4 mmq_load_a(const unsigned* ptr) {
    uint4 r;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "r"(shared));
    return r;
}

__device__ __forceinline__ uint2 mmq_load_b(const unsigned* ptr) {
    uint2 r;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r.x), "=r"(r.y) : "r"(shared));
    return r;
}

__device__ __forceinline__ unsigned mmq_load_packed_q4(const unsigned* ptr) {
    unsigned q;
    unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(ptr));
    asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(q) : "r"(shared));
    return q;
}

template <typename W, int BM, int BN, int BK>
struct MmqBuffers {
    static constexpr bool split_scale = mmq_split_scale<W>;
    static constexpr bool min_scale = mmq_has_min<W>;
    static constexpr int weight_words = mmq_packed_q4<W> ? 4 : 8;

    MmqTile<BM, BK, 8, min_scale> xs[2];
    MmqTile<BN, BK, weight_words, split_scale || min_scale> ws[2];
};

template <typename W, int BM, int BN, int WN, int BK, int WM = BM / 16>
__device__ __forceinline__ void mmq_q8_1_tile(const BlockQ8_1* __restrict__ x, const W* __restrict__ w,
    float* __restrict__ y, int M, int N, int K, int m0, int n0, MmqBuffers<W, BM, BN, BK>& storage) {
    constexpr int warps_n = BN / (WN * 8);
    constexpr bool split_scale = MmqBuffers<W, BM, BN, BK>::split_scale;
    constexpr bool min_scale = MmqBuffers<W, BM, BN, BK>::min_scale;
    auto& xs = storage.xs;
    auto& ws = storage.ws;
    int warp = threadIdx.x / 32;
    int warp_n = (warp % warps_n) * (WN * 8);
    int warp_m = (warp / warps_n) * (WM * 16);
    int lane = threadIdx.x % 32, group = lane / 4, word = lane % 4;
    float4 acc[WM][WN] = {};

    load_mmq_tile<W, BM, BN, BK>(x, w, xs[0], ws[0], m0, n0, M, N, K, 0);
    if constexpr (mmq_async<W>) asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int kb = 0; kb < K / 32; kb += BK) {
        const auto& a = xs[(kb / BK) & 1];
        const auto& b = ws[(kb / BK) & 1];
        if (kb + BK < K / 32) {
            load_mmq_tile<W, BM, BN, BK>(x, w, xs[1 - ((kb / BK) & 1)], ws[1 - ((kb / BK) & 1)],
                                      m0, n0, M, N, K, kb + BK);
        }

        #pragma unroll
        for (int slot = 0; slot < BK; ++slot) {
            uint2 weights[WN];
            float2 scales[WN], other[WN];
            #pragma unroll
            for (int ni = 0; ni < WN; ++ni) {
                int j = warp_n + ni * 8;
                int j0 = j + word * 2, j1 = j0 + 1;
                uint2 bv;
                if constexpr (TileQ4_144<W>) {
                    const unsigned packed = mmq_load_packed_q4(b.qs[j + lane % 8] + slot * 4);
                    if constexpr (TileQ4Trait<W>::symmetric)
                        bv = make_uint2(mmq_signed_q4(packed), mmq_signed_q4(packed >> 4));
                    else
                        bv = make_uint2(unpack_q4_words(packed, 0), unpack_q4_words(packed, 4));
                } else if constexpr (std::is_same_v<W, BlockQ6_A>) {
                    const unsigned* row = b.qs[j + lane % 8] + slot * 8;
                    const auto low = mmq_load_packed_q4(row);
                    const auto high = mmq_load_packed_q4(row + 4);
                    bv = make_uint2((unpack_q6_a(low, high, 0, slot % 2) + 0x60606060u) ^ 0x80808080u,
                                    (unpack_q6_a(low, high, 1, slot % 2) + 0x60606060u) ^ 0x80808080u);
                } else if constexpr (std::is_same_v<W, BlockQ4_K>) {
                    const auto* row = b.qs[j + lane % 8] + (slot & ~1) * 4;
                    const uint2 packed = mmq_load_b(row + ((lane / 8) % 2) * 4);
                    const int shift = (slot & 1) * 4;
                    bv = make_uint2((packed.x >> shift) & kNibbleMask, (packed.y >> shift) & kNibbleMask);
                } else if constexpr (std::is_same_v<W, BlockQ5_K>) {
                    const auto* row = b.qs[j + lane % 8] + ((slot & ~1) + (lane / 8) % 2) * 8;
                    const uint2 packed = mmq_load_b(row);
                    const uint2 high = mmq_load_b(row + 4);
                    const int shift = (slot & 1) * 4;
                    const int bit = (kb + slot) % 8;
                    bv = make_uint2(((packed.x >> shift) & kNibbleMask) | (((high.x >> bit) & 0x01010101u) << 4),
                                    ((packed.y >> shift) & kNibbleMask) | (((high.y >> bit) & 0x01010101u) << 4));
                } else {
                    bv = mmq_load_b(b.qs[j + lane % 8] + slot * 8 + ((lane / 8) % 2) * 4);
                }
                weights[ni] = bv;
                if constexpr (std::is_same_v<W, BlockQ4_A>) {
                    int shift = (slot % 2) * 16;
                    scales[ni] = make_float2(
                        half_bits_to_f32(static_cast<uint16_t>(__float_as_uint(b.scale[slot][j0]) >> shift)),
                        half_bits_to_f32(static_cast<uint16_t>(__float_as_uint(b.scale[slot][j1]) >> shift)));
                } else {
                    scales[ni] = make_float2(b.scale[slot][j0], b.scale[slot][j1]);
                }
                if constexpr (split_scale || min_scale) {
                    other[ni] = make_float2(b.other[slot][j0], b.other[slot][j1]);
                }
            }
            #pragma unroll
            for (int mi = 0; mi < WM; ++mi) {
                const int row = warp_m + mi * 16;
                const int i0 = row + group, i1 = i0 + 8;
                const uint4 av = mmq_load_a(a.qs[row + lane % 16] + slot * 8 + (lane / 16) * 4);
                const float ax = a.scale[slot][i0], ay = a.scale[slot][i1];
                float sx, sy;
                if constexpr (min_scale) {
                    sx = a.other[slot][i0];
                    sy = a.other[slot][i1];
                }
                #pragma unroll
                for (int ni = 0; ni < WN; ++ni) {
                    const uint2 bv = weights[ni];
                    const float bx = scales[ni].x, by = scales[ni].y;
                    int4 c;
                    if constexpr (std::is_same_v<W, BlockQ6_A>) c = mma_s8_k16(av.x, av.y, bv.x);
                    else c = mma_s8(av.x, av.y, av.z, av.w, bv.x, split_scale ? 0u : bv.y);
                    auto& f = acc[mi][ni];
                    if constexpr (split_scale) {
                        int4 hi;
                        if constexpr (std::is_same_v<W, BlockQ6_A>) hi = mma_s8_k16(av.z, av.w, bv.y);
                        else hi = mma_s8(0, 0, av.z, av.w, 0, bv.y);
                        float hx = other[ni].x, hy = other[ni].y;
                        f.x = fmaf(ax, fmaf(hx, hi.x, bx * c.x), f.x);
                        f.y = fmaf(ax, fmaf(hy, hi.y, by * c.y), f.y);
                        f.z = fmaf(ay, fmaf(hx, hi.z, bx * c.z), f.z);
                        f.w = fmaf(ay, fmaf(hy, hi.w, by * c.w), f.w);
                    } else {
                        f.x = fmaf(ax * bx, c.x, f.x);
                        f.y = fmaf(ax * by, c.y, f.y);
                        f.z = fmaf(ay * bx, c.z, f.z);
                        f.w = fmaf(ay * by, c.w, f.w);
                        if constexpr (min_scale) {
                            float dx = other[ni].x, dy = other[ni].y;
                            f.x = fmaf(sx, dx, f.x);
                            f.y = fmaf(sx, dy, f.y);
                            f.z = fmaf(sy, dx, f.z);
                            f.w = fmaf(sy, dy, f.w);
                        }
                    }
                }
            }
        }
        if constexpr (mmq_async<W>) asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < WM; ++mi) {
        int i0 = m0 + warp_m + mi * 16 + group, i1 = i0 + 8;
        #pragma unroll
        for (int ni = 0; ni < WN; ++ni) {
            int j0 = n0 + warp_n + ni * 8 + word * 2, j1 = j0 + 1;
            const auto& f = acc[mi][ni];
            if (i0 < M && j0 < N) y[static_cast<int64_t>(i0) * N + j0] = f.x;
            if (i0 < M && j1 < N) y[static_cast<int64_t>(i0) * N + j1] = f.y;
            if (i1 < M && j0 < N) y[static_cast<int64_t>(i1) * N + j0] = f.z;
            if (i1 < M && j1 < N) y[static_cast<int64_t>(i1) * N + j1] = f.w;
        }
    }
}

template <typename W, int BM, int BN, int WN, int BK = 4, int WM = BM / 16>
__global__ void k_mmq_q8_1(const BlockQ8_1* x, const W* w, float* y, int M, int N, int K) {
    using Buffers = MmqBuffers<W, BM, BN, BK>;
    if constexpr (sizeof(Buffers) > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char mmq_smem[];
        auto& storage = *reinterpret_cast<Buffers*>(mmq_smem);
        mmq_q8_1_tile<W, BM, BN, WN, BK, WM>(x, w, y, M, N, K, blockIdx.y * BM, blockIdx.x * BN, storage);
    } else {
        __shared__ Buffers storage;
        mmq_q8_1_tile<W, BM, BN, WN, BK, WM>(x, w, y, M, N, K, blockIdx.y * BM, blockIdx.x * BN, storage);
    }
}

// Use the same tiles as the large Q4 projection and queue the narrow Q8 projection in the same launch.
__global__ void k_mmq_q4_a_and_q8_a(const BlockQ8_1* x, const BlockQ4_A* w4, const BlockQ8_A* w8,
                                     float* y4, float* y8, int M, int N4, int N8, int K,
                                     int q4_tiles, int q4_blocks) {
    __shared__ union {
        MmqBuffers<BlockQ4_A, 64, 128, 4> q4;
        MmqBuffers<BlockQ8_A, 16, 64, 4> q8;
    } storage;
    const int block = blockIdx.x;
    if (block < q4_blocks) {
        mmq_q8_1_tile<BlockQ4_A, 64, 128, 2, 4>(x, w4, y4, M, N4, K,
            block / q4_tiles * 64, block % q4_tiles * 128, storage.q4);
    } else {
        mmq_q8_1_tile<BlockQ8_A, 16, 64, 1, 4>(x, w8, y8, M, N8, K,
            (block - q4_blocks) * 16, 0, storage.q8);
    }
}

template <typename W, int BM, int BN, int WN, int BK = 4, int WM = BM / 16>
void launch_mmq_tile(const BlockQ8_1* x, const W* w, float* y, int M, int N, int K, cudaStream_t stream) {
    constexpr int threads = (BM / (WM * 16)) * (BN / (WN * 8)) * 32;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    constexpr int bytes = sizeof(MmqBuffers<W, BM, BN, BK>);
    constexpr int dynamic_bytes = bytes > 48 * 1024 ? bytes : 0;
    if constexpr (dynamic_bytes > 0) {
        static thread_local int configured_device = -1;
        static thread_local bool fits = false;
        int device = 0;
        FUSIONRT_CUDA_CHECK(cudaGetDevice(&device));
        if (configured_device != device) {
            int limit = 0;
            FUSIONRT_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            fits = dynamic_bytes <= limit;
            if (fits) {
                FUSIONRT_CUDA_CHECK(cudaFuncSetAttribute(k_mmq_q8_1<W, BM, BN, WN, BK, WM>,
                                                         cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes));
            }
            configured_device = device;
        }
        if (!fits) {
            launch_mmq_tile<W, 64, 64, 2, 4, 2>(x, w, y, M, N, K, stream);
            return;
        }
    }
    k_mmq_q8_1<W, BM, BN, WN, BK, WM><<<grid, threads, dynamic_bytes, stream>>>(x, w, y, M, N, K);
}

template <typename W, MmqTileConfig Config>
constexpr bool mmq_tile_compiled() {
    using enum MmqTileConfig;
    constexpr bool scaled = std::is_same_v<W, BlockQ4_KA> || std::is_same_v<W, BlockQ5_K>;
    switch (Config) {
        case M128N96: case M128N128: return scaled;
        case M128N64: case M64N64: return scaled || std::is_same_v<W, BlockQ6_A>;
        case M64N128: return std::is_same_v<W, BlockQ4_A>;
        case M64N64K256: return std::is_same_v<W, BlockQ8_A>;
        case M64N64W1: return std::is_same_v<W, BlockQ4_K>;
        default: return true;
    }
}

template <typename W, size_t... I>
constexpr auto mmq_tile_menu(std::index_sequence<I...>) {
    struct Menu {
        unsigned mask;
        int shared[sizeof...(I)];
    };
    constexpr auto& tiles = cost_detail::mmq_tiles;
    return Menu{
        (0u | ... | (mmq_tile_compiled<W, static_cast<MmqTileConfig>(I)>() ? 1u << I : 0u)),
        {sizeof(MmqBuffers<W, tiles[I].m, tiles[I].n, tiles[I].k_groups>)...},
    };
}

template <typename W>
void launch_mmq(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                const CostHardware& hw) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    auto* qx = static_cast<const BlockQ8_1*>(x);
    auto* qw = static_cast<const W*>(w);
    constexpr auto menu = mmq_tile_menu<W>(std::make_index_sequence<std::size(cost_detail::mmq_tiles)>{});
    constexpr bool wide = mmq_tile_compiled<W, MmqTileConfig::M128N64>() ||
                          mmq_tile_compiled<W, MmqTileConfig::M64N128>();
    const auto tile = select_mmq_tile(hw, M, N, K, sizeof(W), wide, mmq_async<W>, menu.mask, menu.shared);

    auto launch = [&]<MmqTileConfig Config> {
        if constexpr (mmq_tile_compiled<W, Config>()) {
            constexpr auto t = cost_detail::mmq_tiles[static_cast<unsigned>(Config)];
            launch_mmq_tile<W, t.m, t.n, t.warp_n, t.k_groups, t.warp_m>(qx, qw, y, M, N, K, stream);
        }
    };
    using enum MmqTileConfig;
    switch (tile) {
        case M16N32:   return launch.template operator()<M16N32>();
        case M32N32:   return launch.template operator()<M32N32>();
        case M32N64:   return launch.template operator()<M32N64>();
        case M64N32:   return launch.template operator()<M64N32>();
        case M64N64:   return launch.template operator()<M64N64>();
        case M64N64K256: return launch.template operator()<M64N64K256>();
        case M64N64W1: return launch.template operator()<M64N64W1>();
        case M64N128:  return launch.template operator()<M64N128>();
        case M128N64:  return launch.template operator()<M128N64>();
        case M128N96:  return launch.template operator()<M128N96>();
        case M128N128: return launch.template operator()<M128N128>();
    }
}

__global__ void k_dequant_q8_a_to_f16(const BlockQ8_A* __restrict__ w, half* __restrict__ out, int N, int K) {
    int ns = K >> 8;
    int s_idx = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_s = N * ns;
    if (s_idx >= total_s) return;
    const BlockQ8_A& blk = w[s_idx];
    half* dst = out + static_cast<int64_t>(s_idx) * 256;
    for (int sb = 0; sb < 8; ++sb) {
        float d = half_bits_to_f32(__ldg(&blk.d[sb]));
        int8_t q = reinterpret_cast<const int8_t*>(&blk.qs[sb * 8])[lane];
        dst[sb * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
    }
}

__global__ void k_dequant_q4_0_to_f16(const BlockQ4_0* __restrict__ w, half* __restrict__ out, int N, int K) {
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_b = N * (K >> 5);
    if (b >= total_b) return;
    const BlockQ4_0& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    uint8_t byte = __ldg(&blk.qs[lane & 15]);
    int q = (lane < 16) ? (static_cast<int>(byte & 0x0Fu) - 8) : (static_cast<int>(byte >> 4) - 8);
    out[static_cast<int64_t>(b) * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
}

__global__ void k_dequant_q8_0_to_f16(const BlockQ8_0* __restrict__ w, half* __restrict__ out, int N, int K) {
    int b = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int total_b = N * (K >> 5);
    if (b >= total_b) return;
    const BlockQ8_0& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    int8_t q = __ldg(&blk.qs[lane]);
    out[static_cast<int64_t>(b) * 32 + lane] = __float2half_rn(d * static_cast<float>(q));
}

__global__ void k_dequant_q4_1_to_f16(const BlockQ4_1* __restrict__ w, half* __restrict__ out, int total_b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / 4, word = tid % 4;
    if (b >= total_b) return;
    const BlockQ4_1& blk = w[b];
    float d = half_bits_to_f32(__ldg(&blk.d));
    float m = half_bits_to_f32(__ldg(&blk.m));
    unsigned packed = __ldg(reinterpret_cast<const unsigned*>(blk.qs) + word);
    half2* dst = reinterpret_cast<half2*>(out + static_cast<int64_t>(b) * 32 + word * 4);
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        unsigned q = (packed >> (part * 4)) & kNibbleMask;
        dst[part * 8] = __floats2half2_rn(fmaf(d, q & 15u, m), fmaf(d, (q >> 8) & 15u, m));
        dst[part * 8 + 1] = __floats2half2_rn(fmaf(d, (q >> 16) & 15u, m), fmaf(d, q >> 24, m));
    }
}

template <int WK>
__global__ void k_dequant_generic_to_f16(const void* __restrict__ w, half* __restrict__ out, int N, int K) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * K;
    if (idx >= total) return;
    int n = static_cast<int>(idx / K);
    int k = static_cast<int>(idx % K);
    int k0 = k & ~31;
    int c = k & 31;
    out[idx] = __float2half_rn(dequant_weight_elem<WK>(w, n, k0, c, K));
}

__global__ void k_dequant_f32_or_bf16_to_f16(const void* __restrict__ w, int is_bf16, half* __restrict__ out, int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = is_bf16
        ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w)[idx])
        : reinterpret_cast<const float*>(w)[idx];
    out[idx] = __float2half_rn(v);
}

__global__ void k_rmsnorm_or_copy_to_f16(const float* __restrict__ x,
                                         const float* __restrict__ norm_w,
                                         half* __restrict__ out,
                                         int K, float eps) {
    __shared__ float smem[32];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    const float* xm = x + static_cast<int64_t>(m) * K;
    half* ym = out + static_cast<int64_t>(m) * K;
    if (norm_w != nullptr) {
        float sum_sq = 0.0f;
        for (int c = tid; c < K; c += blockDim.x) {
            float v = __ldg(&xm[c]);
            sum_sq += v * v;
        }
        float total_sq = block_reduce_sum(sum_sq, smem);
        float inv = rsqrtf(total_sq / static_cast<float>(K) + eps);
        for (int c = tid; c < K; c += blockDim.x) {
            ym[c] = __float2half_rn(__ldg(&xm[c]) * inv * __ldg(&norm_w[c]));
        }
    } else {
        for (int c = tid; c < K; c += blockDim.x) {
            ym[c] = __float2half_rn(__ldg(&xm[c]));
        }
    }
}

}  // namespace

namespace matmul {

void launch_gemm_tiled_f32(const float* x, const float* w, float* y,
                           int M, int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    k_linear_gemm_tiled_f32<<<grid, block, 0, stream>>>(x, w, y, M, N, K);
}

template <int WK, bool HAS_NORM>
void launch_wmma(const float* x, const float* norm_w, const void* w, float* y,
                 int M, int N, int K, float eps, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((N + 63) / 64, (M + 31) / 32);
    k_gemm_wmma<WK, HAS_NORM><<<grid, block, 0, stream>>>(x, norm_w, w, y, M, N, K, eps);
}

template void launch_wmma<0, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<0, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<1, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<1, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<2, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<2, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<3, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<3, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<4, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<4, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<5, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<5, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<6, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<6, true>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<7, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<8, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<9, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);
template void launch_wmma<10, false>(const float*, const float*, const void*, float*, int, int, int, float, cudaStream_t);

void launch_mmq_q4_a_and_q8_a(const BlockQ8_1* x, const BlockQ4_A* w4, const BlockQ8_A* w8,
                              float* y4, float* y8, int M, int N4, int N8, int K, cudaStream_t stream) {
    const int q4_tiles = (N4 + 127) / 128;
    const int q4_blocks = q4_tiles * ((M + 63) / 64);
    k_mmq_q4_a_and_q8_a<<<q4_blocks + (M + 15) / 16, 256, 0, stream>>>(
        x, w4, w8, y4, y8, M, N4, N8, K, q4_tiles, q4_blocks);
}

}  // namespace matmul

void mmq_q4_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ4_A>(x, w, y, M, N, K, stream, hw);
}

void codec_mmq_q8_1(const void* x, const void* w, DType dtype, float* y, int M, int N, int K,
                     cudaStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    visit_weight_codec(dtype, [&]<class Codec>() {
        if constexpr (Codec::kHasInt8Sub32) {
            if (K % Codec::kBlockSize) throw std::invalid_argument("Invalid WeightCodec MMQ column count");
            using W = MmqCodecBlock<Codec>;
            const auto* qx = static_cast<const BlockQ8_1*>(x);
            const auto* qw = static_cast<const W*>(w);
            if (M <= 16) launch_mmq_tile<W, 16, 32, 1>(qx, qw, y, M, N, K, stream);
            else if (M <= 32) launch_mmq_tile<W, 32, 64, 2>(qx, qw, y, M, N, K, stream);
            else launch_mmq_tile<W, 64, 64, 2, 4, 2>(qx, qw, y, M, N, K, stream);
        } else throw std::invalid_argument("WeightCodec has no INT8 MMQ basis");
    });
}

void mmq_q8_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ8_A>(x, w, y, M, N, K, stream, hw);
}

void mmq_q6_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ6_K>(x, w, y, M, N, K, stream, hw);
}

void mmq_q4_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ4_K>(x, w, y, M, N, K, stream, hw);
}

void mmq_q4_ka_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ4_KA>(x, w, y, M, N, K, stream, hw);
}

void mmq_q6_a_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ6_A>(x, w, y, M, N, K, stream, hw);
}

void mmq_q5_k_q8_1(const void* x, const void* w, float* y, int M, int N, int K, cudaStream_t stream,
                     const CostHardware& hw) {
    launch_mmq<BlockQ5_K>(x, w, y, M, N, K, stream, hw);
}

void dequantize_to_f16(const void* w, DType dt, uint16_t* out_f16, int N, int K, cudaStream_t stream) {
    half* out = reinterpret_cast<half*>(out_f16);
    if (dt == DType::Q8_A) {
        int total_s = N * (K >> 8);
        dim3 block(32, 4);
        k_dequant_q8_a_to_f16<<<(total_s + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ8_A*>(w), out, N, K);
    } else if (dt == DType::Q4_0) {
        int total_b = N * (K >> 5);
        dim3 block(32, 4);
        k_dequant_q4_0_to_f16<<<(total_b + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ4_0*>(w), out, N, K);
    } else if (dt == DType::Q8_0) {
        int total_b = N * (K >> 5);
        dim3 block(32, 4);
        k_dequant_q8_0_to_f16<<<(total_b + 3) / 4, block, 0, stream>>>(static_cast<const BlockQ8_0*>(w), out, N, K);
    } else if (dt == DType::Q4_1) {
        int total_b = N * (K >> 5);
        k_dequant_q4_1_to_f16<<<(total_b + 63) / 64, 256, 0, stream>>>(
            static_cast<const BlockQ4_1*>(w), out, total_b);
    } else if (dt == DType::Q2_K) {
        const int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_generic_to_f16<10><<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::Q5_K) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_generic_to_f16<7><<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, out, N, K);
    } else if (dt == DType::F32 || dt == DType::BF16) {
        int64_t total = static_cast<int64_t>(N) * K;
        k_dequant_f32_or_bf16_to_f16<<<static_cast<int>((total + 255) / 256), 256, 0, stream>>>(w, dt == DType::BF16 ? 1 : 0, out, total);
    } else if (dt == DType::F16) {
        cudaMemcpyAsync(out, w, static_cast<size_t>(N) * K * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    }
}

void rmsnorm_or_copy_to_f16(const float* x, const float* norm_w, uint16_t* out_f16,
                            int M, int K, float eps, cudaStream_t stream) {
    int threads = std::min(1024, std::max(32, ((K + 31) / 32) * 32));
    k_rmsnorm_or_copy_to_f16<<<M, threads, 0, stream>>>(x, norm_w, reinterpret_cast<half*>(out_f16), K, eps);
}

void cublas_linear_f16_f32(cublasHandle_t handle, const uint16_t* x_f16, const uint16_t* w_f16,
                           float* y, int M, int N, int K, int output_stride, float alpha) {
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w_f16, CUDA_R_16F, K,
        x_f16, CUDA_R_16F, K,
        &beta,
        y, CUDA_R_32F, output_stride ? output_stride : N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasGemmEx failed with status " + std::to_string(static_cast<int>(st)));
    }
}

}  // namespace fusionrt::cuda_ops

from typing import Optional

import torch
import triton
import triton.language as tl

from fbgemm_gpu.experimental.gemm.triton_gemm import utils

_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [128, 256]
    for block_size_k in [128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
    if not (block_size_m == 64 and num_warps == 8)
]

@triton.autotune(
    configs= _NV_CONFIGS,
    key=["G","M_BUCKET", "N", "K"],
)

@triton.jit
def _kernel_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # grid
    NUM_SMS: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
) -> None:
    block_id_x = tl.program_id(0)
    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    c_desc_ptr = workspace + block_id_x * TMA_SIZE

    M_end_offset =0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes+g)
        M_end_offset = M_start_offset + m_size

        if m_size > 0:
            N_start_offset = g *N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            tl.extra.cuda.experimental_device_tensormap_create2d(
                desc_ptr = c_desc_ptr,
                global_address = c_ptr + M_start_offset * N,
                load_size = [BLOCK_SIZE_M, BLOCK_SIZE_N],
                global_size =[m_size, n_size],
                element_ty = c_ptr.dtype.element_ty

            )
            tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while block_id_x >= iterated_tiles and block_id_x < iterated_tiles+num_tiles:
                gindex = block_id_x - iterated_tiles
                # split M first then N
                tile_m_index = gindex % num_m_tiles
                tile_n_index = gindex // num_m_tiles
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32
                # tl.static_assert(K % BLOCK_SIZE_K==0)

                m_offset = (M_start_offset + tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (N_start_offset + tile_n_index * BLOCK_SIZE_N).to(tl.int32))
                for k_offset in range(0, K, BLOCK_SIZE_K):
                    a = tl.experimental_descriptor_load(
                        a_desc_ptr,
                        [m_offset, k_offset],
                        [BLOCK_SIZE_M, BLOCK_SIZE_K],
                        dtype,
                    )
                    b = tl.experimental_descriptor_load (
                        b_desc_ptr,
                        [n_offset, k_offset],
                        [BLOCK_SIZE_N, BLOCK_SIZE_K],
                        dtype,
                    )
                    accumulator += tl.dot(a,b.T)

                m_offset = (tile_m_index * BLOCK_SIZE_M).to(tl.int32)
                n_offset = (tile_n_index * BLOCK_SIZE_N).to(tl.int32)
                tl.experimental._experimental_descriptor_store(
                    c_desc_ptr,
                    accumulator.to(c_ptr.dtype.element_ty),
                    [m_offset, n_offset],
                )
                block_id_x += NUM_SMS
            iterated_tiles += num_tiles

)

/**
 * vllm_npu_direct.h - In-tree minimal Rockchip rknpu userspace driver
 *
 * Zero third-party dependencies: this driver talks DIRECTLY to the Rockchip
 * rknpu kernel driver (a stock system component on RK3588 boards; GPL-2.0
 * sources ship in the Rockchip kernel). It implements:
 *
 *   1. Device open      - auto-detects the driver backend:
 *                          - misc node (/dev/rknpu, legacy RKNPU_* ioctls)
 *                          - DRM render node (/dev/dri/renderD*, rknpu 0.9.x
 *                            DRM GEM backend: DRM_IOCTL_RKNPU_* ioctls)
 *   2. Buffer objects   - DMA-able BOs allocated through the driver:
 *                          - misc: RKNPU_ALLOC_MEM ioctl + mmap
 *                          - DRM : RKNPU_MEM_CREATE + MEM_MAP (fake offset)
 *                            + mmap; dma_addr (IOVA) exposed for regcmd.
 *   3. regcmd lists     - the register-command sequences that configure the
 *                         NPU frontend (CBUF/CNA/DPU) for a group-wise
 *                         int8/int4 GEMM, generated in-tree (no compiler,
 *                         no runtime library).
 *   4. Task submission  - misc: RKNPU_SET_REG_CMD + RKNPU_RUN_CMD;
 *                         DRM : rknpu_task[] array inside a BO + RKNPU_SUBMIT
 *                         (PC mode), synchronous via task_counter.
 *
 * The group-wise GEMM matches the engine's Q8_0/Q4_0 block format 1:1
 * (group G = 32 == one block, per-block scales):
 *   C_f[m,n] = Σ_g a_scale[m,g] * b_scale[n,g] * (int32 partial of K-group g)
 * The engine's weights feed the hardware as-is - NO model conversion.
 *
 * CALIBRATION CONTRACT (honesty boundary): the rknpu ioctl protocol below is
 * stable and documented in the Rockchip kernel sources, but the NPU register
 * offsets/values for the matmul datapath are NOT public beyond RK3588 TRM
 * chapter 36. They live in vllm_npu_direct.c under a single table now guarded
 * by VLLM_NPU_DIRECT_READY (defined; see the VLLM_NPU_REGS_VERIFIED table in
 * the source header - calibrated 2026-08-20 on rknpu 0.9.6 DRM via
 * `--npu-calib`: int8 4x64x64 and int4 4x128x64 both PASS). Run
 * `--npu-selftest` on any new driver/hardware revision: it executes the full
 * pipeline against a CPU reference and reports PASS/FAIL, which is how the
 * table gets re-validated.
 */

#ifndef VLLM_NPU_DIRECT_H
#define VLLM_NPU_DIRECT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vllm_npu_direct_s vllm_npu_direct_t;   /* opaque */

/* ---- Buffer object ---- */
typedef struct {
    uint64_t vaddr;     /* CPU virtual address (mmap) */
    uint64_t haddr;     /* hardware/IOVA address (for regcmd) */
    uint32_t size;
    int      mapped;
    uint32_t handle;    /* DRM GEM handle (MEM_CREATE); 0 on misc backend */
    uint64_t obj_addr;  /* kernel object addr (MEM_CREATE); 0 on misc backend */
} vllm_npu_bo_t;

/* Open the rknpu device (misc or DRM backend auto-detected). Returns NULL if
 * no usable node exists. */
vllm_npu_direct_t *vllm_npu_direct_open(int verbosity, int core_mask);

/* Close the device and release all BOs. */
void vllm_npu_direct_close(vllm_npu_direct_t *d);

/* Allocate + map a DMA BO. Returns 0 on success. */
int vllm_npu_direct_bo_alloc(vllm_npu_direct_t *d, vllm_npu_bo_t *bo,
                             size_t size);

/* Release a BO. */
void vllm_npu_direct_bo_free(vllm_npu_direct_t *d, vllm_npu_bo_t *bo);

/* Driver/hardware info (for the selftest banner). Returns 0 on success. */
int vllm_npu_direct_info(vllm_npu_direct_t *d, uint32_t *hw_version,
                         uint32_t *drv_version);

/**
 * Group-wise int8/int4 GEMM on the NPU:
 *   out[m][n] = Σ_g a_scale[m][g] * b_scale[n][g] * dot(Aq[m][g-block], Wq[n][g-block])
 *   prec 0 = int8 (Q8_0: 32 int8 codes per block), G = 32
 *   prec 1 = int4 (Q4_0: 16 nibble bytes per block), G = 32
 * Inputs are engine-owned RAM buffers; the driver uploads them into BOs.
 *   Aq     [M*K]    int8/int4 codes, row-major, K % 32 == 0
 *   a_scale[M*K/G]  per-row per-K-block fp32 scales
 *   Wq     [N*K]    weight codes, row-major
 *   b_scale[N*K/G]  per-row per-K-block fp32 scales
 * K is submitted one block (G columns) per NPU pass and the dequantised
 * partials are accumulated in fp32 (block-dequant needs per-block partials,
 * a full-K int32 sum cannot be block-scaled).
 * M must be a positive multiple of 4 (M==1 is broken on the HW geometry).
 * N is padded internally to the HW alignment (int8 32 / int4 64) with zero
 * weight rows, so arbitrary output widths are accepted.
 *
 * wkey (0 = no caching) is a caller-supplied stable identity for THIS weight
 * matrix (e.g. (layer,proj)). When nonzero and the bounded persistent
 * weight-BO cache is enabled (VLLM_NPU_WCACHE slots), the full-N packed
 * weight is pre-uploaded into a persistent BO once and reused on later calls
 * (shared-input fusion: the weights are read from DRAM once instead of being
 * re-packed + re-uploaded every decode token). The key disambiguates a
 * pointer-reuse after an engine-side evict. Capacity is bounded; on overflow
 * the oldest slot is evicted (LRU ring).
 * Returns 0 on success, nonzero on failure (caller falls back to CPU).
 */
int vllm_npu_direct_gemm_gw(vllm_npu_direct_t *d, float *out,
                            const int8_t *Aq, const float *a_scale,
                            const int8_t *Wq, const float *b_scale,
                            int M, int N, int K, int G, int prec,
                            uint64_t wkey);

/**
 * Self-test: opens the device, allocates a BO, runs a small group-wise matmul
 * against a CPU reference and prints PASS/FAIL. Returns 0 when all checks
 * pass (or the device is absent), nonzero on a failed check.
 */
int vllm_npu_direct_selftest(vllm_npu_direct_t *d, const char *sdk_tag);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_NPU_DIRECT_H */

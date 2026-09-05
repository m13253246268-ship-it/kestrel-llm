/**
 * vllm_npu_direct.c - In-tree minimal Rockchip rknpu userspace driver
 *
 * Zero third-party dependencies. Implements the device-open / BO / regcmd /
 * submit machinery against the Rockchip rknpu kernel driver ioctl ABI (see
 * the header for the design and the CALIBRATION CONTRACT).
 *
 * SUBMIT MODEL (RK3588 rknpu 0.9.x, DRM backend, PC mode):
 *   - the register-command "program" is an array of 64-bit NPUOP words
 *     (op(16)<<48 | value(32)<<16 | reg(16)), stored in a BO and executed by
 *     the NPU's PC engine (task->regcmd_addr -> PC_DATA_ADDR).
 *   - the rknpu_task[] array lives in its own BO (which MUST be allocated
 *     with RKNPU_MEM_KERNEL_MAPPING - the driver reads it through its kernel
 *     mapping); submit->task_obj_addr = that gem object's kernel address.
 *   - the driver computes PC_DATA_AMOUNT = (regcfg_amount + 4 + scale - 1) /
 *     scale - 1 (RK3588 scale=2; the +4 is the mandatory 4-word PC trailer),
 *     writes INT_MASK/INT_CLEAR/PC_TASK_CONTROL/PC_DMA_BASE_ADDR and kicks
 *     PC_OP_EN. A blocking submit (no RKNPU_JOB_NONBLOCK) returns only after
 *     the completion IRQ matched task->int_mask.
 *
 * The regcmd stream (CNA/CORE/DPU/DPU_RDMA + trailer) is an independent
 * implementation: it was informed by public RK3588 hardware knowledge
 * (register-map facts documented by FOSS projects such as Mesa rkt_regcmd
 * and rocket-userspace were used as references only), and every value was
 * cross-verified on the target board against a working BSP rknpu submission
 * (CNA_CBUF_CON0=0xA2, DPU_FEATURE_MODE_CFG=0x1E4, DPU_BS_CFG/BN_CFG=0x53,
 * DPU_EW_CFG=0x383, DPU_SURFACE_ADD = surf_stride*8<<4 all match).
 * No code from any of those references is copied into this file.
 *
 * CALIBRATION CONTRACT (honesty boundary): the stream below is deliberately
 * NOT shipped as "verified". Until VLLM_NPU_DIRECT_READY is defined, the
 * gemm path returns "uncalibrated" (-2) so every offload point transparently
 * falls back to the CPU - EXCEPT when VLLM_NPU_CALIBRATE=1 or the selftest
 * itself runs the pipeline. `--npu-selftest` exercises the full path against
 * a CPU reference and prints PASS/FAIL per stage; that is how the table gets
 * calibrated for the target driver/hardware revision (env knobs below allow
 * sweeping the uncertain fields without recompiling).
 */

/* ================================================================
 * VLLM_NPU_REGS_VERIFIED - RK3588 rknpu 0.9.6 DRM calibration table
 * ================================================================
 * Target: Orange Pi 5 Plus, rknpu DRM backend (/dev/dri/renderD129,
 * drv=906 hw=1179210309), PC-mode submit. Calibrated 2026-08-20 via
 * `--npu-calib` (selftest vs CPU oracle, PASS/FAIL):
 *
 *   int8 (Q8_0): 4x32x32 PASS, 4x64x64 PASS   max_err 3.8e-06
 *   int4 (Q4_0): 4x64x32 PASS, 4x128x64 PASS  max_err 1.9-3.8e-06
 *   int4 per-k one-hot feature probe: 64/64 k-sites exact (layout proof)
 *
 * ENGINE-SIZE INTEGRATION (K-chunked + N-tiled; one NPU pass per 32-wide
 * K block x 256-wide N tile, fp32 accumulation):
 *   int8 8x4096x2048 PASS (16 tiles x 64 blocks)  max_err 3.05e-05
 *   int4 8x2048x2048 PASS ( 8 tiles x 64 blocks)  max_err 3.05e-05
 *   Nt=64 run: 3200 submits, Nt=256 run: 1536 submits - ZERO job timeouts,
 *   ZERO invalid-irq status. int_mask=0x30 is what makes this stable (below).
 *
 * Two HW/driver limits discovered and worked around:
 *   - single-task N is HW-limited (N=2048 int8 wedged the DPU writer and
 *     timed out every submit; N<=340 passed) -> N-tile at Nt=256.
 *   - int_mask=0x3ff races: a partial-completion IRQ (CNA/CORE done, DPU
 *     still writing - "invalid irq status: 0x5/0x55") makes the 0.9.x
 *     handler clear RKNPU_INT_CLEAR and drop the late DPU bit -> timeout.
 *     Masking to 0x30 (DPU group0|group1 only) leaves CNA/CORE bits masked
 *     off, so the single DPU completion is the only interrupt the job sees.
 *     (Nt<=256 keeps the DPU within one write group so it is not reported
 *     complete before the whole tile is written.)
 *   - once a job times out the 0.9.x driver is wedged (every later submit
 *     returns EINVAL, or a retry piles a second job in the queue and hangs
 *     the process in D-state) - cold boot only. Hence: no submit retries,
 *     a failed block fails the GEMM and the caller falls back to the CPU.
 *
 * Register table (register values independently derived for this chip,
 * informed by public FOSS matmul bring-up documentation such as
 * rocket-userspace gen_matmul_int8/int4 — reference only, no code reused;
 * each value cross-verified on this board):
 *
 *   CNA   CONV_CON1  (in==proc prec<<7|prec<<4|conv_mode)   int8=3 int4=6
 *   CNA   CONV_CON2  feature_grains=(M+1)<<4
 *   CNA   CONV_CON3  1x1 strides
 *   CNA   DATA_SIZE0 (1<<16)|M ; SIZE1 ((K-1)<<16)|K ; SIZE2 1 ; SIZE3 M
 *   CNA   WEIGHT_SIZE0 int8 N*K / int4 N*K/2
 *   CNA   WEIGHT_SIZE1 int8 K   / int4 K/2
 *   CNA   WEIGHT_SIZE2 (1<<24)|(1<<16)|N
 *   CNA   CBUF_CON0   data_bank = fd_banks + 1 (int8, DMA-resonance slack)
 *                     data_bank = fd_banks     (int4); weight_bank=BANKS-data
 *   CNA   CBUF_CON1   data_entries: int8 ceil(K/64), int4 ceil(K/128)
 *   CNA   CVT_CON0    0xB (sign|cvt_type|bypass); CVT_CON1-4 scale=0x10000
 *   CNA   DMA_CON0    (0xF<<16)|0xF ; DMA_CON1 line_stride=4
 *   CNA   DMA_CON2    surf_stride = 4*(M/4 - 1)
 *   CNA   FC_DATA_SIZE0 (1<<16)|M ; FC_DATA_SIZE1 K
 *   CNA   DCOMP_ADDR0 = weight DMA (decompress pass-through)
 *   CORE  MISC_CFG   prec<<8 ; DATAOUT_SIZE_0 (M-1)<<16 ; SIZE_1 N-1
 *   DPU   FEATURE_MODE_CFG 0x1E4 (burst 15, output_mode 2, direct conv)
 *   DPU   DATA_FORMAT (out<<29)|(in<<26)|proc    int8 4/3/3  int4 1/6/6
 *   DPU   DST_BASE_ADD out_dma ; DST_SURF_STRIDE M<<4
 *   DPU   DATA_CUBE_WIDTH 0 ; HEIGHT M-1 ; CHANNEL (N-1)<<16|(N-1)
 *   DPU   BS/BN_CFG 0x53 (all bypass) ; EW_CFG 0x382 (all bypass)
 *   DPU   BS_OW_CFG size_e=7, od_bypass=1 (integer-output stride quirk)
 *   DPU   SURFACE_ADD (M*8)<<4 ; OUT_CVT_SCALE 1
 *   DPU_RDMA S_POINTER 0xE + MRDMA/ERDMA disable + FMC burst 15 (MANDATORY:
 *            without the armed 0x5xxx block the DPU never completes - MRDMA
 *            trap; skipped only via VLLM_NPU_NO_RDMA=1 diagnostic)
 *   trailer: OP_ENABLE reg=0x8 val=0x1D (CNA|CORE|DPU|DPU_RDMA)
 *
 * Cube layouts (engine row-major -> NPU native, see packers below):
 *   int8 feature C2=16 [K/16][M][16] ; int8 weight (N/32,K/32,32,32)
 *   int4 feature C2=32 nibbles [K/32][M][32] (even=low, odd=high)
 *   int4 weight (N/64,K/32,64,32) nibbles (N-align 64 - N=32 hangs the HW)
 *   output int8 int32 C2=4 ; int4 int16 C2=8
 * ================================================================ */
#define VLLM_NPU_DIRECT_READY 1

#include "vllm_npu_direct.h"
#include "vllm_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>

/* ================================================================
 * rknpu kernel driver ioctl ABI (Rockchip GPL-2.0 kernel driver:
 * drivers/rknpu, shipped in the board BSP. Field order must match.)
 * Two backends are supported and auto-detected at open():
 *
 *   A) misc  (legacy): /dev/rknpu, RKNPU_IOC_MAGIC 'R', RKNPU_* ioctls
 *   B) DRM  (0.9.x+) : /dev/dri/renderD*, DRM_IOCTL_RKNPU_* ioctls
 *      (rockchip 6.1 drivers/rknpu/include/rknpu_ioctl.h: MEM_CREATE /
 *       MEM_MAP / MEM_DESTROY / MEM_SYNC / SUBMIT / ACTION)
 *
 * The structs are plain layouts and are defined on every platform so the
 * regcmd generator compiles everywhere; the ioctl macros need sys/ioctl.h
 * and are therefore Linux-only.
 * ================================================================ */
struct rknn_mem {
    uint64_t v_addr;    /* CPU virtual address (mmap) */
    uint32_t size;
    uint32_t flags;
    uint64_t h_addr;    /* hardware (IOVA) address, filled by the driver */
    uint32_t reserve;
};

struct rknn_reg_cmd {
    uint32_t reg_offset;   /* NPU core register offset */
    uint32_t reg_value;
};

struct rknn_task_cmd {
    uint64_t reg_cmd_ptr;  /* pointer to a struct rknn_reg_cmd array */
    uint32_t reg_cmd_num;
    uint32_t int_mask;
    uint64_t ud;
    uint64_t result;       /* execution result written by the driver */
    uint64_t pingpong;
    uint64_t dump_buf_ptr;
    uint64_t dump_buf_size;
    uint32_t core_mask;
    uint32_t reserve;
};

/* ---- DRM backend UAPI (rockchip 6.1 rknpu_ioctl.h) ---- */
struct rknpu_mem_create {
    uint32_t handle;         /* GEM handle, filled by driver */
    uint32_t flags;          /* e_rknpu_mem_type */
    uint64_t size;
    uint64_t obj_addr;       /* kernel object addr (submit task_obj_addr) */
    uint64_t dma_addr;       /* IOVA for regcmd addressing */
    uint64_t sram_size;
    int32_t  iommu_domain_id;
    uint32_t core_mask;
};
struct rknpu_mem_map {
    uint32_t handle;
    uint32_t reserved;
    uint64_t offset;         /* fake mmap offset, filled by driver */
};
struct rknpu_mem_destroy {
    uint32_t handle;
    uint32_t reserved;
    uint64_t obj_addr;
};
struct rknpu_mem_sync {
    uint32_t flags;          /* e_rknpu_mem_sync_mode */
    uint32_t reserved;
    uint64_t obj_addr;
    uint64_t offset;
    uint64_t size;
};
#pragma pack(push, 1)
struct rknpu_task {
    uint32_t flags;          /* e_rknpu_job_mode (PC etc.) */
    uint32_t op_idx;
    uint32_t enable_mask;
    uint32_t int_mask;
    uint32_t int_clear;
    uint32_t int_status;
    uint32_t regcfg_amount;
    uint32_t regcfg_offset;
    uint64_t regcmd_addr;    /* IOVA of the register-command data */
};
#pragma pack(pop)
struct rknpu_subcore_task {
    uint32_t task_start;
    uint32_t task_number;
};
struct rknpu_submit {
    uint32_t flags;          /* e_rknpu_job_mode */
    uint32_t timeout;
    uint32_t task_start;
    uint32_t task_number;
    uint32_t task_counter;   /* filled by driver: completed task count */
    int32_t  priority;
    uint64_t task_obj_addr;  /* kernel address of the task-array gem object */
    uint32_t iommu_domain_id;
    uint32_t reserved;
    uint64_t task_base_addr; /* -> PC_DMA_BASE_ADDR (0 for absolute-IOVA) */
    int64_t  hw_elapse_time; /* filled by driver */
    uint32_t core_mask;
    int32_t  fence_fd;
    struct rknpu_subcore_task subcore_task[5];
};
struct rknpu_action {
    uint32_t flags;          /* e_rknpu_action */
    uint32_t value;          /* GET -> value, SET -> in/out */
};

#define RKNPU_IOC_MAGIC 'R'
#define RKNPU_GET_HW_VERSION    _IO(RKNPU_IOC_MAGIC, 0)
#define RKNPU_SET_IOMMU_ENABLE  _IO(RKNPU_IOC_MAGIC, 1)
#define RKNPU_ALLOC_MEM         _IOWR(RKNPU_IOC_MAGIC, 2, struct rknn_mem)
#define RKNPU_FREE_MEM          _IOW(RKNPU_IOC_MAGIC, 3, struct rknn_mem)
#define RKNPU_SET_REG_CMD       _IOWR(RKNPU_IOC_MAGIC, 4, struct rknn_reg_cmd)
#define RKNPU_RUN_CMD           _IOWR(RKNPU_IOC_MAGIC, 5, struct rknn_task_cmd)
#define RKNPU_GET_TOTAL_MEM     _IO(RKNPU_IOC_MAGIC, 11)
#define RKNPU_GET_USED_MEM      _IO(RKNPU_IOC_MAGIC, 12)

/* DRM backend ioctls (rknpu_ioctl.h: RKNPU_ACTION=0x00 ... RKNPU_MEM_SYNC=0x05)
 * DRM_COMMAND_BASE = 0x40; the driver's rknpu_ioctl matches _IOC_NR only. */
#define DRM_COMMAND_BASE 0x40
#define DRM_RKNPU_ACTION      (DRM_COMMAND_BASE + 0x00)
#define DRM_RKNPU_SUBMIT      (DRM_COMMAND_BASE + 0x01)
#define DRM_RKNPU_MEM_CREATE  (DRM_COMMAND_BASE + 0x02)
#define DRM_RKNPU_MEM_MAP     (DRM_COMMAND_BASE + 0x03)
#define DRM_RKNPU_MEM_DESTROY (DRM_COMMAND_BASE + 0x04)
#define DRM_RKNPU_MEM_SYNC    (DRM_COMMAND_BASE + 0x05)
#define DRM_RKNPU_IOWR(nr, type) _IOWR('d', nr, type)

/* e_rknpu_action */
enum {
    NPU_ACT_GET_HW_VERSION = 0,
    NPU_ACT_GET_DRV_VERSION = 1,
    NPU_ACT_GET_FREQ = 2,
    NPU_ACT_RESET = 6,
    NPU_ACT_POWER_ON = 20,
    NPU_ACT_POWER_OFF = 21,
};
/* e_rknpu_job_mode */
#define NPU_JOB_SLAVE    0x0
#define NPU_JOB_PC       0x1
#define NPU_JOB_NONBLOCK 0x2
#define NPU_JOB_PINGPONG 0x4
/* e_rknpu_mem_type */
#define NPU_MEM_NON_CONTIGUOUS 0x1
#define NPU_MEM_CACHEABLE      0x2
#define NPU_MEM_WRITE_COMBINE  0x4
#define NPU_MEM_KERNEL_MAPPING 0x8
#define NPU_MEM_IOMMU          0x10
/* e_rknpu_mem_sync_mode (MEM_SYNC flags) */
#define RKNPU_MEM_SYNC_TO_DEVICE   (1u << 0)
#define RKNPU_MEM_SYNC_FROM_DEVICE (1u << 1)

/* Which core the task runs on. The DRM driver's rknpu_job_commit switches on
 * the exact per-core masks below - RKNN's "BIT3" convention (0x8) would hit
 * the "Unknown core mask" branch and the job would never commit. */
#define RKNPU_CORE0_MASK 0x1
#define RKNPU_CORE1_MASK 0x2
#define RKNPU_CORE2_MASK 0x4

/* Backend selector */
#define NPU_BACKEND_MISC 0
#define NPU_BACKEND_DRM  1

/* ================================================================
 * RK3588 NPU register map + PC regcmd encodings. Register addresses and
 * field encodings are public hardware facts; PC offsets were cross-checked
 * against the rockchip rknpu_ioctl.h UAPI and public FOSS hardware notes.
 * ================================================================ */
/* PC block */
#define PC_OPERATION_ENABLE 0x0008
#define PC_BASE_ADDRESS     0x0010
#define PC_REGISTER_AMOUNTS 0x0014
/* CNA 0x1xxx */
#define CNA_S_POINTER          0x1004
#define CNA_CONV_CON1          0x100C
#define CNA_CONV_CON2          0x1010
#define CNA_CONV_CON3          0x1014
#define CNA_DATA_SIZE0         0x1020
#define CNA_DATA_SIZE1         0x1024
#define CNA_DATA_SIZE2         0x1028
#define CNA_DATA_SIZE3         0x102C
#define CNA_WEIGHT_SIZE0       0x1030
#define CNA_WEIGHT_SIZE1       0x1034
#define CNA_WEIGHT_SIZE2       0x1038
#define CNA_CBUF_CON0          0x1040
#define CNA_CBUF_CON1          0x1044
#define CNA_CVT_CON0           0x104C
#define CNA_CVT_CON1           0x1050
#define CNA_CVT_CON2           0x1054
#define CNA_CVT_CON3           0x1058
#define CNA_CVT_CON4           0x105C
#define CNA_FC_CON0            0x1060
#define CNA_FC_CON1            0x1064
#define CNA_PAD_CON0           0x1068
#define CNA_FEATURE_DATA_ADDR  0x1070
#define CNA_FC_CON2            0x1074
#define CNA_DMA_CON0           0x1078
#define CNA_DMA_CON1           0x107C
#define CNA_DMA_CON2           0x1080
#define CNA_FC_DATA_SIZE0      0x1084
#define CNA_FC_DATA_SIZE1      0x1088
#define CNA_DCOMP_CTRL         0x1100
#define CNA_DCOMP_REGNUM       0x1104
#define CNA_DCOMP_ADDR0        0x1110
#define CNA_DCOMP_AMOUNT       0x1140
#define CNA_CVT_CON5           0x1180
#define CNA_PAD_CON1           0x1184
/* CORE 0x3xxx */
#define CORE_S_POINTER         0x3004
#define CORE_MISC_CFG          0x3010
#define CORE_DATAOUT_SIZE_0    0x3014
#define CORE_DATAOUT_SIZE_1    0x3018
#define CORE_CLIP_TRUNCATE     0x301C
#define CORE_3030              0x3030
/* DPU 0x4xxx */
#define DPU_S_POINTER            0x4004
#define DPU_FEATURE_MODE_CFG     0x400C
#define DPU_DATA_FORMAT          0x4010
#define DPU_OFFSET_PEND          0x4014
#define DPU_DST_BASE_ADD         0x4020
#define DPU_DST_SURF_STRIDE      0x4024
#define DPU_DATA_CUBE_WIDTH      0x4030
#define DPU_DATA_CUBE_HEIGHT     0x4034
#define DPU_DATA_CUBE_NOTCH_ADDR 0x4038
#define DPU_DATA_CUBE_CHANNEL    0x403C
#define DPU_BS_CFG               0x4040
#define DPU_BS_ALU_CFG           0x4044
#define DPU_BS_MUL_CFG           0x4048
#define DPU_BS_RELUX_CMP_VALUE   0x404C
#define DPU_BS_OW_CFG            0x4050
#define DPU_BS_OW_OP             0x4054
#define DPU_WDMA_SIZE_0          0x4058
#define DPU_WDMA_SIZE_1          0x405C
#define DPU_BN_CFG               0x4060
#define DPU_BN_ALU_CFG           0x4064
#define DPU_BN_MUL_CFG           0x4068
#define DPU_BN_RELUX_CMP_VALUE   0x406C
#define DPU_EW_CFG               0x4070
#define DPU_EW_CVT_OFFSET_VALUE  0x4074
#define DPU_EW_CVT_SCALE_VALUE   0x4078
#define DPU_EW_RELUX_CMP_VALUE   0x407C
#define DPU_OUT_CVT_OFFSET       0x4080
#define DPU_OUT_CVT_SCALE        0x4084
#define DPU_OUT_CVT_SHIFT        0x4088
#define DPU_EW_OP_VALUE_0        0x4090
#define DPU_SURFACE_ADD          0x40C0
#define DPU_40C4                 0x40C4
#define DPU_LUT_ACCESS_CFG       0x4100
#define DPU_LUT_ACCESS_DATA      0x4104
#define DPU_LUT_CFG              0x4108
#define DPU_LUT_INFO             0x410C
#define DPU_LUT_LE_START         0x4110
#define DPU_LUT_LE_END           0x4114
#define DPU_LUT_LO_START         0x4118
#define DPU_LUT_LO_END           0x411C
#define DPU_LUT_LE_SLOPE_SCALE   0x4120
#define DPU_LUT_LE_SLOPE_SHIFT   0x4124
#define DPU_LUT_LO_SLOPE_SCALE   0x4128
#define DPU_LUT_LO_SLOPE_SHIFT   0x412C
/* DPU_RDMA 0x5xxx */
#define DPU_RDMA_S_POINTER          0x5004
#define DPU_RDMA_DATA_CUBE_WIDTH    0x500C
#define DPU_RDMA_DATA_CUBE_HEIGHT   0x5010
#define DPU_RDMA_DATA_CUBE_CHANNEL  0x5014
#define DPU_RDMA_SRC_BASE_ADDR      0x5018
#define DPU_RDMA_BRDMA_CFG          0x501C
#define DPU_RDMA_BS_BASE_ADDR       0x5020
#define DPU_RDMA_NRDMA_CFG          0x5028
#define DPU_RDMA_BN_BASE_ADDR       0x502C
#define DPU_RDMA_ERDMA_CFG          0x5034
#define DPU_RDMA_EW_BASE_ADDR       0x5038
#define DPU_RDMA_EW_SURF_STRIDE     0x5040
#define DPU_RDMA_FEATURE_MODE_CFG   0x5044
#define DPU_RDMA_SRC_DMA_CFG        0x5048
#define DPU_RDMA_SURF_NOTCH         0x504C
#define DPU_RDMA_PAD_CFG            0x5064
#define DPU_RDMA_WEIGHT             0x5068
#define DPU_RDMA_EW_SURF_NOTCH      0x506C

/* Block IDs + PC op-enable encodings */
#define BLOCK_PC       0x0100
#define BLOCK_CNA      0x0200
#define BLOCK_CORE     0x0800
#define BLOCK_DPU      0x1000
#define BLOCK_DPU_RDMA 0x2000
#define PC_OP_01     0x01
#define PC_OP_40     0x40
#define PC_OP_ENABLE 0x80
#define OP_REG_PC   (BLOCK_PC | PC_OP_01)
#define OP_REG_CNA  (BLOCK_CNA | PC_OP_01)
#define OP_REG_CORE (BLOCK_CORE | PC_OP_01)
#define OP_REG_DPU  (BLOCK_DPU | PC_OP_01)
#define OP_REG_DPU_RDMA (BLOCK_DPU_RDMA | PC_OP_01)
#define OP_40     (PC_OP_40 | PC_OP_01)
#define OP_ENABLE (PC_OP_ENABLE | PC_OP_01)
#define OP_NONE   0x0

/* regcmd word: op(16)<<48 | value(32)<<16 | reg(16) */
#define NPUOP(op, value, reg) \
    ((((uint64_t)((op) & 0xffff)) << 48) | \
     (((uint64_t)((value) & 0xffffffffu)) << 16) | ((uint64_t)((reg) & 0xffff)))

/* Precision field encoding (3-bit). int4=6 was HW-established by sweep. */
#define NPU_PREC_INT8  0
#define NPU_PREC_INT16 1
#define NPU_PREC_FP16  2
#define NPU_PREC_INT32 4
#define NPU_PREC_FP32  5
#define NPU_PREC_INT4  6

/* CBUF geometry */
#define NPU_CBUF_BANK_SIZE 32768
#define NPU_CBUF_BANKS 12

/* DPU_RDMA feature-mode fields (plain-conv path: MRDMA disabled) */
#define RDMA_FMC_BURST_LEN(x)     (((x) & 0xF) << 11)
#define RDMA_FMC_MRDMA_DISABLE(x) (((x) & 0x1) << 4)
#define RDMA_FMC_MRDMA_FP16TOFP32(x) (((x) & 0x1) << 3)
#define RDMA_WEIGHT_ALL1 (1u<<24 | 1u<<16 | 1u<<8 | 1u)
#define RDMA_ERDMA_DISABLE 0x1

/* S_POINTER: single-register-group pointer pattern */
#define NPU_S_POINTER 0xE

/* CNA feature-surface stride (bytes) for datain_height rows of line_stride;
 * partial sub-block heights (<4 rows) clamp to 0. */
static inline int npu_feature_surf_stride(int line_stride, int h) {
    int s = line_stride * ((h / 4) - 1);
    return s < 0 ? 0 : s;
}

/* ================================================================
 * Driver instance
 * ================================================================ */
struct vllm_npu_direct_s {
    int  fd;            /* device fd, -1 = not open */
    int  verbosity;
    int  core_mask;
    int  backend;       /* NPU_BACKEND_MISC or NPU_BACKEND_DRM */
    int  iommu;         /* 1 when the IOMMU is enabled on the device */
    uint32_t hw_version;
    uint32_t drv_version;
    uint64_t total_mem;
    uint64_t used_mem;
    /* Cached regcmd/task BOs (fixed size; reused across gemm calls) */
    vllm_npu_bo_t rc_bo, task_bo;
    int          cache_ok;
    /* Weight Pre-packing buffer: full-N NPU-layout copy of the current
     * quantized weight matrix, REBUILT on every gemm_gw call (no identity
     * caching - the engine's 16-slot gw cache can hand the same pointer to
     * different layers after an evict+realloc, so a pointer-keyed cache
     * would serve stale weights). The build cost is one full-N pack per
     * GEMM; every submit then only memcpys its tile slice instead of
     * re-staging the strided rows and re-running the NEON interleave (the
     * old per-nibble loop measured ~17 ms per GATE/UP submit, 75% of the
     * Q4 NPU wall time). */
    uint8_t *wq_packed;         /* full-N NPU layout (N rows), NULL = none */
    size_t  wq_packed_cap;      /* allocated bytes */
    size_t  wq_packed_bytes;
    /* Persistent weight-BO cache (shared-input fusion): full-N packed weights
     * pre-uploaded into cacheable BOs and reused across gemm_gw calls (decode
     * walks the same layers every token, so the weights are a shared input
     * that was previously re-packed + re-uploaded per token). Bounded by
     * VLLM_NPU_WCACHE (slots; 0 = off); keyed by the caller's stable wkey so a
     * pointer-reuse after an engine evict cannot serve stale weights. */
    struct vllm_npu_wc_s {
        uint64_t key;
        int prec, N, K, G, Np, n_blocks, Nt;
        size_t w_seg;           /* packed bytes per (tile,block) payload */
        vllm_npu_bo_t bo;       /* persistent BO: submit-payload weight */
    } *wc;
    int wc_cap;                 /* max slots (env VLLM_NPU_WCACHE) */
    int wc_n;                   /* live slots */
    int wc_evict;               /* ring cursor: next slot to evict */

    /* Submit-failure latch (M5a): a job timeout makes the 0.9.x driver
     * soft-reset the NPU ("RKNPU: soft reset"), after which subsequent
     * submits can silently compute wrong results (observed: every garbage
     * offload run had >=1 reset; the normal gate-only/up-only runs had 0).
     * Once a submit fails, the DPU is no longer trustworthy for the rest of
     * the process - latch and let the caller fall back to the CPU kernel
     * (correctness over offload) instead of blindly retrying on a wedged
     * device. */
    int broken;
};

/* DRM backend helpers --------------------------------------------------- */
static int drm_action(vllm_npu_direct_t *d, int action, uint32_t *value) {
    struct rknpu_action a;
    memset(&a, 0, sizeof(a));
    a.flags = (uint32_t)action;
    if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_ACTION, struct rknpu_action),
              &a) != 0)
        return -1;
    if (value) *value = a.value;
    return 0;
}

/* MEM_CREATE with an explicit flags word. flags=0 = contiguous non-cacheable
 * (validated on the RK3588 0.9.6 DRM backend; IOMMU/NON_CONTIGUOUS flags
 * return EINVAL there). The task-array BO additionally needs
 * RKNPU_MEM_KERNEL_MAPPING so the driver can read it via its kv_addr. */
static int drm_mem_create_flags(vllm_npu_direct_t *d, struct rknpu_mem_create *mc,
                                size_t size, uint32_t flags) {
    memset(mc, 0, sizeof(*mc));
    mc->flags = flags;
    mc->size = (uint64_t)size;
    if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_MEM_CREATE, struct rknpu_mem_create),
              mc) != 0)
        return -1;
    return 0;
}

static int drm_mem_create(vllm_npu_direct_t *d, struct rknpu_mem_create *mc,
                          size_t size) {
    return drm_mem_create_flags(d, mc, size, 0);
}

static int drm_mem_map(vllm_npu_direct_t *d, uint32_t handle, uint64_t *offset) {
    struct rknpu_mem_map mm;
    memset(&mm, 0, sizeof(mm));
    mm.handle = handle;
    if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_MEM_MAP, struct rknpu_mem_map),
              &mm) != 0)
        return -1;
    if (offset) *offset = mm.offset;
    return 0;
}

static int drm_mem_destroy(vllm_npu_direct_t *d, uint32_t handle, uint64_t obj_addr) {
    struct rknpu_mem_destroy md;
    memset(&md, 0, sizeof(md));
    md.handle = handle;
    md.obj_addr = obj_addr;
    return ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_MEM_DESTROY, struct rknpu_mem_destroy),
                 &md);
}

/* Cacheable-BO coherence range variant: sync only [offset, offset+size) so a
 * K-block submit invalidates just its own out segment (1MB) instead of the
 * whole 3MB output BO every time - the per-submit cache-maintenance cost is
 * what makes the accumulate reads pay DRAM round trips. */
static int drm_bo_sync_range(vllm_npu_direct_t *d, vllm_npu_bo_t *bo,
                             uint32_t flags, uint64_t offset, uint64_t size) {
    struct rknpu_mem_sync ms;
    memset(&ms, 0, sizeof(ms));
    ms.flags = flags;
    ms.obj_addr = bo->obj_addr;
    ms.offset = offset;
    ms.size = size;
    if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_MEM_SYNC, struct rknpu_mem_sync),
              &ms) != 0) {
        if (d->verbosity > 1)
            fprintf(stderr, "[NPU-DIRECT] MEM_SYNC_RANGE(flags=%u off=%llu sz=%llu) "
                    "failed: %s\n", flags,
                    (unsigned long long)offset, (unsigned long long)size,
                    strerror(errno));
        return -1;
    }
    return 0;
}

/* Cacheable-BO coherence: the NPU reads/writes through the IOMMU, which is
 * NOT coherent with the CPU on RK3588. A cacheable output BO therefore needs
 * an explicit sync - TO_DEVICE after the CPU writes the sentinel (so a late
 * dirty write-back cannot clobber the NPU result) and FROM_DEVICE after the
 * submit (so the accumulate loop reads the fresh DMA result, not a stale cache
 * line from the previous K-block). The driver only honours MEM_SYNC on
 * CACHEABLE objects (non-cacheable returns EINVAL); bo_in/bo_w are uncached
 * and need no sync. */
static int drm_bo_sync(vllm_npu_direct_t *d, vllm_npu_bo_t *bo, uint32_t flags) {
    struct rknpu_mem_sync ms;
    memset(&ms, 0, sizeof(ms));
    ms.flags = flags;
    ms.obj_addr = bo->obj_addr;
    ms.offset = 0;
    ms.size = (uint64_t)bo->size;
    if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_MEM_SYNC, struct rknpu_mem_sync),
              &ms) != 0) {
        if (d->verbosity > 1)
            fprintf(stderr, "[NPU-DIRECT] MEM_SYNC(flags=%u) failed: %s\n",
                    flags, strerror(errno));
        return -1;
    }
    return 0;
}

/* #region debug-point npu-submit-sochang: forward decl (used before def) */
static void npu_dbg_mark(const char *tag);
/* #endregion */

/* Open the device with the DRM render-node backend (/dev/dri/renderD*).
 * Scans the render nodes and keeps the first one that answers the rknpu
 * ACTION probe (so a display card's render node is skipped). */
static int drm_open(vllm_npu_direct_t *d) {
    const char *cand[] = {
        "/dev/dri/renderD129", "/dev/dri/renderD128", "/dev/dri/renderD130",
        "/dev/dri/renderD131", NULL
    };
    for (int i = 0; cand[i]; i++) {
        int fd = open(cand[i], O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        int save = d->fd;
        d->fd = fd;
        uint32_t v = 0;
        if (drm_action(d, NPU_ACT_GET_DRV_VERSION, &v) == 0) {
            d->drv_version = v;
            drm_action(d, NPU_ACT_GET_HW_VERSION, &d->hw_version);
            d->backend = NPU_BACKEND_DRM;
            /* The NPU register file is NOT reset between jobs or reboots, and
             * a prior crash can leave the hardware wedged: every first PC job
             * then hangs the SoC. Apply the standard power-on + full reset
             * sequence (required by this driver) once before any task is
             * submitted. */
            npu_dbg_mark("open:before-poweron");
            drm_action(d, NPU_ACT_POWER_ON, NULL);
            npu_dbg_mark("open:after-poweron");
            drm_action(d, NPU_ACT_RESET, NULL);
            npu_dbg_mark("open:after-reset");
            if (d->verbosity > 0)
                fprintf(stderr, "[NPU-DIRECT] opened %s (DRM, drv=%u hw=%u "
                        "reset ok)\n", cand[i], d->drv_version, d->hw_version);
            return 0;
        }
        close(fd);
        d->fd = save;
    }
    return -1;
}

/* ================================================================
 * Device open / close
 * ================================================================ */
vllm_npu_direct_t *vllm_npu_direct_open(int verbosity, int core_mask) {
    static const char *const misc_nodes[] = {
        "/dev/rknpu", "/dev/accel/accel0", "/dev/rknpu0", NULL
    };
    vllm_npu_direct_t *d = (vllm_npu_direct_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->fd = -1;
    d->verbosity = verbosity;
    /* DRM backend: the driver commits on exact per-core masks (0x1/0x2/0x4);
     * default to core0. Do NOT use RKNN's 0x8 "BIT3" convention here. */
    d->core_mask = (core_mask == 0) ? RKNPU_CORE0_MASK : core_mask;

    /* Backend A: legacy misc node (/dev/rknpu, RKNPU_* ioctls) */
    for (int i = 0; misc_nodes[i]; i++) {
        int fd = open(misc_nodes[i], O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;
        d->fd = fd;
        d->backend = NPU_BACKEND_MISC;
        /* Enable the IOMMU if the driver supports it (RK3588 does). */
        if (ioctl(fd, RKNPU_SET_IOMMU_ENABLE, 1) == 0)
            d->iommu = 1;
        if (verbosity > 0)
            fprintf(stderr, "[NPU-DIRECT] opened %s (misc, iommu=%d)\n",
                    misc_nodes[i], d->iommu);
        return d;
    }

    /* Backend B: DRM render node (rockchip rknpu 0.9.x DRM GEM backend) */
    if (drm_open(d) == 0)
        return d;

    free(d);
    return NULL;
}

void vllm_npu_direct_close(vllm_npu_direct_t *d) {
    if (!d) return;
    if (d->cache_ok) {
        vllm_npu_direct_bo_free(d, &d->task_bo);
        vllm_npu_direct_bo_free(d, &d->rc_bo);
        d->cache_ok = 0;
    }
    free(d->wq_packed); d->wq_packed = NULL; d->wq_packed_cap = 0;
    if (d->wc) {
        for (int i = 0; i < d->wc_n; i++)
            vllm_npu_direct_bo_free(d, &d->wc[i].bo);
        free(d->wc); d->wc = NULL; d->wc_cap = 0; d->wc_n = 0; d->wc_evict = 0;
    }
    if (d->fd >= 0) close(d->fd);
    free(d);
}

int vllm_npu_direct_info(vllm_npu_direct_t *d, uint32_t *hw_version,
                         uint32_t *drv_version) {
    if (!d) return -1;
    if (hw_version) *hw_version = d->hw_version;
    if (drv_version) *drv_version = d->drv_version;
    return 0;
}

/* Internal alloc with explicit DRM mem-type flags (task BO needs
 * RKNPU_MEM_KERNEL_MAPPING). Public bo_alloc uses flags=0. */
static int vllm_npu_direct_bo_alloc_ex(vllm_npu_direct_t *d, vllm_npu_bo_t *bo,
                                       size_t size, uint32_t flags);

int vllm_npu_direct_bo_alloc(vllm_npu_direct_t *d, vllm_npu_bo_t *bo,
                             size_t size) {
    if (!d || !bo || d->fd < 0 || size == 0) return -1;
    return vllm_npu_direct_bo_alloc_ex(d, bo, size, 0);
}

static int vllm_npu_direct_bo_alloc_ex(vllm_npu_direct_t *d, vllm_npu_bo_t *bo,
                                       size_t size, uint32_t flags) {
    if (!d || !bo || d->fd < 0 || size == 0) return -1;
    memset(bo, 0, sizeof(*bo));

    if (d->backend == NPU_BACKEND_DRM) {
        /* DRM GEM backend: MEM_CREATE -> MEM_MAP (fake offset) -> mmap */
        struct rknpu_mem_create mc;
        if (drm_mem_create_flags(d, &mc, size, flags) != 0) {
            if (d->verbosity > 1)
                fprintf(stderr, "[NPU-DIRECT] MEM_CREATE(%zu, flags=%u) failed: %s\n",
                        size, flags, strerror(errno));
            return -1;
        }
        uint64_t offset = 0;
        if (drm_mem_map(d, mc.handle, &offset) != 0) {
            drm_mem_destroy(d, mc.handle, mc.obj_addr);
            return -1;
        }
        size_t msz = (size_t)(mc.size > 0 ? mc.size : size);
        void *p = mmap(NULL, msz, PROT_READ | PROT_WRITE, MAP_SHARED,
                       d->fd, (off_t)offset);
        if (p == MAP_FAILED) {
            drm_mem_destroy(d, mc.handle, mc.obj_addr);
            return -1;
        }
        bo->vaddr = (uint64_t)(uintptr_t)p;
        bo->haddr = mc.dma_addr;
        bo->size = (uint32_t)msz;
        bo->mapped = 1;
        bo->handle = mc.handle;
        bo->obj_addr = mc.obj_addr;
        return 0;
    }

    /* misc backend: RKNPU_ALLOC_MEM + mmap(h_addr) */
    struct rknn_mem m;
    memset(&m, 0, sizeof(m));
    m.size = (uint32_t)size;
    if (ioctl(d->fd, RKNPU_ALLOC_MEM, &m) != 0) {
        if (d->verbosity > 1)
            fprintf(stderr, "[NPU-DIRECT] ALLOC_MEM(%zu) failed: %s\n",
                    size, strerror(errno));
        return -1;
    }
    void *p = mmap(NULL, m.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                   d->fd, (off_t)m.h_addr);
    if (p == MAP_FAILED) {
        ioctl(d->fd, RKNPU_FREE_MEM, &m);
        return -1;
    }
    bo->vaddr = (uint64_t)(uintptr_t)p;
    bo->haddr = m.h_addr;
    bo->size = m.size;
    bo->mapped = 1;
    return 0;
}

void vllm_npu_direct_bo_free(vllm_npu_direct_t *d, vllm_npu_bo_t *bo) {
    if (!d || !bo || d->fd < 0) return;
    if (bo->mapped && bo->vaddr)
        munmap((void *)(uintptr_t)bo->vaddr, bo->size);
    if (d->backend == NPU_BACKEND_DRM) {
        if (bo->handle)
            drm_mem_destroy(d, bo->handle, bo->obj_addr);
    } else {
        if (bo->vaddr) {
            struct rknn_mem m;
            memset(&m, 0, sizeof(m));
            m.v_addr = bo->vaddr;
            ioctl(d->fd, RKNPU_FREE_MEM, &m);
        }
    }
    memset(bo, 0, sizeof(*bo));
}

/* ================================================================
 * PC register-command program for one int8/int4 matmul (as a 1x1 conv)
 *
 * Emits the NPUOP(op,value,reg) word stream the rknpu driver's PC engine
 * executes in PC mode (task->regcmd_addr -> PC_DATA_ADDR; regcfg_amount =
 * word count INCLUDING the 4-word trailer; the driver computes
 * PC_DATA_AMOUNT = (amount + RKNPU_PC_DATA_EXTRA_AMOUNT(4) + scale(2) - 1)
 * / 2 - 1 and the HW fetches (amount+1)*scale words = amount+4).
 *
 * Field encodings are independently derived for this chip; public FOSS
 * matmul documentation (e.g. Mesa rkt_regcmd, rocket-userspace) was used as
 * reference only, and every value was cross-verified against a working BSP
 * rknpu submission. No code is copied from those references. All addresses
 * are ABSOLUTE IOVAs in the low
 * 32 bits (task_base_addr = 0 so PC_DMA_BASE_ADDR can never add into them).
 *
 * CALIBRATION TABLE: the only fields expected to need HW confirmation are
 * the PC trailer (enable register 0xf008 = RKNPU_OFFSET_ENABLE_MASK, value
 * 0x1D = CNA|CORE|DPU|DPU_RDMA) and the int16-output geometry. Each is
 * env-overridable (VLLM_NPU_*) so `--npu-selftest` can sweep them on the
 * target without recompiling. VLLM_NPU_DIRECT_READY marks the table
 * verified for a given driver/hardware revision.
 * ================================================================ */
#define VLLM_NPU_REGS_MAX 256
/* Max K-blocks per ioctl. NOTE: multi-task submit (task_number>1) is DEAD on
 * this HW/driver - the RK3588 PC engine cannot chain tasks on one core (task
 * counter stalls at 1, no completion IRQ; verified with B=2 and B=16 on drv
 * 0.9.6, and matches the rocket project's chained-task negative result), so
 * gemm_gw submits ONE task per ioctl. The bound is kept for the rc_bo/task_bo
 * sizing and for a future multi-core (subcore_task) schedule. */
#define VLLM_NPU_BATCH_MAX 16

static int npu_direct_gen_matmul_regs(uint64_t *ops, int cap,
                                      uint32_t in_dma, uint32_t w_dma,
                                      uint32_t out_dma,
                                      int M, int N, int K, int prec) {
    int i = 0;
#define EMIT(op, val, reg) do { \
        if (i >= cap) return -1; \
        ops[i++] = NPUOP((op), (uint32_t)(val), (reg)); \
    } while (0)

    /* Calibration knobs (env, read per generation).
     * Default = the rocket-HW-proven form: DPU_RDMA block armed + enable mask
     * 0x1d (OP_EN|CNA|CORE|DPU_RDMA). The DPU_RDMA domain (0x5xxx, mask bit4)
     * was previously skipped on the BSP 0.9.x driver over a lockup concern;
     * the regcmd calibrator now treats it as REQUIRED (MRDMA trap) with
     * VLLM_NPU_NO_RDMA=1 keeping the old form for diagnostics. */
    const char *e;
    uint32_t en_reg  = 0x0008u;   /* PC_OPERATION_ENABLE (rocket-HW-proven: the
                                     regcmd enable trailer targets 0x8, not the
                                     0xf008 ENABLE_MASK absolute register)    */
    uint32_t en_val  = 0x1du;     /* CNA|CORE|DPU|DPU_RDMA (cleared below)    */
    uint32_t pc_amt  = 0u;        /* PC_REGISTER_AMOUNTS = 0 (rocket-HW-proven;
                                     the driver owns PC_DATA_AMOUNT)          */
    int      no_rdma = 0;         /* default: ALWAYS arm the DPU_RDMA block.
                                     rocket-HW-proven MRDMA trap: a 1x1-conv
                                     matmul with no eltwise/bias feed leaves the
                                     DPU read-DMA armed waiting forever unless
                                     S_POINTER + the 0x5xxx block (MRDMA/ERDMA
                                     explicitly disabled) are configured -> DPU
                                     never completes -> garbage output. */
    if ((e = getenv("VLLM_NPU_OP_EN_REG")))  en_reg = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("VLLM_NPU_OP_EN_VAL")))  en_val = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("VLLM_NPU_PC_AMT")))     pc_amt = (uint32_t)strtoul(e, NULL, 0);
    if ((e = getenv("VLLM_NPU_NO_RDMA")) && e[0] && e[0] != '0')
        no_rdma = 1;             /* explicit opt-out (diagnostic only) */
    if (no_rdma)
        en_val &= ~0x10u;        /* drop the DPU_RDMA enable bit */

    unsigned conv_mode = 0;                       /* direct_convolution */
    /* NOTE: int4 mode is int4 x int4 on this HW - the same precision bits
     * feed both the feature (bits 7-9) and the weight (bits 4-6) of
     * CNA_CONV_CON1. Splitting them (feat=int8) made the CNA state machine
     * hang (feature DMA data is packed 4-bit, so an int8-feature config
     * never completes). The ±7 int4 calibration passes; real ±127 engine
     * activations truncate to 4 bits and the Q4 offload is therefore
     * disabled in the engine (VLLM_NPU_Q4 opt-in only). */
    unsigned in_prec   = (prec == 0) ? NPU_PREC_INT8 : NPU_PREC_INT4;
    unsigned out_prec  = (prec == 0) ? NPU_PREC_INT32 : NPU_PREC_INT16;
    unsigned fd_bytes;

    EMIT(OP_REG_DPU, NPU_S_POINTER, DPU_S_POINTER);
    if (!no_rdma)
        EMIT(OP_REG_DPU_RDMA, NPU_S_POINTER, DPU_RDMA_S_POINTER);

    /* ---- CNA ---- */
    EMIT(OP_REG_CNA, ((in_prec & 7) << 7) | ((in_prec & 7) << 4) | (conv_mode & 0xF),
         CNA_CONV_CON1);
    /* feature_grains = M+1 (10-bit), kernel_groups = 0 */
    EMIT(OP_REG_CNA, (((unsigned)(M + 1) & 0x3FF) << 4), CNA_CONV_CON2);
    EMIT(OP_REG_CNA, ((1u & 7) << 3) | (1u & 7), CNA_CONV_CON3);      /* 1x1 strides */
    EMIT(OP_REG_CNA, ((1u & 0x7FF) << 16) | ((unsigned)M & 0x7FF), CNA_DATA_SIZE0);
    EMIT(OP_REG_CNA, (((unsigned)(K - 1) & 0xFFFF) << 16) | ((unsigned)K & 0xFFFF),
         CNA_DATA_SIZE1);
    EMIT(OP_REG_CNA, 1u & 0x7FF, CNA_DATA_SIZE2);                     /* dataout_width */
    EMIT(OP_REG_CNA, (1u * (unsigned)M) & 0x3FFFF, CNA_DATA_SIZE3);   /* dataout_atomics */
    if (prec == 0) {
        EMIT(OP_REG_CNA, (uint32_t)((size_t)N * (size_t)K), CNA_WEIGHT_SIZE0); /* total weight bytes */
        EMIT(OP_REG_CNA, (uint32_t)K & 0x7FFFF, CNA_WEIGHT_SIZE1);             /* per-kernel */
    } else {
        EMIT(OP_REG_CNA, (uint32_t)(((size_t)N * (size_t)K + 1) / 2), CNA_WEIGHT_SIZE0);
        EMIT(OP_REG_CNA, (uint32_t)((K + 1) / 2) & 0x7FFFF, CNA_WEIGHT_SIZE1);
    }
    EMIT(OP_REG_CNA, ((1u & 0x1F) << 24) | ((1u & 0x1F) << 16) | ((unsigned)N & 0x3FFF),
         CNA_WEIGHT_SIZE2);
    /* CBUF banks. int8: +1 slack bank for the C2=16 feature DMA resonance
     * (HW-proven; data_bank=fd_banks+1, weight_bank = BANKS-data_bank). */
    fd_bytes = (prec == 0) ? (unsigned)M * (unsigned)K
                           : ((unsigned)M * (unsigned)K + 1) / 2;
    {
        unsigned fd_banks = (fd_bytes + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE;
        /* Weight cube must also fit: w_bytes = Np_tile * K bytes (int8).
         * G=256 (g256 wmode) uses Nt=1024 x K=256 = 256 KB = 8 banks; the
         * data_bank+1 slack below leaves 10 banks - fit is checked explicitly
         * so a too-large M (feature banks) or Nt/K never silently overflows
         * into garbage weights on the silicon. */
        unsigned w_banks = (prec == 0)
            ? ((unsigned)N * (unsigned)K + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE
            : ((((unsigned)N * (unsigned)K + 1) / 2) + NPU_CBUF_BANK_SIZE - 1) / NPU_CBUF_BANK_SIZE;
        unsigned data_bank, weight_bank;
        if (fd_banks > NPU_CBUF_BANKS - 1) return -1;
        if (prec == 0) {
            if (fd_banks + 1 > NPU_CBUF_BANKS - 1) return -1;
            data_bank = fd_banks + 1;
        } else {
            data_bank = fd_banks;
        }
        weight_bank = NPU_CBUF_BANKS - data_bank;
        if (w_banks > weight_bank) return -1;      /* weights must fit CBUF */
        EMIT(OP_REG_CNA, (weight_bank << 4) | data_bank, CNA_CBUF_CON0);
    }
    {
        unsigned entries = 1u * (unsigned)K;                 /* W * C */
        unsigned div     = (prec == 0) ? 64u : 128u;         /* int8 /64, int4 /128 */
        EMIT(OP_REG_CNA, ((entries + div - 1) / div) & 0x1FFF, CNA_CBUF_CON1);
    }
    EMIT(OP_REG_CNA, (1u << 3) | (1u << 1) | 1u, CNA_CVT_CON0);  /* sign=1, cvt_type=1, bypass */
    EMIT(OP_REG_CNA, 0x00010000u, CNA_CVT_CON1);
    EMIT(OP_REG_CNA, 0x00010000u, CNA_CVT_CON2);
    EMIT(OP_REG_CNA, 0x00010000u, CNA_CVT_CON3);
    EMIT(OP_REG_CNA, 0x00010000u, CNA_CVT_CON4);
    EMIT(OP_REG_CNA, 0u, CNA_FC_CON0);
    EMIT(OP_REG_CNA, 0u, CNA_FC_CON1);
    EMIT(OP_REG_CNA, 0u, CNA_PAD_CON0);
    EMIT(OP_REG_CNA, in_dma, CNA_FEATURE_DATA_ADDR);
    EMIT(OP_REG_CNA, 0u, CNA_FC_CON2);
    EMIT(OP_REG_CNA, (0xFu << 16) | 0xFu, CNA_DMA_CON0);
    EMIT(OP_REG_CNA, (1u * 4u) & 0xFFFFFFF, CNA_DMA_CON1);      /* line_stride = W*4 */
    EMIT(OP_REG_CNA, (uint32_t)npu_feature_surf_stride(4, M) & 0xFFFFFFF, CNA_DMA_CON2);
    EMIT(OP_REG_CNA, ((1u & 0x7FF) << 16) | ((unsigned)M & 0x7FF), CNA_FC_DATA_SIZE0);
    EMIT(OP_REG_CNA, (unsigned)K & 0xFFFF, CNA_FC_DATA_SIZE1);
    EMIT(OP_REG_CNA, 0u, CNA_DCOMP_CTRL);
    EMIT(OP_REG_CNA, 0u, CNA_DCOMP_REGNUM);
    EMIT(OP_REG_CNA, w_dma, CNA_DCOMP_ADDR0);
    for (int a = 0; a < 16; a++)
        EMIT(OP_REG_CNA, 0u, CNA_DCOMP_AMOUNT + (uint32_t)a * 4);
    EMIT(OP_REG_CNA, 0u, CNA_CVT_CON5);
    EMIT(OP_REG_CNA, 0u, CNA_PAD_CON1);

    /* ---- CORE ---- */
    EMIT(OP_REG_CORE, (in_prec & 7) << 8, CORE_MISC_CFG);        /* qd_en=0 */
    EMIT(OP_REG_CORE, (((unsigned)(M - 1) & 0xFFFF) << 16) | (0u & 0xFFFF),
         CORE_DATAOUT_SIZE_0);
    EMIT(OP_REG_CORE, ((unsigned)(N - 1)) & 0xFFFF, CORE_DATAOUT_SIZE_1);
    EMIT(OP_REG_CORE, 0u, CORE_CLIP_TRUNCATE);
    EMIT(OP_REG_CORE, 0u, CORE_3030);

    /* ---- DPU ---- */
    EMIT(OP_REG_DPU, (0xFu << 5) | (0u << 3) | (2u << 1) | 0u, DPU_FEATURE_MODE_CFG);
    EMIT(OP_REG_DPU, ((out_prec & 7) << 29) | ((in_prec & 7) << 26) | (in_prec & 7),
         DPU_DATA_FORMAT);
    EMIT(OP_REG_DPU, 0u, DPU_OFFSET_PEND);
    EMIT(OP_REG_DPU, out_dma, DPU_DST_BASE_ADD);
    EMIT(OP_REG_DPU, ((uint32_t)((unsigned)M) & 0xFFFFFFF) << 4, DPU_DST_SURF_STRIDE);
    EMIT(OP_REG_DPU, 0u & 0x1FFF, DPU_DATA_CUBE_WIDTH);           /* dataout_w-1 = 0 */
    EMIT(OP_REG_DPU, ((unsigned)(M - 1)) & 0x1FFF, DPU_DATA_CUBE_HEIGHT);
    EMIT(OP_REG_DPU, 0u, DPU_DATA_CUBE_NOTCH_ADDR);
    EMIT(OP_REG_DPU, (((unsigned)(N - 1) & 0x1FFF) << 16) | ((unsigned)(N - 1) & 0x1FFF),
         DPU_DATA_CUBE_CHANNEL);
    /* BS / BN / EW all bypassed (no K-accum: integer EW add is HW-dead) */
    EMIT(OP_REG_DPU, (1u << 6) | (1u << 4) | (1u << 1) | 1u, DPU_BS_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BS_ALU_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BS_MUL_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BS_RELUX_CMP_VALUE);
    /* size_e=7 integer-output stride quirk, od_bypass=1 */
    EMIT(OP_REG_DPU, (0u << 27) | (7u << 8) | (7u << 5) | (7u << 2) | (1u << 1),
         DPU_BS_OW_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BS_OW_OP);
    /* WDMA channel = N-1 (13-bit). size_c_wdma MUST stay 0 (rocket default):
     * (N-1)&0x7FF truncates to 31 at N>2048 (2080&0x7FF=31, 2052&0x7FF=3)
     * which wedged the DPU output writer - every block of a large-N GEMM
     * timed out (job timeout) while small-N (<=2048) passed. */
    EMIT(OP_REG_DPU, (0u << 27) | (0u << 16) | ((unsigned)(N - 1) & 0x1FFF),
         DPU_WDMA_SIZE_0);
    EMIT(OP_REG_DPU, (((unsigned)(M - 1) & 0x1FFF) << 16) | (0u & 0x1FFF),
         DPU_WDMA_SIZE_1);
    EMIT(OP_REG_DPU, (1u << 6) | (1u << 4) | (1u << 1) | 1u, DPU_BN_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BN_ALU_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BN_MUL_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_BN_RELUX_CMP_VALUE);
    EMIT(OP_REG_DPU, (1u << 9) | (1u << 8) | (1u << 7) | (1u << 1) | 1u, DPU_EW_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_EW_CVT_OFFSET_VALUE);
    EMIT(OP_REG_DPU, 1u, DPU_EW_CVT_SCALE_VALUE);
    EMIT(OP_REG_DPU, 0u, DPU_EW_RELUX_CMP_VALUE);
    EMIT(OP_REG_DPU, 0u, DPU_OUT_CVT_OFFSET);
    EMIT(OP_REG_DPU, (0u << 16) | 1u, DPU_OUT_CVT_SCALE);
    EMIT(OP_REG_DPU, 0u, DPU_OUT_CVT_SHIFT);
    for (int a = 0; a < 8; a++)
        EMIT(OP_REG_DPU, 0u, DPU_EW_OP_VALUE_0 + (uint32_t)a * 4);
    EMIT(OP_REG_DPU, ((uint32_t)((unsigned)M * 8u) & 0xFFFFFFF) << 4, DPU_SURFACE_ADD);
    EMIT(OP_REG_DPU, 0u, DPU_40C4);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_ACCESS_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_ACCESS_DATA);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_CFG);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_INFO);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LE_START);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LE_END);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LO_START);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LO_END);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LE_SLOPE_SCALE);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LE_SLOPE_SHIFT);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LO_SLOPE_SCALE);
    EMIT(OP_REG_DPU, 0u, DPU_LUT_LO_SLOPE_SHIFT);

    /* ---- DPU_RDMA (MRDMA-trap avoidance: arm + disable; enabled via mask).
     * Skippable via VLLM_NPU_NO_RDMA=1 (the BSP/hello2 form omits the whole
     * 0x5xxx domain and drops the mask bit4). */
    if (!no_rdma) {
        EMIT(OP_REG_DPU_RDMA, 0u & 0x1FFF, DPU_RDMA_DATA_CUBE_WIDTH);
        EMIT(OP_REG_DPU_RDMA, ((unsigned)(M - 1)) & 0x1FFF, DPU_RDMA_DATA_CUBE_HEIGHT);
        EMIT(OP_REG_DPU_RDMA, ((unsigned)(N - 1)) & 0x1FFF, DPU_RDMA_DATA_CUBE_CHANNEL);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_SRC_BASE_ADDR);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_BRDMA_CFG);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_BS_BASE_ADDR);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_NRDMA_CFG);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_BN_BASE_ADDR);
        EMIT(OP_REG_DPU_RDMA, RDMA_ERDMA_DISABLE, DPU_RDMA_ERDMA_CFG);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_EW_BASE_ADDR);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_EW_SURF_STRIDE);
        EMIT(OP_REG_DPU_RDMA, RDMA_FMC_BURST_LEN(15) | RDMA_FMC_MRDMA_DISABLE(1),
             DPU_RDMA_FEATURE_MODE_CFG);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_SRC_DMA_CFG);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_SURF_NOTCH);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_PAD_CFG);
        EMIT(OP_REG_DPU_RDMA, RDMA_WEIGHT_ALL1, DPU_RDMA_WEIGHT);
        EMIT(OP_REG_DPU_RDMA, 0u, DPU_RDMA_EW_SURF_NOTCH);
    }

    /* ---- PC trailer (4 words; the driver's amount math expects exactly
     * these trailing 4 words) ---- */
    EMIT(OP_NONE, 0u, 0u);
    EMIT(OP_REG_PC, pc_amt, PC_REGISTER_AMOUNTS);
    EMIT(OP_40, 0u, 0u);
    EMIT(OP_ENABLE, en_val, en_reg);
#undef EMIT
    return i;
}

/* ================================================================
 * Data-cube packing (engine row-major buffers -> NPU native layouts)
 *
 * int8  feature: NC1HWC2 with C2=16  -> [K/16][M][16] (feature_data)
 * int8  weight : N-group 32, K-group 32 -> [N/32][K/32][32][32]
 *               (weight_int8; index = Ngrp*1024 + Kgrp*32*K + Nwithin + Kwithin*32)
 * int4  feature: C2=32 nibbles -> [K/32][M][32], two K per byte (even=low)
 * int4  weight : N-group 64, K-group 32 -> [N/64][K/32][64][32] nibbles
 * Output cube: int8 -> int32 C2=4; int4 -> int16 C2=8 (calibratable)
 * ================================================================ */
static void npu_pack_feature_int8(int8_t *dst, const int8_t *A, int M, int K) {
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            dst[((k - 1) / 16) * M * 16 + 16 * (m - 1) + (k - 1) % 16] =
                A[(m - 1) * K + (k - 1)];
}

static void npu_pack_weight_int8(int8_t *dst, const int8_t *B, int N, int K) {
    /* HW weight cube (rocket weight_int8): N-group 32 outer, K-group 32,
     * then N-within (stride 32) OUTER, K-within INNERMOST (contiguous).
     *   dst = Kgrp*1024 + Ngrp*32*K + Kwithin + Nwithin*32
     * (Calibrated on HW: a K-inner/N-outer pack made the HW read the weight
     * transposed -> outputs were column-sums of B instead of dot products.) */
#if ST_HAVE_NEON
    /* NEON repack (bit-exact vs scalar, vqtbl4 table lookup): 8 input rows
     * become a 64B table; dst position (Kwithin=r, Nwithin=c) holds
     * src row (bi*8+c) col (bj*8+r) - pure layout repack, NO transpose. */
    const int ngrp = N / 32, kgrp = K / 32;
    for (int ng = 0; ng < ngrp; ng++) {
        for (int kg = 0; kg < kgrp; kg++) {
            const int8_t *src = B + (size_t)ng * 32 * K + (size_t)kg * 32;
            int8_t *out = dst + (size_t)kg * 1024 + (size_t)ng * 32 * K;
            for (int bi = 0; bi < 4; bi++) {
                for (int bj = 0; bj < 4; bj++) {
                    const int8_t *s = src + (size_t)bi * 8 * K + (size_t)bj * 8;
                    int8x16x4_t tab;
                    tab.val[0] = vcombine_s8(vld1_s8(s + (size_t)0 * K),
                                             vld1_s8(s + (size_t)1 * K));
                    tab.val[1] = vcombine_s8(vld1_s8(s + (size_t)2 * K),
                                             vld1_s8(s + (size_t)3 * K));
                    tab.val[2] = vcombine_s8(vld1_s8(s + (size_t)4 * K),
                                             vld1_s8(s + (size_t)5 * K));
                    tab.val[3] = vcombine_s8(vld1_s8(s + (size_t)6 * K),
                                             vld1_s8(s + (size_t)7 * K));
                    int8_t *o = out + (size_t)bj * 8 + (size_t)bi * 8 * 32;
                    for (int c = 0; c < 8; c++) {
                        uint8_t idx[8];
                        for (int r = 0; r < 8; r++) idx[r] = (uint8_t)(c * 8 + r);
                        vst1_s8(o + (size_t)c * 32,
                                vqtbl4_s8(tab, vld1_u8(idx)));
                    }
                }
            }
        }
    }
#else
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            dst[((k - 1) / 32) * 32 * 32 + ((n - 1) / 32) * 32 * K +
                (k - 1) % 32 + ((n - 1) % 32) * 32] =
                B[(n - 1) * K + (k - 1)];
#endif
}

/* (npu_pack_weight_int8_stride was tried and removed: packing straight from
 * the G256 weight matrix rows (K=4096+ stride) thrashed L2 and measured
 * slower than the wpad staging + contiguous pack path.) */

static void npu_put_nibble(uint8_t *buf, size_t idx, int8_t v) {
    uint8_t nib = (uint8_t)(v & 0xF);
    size_t byte = idx >> 1;
    if (idx & 1)
        buf[byte] = (uint8_t)((buf[byte] & 0x0F) | (nib << 4));  /* odd -> high */
    else
        buf[byte] = (uint8_t)((buf[byte] & 0xF0) | nib);         /* even -> low */
}

/* Both int4 packers run with K=Kb=G=32 (one 32-wide group per row), so the
 * NPU cube layout degenerates to "row-major 16-byte nibble rows": row r's 32
 * int8 codes (values -8..7) -> bytes [r*16 .. r*16+16), byte j = low nibble of
 * code 2j | high nibble of code 2j+1 (see npu_put_nibble). NEON: split the
 * 32 codes into even/odd lanes (vuzp), OR the odd lanes shifted left by 4
 * into the even lanes - exactly the nibble interleave. The old per-nibble
 * scalar loop measured ~17 ms per GATE/UP submit (75% of the Q4 NPU wall
 * time); this is the hot path for int4 offload. */
static void npu_pack_feature_int4(uint8_t *dst, const int8_t *A, int M, int K) {
    memset(dst, 0, ((size_t)M * (size_t)K + 1) / 2);
#if ST_HAVE_NEON
    if ((K % 32) == 0) {
        const int K32 = K / 32;   /* groups per row (row r -> group g) */
        uint8x16_t msk = vdupq_n_u8(0x0F);
        for (int m = 0; m < M; m++) {
            const uint8_t *src = (const uint8_t *)(A + (size_t)m * K);
            uint8_t *drow = dst + (size_t)m * 16;   /* group-major: rows packed */
            for (int g = 0; g < K32; g++) {
                uint8x16_t a = vld1q_u8(src + (size_t)g * 32);
                uint8x16_t b = vld1q_u8(src + (size_t)g * 32 + 16);
                uint8x16_t even = vuzp1q_u8(a, b);          /* s[0,2,..30] */
                uint8x16_t odd  = vuzp2q_u8(a, b);          /* s[1,3,..31] */
                uint8x16_t d = vorrq_u8(vandq_u8(even, msk),
                                        vshlq_n_u8(vandq_u8(odd, msk), 4));
                vst1q_u8(drow + (size_t)g * 16, d);
            }
        }
        return;
    }
#endif
    for (int m = 1; m <= M; m++)
        for (int k = 1; k <= K; k++)
            npu_put_nibble(dst, ((k - 1) / 32) * M * 32 + 32 * (m - 1) + (k - 1) % 32,
                           A[(m - 1) * K + (k - 1)]);
}

static void npu_pack_weight_int4(uint8_t *dst, const int8_t *B, int N, int K) {
    int nKgrp = (K + 31) / 32;
    memset(dst, 0, ((size_t)N * (size_t)K + 1) / 2);
#if ST_HAVE_NEON
    if ((K % 32) == 0) {
        /* 64-row cube groups: dst offset for row n = (n/64)*nKgrp*64*16 +
         * (n%64)*16 with a single k-block; general K just adds k-block
         * columns. Rows stay contiguous, so a simple per-row zip works. */
        uint8x16_t msk = vdupq_n_u8(0x0F);
        for (int n = 0; n < N; n++) {
            const uint8_t *src = (const uint8_t *)(B + (size_t)n * K);
            const int n64 = n / 64, n16 = n % 64;
            uint8_t *drow = dst + (size_t)n64 * nKgrp * 64 * 16 + (size_t)n16 * 16;
            for (int kb = 0; kb < nKgrp; kb++) {
                uint8x16_t a = vld1q_u8(src + (size_t)kb * 32);
                uint8x16_t b = vld1q_u8(src + (size_t)kb * 32 + 16);
                uint8x16_t even = vuzp1q_u8(a, b);          /* s[0,2,..30] */
                uint8x16_t odd  = vuzp2q_u8(a, b);          /* s[1,3,..31] */
                uint8x16_t d = vorrq_u8(vandq_u8(even, msk),
                                        vshlq_n_u8(vandq_u8(odd, msk), 4));
                vst1q_u8(drow + (size_t)kb * 64 * 16, d);
            }
        }
        return;
    }
#endif
    for (int n = 1; n <= N; n++)
        for (int k = 1; k <= K; k++)
            npu_put_nibble(dst,
                           ((n - 1) / 64) * nKgrp * 64 * 32 +
                               ((k - 1) / 32) * 64 * 32 +
                               (n - 1) % 64 * 32 + (k - 1) % 32,
                           B[(n - 1) * K + (k - 1)]);
}

/* ================================================================
 * DRM PC-mode task submission (synchronous; the driver waits for the
 * completion IRQ to match task->int_mask and returns).
 * ================================================================ */

/* #region debug-point npu-submit-sochang: synced file markers so a SoC hang
 * leaves the last reached step on /root/marks.log (SD card: ext4 commit=600
 * delays writes, a power-cut loses them, hence the sync after every write). */
static void npu_dbg_mark(const char *tag) {
    if (!getenv("VLLM_NPU_DBG")) return;
    FILE *f = fopen("/root/marks.log", "a");
    if (f) {
        fprintf(f, "%s\n", tag);
        fclose(f);
        sync();
    }
}
/* #endregion */

/* Job timeout (ms) per submit. Default 1000: a healthy task runs in well
 * under 1 ms (Kb<=256, Nt<=1024, M<=44), so 1 s is a generous ceiling that
 * caps the soft-reset penalty at ~1 s instead of the old 3 s when the driver
 * loses the completion IRQ under a sustained submit storm. VLLM_NPU_TIMEOUT
 * overrides (clamped 100..5000). */
static uint32_t npu_submit_timeout_ms(void) {
    const char *e = getenv("VLLM_NPU_TIMEOUT");
    if (e && e[0]) {
        int v = atoi(e);
        if (v >= 100 && v <= 5000) return (uint32_t)v;
    }
    return 1000u;
}

/* Recovery settle (ms) before retrying a timed-out submit: the 0.9.x driver
 * soft-resets on timeout and needs a beat before the next job can queue
 * safely (too-short settle observed to re-timeout). Default 150. */
static long npu_submit_settle_ns(void) {
    const char *e = getenv("VLLM_NPU_SETTLE_MS");
    if (e && e[0]) {
        int v = atoi(e);
        if (v >= 10 && v <= 2000) return (long)v * 1000000L;
    }
    return 150L * 1000000L;
}

static int drm_submit_pc(vllm_npu_direct_t *d, vllm_npu_bo_t *task_bo,
                         const uint64_t *regcmd_haddr, uint32_t n_words,
                         int n_task, int core_mask,
                         const struct rknpu_subcore_task *sct) {
    /* Batch submit: n_task tasks (one per K-block) in one ioctl. Every task
     * has its own regcmd IOVA (geometry identical, data addresses differ).
     * Axiom (MI_fusion_rule): shared-input fusion - collapsing K-block
     * submits cuts the per-ioctl fixed cost (syscall + driver job dispatch
     * + fence) from the hot loop; the NPU still executes each task in turn. */
    struct rknpu_task *tasks = (struct rknpu_task *)(uintptr_t)task_bo->vaddr;
    if (n_task < 1) return -1;
    for (int i = 0; i < n_task; i++) {
        memset(&tasks[i], 0, sizeof(struct rknpu_task));
        tasks[i].flags = 0;
        tasks[i].op_idx = 1;
        tasks[i].enable_mask = 0x7f;   /* not consumed by the 0.9.x driver */
        /* Completion criterion for the 0.9.x IRQ handler: it compares
         * fuzz_status(INT_STATUS) == int_mask. The block-group IRQ bits are
         * two per block (group0|group1); the fuzz() collapses each pair.
         *
         * int_mask = 0x30 (DPU group0|group1) ONLY, NOT 0x3ff (CNA|CORE|DPU).
         * With 0x3ff a partial-completion IRQ (CNA/CORE done while DPU is
         * still writing - observed "invalid irq status: 0x55/0x5" on large-N
         * / many submits) makes the handler clear RKNPU_INT_CLEAR, dropping
         * the late DPU completion bit -> the job waits forever -> 3s timeout.
         * Masking the CNA/CORE bits (0x30 leaves only DPU IRQs live) means
         * the single DPU completion is the ONLY interrupt the job can see -
         * no partial status, no race. N must stay within one DPU write group
         * (Nt<=64) so the DPU completes in a single group; wider N would fire
         * group0 early and be reported complete before the output is fully
         * written. (A too-small mask that never matches just times out -
         * bypass_soft_reset keeps it from hanging the SoC.) */
        tasks[i].int_mask = 0x30;          /* DPU group0|group1 completion */
        tasks[i].int_clear = 0x1ffff;      /* RKNPU_INT_CLEAR */
        /* regcfg_amount = ONE task's program length (words). The HW reads
         * (amount+1)*scale = amount+4 words per task from PC_DATA_ADDR and,
         * when task_number>1, advances by that stride for the next task -
         * the programs must therefore be packed at (n+4)-word stride. A
         * whole-stream amount (n*n_task) made the HW fetch a too-long first
         * task and stall with the task counter stuck at 1 (observed: every
         * batch submit timed out, irq status 0x0). */
        tasks[i].regcfg_amount = n_words;  /* 64-bit NPUOP words, one task */
        tasks[i].regcmd_addr = regcmd_haddr[i]; /* absolute IOVA per task */
    }

    struct rknpu_submit sub;
    memset(&sub, 0, sizeof(sub));
    sub.flags = NPU_JOB_PC;        /* blocking (no RKNPU_JOB_NONBLOCK) */
    sub.timeout = npu_submit_timeout_ms();  /* ms; driver soft-resets on timeout */
    sub.task_start = 0;
    sub.task_number = (uint32_t)n_task;
    sub.task_obj_addr = task_bo->obj_addr; /* kernel gem object addr */
    sub.task_base_addr = 0;        /* PC_DMA_BASE_ADDR=0: absolute-IOVA scheme */
    sub.core_mask = (uint32_t)core_mask;
    sub.fence_fd = -1;
    /* RK3588 driver (num_irqs>1): rknpu_job_subcore_commit_pc takes
     * task_start/task_number from subcore_task[...] instead of the top-level
     * fields. Leaving it zeroed made task_number=0 -> task_end=-1 ->
     * last_task=&task_base[-1] -> kernel OOPS in rknpu_job_subcore_commit
     * (level 3 translation fault, observed: board froze every submit).
     *
     * The index the driver reads depends on use_core_num (popcount of
     * core_mask): 1-2 cores -> subcore_task[core_index]; 3 cores ->
     * subcore_task[core_index+2]. The caller fills the exact entries; a
     * NONZERO core_mask with entries left zeroed reproduces the OOPS
     * (2026-08-20: npu_sim with core_mask=3 froze the SoC at the very first
     * submit, crash point rknpu_job_subcore_commit+0x170). */
    for (int si = 0; si < 5; si++)
        sub.subcore_task[si] = sct[si];

    npu_dbg_mark("submit:ioctl-enter");
    /* Transient IRQ loss under sustained multi-thousand submits: the job
     * times out, the 0.9.x driver soft-resets (dmesg "soft reset, num: N"),
     * and the ioctl returns error. The K-block was never executed, so
     * retrying the SAME submit after the reset settles is safe and keeps the
     * projection on the NPU instead of falling back to the whole-GEMM CPU
     * recompute. Observed: single isolated submits never time out; only the
     * back-to-back storm (G=32 int4: thousands per GEMM) does, 1-2 per
     * request. Each timeout costs ~3s ioctl + ~50ms settle. */
    int last_rc = 0;
    for (int attempt = 0; attempt < 3; attempt++) {
        if (attempt > 0) {
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = npu_submit_settle_ns();  /* let the soft-reset settle */
            nanosleep(&ts, NULL);
        }
        if (ioctl(d->fd, DRM_RKNPU_IOWR(DRM_RKNPU_SUBMIT, struct rknpu_submit),
                  &sub) == 0) {
            npu_dbg_mark("submit:ioctl-rc0");
            if (sub.task_counter == sub.task_number) return 0;
            last_rc = 2;              /* counter mismatch */
        } else {
            npu_dbg_mark("submit:ioctl-err");
            last_rc = 1;              /* ioctl error (timeout / EINTR / ...) */
        }
    }
    if (d->verbosity > 0)
        fprintf(stderr, "[NPU-DIRECT] SUBMIT failed after 3 attempts (rc=%d): %s "
                "(CPU fallback; NPU latched broken)\n",
                last_rc, strerror(errno));
    d->broken = 1;   /* 0.9.x soft-reset leaves the DPU untrustworthy */
    return -1;
}

/* Runtime calibration override: lets `--npu-selftest` (and VLLM_NPU_CALIBRATE)
 * exercise the unverified table before VLLM_NPU_DIRECT_READY is defined.
 * The selftest forces it so the calibration flow needs no extra env; normal
 * engine runs stay gated (CPU fallback) until the table is marked verified. */
static int g_npu_calib_force;

static int npu_calib_ok(void) {
    if (g_npu_calib_force) return 1;
    return getenv("VLLM_NPU_CALIBRATE") != NULL;
}

/* ================================================================
 * Group-wise int8/int4 GEMM (the engine's Q8_0/Q4_0 block weights)
 * ================================================================ */
static int vllm_npu_direct_ensure_cache(vllm_npu_direct_t *d);   /* fwd */

/* Assemble one (tile, block) weight payload of w_seg bytes from the full-N
 * packed buffer wq_src into dst. int4 (Kb=G=32) tile slices are contiguous;
 * int8 (Kb=G=256) tile slices are scattered (kg x ng 1024-byte blocks) and
 * are re-assembled here. Identical to the legacy per-submit copy, so it is
 * shared by both the persistent-BO fill and the legacy bo_w window path. */
static void npu_wtile_copy(uint8_t *dst, const uint8_t *wq_src, int prec,
                           int g, int n0, int ncur, int Np, int Np_tile,
                           int Kb, size_t wblk, size_t wrow) {
    if (prec == 1) {
        if (ncur != Np_tile)
            memset(dst, 0, (size_t)Np_tile * wrow);
        memcpy(dst, wq_src + (size_t)g * wblk + (size_t)n0 * wrow,
               (size_t)ncur * wrow);
    } else {
        const int kgrps = Kb / 32;
        const int ngroups = Np_tile / 32;
        const int ngrp_cache = Np / 32;
        const int ng_off = n0 / 32;
        memset(dst, 0, (size_t)Np_tile * (size_t)Kb);
        for (int kg = 0; kg < kgrps; kg++) {
            const uint8_t *srcg = wq_src + (size_t)g * wblk + (size_t)kg * 1024;
            uint8_t *dstg = dst + (size_t)kg * 1024;
            for (int ngt = 0; ngt < ngroups; ngt++) {
                int cg = ng_off + ngt;
                if (cg >= ngrp_cache) break;
                memcpy(dstg + (size_t)ngt * 32 * Kb,
                       srcg + (size_t)cg * 32 * Kb, 1024);
            }
        }
    }
}

int vllm_npu_direct_gemm_gw(vllm_npu_direct_t *d, float *out,
                            const int8_t *Aq, const float *a_scale,
                            const int8_t *Wq, const float *b_scale,
                            int M, int N, int K, int G, int prec,
                            uint64_t wkey) {
#ifndef VLLM_NPU_DIRECT_READY
    /* Calibration contract: the regcmd table is not yet validated on the
     * target. Returning nonzero makes every offload point fall back to the
     * CPU path - nothing dangerous is ever submitted, except during the
     * explicit calibration flow (selftest / VLLM_NPU_CALIBRATE=1). */
    if (!npu_calib_ok()) {
        (void)d; (void)out; (void)Aq; (void)a_scale; (void)Wq; (void)b_scale;
        (void)M; (void)N; (void)K; (void)G; (void)prec; (void)wkey;
        return -2;
    }
#endif
    if (!d || d->fd < 0 || !out || !Aq || !a_scale || !Wq || !b_scale)
        return -1;
    if (M <= 0 || N <= 0 || K <= 0 || G <= 0) return -1;
    if (prec != 0 && prec != 1) return -1;                     /* 0=int8, 1=int4 */
    if (d->backend != NPU_BACKEND_DRM) return -2;              /* misc not wired */
    if (M % 4 != 0 || K % 32 != 0 || G % 32 != 0 || K % G != 0)
        return -1;
    if (d->broken) return -1;   /* a prior submit failed + soft-reset: CPU only */

#ifdef __linux__
    struct timespec tw0, ts0;
    double t_setup = 0.0;      /* BO alloc + sentinel memset + scratch malloc */
    double t_gen = 0.0, t_sb = 0.0, t_aff = 0.0;  /* hide breakdown */
    double t_p1m = 0.0, t_p1t = 0.0;              /* p1 memcpy vs transpose */
    clock_gettime(CLOCK_MONOTONIC, &tw0);
#endif

    /* K-chunked + N-tiled submission: one NPU pass per (32-wide K block,
     * Nt-wide N tile). Each pass computes a raw int32/int16 partial that is
     * dequantised with the per-block scales and summed in fp32.
     *   - K chunks (Kb = G columns each) because block-dequant needs per-block
     *     partials - a full-K int32 sum cannot be block-scaled.
     *   - N tiles because the single-task N is HW-limited: N=2048 (int8)
     *     wedged the DPU output writer and timed out every submit, while
     *     N<=340 passed. Nt=256 is safely inside the proven range.
     * The regcmd geometry is fixed (K=Kb, N=Np_tile) and every pass reuses
     * the same in/w/out IOVAs - only the cube contents change. N is padded
     * per tile to the HW alignment (int8 32 / int4 64). */
    int n_align = (prec == 1) ? 64 : 32;
    /* N per NPU pass. Nt=1024 is the documented-safe ceiling (the DPU output
     * writer wedges past N=2048 via size_c_wdma truncation; Nt=256 was the
     * earlier conservative default). Larger Nt => fewer N-tiles => fewer
     * ioctls per GEMM (each submit carries one K-block task per core), i.e.
     * Command-Stream batching in the N dimension: N=4096 drops from 16 to 4
     * tiles, and gate/up (N=12288) from 48 to 12. VLLM_NPU_NT overrides
     * (must be 64..1024, aligned). */
    int Nt = 1024;
    {
        const char *e = getenv("VLLM_NPU_NT");
        if (e && e[0]) {
            int v = atoi(e);
            if (v >= 64 && v <= 1024 && (v % n_align) == 0) Nt = v;
        }
    }
    int n_tiles = (N + Nt - 1) / Nt;
    int Np_tile = ((Nt + n_align - 1) / n_align) * n_align;
    int Kb = G;
    int n_blocks = K / G;

    /* Multi-core parallel submit (VLLM_NPU_MC, default ON): NB K-blocks per
     * ioctl, one per NPU core (max 3). Each core executes its OWN single task
     * - no chaining - so the single-core task_number>1 stall does not apply
     * and the otherwise idle cores are used (~3x submit throughput, measured
     * t_sub /3.5 on the G256 prefill path). Set VLLM_NPU_MC=0 to force the
     * single-core serial path. */
    int NB = 3;
    {
        const char *e = getenv("VLLM_NPU_MC");
        if (e && e[0] && e[0] == '0') NB = 1;
    }

    /* Cube / BO sizes (per tile / per block) */
    size_t in_bytes, w_bytes, out_bytes;
    if (prec == 0) {
        in_bytes  = (size_t)M * (size_t)Kb;
        w_bytes   = (size_t)Np_tile * (size_t)Kb;
        out_bytes = (size_t)M * (size_t)Np_tile * 4;           /* int32 */
    } else {
        in_bytes  = ((size_t)M * (size_t)Kb + 1) / 2;
        w_bytes   = ((size_t)Np_tile * (size_t)Kb + 1) / 2;
        out_bytes = (size_t)M * (size_t)Np_tile * 2;           /* int16 */
    }

    /* Per-block cube sizes (one submit per K-block, single task; NB segments
     * when parallel multi-core). */
    size_t in_seg = in_bytes, w_seg = w_bytes, out_seg = out_bytes;

    /* Full-N packed weight size (also the persistent-BO payload). */
    const int Np = (N + 31) & ~31;   /* int8 packer needs N % 32 == 0 */
    const size_t wblk = (prec == 1) ? (size_t)N * (size_t)Kb / 2
                                    : (size_t)Np * (size_t)Kb;
    const size_t wneed = wblk * (size_t)n_blocks;

    /* Persistent weight-BO cache (shared-input fusion): pre-upload the full-N
     * packed weight into a persistent BO keyed by wkey, and reuse it across
     * gemm_gw calls instead of re-packing + re-uploading per token. */
    vllm_npu_bo_t *wbo = NULL;       /* persistent weight BO, NULL = legacy */
    {
        const char *e = getenv("VLLM_NPU_WCACHE");
        int cap = 0;
        if (e && e[0]) { int v = atoi(e); if (v > 0) cap = v; }
        if (cap > 0 && d->wc_cap != cap) {
            for (int i = 0; i < d->wc_n; i++)
                vllm_npu_direct_bo_free(d, &d->wc[i].bo);
            free(d->wc); d->wc = NULL; d->wc_cap = 0; d->wc_n = 0; d->wc_evict = 0;
            d->wc = (struct vllm_npu_wc_s *)calloc((size_t)cap, sizeof(*d->wc));
            if (d->wc) d->wc_cap = cap;
        }
        if (wkey != 0 && d->wc && d->wc_cap > 0) {
            for (int i = 0; i < d->wc_n; i++) {
                struct vllm_npu_wc_s *c = &d->wc[i];
                if (c->key == wkey && c->prec == prec && c->N == N &&
                    c->K == K && c->G == G && c->Np == Np &&
                    c->n_blocks == n_blocks && c->Nt == Nt) {
                    wbo = &c->bo;
                    break;
                }
            }
        }
    }

    vllm_npu_bo_t bo_in, bo_w, bo_out;
    memset(&bo_in, 0, sizeof(bo_in));
    memset(&bo_w, 0, sizeof(bo_w));
    memset(&bo_out, 0, sizeof(bo_out));
#ifdef __linux__
    clock_gettime(CLOCK_MONOTONIC, &ts0);   /* setup window: alloc+memset+scratch */
#endif
    /* VLLM_IN_UC=0 (default 1): allocate bo_in CACHEABLE so t_pack's feature
     * writes hit the L2 instead of paying the uncached 124MB/s write floor;
     * the CPU side then flushes the freshly-packed segment TO_DEVICE before
     * each submit. Experimental switch (build23) - A/B vs the uncached
     * default. */
    {
        static int g_in_uc = -1;
        if (g_in_uc < 0) {
            const char *eu = getenv("VLLM_IN_UC");
            g_in_uc = (!eu || eu[0] == '\0' || eu[0] == '1');
        }
        if (g_in_uc) {
            if (vllm_npu_direct_bo_alloc(d, &bo_in, in_seg * NB) != 0) return -1;
        } else if (vllm_npu_direct_bo_alloc_ex(d, &bo_in, in_seg * NB,
                                               NPU_MEM_CACHEABLE) != 0) {
            return -1;
        }
    }
    if (!wbo && vllm_npu_direct_bo_alloc(d, &bo_w, w_seg * NB) != 0) {
        vllm_npu_direct_bo_free(d, &bo_in); return -1;
    }
    /* out BO: cacheable lets the accumulate read from cache (a DMA-written
     * cacheable BO syncs + invalidates per submit, which measured SLOWER than
     * an uncached streamed read for the transposed accumulate - every line is
     * a fresh DMA result either way). VLLM_OUT_UC=0 forces the old cacheable
     * path; default 1 = uncached, no MEM_SYNC needed (IOMMU DMA is not CPU-
     * coherent, but with no CPU cache there is nothing to keep coherent). */
    {
        static int g_out_uc = -1;
        if (g_out_uc < 0) {
            const char *eu = getenv("VLLM_OUT_UC");
            g_out_uc = (!eu || eu[0] == '\0' || eu[0] == '1');
        }
        if (g_out_uc) {
            if (vllm_npu_direct_bo_alloc_ex(d, &bo_out, (out_seg + 64) * NB,
                                            0) != 0) {
                vllm_npu_direct_bo_free(d, &bo_w);
                vllm_npu_direct_bo_free(d, &bo_in); return -1;
            }
        } else if (vllm_npu_direct_bo_alloc_ex(d, &bo_out, (out_seg + 64) * NB,
                                               NPU_MEM_CACHEABLE) != 0) {
            vllm_npu_direct_bo_free(d, &bo_w);
            vllm_npu_direct_bo_free(d, &bo_in); return -1;
        }
    }
    /* CALIB-DIAG sentinel: pre-fill the output BO so a submit that never
     * reaches the DPU write stage leaves a recognizable pattern. */
    memset((void *)(uintptr_t)bo_out.vaddr, 0x7F, (out_seg + 64) * NB);
    /* Flush the sentinel to DRAM before the NPU overwrites it (cacheable BO) -
     * a dirty line written back later would clobber the DPU result. Uncached
     * BOs are not coherent with the CPU cache and need no flush (the driver
     * rejects MEM_SYNC on non-cacheable objects anyway). */
    {
        const char *eu2 = getenv("VLLM_OUT_UC");
        int out_uc = (!eu2 || eu2[0] == '\0' || eu2[0] == '1');
        if (!out_uc &&
            drm_bo_sync(d, &bo_out, RKNPU_MEM_SYNC_TO_DEVICE) != 0) {
            vllm_npu_direct_bo_free(d, &bo_out);
            vllm_npu_direct_bo_free(d, &bo_w);
            vllm_npu_direct_bo_free(d, &bo_in);
            return -1;
        }
    }

    /* regcmd + task BOs: cached per device (fixed 2KB + 4KB; regcmd BO is
     * zero-filled so any word past the program is a no-op). */
    if (vllm_npu_direct_ensure_cache(d) != 0) {
        vllm_npu_direct_bo_free(d, &bo_out);
        vllm_npu_direct_bo_free(d, &bo_w);
        vllm_npu_direct_bo_free(d, &bo_in);
        return -1;
    }

    /* All addresses must fit the 32-bit regcmd value fields. */
    uint64_t wh = wbo ? wbo->haddr : bo_w.haddr;
    uint64_t addrs[] = { bo_in.haddr, wh, bo_out.haddr, d->rc_bo.haddr };
    for (int i = 0; i < 4; i++) {
        if (addrs[i] >> 32) {
            if (d->verbosity > 0)
                fprintf(stderr, "[NPU-DIRECT] IOVA 0x%llx exceeds 32 bits "
                        "(IOMMU range above 4GB) - CPU fallback\n",
                        (unsigned long long)addrs[i]);
            vllm_npu_direct_bo_free(d, &bo_out);
            vllm_npu_direct_bo_free(d, &bo_w);
            vllm_npu_direct_bo_free(d, &bo_in);
            return -3;
        }
    }

    /* The register-command programs are now generated PER SUBMIT inside the
     * tile/block loop: the weight IOVA differs per (tile, block) when it lives
     * in a persistent BO (shared-input fusion), while the legacy path reuses
     * the fixed bo_w window. See the submit loop below. */

    /* Scratches: block-g activation slice [M][Kb] (rows past the tile's
     * valid width are zero so the HW pads contribute nothing). tbuf8/tbuf4
     * gather the NPU output cube into row-major [m][n] so accumulate reads
     * it contiguously - the raw cube's (nn>>2)*M*4 stride is an L2 miss per
     * group and dominates the wall time. */
    const int ncur_max = (N < Nt) ? N : Nt;
    /* Per-thread staging buffers (bcur <= NB = 3): the block staging/pack
     * loop is OpenMP-parallelized, so each worker gets its own slice.
     * (Weights no longer need a staging slice: they are pre-packed into
     * d->wq_packed and only memcpy'd per tile.) */
    enum { NPU_WPAD_MAX = 3 };
    int8_t *apad = (int8_t *)malloc((size_t)M * (size_t)Kb * NPU_WPAD_MAX);
    void *tbuf = NULL;                      /* int32 (prec=0) / int16 (prec=1) */
    if (prec == 0)
        tbuf = malloc((size_t)M * (size_t)ncur_max * sizeof(int32_t));
    else
        tbuf = malloc((size_t)M * (size_t)ncur_max * sizeof(int16_t));
    /* p1 staging: one sub-block of the cube in a plain cached buffer so the
     * transpose reads L2 instead of paying a DRAM round trip per 64B line of
     * the DMA-written output BO (VLLM_P1=3 memcpy path; see below). */
    int32_t *tmps = (int32_t *)malloc((size_t)M * 128 * sizeof(int32_t));
    if (!apad || !tbuf || !tmps) {
        free(apad); free(tbuf); free(tmps);
        vllm_npu_direct_bo_free(d, &bo_out);
        vllm_npu_direct_bo_free(d, &bo_w);
        vllm_npu_direct_bo_free(d, &bo_in);
        return -1;
    }
#ifdef __linux__
    {
        struct timespec tst;
        clock_gettime(CLOCK_MONOTONIC, &tst);
        t_setup += (tst.tv_sec - ts0.tv_sec) + 1e-9 * (tst.tv_nsec - ts0.tv_nsec);
    }
#endif

#ifdef __linux__
    struct timespec t0, t1;
    double t_pack = 0.0, t_sub = 0.0, t_acc = 0.0;
    double t_bc = 0.0, t_p1 = 0.0, t_p2 = 0.0;   /* t_acc breakdown (dbg) */
    int tacc_cpu_dbg = -1;                        /* A76 pin diagnostic */
#endif
    memset(out, 0, (size_t)M * (size_t)N * sizeof(float));
    int rc = 0;

    /* Build the full-N packed weight into d->wq_packed (shared staging). On a
     * persistent-cache miss it is assembled into a persistent BO and cached;
     * the legacy path memcpys each tile slice from it. wq_src stays NULL only
     * on a cache hit (the weight is already resident in wbo). */
    const size_t wrow = (size_t)Kb / 2;   /* packed bytes per N row (int4) */
    const uint8_t *wq_src = NULL;

    if (wbo == NULL) {
        if (wneed > d->wq_packed_cap) {
            free(d->wq_packed); d->wq_packed = NULL; d->wq_packed_cap = 0;
            d->wq_packed = (uint8_t *)malloc(wneed);
            if (d->wq_packed) d->wq_packed_cap = wneed;
        }
        int8_t *stg = (d->wq_packed)
            ? (int8_t *)malloc((size_t)Np * (size_t)Kb) : NULL;
        if (!d->wq_packed || !stg) {
            free(stg);
            vllm_npu_direct_bo_free(d, &bo_out);
            vllm_npu_direct_bo_free(d, &bo_w);
            vllm_npu_direct_bo_free(d, &bo_in);
            return -1;   /* OOM -> CPU */
        }
        for (int g = 0; g < n_blocks; g++) {
            /* Stage the block's [Np][Kb] slice (row-major contiguous,
             * rows N..Np-1 zero so HW N-pad contributes nothing). */
            for (int n = 0; n < N; n++)
                memcpy(stg + (size_t)n * Kb,
                       Wq + (size_t)n * K + (size_t)g * Kb, (size_t)Kb);
            if (Np != N)
                memset(stg + (size_t)N * Kb, 0, (size_t)(Np - N) * (size_t)Kb);
            if (prec == 1)
                npu_pack_weight_int4(d->wq_packed + (size_t)g * wblk, stg, N, Kb);
            else
                npu_pack_weight_int8((int8_t *)d->wq_packed + (size_t)g * wblk,
                                     stg, Np, Kb);
        }
        d->wq_packed_bytes = wneed;
        free(stg);
        wq_src = d->wq_packed;

        /* Persistent-cache fill on miss (shared-input fusion): pre-assemble
         * every (tile, block) submit payload into a persistent BO so later
         * calls (decode reuses the same weights every token) skip the pack +
         * per-tile memcpy + BO alloc. Non-cacheable (flags=0): written once by
         * the CPU, read many times by the NPU via DMA - no sync needed. */
        if (wkey != 0 && d->wc && d->wc_cap > 0) {
            struct vllm_npu_wc_s tmp;
            memset(&tmp, 0, sizeof(tmp));
            size_t total = (size_t)n_tiles * (size_t)n_blocks * w_seg;
            if (vllm_npu_direct_bo_alloc_ex(d, &tmp.bo, total, 0) == 0 &&
                ((tmp.bo.haddr + total) >> 32) == 0) {
                for (int t = 0; t < n_tiles; t++) {
                    const int tn0 = t * Nt;
                    const int tncur = (N - tn0 < Nt) ? (N - tn0) : Nt;
                    for (int g = 0; g < n_blocks; g++) {
                        uint8_t *dst = (uint8_t *)(uintptr_t)tmp.bo.vaddr
                                       + ((size_t)t * n_blocks + (size_t)g) * w_seg;
                        npu_wtile_copy(dst, wq_src, prec, g, tn0, tncur,
                                       Np, Np_tile, Kb, wblk, wrow);
                    }
                }
                tmp.key = wkey;
                tmp.prec = prec; tmp.N = N; tmp.K = K; tmp.G = G;
                tmp.Np = Np; tmp.n_blocks = n_blocks; tmp.Nt = Nt;
                tmp.w_seg = w_seg;
                struct vllm_npu_wc_s *slot;
                if (d->wc_n < d->wc_cap) {
                    slot = &d->wc[d->wc_n++];
                } else {
                    slot = &d->wc[d->wc_evict];
                    vllm_npu_direct_bo_free(d, &slot->bo);
                    d->wc_evict = (d->wc_evict + 1) % d->wc_cap;
                }
                *slot = tmp;
                wbo = &slot->bo;
            } else {
                vllm_npu_direct_bo_free(d, &tmp.bo);
            }
        }
    }
    for (int t = 0; t < n_tiles && rc == 0; t++) {
        const int n0 = t * Nt;
        const int ncur = (N - n0 < Nt) ? (N - n0) : Nt;   /* valid cols */
        /* Block loop INSIDE the tile loop: every K-block revisits the same
         * out segment [M][n0..n0+Nt), keeping it hot in L2. The outer-block
         * order scattered writes over out[M][N] (4.3MB > L3) and made
         * accumulate ~62% of the wall time; this order is the measured-faster
         * one. */
        for (int g0 = 0; g0 < n_blocks && rc == 0; g0 += NB) {
            const int bcur = (n_blocks - g0 < NB) ? (n_blocks - g0) : NB;
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
            /* Stage + pack the bcur feature slices (repeated per tile; the
             * pack is only ~4% of the wall time, cache locality wins). */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(bcur > 1 && NB > 1)
#endif
            for (int bi = 0; bi < bcur; bi++) {
                const int g = g0 + bi;
                int8_t *ap = apad + (size_t)omp_get_thread_num() * (size_t)M * (size_t)Kb;
                for (int m = 0; m < M; m++)
                    memcpy(ap + (size_t)m * Kb, Aq + (size_t)m * K + (size_t)g * Kb,
                           (size_t)Kb);
                void *in_v = (void *)((uintptr_t)bo_in.vaddr + (size_t)bi * in_seg);
                if (prec == 0)
                    npu_pack_feature_int8((int8_t *)in_v, ap, M, Kb);
                else
                    npu_pack_feature_int4((uint8_t *)in_v, ap, M, Kb);
            }
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_pack += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
#endif
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(bcur > 1 && NB > 1)
#endif
            for (int bi = 0; bi < bcur; bi++) {
                if (wbo) continue;   /* persistent BO: no per-submit weight copy */
                const int g = g0 + bi;
                uint8_t *w_v = (uint8_t *)((uintptr_t)bo_w.vaddr + (size_t)bi * w_seg);
                npu_wtile_copy(w_v, wq_src, prec, g, n0, ncur,
                               Np, Np_tile, Kb, wblk, wrow);
            }
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_pack += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
#endif

            /* Regenerate the regcmd programs for THIS submit. in/out IOVAs are
             * fixed per segment; the weight IOVA differs per (tile, block) when
             * the weight lives in a persistent BO (legacy reuses the fixed bo_w
             * window). */
            uint64_t rc_addrs[3];
            int n = 0;
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t0);   /* t_gen: regcmd regen */
#endif
            for (int bi = 0; bi < bcur; bi++) {
                uint64_t *op = (uint64_t *)((uintptr_t)d->rc_bo.vaddr +
                                            (size_t)bi * (size_t)n * 8);
                uint32_t w_dma = wbo
                    ? (uint32_t)(wbo->haddr +
                                 ((size_t)t * n_blocks + (size_t)(g0 + bi)) * w_seg)
                    : (uint32_t)(bo_w.haddr + (size_t)bi * w_seg);
                int ni = npu_direct_gen_matmul_regs(op, VLLM_NPU_REGS_MAX,
                                                    (uint32_t)(bo_in.haddr + (size_t)bi * in_seg),
                                                    w_dma,
                                                    (uint32_t)(bo_out.haddr + (size_t)bi * (out_seg + 64)),
                                                    M, Np_tile, Kb, prec);
                if (ni < 0 || (bi > 0 && ni != n)) {
                    if (d->verbosity > 0)
                        fprintf(stderr, "[NPU-DIRECT] gen regs failed bi=%d ni=%d\n", bi, ni);
                    rc = -1;
                    break;
                }
                if (bi == 0) n = ni;
                rc_addrs[bi] = d->rc_bo.haddr + (size_t)bi * (size_t)n * 8;
            }
            if (rc != 0) break;
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_gen += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
            clock_gettime(CLOCK_MONOTONIC, &t0);   /* t_sb: sct/tag/mark prep */
#endif

            if (getenv("VLLM_NPU_DBG") && t == 0 && g0 == 0) {
                const uint8_t *ib = (const uint8_t *)(uintptr_t)bo_in.vaddr;
                const uint8_t *wb = (const uint8_t *)(uintptr_t)(wbo ? wbo->vaddr : bo_w.vaddr);
                printf("[NPU-DIRECT] geom M=%d N=%d tiles=%dx%d Kb=%d blocks=%d n=%d\n",
                       M, N, n_tiles, Nt, Kb, n_blocks, n);
                printf("[NPU-DIRECT] haddr in=%x w=%x out=%x rc=%x\n",
                       (unsigned)bo_in.haddr,
                       (unsigned)(wbo ? wbo->haddr : bo_w.haddr),
                       (unsigned)bo_out.haddr, (unsigned)d->rc_bo.haddr);
                printf("[NPU-DIRECT] in[0..15]=%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x\n",
                       ib[0],ib[1],ib[2],ib[3],ib[4],ib[5],ib[6],ib[7],
                       ib[8],ib[9],ib[10],ib[11],ib[12],ib[13],ib[14],ib[15]);
                printf("[NPU-DIRECT] w[0..15]=%02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x %02x%02x%02x%02x\n",
                       wb[0],wb[1],wb[2],wb[3],wb[4],wb[5],wb[6],wb[7],
                       wb[8],wb[9],wb[10],wb[11],wb[12],wb[13],wb[14],wb[15]);
            }

            /* Submit: 1 task on core0 (NB=1) or bcur tasks one per core
             * (NB=3), each core executing its own single task in parallel.
             * core_mask selects the cores; subcore_task entries follow the
             * driver's use_core_num indexing (1-2 cores -> [core_index],
             * 3 cores -> [core_index+2]). */
            int cmask;
            struct rknpu_subcore_task sct[5];
            memset(sct, 0, sizeof(sct));
            if (bcur == 1) {
                cmask = d->core_mask;          /* 0x1: core0 only */
                sct[0].task_start = 0; sct[0].task_number = 1;
            } else if (bcur == 2) {
                cmask = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK;
                sct[0].task_start = 0; sct[0].task_number = 1;
                sct[1].task_start = 1; sct[1].task_number = 1;
            } else {                           /* bcur == 3 */
                cmask = RKNPU_CORE0_MASK | RKNPU_CORE1_MASK | RKNPU_CORE2_MASK;
                sct[2].task_start = 0; sct[2].task_number = 1;
                sct[3].task_start = 1; sct[3].task_number = 1;
                sct[4].task_start = 2; sct[4].task_number = 1;
            }
            char tag[64];
            snprintf(tag, sizeof(tag), "gemm:t%d-b%d-%d", t, g0, bcur);
            npu_dbg_mark(tag);
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_sb += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
            clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
            /* CACHEABLE bo_in (VLLM_IN_UC=0): flush the freshly packed
             * features TO_DEVICE so the NPU's IOMMU DMA sees them. */
            {
                const char *eu3 = getenv("VLLM_IN_UC");
                int in_uc = (!eu3 || eu3[0] == '\0' || eu3[0] == '1');
                if (!in_uc &&
                    drm_bo_sync_range(d, &bo_in, RKNPU_MEM_SYNC_TO_DEVICE, 0,
                                      (uint64_t)in_seg * NB) != 0) {
                    rc = -1;
                    break;
                }
            }
            rc = drm_submit_pc(d, &d->task_bo, rc_addrs, (uint32_t)n, bcur, cmask, sct);
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_sub += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
#endif
            if (rc != 0) {
                /* Do NOT retry: a timed-out job leaves the 0.9.6 driver wedged
                 * in its abort path (observed D-state hang when a new job is
                 * queued while the old one aborts - two jobs pile up and the
                 * process never returns). Return failure so the caller falls
                 * back to the CPU; correctness is preserved, only the offload
                 * is skipped. */
                if (d->verbosity > 0)
                    fprintf(stderr, "[NPU-DIRECT] t%d g%d/%d submit failed: %s "
                            "(CPU fallback)\n", t, g0, n_blocks, strerror(errno));
                break;
            }

            /* Accumulate the bcur partials (each from its own out segment).
             * Hot loop: M x Np fp32 FMA per block. od lives in an UNCACHED NPU
             * BO, so every element read is a DRAM round trip - the raw cube's
             * (nn>>2)*M*4 access stride wastes 3/4 of each 64B line (measured
             * ~62% of wall time). Copy the segment to a cached buffer with one
             * SEQUENTIAL memcpy (full line usage), then accumulate from the
             * cached copy with b_scale gathered to a contiguous bcol. */
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t0);
            struct timespec ta0 = t0;   /* t_acc: NOT clobbered by pass timers */
            /* t_acc is single-threaded and hot: pin to an A76 core so the
             * scheduler never drops it on an A55 (L2 128KB) - the sub-block
             * working set (tbuf+cd+out ~448KB) needs the A76's 512KB L2
             * (measured 4ms on a pinned A76 vs ~60ms in-engine). Restore the
             * original affinity afterwards. */
            cpu_set_t tacc_orig, tacc_pin;
            CPU_ZERO(&tacc_orig);
            sched_getaffinity(0, sizeof(tacc_orig), &tacc_orig);
            CPU_ZERO(&tacc_pin);
            CPU_SET(4, &tacc_pin);
            sched_setaffinity(0, sizeof(tacc_pin), &tacc_pin);
            tacc_cpu_dbg = sched_getcpu();
            {
                struct timespec tq;
                clock_gettime(CLOCK_MONOTONIC, &tq);
                t_aff += (tq.tv_sec - t0.tv_sec) + 1e-9 * (tq.tv_nsec - t0.tv_nsec);
            }
#endif
            for (int bi = 0; bi < bcur; bi++) {
                const int g = g0 + bi;
                /* Invalidate ONLY this segment (1MB) before reading it - the
                 * old full-BO 3MB FROM_DEVICE sync per submit made every cube
                 * read pay DRAM round trips and dominated t_acc. Uncached BOs
                 * (VLLM_OUT_UC default) have no CPU cache to keep coherent and
                 * skip this entirely. */
                {
                    const char *eu2 = getenv("VLLM_OUT_UC");
                    int out_uc = (!eu2 || eu2[0] == '\0' || eu2[0] == '1');
                    if (!out_uc &&
                        drm_bo_sync_range(d, &bo_out, RKNPU_MEM_SYNC_FROM_DEVICE,
                                          (uint64_t)bi * (out_seg + 64),
                                          (uint64_t)out_seg) != 0) {
                        rc = -1;
                        break;
                    }
                }
                const void *ov = (const void *)((uintptr_t)bo_out.vaddr +
                                                (size_t)bi * (out_seg + 64));
                if (prec == 0) {
                    const int32_t *cd = (const int32_t *)ov;   /* NPU cube (cached BO) */
                    float bcol[1024];
#ifdef __linux__
                    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
                    for (int nn = 0; nn < ncur; nn++)
                        bcol[nn] = b_scale[(size_t)(n0 + nn) * n_blocks + g];
#ifdef __linux__
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    t_bc += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
#endif
#if ST_HAVE_NEON
                    /* Cube layout [ncur/4][M][4] int32 (DMA-written, cached BO).
                     * TWO passes per N SUB-BLOCK (NPU_TACC_NB wide):
                     *   pass 1: transpose cube -> row-major tbuf (sub-block view)
                     *   pass 2: accumulate from row-major tbuf, threads over m.
                     * The full-width tbuf is M*ncur*4 = 1MB at M=256, far above
                     * the A76 L2 (512KB), so the scattered 16B tbuf writes in
                     * pass 1 miss L2 and pay a read-modify-write per element.
                     * Splitting N keeps tbuf (M*NB*4) + out segment + cd segment
                     * all L2-hot (~288KB at M=256, NB=128), so the scattered
                     * 16B writes hit cache and pass 2 streams. Per-(m,n)
                     * arithmetic (fma over the g loop, asc*bcol) is unchanged
                     * -> bit-identical to the strided path. */
                    const int tsub = 128;    /* N per sub-block (L2 blocking) */
                    for (int sub0 = 0; sub0 < ncur; sub0 += tsub) {
                        const int subn = (ncur - sub0 < tsub) ? (ncur - sub0) : tsub;
                        const int ng0 = sub0 >> 2;
                        const int sub_ng = subn >> 2;   /* local gic count */
                        const int32_t *c0 = cd + (size_t)ng0 * (M * 4);
#ifdef __linux__
                        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
                        /* Single-threaded accumulate: parallelising pass1/pass2
                         * is a NET LOSS on the 8-core RK3588 (4xA55+4xA76) -
                         * measured 2-3x SLOWER with 8 OMP threads (t_acc 293ms
                         * vs 101ms at M=256 K=6144) from the A55/A76 mix and
                         * shared-tbuf/out cache-coherence traffic. Single-
                         * threaded the 16B scattered writes hit the local L2.
                         *
                         * VLLM_P1=3 (default): copy the sub-block cube into a
                         * plain cached staging buffer (tmps) with one streamed
                         * memcpy, then transpose from tmps. The NPU output BO
                         * is DMA-written through the IOMMU (NOT coherent with
                         * the CPU cache), so every 64B line read straight from
                         * it is a DRAM round trip - the per-gic 16B reads pay
                         * that latency every time and dominate t_acc (measured
                         * p1 56ms vs 4ms for the identical transpose over a
                         * cached buffer). memcpy streams full lines into L2 and
                         * the transpose then runs at cache speed. */
                        int p1mode = 3;
                        {
                            const char *pe = getenv("VLLM_P1");
                            if (pe && pe[0]) p1mode = atoi(pe);
                        }
                        if (p1mode == 3) {
#ifdef __linux__
                            struct timespec tq0, tq1;
                            clock_gettime(CLOCK_MONOTONIC, &tq0);
#endif
                            memcpy(tmps, c0, (size_t)sub_ng * (M * 4) * sizeof(int32_t));
#ifdef __linux__
                            clock_gettime(CLOCK_MONOTONIC, &tq1);
                            t_p1m += (tq1.tv_sec - tq0.tv_sec) + 1e-9 * (tq1.tv_nsec - tq0.tv_nsec);
                            clock_gettime(CLOCK_MONOTONIC, &tq0);
#endif
                            for (int gic = 0; gic < sub_ng; gic++) {
                                const int32_t *sg = tmps + (size_t)gic * (M * 4);
                                for (int m = 0; m < M; m++)
                                    vst1q_s32((int32_t *)tbuf + (size_t)m * subn +
                                                  (size_t)gic * 4,
                                              vld1q_s32(sg + (size_t)m * 4));
                            }
#ifdef __linux__
                            clock_gettime(CLOCK_MONOTONIC, &tq1);
                            t_p1t += (tq1.tv_sec - tq0.tv_sec) + 1e-9 * (tq1.tv_nsec - tq0.tv_nsec);
#endif
                        } else {
                        for (int gic = 0; gic < sub_ng; gic++) {
                            const int32_t *sg = c0 + (size_t)gic * (M * 4);
                            /* Prefetch the next gic's 4KB cd segment: the cube
                             * is freshly DMA-written (cold in cache after the
                             * segment sync), so the 4KB stride miss latency is
                             * what dominates the transpose. */
                            __builtin_prefetch(c0 + (size_t)(gic + 2) * (M * 4));
                            for (int m = 0; m < M; m++)
                                vst1q_s32((int32_t *)tbuf + (size_t)m * subn +
                                              (size_t)gic * 4,
                                          vld1q_s32(sg + (size_t)m * 4));
                        }
                        }
#ifdef __linux__
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        t_p1 += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
                        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
                        for (int m = 0; m < M; m++) {
                            float *orow = &out[m * N + n0 + sub0];
                            const float asc = a_scale[m * n_blocks + g];
                            const float32x4_t asv = vdupq_n_f32(asc);
                            const int32_t *row = (const int32_t *)tbuf +
                                                 (size_t)m * subn;
                            int nn = 0;
                            for (; nn + 4 <= subn; nn += 4) {
                                float32x4_t f   = vcvtq_f32_s32(vld1q_s32(&row[nn]));
                                float32x4_t acc = vld1q_f32(&orow[nn]);
                                acc = vfmaq_f32(acc, vmulq_f32(f, asv),
                                                vld1q_f32(&bcol[sub0 + nn]));
                                vst1q_f32(&orow[nn], acc);
                            }
                            for (; nn < subn; nn++)
                                orow[nn] += (float)row[nn] * asc * bcol[sub0 + nn];
                        }
#ifdef __linux__
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        t_p2 += (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
#endif
                    }
#else
                    const int tsub = 128;    /* N per sub-block (L2 blocking) */
                    for (int sub0 = 0; sub0 < ncur; sub0 += tsub) {
                        const int subn = (ncur - sub0 < tsub) ? (ncur - sub0) : tsub;
                        const int ng0 = sub0 >> 2;
                        const int sub_ng = subn >> 2;   /* local gic count */
                        const int32_t *c0 = cd + (size_t)ng0 * (M * 4);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(M >= 24)
#endif
                        for (int gic = 0; gic < sub_ng; gic++) {
                            const int32_t *sg = c0 + (size_t)gic * (M * 4);
                            for (int m = 0; m < M; m++)
                                for (int i = 0; i < 4; i++)
                                    tbuf[(size_t)m * subn + (size_t)gic * 4 + i] =
                                        sg[(size_t)m * 4 + i];
                        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(M >= 24)
#endif
                        for (int m = 0; m < M; m++) {
                            float *orow = &out[m * N + n0 + sub0];
                            const float asc = a_scale[m * n_blocks + g];
                            const int32_t *row = (const int32_t *)tbuf +
                                                 (size_t)m * subn;
                            for (int nn = 0; nn < subn; nn++)
                                orow[nn] += (float)row[nn] * asc * bcol[sub0 + nn];
                        }
                    }
#endif
                } else {
                    const int16_t *cd = (const int16_t *)ov;
                    float bcol[1024];
                    for (int nn = 0; nn < ncur; nn++)
                        bcol[nn] = b_scale[(size_t)(n0 + nn) * n_blocks + g];
#if ST_HAVE_NEON
                    /* Same transpose for the int16 cube [g][m][8] -> [m][n]. */
                    for (int m = 0; m < M; m++) {
                        int16_t *dst = (int16_t *)tbuf + (size_t)m * ncur;
                        const int16_t *src = cd + 8 * m;
                        int g = 0, ng = ncur >> 3;
                        for (; g + 4 <= ng; g += 4) {
                            int16x8_t v0 = vld1q_s16(src + (size_t)(g + 0) * (M * 8));
                            int16x8_t v1 = vld1q_s16(src + (size_t)(g + 1) * (M * 8));
                            int16x8_t v2 = vld1q_s16(src + (size_t)(g + 2) * (M * 8));
                            int16x8_t v3 = vld1q_s16(src + (size_t)(g + 3) * (M * 8));
                            vst1q_s16(dst + (size_t)g * 8 + 0, v0);
                            vst1q_s16(dst + (size_t)g * 8 + 8, v1);
                            vst1q_s16(dst + (size_t)g * 8 + 16, v2);
                            vst1q_s16(dst + (size_t)g * 8 + 24, v3);
                        }
                        for (; g < ng; g++)
                            vst1q_s16(dst + (size_t)g * 8,
                                      vld1q_s16(src + (size_t)g * (M * 8)));
                    }
                    for (int m = 0; m < M; m++) {
                        float *orow = &out[m * N + n0];
                        const float asc = a_scale[m * n_blocks + g];
                        const float32x4_t asv = vdupq_n_f32(asc);
                        const int16_t *row = (const int16_t *)tbuf + (size_t)m * ncur;
                        int nn = 0;
                        for (; nn + 8 <= ncur; nn += 8) {
                            int16x8_t q = vld1q_s16(&row[nn]);
                            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q)));
                            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q)));
                            float32x4_t acc0 = vld1q_f32(&orow[nn]);
                            float32x4_t acc1 = vld1q_f32(&orow[nn + 4]);
                            acc0 = vfmaq_f32(acc0, vmulq_f32(f0, asv), vld1q_f32(&bcol[nn]));
                            acc1 = vfmaq_f32(acc1, vmulq_f32(f1, asv), vld1q_f32(&bcol[nn + 4]));
                            vst1q_f32(&orow[nn], acc0);
                            vst1q_f32(&orow[nn + 4], acc1);
                        }
                        for (; nn < ncur; nn++)
                            orow[nn] += (float)row[nn] * asc * bcol[nn];
                    }
#else
                    for (int m = 0; m < M; m++) {
                        int16_t *dst = (int16_t *)tbuf + (size_t)m * ncur;
                        const int16_t *src = cd + 8 * m;
                        for (int g = 0; g < ncur >> 3; g++)
                            for (int i = 0; i < 8; i++)
                                dst[(size_t)g * 8 + i] = src[(size_t)g * (M * 8) + i];
                    }
                    for (int m = 0; m < M; m++) {
                        float *orow = &out[m * N + n0];
                        const float asc = a_scale[m * n_blocks + g];
                        const int16_t *row = (const int16_t *)tbuf + (size_t)m * ncur;
                        for (int nn = 0; nn < ncur; nn++)
                            orow[nn] += (float)row[nn] * asc * bcol[nn];
                    }
#endif
                }
            }
#ifdef __linux__
            clock_gettime(CLOCK_MONOTONIC, &t1);
            t_acc += (t1.tv_sec - ta0.tv_sec) + 1e-9 * (t1.tv_nsec - ta0.tv_nsec);
            sched_setaffinity(0, sizeof(tacc_orig), &tacc_orig);
#endif
        }
    }

#ifdef __linux__
    {
        /* getenv() returns "0" (non-NULL) when the supervisor injects
         * VLLM_NPU_TIMING=0 - treat any non-'0' value as the real switch. */
        const char *tv = getenv("VLLM_NPU_TIMING");
        if (tv && tv[0] && tv[0] != '0') {
            struct timespec tw1;
            clock_gettime(CLOCK_MONOTONIC, &tw1);
            double t_wall = (tw1.tv_sec - tw0.tv_sec) + 1e-9 * (tw1.tv_nsec - tw0.tv_nsec);
            double t_hide = t_wall - t_setup - t_pack - t_sub - t_acc - t_gen - t_sb - t_aff;
            printf("[NPU-DIRECT] M=%d N=%d K=%d G=%d prec=%d t_pack=%.0fus "
                   "t_sub=%.0fus t_acc=%.0fus submits=%d setup=%.0fus "
                   "wall=%.0fus hide=%.0fus gen=%.0fus sb=%.0fus aff=%.0fus p1m=%.0fus p1t=%.0fus",
                   M, N, K, G, prec, t_pack * 1e6, t_sub * 1e6, t_acc * 1e6,
                   n_tiles * ((n_blocks + NB - 1) / NB),
                   t_setup * 1e6, t_wall * 1e6, t_hide * 1e6,
                   t_gen * 1e6, t_sb * 1e6, t_aff * 1e6,
                   t_p1m * 1e6, t_p1t * 1e6);
        }
        const char *tv2 = getenv("VLLM_NPU_TIMING2");
        if (tv2 && tv2[0] && tv2[0] != '0')
            printf(" t_bc=%.0fus t_p1=%.0fus t_p2=%.0fus cpu=%d",
                   t_bc * 1e6, t_p1 * 1e6, t_p2 * 1e6, tacc_cpu_dbg);
        if (tv && tv[0] && tv[0] != '0') printf("\n");
        fflush(stdout);   /* stdout is block-buffered under nohup redirect */
    }
#endif

    free(apad);
    free(tbuf);
    free(tmps);
    vllm_npu_direct_bo_free(d, &bo_out);
    vllm_npu_direct_bo_free(d, &bo_w);
    vllm_npu_direct_bo_free(d, &bo_in);
    return rc;
}

/* ================================================================
 * Self test (PASS/FAIL; validates every stage up to the calibration gate)
 *
 * Staged bring-up (VLLM_NPU_STAGE=bo|full, default full):
 *   bo    - BO alloc/free round-trip only (device layer)
 *   full  - + int8 and int4 group-wise matmul vs CPU reference
 * (VLLM_NPU_PREC=0|1 restricts to one dtype. There is deliberately no bare
 * PC-probe stage: enabling the blocks without compute geometry hangs the NPU
 * and locks the board.)
 * ================================================================ */
static int vllm_npu_direct_ensure_cache(vllm_npu_direct_t *d) {
    if (d->cache_ok) return 0;
    /* rc_bo holds VLLM_NPU_BATCH_MAX regcmd programs (one per K-block in a
     * batch submit) packed at (n+4)-word stride, task_bo the matching
     * rknpu_task array. (n+4) covers the 4-word HW fetch tail per program. */
    if (vllm_npu_direct_bo_alloc(d, &d->rc_bo,
                                 (VLLM_NPU_REGS_MAX + 4) * 8 * VLLM_NPU_BATCH_MAX) != 0 ||
        vllm_npu_direct_bo_alloc_ex(d, &d->task_bo, 4096,
                                    NPU_MEM_KERNEL_MAPPING) != 0) {
        vllm_npu_direct_bo_free(d, &d->rc_bo);
        vllm_npu_direct_bo_free(d, &d->task_bo);
        return -1;
    }
    memset((void *)(uintptr_t)d->rc_bo.vaddr, 0, d->rc_bo.size);
    d->cache_ok = 1;
    return 0;
}

int vllm_npu_direct_selftest(vllm_npu_direct_t *d, const char *sdk_tag) {
    if (!d) { fprintf(stderr, "[NPU-DIRECT] FAIL no device handle\n"); return 1; }
    if (d->fd < 0) { fprintf(stderr, "[NPU-DIRECT] SKIP no rknpu node\n"); return 0; }

    printf("[NPU-DIRECT] backend=%s hw_version=%u drv_version=%u [%s]\n",
           d->backend == NPU_BACKEND_DRM ? "DRM" : "misc",
           d->hw_version, d->drv_version, sdk_tag);

    const char *stage = getenv("VLLM_NPU_STAGE");
    if (!stage || !stage[0]) stage = "full";
    int do_gemm = (strcmp(stage, "full") == 0);
    /* VLLM_NPU_PREC: 0 = int8 only, 1 = int4 only, unset = both. */
    int prec_filter = -1;
    {
        const char *pf = getenv("VLLM_NPU_PREC");
        if (pf && pf[0]) prec_filter = (int)strtol(pf, NULL, 0);
    }

    int failures = 0;
    g_npu_calib_force = 1;   /* exercise the full submit path for calibration */
    /* Diagnostic: clear the kernel ring buffer so the submit's driver messages
     * (job timeout / invalid irq status / etc.) are the only ones we dump. */
    system("dmesg -c > /dev/null 2>&1");
    printf("[NPU-DIRECT] stage=%s (regcmd table %s) [%s]\n",
           stage,
#ifndef VLLM_NPU_DIRECT_READY
           "UNVERIFIED - calibration mode",
#else
           "verified",
#endif
           sdk_tag);
    fflush(stdout);

    /* Stage 1: BO alloc/free round-trip */
    vllm_npu_bo_t bo;
    if (vllm_npu_direct_bo_alloc(d, &bo, 65536) == 0) {
        uint64_t v = bo.vaddr, h = bo.haddr;
        memset((void *)(uintptr_t)v, 0xA5, 65536);
        int ok = (v != 0 && h != 0);
        vllm_npu_direct_bo_free(d, &bo);
        printf("[NPU-DIRECT] BO alloc/map/free %s (v=%llx h=%llx)\n",
               ok ? "PASS" : "FAIL", (unsigned long long)v, (unsigned long long)h);
        fflush(stdout);
        if (!ok) failures++;
    } else {
        printf("[NPU-DIRECT] BO alloc FAIL\n");
        fflush(stdout);
        failures++;
    }

    /* Stage 2: group-wise matmul vs CPU reference. Runs the full pipeline
     * (pack -> regcmd -> DRM SUBMIT -> readback) against a CPU oracle; the
     * int8 stage is the primary calibration target, int4 secondary.
     * NOTE: no "bare PC probe" stage - firing the block enable with no
     * compute geometry hangs the NPU and locks the board (observed). */
    if (do_gemm && failures == 0) {
        /* Real-engine shape: G=32 Q8_0/Q4_0 blocks, K=2048 spans 64 blocks
         * (the K-chunked path submits one 32-wide block per pass and
         * accumulates the dequantised partials in fp32). N is deliberately
         * NOT aligned (int8 2052 -> pad 2080; int4 1036 -> pad 1088) so the
         * N-pad path is exercised at full scale. Buffers are malloc'd - the
         * stack cannot hold M*K / N*K at engine sizes. */
        /* int4 real prefill shapes (M=96/128) in addition to the tiny M=8
         * calibration case - the int4 path only ever ran M=8 and the Q4 NPU
         * prefill produced garbage; these pin down the failing geometry. */
        const int Ms[] = { 8, 64, 64, 64, 8, 8, 32, 204 };
        const int Ks[] = { 2048, 4096, 4096, 4096, 2048, 4096, 4096, 4096 };
        const int Ns[] = { 4096, 4096, 12288, 1024, 2048, 4096, 4096, 4096 };
        int G = 32;
        {
            const char *ge = getenv("VLLM_NPU_G");
            if (ge && ge[0]) {
                int v = atoi(ge);
                if (v > 0 && (v % 32) == 0) G = v;   /* g256 wmode: G=256 */
            }
        }
        const int precs[] = { 0, 0, 0, 0, 1, 1, 0, 1 };
        const char *names[] = { "int8 (Q8_0)", "int8 M=64 K=4096 N=4096 (Q proj)",
                                "int8 M=64 K=4096 N=12288 (gate/up)",
                                "int8 M=64 K=4096 N=1024 (K/V proj)",
                                "int4 M=8 K=2048 N=2048 (calib)",
                                "int4 M=8 K=4096 N=4096 (K sweep)",
                                "int8 M=32 K=4096 N=4096 (Q/O projection)",
                                "int4 M=204 K=4096 N=4096 (padded prefill)" };
        for (int p = 0; p < 8; p++) {
            int prec = precs[p];
            if (prec_filter >= 0 && prec != prec_filter) {
                printf("[NPU-DIRECT] gemm_gw %s SKIP (VLLM_NPU_PREC=%d)\n",
                       names[p], prec_filter);
                fflush(stdout);
                continue;
            }
            const int M = Ms[p], K = Ks[p], N = Ns[p];
            const int n_blocks = K / G;
            int8_t *Aq = (int8_t *)malloc((size_t)M * K);
            int8_t *Wq = (int8_t *)malloc((size_t)N * K);
            float  *a_scale = (float  *)malloc((size_t)M * n_blocks * sizeof(float));
            float  *b_scale = (float  *)malloc((size_t)N * n_blocks * sizeof(float));
            float  *out = (float  *)malloc((size_t)M * N * sizeof(float));
            float  *ref = (float  *)malloc((size_t)M * N * sizeof(float));
            if (!Aq || !Wq || !a_scale || !b_scale || !out || !ref) {
                free(Aq); free(Wq); free(a_scale); free(b_scale);
                free(out); free(ref);
                printf("[NPU-DIRECT] gemm_gw %s FAIL (malloc)\n", names[p]);
                fflush(stdout);
                failures++;
                continue;
            }
            /* int4 covers the full LEGAL [-8,7] range with a row-varying
             * pattern ((k*5 + m*3) % 16 - 8) so the M dimension of the
             * feature cube is exercised too - a K-periodic pattern (m*K*5
             * % 16 == 0 at K%16==0) made every row identical and could not
             * see an M-layout error. Note: +8 is NOT representable in 4-bit
             * two's complement (its nibble 0b1000 decodes as -8), which is
             * exactly the earlier (i*5)%17-8 FAIL; Q4_0 weights are [-8,7]
             * so this is the only range the engine will ever present. */
            for (int m2 = 0; m2 < M; m2++)
                for (int k = 0; k < K; k++)
                    Aq[m2 * K + k] = (int8_t)(
                        (prec == 1) ? ((k * 5 + m2 * 3) % 16 - 8)
                                    : ((m2 * K + k) * 5 % 17 - 8));
            /* int4 is [-8,7]: the int8-range Wq ([-9,9]) would be nibble-truncated
             * on the HW (9->-7, -9->7), so the int4 pass uses a fitting range. */
            for (int i = 0; i < N * K; i++)
                Wq[i] = (int8_t)((prec == 1) ? ((i * 7) % 15 - 7)
                                             : ((i * 7) % 19 - 9));
            for (int i = 0; i < M * n_blocks; i++)
                a_scale[i] = 0.5f + 0.1f * (float)(i % 3);
            for (int i = 0; i < N * n_blocks; i++)
                b_scale[i] = 0.2f + 0.05f * (float)(i % 5);
            /* CPU oracle: block-wise dequantisation like the NPU path -
             * per 32-wide block, raw int dot * a_scale * b_scale, summed. */
            for (int m = 0; m < M; m++)
                for (int nn = 0; nn < N; nn++) {
                    double acc = 0.0;
                    for (int g = 0; g < n_blocks; g++) {
                        int64_t p = 0;
                        for (int k = 0; k < G; k++)
                            p += (int64_t)Aq[m * K + g * G + k] *
                                 (int64_t)Wq[nn * K + g * G + k];
                        acc += (double)p * a_scale[m * n_blocks + g] *
                                          b_scale[nn * n_blocks + g];
                    }
                    ref[m * N + nn] = (float)acc;
                }
            int rc = vllm_npu_direct_gemm_gw(d, out, Aq, a_scale, Wq, b_scale,
                                             M, N, K, G, prec, 0);
            /* Diagnostic: append the kernel messages the submit triggered
             * (driver timeout / irq-status diagnostics) to the app log. */
            system("dmesg | tail -60 >> /root/calib.log 2>&1");
            if (rc == 0) {
                double max_err = 0.0;
                for (int i = 0; i < M * N; i++)
                    max_err = fmax(max_err, fabs((double)out[i] - (double)ref[i]));
                int pass = (max_err < 1e-3);
                printf("[NPU-DIRECT] gemm_gw %dx%dx%d %s (max_err=%.2e)\n",
                       M, N, K, pass ? "PASS" : "FAIL", max_err);
                fflush(stdout);
                if (!pass) {
                    for (int m = 0; m < M && m < 4; m++)
                        printf("    row%d: got %.3f %.3f %.3f ref %.3f %.3f %.3f\n",
                               m, out[m*N+0], out[m*N+1], out[m*N+2],
                               ref[m*N+0], ref[m*N+1], ref[m*N+2]);
                    fflush(stdout);
                    failures++;
                }
            } else if (rc == -2) {
                printf("[NPU-DIRECT] gemm_gw %dx%dx%d %s SKIP (uncalibrated)\n",
                       M, N, K, names[p]);
                fflush(stdout);
            } else {
                printf("[NPU-DIRECT] gemm_gw %dx%dx%d %s FAIL (rc=%d)\n",
                       M, N, K, names[p], rc);
                fflush(stdout);
                failures++;
            }
            free(Aq); free(Wq); free(a_scale); free(b_scale);
            free(out); free(ref);
        }
    } else if (!do_gemm && failures == 0) {
        printf("[NPU-DIRECT] gemm stages skipped (VLLM_NPU_STAGE=%s)\n", stage);
        fflush(stdout);
    }
    return failures;
}

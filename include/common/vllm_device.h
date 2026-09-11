/* ================================================================
 * vllm_device.h - Device classification & weight-adaptation profiles
 *
 * Classifies the target hardware along three axes:
 *   architecture : x86 | arm64
 *   class        : pc | embedded
 *   model        : named device (e.g. "rk3588-opi5")
 *
 * Each named model carries a weight-adaptation profile: recommended
 * quantization mode, NPU backend, OMP/worker threads, memory budget,
 * supported-wmode whitelist and a default model (weight) directory.
 * The web admin page pre-fills the config from the profile; the user can
 * override the detected device with --device / config "device".
 * ================================================================ */
#ifndef VLLM_DEVICE_H
#define VLLM_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VDEV_ARCH_UNKNOWN = 0,
    VDEV_ARCH_X86,      /* x86-64 (AVX/AVX2/AVX512) */
    VDEV_ARCH_ARM64,    /* aarch64 (NEON) */
} VDevArch;

typedef enum {
    VDEV_CLASS_UNKNOWN = 0,
    VDEV_CLASS_PC,       /* desktop / server / laptop */
    VDEV_CLASS_EMBEDDED, /* SBC / embedded box (e.g. RK3588 Orange Pi 5) */
} VDevClass;

/* One named device's weight-adaptation profile (see the table in
 * vllm_device.c). NULL strings mean "no recommendation / not supported". */
typedef struct {
    const char *id;             /* canonical id, e.g. "arm-rk3588-opi5" */
    const char *name;           /* human-readable (zh), e.g. "RK3588 香橙派 5" */
    const char *name_en;        /* human-readable (en); 管理页按当前语言二选一 */
    VDevArch    arch;
    VDevClass   cls;
    const char *npu;            /* NPU backend id ("rk3588-direct") or NULL */
    const char *default_wmode;  /* recommended wmode (serve-safe) */
    int         default_omp;    /* OMP_NUM_THREADS recommendation */
    int         default_threads;/* HTTP worker threads recommendation */
    long        mem_limit_mb;   /* memory budget hint (0 = auto) */
    int         default_npu_load;   /* VLLM_NPU_LOAD recommendation (0/1/2) */
    int         default_prefill_q8; /* --prefill-q8 (Q8 prefill + Q4 decode) */
    const char *wmodes;         /* comma-separated supported wmode whitelist */
    const char *default_model_dir;  /* default model/weight directory */
} VDevProfile;

/* Detected/selected device state. */
typedef struct {
    VDevArch arch;
    VDevClass cls;
    const VDevProfile *profile; /* matched profile (never NULL after detect) */
    char model_id[64];          /* profile id or "generic-<arch>" */
    char model_hint[128];       /* detected string (device-tree / cpuinfo) */
} VDevInfo;

const char *vdev_arch_str(VDevArch a);
const char *vdev_class_str(VDevClass c);

/* Look up a profile by id (case-insensitive). NULL if unknown. */
const VDevProfile *vdev_lookup(const char *id);

/* Auto-detect the device (arch from compile flags, model from
 * /proc/device-tree/model and /proc/cpuinfo on Linux; x86 maps to the
 * generic PC profile). info->profile is always set. */
const VDevProfile *vdev_detect(VDevInfo *info);

/* All profiles (terminated by id==NULL) - for the admin page device list. */
const VDevProfile *vdev_all(void);

/* True when wmode is in the profile's whitelist (always true if empty). */
int vdev_wmode_allowed(const VDevProfile *p, const char *wmode);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_DEVICE_H */

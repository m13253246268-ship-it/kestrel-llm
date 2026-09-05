/* ================================================================
 * vllm_device.c - Device classification & weight-adaptation profiles
 * ================================================================ */
#include "vllm_device.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* ---------- device table ----------
 * Keep the whitelists aligned with main.c --wmode parsing and the admin
 * page (q4|q4i|q8|g256|dual). g256 (NPU K-block 256 layout) only exists on
 * the RK3588 profile. */
static const VDevProfile g_devices[] = {
    { "x86-pc", "x86 通用 PC（x86-64）", VDEV_ARCH_X86, VDEV_CLASS_PC,
      NULL, "q8", 8, 8, 0, 0, 0, "q8,q4,q4i,dual", NULL },
    { "arm-rk3588-opi5", "RK3588 香橙派 5（ARM 嵌入式）",
      VDEV_ARCH_ARM64, VDEV_CLASS_EMBEDDED,
      "rk3588-direct", "dual", 4, 4, 0, 1, 0, "g256,q4,q4i,q8,dual",
      "/NewVLLM/Modl/千问3_VL_8B_Instruct" },
    { "arm-generic", "ARM 通用嵌入式", VDEV_ARCH_ARM64, VDEV_CLASS_EMBEDDED,
      NULL, "dual", 4, 4, 0, 0, 0, "q4,q4i,q8,dual", NULL },
    { NULL }
};

const char *vdev_arch_str(VDevArch a) {
    switch (a) {
    case VDEV_ARCH_X86:   return "x86";
    case VDEV_ARCH_ARM64: return "arm64";
    default:              return "unknown";
    }
}

const char *vdev_class_str(VDevClass c) {
    switch (c) {
    case VDEV_CLASS_PC:       return "pc";
    case VDEV_CLASS_EMBEDDED: return "embedded";
    default:                  return "unknown";
    }
}

const VDevProfile *vdev_all(void) {
    return g_devices;
}

const VDevProfile *vdev_lookup(const char *id) {
    if (!id || !id[0]) return NULL;
    for (const VDevProfile *p = g_devices; p->id; p++) {
        if (strcasecmp(p->id, id) == 0) return p;
    }
    return NULL;
}

/* Case-insensitive substring. */
static int str_contains_ci(const char *hay, const char *needle) {
    size_t nl = strlen(needle);
    for (const char *p = hay; *p; p++) {
        if (strncasecmp(p, needle, nl) == 0) return 1;
    }
    return 0;
}

/* Read the ARM board model from /proc/device-tree/model (preferred) or
 * /proc/cpuinfo Hardware/Model. Returns 1 when a string was read. */
static int detect_arm_model(char *buf, size_t cap) {
    FILE *f = fopen("/proc/device-tree/model", "rb");
    if (f) {
        size_t n = fread(buf, 1, cap - 1, f);
        fclose(f);
        buf[n] = '\0';
        return n > 0;
    }
    f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "Hardware", 8) == 0 || strncmp(line, "Model", 5) == 0) {
                const char *v = strchr(line, ':');
                if (v) {
                    v++;
                    while (*v == ' ' || *v == '\t') v++;
                    size_t n = 0;
                    while (v[n] && v[n] != '\n' && v[n] != '\r' && n < cap - 1) {
                        buf[n] = v[n];
                        n++;
                    }
                    buf[n] = '\0';
                    fclose(f);
                    return 1;
                }
            }
        }
        fclose(f);
    }
    return 0;
}

const VDevProfile *vdev_detect(VDevInfo *info) {
    memset(info, 0, sizeof(*info));
#if defined(__aarch64__) || defined(_M_ARM64)
    info->arch = VDEV_ARCH_ARM64;
    info->cls = VDEV_CLASS_EMBEDDED;
#else
    info->arch = VDEV_ARCH_X86;
    info->cls = VDEV_CLASS_PC;
#endif

    const VDevProfile *p = NULL;
    if (info->arch == VDEV_ARCH_ARM64) {
        if (detect_arm_model(info->model_hint, sizeof(info->model_hint))) {
            /* Orange Pi 5 / RK3588 family -> RK3588 profile. */
            if (str_contains_ci(info->model_hint, "RK3588") ||
                str_contains_ci(info->model_hint, "3588") ||
                str_contains_ci(info->model_hint, "OrangePi 5") ||
                str_contains_ci(info->model_hint, "orangepi5")) {
                p = vdev_lookup("arm-rk3588-opi5");
            }
        }
        if (!p) p = vdev_lookup("arm-generic");
    } else {
        p = vdev_lookup("x86-pc");
    }
    info->profile = p;
    if (p) snprintf(info->model_id, sizeof(info->model_id), "%s", p->id);
    else   snprintf(info->model_id, sizeof(info->model_id), "generic-%s",
                    vdev_arch_str(info->arch));
    return p;
}

int vdev_wmode_allowed(const VDevProfile *p, const char *wmode) {
    if (!p || !p->wmodes || !wmode) return 1;
    char copy[128];
    snprintf(copy, sizeof(copy), ",%s,", p->wmodes);
    char want[64];
    snprintf(want, sizeof(want), ",%s,", wmode);
    return str_contains_ci(copy, want);
}

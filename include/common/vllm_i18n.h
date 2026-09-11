/* ================================================================
 * vllm_i18n.h - 引擎侧「少量可现文案」的中英双语。
 *
 * 定位（刻意保持极小）：页面文案由前端字典负责（admin.html / chat.html 的
 * T() 与 data-en），这里只覆盖**前端无法替换**的少数后端串：
 *   - 启动横幅（[SERVE] 管理入口 / 对话入口）——只进日志；
 *   - 少数 API 提示串。
 * 语言由 env VLLM_LANG 决定：zh（默认）/ en。
 *
 * 设备显示名不走这里：/admin/api/device 同时下发 name(zh) 与 name_en，
 * 由管理页按当前语言二选一，切换语言无需重启引擎。
 *
 * 边界：引擎其余日志（约 2000 行 printf/fprintf）保持中文，不在本机制范围内；
 *       日志查看器会原样显示，属已知边界。
 * ================================================================ */
#ifndef VLLM_I18N_H
#define VLLM_I18N_H

#include <stdlib.h>

/* 取当前语言文案。每次调用读 env：getenv 开销可忽略，且免去初始化顺序问题。 */
static inline const char *vllm_tr(const char *zh, const char *en) {
    const char *e = getenv("VLLM_LANG");
    return (e && (e[0] == 'e' || e[0] == 'E')) ? en : zh;
}

#endif /* VLLM_I18N_H */

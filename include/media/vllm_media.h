/**
 * vllm_media.h - 自研媒体解码（零第三方依赖）
 *
 * MP4 (ISO BMFF) 容器解析 + H.264 软件解码器 + YUV→RGB 缩放。
 * 纯 C 标量实现，x86 本地开发 / aarch64 (RK3588) 板上运行同一份源码。
 * 解码帧仅用于视觉推理，采用"从最近 IDR 解码到目标帧"的抽帧策略。
 */
#ifndef VLLM_MEDIA_H
#define VLLM_MEDIA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * MP4 容器解析 (demux)
 * ================================================================ */

#define MPM_MAX_SAMPLES 65536

typedef struct {
    /* 输入（调用方持有） */
    const uint8_t *data;
    size_t         size;

    /* 视频轨道信息 */
    int    width, height;        /* 显示分辨率（crop 后） */
    int    n_frames;             /* 视频 sample 数 */
    int    length_size;          /* AVCC NAL 长度前缀字节数 (1-4) */
    int    timescale;            /* mvhd timescale */
    int64_t duration_ms;         /* 视频时长 (ms) */

    /* SPS / PPS（AVCC 原始 NAL，含 1 字节 NAL header） */
    const uint8_t *sps; int sps_len;
    const uint8_t *pps; int pps_len;

    /* 帧索引 */
    uint32_t *sample_off;        /* [n_frames] 文件内偏移 */
    uint32_t *sample_size;       /* [n_frames] 字节数 */
    uint32_t *sample_pts_ms;     /* [n_frames] 显示时间戳 (ms) */
    uint8_t  *sample_idr;        /* [n_frames] 1 = IDR */

    uint8_t *own_index;          /* 内部一次性分配（含全部数组） */
} MPMovie;

/* 解析 mp4 容器。成功返回 0；data/size 需在解码期间保持有效。 */
int mp4_open(MPMovie *m, const uint8_t *data, size_t size);

/* 释放 mp4_open 分配的内部内存（data 归调用方）。 */
void mp4_close(MPMovie *m);

/* ================================================================
 * H.264 软件解码器
 * ================================================================ */

/* 解码器上下文（一次初始化，多次逐帧解码） */
typedef struct H264Dec H264Dec;

typedef struct {
    const MPMovie *movie;
    H264Dec *dec;              /* 内部解码状态（h264_dec.c） */

    /* 输出帧缓冲（YUV420P，紧凑布局：w*h + w/2*h/2*2）
     * yuv=Y 起点；y_u/y_v 为 U/V 平面（紧邻 Y，便于无 padding 拷贝） */
    uint8_t *yuv;
    uint8_t *y_u, *y_v;
    int      yuv_cap;
} H264Decoder;

/* 初始化解码器（解析 SPS/PPS）。返回 0 成功。 */
int h264_decoder_init(H264Decoder *d, const MPMovie *movie);

/* 释放解码器资源 */
void h264_decoder_free(H264Decoder *d);

/* 解码 frame_idx（1-based sample id），输出到 d->yuv（YUV420P）。
 * 内部从该帧之前最近的 IDR 开始解码到目标帧。
 * 返回 0 成功，-1 失败。 */
int h264_decoder_decode(H264Decoder *d, int frame_idx);

/* ================================================================
 * 帧处理：YUV420 → RGB 缩放
 * ================================================================ */

/* 把 yuv (src_w x src_h) 缩放为 rgb24 (max_edge 最长边限制)。
 * rgb 由调用方分配 (dst_w*dst_h*3)。返回 0 成功。 */
int media_yuv_to_rgb(const uint8_t *yuv, int src_w, int src_h,
                     uint8_t *rgb, int max_edge, int *dst_w, int *dst_h);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_MEDIA_H */

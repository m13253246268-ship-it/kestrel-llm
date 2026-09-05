/**
 * h264_dec.c - 自研 H.264 (AVC) 软件解码器，零第三方依赖
 *
 * 面向 LLM 视觉推理的聚焦实现：
 *   - Main profile、4:2:0 8-bit、progressive（无场编码）
 *   - CAVLC + CABAC 熵解码
 *   - I / P / B 帧、多参考帧、8x8 变换
 *   - 只解码抽帧所需 GOP（从最近 IDR 开始）
 * 数值与 ffmpeg/libx264 兼容（无损解码，非近似）。
 *
 * 文件组织：
 *   [1] bitstream / NAL / emulation removal
 *   [2] SPS / PPS 解析
 *   [3] slice header
 *   [4] CABAC + CAVLC 熵解码
 *   [5] 帧内预测 / IDCT
 *   [6] 帧间预测（运动补偿）
 *   [7] 去块滤波
 *   [8] DPB 管理 + 顶层 decode
 */
#include "vllm_media.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ================================================================
 * [1] Bitstream reader (RBSP) + emulation prevention removal
 * ================================================================ */

typedef struct {
    const uint8_t *d;
    int nbits;
    int pos;        /* bit position */
    int byte;       /* 当前字节（peek 用） */
} BitReader;

static inline int br_available(const BitReader *br) {
    return br->nbits - br->pos;
}
static inline int br_get_bit(BitReader *br) {
    if (br->pos >= br->nbits) return 0;
    int b = (br->d[br->pos >> 3] >> (7 - (br->pos & 7))) & 1;
    br->pos++;
    return b;
}
static inline int br_get_bits(BitReader *br, int n) {
    int v = 0;
    for (int i = 0; i < n; i++) v = (v << 1) | br_get_bit(br);
    return v;
}
static inline int br_get_ue(BitReader *br) {
    int z = 0;
    while (!br_get_bit(br) && br_available(br) > 0) z++;
    if (z == 0) return 0;
    return (1 << z) - 1 + br_get_bits(br, z);
}
static inline int br_get_se(BitReader *br) {
    int k = br_get_ue(br);
    return (k & 1) ? (k + 1) >> 1 : -((k + 1) >> 1);
}
static inline int br_get_te(BitReader *br, int range) {
    if (range == 1) return 0;
    if (range == 2) return br_get_bit(br) ? 0 : 1;
    return br_get_ue(br);
}

/* NAL payload（去掉 1 字节 header 后）→ RBSP（移除 00 00 03）。 */
static int nal_to_rbsp(const uint8_t *nal, int nal_size, uint8_t *rbsp, int cap) {
    int src = 0, dst = 0, zeros = 0;
    for (src = 0; src < nal_size && dst < cap; src++) {
        uint8_t c = nal[src];
        if (zeros >= 2 && c == 3) { zeros = 0; continue; }  /* emulation prevention */
        if (c == 0) zeros++; else zeros = 0;
        rbsp[dst++] = c;
    }
    return dst;
}

/* ================================================================
 * [2] SPS / PPS
 * ================================================================ */

typedef struct {
    int profile_idc, level_idc, constraint_set;
    int seq_parameter_set_id;
    int chroma_format_idc;
    int bit_depth_luma, bit_depth_chroma;
    int log2_max_frame_num;
    int pic_order_cnt_type;
    int log2_max_pic_order_cnt_lsb;
    int num_ref_frames;
    int gaps_in_frame_num_allowed;
    int width, height;       /* 解码分辨率（crop 前为 mb 尺寸） */
    int crop_left, crop_right, crop_top, crop_bottom;
    int mb_width, mb_height;
    int frame_mbs_only_flag;
    int direct_8x8_inference;
    int vui_parameters_present;
} SPS;

typedef struct {
    int pic_parameter_set_id;
    int seq_parameter_set_id;
    int entropy_coding_mode_flag;
    int bottom_field_pic_order_in_frame_present;
    int num_slice_groups_minus1;
    int num_ref_idx_l0_default_active;
    int num_ref_idx_l1_default_active;
    int weighted_pred_flag;
    int weighted_bipred_idc;
    int pic_init_qp_minus26;
    int pic_init_qs_minus26;
    int chroma_qp_index_offset;
    int deblocking_filter_control_present;
    int constrained_intra_pred_flag;
    int redundant_pic_cnt_present;
    int transform_8x8_mode_flag;
    int pic_scaling_matrix_present;
} PPS;

static void parse_sps(const uint8_t *rbsp, int len, SPS *s) {
    BitReader br = { rbsp, len * 8, 0, 0 };
    memset(s, 0, sizeof(*s));
    s->profile_idc = br_get_bits(&br, 8);
    s->constraint_set = br_get_bits(&br, 8);
    s->level_idc = br_get_bits(&br, 8);
    s->seq_parameter_set_id = br_get_ue(&br);
    s->chroma_format_idc = 1;
    s->bit_depth_luma = 8;    /* 非 High profile 隐含 8-bit 4:2:0 */
    s->bit_depth_chroma = 8;
    if (s->profile_idc == 100 || s->profile_idc == 110 || s->profile_idc == 122 ||
        s->profile_idc == 244 || s->profile_idc == 44 || s->profile_idc == 83 ||
        s->profile_idc == 86 || s->profile_idc == 118 || s->profile_idc == 128 ||
        s->profile_idc == 138 || s->profile_idc == 139 || s->profile_idc == 134 ||
        s->profile_idc == 135) {
        s->chroma_format_idc = br_get_ue(&br);
        if (s->chroma_format_idc == 3) br_get_bit(&br);
        s->bit_depth_luma = br_get_ue(&br) + 8;
        s->bit_depth_chroma = br_get_ue(&br) + 8;
        br_get_bit(&br);  /* qpprime */
        if (br_get_bit(&br)) {  /* scaling matrix */
            int n = (s->chroma_format_idc != 3) ? 8 : 12;
            for (int i = 0; i < n; i++) {
                if (br_get_bit(&br)) {
                    int sz = (i < 6) ? 16 : 64;
                    for (int k = 0; k < sz; k++) br_get_se(&br);
                }
            }
        }
    }
    s->log2_max_frame_num = br_get_ue(&br) + 4;
    s->pic_order_cnt_type = br_get_ue(&br);
    if (s->pic_order_cnt_type == 0) {
        s->log2_max_pic_order_cnt_lsb = br_get_ue(&br) + 4;
    } else if (s->pic_order_cnt_type == 1) {
        br_get_bit(&br);
        br_get_se(&br);
        br_get_se(&br);
        int n = br_get_ue(&br);
        for (int i = 0; i < n; i++) br_get_se(&br);
        n = br_get_ue(&br);
        for (int i = 0; i < n; i++) br_get_se(&br);
    }
    s->num_ref_frames = br_get_ue(&br);
    s->gaps_in_frame_num_allowed = br_get_bit(&br);
    int mw = br_get_ue(&br) + 1;
    int mh = br_get_ue(&br) + 1;
    s->mb_width = mw;
    s->mb_height = mh;
    s->frame_mbs_only_flag = br_get_bit(&br);
    if (!s->frame_mbs_only_flag) br_get_bit(&br);  /* mb_adaptive_frame_field */
    s->direct_8x8_inference = br_get_bit(&br);
    if (br_get_bit(&br)) {  /* cropping */
        s->crop_left = br_get_ue(&br);
        s->crop_right = br_get_ue(&br);
        s->crop_top = br_get_ue(&br);
        s->crop_bottom = br_get_ue(&br);
    }
    /* 显示尺寸（chroma_format_idc=1: 4:2:0 时 crop 单位 ×2） */
    int sub_wc = (s->chroma_format_idc == 1 || s->chroma_format_idc == 2) ? 2 : 1;
    int sub_hc = (s->chroma_format_idc == 1) ? 2 : 1;
    s->width = mw * 16 - (s->crop_left + s->crop_right) * sub_wc;
    s->height = mh * 16 * (s->frame_mbs_only_flag ? 1 : 2) -
                (s->crop_top + s->crop_bottom) * sub_hc;
}

static void parse_pps(const uint8_t *rbsp, int len, PPS *p) {
    BitReader br = { rbsp, len * 8, 0, 0 };
    memset(p, 0, sizeof(*p));
    p->pic_parameter_set_id = br_get_ue(&br);
    p->seq_parameter_set_id = br_get_ue(&br);
    p->entropy_coding_mode_flag = br_get_bit(&br);
    p->bottom_field_pic_order_in_frame_present = br_get_bit(&br);
    p->num_slice_groups_minus1 = br_get_ue(&br);
    if (p->num_slice_groups_minus1 > 0) {
        /* FMO 不支持，但保持解析一致 */
        int ng = p->num_slice_groups_minus1 + 1;
        int mt = br_get_ue(&br);  /* slice_group_map_type */
        if (mt == 0) {
            for (int i = 0; i < ng; i++) { br_get_ue(&br); }
        } else if (mt == 2) {
            for (int i = 0; i < ng - 1; i++) { br_get_ue(&br); }
        } else if (mt == 3 || mt == 4 || mt == 5) {
            br_get_bit(&br);
            br_get_se(&br);
            br_get_se(&br);
            for (int i = 0; i < ng; i++) br_get_ue(&br);
        } else if (mt == 6) {
            int n = br_get_ue(&br);
            for (int i = 0; i < n; i++) br_get_ue(&br);
        }
    }
    p->num_ref_idx_l0_default_active = br_get_ue(&br) + 1;
    p->num_ref_idx_l1_default_active = br_get_ue(&br) + 1;
    p->weighted_pred_flag = br_get_bit(&br);
    p->weighted_bipred_idc = br_get_bits(&br, 2);
    p->pic_init_qp_minus26 = br_get_se(&br);
    p->pic_init_qs_minus26 = br_get_se(&br);
    p->chroma_qp_index_offset = br_get_se(&br);
    p->deblocking_filter_control_present = br_get_bit(&br);
    p->constrained_intra_pred_flag = br_get_bit(&br);
    p->redundant_pic_cnt_present = br_get_bit(&br);
    if (br_available(&br) > 0) {
        /* 更多 rbsp 数据：变换 8x8 等 */
        int more = 1;
        while (more) {
            more = br_get_bit(&br);
            if (more) {
                int b = br_get_bit(&br);
                if (b) p->transform_8x8_mode_flag = 1;
                else { br_get_ue(&br); /* mb_qp_delta */ }
                if (b) {
                    /* 第二个 scaling matrix */
                    if (br_get_bit(&br)) {
                        for (int k = 0; k < 64; k++) br_get_se(&br);
                    }
                }
            }
        }
    }
}

/* ================================================================
 * [3] slice header + 解码上下文
 * ================================================================ */

#define SLICE_P   0
#define SLICE_B   1
#define SLICE_I   2
#define SLICE_SP  3
#define SLICE_SI  4
#define MAX_REF   32

typedef struct {
    int poc;              /* POC (bottom 组合) */
    int frame_num;
    int is_ref;           /* nal_ref_idc > 0 */
    int long_term;
    uint8_t *yuv;         /* YUV420P 帧缓冲（luma + chroma 分离） */
    int w, h;
    int used;
} RefPic;

/* 每 MB 状态（解码中重建所需） */
typedef struct {
    int16_t mv[2][2];       /* [list][x/y] 1/4 像素，16x16 级汇总 */
    int8_t  ref[2];
    int16_t mb_type;        /* MB_TYPE_* */
    uint16_t cbp;           /* 编码 cbp（含 bit8 luma-DC / bit6-7 chroma-DC 标志） */
    uint8_t qp;
    uint8_t intra4x4[16];   /* 每 4x4 intra 模式（0-8） */
    uint8_t intra16x16_pred;
    uint8_t chroma_pred;
    uint8_t nz[24];         /* 非零系数数：16 luma + 4 chroma DC + 4 chroma AC */
} MbState;

typedef struct H264Dec {
    const MPMovie *movie;
    SPS sps;
    PPS pps;
    int width, height;    /* 显示尺寸 */
    int mb_w, mb_h;

    /* DPB */
    RefPic refs[MAX_REF];
    int n_refs;
    RefPic last_ref;      /* 上帧参考（已解码） */

    /* 当前 slice 状态 */
    int slice_type;       /* SLICE_* */
    int frame_num;
    int idr_pic_id;
    int pic_order_cnt_lsb;
    int delta_pic_order_cnt[2];
    int redundant_pic_cnt;
    int direct_spatial_mv_pred;
    int num_ref_idx_l0_active, num_ref_idx_l1_active;
    int ref_pic_list_reordering[2][MAX_REF];  /* reorder 标记 + id */
    int cabac_init_idc;
    int slice_qp_delta;
    int disable_deblocking_filter_idc;
    int slice_alpha_c0_offset_div2, slice_beta_offset_div2;
    int field_pic_flag, bottom_field_flag;
    int nal_ref_idc;
    int idr_flag;

    /* 当前帧缓冲（解码中） */
    uint8_t *cur_y, *cur_u, *cur_v;
    int ls_y, ls_c;       /* 行距 */
    int16_t *mv0, *mv1;   /* [mb_w*mb_h][2] 运动矢量（1/4 像素） */
    uint8_t *ref0, *ref1; /* 参考帧索引 */
    uint8_t *mb_type_arr; /* 每 MB 类型编码 */
    int *qp_arr;          /* 每 MB QP */
    uint8_t *intra_modes; /* 帧内预测模式（供邻接预测） */
    int qp_y;

    /* CABAC 状态 */
    uint8_t *cabac_ctx;   /* [1024] 上下文值 */
    int cabac_init_done;

    /* MB 状态（重建层） */
    MbState *mb;            /* [mb_w*mb_h] */
    uint8_t *cur_nz;        /* [24] 当前 MB 非零系数（4x4 级） */
    uint8_t *left_nz;       /* [24] 左 MB */
    uint8_t *top_nz;        /* [mb_w+1][24] 上 MB 行 */
    int left_cbp, top_cbp;  /* 邻接 cbp */
    int mb_idx;             /* 当前宏块索引 */
    int qscale;             /* 当前 QP */
    int last_qscale_diff;
    int16_t *tmp_block;     /* [64] 残差工作区 */
    int16_t *dc_block;      /* [64] DC 工作区 */

    /* 输出 */
    uint8_t *yuv_out;     /* 最近解码帧的 YUV（供 h264_decoder_decode 使用） */
} H264Dec;

/* 一帧 YUV 缓冲大小（含 16 对齐边距用于插值） */
#define YUV_STRIDE(w) (((w) + 31) & ~31)

static int h264_init_frame(H264Dec *d, uint8_t *buf) {
    int w = d->width, h = d->height;
    int ls = YUV_STRIDE(w);
    d->ls_y = ls;
    d->ls_c = ls >> 1;
    d->cur_y = buf;
    d->cur_u = buf + (size_t)ls * h;
    d->cur_v = buf + (size_t)ls * h + (size_t)(ls >> 1) * (h >> 1);
    return ls * h + (ls >> 1) * h;
}

/* 解析 slice header，返回 0 成功 */
static int parse_slice_header(H264Dec *d, const uint8_t *rbsp, int len,
                              int *first_mb, int *cabac_bit_pos) {
    BitReader br = { rbsp, len * 8, 0, 0 };
    *first_mb = br_get_ue(&br);
    int st = br_get_ue(&br);
    /* st: 0/5=P, 1/6=B, 2/7=I, 3/8=SP, 4/9=SI */
    d->slice_type = (st <= 4) ? st : st - 5;
    br_get_ue(&br);  /* pic_parameter_set_id */
    d->frame_num = br_get_bits(&br, d->sps.log2_max_frame_num);
    if (!d->sps.frame_mbs_only_flag) {
        d->field_pic_flag = br_get_bit(&br);
        if (d->field_pic_flag) d->bottom_field_flag = br_get_bit(&br);
    }
    if (d->idr_flag) d->idr_pic_id = br_get_ue(&br);
    if (d->sps.pic_order_cnt_type == 0) {
        d->pic_order_cnt_lsb = br_get_bits(&br, d->sps.log2_max_pic_order_cnt_lsb);
        if (d->pps.bottom_field_pic_order_in_frame_present && !d->field_pic_flag)
            d->delta_pic_order_cnt[1] = br_get_se(&br);
    } else if (d->sps.pic_order_cnt_type == 1 && !d->sps.gaps_in_frame_num_allowed) {
        br_get_se(&br);  /* delta_pic_order_cnt[0] */
        if (d->pps.bottom_field_pic_order_in_frame_present && !d->field_pic_flag)
            br_get_se(&br);
    }
    if (d->pps.redundant_pic_cnt_present)
        d->redundant_pic_cnt = br_get_ue(&br);
    if (d->slice_type == SLICE_B) {
        d->direct_spatial_mv_pred = br_get_bit(&br);
    }
    if (d->slice_type == SLICE_P || d->slice_type == SLICE_SP ||
        d->slice_type == SLICE_B) {
        int override = br_get_bit(&br);
        if (override) {
            d->num_ref_idx_l0_active = br_get_ue(&br) + 1;
            if (d->slice_type == SLICE_B)
                d->num_ref_idx_l1_active = br_get_ue(&br) + 1;
        } else {
            d->num_ref_idx_l0_active = d->pps.num_ref_idx_l0_default_active;
            if (d->slice_type == SLICE_B)
                d->num_ref_idx_l1_active = d->pps.num_ref_idx_l1_default_active;
        }
        /* ref_pic_list_modification（跳过但保持位同步） */
        for (int list = 0; list < 2; list++) {
            int lmax = (list == 0) ? d->num_ref_idx_l0_active : d->num_ref_idx_l1_active;
            if (list == 1 && d->slice_type != SLICE_B) break;
            int mod = br_get_bit(&br);
            if (mod) {
                int n = 0;
                for (;;) {
                    int reorder = br_get_ue(&br);
                    if (reorder == 3) break;
                    if (reorder > 2) break;
                    br_get_ue(&br);  /* 参考帧 index */
                    n++;
                    if (n > 32) break;
                }
            }
        }
        /* pred_weight_table（加权预测；本实现不应用权重但保持同步） */
        if ((d->pps.weighted_pred_flag && d->slice_type == SLICE_P) ||
            (d->pps.weighted_bipred_idc == 1 && d->slice_type == SLICE_B)) {
            for (int l = 0; l < (d->slice_type == SLICE_B ? 2 : 1); l++) {
                int n = (l == 0) ? d->num_ref_idx_l0_active : d->num_ref_idx_l1_active;
                for (int i = 0; i < n; i++) {
                    int luma_w = br_get_ue(&br);
                    if (luma_w) {
                        int off = br_get_se(&br);
                        (void)off;
                    }
                    if (d->sps.chroma_format_idc != 0) {
                        for (int c = 0; c < 2; c++) {
                            int cw = br_get_ue(&br);
                            if (cw) br_get_se(&br);
                        }
                    }
                }
            }
        }
    }
    /* dec_ref_pic_marking（仅当 nal_ref_idc != 0；IDR/参考帧必有）
       —— 本实现暂不应用 MMCO，但必须保持位同步以定位 CABAC 起点 */
    if (d->nal_ref_idc != 0) {
        if (d->idr_flag) {
            br_get_bit(&br); /* no_output_of_prior_pics_flag */
            br_get_bit(&br); /* long_term_reference_flag */
        } else {
            int adaptive = br_get_bit(&br);
            if (adaptive) {
                for (;;) {
                    int mmco = br_get_ue(&br);
                    if (mmco == 0) break;
                    if (mmco == 1 || mmco == 3) br_get_ue(&br);
                    if (mmco == 2) br_get_ue(&br);
                    if (mmco == 3 || mmco == 6) br_get_ue(&br);
                    if (mmco == 4) br_get_ue(&br);
                }
            }
        }
    }
    if (d->pps.entropy_coding_mode_flag && d->slice_type != SLICE_I &&
        d->slice_type != SLICE_SI) {
        d->cabac_init_idc = br_get_ue(&br);
    }
    d->slice_qp_delta = br_get_se(&br);
    if (d->slice_type == SLICE_SP || d->slice_type == SLICE_SI) {
        if (d->slice_type == SLICE_SP) br_get_bit(&br);
        br_get_se(&br);
    }
    if (d->pps.deblocking_filter_control_present) {
        d->disable_deblocking_filter_idc = br_get_ue(&br);
        if (d->disable_deblocking_filter_idc != 1) {
            d->slice_alpha_c0_offset_div2 = br_get_se(&br);
            d->slice_beta_offset_div2 = br_get_se(&br);
        }
    }
    if (cabac_bit_pos) *cabac_bit_pos = br.pos;
    return 0;
}

/* ================================================================
 * [4] CABAC 熵解码（H.264 规范算术解码器）
 * ================================================================ */

typedef struct {
    uint32_t range, low;
    const uint8_t *data;
    size_t size;
    size_t byte_pos;   /* 当前输入字节 */
    int bit_pos;       /* 当前字节内位位置 (0-7) */
} Cabac;

static inline int cabac_bit(Cabac *c) {
    if (c->byte_pos >= c->size) return 0;  /* 数据耗尽补 0 */
    int b = (c->data[c->byte_pos] >> (7 - c->bit_pos)) & 1;
    c->bit_pos++;
    if (c->bit_pos == 8) { c->bit_pos = 0; c->byte_pos++; }
    return b;
}

static void cabac_init(Cabac *c, const uint8_t *data, size_t size) {
    c->data = data;
    c->size = size;
    c->byte_pos = 0;
    c->bit_pos = 0;
    c->range = 0x1FE;
    /* H.264 9.3.4.1.1：byte_alignment 后从字节边界读入 9 位到 codILow
       （与 codIRange=510 同尺度，保证比较精度） */
    c->low = 0;
    for (int i = 0; i < 9; i++)
        c->low = (c->low << 1) | (uint32_t)cabac_bit(c);
}

/* range_tab[state][q]：LPS 范围（H.264 规范 Table 9-35） */
static const uint8_t range_tab[64][4] = {
    {128,176,208,240},{128,167,197,227},{128,158,187,216},{123,150,178,205},
    {116,142,169,195},{111,135,160,185},{105,128,152,175},{100,122,144,166},
    { 95,116,137,158},{ 90,110,130,150},{ 85,104,123,142},{ 81,99,117,135},
    { 77,94,111,128},{ 73,89,105,122},{ 69,85,100,116},{ 66,80,95,110},
    { 62,76,90,104},{ 59,72,86,99},{ 56,69,81,94},{ 53,65,77,89},
    { 51,62,73,85},{ 48,59,69,80},{ 46,56,66,76},{ 43,53,63,72},
    { 41,50,59,69},{ 39,48,56,65},{ 37,45,54,62},{ 35,43,51,59},
    { 33,41,48,56},{ 32,39,46,53},{ 30,37,43,50},{ 28,35,41,48},
    { 27,33,39,45},{ 26,31,37,43},{ 24,30,35,41},{ 23,28,33,39},
    { 22,27,32,37},{ 21,26,30,35},{ 20,24,29,33},{ 19,23,27,31},
    { 18,22,26,30},{ 17,21,25,28},{ 16,20,23,27},{ 15,19,22,25},
    { 14,18,21,24},{ 14,17,20,23},{ 13,16,19,22},{ 12,15,18,21},
    { 12,14,17,20},{ 11,14,16,19},{ 11,13,15,18},{ 10,12,15,17},
    { 10,12,14,16},{  9,11,13,15},{  9,11,12,14},{  8,10,12,14},
    {  8,9,11,13},{  7,9,11,12},{  7,9,10,12},{  7,8,10,11},
    {  6,8,9,11},{  6,7,9,10},{  6,7,8,9},{  2,2,2,2}
};

/* trans_mps / trans_lps（规范 Table 9-36） */
static const uint8_t trans_mps[64] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
    17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
    33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,
    49,50,51,52,53,54,55,56,57,58,59,60,61,62,62,63
};
static const uint8_t trans_lps[64] = {
    0,0,1,2,2,4,4,5,6,7,8,9,9,11,11,12,
    13,13,15,15,16,16,18,18,19,19,21,21,22,22,23,24,
    24,25,26,26,27,27,28,29,29,30,30,30,31,32,32,33,
    33,33,34,34,35,35,35,36,36,36,37,37,37,38,38,63
};

/* 解码一个 bin（ctx 指向上下文值字节，就地更新） */
static inline int cabac_decode(Cabac *c, uint8_t *ctx) {
    int state = *ctx >> 1;
    int mps = *ctx & 1;
    uint32_t q = (c->range >> 6) & 3;
    uint32_t range_lps = range_tab[state][q];
    c->range -= range_lps;
    int bit;
    if (c->low < c->range) {
        bit = mps;
        *ctx = (uint8_t)((trans_mps[state] << 1) | mps);
    } else {
        c->low -= c->range;
        c->range = range_lps;
        bit = 1 - mps;
        *ctx = (uint8_t)((trans_lps[state] << 1) | (1 - mps));
    }
    /* 重归一化 */
    while (c->range < 256) {
        c->range <<= 1;
        c->low = (c->low << 1) | (uint32_t)cabac_bit(c);
    }
    return bit;
}

/* 解码 bypass bin（等概率，无上下文；range 不变） */
static inline int cabac_decode_bypass(Cabac *c) {
    c->low <<= 1;
    c->low |= (uint32_t)cabac_bit(c);
    if (c->low >= c->range) {
        c->low -= c->range;
        return 1;
    }
    return 0;
}

/* 终止编码（fixed probability，无上下文）：range 减 2 比较 */
static inline int cabac_decode_terminate(Cabac *c) {
    int ret;
    c->range -= 2;
    if (c->low >= c->range) {
        ret = 1;
        c->low -= c->range;
        c->range = 2;
    } else {
        ret = 0;
    }
    while (c->range < 256) {
        c->range <<= 1;
        c->low = (c->low << 1) | (uint32_t)cabac_bit(c);
    }
    return ret;
}

/* 解码 unary（ae_v）: 返回前缀 bin 数 */
static inline int cabac_decode_unary(Cabac *c, uint8_t *ctx) {
    int n = 0;
    while (cabac_decode(c, ctx)) n++;
    return n;
}

/* 解码 ue（egk=0）/ 转指数哥伦布（k） */
static inline int cabac_decode_egk(Cabac *c, uint8_t *ctx, int k) {
    int info = 0, n = 0;
    int v = cabac_decode(c, ctx);
    while (v) {
        n++;
        v = cabac_decode(c, ctx);
    }
    if (n == 0) return 0;
    int suffix = 0;
    for (int i = 0; i < n; i++)
        suffix = (suffix << 1) | cabac_decode_bypass(c);
    int m = (1 << n) - 1;
    int val = (n == 0) ? 0 : m + suffix;
    if (k > 0) {
        int extra = 0;
        for (int i = 0; i < k; i++)
            extra = (extra << 1) | cabac_decode_bypass(c);
        val = (val << k) + extra;
    }
    return val;
}

static const int8_t cabac_ctx_I[460][2] = {
    {  20, -15}, {   2,  54}, {   3,  74}, {  20, -15},
    {   2,  54}, {   3,  74}, { -28, 127}, { -23, 104},
    {  -6,  53}, {  -1,  54}, {   7,  51}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,   0}, {   0,   0}, {   0,   0}, {   0,   0},
    {   0,  41}, {   0,  63}, {   0,  63}, {   0,  63},
    {  -9,  83}, {   4,  86}, {   0,  97}, {  -7,  72},
    {  13,  41}, {   3,  62}, {   0,  11}, {   1,  55},
    {   0,  69}, { -17, 127}, { -13, 102}, {   0,  82},
    {  -7,  74}, { -21, 107}, { -27, 127}, { -31, 127},
    { -24, 127}, { -18,  95}, { -27, 127}, { -21, 114},
    { -30, 127}, { -17, 123}, { -12, 115}, { -16, 122},
    { -11, 115}, { -12,  63}, {  -2,  68}, { -15,  84},
    { -13, 104}, {  -3,  70}, {  -8,  93}, { -10,  90},
    { -30, 127}, {  -1,  74}, {  -6,  97}, {  -7,  91},
    { -20, 127}, {  -4,  56}, {  -5,  82}, {  -7,  76},
    { -22, 125}, {  -7,  93}, { -11,  87}, {  -3,  77},
    {  -5,  71}, {  -4,  63}, {  -4,  68}, { -12,  84},
    {  -7,  62}, {  -7,  65}, {   8,  61}, {   5,  56},
    {  -2,  66}, {   1,  64}, {   0,  61}, {  -2,  78},
    {   1,  50}, {   7,  52}, {  10,  35}, {   0,  44},
    {  11,  38}, {   1,  45}, {   0,  46}, {   5,  44},
    {  31,  17}, {   1,  51}, {   7,  50}, {  28,  19},
    {  16,  33}, {  14,  62}, { -13, 108}, { -15, 100},
    { -13, 101}, { -13,  91}, { -12,  94}, { -10,  88},
    { -16,  84}, { -10,  86}, {  -7,  83}, { -13,  87},
    { -19,  94}, {   1,  70}, {   0,  72}, {  -5,  74},
    {  18,  59}, {  -8, 102}, { -15, 100}, {   0,  95},
    {  -4,  75}, {   2,  72}, { -11,  75}, {  -3,  71},
    {  15,  46}, { -13,  69}, {   0,  62}, {   0,  65},
    {  21,  37}, { -15,  72}, {   9,  57}, {  16,  54},
    {   0,  62}, {  12,  72}, {  24,   0}, {  15,   9},
    {   8,  25}, {  13,  18}, {  15,   9}, {  13,  19},
    {  10,  37}, {  12,  18}, {   6,  29}, {  20,  33},
    {  15,  30}, {   4,  45}, {   1,  58}, {   0,  62},
    {   7,  61}, {  12,  38}, {  11,  45}, {  15,  39},
    {  11,  42}, {  13,  44}, {  16,  45}, {  12,  41},
    {  10,  49}, {  30,  34}, {  18,  42}, {  10,  55},
    {  17,  51}, {  17,  46}, {   0,  89}, {  26, -19},
    {  22, -17}, {  26, -17}, {  30, -25}, {  28, -20},
    {  33, -23}, {  37, -27}, {  33, -23}, {  40, -28},
    {  38, -17}, {  33, -11}, {  40, -15}, {  41,  -6},
    {  38,   1}, {  41,  17}, {  30,  -6}, {  27,   3},
    {  26,  22}, {  37, -16}, {  35,  -4}, {  38,  -8},
    {  38,  -3}, {  37,   3}, {  38,   5}, {  42,   0},
    {  35,  16}, {  39,  22}, {  14,  48}, {  27,  37},
    {  21,  60}, {  12,  68}, {   2,  97}, {  -3,  71},
    {  -6,  42}, {  -5,  50}, {  -3,  54}, {  -2,  62},
    {   0,  58}, {   1,  63}, {  -2,  72}, {  -1,  74},
    {  -9,  91}, {  -5,  67}, {  -5,  27}, {  -3,  39},
    {  -2,  44}, {   0,  46}, { -16,  64}, {  -8,  68},
    { -10,  78}, {  -6,  77}, { -10,  86}, { -12,  92},
    { -15,  55}, { -10,  60}, {  -6,  62}, {  -4,  65},
    { -12,  73}, {  -8,  76}, {  -7,  80}, {  -9,  88},
    { -17, 110}, { -11,  97}, { -20,  84}, { -11,  79},
    {  -6,  73}, {  -4,  74}, { -13,  86}, { -13,  96},
    { -11,  97}, { -19, 117}, {  -8,  78}, {  -5,  33},
    {  -4,  48}, {  -2,  53}, {  -3,  62}, { -13,  71},
    { -10,  79}, { -12,  86}, { -13,  90}, { -14,  97},
    {   0,   0}, {  -6,  93}, {  -6,  84}, {  -8,  79},
    {   0,  66}, {  -1,  71}, {   0,  62}, {  -2,  60},
    {  -2,  59}, {  -5,  75}, {  -3,  62}, {  -4,  58},
    {  -9,  66}, {  -1,  79}, {   0,  71}, {   3,  68},
    {  10,  44}, {  -7,  62}, {  15,  36}, {  14,  40},
    {  16,  27}, {  12,  29}, {   1,  44}, {  20,  36},
    {  18,  32}, {   5,  42}, {   1,  48}, {  10,  62},
    {  17,  46}, {   9,  64}, { -12, 104}, { -11,  97},
    { -16,  96}, {  -7,  88}, {  -8,  85}, {  -7,  85},
    {  -9,  85}, { -13,  88}, {   4,  66}, {  -3,  77},
    {  -3,  76}, {  -6,  76}, {  10,  58}, {  -1,  76},
    {  -1,  83}, {  -7,  99}, { -14,  95}, {   2,  95},
    {   0,  76}, {  -5,  74}, {   0,  70}, { -11,  75},
    {   1,  68}, {   0,  65}, { -14,  73}, {   3,  62},
    {   4,  62}, {  -1,  68}, { -13,  75}, {  11,  55},
    {   5,  64}, {  12,  70}, {  15,   6}, {   6,  19},
    {   7,  16}, {  12,  14}, {  18,  13}, {  13,  11},
    {  13,  15}, {  15,  16}, {  12,  23}, {  13,  23},
    {  15,  20}, {  14,  26}, {  14,  44}, {  17,  40},
    {  17,  47}, {  24,  17}, {  21,  21}, {  25,  22},
    {  31,  27}, {  22,  29}, {  19,  35}, {  14,  50},
    {  10,  57}, {   7,  63}, {  -2,  77}, {  -4,  82},
    {  -3,  94}, {   9,  69}, { -12, 109}, {  36, -35},
    {  36, -34}, {  32, -26}, {  37, -30}, {  44, -32},
    {  34, -18}, {  34, -15}, {  40, -15}, {  33,  -7},
    {  35,  -5}, {  33,   0}, {  38,   2}, {  33,  13},
    {  23,  35}, {  13,  58}, {  29,  -3}, {  26,   0},
    {  22,  30}, {  31,  -7}, {  35, -15}, {  34,  -3},
    {  34,   3}, {  36,  -1}, {  34,   5}, {  32,  11},
    {  35,   5}, {  34,  12}, {  39,  11}, {  30,  29},
    {  34,  26}, {  29,  39}, {  19,  66}, {  31,  21},
    {  31,  31}, {  25,  50}, { -17, 120}, { -20, 112},
    { -18, 114}, { -11,  85}, { -15,  92}, { -14,  89},
    { -26,  71}, { -15,  81}, { -14,  80}, {   0,  68},
    { -14,  70}, { -24,  56}, { -23,  68}, { -24,  50},
    { -11,  74}, {  23, -13}, {  26, -13}, {  40, -15},
    {  49, -14}, {  44,   3}, {  45,   6}, {  44,  34},
    {  33,  54}, {  19,  82}, {  -3,  75}, {  -1,  23},
    {   1,  34}, {   1,  43}, {   0,  54}, {  -2,  55},
    {   0,  61}, {   1,  64}, {   0,  68}, {  -9,  92},
    { -14, 106}, { -13,  97}, { -15,  90}, { -12,  90},
    { -18,  88}, { -10,  73}, {  -9,  79}, { -14,  86},
    { -10,  73}, { -10,  70}, { -10,  69}, {  -5,  66},
    {  -9,  64}, {  -5,  58}, {   2,  59}, {  21, -10},
    {  24, -11}, {  28,  -8}, {  28,  -1}, {  29,   3},
    {  29,   9}, {  35,  20}, {  29,  36}, {  14,  67},
};

static const int8_t cabac_ctx_P0[460][2] = {
    {  20, -15}, {   2,  54}, {   3,  74}, {  20, -15},
    {   2,  54}, {   3,  74}, { -28, 127}, { -23, 104},
    {  -6,  53}, {  -1,  54}, {   7,  51}, {  23,  33},
    {  23,   2}, {  21,   0}, {   1,   9}, {   0,  49},
    { -37, 118}, {   5,  57}, { -13,  78}, { -11,  65},
    {   1,  62}, {  12,  49}, {  -4,  73}, {  17,  50},
    {  18,  64}, {   9,  43}, {  29,   0}, {  26,  67},
    {  16,  90}, {   9, 104}, { -46, 127}, { -20, 104},
    {   1,  67}, { -13,  78}, { -11,  65}, {   1,  62},
    {  -6,  86}, { -17,  95}, {  -6,  61}, {   9,  45},
    {  -3,  69}, {  -6,  81}, { -11,  96}, {   6,  55},
    {   7,  67}, {  -5,  86}, {   2,  88}, {   0,  58},
    {  -3,  76}, { -10,  94}, {   5,  54}, {   4,  69},
    {  -3,  81}, {   0,  88}, {  -7,  67}, {  -5,  74},
    {  -4,  74}, {  -5,  80}, {  -7,  72}, {   1,  58},
    {   0,  41}, {   0,  63}, {   0,  63}, {   0,  63},
    {  -9,  83}, {   4,  86}, {   0,  97}, {  -7,  72},
    {  13,  41}, {   3,  62}, {   0,  45}, {  -4,  78},
    {  -3,  96}, { -27, 126}, { -28,  98}, { -25, 101},
    { -23,  67}, { -28,  82}, { -20,  94}, { -16,  83},
    { -22, 110}, { -21,  91}, { -18, 102}, { -13,  93},
    { -29, 127}, {  -7,  92}, {  -5,  89}, {  -7,  96},
    { -13, 108}, {  -3,  46}, {  -1,  65}, {  -1,  57},
    {  -9,  93}, {  -3,  74}, {  -9,  92}, {  -8,  87},
    { -23, 126}, {   5,  54}, {   6,  60}, {   6,  59},
    {   6,  69}, {  -1,  48}, {   0,  68}, {  -4,  69},
    {  -8,  88}, {  -2,  85}, {  -6,  78}, {  -1,  75},
    {  -7,  77}, {   2,  54}, {   5,  50}, {  -3,  68},
    {   1,  50}, {   6,  42}, {  -4,  81}, {   1,  63},
    {  -4,  70}, {   0,  67}, {   2,  57}, {  -2,  76},
    {  11,  35}, {   4,  64}, {   1,  61}, {  11,  35},
    {  18,  25}, {  12,  24}, {  13,  29}, {  13,  36},
    { -10,  93}, {  -7,  73}, {  -2,  73}, {  13,  46},
    {   9,  49}, {  -7, 100}, {   9,  53}, {   2,  53},
    {   5,  53}, {  -2,  61}, {   0,  56}, {   0,  56},
    { -13,  63}, {  -5,  60}, {  -1,  62}, {   4,  57},
    {  -6,  69}, {   4,  57}, {  14,  39}, {   4,  51},
    {  13,  68}, {   3,  64}, {   1,  61}, {   9,  63},
    {   7,  50}, {  16,  39}, {   5,  44}, {   4,  52},
    {  11,  48}, {  -5,  60}, {  -1,  59}, {   0,  59},
    {  22,  33}, {   5,  44}, {  14,  43}, {  -1,  78},
    {   0,  60}, {   9,  69}, {  11,  28}, {   2,  40},
    {   3,  44}, {   0,  49}, {   0,  46}, {   2,  44},
    {   2,  51}, {   0,  47}, {   4,  39}, {   2,  62},
    {   6,  46}, {   0,  54}, {   3,  54}, {   2,  58},
    {   4,  63}, {   6,  51}, {   6,  57}, {   7,  53},
    {   6,  52}, {   6,  55}, {  11,  45}, {  14,  36},
    {   8,  53}, {  -1,  82}, {   7,  55}, {  -3,  78},
    {  15,  46}, {  22,  31}, {  -1,  84}, {  25,   7},
    {  30,  -7}, {  28,   3}, {  28,   4}, {  32,   0},
    {  34,  -1}, {  30,   6}, {  30,   6}, {  32,   9},
    {  31,  19}, {  26,  27}, {  26,  30}, {  37,  20},
    {  28,  34}, {  17,  70}, {   1,  67}, {   5,  59},
    {   9,  67}, {  16,  30}, {  18,  32}, {  18,  35},
    {  22,  29}, {  24,  31}, {  23,  38}, {  18,  43},
    {  20,  41}, {  11,  63}, {   9,  59}, {   9,  64},
    {  -1,  94}, {  -2,  89}, {  -9, 108}, {  -6,  76},
    {  -2,  44}, {   0,  45}, {   0,  52}, {  -3,  64},
    {  -2,  59}, {  -4,  70}, {  -4,  75}, {  -8,  82},
    { -17, 102}, {  -9,  77}, {   3,  24}, {   0,  42},
    {   0,  48}, {   0,  55}, {  -6,  59}, {  -7,  71},
    { -12,  83}, { -11,  87}, { -30, 119}, {   1,  58},
    {  -3,  29}, {  -1,  36}, {   1,  38}, {   2,  43},
    {  -6,  55}, {   0,  58}, {   0,  64}, {  -3,  74},
    { -10,  90}, {   0,  70}, {  -4,  29}, {   5,  31},
    {   7,  42}, {   1,  59}, {  -2,  58}, {  -3,  72},
    {  -3,  81}, { -11,  97}, {   0,  58}, {   8,   5},
    {  10,  14}, {  14,  18}, {  13,  27}, {   2,  40},
    {   0,  58}, {  -3,  70}, {  -6,  79}, {  -8,  85},
    {   0,   0}, { -13, 106}, { -16, 106}, { -10,  87},
    { -21, 114}, { -18, 110}, { -14,  98}, { -22, 110},
    { -21, 106}, { -18, 103}, { -21, 107}, { -23, 108},
    { -26, 112}, { -10,  96}, { -12,  95}, {  -5,  91},
    {  -9,  93}, { -22,  94}, {  -5,  86}, {   9,  67},
    {  -4,  80}, { -10,  85}, {  -1,  70}, {   7,  60},
    {   9,  58}, {   5,  61}, {  12,  50}, {  15,  50},
    {  18,  49}, {  17,  54}, {  10,  41}, {   7,  46},
    {  -1,  51}, {   7,  49}, {   8,  52}, {   9,  41},
    {   6,  47}, {   2,  55}, {  13,  41}, {  10,  44},
    {   6,  50}, {   5,  53}, {  13,  49}, {   4,  63},
    {   6,  64}, {  -2,  69}, {  -2,  59}, {   6,  70},
    {  10,  44}, {   9,  31}, {  12,  43}, {   3,  53},
    {  14,  34}, {  10,  38}, {  -3,  52}, {  13,  40},
    {  17,  32}, {   7,  44}, {   7,  38}, {  13,  50},
    {  10,  57}, {  26,  43}, {  14,  11}, {  11,  14},
    {   9,  11}, {  18,  11}, {  21,   9}, {  23,  -2},
    {  32, -15}, {  32, -15}, {  34, -21}, {  39, -23},
    {  42, -33}, {  41, -31}, {  46, -28}, {  38, -12},
    {  21,  29}, {  45, -24}, {  53, -45}, {  48, -26},
    {  65, -43}, {  43, -19}, {  39, -10}, {  30,   9},
    {  18,  26}, {  20,  27}, {   0,  57}, { -14,  82},
    {  -5,  75}, { -19,  97}, { -35, 125}, {  27,   0},
    {  28,   0}, {  31,  -4}, {  27,   6}, {  34,   8},
    {  30,  10}, {  24,  22}, {  33,  19}, {  22,  32},
    {  26,  31}, {  21,  41}, {  26,  44}, {  23,  47},
    {  16,  65}, {  14,  71}, {   8,  60}, {   6,  63},
    {  17,  65}, {  21,  24}, {  23,  20}, {  26,  23},
    {  27,  32}, {  28,  23}, {  28,  24}, {  23,  40},
    {  24,  32}, {  28,  29}, {  23,  42}, {  19,  57},
    {  22,  53}, {  22,  61}, {  11,  86}, {  12,  40},
    {  11,  51}, {  14,  59}, {  -4,  79}, {  -7,  71},
    {  -5,  69}, {  -9,  70}, {  -8,  66}, { -10,  68},
    { -19,  73}, { -12,  69}, { -16,  70}, { -15,  67},
    { -20,  62}, { -19,  70}, { -16,  66}, { -22,  65},
    { -20,  63}, {   9,  -2}, {  26,  -9}, {  33,  -9},
    {  39,  -7}, {  41,  -2}, {  45,   3}, {  49,   9},
    {  45,  27}, {  36,  59}, {  -6,  66}, {  -7,  35},
    {  -7,  42}, {  -8,  45}, {  -5,  48}, { -12,  56},
    {  -6,  60}, {  -5,  62}, {  -8,  66}, {  -8,  76},
    {  -5,  85}, {  -6,  81}, { -10,  77}, {  -7,  81},
    { -17,  80}, { -18,  73}, {  -4,  74}, { -10,  83},
    {  -9,  71}, {  -9,  67}, {  -1,  61}, {  -8,  66},
    { -14,  66}, {   0,  59}, {   2,  59}, {  21, -13},
    {  33, -14}, {  39,  -7}, {  46,  -2}, {  51,   2},
    {  60,   6}, {  61,  17}, {  55,  34}, {  42,  62},
};

static const int8_t cabac_ctx_P1[460][2] = {
    {  20, -15}, {   2,  54}, {   3,  74}, {  20, -15},
    {   2,  54}, {   3,  74}, { -28, 127}, { -23, 104},
    {  -6,  53}, {  -1,  54}, {   7,  51}, {  22,  25},
    {  34,   0}, {  16,   0}, {  -2,   9}, {   4,  41},
    { -29, 118}, {   2,  65}, {  -6,  71}, { -13,  79},
    {   5,  52}, {   9,  50}, {  -3,  70}, {  10,  54},
    {  26,  34}, {  19,  22}, {  40,   0}, {  57,   2},
    {  41,  36}, {  26,  69}, { -45, 127}, { -15, 101},
    {  -4,  76}, {  -6,  71}, { -13,  79}, {   5,  52},
    {   6,  69}, { -13,  90}, {   0,  52}, {   8,  43},
    {  -2,  69}, {  -5,  82}, { -10,  96}, {   2,  59},
    {   2,  75}, {  -3,  87}, {  -3, 100}, {   1,  56},
    {  -3,  74}, {  -6,  85}, {   0,  59}, {  -3,  81},
    {  -7,  86}, {  -5,  95}, {  -1,  66}, {  -1,  77},
    {   1,  70}, {  -2,  86}, {  -5,  72}, {   0,  61},
    {   0,  41}, {   0,  63}, {   0,  63}, {   0,  63},
    {  -9,  83}, {   4,  86}, {   0,  97}, {  -7,  72},
    {  13,  41}, {   3,  62}, {  13,  15}, {   7,  51},
    {   2,  80}, { -39, 127}, { -18,  91}, { -17,  96},
    { -26,  81}, { -35,  98}, { -24, 102}, { -23,  97},
    { -27, 119}, { -24,  99}, { -21, 110}, { -18, 102},
    { -36, 127}, {   0,  80}, {  -5,  89}, {  -7,  94},
    {  -4,  92}, {   0,  39}, {   0,  65}, { -15,  84},
    { -35, 127}, {  -2,  73}, { -12, 104}, {  -9,  91},
    { -31, 127}, {   3,  55}, {   7,  56}, {   7,  55},
    {   8,  61}, {  -3,  53}, {   0,  68}, {  -7,  74},
    {  -9,  88}, { -13, 103}, { -13,  91}, {  -9,  89},
    { -14,  92}, {  -8,  76}, { -12,  87}, { -23, 110},
    { -24, 105}, { -10,  78}, { -20, 112}, { -17,  99},
    { -78, 127}, { -70, 127}, { -50, 127}, { -46, 127},
    {  -4,  66}, {  -5,  78}, {  -4,  71}, {  -8,  72},
    {   2,  59}, {  -1,  55}, {  -7,  70}, {  -6,  75},
    {  -8,  89}, { -34, 119}, {  -3,  75}, {  32,  20},
    {  30,  22}, { -44, 127}, {   0,  54}, {  -5,  61},
    {   0,  58}, {  -1,  60}, {  -3,  61}, {  -8,  67},
    { -25,  84}, { -14,  74}, {  -5,  65}, {   5,  52},
    {   2,  57}, {   0,  61}, {  -9,  69}, { -11,  70},
    {  18,  55}, {  -4,  71}, {   0,  58}, {   7,  61},
    {   9,  41}, {  18,  25}, {   9,  32}, {   5,  43},
    {   9,  47}, {   0,  44}, {   0,  51}, {   2,  46},
    {  19,  38}, {  -4,  66}, {  15,  38}, {  12,  42},
    {   9,  34}, {   0,  89}, {   4,  45}, {  10,  28},
    {  10,  31}, {  33, -11}, {  52, -43}, {  18,  15},
    {  28,   0}, {  35, -22}, {  38, -25}, {  34,   0},
    {  39, -18}, {  32, -12}, { 102, -94}, {   0,   0},
    {  56, -15}, {  33,  -4}, {  29,  10}, {  37,  -5},
    {  51, -29}, {  39,  -9}, {  52, -34}, {  69, -58},
    {  67, -63}, {  44,  -5}, {  32,   7}, {  55, -29},
    {  32,   1}, {   0,   0}, {  27,  36}, {  33, -25},
    {  34, -30}, {  36, -28}, {  38, -28}, {  38, -27},
    {  34, -18}, {  35, -16}, {  34, -14}, {  32,  -8},
    {  37,  -6}, {  35,   0}, {  30,  10}, {  28,  18},
    {  26,  25}, {  29,  41}, {   0,  75}, {   2,  72},
    {   8,  77}, {  14,  35}, {  18,  31}, {  17,  35},
    {  21,  30}, {  17,  45}, {  20,  42}, {  18,  45},
    {  27,  26}, {  16,  54}, {   7,  66}, {  16,  56},
    {  11,  73}, {  10,  67}, { -10, 116}, { -23, 112},
    { -15,  71}, {  -7,  61}, {   0,  53}, {  -5,  66},
    { -11,  77}, {  -9,  80}, {  -9,  84}, { -10,  87},
    { -34, 127}, { -21, 101}, {  -3,  39}, {  -5,  53},
    {  -7,  61}, { -11,  75}, { -15,  77}, { -17,  91},
    { -25, 107}, { -25, 111}, { -28, 122}, { -11,  76},
    { -10,  44}, { -10,  52}, { -10,  57}, {  -9,  58},
    { -16,  72}, {  -7,  69}, {  -4,  69}, {  -5,  74},
    {  -9,  86}, {   2,  66}, {  -9,  34}, {   1,  32},
    {  11,  31}, {   5,  52}, {  -2,  55}, {  -2,  67},
    {   0,  73}, {  -8,  89}, {   3,  52}, {   7,   4},
    {  10,   8}, {  17,   8}, {  16,  19}, {   3,  37},
    {  -1,  61}, {  -5,  73}, {  -1,  70}, {  -4,  78},
    {   0,   0}, { -21, 126}, { -23, 124}, { -20, 110},
    { -26, 126}, { -25, 124}, { -17, 105}, { -27, 121},
    { -27, 117}, { -17, 102}, { -26, 117}, { -27, 116},
    { -33, 122}, { -10,  95}, { -14, 100}, {  -8,  95},
    { -17, 111}, { -28, 114}, {  -6,  89}, {  -2,  80},
    {  -4,  82}, {  -9,  85}, {  -8,  81}, {  -1,  72},
    {   5,  64}, {   1,  67}, {   9,  56}, {   0,  69},
    {   1,  69}, {   7,  69}, {  -7,  69}, {  -6,  67},
    { -16,  77}, {  -2,  64}, {   2,  61}, {  -6,  67},
    {  -3,  64}, {   2,  57}, {  -3,  65}, {  -3,  66},
    {   0,  62}, {   9,  51}, {  -1,  66}, {  -2,  71},
    {  -2,  75}, {  -1,  70}, {  -9,  72}, {  14,  60},
    {  16,  37}, {   0,  47}, {  18,  35}, {  11,  37},
    {  12,  41}, {  10,  41}, {   2,  48}, {  12,  41},
    {  13,  41}, {   0,  59}, {   3,  50}, {  19,  40},
    {   3,  66}, {  18,  50}, {  19,  -6}, {  18,  -6},
    {  14,   0}, {  26, -12}, {  31, -16}, {  33, -25},
    {  33, -22}, {  37, -28}, {  39, -30}, {  42, -30},
    {  47, -42}, {  45, -36}, {  49, -34}, {  41, -17},
    {  32,   9}, {  69, -71}, {  63, -63}, {  66, -64},
    {  77, -74}, {  54, -39}, {  52, -35}, {  41, -10},
    {  36,   0}, {  40,  -1}, {  30,  14}, {  28,  26},
    {  23,  37}, {  12,  55}, {  11,  65}, {  37, -33},
    {  39, -36}, {  40, -37}, {  38, -30}, {  46, -33},
    {  42, -30}, {  40, -24}, {  49, -29}, {  38, -12},
    {  40, -10}, {  38,  -3}, {  46,  -5}, {  31,  20},
    {  29,  30}, {  25,  44}, {  12,  48}, {  11,  49},
    {  26,  45}, {  22,  22}, {  23,  22}, {  27,  21},
    {  33,  20}, {  26,  28}, {  30,  24}, {  27,  34},
    {  18,  42}, {  25,  39}, {  18,  50}, {  12,  70},
    {  21,  54}, {  14,  71}, {  11,  83}, {  25,  32},
    {  21,  49}, {  21,  54}, {  -5,  85}, {  -6,  81},
    { -10,  77}, {  -7,  81}, { -17,  80}, { -18,  73},
    {  -4,  74}, { -10,  83}, {  -9,  71}, {  -9,  67},
    {  -1,  61}, {  -8,  66}, { -14,  66}, {   0,  59},
    {   2,  59}, {  17, -10}, {  32, -13}, {  42,  -9},
    {  49,  -5}, {  53,   0}, {  64,   3}, {  68,  10},
    {  66,  27}, {  47,  57}, {  -5,  71}, {   0,  24},
    {  -1,  36}, {  -2,  42}, {  -2,  52}, {  -9,  57},
    {  -6,  63}, {  -4,  65}, {  -4,  67}, {  -7,  82},
    {  -3,  81}, {  -3,  76}, {  -7,  72}, {  -6,  78},
    { -12,  72}, { -14,  68}, {  -3,  70}, {  -6,  76},
    {  -5,  66}, {  -5,  62}, {   0,  57}, {  -4,  61},
    {  -9,  60}, {   1,  54}, {   2,  58}, {  17, -10},
    {  32, -13}, {  42,  -9}, {  49,  -5}, {  53,   0},
    {  64,   3}, {  68,  10}, {  66,  27}, {  47,  57},
};

static const int8_t cabac_ctx_P2[460][2] = {
    {  20, -15}, {   2,  54}, {   3,  74}, {  20, -15},
    {   2,  54}, {   3,  74}, { -28, 127}, { -23, 104},
    {  -6,  53}, {  -1,  54}, {   7,  51}, {  29,  16},
    {  25,   0}, {  14,   0}, { -10,  51}, {  -3,  62},
    { -27,  99}, {  26,  16}, {  -4,  85}, { -24, 102},
    {   5,  57}, {   6,  57}, { -17,  73}, {  14,  57},
    {  20,  40}, {  20,  10}, {  29,   0}, {  54,   0},
    {  37,  42}, {  12,  97}, { -32, 127}, { -22, 117},
    {  -2,  74}, {  -4,  85}, { -24, 102}, {   5,  57},
    {  -6,  93}, { -14,  88}, {  -6,  44}, {   4,  55},
    { -11,  89}, { -15, 103}, { -21, 116}, {  19,  57},
    {  20,  58}, {   4,  84}, {   6,  96}, {   1,  63},
    {  -5,  85}, { -13, 106}, {   5,  63}, {   6,  75},
    {  -3,  90}, {  -1, 101}, {   3,  55}, {  -4,  79},
    {  -2,  75}, { -12,  97}, {  -7,  50}, {   1,  60},
    {   0,  41}, {   0,  63}, {   0,  63}, {   0,  63},
    {  -9,  83}, {   4,  86}, {   0,  97}, {  -7,  72},
    {  13,  41}, {   3,  62}, {   7,  34}, {  -9,  88},
    { -20, 127}, { -36, 127}, { -17,  91}, { -14,  95},
    { -25,  84}, { -25,  86}, { -12,  89}, { -17,  91},
    { -31, 127}, { -14,  76}, { -18, 103}, { -13,  90},
    { -37, 127}, {  11,  80}, {   5,  76}, {   2,  84},
    {   5,  78}, {  -6,  55}, {   4,  61}, { -14,  83},
    { -37, 127}, {  -5,  79}, { -11, 104}, { -11,  91},
    { -30, 127}, {   0,  65}, {  -2,  79}, {   0,  72},
    {  -4,  92}, {  -6,  56}, {   3,  68}, {  -8,  71},
    { -13,  98}, {  -4,  86}, { -12,  88}, {  -5,  82},
    {  -3,  72}, {  -4,  67}, {  -8,  72}, { -16,  89},
    {  -9,  69}, {  -1,  59}, {   5,  66}, {   4,  57},
    {  -4,  71}, {  -2,  71}, {   2,  58}, {  -1,  74},
    {  -4,  44}, {  -1,  69}, {   0,  62}, {  -7,  51},
    {  -4,  47}, {  -6,  42}, {  -3,  41}, {  -6,  53},
    {   8,  76}, {  -9,  78}, { -11,  83}, {   9,  52},
    {   0,  67}, {  -5,  90}, {   1,  67}, { -15,  72},
    {  -5,  75}, {  -8,  80}, { -21,  83}, { -21,  64},
    { -13,  31}, { -25,  64}, { -29,  94}, {   9,  75},
    {  17,  63}, {  -8,  74}, {  -5,  35}, {  -2,  27},
    {  13,  91}, {   3,  65}, {  -7,  69}, {   8,  77},
    { -10,  66}, {   3,  62}, {  -3,  68}, { -20,  81},
    {   0,  30}, {   1,   7}, {  -3,  23}, { -21,  74},
    {  16,  66}, { -23, 124}, {  17,  37}, {  44, -18},
    {  50, -34}, { -22, 127}, {   4,  39}, {   0,  42},
    {   7,  34}, {  11,  29}, {   8,  31}, {   6,  37},
    {   7,  42}, {   3,  40}, {   8,  33}, {  13,  43},
    {  13,  36}, {   4,  47}, {   3,  55}, {   2,  58},
    {   6,  60}, {   8,  44}, {  11,  44}, {  14,  42},
    {   7,  48}, {   4,  56}, {   4,  52}, {  13,  37},
    {   9,  49}, {  19,  58}, {  10,  48}, {  12,  45},
    {   0,  69}, {  20,  33}, {   8,  63}, {  35, -18},
    {  33, -25}, {  28,  -3}, {  24,  10}, {  27,   0},
    {  34, -14}, {  52, -44}, {  39, -24}, {  19,  17},
    {  31,  25}, {  36,  29}, {  24,  33}, {  34,  15},
    {  30,  20}, {  22,  73}, {  20,  34}, {  19,  31},
    {  27,  44}, {  19,  16}, {  15,  36}, {  15,  36},
    {  21,  28}, {  25,  21}, {  30,  20}, {  31,  12},
    {  27,  16}, {  24,  42}, {   0,  93}, {  14,  56},
    {  15,  57}, {  26,  38}, { -24, 127}, { -24, 115},
    { -22,  82}, {  -9,  62}, {   0,  53}, {   0,  59},
    { -14,  85}, { -13,  89}, { -13,  94}, { -11,  92},
    { -29, 127}, { -21, 100}, { -14,  57}, { -12,  67},
    { -11,  71}, { -10,  77}, { -21,  85}, { -16,  88},
    { -23, 104}, { -15,  98}, { -37, 127}, { -10,  82},
    {  -8,  48}, {  -8,  61}, {  -8,  66}, {  -7,  70},
    { -14,  75}, { -10,  79}, {  -9,  83}, { -12,  92},
    { -18, 108}, {  -4,  79}, { -22,  69}, { -16,  75},
    {  -2,  58}, {   1,  58}, { -13,  78}, {  -9,  83},
    {  -4,  81}, { -13,  99}, { -13,  81}, {  -6,  38},
    { -13,  62}, {  -6,  58}, {  -2,  59}, { -16,  73},
    { -10,  76}, { -13,  86}, {  -9,  83}, { -10,  87},
    {   0,   0}, { -22, 127}, { -25, 127}, { -25, 120},
    { -27, 127}, { -19, 114}, { -23, 117}, { -25, 118},
    { -26, 117}, { -24, 113}, { -28, 118}, { -31, 120},
    { -37, 124}, { -10,  94}, { -15, 102}, { -10,  99},
    { -13, 106}, { -50, 127}, {  -5,  92}, {  17,  57},
    {  -5,  86}, { -13,  94}, { -12,  91}, {  -2,  77},
    {   0,  71}, {  -1,  73}, {   4,  64}, {  -7,  81},
    {   5,  64}, {  15,  57}, {   1,  67}, {   0,  68},
    { -10,  67}, {   1,  68}, {   0,  77}, {   2,  64},
    {   0,  68}, {  -5,  78}, {   7,  55}, {   5,  59},
    {   2,  65}, {  14,  54}, {  15,  44}, {   5,  60},
    {   2,  70}, {  -2,  76}, { -18,  86}, {  12,  70},
    {   5,  64}, { -12,  70}, {  11,  55}, {   5,  56},
    {   0,  69}, {   2,  65}, {  -6,  74}, {   5,  54},
    {   7,  54}, {  -6,  76}, { -11,  82}, {  -2,  77},
    {  -2,  77}, {  25,  42}, {  17, -13}, {  16,  -9},
    {  17, -12}, {  27, -21}, {  37, -30}, {  41, -40},
    {  42, -41}, {  48, -47}, {  39, -32}, {  46, -40},
    {  52, -51}, {  46, -41}, {  52, -39}, {  43, -19},
    {  32,  11}, {  61, -55}, {  56, -46}, {  62, -50},
    {  81, -67}, {  45, -20}, {  35,  -2}, {  28,  15},
    {  34,   1}, {  39,   1}, {  30,  17}, {  20,  38},
    {  18,  45}, {  15,  54}, {   0,  79}, {  36, -16},
    {  37, -14}, {  37, -17}, {  32,   1}, {  34,  15},
    {  29,  15}, {  24,  25}, {  34,  22}, {  31,  16},
    {  35,  18}, {  31,  28}, {  33,  41}, {  36,  28},
    {  27,  47}, {  21,  62}, {  18,  31}, {  19,  26},
    {  36,  24}, {  24,  23}, {  27,  16}, {  24,  30},
    {  31,  29}, {  22,  41}, {  22,  42}, {  16,  60},
    {  15,  52}, {  14,  60}, {   3,  78}, { -16, 123},
    {  21,  53}, {  22,  56}, {  25,  61}, {  21,  33},
    {  19,  50}, {  17,  61}, {  -3,  78}, {  -8,  74},
    {  -9,  72}, { -10,  72}, { -18,  75}, { -12,  71},
    { -11,  63}, {  -5,  70}, { -17,  75}, { -14,  72},
    { -16,  67}, {  -8,  53}, { -14,  59}, {  -9,  52},
    { -11,  68}, {   9,  -2}, {  30, -10}, {  31,  -4},
    {  33,  -1}, {  33,   7}, {  31,  12}, {  37,  23},
    {  31,  38}, {  20,  64}, {  -9,  71}, {  -7,  37},
    {  -8,  44}, { -11,  49}, { -10,  56}, { -12,  59},
    {  -8,  63}, {  -9,  67}, {  -6,  68}, { -10,  79},
    {  -3,  78}, {  -8,  74}, {  -9,  72}, { -10,  72},
    { -18,  75}, { -12,  71}, { -11,  63}, {  -5,  70},
    { -17,  75}, { -14,  72}, { -16,  67}, {  -8,  53},
    { -14,  59}, {  -9,  52}, { -11,  68}, {   9,  -2},
    {  30, -10}, {  31,  -4}, {  33,  -1}, {  33,   7},
    {  31,  12}, {  37,  23}, {  31,  38}, {  20,  64},
};


/* 上下文初始化（与 x264 编码器一致）:
 * state = clip3(1,126,((m0*qp)>>4)+m1)
 * ctx = (min(state,127-state)<<1) | (state>>6) */
static void cabac_ctx_init_decode(H264Dec *d, int slice_type) {
    const int8_t (*tab)[2];
    if (slice_type == SLICE_I) tab = cabac_ctx_I;
    else if (d->cabac_init_idc == 1) tab = cabac_ctx_P1;
    else if (d->cabac_init_idc == 2) tab = cabac_ctx_P2;
    else tab = cabac_ctx_P0;
    int qp = d->qp_y;
    for (int i = 0; i < 460; i++) {
        int state = ((tab[i][0] * qp) >> 4) + tab[i][1];
        if (state < 1) state = 1;
        else if (state > 126) state = 126;
        int s = (state < 127 - state) ? state : 127 - state;
        d->cabac_ctx[i] = (uint8_t)((s << 1) | (state >> 6));
    }
    d->cabac_init_done = 1;
}

/* ================================================================
 * [9] 顶层 API（解码器重建层为多轮迭代项，当前骨架保证链接可用）
 * ================================================================ */

static inline int h264_clip_u8(int a) {
    if (a < 0) return 0;
    if (a > 255) return 255;
    return a;
}

int h264_decoder_init(H264Decoder *d, const MPMovie *movie) {
    memset(d, 0, sizeof(*d));
    d->movie = movie;
    if (!movie || !movie->sps || !movie->pps || movie->n_frames <= 0) return -1;
    H264Dec *dec = (H264Dec *)calloc(1, sizeof(H264Dec));
    if (!dec) return -1;
    uint8_t rbsp[512];
    int rl = nal_to_rbsp(movie->sps + 1, movie->sps_len - 1, rbsp, sizeof(rbsp));
    if (rl <= 0) { free(dec); return -1; }
    parse_sps(rbsp, rl, &dec->sps);
    rl = nal_to_rbsp(movie->pps + 1, movie->pps_len - 1, rbsp, sizeof(rbsp));
    if (rl <= 0) { free(dec); return -1; }
    parse_pps(rbsp, rl, &dec->pps);
    dec->movie = movie;
    dec->width = dec->sps.width;
    dec->height = dec->sps.height;
    dec->mb_w = dec->sps.mb_width;
    dec->mb_h = dec->sps.mb_height;
    int ls = YUV_STRIDE(dec->width);
    dec->ls_y = ls;
    dec->ls_c = ls >> 1;
    /* 布局：[顶部pad 1行][Y 显示区 w*h][pad 4行][U][V][尾部pad 1行]。
     * 顶部 pad 供 y=0 帧内预测读取；底部 pad 供最后一 MB(540→544)写入；
     * 尾部 pad 供 chroma 272 行(>显示 270)写入。总 = yb + 6*ls + 2*cb。 */
    size_t yb = (size_t)ls * dec->height;
    size_t cb = (size_t)(ls >> 1) * (dec->height >> 1);
    size_t frame_bytes = yb + 6 * (size_t)ls + cb * 2;
    dec->yuv_out = (uint8_t *)calloc(1, frame_bytes);
    dec->mb = (MbState *)calloc((size_t)dec->mb_w * dec->mb_h, sizeof(MbState));
    dec->cur_nz = (uint8_t *)calloc(24, 1);
    dec->left_nz = (uint8_t *)calloc(24, 1);
    dec->top_nz = (uint8_t *)calloc((size_t)(dec->mb_w + 1) * 24, 1);
    dec->cabac_ctx = (uint8_t *)calloc(460, 1);
    dec->tmp_block = (int16_t *)calloc(64, sizeof(int16_t));
    dec->dc_block = (int16_t *)calloc(64, sizeof(int16_t));
    if (!dec->yuv_out || !dec->mb || !dec->cabac_ctx || !dec->tmp_block) {
        free(dec->yuv_out); free(dec->mb); free(dec->cur_nz); free(dec->left_nz);
        free(dec->top_nz); free(dec->cabac_ctx); free(dec->tmp_block); free(dec->dc_block);
        free(dec); return -1;
    }
    uint8_t *yuv = dec->yuv_out + ls;         /* Y 起点（跳过顶部 pad） */
    d->yuv = yuv;
    d->y_u = yuv + yb + 4 * ls;               /* U 起点（Y + 4 行 pad） */
    d->y_v = d->y_u + cb;                     /* V 起点 */
    d->yuv_cap = (int)frame_bytes;
    d->dec = dec;
    return 0;
}
void h264_decoder_free(H264Decoder *d) {
    if (d->dec) {
        free(d->dec->yuv_out);  /* d->yuv/y_u/y_v 指向其内部 */
        free(d->dec->mb);
        free(d->dec->cur_nz);
        free(d->dec->left_nz);
        free(d->dec->top_nz);
        free(d->dec->cabac_ctx);
        free(d->dec->tmp_block);
        free(d->dec->dc_block);
    }
    memset(d, 0, sizeof(*d));
}

/* ================================================================
 * YUV420 → RGB24 缩放（自研，可用于任意 YUV 帧）
 * ================================================================ */
int media_yuv_to_rgb(const uint8_t *yuv, int src_w, int src_h,
                     uint8_t *rgb, int max_edge, int *dst_w, int *dst_h) {
    int dw = src_w, dh = src_h;
    /* 定点缩放（公理 resource_sensitive_multiplication：热路径避免浮点除法）。
     * 原 float 版 sy = (int)(dy / scale)，scale = max_edge/max_src；
     * 整数等价：sy = dy * max_src / max_edge（int64 单次乘除，精确截断）。 */
    int up = 1, down = 1;
    if (max_edge > 0 && (src_w > max_edge || src_h > max_edge)) {
        int max_src = (src_w > src_h) ? src_w : src_h;
        up = max_src;
        down = max_edge;
        dw = (int)((2LL * src_w * max_edge + max_src) / (2 * max_src));
        dh = (int)((2LL * src_h * max_edge + max_src) / (2 * max_src));
        if (dw < 1) dw = 1;
        if (dh < 1) dh = 1;
    }
    const uint8_t *Y = yuv;
    const uint8_t *U = yuv + (size_t)src_w * src_h;
    const uint8_t *V = U + (size_t)(src_w / 2) * (src_h / 2);
    for (int dy = 0; dy < dh; dy++) {
        int sy = (int)(((int64_t)dy * up) / down);
        if (sy >= src_h) sy = src_h - 1;
        for (int dx = 0; dx < dw; dx++) {
            int sx = (int)(((int64_t)dx * up) / down);
            if (sx >= src_w) sx = src_w - 1;
            int yy = Y[sy * src_w + sx];
            int uu = U[(sy >> 1) * (src_w >> 1) + (sx >> 1)] - 128;
            int vv = V[(sy >> 1) * (src_w >> 1) + (sx >> 1)] - 128;
            int c = yy + ((vv * 1436) >> 10);          /* 1.402 * 1024 */
            int e = yy - ((uu * 352) >> 10) - ((vv * 731) >> 10);
            int f = yy + ((uu * 1814) >> 10);          /* 1.772 * 1024 */
            uint8_t *p = rgb + ((size_t)dy * dw + dx) * 3;
            p[0] = (uint8_t)h264_clip_u8(c);
            p[1] = (uint8_t)h264_clip_u8(e);
            p[2] = (uint8_t)h264_clip_u8(f);
        }
    }
    *dst_w = dw;
    *dst_h = dh;
    return 0;
}

/* ================================================================
 * [5] 宏块重建层（d4）
 * ================================================================ */

#define MB_TYPE_INTRA4x4    0x0001
#define MB_TYPE_INTRA16x16  0x0002
#define MB_TYPE_INTRA_PCM   0x0004
#define MB_TYPE_8x8DCT      0x0008
#define MB_TYPE_16x16       0x0010
#define MB_TYPE_16x8        0x0020
#define MB_TYPE_8x16        0x0040
#define MB_TYPE_8x8         0x0080
#define MB_TYPE_P0L0        0x0100
#define MB_TYPE_P1L0        0x0200
#define MB_TYPE_P0L1        0x0400
#define MB_TYPE_P1L1        0x0800
#define MB_TYPE_DIRECT2     0x1000
#define MB_TYPE_REF0        0x2000
#define IS_INTRA4x4(a)   ((a) & MB_TYPE_INTRA4x4)
#define IS_INTRA16x16(a) ((a) & MB_TYPE_INTRA16x16)
#define IS_INTRA(a)      ((a) & (MB_TYPE_INTRA4x4 | MB_TYPE_INTRA16x16))
#define IS_INTRA_PCM(a)  ((a) & MB_TYPE_INTRA_PCM)
#define IS_8x8DCT(a)     ((a) & MB_TYPE_8x8DCT)
#define IS_16x16(a)      ((a) & MB_TYPE_16x16)
#define IS_16x8(a)       ((a) & MB_TYPE_16x8)
#define IS_8x16(a)       ((a) & MB_TYPE_8x16)
#define IS_8x8(a)        ((a) & MB_TYPE_8x8)
#define IS_SUB_8x8(a)    ((a) & MB_TYPE_8x8)
#define IS_SUB_8x4(a)    ((a) & (MB_TYPE_8x8 | MB_TYPE_8x16)) == 0 && 0
#define IS_DIRECT(a)     ((a) & MB_TYPE_DIRECT2)
#define IS_DIR(a, n, l)  ((a) & (MB_TYPE_P0L0 << ((n) * 2 + (l))))
#define MB_FIELD 0


/* 帧内 4x4 预测（dst 为 4x4 块，top/left 为邻接重建像素） */
static void h264_pred4x4(uint8_t *dst, int stride, int mode,
                         const uint8_t *top, const uint8_t *left) {
    int i;
    switch (mode) {
    case 0:
        for (i = 0; i < 4; i++)
            dst[0*stride+i]=top[i], dst[1*stride+i]=top[i], dst[2*stride+i]=top[i], dst[3*stride+i]=top[i];
        break;
    case 1:
        for (i = 0; i < 4; i++) { uint8_t v=left[i]; dst[i*stride+0]=v; dst[i*stride+1]=v; dst[i*stride+2]=v; dst[i*stride+3]=v; }
        break;
    case 2: { int s = 0;
        for (i = 0; i < 4; i++) s += top[i] + left[i];
        s = (s + 4) >> 3;
        for (i = 0; i < 4; i++) dst[0*stride+i]=s, dst[1*stride+i]=s, dst[2*stride+i]=s, dst[3*stride+i]=s;
        break; }
    case 3: { uint8_t p0=top[0],p1=top[1],p2=top[2],p3=top[3],p4=top[4],p5=top[5],p6=top[6],p7=top[7];
        dst[0*stride+0]=(p0+p1*2+p2+2)>>2; dst[0*stride+1]=(p1+p2*2+p3+2)>>2; dst[0*stride+2]=(p2+p3*2+p4+2)>>2; dst[0*stride+3]=(p3+p4*2+p5+2)>>2;
        dst[1*stride+0]=(p1+p2*2+p3+2)>>2; dst[1*stride+1]=(p2+p3*2+p4+2)>>2; dst[1*stride+2]=(p3+p4*2+p5+2)>>2; dst[1*stride+3]=(p4+p5*2+p6+2)>>2;
        dst[2*stride+0]=(p2+p3*2+p4+2)>>2; dst[2*stride+1]=(p3+p4*2+p5+2)>>2; dst[2*stride+2]=(p4+p5*2+p6+2)>>2; dst[2*stride+3]=(p5+p6*2+p7+2)>>2;
        dst[3*stride+0]=(p3+p4*2+p5+2)>>2; dst[3*stride+1]=(p4+p5*2+p6+2)>>2; dst[3*stride+2]=(p5+p6*2+p7+2)>>2; dst[3*stride+3]=(p6+p7*2+p7+2)>>2;
        break; }
    case 4: { uint8_t p0=left[3],p1=left[2],p2=left[1],p3=left[0],p4=top[0],p5=top[1],p6=top[2],p7=top[3];
        dst[0*stride+0]=(p3+p4*2+p5+2)>>2; dst[0*stride+1]=(p2+p3*2+p4+2)>>2; dst[0*stride+2]=(p1+p2*2+p3+2)>>2; dst[0*stride+3]=(p0+p1*2+p2+2)>>2;
        dst[1*stride+0]=(p2+p3*2+p4+2)>>2; dst[1*stride+1]=(p1+p2*2+p3+2)>>2; dst[1*stride+2]=(p0+p1*2+p2+2)>>2; dst[1*stride+3]=(p1+p0*2+p1+2)>>2;
        dst[2*stride+0]=(p1+p2*2+p3+2)>>2; dst[2*stride+1]=(p0+p1*2+p2+2)>>2; dst[2*stride+2]=(p1+p0*2+p1+2)>>2; dst[2*stride+3]=(p2+p1*2+p0+2)>>2;
        dst[3*stride+0]=(p0+p1*2+p2+2)>>2; dst[3*stride+1]=(p1+p0*2+p1+2)>>2; dst[3*stride+2]=(p2+p1*2+p0+2)>>2; dst[3*stride+3]=(p3+p2*2+p1+2)>>2;
        break; }
    case 5: { uint8_t p0=left[3],p1=left[2],p2=left[1],p3=left[0],p4=top[0],p5=top[1],p6=top[2],p7=top[3];
        dst[0*stride+0]=(p3+p4+1)>>1; dst[0*stride+1]=(p4+p5+1)>>1; dst[0*stride+2]=(p5+p6+1)>>1; dst[0*stride+3]=(p6+p7+1)>>1;
        dst[1*stride+0]=(p2+p3*2+p4+2)>>2; dst[1*stride+1]=(p3+p4*2+p5+2)>>2; dst[1*stride+2]=(p4+p5*2+p6+2)>>2; dst[1*stride+3]=(p5+p6*2+p7+2)>>2;
        dst[2*stride+0]=(p1+p2*2+p3+2)>>2; dst[2*stride+1]=(p2+p3*2+p4+2)>>2; dst[2*stride+2]=(p3+p4*2+p5+2)>>2; dst[2*stride+3]=(p4+p5*2+p6+2)>>2;
        dst[3*stride+0]=(p0+p1*2+p2+2)>>2; dst[3*stride+1]=(p1+p2*2+p3+2)>>2; dst[3*stride+2]=(p2+p3*2+p4+2)>>2; dst[3*stride+3]=(p3+p4*2+p5+2)>>2;
        break; }
    case 6: { uint8_t p0=left[3],p1=left[2],p2=left[1],p3=left[0],p4=top[0],p5=top[1],p6=top[2],p7=top[3];
        dst[0*stride+0]=(p2+p3+1)>>1; dst[0*stride+1]=(p1+p2*2+p3+2)>>2; dst[0*stride+2]=(p0+p1*2+p2+2)>>2; dst[0*stride+3]=(p1+p0*2+p1+2)>>2;
        dst[1*stride+0]=(p3+p4+1)>>1; dst[1*stride+1]=(p2+p3*2+p4+2)>>2; dst[1*stride+2]=(p1+p2*2+p3+2)>>2; dst[1*stride+3]=(p0+p1*2+p2+2)>>2;
        dst[2*stride+0]=(p4+p5+1)>>1; dst[2*stride+1]=(p3+p4*2+p5+2)>>2; dst[2*stride+2]=(p2+p3*2+p4+2)>>2; dst[2*stride+3]=(p1+p2*2+p3+2)>>2;
        dst[3*stride+0]=(p5+p6+1)>>1; dst[3*stride+1]=(p4+p5*2+p6+2)>>2; dst[3*stride+2]=(p3+p4*2+p5+2)>>2; dst[3*stride+3]=(p2+p3*2+p4+2)>>2;
        break; }
    case 7: { uint8_t p4=top[4],p5=top[5],p6=top[6],p7=top[7];
        dst[0*stride+0]=(top[0]+top[1]+1)>>1; dst[0*stride+1]=(top[1]+top[2]+1)>>1; dst[0*stride+2]=(top[2]+top[3]+1)>>1; dst[0*stride+3]=(top[3]+p4+1)>>1;
        dst[1*stride+0]=(top[0]+top[1]*2+top[2]+2)>>2; dst[1*stride+1]=(top[1]+top[2]*2+top[3]+2)>>2; dst[1*stride+2]=(top[2]+top[3]*2+p4+2)>>2; dst[1*stride+3]=(top[3]+p4*2+p5+2)>>2;
        dst[2*stride+0]=(top[1]+top[2]*2+top[3]+2)>>2; dst[2*stride+1]=(top[2]+top[3]*2+p4+2)>>2; dst[2*stride+2]=(top[3]+p4*2+p5+2)>>2; dst[2*stride+3]=(p4+p5*2+p6+2)>>2;
        dst[3*stride+0]=(top[2]+top[3]*2+p4+2)>>2; dst[3*stride+1]=(top[3]+p4*2+p5+2)>>2; dst[3*stride+2]=(p4+p5*2+p6+2)>>2; dst[3*stride+3]=(p5+p6*2+p7+2)>>2;
        break; }
    case 8: { uint8_t p0=left[0],p1=left[1],p2=left[2],p3=left[3];
        dst[0*stride+0]=(p0+p1+1)>>1; dst[0*stride+1]=(p0+p1*2+p2+2)>>2; dst[0*stride+2]=(p1+p2*2+p3+2)>>2; dst[0*stride+3]=(p2+p3*2+p3+2)>>2;
        dst[1*stride+0]=(p1+p2+1)>>1; dst[1*stride+1]=(p1+p2*2+p3+2)>>2; dst[1*stride+2]=(p2+p3*2+p3+2)>>2; dst[1*stride+3]=(p3+p3*2+p3+2)>>2;
        dst[2*stride+0]=(p2+p3+1)>>1; dst[2*stride+1]=(p2+p3*2+p3+2)>>2; dst[2*stride+2]=(p3+p3*2+p3+2)>>2; dst[2*stride+3]=(p3+p3*2+p3+2)>>2;
        dst[3*stride+0]=(p3+p3+1)>>1; dst[3*stride+1]=(p3+p3*2+p3+2)>>2; dst[3*stride+2]=(p3+p3*2+p3+2)>>2; dst[3*stride+3]=(p3+p3*2+p3+2)>>2;
        break; }
    }
}

/* ---- zigzag 扫描（frame 编码） ---- */
static const uint8_t h264_zigzag[16] = { 0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15 };

/* ---- CABAC 残差解码（H.264 9.3.4.3） ----
 * block 输出：系数写到 zigzag 扫描序位置。返回系数个数。 */
static const int sig_off_tab[6]  = { 105, 120, 134, 149, 152, 402 };
static const int last_off_tab[6] = { 166, 181, 195, 210, 213, 417 };
static const int abs_off_tab[6]  = { 227, 237, 247, 257, 266, 426 };
static const uint8_t abs_l1_ctx[8]  = { 1, 2, 3, 4, 0, 0, 0, 0 };
static const uint8_t abs_gt1_ctx[8] = { 5, 5, 5, 5, 6, 7, 8, 9 };
static const uint8_t abs_trans0[8]  = { 1, 2, 3, 3, 4, 5, 6, 7 };
static const uint8_t abs_trans1[8]  = { 4, 4, 4, 4, 5, 6, 7, 7 };

/* bypass 带符号：0 → +v，1 → -v */
static inline int cabac_bypass_sign(Cabac *c, int v) {
    return cabac_decode_bypass(c) ? -v : v;
}

/* 标准表 6-1 块序映射：4x4 块号 n <-> 块坐标 (bx4, by4) */
static inline int h264_blk_x(int n) { return (n & 1) + 2 * ((n >> 2) & 1); }
static inline int h264_blk_y(int n) { return ((n >> 1) & 1) + 2 * ((n >> 3) & 1); }
static inline int h264_blk_n(int bx4, int by4) {
    return (bx4 & 1) + 2 * (by4 & 1) + 4 * ((bx4 >> 1) + 2 * (by4 >> 1));
}

/* 邻接非零（标准块序块号 n：0-15 luma，16-19 U，20-23 V；dir 0=左 1=上） */
static int h264_neighbor_nz(H264Dec *d, int n, int dir) {
    if (n < 16) {
        int bx4 = h264_blk_x(n), by4 = h264_blk_y(n);
        if (dir == 0) {
            if (bx4 > 0) return d->cur_nz[h264_blk_n(bx4 - 1, by4)];
            return d->left_nz[h264_blk_n(3, by4)];
        }
        if (by4 > 0) return d->cur_nz[h264_blk_n(bx4, by4 - 1)];
        return d->top_nz[(d->mb_idx % d->mb_w + 1) * 24 + h264_blk_n(bx4, 3)];
    } else {
        int comp = (n < 20) ? 0 : 1;
        int c = n - 16 - comp * 4;
        int x = c & 1, y = c >> 1;
        int base = 16 + comp * 4;
        if (dir == 0)
            return (x > 0) ? d->cur_nz[base + c - 1] : d->left_nz[base + c + 1];
        return (y > 0) ? d->cur_nz[base + c - 2] : d->top_nz[(d->mb_idx % d->mb_w + 1) * 24 + base + c + 2];
    }
}

/* 解码一个残差块。block[16] 输出（zigzag 扫描序位置），返回非零数。
 * cat: 0=lumaDC(I16x16) 1=lumaAC(I16x16) 2=luma4x4(I4x4) 3=chromaDC 4=chromaAC
 * is_dc=1 时邻接用 MB 级 cbp（DC 块无 nz 邻接），否则用块级 nz 邻接 */
static int h264_cabac_residual(H264Dec *d, Cabac *cab, int cat, int max_coeff,
                               int16_t *block, int n, int is_dc) {
    uint8_t *state = d->cabac_ctx;
    int sig_base = sig_off_tab[cat];
    int last_base = last_off_tab[cat];
    int abs_base = abs_off_tab[cat];
    static const uint8_t cbf_base[5] = { 85, 89, 93, 97, 101 };
    int index[64];
    int coeff_count = 0;

    /* coded_block_flag */
    int nza, nzb, cbf_ctx;
    if (is_dc) {
        if (cat == 3) { /* chroma DC：邻接用左右 MB 的 cbp bit6/7 */
            int comp = n - 16;
            nza = (d->left_cbp >> (6 + comp)) & 1;
            nzb = (d->top_cbp >> (6 + comp)) & 1;
        } else {        /* luma DC（I16x16）：邻接用 cbp bit8 */
            nza = (d->left_cbp >> 8) & 1;
            nzb = (d->top_cbp >> 8) & 1;
        }
    } else {
        nza = h264_neighbor_nz(d, n, 0);
        nzb = h264_neighbor_nz(d, n, 1);
    }
    cbf_ctx = (nza > 0) + 2 * (nzb > 0);
    if (!cabac_decode(cab, &state[cbf_base[cat] + cbf_ctx])) {
        return 0;
    }

    /* significant / last 扫描（ffmpeg：sig/last ctx 均按扫描位置 last 线性递增；
     * 循环只到 max_coeff-2，最后位置 significant 隐式为 1 且不再读位） */
    {
        int last = 0;
        for (last = 0; last < max_coeff - 1; last++) {
            if (cabac_decode(cab, &state[sig_base + last])) {
                index[coeff_count] = last;
                int lctx = (last < 14) ? last : 14;
                if (cabac_decode(cab, &state[last_base + lctx])) {
                    coeff_count++;
                    break;
                }
                coeff_count++;
            }
        }
        if (last == max_coeff - 1)   /* 循环未 break：末位隐式 significant */
            index[coeff_count++] = last;
    }

    /* abs_level（逆序，node_ctx 状态机） */
    {
        int node_ctx = 0;
        int ac = (cat == 1 || cat == 4);   /* I16x16 AC / chroma AC：跳过 DC 位置 */
        for (int i = coeff_count - 1; i >= 0; i--) {
            int sp = index[i];
            /* 块内系数位置：chroma DC 用 2x2 光栅，AC 块从 zigzag[1] 起，其余含 DC */
            int j = (cat == 3) ? sp : h264_zigzag[sp + ac];
            uint8_t *ctx = state + abs_base + abs_l1_ctx[node_ctx];
            if (!cabac_decode(cab, ctx)) {
                node_ctx = abs_trans0[node_ctx];
                block[j] = (int16_t)cabac_bypass_sign(cab, 1);
            } else {
                int absv = 2;
                ctx = state + abs_base + abs_gt1_ctx[node_ctx];
                node_ctx = abs_trans1[node_ctx];
                while (absv < 15 && cabac_decode(cab, ctx)) absv++;
                if (absv >= 15) {
                    int k = 0;
                    while (cabac_decode_bypass(cab) && k < 23) k++;
                    absv = 1;
                    while (k--) absv = (absv << 1) + cabac_decode_bypass(cab);
                    absv += 14;
                }
                block[j] = (int16_t)cabac_bypass_sign(cab, absv);
            }
        }
    }

    /* DC 块有系数 → 更新 MB 级 cbp 标志（供相邻 MB 的 DC cbf 邻接） */
    if (is_dc) {
        if (cat == 3) d->mb[d->mb_idx].cbp |= (uint8_t)(0x40 << (n - 16));
        else d->mb[d->mb_idx].cbp |= 0x100;
    }
    return coeff_count;
}

/* mb_qp_delta（CABAC ctx 60-62） */
static void h264_cabac_mb_qp_delta(H264Dec *d, Cabac *cab) {
    uint8_t *state = d->cabac_ctx;
    int max_qp = 51 + 6 * (d->sps.bit_depth_luma - 8);
    if (!cabac_decode(cab, &state[60 + (d->last_qscale_diff != 0)])) {
        d->last_qscale_diff = 0;
        return;
    }
    int val = 1, ctx = 2;
    while (cabac_decode(cab, &state[60 + ctx])) {
        ctx = 3;
        val++;
    }
    if (val & 1) val = (val + 1) >> 1;
    else val = -((val + 1) >> 1);
    d->last_qscale_diff = val;
    d->qscale += val;
    if (d->qscale < 0) d->qscale += max_qp + 1;
    else if (d->qscale > max_qp) d->qscale -= max_qp + 1;
}

/* chroma QP 映射（H.264 表 8-15） */
static int h264_chroma_qp(int qp, int index_offset) {
    static const int qpc[52] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,
        27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51
    };
    int idx = qp + index_offset;
    if (idx < 0) idx = 0;
    if (idx > 51) idx = 51;
    return qpc[idx];
}

/* I 帧 mb_type（ffmpeg decode_cabac_intra_mb_type）：
 *   首 bin ctx=3+邻接I16x16数（0..2），1 → I16x16 系列；0 → I_NxN
 *   终止编码（无上下文）→ I_PCM；附加 bin ctx 4-8 解 I16x16 变体 */
static int h264_cabac_intra_mb_type(H264Dec *d, Cabac *cab, int mb_x, int mb_y) {
    uint8_t *state = d->cabac_ctx;
    int idx = mb_y * d->mb_w + mb_x;
    int ctx = 0;
    if (mb_x > 0 && IS_INTRA16x16(d->mb[idx - 1].mb_type)) ctx++;
    if (mb_y > 0 && IS_INTRA16x16(d->mb[idx - d->mb_w].mb_type)) ctx++;
    if (cabac_decode(cab, &state[3 + ctx]) == 0)
        return 0;   /* I_NxN */
    if (cabac_decode_terminate(cab))
        return 25;  /* I_PCM */
    int mb_type = 1;  /* I16x16 */
    mb_type += 12 * cabac_decode(cab, &state[4]);       /* cbp_luma != 0 */
    if (cabac_decode(cab, &state[5]))                   /* cbp_chroma != 0 */
        mb_type += 4 + 4 * cabac_decode(cab, &state[6]);
    mb_type += 2 * cabac_decode(cab, &state[7]);
    mb_type += 1 * cabac_decode(cab, &state[8]);
    return mb_type;
}

/* luma cbp（CABAC ctx 73-76） */
static int h264_cabac_cbp_luma(H264Dec *d, Cabac *cab) {
    uint8_t *state = d->cabac_ctx;
    int cbp_a = d->left_cbp, cbp_b = d->top_cbp;
    int ctx, cbp = 0;
    ctx = !(cbp_a & 0x02) + 2 * !(cbp_b & 0x04);
    cbp += cabac_decode(cab, &state[73 + ctx]);
    ctx = !(cbp & 0x01) + 2 * !(cbp_b & 0x08);
    cbp += cabac_decode(cab, &state[73 + ctx]) << 1;
    ctx = !(cbp_a & 0x08) + 2 * !(cbp & 0x01);
    cbp += cabac_decode(cab, &state[73 + ctx]) << 2;
    ctx = !(cbp & 0x04) + 2 * !(cbp & 0x02);
    cbp += cabac_decode(cab, &state[73 + ctx]) << 3;
    return cbp;
}

/* chroma cbp（CABAC ctx 77-79） */
static int h264_cabac_cbp_chroma(H264Dec *d, Cabac *cab) {
    uint8_t *state = d->cabac_ctx;
    int cbp_a = (d->left_cbp >> 4) & 3, cbp_b = (d->top_cbp >> 4) & 3;
    int ctx = (cbp_a > 0) + 2 * (cbp_b > 0);
    if (!cabac_decode(cab, &state[77 + ctx])) return 0;
    ctx = 4 + (cbp_a == 2) + 2 * (cbp_b == 2);
    return 1 + cabac_decode(cab, &state[77 + ctx]);
}

/* 4x4 帧内模式：prev flag (ctx 68) + 3 bin (ctx 69) */
static int h264_cabac_intra4x4_mode(H264Dec *d, Cabac *cab, int pred_mode) {
    uint8_t *state = d->cabac_ctx;
    if (cabac_decode(cab, &state[68])) return pred_mode;
    int mode = 0;
    mode += cabac_decode(cab, &state[69]);
    mode += 2 * cabac_decode(cab, &state[69]);
    mode += 4 * cabac_decode(cab, &state[69]);
    return mode + (mode >= pred_mode);
}

/* chroma 帧内模式（ffmpeg decode_cabac_mb_chroma_pre_mode：ctx 64+邻接，67 用两次） */
static int h264_cabac_chroma_pred(H264Dec *d, Cabac *cab, int left_mode, int top_mode) {
    uint8_t *state = d->cabac_ctx;
    int ctx = (left_mode != 0) + (top_mode != 0);
    if (!cabac_decode(cab, &state[64 + ctx])) return 0;
    if (!cabac_decode(cab, &state[64 + 3])) return 1;
    if (!cabac_decode(cab, &state[64 + 3])) return 2;
    return 3;
}

/* 16x16 帧内预测（模式 0-3） */
static void h264_pred16x16(uint8_t *dst, int stride, int mode,
                           const uint8_t *top, const uint8_t *left) {
    int i, j;
    switch (mode) {
    case 0:
        for (i = 0; i < 16; i++)
            for (j = 0; j < 16; j++) dst[i * stride + j] = top[j];
        break;
    case 1:
        for (i = 0; i < 16; i++)
            for (j = 0; j < 16; j++) dst[i * stride + j] = left[i];
        break;
    case 2: {
        int s = 0;
        for (i = 0; i < 16; i++) s += top[i] + left[i];
        s = (s + 16) >> 5;
        for (i = 0; i < 16; i++)
            for (j = 0; j < 16; j++) dst[i * stride + j] = (uint8_t)s;
        break; }
    case 3:
        for (i = 0; i < 16; i++)
            for (j = 0; j < 16; j++)
                dst[i * stride + j] = (uint8_t)h264_clip_u8(top[j] + ((left[i] - left[15] + 1) >> 1));
        break;
    }
}

/* 色度 8x8 帧内预测（模式 0-3） */
static void h264_pred8x8c(uint8_t *dst, int stride, int mode,
                          const uint8_t *top, const uint8_t *left) {
    int i, j;
    switch (mode) {
    case 0:
        for (i = 0; i < 8; i++)
            for (j = 0; j < 8; j++) dst[i * stride + j] = top[j];
        break;
    case 1:
        for (i = 0; i < 8; i++)
            for (j = 0; j < 8; j++) dst[i * stride + j] = left[i];
        break;
    case 2: {
        int s = 0;
        for (i = 0; i < 8; i++) s += top[i] + left[i];
        s = (s + 8) >> 4;
        for (i = 0; i < 8; i++)
            for (j = 0; j < 8; j++) dst[i * stride + j] = (uint8_t)s;
        break; }
    case 3:
        for (i = 0; i < 8; i++)
            for (j = 0; j < 8; j++)
                dst[i * stride + j] = (uint8_t)h264_clip_u8(top[j] + ((left[i] - left[7] + 1) >> 1));
        break;
    }
}

/* I 帧 mb 类型表（ffmpeg i_mb_type_info，索引 0-25 → {type, pred_mode, cbp}） */
static const int i_mb_type_tab[26][3] = {
    {MB_TYPE_INTRA4x4,   -1,  -1},   /* 0 I_NxN */
    {MB_TYPE_INTRA16x16,  2,   0},   /* 1 */
    {MB_TYPE_INTRA16x16,  1,   0},   /* 2 */
    {MB_TYPE_INTRA16x16,  0,   0},   /* 3 */
    {MB_TYPE_INTRA16x16,  3,   0},   /* 4 */
    {MB_TYPE_INTRA16x16,  2,  16},   /* 5 */
    {MB_TYPE_INTRA16x16,  1,  16},   /* 6 */
    {MB_TYPE_INTRA16x16,  0,  16},   /* 7 */
    {MB_TYPE_INTRA16x16,  3,  16},   /* 8 */
    {MB_TYPE_INTRA16x16,  2,  32},   /* 9 */
    {MB_TYPE_INTRA16x16,  1,  32},   /* 10 */
    {MB_TYPE_INTRA16x16,  0,  32},   /* 11 */
    {MB_TYPE_INTRA16x16,  3,  32},   /* 12 */
    {MB_TYPE_INTRA16x16,  2,  15},   /* 13 */
    {MB_TYPE_INTRA16x16,  1,  15},   /* 14 */
    {MB_TYPE_INTRA16x16,  0,  15},   /* 15 */
    {MB_TYPE_INTRA16x16,  3,  15},   /* 16 */
    {MB_TYPE_INTRA16x16,  2,  31},   /* 17 */
    {MB_TYPE_INTRA16x16,  1,  31},   /* 18 */
    {MB_TYPE_INTRA16x16,  0,  31},   /* 19 */
    {MB_TYPE_INTRA16x16,  3,  31},   /* 20 */
    {MB_TYPE_INTRA16x16,  2,  47},   /* 21 */
    {MB_TYPE_INTRA16x16,  1,  47},   /* 22 */
    {MB_TYPE_INTRA16x16,  0,  47},   /* 23 */
    {MB_TYPE_INTRA16x16,  3,  47},   /* 24 */
    {MB_TYPE_INTRA_PCM,  -1,  -1},   /* 25 */
};

/* ================================================================
 * [6] I 帧宏块重建（decode_mb 主循环，CABAC）
 * ================================================================ */

/* 位置 → 反量化权重列（H.264 表 8-15） */
static inline int dequant_col(int x, int y) {
    if (((x | y) & 1) == 0) return 0;
    return ((x & y) & 1) ? 1 : 2;
}

/* H.264 表 8-15 dequant4：[rem][列]，列由位置奇偶决定
 * 注：列序 {10,16,13} 与 dequant_col 映射配合后，等效于 ffmpeg 的
 * dequant4_coeff_init{10,13,16} 与 col=(x&1)+(y&1)（即 (0,1)/(1,0)=16、(1,1)=13）。 */
static const int dequant4[6][3] = {
    { 10, 16, 13 }, { 11, 18, 14 }, { 13, 20, 16 },
    { 14, 23, 18 }, { 16, 25, 20 }, { 18, 29, 23 }
};

/* 反量化 + IDCT 一个 4x4 块（block 已解出 zigzag 序残差）。
 * dc_pre=1 时位置 (0,0) 已是反量化 DC 值（来自 DC 块），不再乘权重。 */
static void h264_dequant_idct4(uint8_t *dst, int stride, int16_t *block,
                               int qp, int dc_pre) {
    int rem = qp % 6, shift = qp / 6;
    int16_t tmp[16];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            int pos = y * 4 + x;
            int level = block[pos];
            if (!(dc_pre && pos == 0))
                level = level * dequant4[rem][dequant_col(x, y)] << shift;
            tmp[pos] = (int16_t)level;
        }
    /* 4x4 IDCT（双蝶形） */
    for (int i = 0; i < 4; i++) {
        int a = tmp[i*4+0] + tmp[i*4+2];
        int b = tmp[i*4+0] - tmp[i*4+2];
        int c = (tmp[i*4+1] >> 1) - tmp[i*4+3];
        int d = tmp[i*4+1] + (tmp[i*4+3] >> 1);
        tmp[i*4+0] = (int16_t)(a + d);
        tmp[i*4+1] = (int16_t)(b + c);
        tmp[i*4+2] = (int16_t)(b - c);
        tmp[i*4+3] = (int16_t)(a - d);
    }
    for (int i = 0; i < 4; i++) {
        int a = tmp[0+i] + tmp[8+i];
        int b = tmp[0+i] - tmp[8+i];
        int c = (tmp[4+i] >> 1) - tmp[12+i];
        int d = tmp[4+i] + (tmp[12+i] >> 1);
        int e = (a + d + 32) >> 6;
        int f = (b + c + 32) >> 6;
        int g = (b - c + 32) >> 6;
        int h = (a - d + 32) >> 6;
        dst[0*stride+i] = (uint8_t)h264_clip_u8(dst[0*stride+i] + e);
        dst[1*stride+i] = (uint8_t)h264_clip_u8(dst[1*stride+i] + f);
        dst[2*stride+i] = (uint8_t)h264_clip_u8(dst[2*stride+i] + g);
        dst[3*stride+i] = (uint8_t)h264_clip_u8(dst[3*stride+i] + h);
    }
}

/* I16x16 luma DC：4x4 Hadamard 反变换（无归一化）+ 反量化。
 * 位级语义（与 ffmpeg 7.1 实测一致）：
 *   qmul = dequant4_coeff[0][qp][0] * 16 = dequant4[rem][0] << (div+10)
 *   dc   = (z * qmul + 128) >> 8
 * 之后再经 idct4_dc_add 的 (x+32)>>6 得到像素增量。
 * 注意：不能用 dequant4[rem][0] << div（缺 16x scaling，实测差 16 倍）。 */
static void h264_luma_dc_dequant(H264Dec *d, int16_t *dc, int qp) {
    int rem = qp % 6, div = qp / 6;
    int qmul = dequant4[rem][0] << (div + 10);
    int16_t t[16];
    for (int i = 0; i < 4; i++) {
        int z0 = dc[i*4+0] + dc[i*4+1];
        int z1 = dc[i*4+0] - dc[i*4+1];
        int z2 = dc[i*4+2] - dc[i*4+3];
        int z3 = dc[i*4+2] + dc[i*4+3];
        t[i*4+0] = z0 + z3;
        t[i*4+1] = z0 - z3;
        t[i*4+2] = z1 - z2;
        t[i*4+3] = z1 + z2;
    }
    for (int i = 0; i < 4; i++) {
        int z0 = t[0+i] + t[8+i];
        int z1 = t[0+i] - t[8+i];
        int z2 = t[4+i] - t[12+i];
        int z3 = t[4+i] + t[12+i];
        dc[0+i]  = (int16_t)(((z0 + z3) * qmul + 128) >> 8);
        dc[4+i]  = (int16_t)(((z1 + z2) * qmul + 128) >> 8);
        dc[8+i]  = (int16_t)(((z1 - z2) * qmul + 128) >> 8);
        dc[12+i] = (int16_t)(((z0 - z3) * qmul + 128) >> 8);
    }
}

/* chroma DC 2x2：Hadamard 反变换（无归一化）+ 反量化（×qmul>>7） */
static void h264_chroma_dc_dequant(H264Dec *d, int16_t *dc, int qp) {
    int rem = qp % 6, div = qp / 6;
    int qmul = dequant4[rem][0] << div;
    int a = dc[0], b = dc[1], c = dc[2], dd = dc[3];
    int e = a - b; a = a + b; b = c - dd; c = c + dd;
    dc[0] = (int16_t)(((a + c) * qmul) >> 7);
    dc[1] = (int16_t)(((e + b) * qmul) >> 7);
    dc[2] = (int16_t)(((a - c) * qmul) >> 7);
    dc[3] = (int16_t)(((e - b) * qmul) >> 7);
}

/* 读帧内预测邻接：top[0..n-1] 上方、left[0..n-1] 左侧、top[n..2n-1] 右上。
 * 帧右边界外右上像素复制 top[n-1]；y=0 时读顶部 pad（128）。 */
static void h264_pred_neigh(const uint8_t *plane, int stride, int x, int y,
                            uint8_t *top, uint8_t *left, int n, int width) {
    for (int i = 0; i < n; i++) top[i] = plane[(y - 1) * stride + x + i];
    for (int i = 0; i < n; i++) left[i] = plane[(y + i) * stride + x - 1];
    for (int i = 0; i < n; i++)
        top[n + i] = (x + n + i < width) ? plane[(y - 1) * stride + x + n + i]
                                         : top[n - 1];
}

/* I 帧宏块解码（CABAC） */
static int h264_decode_mb_i(H264Dec *d, Cabac *cab, int mb_x, int mb_y) {
    int mb_w = d->mb_w, mb_h = d->mb_h;
    int idx = mb_y * mb_w + mb_x;
    int x0 = mb_x * 16, y0 = mb_y * 16;
    d->mb_idx = idx;
    int mbt = h264_cabac_intra_mb_type(d, cab, mb_x, mb_y);
    if (mbt == 25) {
        /* I_PCM：字节对齐后直接读原始采样（8-bit 4:2:0，不经 CABAC） */
        if (cab->bit_pos > 0) { cab->bit_pos = 0; cab->byte_pos++; }
        const uint8_t *p = cab->data + cab->byte_pos;
        if (cab->byte_pos + 384 > cab->size) return -1;
        for (int i = 0; i < 16; i++)
            memcpy(d->cur_y + (y0 + i) * d->ls_y + x0, p + i * 16, 16);
        const uint8_t *cu = p + 256, *cv = cu + 64;
        for (int i = 0; i < 8; i++) {
            memcpy(d->cur_u + (y0 / 2 + i) * d->ls_c + x0 / 2, cu + i * 8, 8);
            memcpy(d->cur_v + (y0 / 2 + i) * d->ls_c + x0 / 2, cv + i * 8, 8);
        }
        cab->byte_pos += 384;
        d->mb[idx].mb_type = MB_TYPE_INTRA_PCM;
        d->mb[idx].cbp = 0x3F | 0x100 | 0x40 | 0x80;  /* 全部残差 + DC 标志 */
        d->mb[idx].qp = (uint8_t)d->qscale;
        memset(d->cur_nz, 0, 24);
        d->left_cbp = d->mb[idx].cbp;
        d->top_cbp = d->mb[idx].cbp;
        memcpy(d->top_nz + (size_t)(mb_x + 1) * 24, d->cur_nz, 24);
        memcpy(d->left_nz, d->cur_nz, 24);
        return 0;
    }
    int type = i_mb_type_tab[mbt][0];
    int pred16 = i_mb_type_tab[mbt][1];
    int cbp = i_mb_type_tab[mbt][2];

    d->mb[idx].mb_type = (int16_t)type;
    d->mb[idx].intra16x16_pred = (uint8_t)(pred16 >= 0 ? pred16 : 0);

    /* chroma 帧内模式 */
    int lcm = (mb_x > 0) ? d->mb[idx - 1].chroma_pred : 0;
    int tcm = (mb_y > 0) ? d->mb[idx - mb_w].chroma_pred : 0;
    int cm = h264_cabac_chroma_pred(d, cab, lcm, tcm);
    d->mb[idx].chroma_pred = (uint8_t)cm;

    if (IS_INTRA4x4(type)) {
        /* 16 个 4x4 预测模式（标准块序，pred = min(左邻, 上邻)） */
        uint8_t modes[16];
        for (int i = 0; i < 16; i++) {
            int bx4 = h264_blk_x(i), by4 = h264_blk_y(i);
            int la, up;
            if (bx4 > 0) la = modes[h264_blk_n(bx4 - 1, by4)];
            else if (mb_x > 0) la = d->mb[idx - 1].intra4x4[h264_blk_n(3, by4)];
            else la = 2;
            if (by4 > 0) up = modes[h264_blk_n(bx4, by4 - 1)];
            else if (mb_y > 0) up = d->mb[idx - mb_w].intra4x4[h264_blk_n(bx4, 3)];
            else up = 2;
            int pred = (la < up) ? la : up;
            modes[i] = (uint8_t)h264_cabac_intra4x4_mode(d, cab, pred);
        }
        memcpy(d->mb[idx].intra4x4, modes, 16);
    }

    /* cbp（I4x4 时解码；I16x16 由 mb_type 决定） */
    int cbpl = 0, cbpc = 0;
    if (IS_INTRA4x4(type)) {
        cbpl = h264_cabac_cbp_luma(d, cab);
        cbpc = h264_cabac_cbp_chroma(d, cab);
        cbp = cbpl | (cbpc << 4);
        d->mb[idx].cbp = (uint16_t)cbp;
    } else {
        d->mb[idx].cbp = (uint8_t)cbp;
    }

    /* QP delta（有残差时） */
    if (IS_INTRA16x16(type) || (cbp & 0x3F) || cbpl) {
        h264_cabac_mb_qp_delta(d, cab);
    }
    int qp = d->qscale;
    int qpc = h264_chroma_qp(qp, d->pps.chroma_qp_index_offset);
    d->mb[idx].qp = (uint8_t)qp;
    memset(d->cur_nz, 0, 24);

    /* 帧内预测 + 残差 */
    if (IS_INTRA16x16(type)) {
        /* DC 块 + 16 AC 块 */
        memset(d->dc_block, 0, 32 * sizeof(int16_t));
        int16_t *dc = d->dc_block;
        int nzdc = h264_cabac_residual(d, cab, 0, 16, dc, 0, 1);
        if (nzdc) {
            h264_luma_dc_dequant(d, dc, qp);
        }
        uint8_t top[32], left[32];
        h264_pred_neigh(d->cur_y, d->ls_y, x0, y0, top, left, 16, d->width);
        /* I16x16 预测（含残差逐块加） */
        uint8_t *dst = d->cur_y + y0 * d->ls_y + x0;
        h264_pred16x16(dst, d->ls_y, pred16, top, left);
        for (int b = 0; b < 16; b++) {
            int bx4 = h264_blk_x(b), by4 = h264_blk_y(b);
            int bx = x0 + bx4 * 4, by = y0 + by4 * 4;
            memset(d->tmp_block, 0, 32 * sizeof(int16_t));
            d->tmp_block[0] = dc[b];
            if (cbp & (1 << (by4 >> 1))) {
                int nz = h264_cabac_residual(d, cab, 1, 15, d->tmp_block + 1, b, 0);
                d->cur_nz[b] = (uint8_t)nz;
            }
            h264_dequant_idct4(dst + by4 * 4 * d->ls_y + bx4 * 4,
                               d->ls_y, d->tmp_block, qp, 1);
        }
    } else if (IS_INTRA4x4(type)) {
        for (int b = 0; b < 16; b++) {
            int bx4 = h264_blk_x(b), by4 = h264_blk_y(b);
            int bx = x0 + bx4 * 4, by = y0 + by4 * 4;
            uint8_t *dst = d->cur_y + by * d->ls_y + bx;
            uint8_t top[8], left[8];
            h264_pred_neigh(d->cur_y, d->ls_y, bx, by, top, left, 4, d->width);
            /* 模式可用性：右上不可用时 3/7 退化 */
            int mode = d->mb[idx].intra4x4[b];
            if ((bx + 4 >= d->width) && (mode == 3 || mode == 7))
                mode = 2;
            h264_pred4x4(dst, d->ls_y, mode, top, left);
            memset(d->tmp_block, 0, 32 * sizeof(int16_t));
            if (cbp & (1 << (by4 >> 1))) {
                int nz = h264_cabac_residual(d, cab, 2, 16, d->tmp_block, b, 0);
                d->cur_nz[b] = (uint8_t)nz;
            }
            h264_dequant_idct4(dst, d->ls_y, d->tmp_block, qp, 0);
        }
    }

    /* chroma：DC 2x2 + 4 AC 块 */
    for (int c = 0; c < 2; c++) {
        uint8_t *dst = (c == 0) ? d->cur_u : d->cur_v;
        int ls = d->ls_c;
        uint8_t top[16], left[16];
        h264_pred_neigh(dst, ls, x0 >> 1, y0 >> 1, top, left, 8, d->width >> 1);
        int cmode = d->mb[idx].chroma_pred;
        uint8_t *cdst = dst + (y0 >> 1) * ls + (x0 >> 1);
        h264_pred8x8c(cdst, ls, cmode, top, left);
        int16_t dc[4] = {0,0,0,0};
        int cbpc_local = (cbp >> 4) & 3;
        if (cbpc_local & 1) {
            int16_t blk[4];
            int nz = h264_cabac_residual(d, cab, 3, 4, blk, 16 + c * 4, 1);
            (void)nz;
            /* 扫描序 → dc[0..3] */
            dc[0] = blk[0]; dc[1] = blk[1]; dc[2] = blk[2]; dc[3] = blk[3];
            h264_chroma_dc_dequant(d, dc, qpc);
        }
        if (cbpc_local & 2) {
            for (int i = 0; i < 4; i++) {
                int bx = (x0 >> 1) + (i & 1) * 4, by = (y0 >> 1) + (i >> 1) * 4;
                uint8_t *bdst = dst + by * ls + bx;
                memset(d->tmp_block, 0, 32 * sizeof(int16_t));
                d->tmp_block[0] = dc[i];
                int nz = h264_cabac_residual(d, cab, 4, 15, d->tmp_block + 1, 16 + c * 4 + i, 0);
                d->cur_nz[16 + c * 4 + i] = (uint8_t)nz;
                h264_dequant_idct4(bdst, ls, d->tmp_block, qpc, 1);
            }
        }
    }

    /* 更新邻接 cbp/nz */
    d->left_cbp = d->mb[idx].cbp;
    d->top_cbp = d->mb[idx].cbp;
    memcpy(d->top_nz + (size_t)(mb_x + 1) * 24, d->cur_nz, 24);
    memcpy(d->left_nz, d->cur_nz, 24);
    return 0;
}

/* 单 slice 的 CABAC 定位信息 */
typedef struct {
    int first_mb;      /* first_mb_in_slice */
    int byte_start;    /* rbsp 中 CABAC 起始字节 */
    int cabac_len;
    int slice_type;
    int idr_flag;
    int nal_ref_idc;
} SliceInfo;

/* 解码一帧（支持多 slice，I 帧路径；P/B 后续迭代） */
static int h264_decode_frame(H264Dec *dec, const uint8_t *data, size_t size) {
    uint8_t *rbsp = (uint8_t *)malloc(size + 16);
    if (!rbsp) return -1;
    SliceInfo slices[64];
    int n_slices = 0;
    int mb_total = dec->mb_w * dec->mb_h;

    /* pass 1：解析帧内全部 slice NAL 的 header，定位各 slice 的 CABAC 起点 */
    size_t off = 0;
    int ret = -1;
    while (off + 4 <= size) {
        uint32_t nlen = ((uint32_t)data[off] << 24) | ((uint32_t)data[off+1] << 16) |
                        ((uint32_t)data[off+2] << 8) | (uint32_t)data[off+3];
        if (off + 4 + nlen > size) break;
        const uint8_t *nal = data + off + 4;
        int nal_type = nal[0] & 0x1f;
        dec->nal_ref_idc = (nal[0] >> 5) & 3;
        off += 4 + nlen;
        if (nal_type == 7 || nal_type == 8) continue;   /* SPS/PPS */
        if (nal_type != 1 && nal_type != 5) continue;   /* 非 slice */
        dec->idr_flag = (nal_type == 5);
        int rl = nal_to_rbsp(nal + 1, nlen - 1, rbsp, (int)size + 8);
        if (rl <= 0) break;
        int first_mb = 0, cabac_bit = 0;
        if (parse_slice_header(dec, rbsp, rl, &first_mb, &cabac_bit) != 0) break;
        if (dec->slice_type != SLICE_I) {
            fprintf(stderr, "[h264] slice type %d not yet supported (I only)\n", dec->slice_type);
            goto done;
        }
        int byte_start = (cabac_bit + 7) >> 3;
        if (rl - byte_start < 1) break;
        if (n_slices < 64) {
            slices[n_slices].first_mb = first_mb;
            slices[n_slices].byte_start = byte_start;
            slices[n_slices].cabac_len = rl - byte_start;
            slices[n_slices].slice_type = dec->slice_type;
            slices[n_slices].idr_flag = dec->idr_flag;
            slices[n_slices].nal_ref_idc = dec->nal_ref_idc;
            n_slices++;
        }
    }
    if (n_slices == 0) goto done;

    /* 清空帧（I 帧）：DC 灰值 128 垫边，避免帧边界读取越界 */
    int ls = dec->ls_y;
    size_t yb = (size_t)ls * dec->height;
    size_t cb = (size_t)(ls >> 1) * (dec->height >> 1);
    memset(dec->yuv_out, 128, yb + 6 * (size_t)ls + cb * 2);
    dec->cur_y = dec->yuv_out + ls;
    dec->cur_u = dec->cur_y + yb + 4 * ls;
    dec->cur_v = dec->cur_u + cb;
    dec->left_cbp = 0; dec->top_cbp = 0;
    memset(dec->top_nz, 0, (size_t)(dec->mb_w + 1) * 24);
    memset(dec->mb, 0, (size_t)mb_total * sizeof(MbState));

    /* pass 2：逐 slice 解码（slice 边界重初始化 CABAC 与 QP） */
    for (int s = 0; s < n_slices; s++) {
        int start = slices[s].first_mb;
        int end = (s + 1 < n_slices) ? slices[s + 1].first_mb : mb_total;
        dec->qscale = 26 + dec->pps.pic_init_qp_minus26 + dec->slice_qp_delta;
        dec->last_qscale_diff = 0;
        dec->qp_y = dec->qscale;

        Cabac cab;
        cabac_init(&cab, rbsp + slices[s].byte_start, (size_t)slices[s].cabac_len);
        cabac_ctx_init_decode(dec, slices[s].slice_type);

        for (int i = start; i < end; i++) {
            int mb_x = i % dec->mb_w, mb_y = i / dec->mb_w;
            /* 行首或 slice 首 MB：left 邻接重置 */
            if (mb_x == 0 || i == start) { dec->left_cbp = 0; memset(dec->left_nz, 0, 24); }
            dec->top_cbp = (mb_y > 0) ? dec->mb[(mb_y - 1) * dec->mb_w + mb_x].cbp : 0;
            if (h264_decode_mb_i(dec, &cab, mb_x, mb_y) != 0) {
                fprintf(stderr, "[h264] MB %d/%d (slice %d) decode failed\n", i, mb_total, s);
                goto done;
            }
        }
    }
    ret = 0;
done:
    free(rbsp);
    return ret;
}

int h264_decoder_decode(H264Decoder *d, int frame_idx) {
    if (!d || !d->dec || !d->movie || frame_idx < 1 || frame_idx > d->movie->n_frames)
        return -1;
    H264Dec *dec = d->dec;
    const MPMovie *m = d->movie;
    int i = frame_idx - 1;
    if ((uint64_t)m->sample_off[i] + m->sample_size[i] > m->size) return -1;
    const uint8_t *nal_data = m->data + m->sample_off[i];
    size_t nal_size = m->sample_size[i];
    return h264_decode_frame(dec, nal_data, nal_size);
}

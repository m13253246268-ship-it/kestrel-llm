/**
 * vllm_mp4.c - MP4 (ISO BMFF) 容器解析，零第三方依赖
 *
 * 解析 ftyp/moov(trak/mdia/minf/stbl) + mdat，从 stbl 提取：
 *   - stsd: avcC (SPS/PPS + NAL 长度前缀), width/height
 *   - stts/stss/stsz/stco/stsc/ctts: 帧索引（offset/size/PTS/IDR 标记）
 * 全文件线性扫描 box 树，无外部库。
 */
#include "vllm_media.h"
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static uint64_t rd64(const uint8_t *p) {
    return ((uint64_t)rd32(p) << 32) | rd32(p + 4);
}
static uint16_t rd16_un(const uint8_t *p, size_t o) {
    return (uint16_t)(((uint16_t)p[o] << 8) | (uint16_t)p[o + 1]);
}

/* 在 [off, end) 内查找单层 type 匹配的 box，返回内容起始偏移（0=未找到），
 * 并通过 *content_end 返回该 box 的内容结束偏移。 */
static size_t find_box(const uint8_t *b, size_t off, size_t end,
                       const char *type, size_t *content_end) {
    while (off + 8 <= end) {
        uint32_t sz = rd32(b + off);
        size_t hd = 8, box_size;
        if (sz == 1) {
            if (off + 16 > end) return 0;
            box_size = (size_t)rd64(b + off + 8);
            hd = 16;
        } else if (sz == 0) {
            box_size = end - off;
        } else if (sz < 8) {
            return 0;
        } else {
            box_size = (size_t)sz;
        }
        if (memcmp(b + off + 4, type, 4) == 0) {
            if (content_end) *content_end = off + box_size;
            return off + hd;
        }
        off += box_size;
    }
    return 0;
}

int mp4_open(MPMovie *m, const uint8_t *data, size_t size) {
    memset(m, 0, sizeof(*m));
    m->data = data;
    m->size = size;

    /* ---- moov ---- */
    size_t moov_end = 0;
    size_t moov = find_box(data, 0, size, "moov", &moov_end);
    if (!moov) return -1;

    /* ---- mvhd: timescale / duration ---- */
    size_t m_end = 0;
    size_t mvhd = find_box(data, moov, moov_end, "mvhd", &m_end);
    if (mvhd) {
        int ver = data[mvhd];
        int ts_off = (ver == 1) ? 20 : 12;
        int dur_off = (ver == 1) ? 24 : 16;
        m->timescale = (int)rd32(data + mvhd + ts_off);
        int64_t dur = (ver == 1) ? (int64_t)rd64(data + mvhd + dur_off)
                                 : (int64_t)rd32(data + mvhd + dur_off);
        if (m->timescale > 0) m->duration_ms = dur * 1000 / m->timescale;
    }

    /* ---- 遍历 trak 找视频轨道 ---- */
    size_t off = moov;
    while (off + 8 <= moov_end) {
        uint32_t sz = rd32(data + off);
        size_t hd = 8, box_size;
        if (sz == 1) {
            if (off + 16 > moov_end) break;
            box_size = (size_t)rd64(data + off + 8);
            hd = 16;
        } else if (sz == 0) {
            box_size = moov_end - off;
        } else if (sz < 8) {
            break;
        } else {
            box_size = (size_t)sz;
        }
        int is_trak = (memcmp(data + off + 4, "trak", 4) == 0);
        size_t content = off + hd;
        size_t trak_end = off + box_size;
        off = trak_end;
        if (!is_trak) continue;

        /* hdlr -> handler_type == 'vide'? */
        size_t mdia_end = 0;
        size_t mdia = find_box(data, content, trak_end, "mdia", &mdia_end);
        if (!mdia) continue;
        /* mdhd: track 媒体时间尺度（PTS 基准，可能不同于 mvhd） */
        size_t dh_end = 0;
        size_t mdhd = find_box(data, mdia, mdia_end, "mdhd", &dh_end);
        if (mdhd) {
            int ver = data[mdhd];
            int ts_off = (ver == 1) ? 20 : 12;
            m->timescale = (int)rd32(data + mdhd + ts_off);
        }
        size_t h_end = 0;
        size_t hdlr = find_box(data, mdia, mdia_end, "hdlr", &h_end);
        if (!hdlr || hdlr + 12 > trak_end) continue;
        if (!(data[hdlr+8] == 'v' && data[hdlr+9] == 'i' &&
              data[hdlr+10] == 'd' && data[hdlr+11] == 'e')) continue;

        /* stbl */
        size_t minf_end = 0;
        size_t minf = find_box(data, mdia, mdia_end, "minf", &minf_end);
        if (!minf) continue;
        size_t stbl_end = 0;
        size_t stbl = find_box(data, minf, minf_end, "stbl", &stbl_end);
        if (!stbl) continue;

        /* stsd: 取宽高 + avcC */
        size_t sd_end = 0;
        size_t stsd = find_box(data, stbl, stbl_end, "stsd", &sd_end);
        if (!stsd) continue;
        if (sd_end - stsd < 12) continue;
        size_t sentry = stsd + 8;  /* 第一个 sample entry（跳过 version/flags+count） */
        if (!(data[sentry+4] == 'a' && data[sentry+5] == 'v' &&
              data[sentry+6] == 'c' && data[sentry+7] == '1')) continue;

        m->width  = (int)rd16_un(data, sentry + 32);
        m->height = (int)rd16_un(data, sentry + 34);

        /* avcC box 在 sample entry 内（VisualSampleEntry 86B 裸字段之后） */
        size_t se_size = (size_t)rd32(data + sentry);
        size_t se_end = (se_size >= 8) ? sentry + se_size : sd_end;
        if (se_end > sd_end) se_end = sd_end;
        size_t avcc_end = 0;
        size_t avcc = find_box(data, sentry + 86, se_end, "avcC", &avcc_end);
        if (avcc && avcc + 6 <= avcc_end) {
            m->length_size = (data[avcc + 4] & 3) + 1;
            int n_sps = data[avcc + 5] & 31;
            size_t p = avcc + 6;
            for (int i = 0; i < n_sps && p + 2 <= avcc_end; i++) {
                int sl = (int)rd16_un(data, p);
                if (p + 2 + (size_t)sl > avcc_end) break;
                if (i == 0) { m->sps = data + p + 2; m->sps_len = sl; }
                p += 2 + (size_t)sl;
            }
            if (p < avcc_end) {
                int n_pps = data[p]; p += 1;
                for (int i = 0; i < n_pps && p + 2 <= avcc_end; i++) {
                    int pl = (int)rd16_un(data, p);
                    if (p + 2 + (size_t)pl > avcc_end) break;
                    if (i == 0) { m->pps = data + p + 2; m->pps_len = pl; }
                    p += 2 + (size_t)pl;
                }
            }
        }

        /* ---- 帧索引: stts/stsz/stco/stsc + stss ---- */
        int n_frames = 0;
        uint32_t *sizes = NULL, *offsets = NULL, *pts = NULL;
        uint8_t *idr = NULL;

        size_t z_end = 0;
        size_t stsz = find_box(data, stbl, stbl_end, "stsz", &z_end);
        if (stsz) {
            uint32_t sample_size = rd32(data + stsz + 4);
            n_frames = (int)rd32(data + stsz + 8);
            if (n_frames > MPM_MAX_SAMPLES) n_frames = MPM_MAX_SAMPLES;
            sizes = (uint32_t *)malloc((size_t)n_frames * sizeof(uint32_t));
            if (sample_size != 0) {
                for (int i = 0; i < n_frames; i++) sizes[i] = sample_size;
            } else if (z_end - stsz >= (size_t)12 + (size_t)n_frames * 4) {
                for (int i = 0; i < n_frames; i++)
                    sizes[i] = rd32(data + stsz + 12 + (size_t)i * 4);
            }
        }
        if (!sizes || n_frames <= 0) { free(sizes); break; }

        offsets = (uint32_t *)malloc((size_t)n_frames * sizeof(uint32_t));
        pts     = (uint32_t *)malloc((size_t)n_frames * sizeof(uint32_t));
        idr     = (uint8_t *)calloc((size_t)n_frames, 1);
        if (!offsets || !pts || !idr) { free(sizes); free(offsets); free(pts); free(idr); break; }

        /* stsc + stco -> 每 sample 偏移 */
        size_t c_end = 0;
        size_t stsc = find_box(data, stbl, stbl_end, "stsc", &c_end);
        size_t o_end = 0;
        size_t stco = find_box(data, stbl, stbl_end, "stco", &o_end);
        if (stsc && stco) {
            int n_sc = (int)rd32(data + stsc + 4);
            int n_co = (int)rd32(data + stco + 4);
            int sc = 0, first_chunk = 1, samples_per_chunk = 1;
            int chunk = 0, sample = 0, off32 = 0;
            while (sample < n_frames && off32 < n_co) {
                if (sc < n_sc) {
                    int fc = (int)rd32(data + stsc + 8 + (size_t)sc * 12);
                    if (chunk + 1 == fc) {
                        samples_per_chunk = (int)rd32(data + stsc + 8 + (size_t)sc * 12 + 4);
                        sc++;
                    }
                }
                uint32_t chunk_off = rd32(data + stco + 8 + (size_t)off32 * 4);
                for (int s = 0; s < samples_per_chunk && sample < n_frames; s++) {
                    offsets[sample] = chunk_off;
                    chunk_off += sizes[sample];
                    sample++;
                }
                off32++;
                chunk++;
            }
        }

        /* stts -> PTS (显示顺序，仅第一项常用) */
        size_t t_end = 0;
        size_t stts = find_box(data, stbl, stbl_end, "stts", &t_end);
        uint32_t ts = 0;
        if (stts) {
            int n_e = (int)rd32(data + stts + 4);
            int cnt = 0;
            for (int i = 0; i < n_e; i++) {
                int c = (int)rd32(data + stts + 8 + (size_t)i * 8);
                uint32_t d = rd32(data + stts + 12 + (size_t)i * 8);
                for (int k = 0; k < c && cnt < n_frames; k++) {
                    pts[cnt] = ts;
                    cnt++; ts += d;
                }
            }
        }

        /* stss -> IDR 标记 */
        size_t s_end = 0;
        size_t stss = find_box(data, stbl, stbl_end, "stss", &s_end);
        if (stss) {
            int n_i = (int)rd32(data + stss + 4);
            for (int i = 0; i < n_i; i++) {
                uint32_t sid = rd32(data + stss + 8 + (size_t)i * 4);
                if (sid >= 1 && sid <= (uint32_t)n_frames) idr[sid - 1] = 1;
            }
        } else {
            idr[0] = 1;  /* 无 stss 则假设首帧为 IDR */
        }

        /* 时间戳换算 ms */
        if (m->timescale > 0) {
            for (int i = 0; i < n_frames; i++)
                pts[i] = (uint32_t)((uint64_t)pts[i] * 1000 / m->timescale);
        }

        /* 存入结构 */
        size_t total = (size_t)n_frames * (4 + 4 + 4) + (size_t)n_frames;
        uint8_t *blk = (uint8_t *)malloc(total);
        if (!blk) { free(sizes); free(offsets); free(pts); free(idr); break; }
        m->own_index = blk;
        m->sample_off = (uint32_t *)blk;
        m->sample_size = (uint32_t *)(blk + (size_t)n_frames * 4);
        m->sample_pts_ms = (uint32_t *)(blk + (size_t)n_frames * 8);
        m->sample_idr = blk + (size_t)n_frames * 12;
        memcpy(m->sample_off, offsets, (size_t)n_frames * 4);
        memcpy(m->sample_size, sizes, (size_t)n_frames * 4);
        memcpy(m->sample_pts_ms, pts, (size_t)n_frames * 4);
        memcpy(m->sample_idr, idr, (size_t)n_frames);
        m->n_frames = n_frames;

        free(sizes); free(offsets); free(pts); free(idr);
        return 0;
    }
    return -1;
}

void mp4_close(MPMovie *m) {
    free(m->own_index);
    memset(m, 0, sizeof(*m));
}

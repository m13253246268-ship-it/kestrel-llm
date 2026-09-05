/**
 * vllm_media.h - Real Image & Video Frame Loading
 *
 * Supports JPEG, PNG, BMP, TGA, etc. via stb_image.
 * Video: load numbered frame images from a directory.
 */

#ifndef VLLM_MEDIA_H
#define VLLM_MEDIA_H

#include <stdint.h>
#include <stddef.h>   /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load an image file (JPEG, PNG, BMP, TGA, etc.) into RGB uint8 buffer.
 *   path:     file path (UTF-8 or ANSI on Windows)
 *   out_w:    [out] image width
 *   out_h:    [out] image height
 * Returns:    malloc'd RGB data [height * width * 3], or NULL on failure.
 *             Caller must free with media_free_image().
 */
uint8_t* media_load_image(const char *path, int *out_w, int *out_h);

/**
 * Decode an encoded image (JPEG/PNG/BMP/... payload) from memory into RGB.
 *   data:     encoded image bytes (e.g. base64-decoded JPEG)
 *   len:      number of bytes
 *   out_w:    [out] image width
 *   out_h:    [out] image height
 * Returns:    malloc'd RGB data [height * width * 3], or NULL on failure.
 *             Caller must free with media_free_image().
 */
uint8_t* media_load_image_memory(const uint8_t *data, size_t len,
                                 int *out_w, int *out_h);

/**
 * Decode a standard base64 string into raw bytes.
 *   b64:      NUL-terminated base64 text (whitespace tolerated)
 *   out_len:  [out] decoded byte count
 * Returns:    malloc'd decoded bytes, or NULL on failure.
 *             Caller must free with free().
 */
uint8_t* media_base64_decode(const char *b64, size_t *out_len);

/**
 * Free image buffer returned by media_load_image().
 */
void media_free_image(uint8_t *data);

/**
 * Load video frames from a directory containing numbered image files.
 * Looks for files matching: frame_%d.png, frame_%d.jpg, %04d.png, etc.
 *   dir:         directory path containing frame images
 *   out_w:       [out] frame width (all frames must be same size)
 *   out_h:       [out] frame height
 *   out_n:       [out] number of frames loaded
 *   max_frames:  maximum number of frames to load (0 = unlimited)
 * Returns:       array of frame pointers, each [height * width * 3] RGB uint8.
 *                Caller must free with media_free_video_frames().
 *                Returns NULL on failure.
 */
uint8_t** media_load_video_frames(const char *dir,
                                   int *out_w, int *out_h, int *out_n,
                                   int max_frames);

/**
 * Free video frame array returned by media_load_video_frames().
 */
void media_free_video_frames(uint8_t **frames, int n_frames);

/**
 * Scale an RGB image using bilinear interpolation.
 *   dst:       destination buffer [dst_w * dst_h * 3]
 *   src:       source buffer [src_w * src_h * 3]
 *   src_w, src_h: source dimensions
 *   dst_w, dst_h: destination dimensions
 */
void media_resize_rgb(uint8_t *dst, const uint8_t *src,
                       int src_w, int src_h, int dst_w, int dst_h);

#ifdef __cplusplus
}
#endif

#endif /* VLLM_MEDIA_H */

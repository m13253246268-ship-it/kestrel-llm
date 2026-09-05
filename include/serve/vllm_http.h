/* ================================================================
 * vllm_http.h - Zero-dependency HTTP/1.1 server + minimal JSON
 *
 * Uses only OS system calls (winsock2 on Windows, POSIX sockets
 * elsewhere). No third-party libraries. Provides:
 *   - a tiny JSON value tree (parse / serialize / query)
 *   - a single-listener HTTP server with a per-connection handler
 * ================================================================ */
#ifndef VLLM_HTTP_H
#define VLLM_HTTP_H

#include <stddef.h>

/* ---------------- JSON ---------------- */

typedef enum {
    VJ_NULL, VJ_BOOL, VJ_NUMBER, VJ_STRING, VJ_ARRAY, VJ_OBJECT
} VJsonType;

typedef struct VJson VJson;
typedef struct VJsonPair { char *key; VJson *val; } VJsonPair;

struct VJson {
    VJsonType type;
    union {
        int boolean;                     /* VJ_BOOL */
        double number;                   /* VJ_NUMBER */
        char *str;                       /* VJ_STRING (owned, UTF-8) */
        struct { VJson **items; size_t n, cap; } arr;
        struct { VJsonPair *pairs; size_t n, cap; } obj;
    } u;
};

/* Parse a NUL-terminated JSON text. Returns NULL on syntax error. */
VJson *vjson_parse(const char *text);
/* Deep free a parsed/built tree. */
void vjson_free(VJson *v);
/* Deep-copy a value tree into a new independent tree (free with vjson_free). */
VJson *vjson_clone(const VJson *v);

/* Query helpers (return NULL / 0 when absent or type mismatch). */
VJson *vjson_obj_get(const VJson *obj, const char *key);
const char *vjson_str(const VJson *v);
double vjson_num(const VJson *v);
int vjson_bool(const VJson *v);
size_t vjson_array_len(const VJson *v);
VJson *vjson_array_get(const VJson *v, size_t i);

/* Builders. The returned tree owns all its memory (free with vjson_free).
 * vjson_obj_set / vjson_array_push take ownership of the child value. */
VJson *vjson_new_string(const char *s);
VJson *vjson_new_number(double n);
VJson *vjson_new_bool(int b);
VJson *vjson_new_null(void);
VJson *vjson_new_object(void);
VJson *vjson_new_array(void);
void vjson_obj_set(VJson *obj, const char *key, VJson *val);
void vjson_array_push(VJson *arr, VJson *val);

/* Compact serialization into buf (NUL-terminated). Returns bytes written
 * (excluding NUL), or 0 if the buffer is too small. */
size_t vjson_serialize(const VJson *v, char *buf, size_t cap);

/* ---------------- HTTP server ---------------- */

typedef struct VHttpRequest {
    const char *method;      /* GET / POST / ... (points into req buffer) */
    const char *path;        /* path (points into req buffer) */
    const char *query;       /* raw query or "" */
    const char *body;        /* body ("" if none) */
    size_t body_len;
    const char *raw;         /* full raw request (for header lookup) */
} VHttpRequest;

typedef struct VHttpResponse {
    int status;              /* 200 / 400 / 404 / 500 ... */
    const char *content_type;/* e.g. "application/json"; NULL = text/plain */
    const char *body;        /* response body (valid until handler returns,
                              * or until the server sends it when body_owned
                              * is set) */
    size_t body_len;         /* 0 = strlen(body) */
    char *body_owned;        /* if non-NULL: heap buffer owned by the server;
                              * freed after the response is sent. Set this
                              * for per-request malloc'd bodies (thread-safe
                              * replacement for static buffers). */
    int stream;              /* set 1: handler owns the connection and must
                              * call vhttp_stream_write() then vhttp_stream_done() */
} VHttpResponse;

/* Connection handle used by streaming writes. */
typedef struct VHttpConn VHttpConn;

/* Route handler: parse request, fill response. Called serially (one at a
 * time) from the accept loop, so no locking is required inside. */
typedef void (*VHttpHandler)(const VHttpRequest *req, VHttpResponse *resp,
                             VHttpConn *conn, void *userdata);

/* Start the server on `port` (0 = ephemeral). on_start(actual_port, ud) is
 * invoked once the listener is up. Blocks forever serving requests.
 * Multi-threaded: one accept thread + `n_threads` worker threads; a bounded
 * connection queue protects against backlog flooding. */
int vhttp_serve_ex(int port, VHttpHandler handler, void *userdata,
                   void (*on_start)(int actual_port, void *ud), int n_threads);
int vhttp_serve(int port, VHttpHandler handler, void *userdata,
                void (*on_start)(int actual_port, void *ud));

/* Ask the running server to stop accepting connections and return from
 * vhttp_serve_ex after the current request finishes (graceful shutdown).
 * Safe to call from a route handler (e.g. an /admin shutdown endpoint). */
void vhttp_stop(void);

/* Current number of open client connections (accepted but not yet closed),
 * for the admin "current users / connections" view. */
long vhttp_active_conns(void);

/* Streaming helpers - only valid inside a handler after resp->stream = 1. */
/* Send the HTTP response header for a streaming response (must be called
 * before the first vhttp_stream_write). */
int vhttp_stream_begin(VHttpConn *conn, int status, const char *content_type);
int vhttp_stream_write(VHttpConn *conn, const char *data, size_t len);
int vhttp_stream_flush(VHttpConn *conn);
/* Finish a streaming response (sends final chunk + closes). */
int vhttp_stream_done(VHttpConn *conn);

/* ---------------- cross-platform mutex ---------------- */

/* Reusable mutex for serializing engine-critical sections across the
 * worker threads (e.g. one inference state used by all clients). */
typedef struct VHttpMutex VHttpMutex;
VHttpMutex *vhttp_mutex_new(void);
void vhttp_mutex_free(VHttpMutex *m);
void vhttp_mutex_lock(VHttpMutex *m);
void vhttp_mutex_unlock(VHttpMutex *m);

/* Tiny helpers used by route handlers. */
const char *vhttp_json_error(const char *message, char *buf, size_t cap);
/* Case-insensitive header lookup over the raw request ("" if absent). */
const char *vhttp_req_header(const VHttpRequest *r, const char *name);

#endif /* VLLM_HTTP_H */

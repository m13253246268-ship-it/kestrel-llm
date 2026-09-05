/* ================================================================
 * vllm_http.c - Zero-dependency HTTP/1.1 server + minimal JSON
 *
 * Depends only on the OS socket API (winsock2 / POSIX) and the C
 * runtime. A hand-written recursive-descent JSON parser/serializer
 * covers the OpenAI API payloads (objects, arrays, strings, numbers,
 * booleans, null) with \uXXXX escape decoding.
 *
 * The accept loop is single-threaded and serial: one request (or one
 * SSE stream) at a time, so route handlers need no locking.
 * ================================================================ */
#include "vllm_http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>
#include <netdb.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

/* Graceful-stop flag: vhttp_stop() sets it and the accept loop polls it
 * through a select() with a short timeout, so a route handler (e.g. the
 * /admin shutdown endpoint) can ask the server to exit cleanly after the
 * in-flight request has been answered. */
static volatile int g_vhttp_stop = 0;

void vhttp_stop(void) {
    g_vhttp_stop = 1;
}

/* Open client-connection counter (accepted, not yet closed). */
static volatile long g_vhttp_conns = 0;

long vhttp_active_conns(void) {
    return g_vhttp_conns;
}

/* ================================================================
 * JSON
 * ================================================================ */

typedef struct {
    const char *p;
    const char *end;
    int depth;
} VJsonParser;

static void skip_ws(VJsonParser *P) {
    while (P->p < P->end) {
        char c = *P->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') P->p++;
        else break;
    }
}

/* Append one UTF-8 code point (from \uXXXX, optionally a surrogate pair)
 * to a growing byte buffer. */
static void utf8_append(char **buf, size_t *len, size_t *cap, uint32_t cp) {
    if (cp < 0x80) {
        if (*len + 1 > *cap) { *cap = (*cap ? *cap * 2 : 32); *buf = (char *)realloc(*buf, *cap); }
        (*buf)[(*len)++] = (char)cp;
    } else if (cp < 0x800) {
        if (*len + 2 > *cap) { *cap = (*cap ? *cap * 2 : 32); *buf = (char *)realloc(*buf, *cap); }
        (*buf)[(*len)++] = (char)(0xC0 | (cp >> 6));
        (*buf)[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        if (*len + 3 > *cap) { *cap = (*cap ? *cap * 2 : 32); *buf = (char *)realloc(*buf, *cap); }
        (*buf)[(*len)++] = (char)(0xE0 | (cp >> 12));
        (*buf)[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        (*buf)[(*len)++] = (char)(0x80 | (cp & 0x3F));
    } else {
        if (*len + 4 > *cap) { *cap = (*cap ? *cap * 2 : 32); *buf = (char *)realloc(*buf, *cap); }
        (*buf)[(*len)++] = (char)(0xF0 | (cp >> 18));
        (*buf)[(*len)++] = (char)(0x80 | ((cp >> 12) & 0x3F));
        (*buf)[(*len)++] = (char)(0x80 | ((cp >> 6) & 0x3F));
        (*buf)[(*len)++] = (char)(0x80 | (cp & 0x3F));
    }
}

static int hex4(const char *s) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        if (c >= '0' && c <= '9') v = (v << 4) | (c - '0');
        else if (c >= 'a' && c <= 'f') v = (v << 4) | (c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v = (v << 4) | (c - 'A' + 10);
        else return -1;
    }
    return v;
}

/* Parse a JSON string (after the opening quote). Returns malloc'd UTF-8. */
static char *parse_string_body(VJsonParser *P) {
    char *buf = NULL;
    size_t len = 0, cap = 0;
    while (P->p < P->end) {
        char c = *P->p;
        if (c == '"') { P->p++; break; }
        if (c == '\\') {
            P->p++;
            if (P->p >= P->end) { free(buf); return NULL; }
            char e = *P->p++;
            switch (e) {
            case '"': utf8_append(&buf, &len, &cap, '"'); break;
            case '\\': utf8_append(&buf, &len, &cap, '\\'); break;
            case '/': utf8_append(&buf, &len, &cap, '/'); break;
            case 'b': utf8_append(&buf, &len, &cap, '\b'); break;
            case 'f': utf8_append(&buf, &len, &cap, '\f'); break;
            case 'n': utf8_append(&buf, &len, &cap, '\n'); break;
            case 'r': utf8_append(&buf, &len, &cap, '\r'); break;
            case 't': utf8_append(&buf, &len, &cap, '\t'); break;
            case 'u': {
                if (P->end - P->p < 4) { free(buf); return NULL; }
                int hi = hex4(P->p); P->p += 4;
                if (hi < 0) { free(buf); return NULL; }
                uint32_t cp = (uint32_t)hi;
                if (hi >= 0xD800 && hi <= 0xDBFF && P->end - P->p >= 6 &&
                    P->p[0] == '\\' && P->p[1] == 'u') {
                    int lo = hex4(P->p + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        P->p += 6;
                        cp = 0x10000 + ((uint32_t)(hi - 0xD800) << 10) + (uint32_t)(lo - 0xDC00);
                    }
                }
                utf8_append(&buf, &len, &cap, cp);
                break;
            }
            default: free(buf); return NULL;
            }
        } else {
            if ((unsigned char)c < 0x20) { free(buf); return NULL; }
            /* Pass through raw UTF-8 multi-byte sequences unchanged.
             * Feeding each byte to utf8_append() as its own code point
             * would re-encode a 3-byte char (e.g. 0xE5 0x8D 0x83 "千")
             * into 6 bytes (å +  + ), corrupting non-ASCII paths. */
            unsigned char u = (unsigned char)c;
            int extra;
            if (u >= 0xF0 && u <= 0xF4)      extra = 3;  /* 4-byte UTF-8 */
            else if (u >= 0xE0 && u <= 0xEF) extra = 2;  /* 3-byte UTF-8 */
            else if (u >= 0xC2 && u <= 0xDF) extra = 1;  /* 2-byte UTF-8 */
            else if (u >= 0x80) { free(buf); return NULL; }  /* stray continuation byte */
            else { utf8_append(&buf, &len, &cap, u); P->p++; continue; }
            if (P->p + extra >= P->end) { free(buf); return NULL; }
            for (int i = 1; i <= extra; i++)
                if (((unsigned char)P->p[i] & 0xC0) != 0x80) { free(buf); return NULL; }
            if (len + (size_t)extra + 1 > cap) {
                cap = cap ? cap * 2 : 32;
                while (cap < len + (size_t)extra + 1) cap *= 2;
                char *nb = (char *)realloc(buf, cap);
                if (!nb) { free(buf); return NULL; }
                buf = nb;
            }
            memcpy(buf + len, P->p, (size_t)extra + 1);
            len += (size_t)extra + 1;
            P->p += extra + 1;
        }
    }
    if (!buf) { buf = (char *)malloc(1); cap = 1; }
    buf[len] = '\0';
    return buf;
}

static VJson *parse_value(VJsonParser *P);

static VJson *parse_object(VJsonParser *P) {
    VJson *obj = vjson_new_object();
    P->p++; /* consume '{' */
    skip_ws(P);
    if (P->p < P->end && *P->p == '}') { P->p++; return obj; }
    for (;;) {
        skip_ws(P);
        if (P->p >= P->end || *P->p != '"') { vjson_free(obj); return NULL; }
        P->p++;
        char *key = parse_string_body(P);
        if (!key) { vjson_free(obj); return NULL; }
        skip_ws(P);
        if (P->p >= P->end || *P->p != ':') { free(key); vjson_free(obj); return NULL; }
        P->p++;
        skip_ws(P);
        VJson *val = parse_value(P);
        if (!val) { free(key); vjson_free(obj); return NULL; }
        /* Keep the pair's key; object may be modified via vjson_obj_set too. */
        if (obj->u.obj.n == obj->u.obj.cap) {
            obj->u.obj.cap = obj->u.obj.cap ? obj->u.obj.cap * 2 : 8;
            obj->u.obj.pairs = (VJsonPair *)realloc(obj->u.obj.pairs, obj->u.obj.cap * sizeof(VJsonPair));
        }
        obj->u.obj.pairs[obj->u.obj.n].key = key;
        obj->u.obj.pairs[obj->u.obj.n].val = val;
        obj->u.obj.n++;
        skip_ws(P);
        if (P->p >= P->end) { vjson_free(obj); return NULL; }
        if (*P->p == '}') { P->p++; return obj; }
        if (*P->p != ',') { vjson_free(obj); return NULL; }
        P->p++;
    }
}

static VJson *parse_array(VJsonParser *P) {
    VJson *arr = vjson_new_array();
    P->p++; /* consume '[' */
    skip_ws(P);
    if (P->p < P->end && *P->p == ']') { P->p++; return arr; }
    for (;;) {
        skip_ws(P);
        VJson *val = parse_value(P);
        if (!val) { vjson_free(arr); return NULL; }
        vjson_array_push(arr, val);
        skip_ws(P);
        if (P->p >= P->end) { vjson_free(arr); return NULL; }
        if (*P->p == ']') { P->p++; return arr; }
        if (*P->p != ',') { vjson_free(arr); return NULL; }
        P->p++;
    }
}

static VJson *parse_value(VJsonParser *P) {
    if (P->depth > 64) return NULL;   /* recursion guard */
    skip_ws(P);
    if (P->p >= P->end) return NULL;
    char c = *P->p;
    if (c == '{') { P->depth++; VJson *v = parse_object(P); P->depth--; return v; }
    if (c == '[') { P->depth++; VJson *v = parse_array(P); P->depth--; return v; }
    if (c == '"') { P->p++; char *s = parse_string_body(P); if (!s) return NULL; return vjson_new_string(s); }
    if (c == 't') { if (P->end - P->p >= 4 && memcmp(P->p, "true", 4) == 0) { P->p += 4; return vjson_new_bool(1); } return NULL; }
    if (c == 'f') { if (P->end - P->p >= 5 && memcmp(P->p, "false", 5) == 0) { P->p += 5; return vjson_new_bool(0); } return NULL; }
    if (c == 'n') { if (P->end - P->p >= 4 && memcmp(P->p, "null", 4) == 0) { P->p += 4; return vjson_new_null(); } return NULL; }
    /* number */
    {
        char *endp = NULL;
        double d = strtod(P->p, &endp);
        if (endp == P->p) return NULL;
        P->p = endp;
        return vjson_new_number(d);
    }
}

VJson *vjson_parse(const char *text) {
    if (!text) return NULL;
    VJsonParser P = { text, text + strlen(text), 0 };
    skip_ws(&P);
    VJson *v = parse_value(&P);
    if (!v) return NULL;
    skip_ws(&P);
    if (P.p != P.end) { vjson_free(v); return NULL; }
    return v;
}

/* ---------------- builders / serialization ---------------- */

VJson *vjson_new_string(const char *s) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_STRING;
    v->u.str = s ? strdup(s) : strdup("");
    return v;
}
VJson *vjson_new_number(double n) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_NUMBER;
    v->u.number = n;
    return v;
}
VJson *vjson_new_bool(int b) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_BOOL;
    v->u.boolean = b ? 1 : 0;
    return v;
}
VJson *vjson_new_object(void) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_OBJECT;
    return v;
}
VJson *vjson_new_null(void) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_NULL;
    return v;
}
VJson *vjson_new_array(void) {
    VJson *v = (VJson *)calloc(1, sizeof(VJson));
    v->type = VJ_ARRAY;
    return v;
}

void vjson_obj_set(VJson *obj, const char *key, VJson *val) {
    if (!obj || obj->type != VJ_OBJECT || !key || !val) { if (val) vjson_free(val); return; }
    /* Replace existing key if present. */
    for (size_t i = 0; i < obj->u.obj.n; i++) {
        if (strcmp(obj->u.obj.pairs[i].key, key) == 0) {
            vjson_free(obj->u.obj.pairs[i].val);
            obj->u.obj.pairs[i].val = val;
            return;
        }
    }
    if (obj->u.obj.n == obj->u.obj.cap) {
        obj->u.obj.cap = obj->u.obj.cap ? obj->u.obj.cap * 2 : 8;
        obj->u.obj.pairs = (VJsonPair *)realloc(obj->u.obj.pairs, obj->u.obj.cap * sizeof(VJsonPair));
    }
    obj->u.obj.pairs[obj->u.obj.n].key = strdup(key);
    obj->u.obj.pairs[obj->u.obj.n].val = val;
    obj->u.obj.n++;
}

void vjson_array_push(VJson *arr, VJson *val) {
    if (!arr || arr->type != VJ_ARRAY || !val) { if (val) vjson_free(val); return; }
    if (arr->u.arr.n == arr->u.arr.cap) {
        arr->u.arr.cap = arr->u.arr.cap ? arr->u.arr.cap * 2 : 8;
        arr->u.arr.items = (VJson **)realloc(arr->u.arr.items, arr->u.arr.cap * sizeof(VJson *));
    }
    arr->u.arr.items[arr->u.arr.n++] = val;
}

VJson *vjson_clone(const VJson *v) {
    if (!v) return NULL;
    switch (v->type) {
    case VJ_NULL:   return vjson_new_null();
    case VJ_BOOL:   return vjson_new_bool(v->u.boolean);
    case VJ_NUMBER: return vjson_new_number(v->u.number);
    case VJ_STRING: return vjson_new_string(v->u.str);
    case VJ_ARRAY: {
        VJson *a = vjson_new_array();
        for (size_t i = 0; i < v->u.arr.n; i++)
            vjson_array_push(a, vjson_clone(v->u.arr.items[i]));
        return a;
    }
    case VJ_OBJECT: {
        VJson *o = vjson_new_object();
        for (size_t i = 0; i < v->u.obj.n; i++)
            vjson_obj_set(o, v->u.obj.pairs[i].key,
                          vjson_clone(v->u.obj.pairs[i].val));
        return o;
    }
    }
    return NULL;
}

void vjson_free(VJson *v) {
    if (!v) return;
    switch (v->type) {
    case VJ_STRING: free(v->u.str); break;
    case VJ_ARRAY:
        for (size_t i = 0; i < v->u.arr.n; i++) vjson_free(v->u.arr.items[i]);
        free(v->u.arr.items);
        break;
    case VJ_OBJECT:
        for (size_t i = 0; i < v->u.obj.n; i++) {
            free(v->u.obj.pairs[i].key);
            vjson_free(v->u.obj.pairs[i].val);
        }
        free(v->u.obj.pairs);
        break;
    default: break;
    }
    free(v);
}

VJson *vjson_obj_get(const VJson *obj, const char *key) {
    if (!obj || obj->type != VJ_OBJECT || !key) return NULL;
    for (size_t i = 0; i < obj->u.obj.n; i++)
        if (strcmp(obj->u.obj.pairs[i].key, key) == 0) return obj->u.obj.pairs[i].val;
    return NULL;
}
const char *vjson_str(const VJson *v) { return (v && v->type == VJ_STRING) ? v->u.str : NULL; }
double vjson_num(const VJson *v) { return (v && v->type == VJ_NUMBER) ? v->u.number : 0.0; }
int vjson_bool(const VJson *v) { return (v && v->type == VJ_BOOL) ? v->u.boolean : 0; }
size_t vjson_array_len(const VJson *v) { return (v && v->type == VJ_ARRAY) ? v->u.arr.n : 0; }
VJson *vjson_array_get(const VJson *v, size_t i) {
    if (!v || v->type != VJ_ARRAY || i >= v->u.arr.n) return NULL;
    return v->u.arr.items[i];
}

static void serialize_str(const char *s, char *buf, size_t cap, size_t *len) {
    if (*len + 1 >= cap) return;
    buf[(*len)++] = '"';
    for (const unsigned char *p = (const unsigned char *)s; *p && *len + 8 < cap; p++) {
        switch (*p) {
        case '"': buf[(*len)++] = '\\'; buf[(*len)++] = '"'; break;
        case '\\': buf[(*len)++] = '\\'; buf[(*len)++] = '\\'; break;
        case '\n': buf[(*len)++] = '\\'; buf[(*len)++] = 'n'; break;
        case '\r': buf[(*len)++] = '\\'; buf[(*len)++] = 'r'; break;
        case '\t': buf[(*len)++] = '\\'; buf[(*len)++] = 't'; break;
        case '\b': buf[(*len)++] = '\\'; buf[(*len)++] = 'b'; break;
        case '\f': buf[(*len)++] = '\\'; buf[(*len)++] = 'f'; break;
        default:
            if (*p < 0x20) {
                buf[(*len)++] = '\\'; buf[(*len)++] = 'u';
                buf[(*len)++] = '0'; buf[(*len)++] = '0';
                buf[(*len)++] = "0123456789abcdef"[(*p >> 4) & 0xF];
                buf[(*len)++] = "0123456789abcdef"[*p & 0xF];
            } else {
                buf[(*len)++] = (char)*p;
            }
        }
    }
    if (*len + 1 < cap) buf[(*len)++] = '"';
}

static void serialize_value(const VJson *v, char *buf, size_t cap, size_t *len) {
    if (!v || *len + 1 >= cap) return;
    switch (v->type) {
    case VJ_NULL: {
        const char *s = "null"; size_t n = 4;
        if (*len + n < cap) { memcpy(buf + *len, s, n); *len += n; }
        break;
    }
    case VJ_BOOL: {
        const char *s = v->u.boolean ? "true" : "false"; size_t n = v->u.boolean ? 4 : 5;
        if (*len + n < cap) { memcpy(buf + *len, s, n); *len += n; }
        break;
    }
    case VJ_NUMBER: {
        char tmp[48];
        snprintf(tmp, sizeof(tmp), "%.17g", v->u.number);
        size_t n = strlen(tmp);
        if (*len + n < cap) { memcpy(buf + *len, tmp, n); *len += n; }
        break;
    }
    case VJ_STRING:
        serialize_str(v->u.str, buf, cap, len);
        break;
    case VJ_ARRAY: {
        if (*len + 1 >= cap) return;
        buf[(*len)++] = '[';
        for (size_t i = 0; i < v->u.arr.n; i++) {
            if (i) { if (*len + 1 >= cap) return; buf[(*len)++] = ','; }
            serialize_value(v->u.arr.items[i], buf, cap, len);
        }
        if (*len + 1 >= cap) return;
        buf[(*len)++] = ']';
        break;
    }
    case VJ_OBJECT: {
        if (*len + 1 >= cap) return;
        buf[(*len)++] = '{';
        for (size_t i = 0; i < v->u.obj.n; i++) {
            if (i) { if (*len + 1 >= cap) return; buf[(*len)++] = ','; }
            serialize_str(v->u.obj.pairs[i].key, buf, cap, len);
            if (*len + 1 >= cap) return;
            buf[(*len)++] = ':';
            serialize_value(v->u.obj.pairs[i].val, buf, cap, len);
        }
        if (*len + 1 >= cap) return;
        buf[(*len)++] = '}';
        break;
    }
    }
}

size_t vjson_serialize(const VJson *v, char *buf, size_t cap) {
    if (!v || !buf || cap == 0) return 0;
    size_t len = 0;
    serialize_value(v, buf, cap, &len);
    if (len + 1 >= cap) return 0;
    buf[len] = '\0';
    return len;
}

/* ================================================================
 * HTTP server (multi-threaded)
 *
 * One accept thread pushes accepted connections onto a bounded queue;
 * worker threads pull connections, parse the request, run the handler
 * and write the response. Inference-critical sections are serialized by
 * the route layer (vllm_server) with its own lock, so workers only
 * contend there - metadata/error endpoints run concurrently.
 *
 * Robustness guards:
 *   - bounded connection queue (flood -> connection refused, not stall)
 *   - SO_RCVTIMEO on every accepted socket (slow/half-open clients)
 *   - per-connection request buffers (no shared static state)
 * ================================================================ */

/* Upper bound for a request body (base64 image / video-frame uploads are
 * the largest payloads). Larger bodies are refused to keep a single client
 * from exhausting RAM. */
#define VHTTP_MAX_BODY (64u << 20)

struct VHttpConn {
    int fd;
    char rbuf[1 << 16];     /* 64 KiB request-header buffer */
    char *body;             /* dynamic request body (NULL = none) */
    size_t body_len;
    size_t rlen;
    char method_buf[16];    /* parsed request line (per-connection) */
    char path_buf[2048];
    char hdr_buf[1024];     /* header value scratch (per-connection) */
};

static int vnet_recv(VHttpConn *c, char *buf, size_t len) {
    return (int)recv(c->fd, buf, len, 0);
}
static int vnet_send(VHttpConn *c, const char *buf, size_t len) {
    if (len == 0) return 0;
    int r = (int)send(c->fd, buf, len, 0);
    return r;
}
static void vnet_close(VHttpConn *c) {
    close(c->fd);
    if (g_vhttp_conns > 0) g_vhttp_conns--;
    free(c->body);
    free(c);
}

/* Set a receive timeout so a stalled client cannot pin a worker forever. */
static void vnet_set_recv_timeout(VHttpConn *c, int seconds) {
    struct timeval tv;
    tv.tv_sec = seconds; tv.tv_usec = 0;
    setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

/* Case-insensitive header lookup over the raw request. Writes the value
 * into out[outsz] ("" if absent). */
static void find_header(const char *raw, const char *name,
                        char *out, size_t outsz) {
    const char *p = raw;
    size_t nlen = strlen(name);
    out[0] = '\0';
    while (*p) {
        const char *line = p;
        const char *eol = NULL;
        for (const char *q = p; *q; q++) {
            if (q[0] == '\r' && q[1] == '\n') { eol = q; break; }
        }
        if (!eol) break;
        if (eol == p) break;   /* end of headers */
        if (strncasecmp(line, name, nlen) == 0 && line[nlen] == ':') {
            const char *v = line + nlen + 1;
            while (*v == ' ' || *v == '\t') v++;
            size_t n = (size_t)(eol - v);
            if (n >= outsz) n = outsz - 1;
            memcpy(out, v, n);
            out[n] = '\0';
            while (n > 0 && (out[n-1] == ' ' || out[n-1] == '\t')) out[--n] = '\0';
            return;
        }
        p = eol + 2;
    }
}

const char *vhttp_req_header(const VHttpRequest *r, const char *name) {
    /* The route layer does not use headers today; kept for completeness.
     * Callers must supply their own scratch buffer if needed. */
    (void)r; (void)name;
    return "";
}

/* Read one HTTP request into conn->rbuf. Returns 0 on success. */
static int read_request(VHttpConn *c, VHttpRequest *req) {
    /* Read until end of headers. */
    size_t hdr_end = 0;
    for (;;) {
        if (c->rlen >= 4) {
            const char *found = NULL;
            for (size_t i = 0; i + 3 < c->rlen; i++) {
                if (c->rbuf[i] == '\r' && c->rbuf[i+1] == '\n' &&
                    c->rbuf[i+2] == '\r' && c->rbuf[i+3] == '\n') {
                    found = c->rbuf + i;
                    break;
                }
            }
            if (found) { hdr_end = (size_t)(found - c->rbuf) + 4; break; }
        }
        if (c->rlen >= sizeof(c->rbuf)) return -1;
        int n = vnet_recv(c, c->rbuf + c->rlen, sizeof(c->rbuf) - c->rlen);
        if (n <= 0) return -1;
        c->rlen += (size_t)n;
    }
    /* Parse request line WITHOUT modifying rbuf: find_header() scans the
     * intact buffer afterwards, so no NUL may be written into it. */
    char *line = c->rbuf;
    char *eol = strstr(line, "\r\n");
    if (!eol) return -1;
    char *sp1 = strchr(line, ' ');
    char *sp2 = sp1 ? strchr(sp1 + 1, ' ') : NULL;
    char *sp3 = sp2 ? strchr(sp2 + 1, '\r') : NULL;
    if (!sp1 || !sp2 || !sp3) return -1;
    size_t ml = (size_t)(sp1 - line);
    size_t pl = (size_t)(sp2 - sp1 - 1);
    size_t vl = (size_t)(sp3 - sp2 - 1);
    /* Strict request line: METHOD SP PATH SP HTTP/1.x.  Reject junk like
     * "HTTP/9.9", a missing version (which previously let the second space
     * match inside a header line and corrupted the parsed path), or an
     * unknown method that would otherwise fall through to a 200 on "/". */
    if (ml < 1 || pl < 1 || vl != 8 ||
        memcmp(sp2 + 1, "HTTP/1.", 7) != 0 ||
        (sp2[8] != '0' && sp2[8] != '1') ||
        !((ml == 3 && memcmp(line, "GET", 3) == 0) ||
          (ml == 4 && memcmp(line, "POST", 4) == 0)))
        return -1;
    if (ml >= sizeof(c->method_buf) || pl >= sizeof(c->path_buf)) return -1;
    memcpy(c->method_buf, line, ml); c->method_buf[ml] = '\0';
    memcpy(c->path_buf, sp1 + 1, pl); c->path_buf[pl] = '\0';
    req->method = c->method_buf;
    req->path = c->path_buf;
    char *qmark = strchr(c->path_buf, '?');
    if (qmark) { *qmark = '\0'; req->query = qmark + 1; }
    else req->query = "";

    /* Body: read into a heap buffer sized to the declared Content-Length.
     * The header recv loop above may have over-read into the body, so copy
     * those bytes across first. */
    find_header(c->rbuf, "Content-Length", c->hdr_buf, sizeof(c->hdr_buf));
    size_t body_len = c->hdr_buf[0] ? (size_t)strtoul(c->hdr_buf, NULL, 10) : 0;
    if (body_len > VHTTP_MAX_BODY) return -1;
    if (body_len) {
        c->body = (char *)malloc(body_len + 1);
        if (!c->body) return -1;
        c->body_len = body_len;
        size_t already = c->rlen > hdr_end ? c->rlen - hdr_end : 0;
        if (already > body_len) already = body_len;
        if (already) memcpy(c->body, c->rbuf + hdr_end, already);
        size_t got = already;
        while (got < body_len) {
            int n = vnet_recv(c, c->body + got, body_len - got);
            if (n <= 0) return -1;
            got += (size_t)n;
        }
        c->body[body_len] = '\0';
        req->body = c->body;
        req->body_len = body_len;
    } else {
        req->body = "";
        req->body_len = 0;
    }
    req->raw = c->rbuf;
    return 0;
}

/* Send a complete non-streaming response. */
static int send_response(VHttpConn *c, const VHttpResponse *resp) {
    char head[512];
    const char *ct = resp->content_type ? resp->content_type : "text/plain";
    size_t blen = resp->body_len ? resp->body_len : (resp->body ? strlen(resp->body) : 0);
    /* HTML 应用页（chat/admin/convert）每次拉取最新版本，避免浏览器缓存旧页。 */
    const char *nocache = (ct && strcmp(ct, "text/html; charset=utf-8") == 0)
                          ? "Cache-Control: no-store\r\n" : "";
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Connection: close\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "%s"
                     "\r\n",
                     resp->status,
                     resp->status == 200 ? "OK" :
                     resp->status == 400 ? "Bad Request" :
                     resp->status == 404 ? "Not Found" :
                     resp->status == 408 ? "Request Timeout" :
                     resp->status == 413 ? "Payload Too Large" :
                     resp->status == 429 ? "Too Many Requests" :
                     resp->status == 500 ? "Internal Server Error" :
                     resp->status == 503 ? "Service Unavailable" : "Error",
                     ct, blen, nocache);
    if (vnet_send(c, head, (size_t)n) < 0) return -1;
    if (blen && vnet_send(c, resp->body, blen) < 0) return -1;
    return 0;
}

int vhttp_stream_begin(VHttpConn *conn, int status, const char *content_type) {
    char head[256];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Connection: close\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "\r\n",
                     status, status == 200 ? "OK" : "Error",
                     content_type ? content_type : "text/event-stream");
    return vnet_send(conn, head, (size_t)n) < 0 ? -1 : 0;
}
int vhttp_stream_write(VHttpConn *conn, const char *data, size_t len) {
    return vnet_send(conn, data, len) < 0 ? -1 : 0;
}
int vhttp_stream_flush(VHttpConn *conn) {
    (void)conn;
    return 0;   /* sockets are unbuffered; every write reaches the wire */
}
int vhttp_stream_done(VHttpConn *conn) {
    const char *done = "data: [DONE]\n\n";
    vnet_send(conn, done, strlen(done));
    vnet_close(conn);
    return 0;
}

static void default_handler(const VHttpRequest *req, VHttpResponse *resp,
                            VHttpConn *conn, void *ud) {
    (void)req; (void)conn; (void)ud;
    static const char not_found[] =
        "{\"error\":{\"message\":\"Not Found\",\"type\":\"invalid_request_error\",\"code\":404}}";
    resp->status = 404;
    resp->content_type = "application/json";
    resp->body = not_found;
    resp->body_len = sizeof(not_found) - 1;
}

/* ---------------- worker pool ---------------- */

#include <pthread.h>
typedef pthread_mutex_t VLock;
typedef pthread_cond_t VCond;
#define V_LOCK_INIT(l)   pthread_mutex_init(l, NULL)
#define V_LOCK(l)        pthread_mutex_lock(l)
#define V_UNLOCK(l)      pthread_mutex_unlock(l)
#define V_COND_INIT(c)   pthread_cond_init(c, NULL)
#define V_COND_WAIT(c,l) pthread_cond_wait(c, l)
#define V_COND_SIGNAL(c)    pthread_cond_signal(c)
#define V_COND_BROADCAST(c) pthread_cond_broadcast(c)

typedef struct {
    VHttpConn **items;
    int head, tail, count, cap;
    VLock lock;
    VCond cond;
} ConnQueue;

typedef struct {
    VHttpHandler handler;
    void *userdata;
    ConnQueue *q;
} WorkerCtx;

static void queue_init(ConnQueue *q, int cap) {
    q->items = (VHttpConn **)calloc((size_t)cap, sizeof(VHttpConn *));
    q->head = q->tail = q->count = 0;
    q->cap = cap;
    V_LOCK_INIT(&q->lock);
    V_COND_INIT(&q->cond);
}
static void queue_push(ConnQueue *q, VHttpConn *c) {
    V_LOCK(&q->lock);
    while (q->count >= q->cap) {
        /* Queue full: drop the oldest connection to stay responsive. */
        VHttpConn *drop = q->items[q->head];
        q->head = (q->head + 1) % q->cap;
        q->count--;
        V_UNLOCK(&q->lock);
        vnet_close(drop);
        V_LOCK(&q->lock);
    }
    q->items[q->tail] = c;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    V_COND_SIGNAL(&q->cond);
    V_UNLOCK(&q->lock);
}
static VHttpConn *queue_pop(ConnQueue *q) {
    V_LOCK(&q->lock);
    while (q->count == 0)
        V_COND_WAIT(&q->cond, &q->lock);
    VHttpConn *c = q->items[q->head];
    q->head = (q->head + 1) % q->cap;
    q->count--;
    V_UNLOCK(&q->lock);
    return c;
}

/* Process one connection: parse, run handler, respond. */
static void serve_conn(WorkerCtx *ctx, VHttpConn *conn) {
    VHttpRequest req;
    memset(&req, 0, sizeof(req));
    VHttpResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.status = 200;

    if (read_request(conn, &req) != 0) {
        static const char bad400[] =
            "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n"
            "Content-Length: 11\r\nConnection: close\r\n\r\nBad Request";
        vnet_send(conn, bad400, sizeof(bad400) - 1);
        vnet_close(conn);
        return;
    }
    ctx->handler(&req, &resp, conn, ctx->userdata);
    if (resp.stream) {
        /* Handler owns the connection (vhttp_stream_done closes it). */
        free(resp.body_owned);
        return;
    }
    send_response(conn, &resp);
    vnet_close(conn);
    free(resp.body_owned);   /* per-request heap body, if any */
}

static void *worker_main(void *arg) {
    WorkerCtx *ctx = (WorkerCtx *)arg;
    for (;;) {
        VHttpConn *conn = queue_pop(ctx->q);
        serve_conn(ctx, conn);
    }
}

static void spawn_worker(WorkerCtx *arg) {
    pthread_t t;
    pthread_create(&t, NULL, worker_main, arg);
}

int vhttp_serve_ex(int port, VHttpHandler handler, void *userdata,
                   void (*on_start)(int actual_port, void *ud), int n_threads) {
    /* A client that disconnects mid-stream (e.g. curl | head -c N) makes the
     * next write() fail with EPIPE; without this ignore the default SIGPIPE
     * action would terminate the whole server process. */
    signal(SIGPIPE, SIG_IGN);
    VHttpConn listener;
    memset(&listener, 0, sizeof(listener));
    listener.fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listener.fd < 0) return -1;
    {
        int one = 1;
        setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((unsigned short)port);
    if (bind(listener.fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) return -1;
    if (listen(listener.fd, 32) != 0) return -1;

    if (on_start) {
        int actual = port;
        if (port == 0) {
            struct sockaddr_in la;
            socklen_t alen = sizeof(la);
            if (getsockname(listener.fd, (struct sockaddr *)&la, &alen) == 0)
                actual = ntohs(la.sin_port);
        }
        on_start(actual, userdata);
    }

    if (!handler) handler = default_handler;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 32) n_threads = 32;

    ConnQueue q;
    queue_init(&q, 128);
    WorkerCtx ctx;
    ctx.handler = handler;
    ctx.userdata = userdata;
    ctx.q = &q;
    for (int i = 0; i < n_threads; i++)
        spawn_worker(&ctx);

    for (;;) {
        /* Poll the graceful-stop flag so vhttp_stop() (from an /admin
         * shutdown handler) can terminate the accept loop cleanly. */
        if (g_vhttp_stop) break;
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(listener.fd, &rfds);
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        int sr = select(listener.fd + 1, &rfds, NULL, NULL, &tv);
        if (sr < 0) {
            if (g_vhttp_stop) break;
            continue;
        }
        if (sr == 0) continue;   /* timeout: re-check stop flag */
        struct sockaddr_in ca;
        socklen_t calen = sizeof(ca);
        int cf = accept(listener.fd, (struct sockaddr *)&ca, &calen);
        if (cf < 0) {
            if (g_vhttp_stop) break;
            continue;
        }
        VHttpConn *conn = (VHttpConn *)calloc(1, sizeof(VHttpConn));
        if (!conn) {
            close(cf);
            continue;
        }
        conn->fd = cf;
        vnet_set_recv_timeout(conn, 30);   /* slow-client guard */
        g_vhttp_conns++;
        queue_push(&q, conn);
    }
    return 0;
}

int vhttp_serve(int port, VHttpHandler handler, void *userdata,
                void (*on_start)(int actual_port, void *ud)) {
    return vhttp_serve_ex(port, handler, userdata, on_start, 4);
}

/* ---------------- cross-platform mutex ---------------- */

struct VHttpMutex {
    VLock lock;
};

VHttpMutex *vhttp_mutex_new(void) {
    VHttpMutex *m = (VHttpMutex *)malloc(sizeof(VHttpMutex));
    if (m) V_LOCK_INIT(&m->lock);
    return m;
}
void vhttp_mutex_free(VHttpMutex *m) { free(m); }
void vhttp_mutex_lock(VHttpMutex *m) { V_LOCK(&m->lock); }
void vhttp_mutex_unlock(VHttpMutex *m) { V_UNLOCK(&m->lock); }

const char *vhttp_json_error(const char *message, char *buf, size_t cap) {
    VJson *root = vjson_new_object();
    VJson *err = vjson_new_object();
    vjson_obj_set(err, "message", vjson_new_string(message));
    vjson_obj_set(err, "type", vjson_new_string("invalid_request_error"));
    vjson_obj_set(err, "code", vjson_new_number(0));
    vjson_obj_set(root, "error", err);
    vjson_serialize(root, buf, cap);
    vjson_free(root);
    return buf;
}

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_attest.py - vllm_kestrel 可验证推理凭证（attest）一键离线验证脚本。

用法（对归档的推理记录出证）：
    python tools/verify_attest.py \
        --proof proof.json        # 响应 JSON 中的 "attest" 对象（原样保存）
        --request req.json        # 客户端当时 POST 的原始请求体（逐字节）
        --output out.txt          # 当时收到的 assistant 输出文本（逐字节）
        --pub vllm_attest.pub     # 设备公钥文件（128 hex，与请求目录同款）
        [--quiet]                 # 只输出 PASS/FAIL 摘要

原理（schema=3，见 include/serve/vllm_attest.h；兼容校验 schema=2 旧凭证）：
    body_sha = SM3(请求体原始字节)
    digest   = SM3( VLLM-AT-3
                 || F(model_fp) || F(model_id) || F(device_id) || F(user)
                 || F(params) || F(n_prompt_tokens) || F(n_gen_tokens)
                 || F(finish) || F(ts) || F(body_sha) || F(输出文本) )
    其中 F(x) = u32be(len(x)) || x。
    schema=3 与 schema=2 的帧布局完全相同，区别只在 params 字符串：
    v2 = "t=%.4g,p=%.4g,m=%.4g,mt=%d"
    v3 = "t=%.4g,p=%.4g,m=%.4g,k=%d,mt=%d,th=%d"（k=top_k，th=enable_thinking）。
    本脚本按 proof 里的 magic 前缀（由 proof["schema"] 决定）复算；params 一律
    取 proof 中的原串，故两种版本的凭证都能验。
    校验 = SM2(digest) 验签通过 且 digest 与复算一致 且 body_sha 一致。

零第三方依赖：SM3 与 SM2 验签均为本文件内的纯 Python 实现（国密标准
GB/T 32905-2016 / GM/T 0003.2），可在任意装有 Python 3 的主机运行。
退出码：0 = PASS，1 = FAIL。
"""
import argparse
import hashlib  # 仅用于提示；SM3 用本文件实现
import json
import struct
import sys

# ================================================================
# SM3 (GB/T 32905-2016) —— 移植自 src/common/vllm_crypto.c（KAT 已验）
# ================================================================
_SM3_IV = [0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
           0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e]
_MASK = 0xffffffff


def _rotl(x, n):
    n &= 31
    return ((x << n) | (x >> (32 - n))) & _MASK


def _sm3_p0(x):
    return x ^ _rotl(x, 9) ^ _rotl(x, 17)


def _sm3_p1(x):
    return x ^ _rotl(x, 15) ^ _rotl(x, 23)


def _sm3_compress(s, block):
    w = list(struct.unpack(">16I", block))
    for j in range(16, 68):
        w.append(_sm3_p1(w[j - 16] ^ w[j - 9] ^ _rotl(w[j - 3], 15)) ^
                 _rotl(w[j - 13], 7) ^ w[j - 6])
    wp = [w[j] ^ w[j + 4] for j in range(64)]
    a, b, c, d, e, f, g, h = s
    for j in range(64):
        tj = 0x79cc4519 if j < 16 else 0x7a879d8a
        ss1 = _rotl((_rotl(a, 12) + e + _rotl(tj, j)) & _MASK, 7)
        ss2 = ss1 ^ _rotl(a, 12)
        if j < 16:
            ff = a ^ b ^ c
            gg = e ^ f ^ g
        else:
            ff = (a & b) | (a & c) | (b & c)
            gg = (e & f) | ((~e) & g)
        tt1 = (ff + d + ss2 + wp[j]) & _MASK
        tt2 = (gg + h + ss1 + w[j]) & _MASK
        d, c, b, a = c, _rotl(b, 9), a, tt1
        h, g, f, e = g, _rotl(f, 19), e, _sm3_p0(tt2)
    for i, x in enumerate((a, b, c, d, e, f, g, h)):
        s[i] ^= x


def sm3(data: bytes) -> bytes:
    s = _SM3_IV[:]
    n = len(data)
    padded = data + b"\x80"
    while len(padded) % 64 != 56:
        padded += b"\x00"
    padded += struct.pack(">Q", n * 8)
    for i in range(0, len(padded), 64):
        _sm3_compress(s, padded[i:i + 64])
    return struct.pack(">8I", *s)


# ================================================================
# SM2 椭圆曲线 (GM/T 0003.2) —— 与 vllm_crypto.c 同参数
# ================================================================
P = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF
A = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC
B = 0x28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93
N = 0xFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123
GX = 0x32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7
GY = 0xBC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0
INF = None


def _inv(x):
    return pow(x, P - 2, P)


def _ec_add(p1, p2):
    if p1 is INF:
        return p2
    if p2 is INF:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P == 0:
            return INF
        # 倍点
        lam = (3 * x1 * x1 + A) * _inv(2 * y1) % P
    else:
        lam = (y2 - y1) * _inv((x2 - x1) % P) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def _ec_mul(k, pt):
    k = k % N
    r = INF
    while k:
        if k & 1:
            r = _ec_add(r, pt)
        pt = _ec_add(pt, pt)
        k >>= 1
    return r


def sm2_verify(pub_hex, digest: bytes, sig_hex: str, user_id: bytes = b"VLLM-ATTEST-1") -> bool:
    """pub_hex: 128 hex (x||y 各 64)。sig_hex: r||s 各 64 hex。"""
    if len(sig_hex) != 128:
        return False
    if len(pub_hex) != 128:
        return False
    try:
        r = int(sig_hex[:64], 16)
        s = int(sig_hex[64:], 16)
        qx = int(pub_hex[:64], 16)
        qy = int(pub_hex[64:], 16)
    except ValueError:
        return False
    if not (1 <= r < N and 1 <= s < N):
        return False
    if not (0 <= qx < P and 0 <= qy < P):
        return False
    if (qx * qx * qx + A * qx + B - qy * qy) % P != 0:
        return False

    def be(x, nb=32):
        return x.to_bytes(nb, "big")

    entl = struct.pack(">H", len(user_id) * 8)
    za = sm3(entl + user_id + be(A) + be(B) + be(GX) + be(GY) + be(qx) + be(qy))
    e = int.from_bytes(sm3(za + digest), "big")
    t = (r + s) % N
    if t == 0:
        return False
    x1, _ = _ec_add(_ec_mul(s, (GX, GY)), _ec_mul(t, (qx, qy)))
    return (e + x1) % N == r


# ================================================================
# digest 复算（schema=3 帧规范；schema=2 旧凭证按旧 magic 复算）
# ================================================================
def frame(b):
    return struct.pack(">I", len(b)) + b


def magic_for(schema):
    """帧首 magic 前缀。schema=3 -> VLLM-AT-3；schema=2 -> VLLM-AT-2（旧凭证）。"""
    if schema == 3:
        return b"VLLM-AT-3"
    if schema == 2:
        return b"VLLM-AT-2"
    return None


def recompute_digest(proof, body_sha_hex, text_bytes):
    def f(s):
        return frame(s.encode("utf-8"))

    magic = magic_for(proof.get("schema"))
    if magic is None:
        return None
    digest = sm3(
        magic
        + f(proof["model_fp"])
        + f(proof.get("model", ""))
        + f(proof.get("device", ""))
        + f(proof.get("user", ""))
        + f(proof.get("params", ""))
        + f(str(proof.get("prompt_tokens", 0)))
        + f(str(proof.get("n_tokens", 0)))
        + f(proof.get("finish", "stop"))
        + f(str(proof.get("ts", 0)))
        + f(body_sha_hex)
        + frame(text_bytes)
    )
    return digest.hex()


# ================================================================
def main():
    ap = argparse.ArgumentParser(description="vllm_kestrel attest 凭证离线验证")
    ap.add_argument("--proof", help="响应中的 attest JSON 对象文件")
    ap.add_argument("--request", help="POST 的原始请求体文件（逐字节）")
    ap.add_argument("--output", help="assistant 输出文本文件（逐字节）")
    ap.add_argument("--pub", help="设备公钥文件（128 hex）")
    ap.add_argument("--pub-hex", help="或直接给公钥 hex")
    ap.add_argument("--selftest", action="store_true", help="SM3/SM2 原语自检")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = True
        # SM3 KAT：SM3("abc")
        v = sm3(b"abc").hex()
        if v != "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0":
            print("[VERIFY] SM3 KAT        : FAIL"); ok = False
        else:
            print("[VERIFY] SM3 KAT        : PASS")
        # EC：G 在曲线上；n*G == 无穷远
        lhs = (GX * GX * GX + A * GX + B - GY * GY) % P
        if lhs != 0:
            print("[VERIFY] G on-curve     : FAIL"); ok = False
        else:
            print("[VERIFY] G on-curve     : PASS")
        if _ec_mul(N, (GX, GY)) is not INF:
            print("[VERIFY] n*G == INF     : FAIL"); ok = False
        else:
            print("[VERIFY] n*G == INF     : PASS")
        print("[VERIFY] selftest       : %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)

    pub_hex = args.pub_hex
    if not pub_hex and args.pub:
        with open(args.pub, "r", encoding="ascii") as fp:
            pub_hex = fp.read().strip()
    if not pub_hex:
        print("FAIL: 需要 --pub 或 --pub-hex")
        sys.exit(1)
    if not (args.proof and args.request and args.output):
        print("FAIL: 需要 --proof --request --output")
        sys.exit(1)

    with open(args.proof, "r", encoding="utf-8-sig") as fp:
        proof = json.load(fp)
    with open(args.request, "rb") as fp:
        req_bytes = fp.read()
    with open(args.output, "rb") as fp:
        out_bytes = fp.read()

    body_sha = sm3(req_bytes).hex()

    checks = []

    def chk(name, cond, extra=""):
        checks.append((name, bool(cond)))
        if not args.quiet:
            print("[VERIFY] %-16s : %s %s" % (name, "PASS" if cond else "FAIL", extra))

    chk("body_sha 复算", body_sha == proof.get("body_sha", ""),
        "(proof=%s req=%s)" % (proof.get("body_sha", "")[:16], body_sha[:16]))
    chk("schema", proof.get("schema") in (2, 3),
        "schema=%s (支持 2/3)" % proof.get("schema"))

    want = proof.get("digest", "")
    got = recompute_digest(proof, body_sha, out_bytes)
    chk("digest 复算", got is not None and got == want,
        "(want=%s got=%s)" % (want[:16], (got or "-")[:16]))

    sig_ok = bool(got) and sm2_verify(pub_hex, bytes.fromhex(got),
                                      proof.get("signature", ""))
    chk("SM2 验签", sig_ok, "(pub=%s...)" % pub_hex[:16])

    all_ok = all(c for _, c in checks)
    print("RESULT: %s" % ("PASS" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

/**
 * vllm_superpos.c - SHS Superposition Operations Implementation
 * 
 * Implements the axiom registry operations on set-states.
 * All operations follow the SHS (Superposition Hybrid System) semantics
 * defined in axiom_registry.json.
 */

#include "vllm_superpos.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ================================================================
 * Internal Helpers
 * ================================================================ */

static uint32_t mod_pow(uint32_t base, int64_t exp, uint32_t modulus) {
    uint64_t result = 1;
    uint64_t b = base % modulus;
    int64_t e = exp;
    while (e > 0) {
        if (e & 1) result = (result * b) % modulus;
        b = (b * b) % modulus;
        e >>= 1;
    }
    return (uint32_t)result;
}

void setstate_grow(SHS_SetState *S) {
    if (S->count >= S->capacity) {
        size_t new_cap = S->capacity ? S->capacity * 2 : 16;
        S->states = realloc(S->states, new_cap * sizeof(SHS_State));
        S->capacity = new_cap;
    }
}

void setstate_add_element(SHS_SetState *S, uint32_t value,
                                  int64_t exponent, float amplitude) {
    setstate_grow(S);
    S->states[S->count].value = value;
    S->states[S->count].exponent = exponent;
    S->states[S->count].amplitude = amplitude;
    S->count++;
}

/* Remove duplicate (value, exponent) entries, keep higher amplitude */
void setstate_dedup(SHS_SetState *S) {
    if (S->count <= 1) return;
    for (size_t i = 0; i < S->count; i++) {
        for (size_t j = i + 1; j < S->count; j++) {
            if (S->states[i].value == S->states[j].value &&
                S->states[i].exponent == S->states[j].exponent) {
                /* Merge: keep higher amplitude */
                if (S->states[j].amplitude > S->states[i].amplitude) {
                    S->states[i].amplitude = S->states[j].amplitude;
                }
                /* Remove j by swapping with last */
                S->states[j] = S->states[S->count - 1];
                S->count--;
                j--;
            }
        }
    }
}

void setstate_init(SHS_SetState *S, uint32_t modulus) {
    S->states = NULL;
    S->count = 0;
    S->capacity = 0;
    S->modulus = modulus;
}

void setstate_free(SHS_SetState *S) {
    free(S->states);
    S->states = NULL;
    S->count = S->capacity = 0;
}

void setstate_copy(SHS_SetState *dst, const SHS_SetState *src) {
    setstate_free(dst);
    dst->modulus = src->modulus;
    dst->capacity = src->count;
    dst->count = src->count;
    dst->states = malloc(dst->capacity * sizeof(SHS_State));
    memcpy(dst->states, src->states, src->count * sizeof(SHS_State));
}

/* ================================================================
 * existence_axiom (id: existence_axiom_a03280c0)
 * S^(1) = {a mod N}
 * ================================================================ */
SHS_SetState shs_existence(uint32_t a, uint32_t modulus) {
    SHS_SetState S;
    setstate_init(&S, modulus);
    setstate_add_element(&S, a % modulus, 0, 1.0f);
    return S;
}

/* ================================================================
 * exponential_jump_set (id: exponential_jump_set_259264ad)
 * S' = { (v * base^k) mod N | v in S, k in K }
 * ================================================================ */
void shs_exponential_jump_set(SHS_SetState *S, uint32_t base,
                               const int64_t *jumps, size_t num_jumps) {
    size_t orig_count = S->count;
    SHS_State *orig = malloc(orig_count * sizeof(SHS_State));
    memcpy(orig, S->states, orig_count * sizeof(SHS_State));

    S->count = 0; /* reset, will re-add */

    for (size_t i = 0; i < orig_count; i++) {
        for (size_t k = 0; k < num_jumps; k++) {
            uint64_t prod = ((uint64_t)orig[i].value *
                             mod_pow(base, jumps[k], S->modulus)) % S->modulus;
            int64_t new_exp = orig[i].exponent + jumps[k];
            setstate_add_element(S, (uint32_t)prod, new_exp,
                                 orig[i].amplitude * 0.95f);
        }
    }
    free(orig);
    setstate_dedup(S);
}

/* ================================================================
 * superposition_addition_clear (id: superposition_addition_clear_e6cf6fd3)
 * (1,1) -> {2, 3}  (special case)
 * Otherwise: (x,y) -> {(x+y) mod N}
 * ================================================================ */
void shs_superposition_add(SHS_SetState *result, uint32_t x, uint32_t y,
                            uint32_t modulus) {
    setstate_init(result, modulus);
    if (x == 1 && y == 1) {
        /* Axiom special case: 1+1 produces {2, 3} superposition */
        setstate_add_element(result, 2 % modulus, 1, 0.7071f);
        setstate_add_element(result, 3 % modulus, 1, 0.7071f);
    } else {
        uint32_t sum = (x + y) % modulus;
        setstate_add_element(result, sum, 1, 1.0f);
        /* Non-classical: if both odd, add interference term */
        if ((x & 1) && (y & 1) && !(x == 1 && y == 1)) {
            uint32_t extra = (x + y + 1) % modulus;
            setstate_add_element(result, extra, 2, 0.3f);
        }
    }
}

/* ================================================================
 * general_superposition_addition (id: general_superposition_addition_001)
 * If both singletons: apply superposition_addition_clear
 * If X is set, Y is single: element-wise
 * If both are sets: Cartesian product union
 * ================================================================ */
void shs_general_superposition_add(SHS_SetState *result,
                                    const SHS_SetState *X,
                                    const SHS_SetState *Y) {
    setstate_init(result, X->modulus);

    /* Both singletons */
    if (X->count == 1 && Y->count == 1) {
        uint32_t x = X->states[0].value;
        uint32_t y = Y->states[0].value;
        if (x == 1 && y == 1) {
            setstate_add_element(result, 2 % X->modulus, 1, 0.7071f);
            setstate_add_element(result, 3 % X->modulus, 1, 0.7071f);
        } else {
            uint32_t sum = (x + y) % X->modulus;
            setstate_add_element(result, sum, 1, 1.0f);
            if ((x & 1) && (y & 1) && !(x == 1 && y == 1)) {
                setstate_add_element(result, (x + y + 1) % X->modulus, 2, 0.3f);
            }
        }
        return;
    }

    /* Cartesian product for set × set or set × singleton */
    for (size_t i = 0; i < X->count; i++) {
        for (size_t j = 0; j < Y->count; j++) {
            /* Cap to prevent exponential blow-up */
            if (result->count >= SHS_MAX_SUPERPOSITION_SIZE) goto done;

            uint32_t sum = (X->states[i].value + Y->states[j].value) % X->modulus;
            int64_t new_exp = X->states[i].exponent + Y->states[j].exponent;
            float amp = X->states[i].amplitude * Y->states[j].amplitude;
            setstate_add_element(result, sum, new_exp, amp);

            /* Interference: both odd → extra term */
            if ((X->states[i].value & 1) && (Y->states[j].value & 1)) {
                if (result->count >= SHS_MAX_SUPERPOSITION_SIZE) goto done;
                uint32_t inter = (X->states[i].value + Y->states[j].value + 1) %
                                 X->modulus;
                setstate_add_element(result, inter, new_exp + 1, amp * 0.3f);
            }
        }
    }
done:
    setstate_dedup(result);
}

/* ================================================================
 * odd_even_interference_logic (id: odd_even_interference_logic_85b52caf)
 * If (e1%2) != (e2%2): keep state with smaller exponent
 * If (e1%2) == (e2%2): keep both
 * ================================================================ */
void shs_odd_even_interference(SHS_SetState *S) {
    if (S->count <= 1) return;

    bool *keep = calloc(S->count, sizeof(bool));
    for (size_t i = 0; i < S->count; i++) keep[i] = true;

    for (size_t i = 0; i < S->count; i++) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < S->count; j++) {
            if (!keep[j]) continue;
            if (S->states[i].value != S->states[j].value) continue;

            int parity_i = S->states[i].exponent & 1;
            int parity_j = S->states[j].exponent & 1;

            if (parity_i != parity_j) {
                /* Destructive interference: keep smaller exponent */
                if (S->states[i].exponent < S->states[j].exponent) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break; /* i eliminated, next i */
                }
            }
        }
    }

    /* Compact kept states */
    size_t new_count = 0;
    for (size_t i = 0; i < S->count; i++) {
        if (keep[i]) {
            if (new_count != i) {
                S->states[new_count] = S->states[i];
            }
            new_count++;
        }
    }
    S->count = new_count;
    free(keep);
}

/* ================================================================
 * collapse_priority_rule (id: collapse_priority_rule_e16488e2)
 * Priority:
 *   1. Singleton {v} → return v
 *   2. 1 ∈ S → return 1
 *   3. size(S) > 1 & 1 ∈ S → return 1
 *   4. Otherwise → no collapse
 * ================================================================ */
bool shs_collapse_priority(SHS_SetState *S, uint32_t *result) {
    if (S->count == 0) return false;

    /* Priority 1: singleton */
    if (S->count == 1) {
        *result = S->states[0].value;
        return true;
    }

    /* Priority 2 & 3: contains 1 */
    for (size_t i = 0; i < S->count; i++) {
        if (S->states[i].value == 1) {
            *result = 1;
            S->count = 1;
            S->states[0].value = 1;
            S->states[0].exponent = 0;
            S->states[0].amplitude = 1.0f;
            return true;
        }
    }

    return false;
}

/* ================================================================
 * value_mod_interference (id: axiom_value_mod_interf_001)
 * If (v1 mod m) == (v2 mod m): keep state with smaller exponent
 * ================================================================ */
void shs_value_mod_interference(SHS_SetState *S, uint32_t modulus_m) {
    if (S->count <= 1) return;

    bool *keep = calloc(S->count, sizeof(bool));
    for (size_t i = 0; i < S->count; i++) keep[i] = true;

    for (size_t i = 0; i < S->count; i++) {
        if (!keep[i]) continue;
        for (size_t j = i + 1; j < S->count; j++) {
            if (!keep[j]) continue;
            if ((S->states[i].value % modulus_m) ==
                (S->states[j].value % modulus_m)) {
                if (S->states[i].exponent <= S->states[j].exponent) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }

    size_t new_count = 0;
    for (size_t i = 0; i < S->count; i++) {
        if (keep[i]) S->states[new_count++] = S->states[i];
    }
    S->count = new_count;
    free(keep);
}

/* ================================================================
 * tensor_iterated_addition (id: tensor_iterated_addition_e02cfff4)
 * S' = { (x ⊕ a) mod N | x in S }
 * ================================================================ */
void shs_tensor_iterated_add(SHS_SetState *S, uint32_t element,
                              uint32_t modulus) {
    for (size_t i = 0; i < S->count; i++) {
        uint32_t x = S->states[i].value;
        if (x == 1 && element == 1) {
            /* superposition_addition_clear special case: 1⊕1 = {2,3} */
            S->states[i].value = 2 % modulus;
            /* Insert 3 into the set */
            setstate_add_element(S, 3 % modulus,
                                 S->states[i].exponent, 0.7071f);
        } else {
            S->states[i].value = (x + element) % modulus;
            S->states[i].exponent += 1;
        }
    }
    S->modulus = modulus;
    setstate_dedup(S);
}

/* ================================================================
 * nonclassical_multiplication: ⊗_nc (id: _2297_nc_6fec81b9)
 * X ⊗_nc Y = { (x*y) mod N | x in X, y in Y }
 * Interference: if (x1*y1 == x2*y2) but (x1,y1) != (x2,y2), add
 *   ((x1*y1 + x2*y2)/2) mod N as interference term
 * ================================================================ */
void shs_nc_multiplication(SHS_SetState *result,
                            const SHS_SetState *X,
                            const SHS_SetState *Y) {
    setstate_init(result, X->modulus);

    /* Both singletons → classic multiplication */
    if (X->count == 1 && Y->count == 1) {
        uint64_t prod = ((uint64_t)X->states[0].value *
                         Y->states[0].value) % X->modulus;
        int64_t exp = X->states[0].exponent + Y->states[0].exponent;
        float amp = X->states[0].amplitude * Y->states[0].amplitude;
        setstate_add_element(result, (uint32_t)prod, exp, amp);
        return;
    }

    /* Cartesian product */
    for (size_t i = 0; i < X->count; i++) {
        for (size_t j = 0; j < Y->count; j++) {
            uint64_t prod = ((uint64_t)X->states[i].value *
                             Y->states[j].value) % X->modulus;
            int64_t exp = X->states[i].exponent + Y->states[j].exponent;
            float amp = X->states[i].amplitude * Y->states[j].amplitude;
            setstate_add_element(result, (uint32_t)prod, exp, amp);
        }
    }

    /* Interference detection: simplified for performance */
    setstate_dedup(result);

    /* Non-classical interference: only for small sets */
    if (X->count * Y->count <= 256) {
        for (size_t i = 0; i < X->count; i++) {
            for (size_t j = i + 1; j < X->count; j++) {
                for (size_t k = 0; k < Y->count; k++) {
                    for (size_t l = k + 1; l < Y->count; l++) {
                        uint64_t prod1 = ((uint64_t)X->states[i].value *
                                          Y->states[k].value) % X->modulus;
                        uint64_t prod2 = ((uint64_t)X->states[j].value *
                                          Y->states[l].value) % X->modulus;
                        if (prod1 == prod2) {
                            uint32_t inter = (uint32_t)((prod1 + prod2) / 2) %
                                             X->modulus;
                            int64_t exp = (X->states[i].exponent +
                                           X->states[j].exponent +
                                           Y->states[k].exponent +
                                           Y->states[l].exponent) / 2;
                            setstate_add_element(result, inter, exp, 0.5f);
                        }
                    }
                }
            }
        }
    }
}

/* ================================================================
 * selective_interference_for_compression
 * (id: selective_interference_for_compression_8ecadfb1)
 * ================================================================ */
void shs_selective_interference_compress(SHS_SetState *S) {
    if (S->count <= 1) return;

    /* Apply odd_even interference */
    shs_odd_even_interference(S);

    /* Apply value_mod interference with secondary modulus */
    shs_value_mod_interference(S, SHS_MODULUS_M);

    /* Dedup after interference */
    setstate_dedup(S);
}

/* ================================================================
 * NTT Transform Support (from ntt_isomorphism & modal_transition axioms)
 * ================================================================ */

static uint32_t ntt_pow_mod(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t res = 1, b = base % mod;
    uint32_t e = exp;
    while (e) {
        if (e & 1) res = (res * b) % mod;
        b = (b * b) % mod;
        e >>= 1;
    }
    return (uint32_t)res;
}

/**
 * Find primitive n-th root of unity modulo prime q.
 * Uses axiom: parameter_declaration for modulus q (id: parameter_declaration_33fccb0e)
 */
static uint32_t find_primitive_root(uint32_t n, uint32_t q) {
    uint32_t phi = q - 1;
    if (phi % n != 0) return 0;

    uint32_t g = 2;
    for (; g < q; g++) {
        bool ok = true;
        uint32_t temp = phi;
        for (uint32_t p = 2; p * p <= temp; p++) {
            if (temp % p == 0) {
                if (ntt_pow_mod(g, phi / p, q) == 1) { ok = false; break; }
                while (temp % p == 0) temp /= p;
            }
        }
        if (temp > 1 && ntt_pow_mod(g, phi / temp, q) == 1) ok = false;
        if (ok) break;
    }
    return ntt_pow_mod(g, phi / n, q);
}

/**
 * Bit-reverse permutation for NTT
 */
static void bit_reverse(uint32_t *data, size_t n) {
    for (size_t i = 1, j = 0; i < n; i++) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            uint32_t tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

/**
 * shs_ntt_forward: NTT transform
 * Maps to: axiom_isomorphic_superposition_004
 * Superposition state via NTT, convolution → pointwise multiplication
 */
void shs_ntt_forward(uint32_t *data, size_t n, uint32_t modulus,
                      uint32_t primitive_root) {
    if (n <= 1) return;

    bit_reverse(data, n);

    for (size_t len = 2; len <= n; len <<= 1) {
        uint32_t wlen = ntt_pow_mod(primitive_root, (uint32_t)(n / len), modulus);
        size_t half = len / 2;

        for (size_t i = 0; i < n; i += len) {
            uint32_t w = 1;
            for (size_t j = 0; j < half; j++) {
                uint32_t u = data[i + j];
                uint32_t v = (uint32_t)(((uint64_t)data[i + j + half] * w) % modulus);
                data[i + j] = (u + v) % modulus;
                data[i + j + half] = (u + modulus - v) % modulus;
                w = (uint32_t)(((uint64_t)w * wlen) % modulus);
            }
        }
    }
}

/**
 * shs_ntt_inverse: inverse NTT
 * Maps to: axiom_modal_transition_conservation_005
 * iNTT(NTT(x)) = x (bijection)
 */
void shs_ntt_inverse(uint32_t *data, size_t n, uint32_t modulus,
                      uint32_t primitive_root) {
    if (n <= 1) return;

    /* Inverse uses root^{-1} = root^{modulus-2} */
    uint32_t inv_root = ntt_pow_mod(primitive_root, modulus - 2, modulus);

    bit_reverse(data, n);

    for (size_t len = 2; len <= n; len <<= 1) {
        uint32_t wlen = ntt_pow_mod(inv_root, (uint32_t)(n / len), modulus);
        size_t half = len / 2;

        for (size_t i = 0; i < n; i += len) {
            uint32_t w = 1;
            for (size_t j = 0; j < half; j++) {
                uint32_t u = data[i + j];
                uint32_t v = (uint32_t)(((uint64_t)data[i + j + half] * w) % modulus);
                data[i + j] = (u + v) % modulus;
                data[i + j + half] = (u + modulus - v) % modulus;
                w = (uint32_t)(((uint64_t)w * wlen) % modulus);
            }
        }
    }

    /* Scale by n^{-1} */
    uint32_t inv_n = ntt_pow_mod((uint32_t)n, modulus - 2, modulus);
    for (size_t i = 0; i < n; i++) {
        data[i] = (uint32_t)(((uint64_t)data[i] * inv_n) % modulus);
    }
}

/**
 * shs_ntt_pointwise_multiply: NTT-domain multiplication
 * Equivalent to convolution in time domain.
 */
void shs_ntt_pointwise_multiply(uint32_t *a, const uint32_t *b,
                                 size_t n, uint32_t modulus) {
    for (size_t i = 0; i < n; i++) {
        a[i] = (uint32_t)(((uint64_t)a[i] * b[i]) % modulus);
    }
}

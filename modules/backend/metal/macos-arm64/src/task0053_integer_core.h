#ifndef SYNAPTIK_TASK0053_INTEGER_CORE_H
#define SYNAPTIK_TASK0053_INTEGER_CORE_H

#if defined(TASK0053_MSL)
#define T53_INLINE inline
#define T53_U32 uint
#define T53_U64 ulong
#define T53_I64 long
#define T53_U32_C(value) value##u
#define T53_U64_C(value) value##ul
#define T53_I64_C(value) value##l
#define T53_TO_U32(value) uint(value)
#define T53_TO_U64(value) ulong(value)
#define T53_TO_I64(value) long(value)
#define T53_TABLE constant uint
#define T53_I64_MAX 9223372036854775807l
#define T53_I64_MIN (-9223372036854775807l - 1l)
#else
#include <stdbool.h>
#include <stdint.h>
#define T53_INLINE static inline
#define T53_U32 uint32_t
#define T53_U64 uint64_t
#define T53_I64 int64_t
#define T53_U32_C(value) UINT32_C(value)
#define T53_U64_C(value) UINT64_C(value)
#define T53_I64_C(value) INT64_C(value)
#define T53_TO_U32(value) ((uint32_t)(value))
#define T53_TO_U64(value) ((uint64_t)(value))
#define T53_TO_I64(value) ((int64_t)(value))
#define T53_TABLE static const uint32_t
#define T53_I64_MAX INT64_MAX
#define T53_I64_MIN INT64_MIN
#endif

#define T53_U64_MAX T53_U64_C(0xffffffffffffffff)
#define T53_LN2_OVER_128_Q48 T53_U64_C(0x00000162e42fefa4)

T53_TABLE T53_EXP2_Q31[128] = {
    T53_U32_C(0x80000000), T53_U32_C(0x80b1ed50), T53_U32_C(0x8164d1f4), T53_U32_C(0x8218af43),
    T53_U32_C(0x82cd8699), T53_U32_C(0x8383594f), T53_U32_C(0x843a28c4), T53_U32_C(0x84f1f656),
    T53_U32_C(0x85aac368), T53_U32_C(0x8664915c), T53_U32_C(0x871f6197), T53_U32_C(0x87db3580),
    T53_U32_C(0x88980e81), T53_U32_C(0x8955ee03), T53_U32_C(0x8a14d575), T53_U32_C(0x8ad4c645),
    T53_U32_C(0x8b95c1e4), T53_U32_C(0x8c57c9c4), T53_U32_C(0x8d1adf5b), T53_U32_C(0x8ddf0420),
    T53_U32_C(0x8ea4398b), T53_U32_C(0x8f6a8118), T53_U32_C(0x9031dc43), T53_U32_C(0x90fa4c8c),
    T53_U32_C(0x91c3d374), T53_U32_C(0x928e727e), T53_U32_C(0x935a2b2f), T53_U32_C(0x9426ff10),
    T53_U32_C(0x94f4efa9), T53_U32_C(0x95c3fe87), T53_U32_C(0x96942d37), T53_U32_C(0x97657d4a),
    T53_U32_C(0x9837f052), T53_U32_C(0x990b87e2), T53_U32_C(0x99e04593), T53_U32_C(0x9ab62afd),
    T53_U32_C(0x9b8d39ba), T53_U32_C(0x9c657368), T53_U32_C(0x9d3ed9a7), T53_U32_C(0x9e196e19),
    T53_U32_C(0x9ef53261), T53_U32_C(0x9fd22825), T53_U32_C(0xa0b05110), T53_U32_C(0xa18faecb),
    T53_U32_C(0xa2704303), T53_U32_C(0xa3520f69), T53_U32_C(0xa43515ae), T53_U32_C(0xa5195787),
    T53_U32_C(0xa5fed6aa), T53_U32_C(0xa6e594d0), T53_U32_C(0xa7cd93b5), T53_U32_C(0xa8b6d516),
    T53_U32_C(0xa9a15ab5), T53_U32_C(0xaa8d2653), T53_U32_C(0xab7a39b6), T53_U32_C(0xac6896a5),
    T53_U32_C(0xad583eea), T53_U32_C(0xae493453), T53_U32_C(0xaf3b78ad), T53_U32_C(0xb02f0dcc),
    T53_U32_C(0xb123f582), T53_U32_C(0xb21a31a6), T53_U32_C(0xb311c413), T53_U32_C(0xb40aaea2),
    T53_U32_C(0xb504f334), T53_U32_C(0xb60093a8), T53_U32_C(0xb6fd91e3), T53_U32_C(0xb7fbefcb),
    T53_U32_C(0xb8fbaf47), T53_U32_C(0xb9fcd245), T53_U32_C(0xbaff5ab2), T53_U32_C(0xbc034a7f),
    T53_U32_C(0xbd08a39f), T53_U32_C(0xbe0f680a), T53_U32_C(0xbf1799b6), T53_U32_C(0xc0213aa2),
    T53_U32_C(0xc12c4cca), T53_U32_C(0xc238d231), T53_U32_C(0xc346ccda), T53_U32_C(0xc4563ecc),
    T53_U32_C(0xc5672a11), T53_U32_C(0xc67990b6), T53_U32_C(0xc78d74c9), T53_U32_C(0xc8a2d85d),
    T53_U32_C(0xc9b9bd86), T53_U32_C(0xcad2265e), T53_U32_C(0xcbec14ff), T53_U32_C(0xcd078b86),
    T53_U32_C(0xce248c15), T53_U32_C(0xcf4318cf), T53_U32_C(0xd06333db), T53_U32_C(0xd184df62),
    T53_U32_C(0xd2a81d92), T53_U32_C(0xd3ccf09a), T53_U32_C(0xd4f35aac), T53_U32_C(0xd61b5dff),
    T53_U32_C(0xd744fccb), T53_U32_C(0xd870394c), T53_U32_C(0xd99d15c2), T53_U32_C(0xdacb946f),
    T53_U32_C(0xdbfbb798), T53_U32_C(0xdd2d8185), T53_U32_C(0xde60f482), T53_U32_C(0xdf9612df),
    T53_U32_C(0xe0ccdeec), T53_U32_C(0xe2055b00), T53_U32_C(0xe33f8973), T53_U32_C(0xe47b6ca0),
    T53_U32_C(0xe5b906e7), T53_U32_C(0xe6f85aab), T53_U32_C(0xe8396a50), T53_U32_C(0xe97c3840),
    T53_U32_C(0xeac0c6e8), T53_U32_C(0xec0718b6), T53_U32_C(0xed4f301f), T53_U32_C(0xee990f98),
    T53_U32_C(0xefe4b99c), T53_U32_C(0xf13230a8), T53_U32_C(0xf281773c), T53_U32_C(0xf3d28fde),
    T53_U32_C(0xf5257d15), T53_U32_C(0xf67a416c), T53_U32_C(0xf7d0df73), T53_U32_C(0xf92959bb),
    T53_U32_C(0xfa83b2db), T53_U32_C(0xfbdfed6d), T53_U32_C(0xfd3e0c0d), T53_U32_C(0xfe9e115c)
};

T53_INLINE T53_U64 task0053_abs_i64(T53_I64 value) {
    return value < 0 ? T53_U64_C(0) - T53_TO_U64(value) : T53_TO_U64(value);
}

T53_INLINE T53_I64 task0053_i64_from_magnitude(T53_U64 magnitude, bool negative) {
    if (!negative) return magnitude > T53_U64_C(0x7fffffffffffffff)
            ? T53_I64_MAX : T53_TO_I64(magnitude);
    if (magnitude >= T53_U64_C(0x8000000000000000)) return T53_I64_MIN;
    return -T53_TO_I64(magnitude);
}

T53_INLINE T53_U64 task0053_saturating_mul_u64(T53_U64 left, T53_U64 right) {
    if (right != T53_U64_C(0) && left > T53_U64_MAX / right) return T53_U64_MAX;
    return left * right;
}

T53_INLINE T53_I64 task0053_saturating_add_i64(T53_I64 left, T53_I64 right) {
    if (right > 0 && left > T53_I64_MAX - right) return T53_I64_MAX;
    if (right < 0 && left < T53_I64_MIN - right) return T53_I64_MIN;
    return left + right;
}

T53_INLINE T53_I64 task0053_saturating_sub_i64(T53_I64 left, T53_I64 right) {
    if (right > 0 && left < T53_I64_MIN + right) return T53_I64_MIN;
    if (right < 0 && left > T53_I64_MAX + right) return T53_I64_MAX;
    return left - right;
}

T53_INLINE T53_U64 task0053_round_shift_even(T53_U64 value, T53_U32 shift) {
    if (shift == T53_U32_C(0)) return value;
    if (shift >= T53_U32_C(64)) return T53_U64_C(0);
    T53_U64 quotient = value >> shift;
    T53_U64 remainder = value & ((T53_U64_C(1) << shift) - T53_U64_C(1));
    T53_U64 midpoint = T53_U64_C(1) << (shift - T53_U32_C(1));
    return quotient + T53_TO_U64(remainder > midpoint
            || (remainder == midpoint && (quotient & T53_U64_C(1)) != T53_U64_C(0)));
}

T53_INLINE T53_I64 task0053_round_signed_div(T53_I64 value, T53_U64 divisor) {
    if (divisor == T53_U64_C(0)) return value < 0 ? T53_I64_MIN : T53_I64_MAX;
    bool negative = value < 0;
    T53_U64 magnitude = task0053_abs_i64(value);
    T53_U64 quotient = magnitude / divisor;
    T53_U64 remainder = magnitude % divisor;
    quotient += T53_TO_U64(remainder > divisor - remainder
            || (remainder == divisor - remainder && (quotient & T53_U64_C(1)) != T53_U64_C(0)));
    return task0053_i64_from_magnitude(quotient, negative);
}

T53_INLINE T53_I64 task0053_raw_to_q48(T53_U32 bits) {
    T53_U32 exponent = (bits >> T53_U32_C(23)) & T53_U32_C(0xff);
    T53_U64 significand = T53_TO_U64((bits & T53_U32_C(0x007fffff)) | T53_U32_C(0x00800000));
    T53_I64 shift = T53_TO_I64(exponent) - T53_I64_C(102);
    T53_U64 magnitude;
    if (shift < 0) {
        magnitude = task0053_round_shift_even(significand, T53_TO_U32(-shift));
    } else if (shift >= T53_I64_C(64)
            || significand > (T53_U64_MAX >> T53_TO_U32(shift))) {
        magnitude = T53_U64_MAX;
    } else {
        magnitude = significand << T53_TO_U32(shift);
    }
    return task0053_i64_from_magnitude(magnitude, (bits & T53_U32_C(0x80000000)) != 0);
}

T53_INLINE T53_I64 task0053_mul_q32(T53_I64 left, T53_I64 right) {
    bool negative = (left < 0) != (right < 0);
    T53_U64 product = task0053_saturating_mul_u64(
            task0053_abs_i64(left), task0053_abs_i64(right));
    return task0053_i64_from_magnitude(task0053_round_shift_even(product, T53_U32_C(32)), negative);
}

T53_INLINE T53_U32 task0053_pack_q31(T53_U64 value, T53_I64 exponent) {
    if (value < T53_U64_C(0x80000000)) {
        value = value > T53_U64_MAX / T53_U64_C(2) ? T53_U64_MAX : value << T53_U32_C(1);
        exponent = task0053_saturating_sub_i64(exponent, T53_I64_C(1));
    } else if (value >= T53_U64_C(0x100000000)) {
        value = task0053_round_shift_even(value, T53_U32_C(1));
        exponent = task0053_saturating_add_i64(exponent, T53_I64_C(1));
    }
    T53_U64 significand = task0053_round_shift_even(value, T53_U32_C(8));
    if (significand == T53_U64_C(0x01000000)) {
        significand >>= T53_U32_C(1);
        exponent = task0053_saturating_add_i64(exponent, T53_I64_C(1));
    }
    T53_I64 biased = task0053_saturating_add_i64(exponent, T53_I64_C(127));
    if (biased >= T53_I64_C(255)) return T53_U32_C(0x7f800000);
    if (biased <= T53_I64_C(0)) return T53_U32_C(0x00000000);
    return (T53_TO_U32(biased) << T53_U32_C(23))
            | (T53_TO_U32(significand) & T53_U32_C(0x007fffff));
}

T53_INLINE T53_U32 task0053_exp_word(T53_U32 bits) {
    T53_U32 magnitude = bits & T53_U32_C(0x7fffffff);
    T53_U32 exponent = magnitude >> T53_U32_C(23);
    T53_U32 fraction = magnitude & T53_U32_C(0x007fffff);
    if (exponent == T53_U32_C(0xff)) {
        if (fraction != 0) return bits | T53_U32_C(0x00400000);
        return (bits & T53_U32_C(0x80000000)) != 0
                ? T53_U32_C(0x00000000) : T53_U32_C(0x7f800000);
    }
    if (magnitude == 0 || exponent == 0) return T53_U32_C(0x3f800000);
    if ((bits & T53_U32_C(0x80000000)) == 0
            && magnitude >= T53_U32_C(0x42b17218)) return T53_U32_C(0x7f800000);
    if ((bits & T53_U32_C(0x80000000)) != 0
            && magnitude >= T53_U32_C(0x42aeac50)) return T53_U32_C(0x00000000);

    T53_I64 x = task0053_raw_to_q48(bits);
    T53_I64 quotient = task0053_round_signed_div(x, T53_LN2_OVER_128_Q48);
    T53_U64 scaled_magnitude = task0053_saturating_mul_u64(
            task0053_abs_i64(quotient), T53_LN2_OVER_128_Q48);
    T53_I64 scaled = task0053_i64_from_magnitude(scaled_magnitude, quotient < 0);
    T53_I64 residual_q48 = task0053_saturating_sub_i64(x, scaled);
    T53_I64 residual = task0053_round_signed_div(residual_q48, T53_U64_C(0x10000));

    T53_I64 polynomial = T53_I64_C(5965232);
    polynomial = task0053_saturating_add_i64(T53_I64_C(35791394), task0053_mul_q32(residual, polynomial));
    polynomial = task0053_saturating_add_i64(T53_I64_C(178956971), task0053_mul_q32(residual, polynomial));
    polynomial = task0053_saturating_add_i64(T53_I64_C(715827883), task0053_mul_q32(residual, polynomial));
    polynomial = task0053_saturating_add_i64(T53_I64_C(2147483648), task0053_mul_q32(residual, polynomial));
    polynomial = task0053_saturating_add_i64(T53_I64_C(4294967296), task0053_mul_q32(residual, polynomial));
    polynomial = task0053_saturating_add_i64(T53_I64_C(4294967296), task0053_mul_q32(residual, polynomial));

    T53_I64 scale_exponent = quotient / T53_I64_C(128);
    T53_I64 table_index = quotient % T53_I64_C(128);
    if (table_index < 0) {
        table_index = task0053_saturating_add_i64(table_index, T53_I64_C(128));
        scale_exponent = task0053_saturating_sub_i64(scale_exponent, T53_I64_C(1));
    }
    if (table_index < 0 || table_index >= T53_I64_C(128) || polynomial <= 0)
        return T53_U32_C(0x00000000);
    T53_U64 reconstructed = task0053_round_shift_even(
            task0053_saturating_mul_u64(
                    T53_TO_U64(T53_EXP2_Q31[T53_TO_U32(table_index)]),
                    T53_TO_U64(polynomial)),
            T53_U32_C(32));
    return task0053_pack_q31(reconstructed, scale_exponent);
}

T53_INLINE T53_U32 task0053_add_one(T53_U32 value) {
    if (value == 0) return T53_U32_C(0x3f800000);
    if (value == T53_U32_C(0x3f800000)) return T53_U32_C(0x40000000);
    T53_U32 exponent = (value >> T53_U32_C(23)) & T53_U32_C(0xff);
    if (exponent == 0) return T53_U32_C(0x3f800000);
    if (exponent >= T53_U32_C(0xff)) return T53_U32_C(0x7f800000);
    if (exponent > T53_U32_C(127)) return value;
    T53_U64 significand = T53_TO_U64((value & T53_U32_C(0x007fffff)) | T53_U32_C(0x00800000));
    T53_U32 shift = T53_U32_C(127) - exponent;
    T53_U64 sum = T53_U64_C(0x00800000) + task0053_round_shift_even(significand, shift);
    T53_U32 output_exponent = T53_U32_C(127);
    if (sum >= T53_U64_C(0x01000000)) {
        sum = task0053_round_shift_even(sum, T53_U32_C(1));
        output_exponent = T53_U32_C(128);
    }
    return (output_exponent << T53_U32_C(23))
            | (T53_TO_U32(sum) & T53_U32_C(0x007fffff));
}

T53_INLINE T53_U32 task0053_divide_positive(T53_U32 numerator, T53_U32 denominator) {
    if (numerator == 0) return T53_U32_C(0x00000000);
    T53_U32 numerator_exponent = (numerator >> T53_U32_C(23)) & T53_U32_C(0xff);
    T53_U32 denominator_exponent = (denominator >> T53_U32_C(23)) & T53_U32_C(0xff);
    if (numerator_exponent == 0 || denominator_exponent == 0) return T53_U32_C(0x00000000);
    if (denominator_exponent == T53_U32_C(0xff)) return T53_U32_C(0x00000000);
    if (numerator_exponent == T53_U32_C(0xff) || denominator == 0)
        return T53_U32_C(0x7f800000);
    T53_I64 exponent = T53_TO_I64(numerator_exponent) - T53_TO_I64(denominator_exponent);
    T53_U64 left = T53_TO_U64((numerator & T53_U32_C(0x007fffff)) | T53_U32_C(0x00800000));
    T53_U64 right = T53_TO_U64((denominator & T53_U32_C(0x007fffff)) | T53_U32_C(0x00800000));
    if (left < right) {
        left <<= T53_U32_C(1);
        exponent = task0053_saturating_sub_i64(exponent, T53_I64_C(1));
    }
    T53_U64 scaled = left << T53_U32_C(23);
    T53_U64 significand = scaled / right;
    T53_U64 remainder = scaled % right;
    significand += T53_TO_U64(remainder > right - remainder
            || (remainder == right - remainder && (significand & T53_U64_C(1)) != 0));
    if (significand >= T53_U64_C(0x01000000)) {
        significand >>= T53_U32_C(1);
        exponent = task0053_saturating_add_i64(exponent, T53_I64_C(1));
    }
    if (exponent < -T53_I64_C(126)) return T53_U32_C(0x00000000);
    if (exponent > T53_I64_C(127)) return T53_U32_C(0x7f800000);
    return (T53_TO_U32(exponent + T53_I64_C(127)) << T53_U32_C(23))
            | (T53_TO_U32(significand) & T53_U32_C(0x007fffff));
}

T53_INLINE T53_U32 task0053_sigmoid_word(T53_U32 bits) {
    T53_U32 magnitude = bits & T53_U32_C(0x7fffffff);
    if ((magnitude & T53_U32_C(0x7f800000)) == T53_U32_C(0x7f800000)
            && (magnitude & T53_U32_C(0x007fffff)) != 0)
        return bits | T53_U32_C(0x00400000);
    bool negative = (bits & T53_U32_C(0x80000000)) != 0;
    T53_U32 exponential = task0053_exp_word(
            negative ? bits : bits ^ T53_U32_C(0x80000000));
    T53_U32 denominator = task0053_add_one(exponential);
    T53_U32 numerator = negative ? exponential : T53_U32_C(0x3f800000);
    return task0053_divide_positive(numerator, denominator);
}

#undef T53_INLINE
#undef T53_U32
#undef T53_U64
#undef T53_I64
#undef T53_U32_C
#undef T53_U64_C
#undef T53_I64_C
#undef T53_TO_U32
#undef T53_TO_U64
#undef T53_TO_I64
#undef T53_TABLE
#undef T53_I64_MAX
#undef T53_I64_MIN
#undef T53_U64_MAX
#undef T53_LN2_OVER_128_Q48

#endif

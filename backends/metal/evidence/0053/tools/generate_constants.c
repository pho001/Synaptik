#include <stdio.h>

#include <gmp.h>
#include <mpfr.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint32_t exp_reference(mpfr_t input, mpfr_t output, uint32_t bits) {
    mpfr_set_flt(input, from_bits(bits), MPFR_RNDN);
    mpfr_exp(output, input, MPFR_RNDN);
    return to_bits(mpfr_get_flt(output, MPFR_RNDN));
}

static uint32_t overflow_threshold(mpfr_t input, mpfr_t output) {
    uint32_t low = UINT32_C(0x3f800000), high = UINT32_C(0x7f7fffff);
    while (low < high) {
        uint32_t middle = low + (high - low) / 2U;
        if (exp_reference(input, output, middle) == UINT32_C(0x7f800000)) high = middle;
        else low = middle + 1U;
    }
    return low;
}

static uint32_t subnormal_threshold(mpfr_t input, mpfr_t output) {
    uint32_t low = UINT32_C(0x3f800000), high = UINT32_C(0x7f7fffff);
    while (low < high) {
        uint32_t middle = low + (high - low) / 2U;
        uint32_t reference = exp_reference(input, output, middle | UINT32_C(0x80000000));
        if ((reference & UINT32_C(0x7f800000)) == 0U) high = middle;
        else low = middle + 1U;
    }
    return low;
}
static uint32_t first_not_rounding_to_one(
        mpfr_t input, mpfr_t output, bool negative) {
    uint32_t low = UINT32_C(0x00800000), high = UINT32_C(0x3f800000);
    while (low < high) {
        uint32_t middle = low + (high - low) / 2U;
        uint32_t bits = negative ? middle | UINT32_C(0x80000000) : middle;
        if (exp_reference(input, output, bits) != UINT32_C(0x3f800000)) high = middle;
        else low = middle + 1U;
    }
    return low;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: generate_constants HEADER JSON CERTIFICATES.json\n");
        return 2;
    }
    mpfr_t value, input, output, lower, upper;
    mpfr_init2(value, 512);
    mpfr_init2(input, 512);
    mpfr_init2(output, 512);
    mpfr_init2(lower, 512);
    mpfr_init2(upper, 512);

    mpfr_const_log2(value, MPFR_RNDN);
    mpfr_div_2ui(value, value, 7U, MPFR_RNDN);
    mpfr_mul_2ui(value, value, 48U, MPFR_RNDN);
    uint64_t logarithm = mpfr_get_ui(value, MPFR_RNDN);

    uint32_t table[128];
    for (unsigned index = 0; index < 128U; index++) {
        mpfr_set_ui(value, index, MPFR_RNDN);
        mpfr_div_2ui(value, value, 7U, MPFR_RNDN);
        mpfr_exp2(value, value, MPFR_RNDN);
        mpfr_mul_2ui(value, value, 31U, MPFR_RNDN);
        table[index] = (uint32_t)mpfr_get_ui(value, MPFR_RNDN);
    }
    uint32_t overflow = overflow_threshold(input, output);
    uint32_t subnormal = subnormal_threshold(input, output);
    uint32_t positive_not_one = first_not_rounding_to_one(input, output, false);
    uint32_t negative_not_one = first_not_rounding_to_one(input, output, true);

    FILE *header = fopen(argv[1], "wb");
    FILE *json = fopen(argv[2], "wb");
    FILE *certificates = fopen(argv[3], "wb");
    if (header == NULL || json == NULL || certificates == NULL) return 2;
    fprintf(header,
            "#ifndef SYNAPTIK_TASK0053_CONSTANTS_H\n"
            "#define SYNAPTIK_TASK0053_CONSTANTS_H\n\n"
            "#include <stdint.h>\n\n"
            "#define TASK0053_LN2_OVER_128_Q48 UINT64_C(0x%016llx)\n"
            "#define TASK0053_OVERFLOW_FIRST UINT32_C(0x%08x)\n"
            "#define TASK0053_SUBNORMAL_FIRST_MAGNITUDE UINT32_C(0x%08x)\n"
            "#define TASK0053_POSITIVE_EXP_NOT_ONE_FIRST UINT32_C(0x%08x)\n"
            "#define TASK0053_NEGATIVE_EXP_NOT_ONE_FIRST UINT32_C(0x%08x)\n\n"
            "static const int64_t TASK0053_EXP_POLYNOMIAL_Q32[7] = {\n"
            "    INT64_C(4294967296), INT64_C(4294967296), INT64_C(2147483648),\n"
            "    INT64_C(715827883), INT64_C(178956971), INT64_C(35791394), INT64_C(5965232)\n"
            "};\n\n"
            "static const uint32_t TASK0053_EXP2_Q31[128] = {\n",
            (unsigned long long)logarithm, overflow, subnormal,
            positive_not_one, negative_not_one);
    for (unsigned index = 0; index < 128U; index++) {
        if (index % 4U == 0U) fprintf(header, "    ");
        fprintf(header, "UINT32_C(0x%08x)%s", table[index], index == 127U ? "" : ",");
        if (index % 4U == 3U) fputc('\n', header);
        else fputc(' ', header);
    }
    fprintf(header, "};\n\n#endif\n");

    fprintf(json,
            "{\n"
            "  \"generator\": \"task0053-mpfr-constants-v1\",\n"
            "  \"gmpVersion\": \"%s\",\n"
            "  \"mpfrVersion\": \"%s\",\n"
            "  \"ln2Over128Q48\": \"0x%016llx\",\n"
            "  \"overflowFirst\": \"0x%08x\",\n"
            "  \"subnormalFirstMagnitude\": \"0x%08x\",\n"
            "  \"positiveExpNotOneFirst\": \"0x%08x\",\n"
            "  \"negativeExpNotOneFirst\": \"0x%08x\",\n"
            "  \"exp2Q31\": [\n",
            gmp_version, mpfr_get_version(), (unsigned long long)logarithm,
            overflow, subnormal, positive_not_one, negative_not_one);
    for (unsigned index = 0; index < 128U; index++)
        fprintf(json, "    \"0x%08x\"%s\n", table[index], index == 127U ? "" : ",");
    fprintf(json, "  ]\n}\n");
    fprintf(certificates,
            "{\n"
            "  \"schema\": \"task0053-directed-interval-certificates-v1\",\n"
            "  \"generator\": \"MPFR %s with explicit RNDD/RNDU\",\n"
            "  \"thresholds\": [\n",
            mpfr_get_version());
    const char *threshold_names[4] = {
        "positive-overflow", "negative-subnormal-or-zero",
        "positive-not-one", "negative-not-one"
    };
    uint32_t threshold_inputs[4] = {
        overflow, subnormal | UINT32_C(0x80000000),
        positive_not_one, negative_not_one | UINT32_C(0x80000000)
    };
    for (unsigned index = 0; index < 4U; index++) {
        uint32_t first = threshold_inputs[index];
        uint32_t previous = index == 1U || index == 3U
                ? (first & UINT32_C(0x80000000)) | ((first & UINT32_C(0x7fffffff)) - 1U)
                : first - 1U;
        fprintf(certificates,
                "    {\"name\":\"%s\",\"previousInput\":\"0x%08x\","
                "\"previousReference\":\"0x%08x\",\"firstInput\":\"0x%08x\","
                "\"firstReference\":\"0x%08x\"}%s\n",
                threshold_names[index], previous,
                exp_reference(input, output, previous), first,
                exp_reference(input, output, first), index == 3U ? "" : ",");
    }
    fprintf(certificates,
            "  ],\n"
            "  \"tableScale\": 31,\n"
            "  \"directedDenominatorPower\": 256,\n"
            "  \"table\": [\n");
    mpz_t lower_numerator, upper_numerator;
    mpz_init(lower_numerator);
    mpz_init(upper_numerator);
    for (unsigned index = 0; index < 128U; index++) {
        mpfr_set_ui(lower, index, MPFR_RNDD);
        mpfr_div_2ui(lower, lower, 7U, MPFR_RNDD);
        mpfr_exp2(lower, lower, MPFR_RNDD);
        mpfr_mul_2ui(lower, lower, 31U + 256U, MPFR_RNDD);
        mpfr_get_z(lower_numerator, lower, MPFR_RNDD);
        mpfr_set_ui(upper, index, MPFR_RNDU);
        mpfr_div_2ui(upper, upper, 7U, MPFR_RNDU);
        mpfr_exp2(upper, upper, MPFR_RNDU);
        mpfr_mul_2ui(upper, upper, 31U + 256U, MPFR_RNDU);
        mpfr_get_z(upper_numerator, upper, MPFR_RNDU);
        gmp_fprintf(certificates,
                "    {\"index\":%u,\"rounded\":\"0x%08x\","
                "\"predecessor\":\"0x%08x\",\"successor\":\"0x%08x\","
                "\"lowerNumerator\":\"0x%Zx\",\"upperNumerator\":\"0x%Zx\"}%s\n",
                index, table[index], table[index] - 1U, table[index] + 1U,
                lower_numerator, upper_numerator, index == 127U ? "" : ",");
    }
    fprintf(certificates, "  ]\n}\n");
    mpz_clear(lower_numerator);
    mpz_clear(upper_numerator);
    fclose(header);
    fclose(json);
    fclose(certificates);
    mpfr_clears(value, input, output, lower, upper, (mpfr_ptr)0);
    return 0;
}

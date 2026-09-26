#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define WORKER_COUNT 16U
#define WORDS_PER_WORKER (UINT64_C(1) << 28)
#define FNV_OFFSET UINT64_C(1469598103934665603)
#define FNV_PRIME UINT64_C(1099511628211)

enum Partition {
    PART_NAN,
    PART_INFINITY,
    PART_ZERO,
    PART_MAGNITUDE_BELOW_ONE,
    PART_INTEGRAL_FINITE,
    PART_FRACTIONAL_FINITE,
    PART_COUNT
};

typedef struct {
    uint32_t worker;
    uint64_t counts[PART_COUNT];
    uint64_t failures;
    uint64_t partition_failures;
    uint64_t digest;
} Worker;

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

/* Candidate transcription: deliberately independent of the libm reference below. */
static uint32_t candidate_floor(uint32_t word) {
    uint32_t magnitude = word & UINT32_C(0x7fffffff);
    uint32_t exponent_bits = magnitude >> 23U;
    if (exponent_bits == UINT32_C(0xff) || magnitude == 0U) return word;
    int exponent = (int)exponent_bits - 127;
    if (exponent < 0)
        return (word & UINT32_C(0x80000000)) != 0U
                ? UINT32_C(0xbf800000) : UINT32_C(0x00000000);
    if (exponent >= 23) return word;
    uint32_t unit = UINT32_C(1) << (uint32_t)(23 - exponent);
    uint32_t mask = unit - 1U;
    if ((word & mask) == 0U) return word;
    uint32_t truncated = word & ~mask;
    return (word & UINT32_C(0x80000000)) != 0U ? truncated + unit : truncated;
}

static uint32_t candidate_ceil(uint32_t word) {
    uint32_t magnitude = word & UINT32_C(0x7fffffff);
    uint32_t exponent_bits = magnitude >> 23U;
    if (exponent_bits == UINT32_C(0xff) || magnitude == 0U) return word;
    int exponent = (int)exponent_bits - 127;
    if (exponent < 0)
        return (word & UINT32_C(0x80000000)) != 0U
                ? UINT32_C(0x80000000) : UINT32_C(0x3f800000);
    if (exponent >= 23) return word;
    uint32_t unit = UINT32_C(1) << (uint32_t)(23 - exponent);
    uint32_t mask = unit - 1U;
    if ((word & mask) == 0U) return word;
    uint32_t truncated = word & ~mask;
    return (word & UINT32_C(0x80000000)) != 0U ? truncated : truncated + unit;
}

static bool bits_are_nan(uint32_t word) {
    return (word & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)
            && (word & UINT32_C(0x007fffff)) != 0U;
}

static uint32_t candidate_sign(uint32_t word) {
    if (bits_are_nan(word) || (word & UINT32_C(0x7fffffff)) == 0U) return word;
    return (word & UINT32_C(0x80000000)) != 0U
            ? UINT32_C(0xbf800000) : UINT32_C(0x3f800000);
}

static uint32_t candidate_relu(uint32_t word) {
    if (bits_are_nan(word)) return word;
    return (word & UINT32_C(0x80000000)) != 0U ? 0U : word;
}

/*
 * Independent partition predicates use host binary32 classification and libm truncation rather
 * than the candidate's exponent/mask transform. Exactly one predicate must hold for every word.
 */
static uint32_t partition_mask(uint32_t word) {
    float value = from_bits(word);
    bool nan = isnan(value);
    bool infinity = isinf(value);
    bool zero = value == 0.0f;
    bool below_one = isfinite(value) && !zero && fabsf(value) < 1.0f;
    bool integral = isfinite(value) && !zero && !below_one && truncf(value) == value;
    bool fractional = isfinite(value) && !zero && !below_one && truncf(value) != value;
    return ((uint32_t)nan << PART_NAN)
            | ((uint32_t)infinity << PART_INFINITY)
            | ((uint32_t)zero << PART_ZERO)
            | ((uint32_t)below_one << PART_MAGNITUDE_BELOW_ONE)
            | ((uint32_t)integral << PART_INTEGRAL_FINITE)
            | ((uint32_t)fractional << PART_FRACTIONAL_FINITE);
}

static uint32_t reference_sign(uint32_t word) {
    float value = from_bits(word);
    if (isnan(value)) return to_bits(value);
    if (value == 0.0f) return word;
    return signbit(value) ? UINT32_C(0xbf800000) : UINT32_C(0x3f800000);
}

static uint32_t reference_relu(uint32_t word) {
    float value = from_bits(word);
    if (isnan(value)) return to_bits(value);
    if (signbit(value)) return UINT32_C(0x00000000);
    return word;
}

static bool model_equal(uint32_t input, uint32_t actual, uint32_t expected) {
    return bits_are_nan(input) ? bits_are_nan(actual) && bits_are_nan(expected)
            : actual == expected;
}

static uint64_t mix(uint64_t digest, uint32_t word) {
    digest ^= word;
    return digest * FNV_PRIME;
}

static bool check_word(uint32_t word, uint64_t *digest) {
    float value = from_bits(word);
    uint32_t floor_actual = candidate_floor(word);
    uint32_t ceil_actual = candidate_ceil(word);
    uint32_t sign_actual = candidate_sign(word);
    uint32_t relu_actual = candidate_relu(word);
    uint32_t floor_expected = to_bits(floorf(value));
    uint32_t ceil_expected = to_bits(ceilf(value));
    uint32_t sign_expected = reference_sign(word);
    uint32_t relu_expected = reference_relu(word);
    *digest = mix(mix(mix(mix(mix(*digest, word), floor_actual), ceil_actual),
            sign_actual), relu_actual);
    return model_equal(word, floor_actual, floor_expected)
            && model_equal(word, ceil_actual, ceil_expected)
            && model_equal(word, sign_actual, sign_expected)
            && model_equal(word, relu_actual, relu_expected);
}

static void *run_worker(void *opaque) {
    Worker *worker = opaque;
    uint64_t start = (uint64_t)worker->worker * WORDS_PER_WORKER;
    uint64_t end = start + WORDS_PER_WORKER;
    uint64_t digest = FNV_OFFSET;
    for (uint64_t ordinal = start; ordinal < end; ordinal++) {
        uint32_t word = (uint32_t)ordinal;
        uint32_t mask = partition_mask(word);
        if (mask == 0U || (mask & (mask - 1U)) != 0U) {
            worker->partition_failures++;
        } else {
            unsigned partition = (unsigned)__builtin_ctz(mask);
            worker->counts[partition]++;
        }
        if (!check_word(word, &digest)) worker->failures++;
    }
    worker->digest = digest;
    return NULL;
}

static int check_boundaries(uint64_t *out_count, uint64_t *out_digest) {
    static const uint32_t fixed[] = {
        0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U,
        0x007fffffU, 0x807fffffU, 0x00800000U, 0x80800000U,
        0x3f7fffffU, 0xbf7fffffU, 0x3f800000U, 0xbf800000U,
        0x3f800001U, 0xbf800001U, 0x40000000U, 0xc0000000U,
        0x4affffffU, 0xcaffffffU, 0x4b000000U, 0xcb000000U,
        0x7f7fffffU, 0xff7fffffU, 0x7f800000U, 0xff800000U,
        0x7f800001U, 0xff800001U, 0x7fc00000U, 0xffc00000U,
        0x7fffffffU, 0xffffffffU
    };
    uint64_t digest = FNV_OFFSET;
    uint64_t count = 0U;
    for (size_t index = 0U; index < sizeof(fixed) / sizeof(fixed[0]); index++) {
        if (!check_word(fixed[index], &digest)) return 1;
        count++;
    }
    for (uint32_t exponent = 0U; exponent <= UINT32_C(0xff); exponent++) {
        const uint32_t fractions[] = {0U, 1U, UINT32_C(0x003fffff),
                UINT32_C(0x007ffffe), UINT32_C(0x007fffff)};
        for (uint32_t sign = 0U; sign <= 1U; sign++) {
            for (size_t index = 0U; index < sizeof(fractions) / sizeof(fractions[0]); index++) {
                uint32_t word = (sign << 31U) | (exponent << 23U) | fractions[index];
                if (!check_word(word, &digest)) return 1;
                count++;
            }
        }
    }
    *out_count = count;
    *out_digest = digest;
    return 0;
}

int main(void) {
    _Static_assert(sizeof(float) == sizeof(uint32_t), "checker requires binary32-sized float");
    pthread_t threads[WORKER_COUNT];
    Worker workers[WORKER_COUNT] = {0};
    for (uint32_t index = 0U; index < WORKER_COUNT; index++) {
        workers[index].worker = index;
        if (pthread_create(&threads[index], NULL, run_worker, &workers[index]) != 0) return 2;
    }
    for (uint32_t index = 0U; index < WORKER_COUNT; index++)
        if (pthread_join(threads[index], NULL) != 0) return 3;

    uint64_t totals[PART_COUNT] = {0};
    uint64_t failures = 0U;
    uint64_t partition_failures = 0U;
    uint64_t digest = FNV_OFFSET;
    for (uint32_t index = 0U; index < WORKER_COUNT; index++) {
        failures += workers[index].failures;
        partition_failures += workers[index].partition_failures;
        for (unsigned partition = 0U; partition < PART_COUNT; partition++)
            totals[partition] += workers[index].counts[partition];
        digest = mix(digest, (uint32_t)workers[index].digest);
        digest = mix(digest, (uint32_t)(workers[index].digest >> 32U));
    }
    uint64_t covered = 0U;
    for (unsigned partition = 0U; partition < PART_COUNT; partition++) covered += totals[partition];
    uint64_t boundary_count = 0U;
    uint64_t boundary_digest = 0U;
    int boundary_status = check_boundaries(&boundary_count, &boundary_digest);

    printf("words=%llu failures=%llu partition_failures=%llu covered=%llu\n",
            (unsigned long long)(UINT64_C(1) << 32),
            (unsigned long long)failures,
            (unsigned long long)partition_failures,
            (unsigned long long)covered);
    printf("partitions nan=%llu infinity=%llu zero=%llu below_one=%llu integral=%llu fractional=%llu\n",
            (unsigned long long)totals[PART_NAN],
            (unsigned long long)totals[PART_INFINITY],
            (unsigned long long)totals[PART_ZERO],
            (unsigned long long)totals[PART_MAGNITUDE_BELOW_ONE],
            (unsigned long long)totals[PART_INTEGRAL_FINITE],
            (unsigned long long)totals[PART_FRACTIONAL_FINITE]);
    printf("digest=%016llx boundary_words=%llu boundary_digest=%016llx\n",
            (unsigned long long)digest,
            (unsigned long long)boundary_count,
            (unsigned long long)boundary_digest);
    if (failures != 0U || partition_failures != 0U
            || covered != (UINT64_C(1) << 32) || boundary_status != 0)
        return 1;
    return 0;
}

#include "../model/task0053_model.h"
#include "../generated/task0053_constants.h"

#include <mpfr.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#define MAX_REFERENCE_PRECISION 1024

typedef struct {
    uint64_t begin;
    uint64_t end;
    uint64_t words;
    uint64_t nan_inputs;
    uint64_t infinity_inputs;
    uint64_t zero_inputs;
    uint64_t subnormal_inputs;
    uint64_t normal_inputs;
    uint64_t ordinary_exp_results;
    uint64_t exp_class_failures;
    uint64_t exp_distance_failures;
    uint64_t sigmoid_failures;
    uint64_t add_site_failures;
    uint64_t div_site_failures;
    uint64_t unresolved;
    uint64_t maximum_distance;
    uint32_t maximum_distance_word;
    uint32_t maximum_distance_actual;
    uint32_t maximum_distance_reference;
    uint32_t first_failure_word;
    uint32_t first_failure_code;
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

static bool nan_word(uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)
            && (bits & UINT32_C(0x007fffff)) != 0U;
}

static uint64_t ordered_key(uint32_t bits) {
    return (bits & UINT32_C(0x80000000)) != 0U
            ? (uint64_t)(~bits) : (uint64_t)(bits ^ UINT32_C(0x80000000));
}

static uint64_t ordered_distance(uint32_t left, uint32_t right) {
    uint64_t a = ordered_key(left);
    uint64_t b = ordered_key(right);
    return a >= b ? a - b : b - a;
}

static void fail(Worker *worker, uint32_t word, uint32_t code) {
    if (worker->first_failure_code == 0U || word < worker->first_failure_word) {
        worker->first_failure_word = word;
        worker->first_failure_code = code;
    }
}

static bool rounded_exp_word(mpfr_t input, mpfr_t output, uint32_t bits, uint32_t *result) {
    mpfr_set_flt(input, from_bits(bits), MPFR_RNDN);
    for (mpfr_prec_t precision = 160; precision <= MAX_REFERENCE_PRECISION; precision *= 2) {
        mpfr_set_prec(output, precision);
        mpfr_exp(output, input, MPFR_RNDN);
        if (mpfr_can_round(output, precision - 2, MPFR_RNDN, MPFR_RNDN, 24)) {
            *result = to_bits(mpfr_get_flt(output, MPFR_RNDN));
            return true;
        }
    }
    return false;
}

static bool rounded_div_word(
        mpfr_t left, mpfr_t right, mpfr_t output,
        uint32_t numerator, uint32_t denominator, uint32_t *result) {
    mpfr_set_flt(left, from_bits(numerator), MPFR_RNDN);
    mpfr_set_flt(right, from_bits(denominator), MPFR_RNDN);
    for (mpfr_prec_t precision = 160; precision <= MAX_REFERENCE_PRECISION; precision *= 2) {
        mpfr_set_prec(output, precision);
        mpfr_div(output, left, right, MPFR_RNDN);
        if (mpfr_can_round(output, precision - 2, MPFR_RNDN, MPFR_RNDN, 24)) {
            *result = to_bits(mpfr_get_flt(output, MPFR_RNDN));
            return true;
        }
    }
    return false;
}

static bool exp_result_allowed(
        Worker *worker, mpfr_t input, mpfr_t output, uint32_t word, uint32_t actual) {
    uint32_t magnitude = word & UINT32_C(0x7fffffff);
    uint32_t exponent = magnitude >> 23U;
    uint32_t fraction = magnitude & UINT32_C(0x007fffff);
    worker->words++;
    if (exponent == 0xffU) {
        if (fraction != 0U) {
            worker->nan_inputs++;
            if (!nan_word(actual)) {
                worker->exp_class_failures++;
                fail(worker, word, 1U);
                return false;
            }
            return true;
        }
        worker->infinity_inputs++;
        uint32_t expected = (word & UINT32_C(0x80000000)) != 0U
                ? 0U : UINT32_C(0x7f800000);
        if (actual != expected) {
            worker->exp_class_failures++;
            fail(worker, word, 2U);
            return false;
        }
        return true;
    }
    if (magnitude == 0U) {
        worker->zero_inputs++;
        if (actual != UINT32_C(0x3f800000)) {
            worker->exp_class_failures++;
            fail(worker, word, 3U);
            return false;
        }
        return true;
    }
    if (exponent == 0U) {
        worker->subnormal_inputs++;
        if (actual != UINT32_C(0x3f800000)) {
            worker->exp_class_failures++;
            fail(worker, word, 4U);
            return false;
        }
        return true;
    }
    worker->normal_inputs++;
    uint32_t reference;
    bool negative = (word & UINT32_C(0x80000000)) != 0U;
    if (!negative && magnitude >= TASK0053_OVERFLOW_FIRST) {
        reference = UINT32_C(0x7f800000);
    } else if (negative && magnitude >= TASK0053_SUBNORMAL_FIRST_MAGNITUDE) {
        reference = UINT32_C(0x00000001);
    } else if ((!negative && magnitude < TASK0053_POSITIVE_EXP_NOT_ONE_FIRST)
            || (negative && magnitude < TASK0053_NEGATIVE_EXP_NOT_ONE_FIRST)) {
        reference = UINT32_C(0x3f800000);
    } else if (!rounded_exp_word(input, output, word, &reference)) {
        worker->unresolved++;
        fail(worker, word, 5U);
        return false;
    }
    if (reference == UINT32_C(0x7f800000) || reference == 0U) {
        if (actual != reference) {
            worker->exp_class_failures++;
            fail(worker, word, 6U);
            return false;
        }
        return true;
    }
    if ((reference & UINT32_C(0x7f800000)) == 0U) {
        if (actual != 0U && actual != reference) {
            worker->exp_class_failures++;
            fail(worker, word, 7U);
            return false;
        }
        return true;
    }
    if ((actual & UINT32_C(0x80000000)) != 0U
            || (actual & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)
            || (actual & UINT32_C(0x7f800000)) == 0U) {
        worker->exp_class_failures++;
        fail(worker, word, 8U);
        return false;
    }
    worker->ordinary_exp_results++;
    uint64_t distance = ordered_distance(actual, reference);
    if (distance > worker->maximum_distance
            || (distance == worker->maximum_distance
                    && word < worker->maximum_distance_word)) {
        worker->maximum_distance = distance;
        worker->maximum_distance_word = word;
        worker->maximum_distance_actual = actual;
        worker->maximum_distance_reference = reference;
    }
    if (distance > 5U) {
        worker->exp_distance_failures++;
        fail(worker, word, 9U);
        return false;
    }
    return true;
}

static uint32_t rounded_add_one_word(mpfr_t input, mpfr_t output, uint32_t word) {
    mpfr_set_flt(input, from_bits(word), MPFR_RNDN);
    mpfr_add_ui(output, input, 1U, MPFR_RNDN);
    return to_bits(mpfr_get_flt(output, MPFR_RNDN));
}

static void check_sigmoid(
        Worker *worker, mpfr_t left, mpfr_t right, mpfr_t output, uint32_t word) {
    uint32_t magnitude = word & UINT32_C(0x7fffffff);
    uint32_t actual = task0053_model_sigmoid(word);
    if ((magnitude & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)
            && (magnitude & UINT32_C(0x007fffff)) != 0U) {
        if (!nan_word(actual)) {
            worker->sigmoid_failures++;
            fail(worker, word, 10U);
        }
        return;
    }
    bool negative = (word & UINT32_C(0x80000000)) != 0U;
    uint32_t exp_input = negative ? word : word ^ UINT32_C(0x80000000);
    uint32_t exponential = task0053_model_exp(exp_input);
    uint32_t denominator = task0053_model_add_one(exponential);
    uint32_t add_reference = exponential == 0U
            ? UINT32_C(0x3f800000)
            : (exponential == UINT32_C(0x3f800000)
                    ? UINT32_C(0x40000000)
                    : rounded_add_one_word(left, output, exponential));
    if (denominator != add_reference) {
        worker->add_site_failures++;
        fail(worker, word, 11U);
    }
    uint32_t numerator = negative ? exponential : UINT32_C(0x3f800000);
    uint32_t div_reference;
    if (exponential == 0U) {
        div_reference = negative ? 0U : UINT32_C(0x3f800000);
    } else if (exponential == UINT32_C(0x3f800000)) {
        div_reference = UINT32_C(0x3f000000);
    } else if (!rounded_div_word(
            left, right, output, numerator, denominator, &div_reference)) {
        worker->unresolved++;
        fail(worker, word, 12U);
        return;
    }
    uint32_t division = task0053_model_divide_positive(numerator, denominator);
    bool subnormal_reference = (div_reference & UINT32_C(0x7f800000)) == 0U
            && (div_reference & UINT32_C(0x007fffff)) != 0U;
    if (division != div_reference && !(subnormal_reference && division == 0U)) {
        worker->div_site_failures++;
        fail(worker, word, 13U);
    }
    if (actual != division) {
        worker->sigmoid_failures++;
        fail(worker, word, 14U);
    }
}

static void *run_worker(void *argument) {
    Worker *worker = argument;
    mpfr_t left, right, output;
    mpfr_init2(left, 160);
    mpfr_init2(right, 160);
    mpfr_init2(output, 160);
    worker->maximum_distance_word = UINT32_MAX;
    for (uint64_t magnitude = worker->begin; magnitude < worker->end; magnitude++) {
        uint32_t positive = (uint32_t)magnitude;
        uint32_t negative = positive | UINT32_C(0x80000000);
        (void)exp_result_allowed(
                worker, left, output, positive, task0053_model_exp(positive));
        (void)exp_result_allowed(
                worker, left, output, negative, task0053_model_exp(negative));
        check_sigmoid(worker, left, right, output, positive);
        check_sigmoid(worker, left, right, output, negative);
    }
    mpfr_clears(left, right, output, (mpfr_ptr)0);
    return NULL;
}

static uint64_t sum_field(Worker *workers, unsigned count, size_t offset) {
    uint64_t sum = 0U;
    for (unsigned index = 0; index < count; index++)
        sum += *(uint64_t *)((unsigned char *)&workers[index] + offset);
    return sum;
}

#define SUM(field) sum_field(workers, threads, offsetof(Worker, field))

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        fprintf(stderr,
                "usage: task0053_all_words THREADS REPORT.json [BEGIN_MAG END_MAG]\n");
        return 2;
    }
    char *end = NULL;
    unsigned long parsed = strtoul(argv[1], &end, 10);
    if (*argv[1] == '\0' || *end != '\0' || parsed == 0U || parsed > 64U) {
        fprintf(stderr, "thread count must be in 1..64\n");
        return 2;
    }
    unsigned threads = (unsigned)parsed;
    uint64_t range_begin = 0U;
    uint64_t range_end = UINT64_C(0x80000000);
    if (argc == 5) {
        char *begin_end = NULL;
        char *range_end_end = NULL;
        range_begin = strtoull(argv[3], &begin_end, 0);
        range_end = strtoull(argv[4], &range_end_end, 0);
        if (*argv[3] == '\0' || *begin_end != '\0'
                || *argv[4] == '\0' || *range_end_end != '\0'
                || range_begin >= range_end || range_end > UINT64_C(0x80000000)) {
            fprintf(stderr, "magnitude range is invalid\n");
            return 2;
        }
    }
    Worker *workers = calloc(threads, sizeof(*workers));
    pthread_t *handles = calloc(threads, sizeof(*handles));
    if (workers == NULL || handles == NULL) return 2;
    uint64_t magnitudes = range_end - range_begin;
    for (unsigned index = 0; index < threads; index++) {
        workers[index].begin = range_begin + magnitudes * index / threads;
        workers[index].end = range_begin + magnitudes * (index + 1U) / threads;
        int status = pthread_create(&handles[index], NULL, run_worker, &workers[index]);
        if (status != 0) {
            fprintf(stderr, "pthread_create failed: %d\n", status);
            return 2;
        }
    }
    for (unsigned index = 0; index < threads; index++) pthread_join(handles[index], NULL);

    uint64_t maximum_distance = 0U;
    uint32_t maximum_word = UINT32_MAX, maximum_actual = 0U, maximum_reference = 0U;
    uint32_t first_failure_word = UINT32_MAX, first_failure_code = 0U;
    for (unsigned index = 0; index < threads; index++) {
        Worker *worker = &workers[index];
        if (worker->maximum_distance > maximum_distance
                || (worker->maximum_distance == maximum_distance
                        && worker->maximum_distance_word < maximum_word)) {
            maximum_distance = worker->maximum_distance;
            maximum_word = worker->maximum_distance_word;
            maximum_actual = worker->maximum_distance_actual;
            maximum_reference = worker->maximum_distance_reference;
        }
        if (worker->first_failure_code != 0U
                && worker->first_failure_word < first_failure_word) {
            first_failure_word = worker->first_failure_word;
            first_failure_code = worker->first_failure_code;
        }
    }
    uint64_t failures = SUM(exp_class_failures) + SUM(exp_distance_failures)
            + SUM(sigmoid_failures) + SUM(add_site_failures) + SUM(div_site_failures);
    FILE *report = fopen(argv[2], "wb");
    uint64_t unresolved = SUM(unresolved);
    if (report == NULL) return 2;
    fprintf(report,
            "{\n"
            "  \"checker\": \"task0053-mpfr-partitioned-all-binary32-v2\",\n"
            "  \"threads\": %u,\n"
            "  \"beginMagnitude\": \"0x%08llx\",\n"
            "  \"endMagnitude\": \"0x%08llx\",\n"
            "  \"words\": %llu,\n"
            "  \"nanInputs\": %llu,\n"
            "  \"infinityInputs\": %llu,\n"
            "  \"zeroInputs\": %llu,\n"
            "  \"subnormalInputs\": %llu,\n"
            "  \"normalInputs\": %llu,\n"
            "  \"ordinaryExpResults\": %llu,\n"
            "  \"maximumOrderedDistance\": %llu,\n"
            "  \"maximumDistanceWord\": \"0x%08x\",\n"
            "  \"maximumDistanceActual\": \"0x%08x\",\n"
            "  \"maximumDistanceReference\": \"0x%08x\",\n"
            "  \"expClassFailures\": %llu,\n"
            "  \"expDistanceFailures\": %llu,\n"
            "  \"sigmoidFailures\": %llu,\n"
            "  \"addSiteFailures\": %llu,\n"
            "  \"divSiteFailures\": %llu,\n"
            "  \"unresolved\": %llu,\n"
            "  \"firstFailureWord\": \"0x%08x\",\n"
            "  \"firstFailureCode\": %u,\n"
            "  \"verdict\": \"%s\"\n"
            "}\n",
            threads,
            (unsigned long long)range_begin,
            (unsigned long long)range_end,
            (unsigned long long)SUM(words),
            (unsigned long long)SUM(nan_inputs),
            (unsigned long long)SUM(infinity_inputs),
            (unsigned long long)SUM(zero_inputs),
            (unsigned long long)SUM(subnormal_inputs),
            (unsigned long long)SUM(normal_inputs),
            (unsigned long long)SUM(ordinary_exp_results),
            (unsigned long long)maximum_distance,
            maximum_word, maximum_actual, maximum_reference,
            (unsigned long long)SUM(exp_class_failures),
            (unsigned long long)SUM(exp_distance_failures),
            (unsigned long long)SUM(sigmoid_failures),
            (unsigned long long)SUM(add_site_failures),
            (unsigned long long)SUM(div_site_failures),
            (unsigned long long)unresolved,
            first_failure_word, first_failure_code,
            failures == 0U && unresolved == 0U ? "PASS" : "FAIL");
    fclose(report);
    free(handles);
    free(workers);
    return failures == 0U && unresolved == 0U ? 0 : 1;
}

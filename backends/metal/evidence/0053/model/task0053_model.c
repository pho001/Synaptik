#include "task0053_model.h"

#include "../../../../../native/metal-macos-arm64/src/task0053_integer_core.h"

uint32_t task0053_model_exp(uint32_t bits) {
    return task0053_exp_word(bits);
}

uint32_t task0053_model_add_one(uint32_t value) {
    return task0053_add_one(value);
}

uint32_t task0053_model_divide_positive(uint32_t numerator, uint32_t denominator) {
    return task0053_divide_positive(numerator, denominator);
}

uint32_t task0053_model_sigmoid(uint32_t bits) {
    return task0053_sigmoid_word(bits);
}

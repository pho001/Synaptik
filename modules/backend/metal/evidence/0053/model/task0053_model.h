#ifndef SYNAPTIK_TASK0053_MODEL_H
#define SYNAPTIK_TASK0053_MODEL_H

#include <stdint.h>

uint32_t task0053_model_exp(uint32_t bits);
uint32_t task0053_model_add_one(uint32_t bits);
uint32_t task0053_model_divide_positive(uint32_t numerator, uint32_t denominator);
uint32_t task0053_model_sigmoid(uint32_t bits);

#endif

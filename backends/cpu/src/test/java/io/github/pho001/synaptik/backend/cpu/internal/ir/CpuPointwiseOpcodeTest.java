package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.util.EnumSet;
import org.junit.jupiter.api.Test;

class CpuPointwiseOpcodeTest {
    @Test void exposesOneBoundedFamilyOrientedOpcodeVocabulary() {
        assertAll(
                () -> assertEquals(48, CpuPointwiseOpcode.values().length),
                () -> assertEquals(2, CpuPointwiseOpcode.ADD.arity()),
                () -> assertEquals(3, CpuPointwiseOpcode.WHERE.arity()),
                () -> assertTrue(CpuPointwiseOpcode.SCALAR_MUL.carriesScalarImmediate()),
                () -> assertFalse(CpuPointwiseOpcode.CAST.carriesScalarImmediate()),
                () -> assertEquals(EnumSet.of(CpuPointwiseOpcode.ADD, CpuPointwiseOpcode.SUB,
                                CpuPointwiseOpcode.MUL, CpuPointwiseOpcode.DIV,
                                CpuPointwiseOpcode.MIN, CpuPointwiseOpcode.MAX,
                                CpuPointwiseOpcode.SCALAR_ADD,
                                CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL,
                                CpuPointwiseOpcode.SCALAR_DIV, CpuPointwiseOpcode.SCALAR_POW,
                                CpuPointwiseOpcode.SCALAR_MIN, CpuPointwiseOpcode.SCALAR_MAX,
                                CpuPointwiseOpcode.SCALAR_CLAMP,
                                CpuPointwiseOpcode.NEG, CpuPointwiseOpcode.ABS,
                                CpuPointwiseOpcode.RECIPROCAL, CpuPointwiseOpcode.LOG,
                                CpuPointwiseOpcode.LOG1P, CpuPointwiseOpcode.EXP,
                                CpuPointwiseOpcode.EXPM1, CpuPointwiseOpcode.ERF,
                                CpuPointwiseOpcode.SQRT, CpuPointwiseOpcode.RSQRT,
                                CpuPointwiseOpcode.SIGN, CpuPointwiseOpcode.RELU,
                                CpuPointwiseOpcode.TANH, CpuPointwiseOpcode.GELU_EXACT,
                                CpuPointwiseOpcode.IS_FINITE, CpuPointwiseOpcode.IS_NAN,
                                CpuPointwiseOpcode.IS_INF, CpuPointwiseOpcode.GREATER_THAN,
                                CpuPointwiseOpcode.GREATER_OR_EQUAL, CpuPointwiseOpcode.LESS_THAN,
                                CpuPointwiseOpcode.LESS_OR_EQUAL, CpuPointwiseOpcode.EQUAL,
                                CpuPointwiseOpcode.NOT_EQUAL, CpuPointwiseOpcode.LOGICAL_AND,
                                CpuPointwiseOpcode.LOGICAL_OR, CpuPointwiseOpcode.LOGICAL_NOT,
                                CpuPointwiseOpcode.WHERE, CpuPointwiseOpcode.CAST),
                        EnumSet.copyOf(java.util.Arrays.stream(CpuPointwiseOpcode.values())
                                .filter(CpuPointwiseOpcode::vectorEligible).toList())),
                () -> assertEquals(CpuPointwiseOpcode.VectorForm.MASK_PRODUCER,
                        CpuPointwiseOpcode.IS_NAN.vectorForm()),
                () -> assertEquals(CpuPointwiseOpcode.VectorForm.VALUE_OR_MASK,
                        CpuPointwiseOpcode.LOGICAL_AND.vectorForm()),
                () -> assertEquals(CpuPointwiseOpcode.VectorForm.NONE,
                        CpuPointwiseOpcode.FLOOR.vectorForm()));
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import java.util.EnumSet;
import org.junit.jupiter.api.Test;

class CpuPointwiseOpcodeTest {
    @Test void exposesOneBoundedFamilyOrientedOpcodeVocabulary() {
        assertAll(
                () -> assertEquals(22, CpuPointwiseOpcode.values().length),
                () -> assertEquals(2, CpuPointwiseOpcode.ADD.arity()),
                () -> assertEquals(3, CpuPointwiseOpcode.WHERE.arity()),
                () -> assertTrue(CpuPointwiseOpcode.SCALAR_MUL.carriesScalarImmediate()),
                () -> assertFalse(CpuPointwiseOpcode.CAST.carriesScalarImmediate()),
                () -> assertEquals(EnumSet.of(CpuPointwiseOpcode.ADD, CpuPointwiseOpcode.SUB,
                                CpuPointwiseOpcode.MUL, CpuPointwiseOpcode.DIV,
                                CpuPointwiseOpcode.SCALAR_ADD,
                                CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL,
                                CpuPointwiseOpcode.SCALAR_DIV, CpuPointwiseOpcode.SCALAR_POW,
                                CpuPointwiseOpcode.NEG, CpuPointwiseOpcode.GELU_EXACT),
                        EnumSet.copyOf(java.util.Arrays.stream(CpuPointwiseOpcode.values())
                                .filter(CpuPointwiseOpcode::vectorEligible).toList())));
    }
}

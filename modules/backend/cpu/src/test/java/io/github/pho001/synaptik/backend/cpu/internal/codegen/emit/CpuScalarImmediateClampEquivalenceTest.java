package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

/** Behavioral comparison of represented scalar multiplication against independent typed loops. */
class CpuScalarImmediateClampEquivalenceTest {
    @Test void generatedClassesExecuteAgainstIndependentTypedCleanJavaLoopsOnOrdinaryAndEdgeInputs()
            throws Throwable {
        short[] input = {b(0.0f), b(-0.0f), b(1.5f), b(-2.5f), b(Float.NaN),
                b(Float.POSITIVE_INFINITY), b(Float.NEGATIVE_INFINITY), b(Float.MAX_VALUE)};
        for (var fixture : CpuScalarImmediateClampEquivalenceOracle.fixtures()) {
            var artifact = CpuScalarImmediateClampEquivalenceOracle.artifact(fixture);
            assertArrayEquals(CpuScalarImmediateClampEquivalenceOracle.cleanJava(fixture, input),
                    CpuScalarImmediateClampEquivalenceOracle.invoke(artifact, input), fixture.id());
        }
    }

    private static short b(float value) {
        return CpuScalarImmediateClampEquivalenceOracle.bfloat(value);
    }
}

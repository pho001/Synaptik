package backend.cpu1.fused;

import backend.cpu1.kernels.fused.codegen.support.Cpu1FusedGeneratedSupport;
import backend.cpu1.kernels.fused.codegen.support.Cpu1FusedMathSupport;
import org.junit.jupiter.api.Test;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;
import utils.SpecialFunctions;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1FusedGeneratedSupportTest {
    @Test
    void bf16HelpersMatchTensorDTypeOpsForRepresentativeValues() {
        float[] values = {
                -Float.MAX_VALUE,
                -3.5f,
                -0.0f,
                0.0f,
                1.0f,
                1.0f / 3.0f,
                Float.MIN_NORMAL,
                Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY,
                Float.NaN
        };

        for (float value : values) {
            short expectedBits = TensorDTypeOps.toBFloat16Bits(value);

            assertEquals(expectedBits, Cpu1FusedGeneratedSupport.floatToBf16(value));
            assertSameFloatBits(
                    TensorDTypeOps.fromBFloat16Bits(expectedBits),
                    Cpu1FusedGeneratedSupport.bf16ToFloat(expectedBits)
            );
        }
    }

    @Test
    void boolHelperTreatsOnlyZeroAsFalse() {
        assertTrue(Cpu1FusedGeneratedSupport.bool((byte) 1));
        assertTrue(Cpu1FusedGeneratedSupport.bool((byte) -1));
        assertTrue(Cpu1FusedGeneratedSupport.bool(Byte.MIN_VALUE));
        assertFalse(Cpu1FusedGeneratedSupport.bool((byte) 0));
    }

    @Test
    void f32BasicMathHelpersMatchJavaSemantics() {
        assertFloatEquals(Math.max(0.0f, -4.0f), Cpu1FusedMathSupport.reluF32(-4.0f), 0.0f);
        assertFloatEquals(Math.max(0.0f, 2.5f), Cpu1FusedMathSupport.reluF32(2.5f), 0.0f);
        assertFloatEquals(Math.min(-2.0f, 5.0f), Cpu1FusedMathSupport.minF32(-2.0f, 5.0f), 0.0f);
        assertFloatEquals(Math.max(-2.0f, 5.0f), Cpu1FusedMathSupport.maxF32(-2.0f, 5.0f), 0.0f);
        assertFloatEquals(Math.abs(-7.25f), Cpu1FusedMathSupport.absF32(-7.25f), 0.0f);
    }

    @Test
    void f64BasicMathHelpersMatchJavaSemantics() {
        assertDoubleEquals(Math.max(0.0d, -4.0d), Cpu1FusedMathSupport.reluF64(-4.0d), 0.0d);
        assertDoubleEquals(Math.max(0.0d, 2.5d), Cpu1FusedMathSupport.reluF64(2.5d), 0.0d);
        assertDoubleEquals(Math.min(-2.0d, 5.0d), Cpu1FusedMathSupport.minF64(-2.0d, 5.0d), 0.0d);
        assertDoubleEquals(Math.max(-2.0d, 5.0d), Cpu1FusedMathSupport.maxF64(-2.0d, 5.0d), 0.0d);
        assertDoubleEquals(Math.abs(-7.25d), Cpu1FusedMathSupport.absF64(-7.25d), 0.0d);
    }

    @Test
    void f32IntrinsicMathHelpersMatchBackingImplementations() {
        float value = 0.75f;

        assertFloatEquals((float) Math.exp(value), Cpu1FusedMathSupport.expF32(value), 0.0f);
        assertFloatEquals(FastTranscendentals.fastExpF32(value), Cpu1FusedMathSupport.fastExpF32(value), 0.0f);
        assertFloatEquals((float) Math.log(value), Cpu1FusedMathSupport.logF32(value), 0.0f);
        assertFloatEquals((float) Math.tanh(value), Cpu1FusedMathSupport.tanhF32(value), 0.0f);
        assertFloatEquals(FastTranscendentals.fastTanhF32(value), Cpu1FusedMathSupport.fastTanhF32(value), 0.0f);
        assertFloatEquals(SpecialFunctions.erf(value), Cpu1FusedMathSupport.erfF32(value), 0.0f);
        assertFloatEquals((float) Math.sqrt(value), Cpu1FusedMathSupport.sqrtF32(value), 0.0f);
        assertFloatEquals(1.0f / (1.0f + (float) Math.exp(-value)), Cpu1FusedMathSupport.sigmoidF32(value), 0.0f);
        assertFloatEquals((float) Math.floor(-1.25f), Cpu1FusedMathSupport.floorF32(-1.25f), 0.0f);
        assertFloatEquals((float) Math.ceil(-1.25f), Cpu1FusedMathSupport.ceilF32(-1.25f), 0.0f);
        assertFloatEquals(-1.0f, Cpu1FusedMathSupport.signF32(-3.0f), 0.0f);
        assertFloatEquals(0.0f, Cpu1FusedMathSupport.signF32(0.0f), 0.0f);
        assertFloatEquals(1.0f, Cpu1FusedMathSupport.signF32(3.0f), 0.0f);
    }

    @Test
    void f64IntrinsicMathHelpersMatchBackingImplementations() {
        double value = 0.75d;

        assertDoubleEquals(Math.exp(value), Cpu1FusedMathSupport.expF64(value), 0.0d);
        assertDoubleEquals(FastTranscendentals.fastExpF64(value), Cpu1FusedMathSupport.fastExpF64(value), 0.0d);
        assertDoubleEquals(Math.log(value), Cpu1FusedMathSupport.logF64(value), 0.0d);
        assertDoubleEquals(Math.tanh(value), Cpu1FusedMathSupport.tanhF64(value), 0.0d);
        assertDoubleEquals(FastTranscendentals.fastTanhF64(value), Cpu1FusedMathSupport.fastTanhF64(value), 0.0d);
        assertDoubleEquals(SpecialFunctions.erf(value), Cpu1FusedMathSupport.erfF64(value), 0.0d);
        assertDoubleEquals(Math.sqrt(value), Cpu1FusedMathSupport.sqrtF64(value), 0.0d);
        assertDoubleEquals(1.0d / (1.0d + Math.exp(-value)), Cpu1FusedMathSupport.sigmoidF64(value), 0.0d);
        assertDoubleEquals(Math.floor(-1.25d), Cpu1FusedMathSupport.floorF64(-1.25d), 0.0d);
        assertDoubleEquals(Math.ceil(-1.25d), Cpu1FusedMathSupport.ceilF64(-1.25d), 0.0d);
        assertDoubleEquals(-1.0d, Cpu1FusedMathSupport.signF64(-3.0d), 0.0d);
        assertDoubleEquals(0.0d, Cpu1FusedMathSupport.signF64(0.0d), 0.0d);
        assertDoubleEquals(1.0d, Cpu1FusedMathSupport.signF64(3.0d), 0.0d);
    }

    @Test
    void f32PowHelperCoversFastCasesAndGeneralExponent() {
        assertFloatEquals(1.0f, Cpu1FusedMathSupport.powF32(4.0f, 0.0f), 0.0f);
        assertFloatEquals(4.0f, Cpu1FusedMathSupport.powF32(4.0f, 1.0f), 0.0f);
        assertFloatEquals(16.0f, Cpu1FusedMathSupport.powF32(4.0f, 2.0f), 0.0f);
        assertFloatEquals(2.0f, Cpu1FusedMathSupport.powF32(4.0f, 0.5f), 0.0f);
        assertFloatEquals(0.25f, Cpu1FusedMathSupport.powF32(4.0f, -1.0f), 0.0f);
        assertFloatEquals((float) Math.pow(4.25f, 1.75f), Cpu1FusedMathSupport.powF32(4.25f, 1.75f), 0.0f);
    }

    @Test
    void f64PowHelperCoversFastCasesAndGeneralExponent() {
        assertDoubleEquals(1.0d, Cpu1FusedMathSupport.powF64(4.0d, 0.0d), 0.0d);
        assertDoubleEquals(4.0d, Cpu1FusedMathSupport.powF64(4.0d, 1.0d), 0.0d);
        assertDoubleEquals(16.0d, Cpu1FusedMathSupport.powF64(4.0d, 2.0d), 0.0d);
        assertDoubleEquals(2.0d, Cpu1FusedMathSupport.powF64(4.0d, 0.5d), 0.0d);
        assertDoubleEquals(0.25d, Cpu1FusedMathSupport.powF64(4.0d, -1.0d), 0.0d);
        assertDoubleEquals(Math.pow(4.25d, 1.75d), Cpu1FusedMathSupport.powF64(4.25d, 1.75d), 0.0d);
    }

    private static void assertSameFloatBits(float expected, float actual) {
        assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual));
    }

    private static void assertFloatEquals(float expected, float actual, float tolerance) {
        if (Float.isNaN(expected)) {
            assertTrue(Float.isNaN(actual));
            return;
        }
        assertEquals(expected, actual, tolerance);
    }

    private static void assertDoubleEquals(double expected, double actual, double tolerance) {
        if (Double.isNaN(expected)) {
            assertTrue(Double.isNaN(actual));
            return;
        }
        assertEquals(expected, actual, tolerance);
    }
}

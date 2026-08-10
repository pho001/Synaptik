package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import java.lang.reflect.Field;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

class CpuVectorMathTest {
    @Test void floatCoefficientTablesAreExactRoundedDerivations() throws Exception {
        assertDerived("ERF_T");
        assertDerived("ERF_U");
        assertDerived("ERFC_P");
        assertDerived("ERFC_Q");
        assertDerived("ERFC_R");
        assertDerived("ERFC_S");
    }

    @Test void floatErfAndGeluMeetIndependentOracleAtBreakpointsAndRandomValues() {
        float[] directed = {Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY,
                -8.0f, Math.nextDown(-8.0f), Math.nextUp(-8.0f), -1.0f,
                Math.nextDown(-1.0f), Math.nextUp(-1.0f), -0.0f, 0.0f,
                Math.nextDown(1.0f), 1.0f, Math.nextUp(1.0f), Math.nextDown(8.0f), 8.0f,
                Math.nextUp(8.0f), -Float.MIN_NORMAL, Float.MIN_NORMAL,
                -Float.MIN_VALUE, Float.MIN_VALUE, -12.0f, 12.0f};
        check(directed);
        var random = new Random(0x5_0005_1L);
        float[] values = new float[4096];
        for (int i = 0; i < values.length; i++) values[i] = -10.0f + 20.0f * random.nextFloat();
        check(values);
    }

    private static void check(float[] values) {
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        for (int offset = 0; offset < values.length; offset += lanes) {
            float[] input = new float[lanes];
            for (int lane = 0; lane < lanes; lane++) input[lane] = values[
                    Math.min(values.length - 1, offset + lane)];
            FloatVector vector = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, input, 0);
            float[] erf = new float[lanes];
            float[] gelu = new float[lanes];
            CpuVectorMath.erf(vector).intoArray(erf, 0);
            CpuVectorMath.gelu(vector).intoArray(gelu, 0);
            for (int lane = 0; lane < lanes && offset + lane < values.length; lane++) {
                float value = input[lane];
                if (Float.isNaN(value)) {
                    assertTrue(Float.isNaN(erf[lane]));
                    assertTrue(Float.isNaN(gelu[lane]));
                    continue;
                }
                float expectedErf = (float) oracleErf(value);
                float expectedGelu = value == Float.NEGATIVE_INFINITY ? -0.0f
                        : (float) (0.5d * value * (1.0d + oracleErf(value / Math.sqrt(2.0d))));
                assertBound(expectedErf, erf[lane], "erf " + value);
                assertBound(expectedGelu, gelu[lane], "gelu " + value);
            }
        }
    }

    private static void assertBound(float expected, float actual, String message) {
        if (expected == 0.0f || Float.isInfinite(expected)) {
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual), message);
        } else {
            assertEquals(expected, actual, Math.max(2e-5f, 2e-5f * Math.abs(expected)), message);
        }
    }

    private static double oracleErf(double value) {
        if (Double.isNaN(value)) return Double.NaN;
        if (value == Double.POSITIVE_INFINITY) return 1.0d;
        if (value == Double.NEGATIVE_INFINITY) return -1.0d;
        if (value == 0.0d) return value;
        double magnitude = Math.abs(value);
        if (magnitude >= 8.0d) return Math.copySign(1.0d, value);
        int intervals = 4096;
        double step = magnitude / intervals;
        double sum = 1.0d + StrictMath.exp(-magnitude * magnitude);
        for (int i = 1; i < intervals; i++) {
            double x = i * step;
            sum += (i & 1) == 0 ? 2.0d * StrictMath.exp(-x * x)
                    : 4.0d * StrictMath.exp(-x * x);
        }
        double integral = step * sum / 3.0d;
        return Math.copySign(2.0d / Math.sqrt(Math.PI) * integral, value);
    }

    private static void assertDerived(String suffix) throws Exception {
        double[] doubles = (double[]) field("DOUBLE_" + suffix).get(null);
        float[] floats = (float[]) field("FLOAT_" + suffix).get(null);
        assertEquals(doubles.length, floats.length);
        for (int i = 0; i < doubles.length; i++) assertEquals(
                Float.floatToRawIntBits((float) doubles[i]), Float.floatToRawIntBits(floats[i]),
                suffix + " coefficient " + i);
    }

    private static Field field(String name) throws Exception {
        Field result = CpuVectorMath.class.getDeclaredField(name);
        result.setAccessible(true);
        return result;
    }
}

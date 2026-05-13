package onnx;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class OnnxPrimitiveRoundTripEvidenceTest {
    @Test
    void arithmeticBinaryOpsRoundTrip() {
        Tensor x = input("x");
        Tensor y = input("y");

        assertArrayEquals(new double[]{5.0, 4.0, 9.0}, roundTrip(x.add(y)), 1e-6);
        assertArrayEquals(new double[]{-3.0, 0.0, -3.0}, roundTrip(x.sub(y)), 1e-6);
        assertArrayEquals(new double[]{4.0, 4.0, 18.0}, roundTrip(x.mul(y)), 1e-6);
        assertArrayEquals(new double[]{0.25, 1.0, 0.5}, roundTrip(x.div(y)), 1e-6);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, roundTrip(x.min(y)), 1e-6);
        assertArrayEquals(new double[]{4.0, 2.0, 6.0}, roundTrip(x.max(y)), 1e-6);
        assertArrayEquals(new double[]{1.0, 4.0, 9.0}, roundTrip(x.pow(2.0), Map.of("x", new float[]{1f, 2f, 3f})), 1e-6);
    }

    @Test
    void compareAndLogicalOpsRoundTrip() {
        Tensor x = input("x");
        Tensor y = input("y");

        assertArrayEquals(new boolean[]{false, true, false}, roundTripBool(x.equalTo(y), numericInputs()));
        assertArrayEquals(new boolean[]{false, false, false}, roundTripBool(x.greaterThan(y), numericInputs()));
        assertArrayEquals(new boolean[]{true, false, true}, roundTripBool(x.lessThan(y), numericInputs()));

        Tensor a = new Tensor(new byte[3], new int[]{3}, null, "a", DataType.BOOL);
        Tensor b = new Tensor(new byte[3], new int[]{3}, null, "b", DataType.BOOL);
        Map<String, byte[]> boolInputs = Map.of(
                "a", new byte[]{1, 0, 1},
                "b", new byte[]{1, 1, 0}
        );
        assertArrayEquals(new boolean[]{true, false, false}, roundTripBool(a.logicalAnd(b), boolInputs));
        assertArrayEquals(new boolean[]{true, true, true}, roundTripBool(a.logicalOr(b), boolInputs));
        assertArrayEquals(new boolean[]{false, true, false}, roundTripBool(a.logicalNot(), Map.of("a", new byte[]{1, 0, 1})));
    }

    @Test
    void coreReductionsRoundTrip() {
        Tensor x = new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);

        assertArrayEquals(new double[]{6.0, 15.0}, roundTrip(x.sum(1, false), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{2.0, 5.0}, roundTrip(x.mean(1, false), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{3.0, 6.0}, roundTrip(x.max(1, false), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{1.0, 4.0}, roundTrip(x.min(1, false), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{6.0, 120.0}, roundTrip(x.prod(1, false), matrixInputs()), 1e-6);
    }

    private static Tensor input(String label) {
        return new Tensor(new float[3], new int[]{3}, null, label, DataType.FLOAT32);
    }

    private static double[] roundTrip(Tensor output) {
        return roundTrip(output, numericInputs());
    }

    private static double[] roundTrip(Tensor output, Map<String, float[]> inputs) {
        output.setLabel("out");
        return OnnxRoundTripTestSupport.executeRoundTrip(output, "out", inputs);
    }

    private static boolean[] roundTripBool(Tensor output, Map<String, ?> inputs) {
        output.setLabel("out");
        ImportedOnnxModel imported = OnnxRoundTripTestSupport.exportImport(output);
        for (Map.Entry<String, ?> entry : inputs.entrySet()) {
            Object value = entry.getValue();
            if (value instanceof float[] floats) {
                imported.input(entry.getKey()).setData(floats);
            } else if (value instanceof byte[] bytes) {
                imported.input(entry.getKey()).setData(bytes);
            } else {
                throw new IllegalArgumentException("Unsupported test input type: " + value.getClass());
            }
        }
        imported.compile("out", CompileConfig.inference())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        return imported.output("out").toBooleanArrayCopy();
    }

    private static Map<String, float[]> numericInputs() {
        return Map.of(
                "x", new float[]{1f, 2f, 3f},
                "y", new float[]{4f, 2f, 6f}
        );
    }

    private static Map<String, float[]> matrixInputs() {
        return Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f});
    }
}

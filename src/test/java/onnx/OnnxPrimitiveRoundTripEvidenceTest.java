package onnx;

import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Pool2dOptions;
import tensor.options.Window2dOptions;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

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
    void unaryMathAndActivationOpsRoundTrip() {
        Tensor x = input("x");
        Map<String, float[]> signedInputs = Map.of("x", new float[]{-1f, 0f, 2f});
        Map<String, float[]> positiveInputs = Map.of("x", new float[]{1f, 4f, 9f});

        assertArrayEquals(new double[]{1.0, -0.0, -2.0}, roundTrip(x.neg(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{1.0, 0.0, 2.0}, roundTrip(x.abs(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{0.0, 0.0, 2.0}, roundTrip(x.relu(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{Math.tanh(-1.0), 0.0, Math.tanh(2.0)}, roundTrip(x.tanh(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{sigmoid(-1.0), 0.5, sigmoid(2.0)}, roundTrip(x.sigmoid(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{Math.exp(-1.0), 1.0, Math.exp(2.0)}, roundTrip(x.exp(), signedInputs), 1e-6);
        assertArrayEquals(new double[]{0.0, Math.log(4.0), Math.log(9.0)}, roundTrip(x.log(), positiveInputs), 1e-6);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, roundTrip(x.sqrt(), positiveInputs), 1e-6);
    }

    @Test
    void compareAndLogicalOpsRoundTrip() {
        Tensor x = input("x");
        Tensor y = input("y");

        assertArrayEquals(new boolean[]{false, true, false}, roundTripBool(x.equalTo(y), numericInputs()));
        assertArrayEquals(new boolean[]{false, false, false}, roundTripBool(x.greaterThan(y), numericInputs()));
        assertArrayEquals(new boolean[]{false, true, false}, roundTripBool(x.greaterOrEqual(y), numericInputs()));
        assertArrayEquals(new boolean[]{true, false, true}, roundTripBool(x.lessThan(y), numericInputs()));
        assertArrayEquals(new boolean[]{true, true, true}, roundTripBool(x.lessOrEqual(y), numericInputs()));

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
    void selectClampAndCastOpsRoundTrip() {
        Tensor x = input("x");
        Tensor y = input("y");
        Tensor condition = new Tensor(new byte[3], new int[]{3}, null, "condition", DataType.BOOL);

        assertArrayEquals(new double[]{1.0, 2.0, 6.0},
                roundTrip(Tensor.where(condition, x, y), Map.of(
                        "condition", new byte[]{1, 1, 0},
                        "x", new float[]{1f, 2f, 3f},
                        "y", new float[]{4f, 2f, 6f}
                )), 1e-6);
        assertArrayEquals(new double[]{1.5, 2.0, 3.0}, roundTrip(x.clampMin(1.5), xInputs()), 1e-6);
        assertArrayEquals(new double[]{1.0, 2.0, 2.5}, roundTrip(x.clampMax(2.5), xInputs()), 1e-6);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, roundTrip(x.cast(DataType.FLOAT64), xInputs()), 1e-9);
    }

    @Test
    void nnInferenceOpsRoundTrip() {
        Tensor input = new Tensor(new float[4], new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[6], new int[]{2, 3}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[3], new int[]{3}, null, "bias", DataType.FLOAT32);
        assertArrayEquals(new double[]{11.0, 22.0, 38.0, 13.0, 24.0, 48.0},
                roundTrip(input.linear(weight, bias), Map.of(
                        "input", new float[]{1f, 2f, 3f, 4f},
                        "weight", new float[]{1f, 0f, 2f, 0f, 1f, 3f},
                        "bias", new float[]{10f, 20f, 30f}
                )), 1e-6);

        Tensor pool = new Tensor(new float[16], new int[]{1, 1, 4, 4}, null, "pool", DataType.FLOAT32);
        Map<String, float[]> poolInputs = Map.of("pool", new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        });
        assertArrayEquals(new double[]{6.0, 8.0, 14.0, 16.0},
                roundTrip(pool.maxPool2d(Pool2dOptions.square(2)), poolInputs), 1e-6);
        assertArrayEquals(new double[]{3.5, 5.5, 11.5, 13.5},
                roundTrip(pool.avgPool2d(Pool2dOptions.square(2)), poolInputs), 1e-6);
    }

    @Test
    void layoutOpsRoundTrip() {
        Tensor x = new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);
        Tensor y = new Tensor(new float[6], new int[]{2, 3}, null, "y", DataType.FLOAT32);
        Tensor cube = new Tensor(new float[24], new int[]{2, 3, 4}, null, "cube", DataType.FLOAT32);
        Tensor row = new Tensor(new float[2], new int[]{1, 2}, null, "row", DataType.FLOAT32);
        Tensor singleton = new Tensor(new float[6], new int[]{2, 1, 3}, null, "singleton", DataType.FLOAT32);

        assertArrayEquals(new double[]{1.0, 4.0, 2.0, 5.0, 3.0, 6.0},
                roundTrip(x.transpose(), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                roundTrip(x.reshape(3, 2), matrixInputs()), 1e-6);

        Tensor flattened = cube.reshape(2, 12);
        flattened.setLabel("out");
        assertEquals("Flatten", Onnx.exportModel(flattened, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS))
                .proto().getGraph().getNode(0).getOpType());
        assertArrayEquals(sequence(24), roundTrip(flattened, Map.of("cube", sequenceF(24))), 1e-6);

        assertArrayEquals(new double[]{10.0, 20.0, 10.0, 20.0, 10.0, 20.0},
                roundTrip(row.expand(3, 2), Map.of("row", new float[]{10f, 20f})), 1e-6);
        assertArrayEquals(new double[]{-1, -1, -1, 1, 2, -1, 3, 4, -1},
                roundTrip(new Tensor(new float[4], new int[]{2, 2}, null, "pad", DataType.FLOAT32)
                        .pad(new int[]{1, 0}, new int[]{0, 1}, -1.0), Map.of("pad", new float[]{1f, 2f, 3f, 4f})), 1e-6);
        assertArrayEquals(new double[]{
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4,
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4
        }, roundTrip(new Tensor(new float[4], new int[]{2, 2}, null, "tile", DataType.FLOAT32)
                .tile(2, 3), Map.of("tile", new float[]{1f, 2f, 3f, 4f})), 1e-6);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6},
                roundTrip(singleton.squeeze(1), Map.of("singleton", new float[]{1f, 2f, 3f, 4f, 5f, 6f})), 1e-6);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6},
                roundTrip(x.expandDims(1), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{2.0, 3.0, 5.0, 6.0},
                roundTrip(x.slice(new int[]{0, 1}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 1}), matrixInputs()), 1e-6);
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60},
                roundTrip(Tensor.concat(0, x, y), Map.of(
                        "x", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "y", new float[]{10f, 20f, 30f, 40f, 50f, 60f}
                )), 1e-6);
    }

    @Test
    void col2ImRoundTripsThroughFold2d() {
        Tensor columns = new Tensor(new float[16], new int[]{1, 4, 4}, null, "columns", DataType.FLOAT32);
        Tensor folded = columns.fold2d(new int[]{1, 1, 3, 3}, Window2dOptions.of(2, 2));
        folded.setLabel("out");
        assertEquals("Col2Im", Onnx.exportModel(folded, OnnxExportOptions.defaults().withLeafTensorPolicy(OnnxLeafTensorPolicy.INPUTS))
                .proto().getGraph().getNode(0).getOpType());
        assertArrayEquals(new double[]{
                1, 4, 3,
                8, 20, 12,
                7, 16, 9
        }, roundTrip(folded, Map.of("columns", new float[]{
                1f, 2f, 4f, 5f,
                2f, 3f, 5f, 6f,
                4f, 5f, 7f, 8f,
                5f, 6f, 8f, 9f
        })), 1e-6);
    }

    @Test
    void indexAndScanOpsRoundTrip() {
        Tensor x = new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[2], new int[]{2}, null, "indices", DataType.INT32);

        assertArrayEquals(new double[]{3.0, 1.0, 6.0, 4.0},
                roundTrip(x.gatherAxis(indices, 1), Map.of(
                        "x", new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                        "indices", new int[]{2, 0}
                )), 1e-6);
        assertArrayEquals(new double[]{1.0, 0.0}, roundTrip(x.argMax(1, false), Map.of(
                "x", new float[]{1f, 4f, 4f, 7f, 6f, 7f}
        )), 1e-6);
        assertArrayEquals(new double[]{1.0, 3.0, 6.0, 4.0, 9.0, 15.0},
                roundTrip(x.cumSum(1), matrixInputs()), 1e-6);
    }

    @Test
    void softmaxFamilyRoundTrips() {
        Tensor x = new Tensor(new float[6], new int[]{2, 3}, null, "x", DataType.FLOAT32);
        float[] values = new float[]{1f, 2f, 3f, 1f, 1f, 1f};

        assertArrayEquals(softmaxRows(values, 2, 3), roundTrip(x.softmax(1), Map.of("x", values)), 1e-6);
        assertArrayEquals(logSoftmaxRows(values, 2, 3), roundTrip(x.logSoftmax(1), Map.of("x", values)), 1e-6);
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

    private static double[] roundTrip(Tensor output, Map<String, ?> inputs) {
        output.setLabel("out");
        return OnnxRoundTripTestSupport.executeRoundTrip(output, "out", inputs);
    }

    private static boolean[] roundTripBool(Tensor output, Map<String, ?> inputs) {
        output.setLabel("out");
        return OnnxRoundTripTestSupport.executeRoundTripBool(output, "out", inputs);
    }

    private static Map<String, float[]> numericInputs() {
        return Map.of(
                "x", new float[]{1f, 2f, 3f},
                "y", new float[]{4f, 2f, 6f}
        );
    }

    private static Map<String, float[]> xInputs() {
        return Map.of("x", new float[]{1f, 2f, 3f});
    }

    private static Map<String, float[]> matrixInputs() {
        return Map.of("x", new float[]{1f, 2f, 3f, 4f, 5f, 6f});
    }

    private static double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    private static float[] sequenceF(int length) {
        float[] out = new float[length];
        for (int i = 0; i < length; i++) {
            out[i] = i + 1;
        }
        return out;
    }

    private static double[] sequence(int length) {
        double[] out = new double[length];
        for (int i = 0; i < length; i++) {
            out[i] = i + 1;
        }
        return out;
    }

    private static double[] softmaxRows(float[] values, int rows, int cols) {
        double[] out = new double[values.length];
        for (int row = 0; row < rows; row++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int col = 0; col < cols; col++) {
                max = Math.max(max, values[row * cols + col]);
            }
            double sum = 0.0;
            for (int col = 0; col < cols; col++) {
                sum += Math.exp(values[row * cols + col] - max);
            }
            for (int col = 0; col < cols; col++) {
                out[row * cols + col] = Math.exp(values[row * cols + col] - max) / sum;
            }
        }
        return out;
    }

    private static double[] logSoftmaxRows(float[] values, int rows, int cols) {
        double[] softmax = softmaxRows(values, rows, cols);
        for (int i = 0; i < softmax.length; i++) {
            softmax[i] = Math.log(softmax[i]);
        }
        return softmax;
    }
}

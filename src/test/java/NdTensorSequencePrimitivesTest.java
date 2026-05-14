import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NdTensorSequencePrimitivesTest {

    @Test
    void factoriesAndShapeUtilitiesExposeCommonTensorBoilerplate() {
        Tensor zeros = Tensor.zeros(new int[]{2, 3}, DataType.FLOAT32, "z");
        Tensor ones = Tensor.ones(new int[]{2, 2}, DataType.INT64, "o");
        Tensor range = Tensor.arange(1, 7, 2, DataType.INT32);

        assertEquals(2, zeros.rank());
        assertEquals(6, zeros.size());
        assertEquals(3, zeros.lastDim());
        assertTrue(zeros.shapeEquals(2, 3));
        assertArrayEquals(new int[]{2, 3}, zeros.shapeCopy());
        assertEquals(DataType.FLOAT32, zeros.getDataType());
        assertArrayEquals(new double[]{0, 0, 0, 0, 0, 0}, zeros.toDoubleArrayCopy(), 1e-9);
        assertEquals(DataType.INT64, ones.getDataType());
        assertArrayEquals(new double[]{1, 1, 1, 1}, ones.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 3, 5}, range.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void ndLinearSupportsThreeDimensionalInputAndRowBiasWithGradients() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
        }, new int[]{2, 2, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 10,
                2, 20,
                3, 30
        }, new int[]{3, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{0.5, -0.5}, new int[]{1, 2}, null, "bias", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);
        bias.setRequiresGrad(true);

        Tensor loss = input.linear(weight, bias).sum();
        CompiledGraph.compile(loss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor out = input.linear(weight, bias);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                14.5, 139.5,
                32.5, 319.5,
                50.5, 499.5,
                68.5, 679.5
        }, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{
                11, 22, 33,
                11, 22, 33,
                11, 22, 33,
                11, 22, 33
        }, input.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{22, 22, 26, 26, 30, 30}, weight.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{1, 2}, bias.getGradient().getShape());
        assertArrayEquals(new double[]{4, 4}, bias.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void stackAndUnstackComposeViewsAndPreserveGradientFlow() {
        Tensor a = matrix("a", 1, 2, 3, 4);
        Tensor b = matrix("b", 5, 6, 7, 8);
        Tensor c = matrix("c", 9, 10, 11, 12);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        c.setRequiresGrad(true);

        Tensor stacked = Tensor.stack(1, a, b, c);
        CompiledGraph.compile(stacked, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{2, 3, 2}, stacked.getShape());
        assertArrayEquals(new double[]{
                1, 2,
                5, 6,
                9, 10,
                3, 4,
                7, 8,
                11, 12
        }, stacked.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1, 1}, c.getGradient().toDoubleArrayCopy(), 1e-9);

        Tensor[] parts = stacked.unstack(1);
        Tensor restacked = Tensor.stack(1, parts);
        CompiledGraph.compile(restacked, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertEquals(3, parts.length);
        assertArrayEquals(stacked.toDoubleArrayCopy(), restacked.toDoubleArrayCopy(), 1e-9);

        assertThrows(IllegalArgumentException.class, () -> Tensor.stack(0, a, Tensor.ones(new int[]{1, 4})));
    }

    @Test
    void takeAndSliceAxisWrapExistingGradientSafeIndexingPrimitives() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 4}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);

        Tensor taken = x.take(1, new int[]{3, 1});
        Tensor sliced = x.sliceAxis(1, 1, 3);
        Tensor loss = taken.sum();
        CompiledGraph.compile(loss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        CompiledGraph.compile(taken, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(sliced, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2}, taken.getShape());
        assertArrayEquals(new double[]{4, 2, 8, 6}, taken.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0, 1, 0, 1, 0, 1, 0, 1}, x.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{2, 2}, sliced.getShape());
        assertArrayEquals(new double[]{2, 3, 6, 7}, sliced.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void maskedReductionsIgnorePaddedSequencePositions() {
        Tensor x = new Tensor(new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12
        }, new int[]{2, 3, 2}, null, "x", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);

        Tensor sum = x.sum(1, mask);
        Tensor mean = x.mean(1, mask);
        CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{4, 6, 7, 8}, sum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 3, 7, 8}, mean.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void maskedCrossEntropyNormalizesByValidSamplesAndZerosMaskedGradients() {
        Tensor logits = new Tensor(new double[]{
                1, 2, 3,
                2, 1, 0,
                0, 1, 0,
                3, 0, 0
        }, new int[]{2, 2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0, 0, 1,
                1, 0, 0,
                0, 1, 0,
                1, 0, 0
        }, new int[]{2, 2, 3}, null, "targets", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{2, 2}, null, "mask", DataType.BOOL);
        logits.setRequiresGrad(true);

        Tensor loss = logits.crossEntropyLoss(targets, 2, mask);
        CompiledGraph.compile(loss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        double[] logitsData = logits.toDoubleArrayCopy();
        double[] targetData = targets.toDoubleArrayCopy();
        boolean[] valid = mask.toBooleanArrayCopy();
        assertArrayEquals(new double[]{expectedMaskedCrossEntropy(logitsData, targetData, valid, 3)},
                loss.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(expectedMaskedCrossEntropyGrad(logitsData, targetData, valid, 3),
                logits.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static Tensor matrix(String label, double... values) {
        return new Tensor(values, new int[]{2, 2}, null, label, DataType.FLOAT64);
    }

    private static double expectedMaskedCrossEntropy(double[] logits, double[] targets, boolean[] valid, int classes) {
        double total = 0.0d;
        int count = 0;
        for (int sample = 0; sample < valid.length; sample++) {
            if (!valid[sample]) {
                continue;
            }
            count++;
            int base = sample * classes;
            double logDenominator = logSumExp(logits, base, classes);
            for (int c = 0; c < classes; c++) {
                total -= targets[base + c] * (logits[base + c] - logDenominator);
            }
        }
        return total / count;
    }

    private static double[] expectedMaskedCrossEntropyGrad(double[] logits, double[] targets, boolean[] valid, int classes) {
        double[] grad = new double[logits.length];
        int count = 0;
        for (boolean value : valid) {
            if (value) {
                count++;
            }
        }
        for (int sample = 0; sample < valid.length; sample++) {
            if (!valid[sample]) {
                continue;
            }
            int base = sample * classes;
            double denominator = 0.0d;
            double max = max(logits, base, classes);
            for (int c = 0; c < classes; c++) {
                denominator += Math.exp(logits[base + c] - max);
            }
            for (int c = 0; c < classes; c++) {
                double probability = Math.exp(logits[base + c] - max) / denominator;
                grad[base + c] = (probability - targets[base + c]) / count;
            }
        }
        return grad;
    }

    private static double logSumExp(double[] values, int base, int classes) {
        double max = max(values, base, classes);
        double sum = 0.0d;
        for (int c = 0; c < classes; c++) {
            sum += Math.exp(values[base + c] - max);
        }
        return max + Math.log(sum);
    }

    private static double max(double[] values, int base, int classes) {
        double max = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < classes; c++) {
            max = Math.max(max, values[base + c]);
        }
        return max;
    }
}

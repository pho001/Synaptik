import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SelectExecutionTest {

    @Test
    void selectRemovesAxisAndExtractsRequestedSlice() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor y = x.select(1, 2);

        CompiledGraph compiledGraph = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, y.getShape());
        assertArrayEquals(new double[]{3.0, 6.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SELECT));
    }

    @Test
    void selectSupportsNegativeIndices() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor y = x.select(1, -1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 6.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectAliasesBaseStorageAtRuntime() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor y = x.select(0, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        x.setDataAt(3, 40.0);
        x.setDataAt(4, 50.0);
        x.setDataAt(5, 60.0);

        assertArrayEquals(new double[]{40.0, 50.0, 60.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectRespectsOffsetAndStrideOnPermutedInput() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor y = x.permute(1, 0).select(0, 1);

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{2.0, 5.0}, y.toDoubleArrayCopy(), 1e-9);

        x.setDataAt(1, 20.0);
        x.setDataAt(4, 50.0);
        assertArrayEquals(new double[]{20.0, 50.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedReductionCorrectly() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor loss = x.select(0, 1).sum();

        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{15.0}, loss.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedMeanAndMinMaxReductions() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);

        Tensor mean = x.select(0, 1).mean();
        Tensor min = x.select(0, 1).min();
        Tensor max = x.select(0, 1).max();

        CompiledGraph.compile(max, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(min, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{5.0}, mean.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4.0}, min.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{6.0}, max.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedSoftmaxAndLogSoftmax() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);

        Tensor softmax = x.select(0, 1).softmax(0);
        Tensor logSoftmax = x.select(0, 1).logSoftmax(0);

        CompiledGraph.compile(logSoftmax, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(softmax, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expected = new double[]{
                0.09003057317038046,
                0.24472847105479764,
                0.6652409557748218
        };
        assertArrayEquals(expected, softmax.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{
                Math.log(expected[0]),
                Math.log(expected[1]),
                Math.log(expected[2])
        }, logSoftmax.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedDenseTargetLosses() {
        Tensor logits = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0, 0, 1,
                0, 1, 0
        }, new int[]{2, 3}, null, "targets", DataType.FLOAT64);

        Tensor nll = logits.logSoftmax(1).select(0, 1).nllLoss(targets.select(0, 1), 0);
        Tensor crossEntropy = logits.select(0, 1).crossEntropyLoss(targets.select(0, 1), 0);

        CompiledGraph.compile(crossEntropy, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(nll, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1.4076059644443804}, nll.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.4076059644443804}, crossEntropy.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedGatherTakeAlongAxisAndScatterAdd() {
        Tensor base = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "base", DataType.FLOAT64);

        Tensor gather = base.select(0, 1).gather(
                new Tensor(new int[]{2}, new int[]{1}, null, "gatherIndices", DataType.INT32),
                0
        );

        Tensor takeAlongAxis = base.select(0, 1).takeAlongAxis(
                new Tensor(new int[]{2, 1}, new int[]{2}, null, "taaIndices", DataType.INT32),
                0
        );

        Tensor scatterAdd = base.select(0, 1).scatterAdd(
                new Tensor(new int[]{2}, new int[]{1}, null, "scatterIndices", DataType.INT32),
                new Tensor(new double[]{10}, new int[]{1}, null, "scatterSrc", DataType.FLOAT64),
                0
        );

        CompiledGraph.compile(scatterAdd, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(takeAlongAxis, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(gather, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{6.0}, gather.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{6.0, 5.0}, takeAlongAxis.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4.0, 5.0, 16.0}, scatterAdd.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectSupportsBackwardThroughMinAndMaxReductions() {
        Tensor xMin = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "xMin", DataType.FLOAT64);
        xMin.setRequiresGrad(true);
        Tensor minLoss = xMin.select(0, 1).min();

        CompiledGraph.compile(minLoss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0,
                1, 0, 0
        }, xMin.getGradient().toDoubleArrayCopy(), 1e-9);

        Tensor xMax = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "xMax", DataType.FLOAT64);
        xMax.setRequiresGrad(true);
        Tensor maxLoss = xMax.select(0, 1).max();

        CompiledGraph.compile(maxLoss, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0,
                0, 0, 1
        }, xMax.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedCompareWithoutMaterializingInputView() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                0, 3, 3,
                5, 4, 7
        }, new int[]{2, 3}, null, "b", DataType.FLOAT64);
        Tensor y = a.select(0, 1).greaterThan(b.select(0, 1));

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{false, true, false}, y.toBooleanArrayCopy());
    }

    @Test
    void selectCanFeedLogicalOpsWithoutMaterializingInputView() {
        Tensor mask = new Tensor(new byte[]{
                1, 0, 1,
                0, 1, 0
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);
        Tensor other = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0
        }, new int[]{2, 3}, null, "other", DataType.BOOL);
        Tensor y = mask.select(0, 1).logicalOr(other.select(0, 1));

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, true, false}, y.toBooleanArrayCopy());
    }

    @Test
    void selectCanFeedWhereWithoutMaterializingInputView() {
        Tensor mask = new Tensor(new byte[]{
                1, 0, 1,
                0, 1, 0
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "ifTrue", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "ifFalse", DataType.FLOAT64);

        Tensor y = Tensor.where(mask.select(0, 1), ifTrue.select(0, 1), ifFalse.select(0, 1));

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{4.0, 50.0, 6.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void selectCanFeedBoolReductions() {
        Tensor mask = new Tensor(new byte[]{
                1, 0, 1,
                0, 1, 1
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);

        Tensor all = mask.select(0, 1).all();
        Tensor any = mask.select(0, 1).any();

        CompiledGraph.compile(any, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(all, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{false}, all.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{true}, any.toBooleanArrayCopy());
    }

    @Test
    void selectSupportsBackwardThroughBinaryMinAndMax() {
        Tensor aMin = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "aMin", DataType.FLOAT64);
        Tensor bMin = new Tensor(new double[]{
                6, 5, 4,
                3, 2, 1
        }, new int[]{2, 3}, null, "bMin", DataType.FLOAT64);
        aMin.setRequiresGrad(true);
        bMin.setRequiresGrad(true);
        Tensor minLoss = aMin.select(0, 1).min(bMin.select(0, 1)).sum();

        CompiledGraph compiledMin = CompiledGraph.compile(minLoss, CompileConfig.training());
        assertTrue(containsOp(compiledMin, Operation.OpType.MIN_GRAD));
        compiledMin.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0,
                0, 0, 0
        }, aMin.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{
                0, 0, 0,
                1, 1, 1
        }, bMin.getGradient().toDoubleArrayCopy(), 1e-9);

        Tensor aMax = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "aMax", DataType.FLOAT64);
        Tensor bMax = new Tensor(new double[]{
                6, 5, 4,
                3, 2, 1
        }, new int[]{2, 3}, null, "bMax", DataType.FLOAT64);
        aMax.setRequiresGrad(true);
        bMax.setRequiresGrad(true);
        Tensor maxLoss = aMax.select(0, 1).max(bMax.select(0, 1)).sum();

        CompiledGraph compiledMax = CompiledGraph.compile(maxLoss, CompileConfig.training());
        assertTrue(containsOp(compiledMax, Operation.OpType.MAX_GRAD));
        compiledMax.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0, 0, 0,
                1, 1, 1
        }, aMax.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{
                0, 0, 0,
                0, 0, 0
        }, bMax.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}

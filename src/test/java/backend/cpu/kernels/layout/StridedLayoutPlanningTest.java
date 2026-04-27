package backend.cpu.kernels.layout;

import backend.cpu.plan.CpuPreparedInput;
import backend.cpu.kernels.elementwise.strided.StridedLayoutDecision;
import backend.cpu.kernels.elementwise.strided.StridedPathEligibility;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import backend.cpu.kernels.plan.PreparedTypeContract;
import config.backend.CpuKernelConfig;
import operations.elementwise.binary.add;
import operations.elementwise.where.where;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

public class StridedLayoutPlanningTest {
    @Test
    public void resolvesSingleNonContiguousBinaryInputToSelectiveMaterialization() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(config(0, 0, 0, 0, 0));
        Tensor left = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "right_contig", DataType.FLOAT64);
        Tensor out = new Tensor(new int[]{2, 2}, List.of(left, right), new add(), "out", DataType.FLOAT64);

        StridedLayoutDecision decision = StridedPathEligibility.resolve(
                new add(),
                List.of(left, right),
                out,
                DataType.FLOAT64,
                planner
        );

        assertEquals(StridedLayoutDecision.MATERIALIZE_INPUT_0, decision);
    }

    @Test
    public void preparedInputPlannerMaterializesOnlySelectedInput() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(config(1_000_000, 1_000_000, 1_000_000, 1_000_000, 1_000_000));
        Tensor left = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "right_contig", DataType.FLOAT64);
        Tensor out = new Tensor(new int[]{2, 2}, List.of(left, right), new add(), "out", DataType.FLOAT64);

        PreparedInputsResult result = PreparedInputPlanner.plan(
                new add(),
                List.of(left, right),
                out,
                new PreparedTypeContract(DataType.FLOAT64, List.of(DataType.FLOAT64, DataType.FLOAT64)),
                planner,
                StridedLayoutDecision.MATERIALIZE_INPUT_0
        );

        assertEquals(1, result.preparedInputs().size());
        CpuPreparedInput prepared = result.preparedInputs().get(0);
        assertEquals(0, prepared.inputIndex());
        assertNotSame(left, result.runtimeInputs().get(0));
        assertSame(right, result.runtimeInputs().get(1));
    }

    @Test
    public void cheapF64ThresholdCanForceMaterializationEvenWhenGenericThresholdIsHigh() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(config(1_000_000, 0, 1_000_000, 1_000_000, 1_000_000));
        Tensor left = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "right_contig", DataType.FLOAT64);
        Tensor out = new Tensor(new int[]{2, 2}, List.of(left, right), new add(), "out", DataType.FLOAT64);

        StridedLayoutDecision decision = StridedPathEligibility.resolve(
                new add(),
                List.of(left, right),
                out,
                DataType.FLOAT64,
                planner
        );

        assertEquals(StridedLayoutDecision.MATERIALIZE_INPUT_0, decision);
    }

    @Test
    public void whereThresholdCanForceMaterializationIndependentlyOfGenericThreshold() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(config(1_000_000, 1_000_000, 1_000_000, 1_000_000, 0));
        Tensor cond = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, new int[]{1, 2}, null, "cond_noncontig", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "true_contig", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "false_contig", DataType.FLOAT64);
        Tensor out = new Tensor(new int[]{2, 2}, List.of(cond, ifTrue, ifFalse), new where(), "out", DataType.FLOAT64);

        StridedLayoutDecision decision = StridedPathEligibility.resolve(
                new where(),
                List.of(cond, ifTrue, ifFalse),
                out,
                DataType.FLOAT64,
                planner
        );

        assertEquals(StridedLayoutDecision.MATERIALIZE_ALL, decision);
    }

    @Test
    public void rowBroadcastStrideClassDoesNotUseCheapMaterializationThreshold() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(config(1_000_000, 0, 1_000_000, 1_000_000, 1_000_000));
        Tensor left = new Tensor(new double[]{1, 2, 0, 0}, new int[]{2, 2}, new int[]{1, 0}, null, "left_row_broadcast", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "right_contig", DataType.FLOAT64);
        Tensor out = new Tensor(new int[]{2, 2}, List.of(left, right), new add(), "out", DataType.FLOAT64);

        StridedLayoutDecision decision = StridedPathEligibility.resolve(
                new add(),
                List.of(left, right),
                out,
                DataType.FLOAT64,
                planner
        );

        assertEquals(StridedLayoutDecision.KEEP_STRIDED, decision);
    }

    private static CpuKernelConfig config(
            int contiguousThreshold,
            int cheapF64Threshold,
            int cheapF32Threshold,
            int cheapBF16Threshold,
            int whereThreshold
    ) {
        CpuKernelConfig base = CpuKernelConfig.defaultsInference();
        return new CpuKernelConfig(
                base.loopUnrollFactor(),
                base.matMulTileM(),
                base.matMulTileN(),
                base.matMulTileK(),
                base.cheapVectorMinSize(),
                base.transcendentalVectorMinSize(),
                base.fusedCheapVectorMinSize(),
                base.fusedTranscendentalVectorMinSize(),
                base.reductionVectorMinSize(),
                base.attentionVectorMinSize(),
                base.cheapParallelMinSize(),
                base.transcendentalParallelMinSize(),
                base.fusedCheapParallelMinSize(),
                base.fusedTranscendentalParallelMinSize(),
                base.reductionParallelMinSize(),
                base.attentionParallelMinSize(),
                contiguousThreshold,
                cheapF64Threshold,
                cheapF32Threshold,
                cheapBF16Threshold,
                whereThreshold,
                base.lowCostTargetChunksPerWorker(),
                base.mediumCostTargetChunksPerWorker(),
                base.highCostTargetChunksPerWorker(),
                base.minScalarChunkSize(),
                base.minVectorChunkSize(),
                base.minReductionChunkSize(),
                base.commonPoolLowCostMaxWorkPerWorker(),
                base.fusedCheapContiguousAsmVectorWidth(),
                base.fusedCheapStridedAsmVectorWidth(),
                base.fusedNonCheapContiguousAsmVectorWidth(),
                base.fusedNonCheapStridedAsmVectorWidth(),
                base.sumAccuracyMode(),
                base.matMulParallelMinSize(),
                base.attentionMatMulPolicy(),
                base.matMulMicroKernel(),
                base.attentionMatMulMicroKernel(),
                base.attentionMatMulTileM(),
                base.attentionMatMulTileN(),
                base.attentionMatMulTileK()
        );
    }
}

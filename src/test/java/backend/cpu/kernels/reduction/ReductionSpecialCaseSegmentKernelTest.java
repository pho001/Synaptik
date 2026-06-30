package backend.cpu.kernels.reduction;

import backend.contract.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageView;
import runtime.execution.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.runtime.RuntimeConfig;
import runtime.execution.PreparedStepMetadata;
import operations.Operation;
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.argMax;
import operations.reduction.cumSum;
import operations.reduction.reduceProd;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ReductionSpecialCaseSegmentKernelTest {
    @Test
    void reduceProdReadsStridedF32MemorySegmentInputAndWritesStridedMemorySegmentOutput() {
        reduceProd op = new reduceProd(1, false);
        Tensor input = new Tensor(new int[]{2, 3}, new int[]{1, 2}, 1,
                null, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{2}, new int[]{2}, 1,
                List.of(input), op, "output", DataType.FLOAT32);
        float[] inputStorage = {0.0f, 2.0f, 5.0f, 3.0f, 6.0f, 4.0f, 7.0f};
        float[] outputStorage = {-1.0f, -1.0f, -1.0f, -1.0f};

        new CpuReduceProdKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(CpuStorageView.segment(
                        DataType.FLOAT32,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{2, 3},
                        new int[]{1, 2},
                        1,
                        6)),
                CpuStorageView.segment(
                        DataType.FLOAT32,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{2},
                        new int[]{2},
                        1,
                        2)));

        assertArrayEquals(new float[]{-1.0f, 24.0f, -1.0f, 210.0f}, outputStorage, 0.0f);
    }

    @Test
    void cumSumReadsStridedInt64MemorySegmentInputAndWritesReverseExclusiveSegmentOutput() {
        cumSum op = new cumSum(1, true, true);
        Tensor input = new Tensor(new int[]{2, 3}, new int[]{1, 2}, 1,
                null, null, "input", DataType.INT64);
        Tensor output = new Tensor(new int[]{2, 3}, new int[]{1, 2}, 1,
                List.of(input), op, "output", DataType.INT64);
        long[] inputStorage = {0L, 2L, 5L, 3L, 6L, 4L, 7L};
        long[] outputStorage = {-1L, -1L, -1L, -1L, -1L, -1L, -1L};

        new CpuCumSumKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(CpuStorageView.segment(
                        DataType.INT64,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{2, 3},
                        new int[]{1, 2},
                        1,
                        6)),
                CpuStorageView.segment(
                        DataType.INT64,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{2, 3},
                        new int[]{1, 2},
                        1,
                        6)));

        assertArrayEquals(new long[]{-1L, 7L, 13L, 4L, 7L, 0L, 0L}, outputStorage);
    }

    @Test
    void argMaxReadsF32MemorySegmentTiesAndWritesStridedInt64MemorySegmentOutput() {
        argMax op = new argMax(1, false, ArgMaxTiePolicy.LAST_INDEX);
        Tensor input = new Tensor(new int[]{2, 4}, new int[]{1, 2}, 1,
                null, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{2}, new int[]{2}, 1,
                List.of(input), op, "output", DataType.INT64);
        float[] inputStorage = {0.0f, 1.0f, 3.0f, 7.0f, 3.0f, 7.0f, 1.0f, 2.0f, 3.0f};
        long[] outputStorage = {-1L, -1L, -1L, -1L};

        new CpuArgMaxKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(CpuStorageView.segment(
                        DataType.FLOAT32,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{2, 4},
                        new int[]{1, 2},
                        1,
                        8)),
                CpuStorageView.segment(
                        DataType.INT64,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{2},
                        new int[]{2},
                        1,
                        2)));

        assertArrayEquals(new long[]{-1L, 2L, -1L, 3L}, outputStorage);
    }

    private static CpuKernelCall call(
            Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        CpuNodeExecutionPlan plan = plan(output.dtype());
        PreparedStepMetadata metadata = new PreparedStepMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(),
                testsupport.MetadataArtifacts.noopExecutable(),
                runtime.execution.InputResidencyRequirement.cpuReadableAll(),
                runtime.execution.OutputResidencyEffect.cpuCurrentPreserveNative()
                );
        CpuKernelContext context = new CpuKernelContext(
                1,
                inputTensors.stream().map(tensor -> 0).toList(),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                operation);
        return new CpuKernelCall(operation, inputTensors, outputTensor, inputs, output, plan, context, null);
    }

    private static CpuNodeExecutionPlan plan(DataType dtype) {
        return new CpuNodeExecutionPlan(
                new CpuLayoutPlan(StridedLayoutDecision.NONE, dtype, 0, null, null, List.of()),
                null,
                false,
                1,
                0,
                null,
                null,
                null,
                null,
                null);
    }
}

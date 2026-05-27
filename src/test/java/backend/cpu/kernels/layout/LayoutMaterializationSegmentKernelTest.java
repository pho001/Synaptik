package backend.cpu.kernels.layout;

import backend.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.layout.pad;
import operations.layout.tile;
import operations.layout.unfold2d;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.options.Window2dOptions;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LayoutMaterializationSegmentKernelTest {
    @Test
    void padReadsStridedBf16MemorySegmentInputAndWritesRoundedMemorySegmentOutput() {
        Tensor input = new Tensor(new int[]{2}, new int[]{2}, 1, null, null, "input", DataType.BFLOAT16);
        Tensor out = new Tensor(new int[]{4}, List.of(input), new pad(new int[]{1}, new int[]{1}, -1.0d),
                "out", DataType.BFLOAT16);
        short[] inputStorage = {bf16(0.0f), bf16(2.0f), bf16(0.0f), bf16(4.0f), bf16(0.0f)};
        short[] outputStorage = new short[4];

        new CpuPadKernel().execute(call(
                new pad(new int[]{1}, new int[]{1}, -1.0d),
                List.of(input),
                out,
                List.of(CpuStorageView.segment(
                        DataType.BFLOAT16,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{2},
                        new int[]{2},
                        1,
                        2)),
                CpuStorageView.segment(
                        DataType.BFLOAT16,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{4},
                        new int[]{1},
                        0,
                        4)));

        assertArrayEquals(new short[]{bf16(-1.0f), bf16(2.0f), bf16(4.0f), bf16(-1.0f)}, outputStorage);
    }

    @Test
    void tilePreservesBoolValuesAcrossMemorySegmentViews() {
        Tensor input = new Tensor(new int[]{2}, new int[]{2}, 0, null, null, "input", DataType.BOOL);
        Tensor out = new Tensor(new int[]{4}, List.of(input), new tile(new int[]{2}), "out", DataType.BOOL);
        byte[] inputStorage = {1, 0, 0};
        byte[] outputStorage = new byte[4];

        new CpuTileKernel().execute(call(
                new tile(new int[]{2}),
                List.of(input),
                out,
                List.of(CpuStorageView.segment(
                        DataType.BOOL,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{2},
                        new int[]{2},
                        0,
                        2)),
                CpuStorageView.segment(
                        DataType.BOOL,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{4},
                        new int[]{1},
                        0,
                        4)));

        assertArrayEquals(new byte[]{1, 0, 1, 0}, outputStorage);
    }

    @Test
    void unfold2dReadsOffsetMemorySegmentInputAndWritesColumnsToMemorySegmentOutput() {
        Window2dOptions options = Window2dOptions.of(2, 2);
        Tensor input = new Tensor(new int[]{1, 1, 2, 2}, new int[]{4, 4, 2, 1}, 1,
                null, null, "input", DataType.FLOAT32);
        Tensor out = new Tensor(new int[]{1, 4, 1}, List.of(input), new unfold2d(options), "out", DataType.FLOAT32);
        float[] inputStorage = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
        float[] outputStorage = new float[4];

        new CpuUnfold2dKernel().execute(call(
                new unfold2d(options),
                List.of(input),
                out,
                List.of(CpuStorageView.segment(
                        DataType.FLOAT32,
                        MemorySegment.ofArray(inputStorage),
                        new int[]{1, 1, 2, 2},
                        new int[]{4, 4, 2, 1},
                        1,
                        4)),
                CpuStorageView.segment(
                        DataType.FLOAT32,
                        MemorySegment.ofArray(outputStorage),
                        new int[]{1, 4, 1},
                        new int[]{4, 1, 1},
                        0,
                        4)));

        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, outputStorage, 0.0f);
    }

    @Test
    void unfold2dRejectsBoolMemorySegmentView() {
        Window2dOptions options = Window2dOptions.of(1, 1);
        Tensor input = new Tensor(new byte[]{1}, new int[]{1, 1, 1, 1}, null, "input", DataType.BOOL);
        Tensor out = new Tensor(new int[]{1, 1, 1}, List.of(input), new unfold2d(options), "out", DataType.BOOL);

        CpuKernelCall kernelCall = call(
                new unfold2d(options),
                List.of(input),
                out,
                List.of(CpuStorageView.segment(
                        DataType.BOOL,
                        MemorySegment.ofArray(new byte[]{1}),
                        new int[]{1, 1, 1, 1},
                        new int[]{1, 1, 1, 1},
                        0,
                        1)),
                CpuStorageView.segment(
                        DataType.BOOL,
                        MemorySegment.ofArray(new byte[1]),
                        new int[]{1, 1, 1},
                        new int[]{1, 1, 1},
                        0,
                        1));

        assertThrows(UnsupportedOperationException.class, () -> new CpuUnfold2dKernel().execute(kernelCall));
    }

    private static CpuKernelCall call(
            operations.Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        CpuNodeExecutionPlan plan = plan(output.dtype());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(),
                null);
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

    private static short bf16(float value) {
        return TensorDTypeOps.toBFloat16Bits(value);
    }
}

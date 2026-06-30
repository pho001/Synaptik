package backend.cpu.kernels.nn;

import backend.contract.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuNodeWorkspace;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import operations.nn.conv.conv2d;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class ConvPoolSegmentKernelTest {
    @Test
    void conv2dF32ReadsAndWritesMemorySegmentStorageViews() {
        conv2d op = new conv2d(Conv2dOptions.defaults(), true);
        Tensor input = tensor(new int[]{1, 1, 3, 3}, new int[]{9, 9, 3, 1}, 1, DataType.FLOAT32);
        Tensor weight = tensor(new int[]{1, 1, 2, 2}, new int[]{4, 4, 2, 1}, 1, DataType.FLOAT32);
        Tensor bias = tensor(new int[]{1}, new int[]{1}, 1, DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{1, 1, 2, 2}, new int[]{4, 4, 2, 1}, 1,
                List.of(input, weight, bias), op, "output", DataType.FLOAT32);

        float sentinel = -99.0f;
        float[] inputStorage = {sentinel, 1, 2, 3, 4, 5, 6, 7, 8, 9, sentinel};
        float[] weightStorage = {sentinel, 1, 0, 0, -1, sentinel};
        float[] biasStorage = {sentinel, 0.5f, sentinel};
        float[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

        new CpuConv2dKernel().execute(call(
                op,
                List.of(input, weight, bias),
                output,
                List.of(
                        f32Segment(inputStorage, input),
                        f32Segment(weightStorage, weight),
                        f32Segment(biasStorage, bias)
                ),
                f32Segment(outputStorage, output),
                null));

        assertArrayEquals(new float[]{sentinel, -3.5f, -3.5f, -3.5f, -3.5f, sentinel}, outputStorage, 1.0e-6f);
    }

    @Test
    void maxPool2dF32ReadsAndWritesMemorySegmentStorageViews() {
        maxPool2d op = new maxPool2d(Pool2dOptions.square(2));
        Tensor input = tensor(new int[]{1, 1, 4, 4}, new int[]{16, 16, 4, 1}, 1, DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{1, 1, 2, 2}, new int[]{4, 4, 2, 1}, 1,
                List.of(input), op, "output", DataType.FLOAT32);

        float sentinel = -99.0f;
        float[] inputStorage = {
                sentinel,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                sentinel
        };
        float[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

        new CpuMaxPool2dKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(f32Segment(inputStorage, input)),
                f32Segment(outputStorage, output),
                CpuNodeWorkspace.withIntWorkspace(output.getFlatDataSize())));

        assertArrayEquals(new float[]{sentinel, 6, 8, 14, 16, sentinel}, outputStorage, 1.0e-6f);
    }

    @Test
    void avgPool2dF32ReadsAndWritesMemorySegmentStorageViews() {
        avgPool2d op = new avgPool2d(Pool2dOptions.square(2));
        Tensor input = tensor(new int[]{1, 1, 4, 4}, new int[]{16, 16, 4, 1}, 1, DataType.FLOAT32);
        Tensor output = new Tensor(new int[]{1, 1, 2, 2}, new int[]{4, 4, 2, 1}, 1,
                List.of(input), op, "output", DataType.FLOAT32);

        float sentinel = -99.0f;
        float[] inputStorage = {
                sentinel,
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                sentinel
        };
        float[] outputStorage = {sentinel, sentinel, sentinel, sentinel, sentinel, sentinel};

        new CpuAvgPool2dKernel().execute(call(
                op,
                List.of(input),
                output,
                List.of(f32Segment(inputStorage, input)),
                f32Segment(outputStorage, output),
                null));

        assertArrayEquals(new float[]{sentinel, 3.5f, 5.5f, 11.5f, 13.5f, sentinel}, outputStorage, 1.0e-6f);
    }

    private static Tensor tensor(int[] shape, int[] strides, int storageOffset, DataType dtype) {
        return new Tensor(shape, strides, storageOffset, null, null, "input", dtype);
    }

    private static CpuStorageView f32Segment(float[] storage, Tensor tensor) {
        return CpuStorageView.segment(
                DataType.FLOAT32,
                MemorySegment.ofArray(storage),
                tensor.getShapeUnsafe(),
                tensor.getStridesUnsafe(),
                tensor.getStorageOffsetUnsafe(),
                tensor.getFlatDataSize());
    }

    private static CpuKernelCall call(
            Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output,
            CpuNodeWorkspace workspace
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
        return new CpuKernelCall(operation, inputTensors, outputTensor, inputs, output, plan, context, workspace);
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

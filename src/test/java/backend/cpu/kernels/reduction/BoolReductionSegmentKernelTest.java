package backend.cpu.kernels.reduction;

import backend.contract.ComputeBackend;
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
import operations.Operation;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class BoolReductionSegmentKernelTest {
    @Test
    void allAndAnyAllReduceReadStridedMemorySegmentInputAndWriteMemorySegmentOutput() {
        assertAllReduce(new reduceAll(-1), false);
        assertAllReduce(new reduceAny(-1), true);
    }

    @Test
    void allAndAnyAxisReduceReadAndWriteStridedMemorySegments() {
        assertAxisReduce(new reduceAll(1), new byte[]{1, 0});
        assertAxisReduce(new reduceAny(1), new byte[]{1, 1});
    }

    private static void assertAllReduce(Operation operation, boolean expected) {
        Tensor inputTensor = tensor(new int[]{2, 3}, new int[]{1, 2}, 1, "input");
        Tensor outputTensor = tensor(new int[]{1}, new int[]{1}, 1, "output");
        byte[] inputStorage = bools(0, 1, 0, 1, 1, 1, 0);
        byte[] outputStorage = bools(1, 1);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(outputStorage, new int[]{1}, new int[]{1}, 1)
        ));

        assertArrayEquals(bools(1, expected ? 1 : 0), outputStorage);
    }

    private static void assertAxisReduce(Operation operation, byte[] expected) {
        Tensor inputTensor = tensor(new int[]{2, 3}, new int[]{1, 2}, 1, "input");
        Tensor outputTensor = tensor(new int[]{2}, new int[]{2}, 1, "output");
        byte[] inputStorage = bools(0, 1, 0, 1, 1, 1, 0);
        byte[] outputStorage = bools(0, 0, 0, 0);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(outputStorage, new int[]{2}, new int[]{2}, 1)
        ));

        assertArrayEquals(bools(0, expected[0], 0, expected[1]), outputStorage);
    }

    private static StorageAwareBoolReductionKernel kernel(Operation operation) {
        return switch (operation.opType()) {
            case REDUCE_ALL -> new CpuReduceAllKernel();
            case REDUCE_ANY -> new CpuReduceAnyKernel();
            default -> throw new IllegalArgumentException("Unsupported test operation " + operation.opType());
        };
    }

    private static CpuKernelCall call(
            Operation operation,
            Tensor inputTensor,
            Tensor outputTensor,
            CpuStorageView input,
            CpuStorageView output
    ) {
        CpuNodeExecutionPlan plan = plan(output.dtype());
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(1),
                null
        );
        CpuKernelContext context = new CpuKernelContext(
                2,
                List.of(1),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                operation
        );
        return new CpuKernelCall(operation, List.of(inputTensor), outputTensor, List.of(input), output, plan, context, null);
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
                null
        );
    }

    private static Tensor tensor(int[] shape, int[] strides, int storageOffset, String label) {
        return new Tensor(shape, strides, storageOffset, null, null, label, DataType.BOOL);
    }

    private static CpuStorageView segment(byte[] storage, int[] shape, int[] strides, int storageOffset) {
        return CpuStorageView.segment(
                DataType.BOOL,
                MemorySegment.ofArray(storage),
                shape,
                strides,
                storageOffset,
                TensorShape.checkedFlatSize(shape)
        );
    }

    private static byte[] bools(int... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] == 0 ? (byte) 0 : (byte) 1;
        }
        return out;
    }
}

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
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MinMaxReductionSegmentKernelTest {
    @Test
    void minMaxAllReduceReadStridedMemorySegmentsByDtype() {
        assertAllReduce(DataType.FLOAT64, new reduceMin(-1), -4.0d);
        assertAllReduce(DataType.FLOAT64, new reduceMax(-1), 7.0d);
        assertAllReduce(DataType.FLOAT32, new reduceMin(-1), -4.0d);
        assertAllReduce(DataType.FLOAT32, new reduceMax(-1), 7.0d);
        assertAllReduce(DataType.BFLOAT16, new reduceMin(-1), -4.0d);
        assertAllReduce(DataType.BFLOAT16, new reduceMax(-1), 7.0d);
        assertAllReduce(DataType.INT32, new reduceMin(-1), -4.0d);
        assertAllReduce(DataType.INT32, new reduceMax(-1), 7.0d);
        assertAllReduce(DataType.INT64, new reduceMin(-1), -4.0d);
        assertAllReduce(DataType.INT64, new reduceMax(-1), 7.0d);
        assertAllReduce(DataType.BOOL, new reduceMin(-1), 0.0d);
        assertAllReduce(DataType.BOOL, new reduceMax(-1), 1.0d);
    }

    @Test
    void minMaxAxisReduceReadAndWriteStridedMemorySegmentsByDtype() {
        assertAxisReduce(DataType.FLOAT64, new reduceMin(1), new double[]{2.0d, -4.0d});
        assertAxisReduce(DataType.FLOAT64, new reduceMax(1), new double[]{7.0d, 5.0d});
        assertAxisReduce(DataType.FLOAT32, new reduceMin(1), new double[]{2.0d, -4.0d});
        assertAxisReduce(DataType.FLOAT32, new reduceMax(1), new double[]{7.0d, 5.0d});
        assertAxisReduce(DataType.BFLOAT16, new reduceMin(1), new double[]{2.0d, -4.0d});
        assertAxisReduce(DataType.BFLOAT16, new reduceMax(1), new double[]{7.0d, 5.0d});
        assertAxisReduce(DataType.INT32, new reduceMin(1), new double[]{2.0d, -4.0d});
        assertAxisReduce(DataType.INT32, new reduceMax(1), new double[]{7.0d, 5.0d});
        assertAxisReduce(DataType.INT64, new reduceMin(1), new double[]{2.0d, -4.0d});
        assertAxisReduce(DataType.INT64, new reduceMax(1), new double[]{7.0d, 5.0d});
        assertAxisReduce(DataType.BOOL, new reduceMin(1), new double[]{0.0d, 1.0d});
        assertAxisReduce(DataType.BOOL, new reduceMax(1), new double[]{1.0d, 1.0d});
    }

    private static void assertAllReduce(DataType dtype, Operation operation, double expected) {
        Tensor inputTensor = tensor(dtype, new int[]{2, 3}, new int[]{1, 2}, 1, "input");
        Tensor outputTensor = tensor(dtype, new int[]{1}, new int[]{1}, 1, "output");
        Object inputStorage = inputStorage(dtype);
        Object outputStorage = storage(dtype, -1.0d, -1.0d);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(dtype, inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(dtype, outputStorage, new int[]{1}, new int[]{1}, 1)
        ));

        assertStorageValue(dtype, outputStorage, 0, sentinel(dtype));
        assertStorageValue(dtype, outputStorage, 1, expected);
    }

    private static void assertAxisReduce(DataType dtype, Operation operation, double[] expected) {
        Tensor inputTensor = tensor(dtype, new int[]{2, 3}, new int[]{1, 2}, 1, "input");
        Tensor outputTensor = tensor(dtype, new int[]{2}, new int[]{2}, 1, "output");
        Object inputStorage = inputStorage(dtype);
        Object outputStorage = storage(dtype, -1.0d, -1.0d, -1.0d, -1.0d);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(dtype, inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(dtype, outputStorage, new int[]{2}, new int[]{2}, 1)
        ));

        assertStorageValue(dtype, outputStorage, 0, sentinel(dtype));
        assertStorageValue(dtype, outputStorage, 1, expected[0]);
        assertStorageValue(dtype, outputStorage, 2, sentinel(dtype));
        assertStorageValue(dtype, outputStorage, 3, expected[1]);
    }

    private static StorageAwareMinMaxReductionKernel kernel(Operation operation) {
        return switch (operation.opType()) {
            case REDUCE_MIN -> new CpuReduceMinKernel();
            case REDUCE_MAX -> new CpuReduceMaxKernel();
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

    private static Tensor tensor(DataType dtype, int[] shape, int[] strides, int storageOffset, String label) {
        return new Tensor(shape, strides, storageOffset, null, null, label, dtype);
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        MemorySegment memorySegment = switch (dtype) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) storage);
            case FLOAT32 -> MemorySegment.ofArray((float[]) storage);
            case BFLOAT16 -> MemorySegment.ofArray((short[]) storage);
            case INT32 -> MemorySegment.ofArray((int[]) storage);
            case INT64 -> MemorySegment.ofArray((long[]) storage);
            case BOOL -> MemorySegment.ofArray((byte[]) storage);
        };
        return CpuStorageView.segment(
                dtype,
                memorySegment,
                shape,
                strides,
                storageOffset,
                TensorShape.checkedFlatSize(shape)
        );
    }

    private static Object inputStorage(DataType dtype) {
        return switch (dtype) {
            case BOOL -> storage(dtype, 1.0d, 1.0d, 1.0d, 0.0d, 1.0d, 1.0d, 1.0d);
            case FLOAT64, FLOAT32, BFLOAT16, INT32, INT64 -> storage(dtype, 99.0d, 2.0d, -4.0d, 7.0d, 5.0d, 3.0d, 1.0d);
        };
    }

    private static Object storage(DataType dtype, double... values) {
        return switch (dtype) {
            case FLOAT64 -> values.clone();
            case FLOAT32 -> {
                float[] out = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = (float) values[i];
                }
                yield out;
            }
            case BFLOAT16 -> {
                short[] out = new short[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = TensorDTypeOps.toBFloat16Bits((float) values[i]);
                }
                yield out;
            }
            case INT32 -> {
                int[] out = new int[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = (int) values[i];
                }
                yield out;
            }
            case INT64 -> {
                long[] out = new long[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = (long) values[i];
                }
                yield out;
            }
            case BOOL -> {
                byte[] out = new byte[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = values[i] == 0.0d ? (byte) 0 : (byte) 1;
                }
                yield out;
            }
        };
    }

    private static double sentinel(DataType dtype) {
        return dtype == DataType.BOOL ? 1.0d : -1.0d;
    }

    private static void assertStorageValue(DataType dtype, Object storage, int offset, double expected) {
        double actual = switch (dtype) {
            case FLOAT64 -> ((double[]) storage)[offset];
            case FLOAT32 -> ((float[]) storage)[offset];
            case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(((short[]) storage)[offset]);
            case INT32 -> ((int[]) storage)[offset];
            case INT64 -> ((long[]) storage)[offset];
            case BOOL -> ((byte[]) storage)[offset] == 0 ? 0.0d : 1.0d;
        };
        double tolerance = dtype == DataType.FLOAT64 ? 1.0e-12 : 1.0e-5;
        assertEquals(expected, actual, tolerance);
    }
}

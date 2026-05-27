package backend.cpu.kernels.reduction;

import backend.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.plan.reduction.ResolvedReductionHints;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.SumAccuracyMode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import operations.reduction.mean;
import operations.reduction.sum;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SumLikeReductionSegmentKernelTest {
    @Test
    void sumAndMeanAllReduceReadStridedMemorySegmentsByDtype() {
        assertAllReduce(DataType.FLOAT64, new sum(-1), 21.0d);
        assertAllReduce(DataType.FLOAT64, new mean(-1), 3.5d);
        assertAllReduce(DataType.FLOAT32, new sum(-1), 21.0d);
        assertAllReduce(DataType.FLOAT32, new mean(-1), 3.5d);
        assertAllReduce(DataType.BFLOAT16, new sum(-1), 21.0d);
        assertAllReduce(DataType.BFLOAT16, new mean(-1), 3.5d);
    }

    @Test
    void sumAndMeanAxisReduceReadAndWriteStridedMemorySegmentsByDtype() {
        assertAxisReduce(DataType.FLOAT64, new sum(1), new double[]{6.0d, 15.0d});
        assertAxisReduce(DataType.FLOAT64, new mean(1), new double[]{2.0d, 5.0d});
        assertAxisReduce(DataType.FLOAT32, new sum(1), new double[]{6.0d, 15.0d});
        assertAxisReduce(DataType.FLOAT32, new mean(1), new double[]{2.0d, 5.0d});
        assertAxisReduce(DataType.BFLOAT16, new sum(1), new double[]{6.0d, 15.0d});
        assertAxisReduce(DataType.BFLOAT16, new mean(1), new double[]{2.0d, 5.0d});
    }

    @Test
    void bf16FloatContinuationWritesAxisReductionToMemorySegmentOutput() {
        int[] inputShape = {2, 3};
        int[] inputStrides = TensorMetadata.computeStrides(inputShape);
        short[] ignoredInputStorage = bf16(9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f);
        short[] outputStorage = bf16(-1.0f, -1.0f, -1.0f, -1.0f);
        float[] continuation = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

        SumLoops.executeF32ToBF16(
                segment(DataType.BFLOAT16, ignoredInputStorage, inputShape, inputStrides, 0),
                continuation,
                segment(DataType.BFLOAT16, outputStorage, new int[]{2}, new int[]{2}, 1),
                1,
                context(new sum(1), DataType.BFLOAT16, 6)
        );

        assertEquals(-1.0f, fromBF16(outputStorage[0]), 0.0f);
        assertEquals(6.0f, fromBF16(outputStorage[1]), 0.0f);
        assertEquals(-1.0f, fromBF16(outputStorage[2]), 0.0f);
        assertEquals(15.0f, fromBF16(outputStorage[3]), 0.0f);
    }

    private static void assertAllReduce(DataType dtype, Operation operation, double expected) {
        Tensor inputTensor = tensor(dtype, new int[]{2, 3}, "input");
        Tensor outputTensor = tensor(dtype, new int[]{1}, "output");
        Object inputStorage = storage(dtype, 0.0d, 1.0d, 4.0d, 2.0d, 5.0d, 3.0d, 6.0d);
        Object outputStorage = storage(dtype, -1.0d, -1.0d);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(dtype, inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(dtype, outputStorage, new int[]{1}, new int[]{1}, 1)
        ));

        assertStorageValue(dtype, outputStorage, 0, -1.0d);
        assertStorageValue(dtype, outputStorage, 1, expected);
    }

    private static void assertAxisReduce(DataType dtype, Operation operation, double[] expected) {
        Tensor inputTensor = tensor(dtype, new int[]{2, 3}, "input");
        Tensor outputTensor = tensor(dtype, new int[]{2}, "output");
        Object inputStorage = storage(dtype, 0.0d, 1.0d, 4.0d, 2.0d, 5.0d, 3.0d, 6.0d);
        Object outputStorage = storage(dtype, -1.0d, -1.0d, -1.0d, -1.0d);

        kernel(operation).execute(call(
                operation,
                inputTensor,
                outputTensor,
                segment(dtype, inputStorage, new int[]{2, 3}, new int[]{1, 2}, 1),
                segment(dtype, outputStorage, new int[]{2}, new int[]{2}, 1)
        ));

        assertStorageValue(dtype, outputStorage, 0, -1.0d);
        assertStorageValue(dtype, outputStorage, 1, expected[0]);
        assertStorageValue(dtype, outputStorage, 2, -1.0d);
        assertStorageValue(dtype, outputStorage, 3, expected[1]);
    }

    private static StorageAwareSumLikeReductionKernel kernel(Operation operation) {
        return switch (operation.opType()) {
            case SUM -> new CpuSumKernel();
            case MEAN -> new CpuMeanKernel();
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
        CpuKernelContext context = context(operation, output.dtype(), input.logicalSize());
        return new CpuKernelCall(
                operation,
                List.of(inputTensor),
                outputTensor,
                List.of(input),
                output,
                context.nodePlan(),
                context,
                null
        );
    }

    private static CpuKernelContext context(Operation operation, DataType dtype, int logicalSize) {
        CpuNodeExecutionPlan plan = new CpuNodeExecutionPlan(
                new CpuLayoutPlan(StridedLayoutDecision.NONE, dtype, 0, null, null, List.of()),
                null,
                false,
                1,
                Integer.MAX_VALUE,
                null,
                new ResolvedReductionHints(
                        logicalSize,
                        CpuExecutionMode.SCALAR,
                        32,
                        1,
                        1,
                        SumAccuracyMode.FAST
                ),
                null,
                null,
                null
        );
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(1),
                null
        );
        return new CpuKernelContext(
                2,
                List.of(1),
                plan,
                new ExecutionContext(ExecutionMode.FORWARD, false, false),
                metadata,
                List.of(),
                operation
        );
    }

    private static Tensor tensor(DataType dtype, int[] shape, String label) {
        int length = TensorShape.checkedFlatSize(shape);
        return switch (dtype) {
            case FLOAT64 -> new Tensor(new double[length], shape, null, label, dtype);
            case FLOAT32 -> new Tensor(new float[length], shape, null, label, dtype);
            case BFLOAT16 -> new Tensor(new double[length], shape, null, label, dtype);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported test dtype " + dtype);
        };
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        MemorySegment memorySegment = switch (dtype) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) storage);
            case FLOAT32 -> MemorySegment.ofArray((float[]) storage);
            case BFLOAT16 -> MemorySegment.ofArray((short[]) storage);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported test dtype " + dtype);
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
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported test dtype " + dtype);
        };
    }

    private static void assertStorageValue(DataType dtype, Object storage, int offset, double expected) {
        double actual = switch (dtype) {
            case FLOAT64 -> ((double[]) storage)[offset];
            case FLOAT32 -> ((float[]) storage)[offset];
            case BFLOAT16 -> fromBF16(((short[]) storage)[offset]);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported test dtype " + dtype);
        };
        double tolerance = dtype == DataType.FLOAT64 ? 1.0e-12 : 1.0e-5;
        assertEquals(expected, actual, tolerance);
    }

    private static short[] bf16(float... values) {
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return out;
    }

    private static float fromBF16(short value) {
        return TensorDTypeOps.fromBFloat16Bits(value);
    }
}

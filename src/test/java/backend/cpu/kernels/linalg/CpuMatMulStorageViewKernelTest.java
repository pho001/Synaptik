package backend.cpu.kernels.linalg;

import backend.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.plan.linalg.matmul.MatMulExecutionRoute;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.CpuMatMulMicroKernel;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import operations.linalg.matmul;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CpuMatMulStorageViewKernelTest {
    @Test
    void denseZeroOffsetArrayInputsAndOutputUsePreparedExecutable() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT32, new int[]{2, 3}, "a");
        Tensor b = tensor(DataType.FLOAT32, new int[]{3, 2}, "b");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        CountingPreparedMatMulExecutable executable = new CountingPreparedMatMulExecutable();

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                array(DataType.FLOAT32, new float[6], new int[]{2, 3}, new int[]{3, 1}, 0),
                array(DataType.FLOAT32, new float[6], new int[]{3, 2}, new int[]{2, 1}, 0),
                array(DataType.FLOAT32, new float[4], new int[]{2, 2}, new int[]{2, 1}, 0),
                executable
        ));

        assertEquals(1, executable.calls);
    }

    @Test
    void denseZeroOffsetNativeSegmentInputsUsePreparedExecutableWhenProviderAcceptsNativeInputs() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT32, new int[]{2, 3}, "a");
        Tensor b = tensor(DataType.FLOAT32, new int[]{3, 2}, "b");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        NativeCountingPreparedMatMulExecutable executable = new NativeCountingPreparedMatMulExecutable();

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                segment(DataType.FLOAT32, new float[6], new int[]{2, 3}, new int[]{3, 1}, 0),
                segment(DataType.FLOAT32, new float[6], new int[]{3, 2}, new int[]{2, 1}, 0),
                array(DataType.FLOAT32, new float[4], new int[]{2, 2}, new int[]{2, 1}, 0),
                executable
        ));

        assertEquals(1, executable.calls);
    }

    @Test
    void stridedArrayInputsUseStorageViewPathInsteadOfPreparedExecutable() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT32, new int[]{2, 3}, "a");
        Tensor b = tensor(DataType.FLOAT32, new int[]{3, 2}, "b");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        CountingPreparedMatMulExecutable executable = new CountingPreparedMatMulExecutable();
        float[] aStorage = new float[]{-1, 1, 2, 3, -1, 4, 5, 6};
        float[] bStorage = new float[]{-1, -1, 7, 8, -1, 9, 10, -1, 11, 12};
        float[] outStorage = new float[]{-1, -1, -1, -1, -1, -1};

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                array(DataType.FLOAT32, aStorage, new int[]{2, 3}, new int[]{4, 1}, 1),
                array(DataType.FLOAT32, bStorage, new int[]{3, 2}, new int[]{3, 1}, 2),
                array(DataType.FLOAT32, outStorage, new int[]{2, 2}, new int[]{3, 1}, 1),
                executable
        ));

        assertEquals(0, executable.calls);
        assertEquals(58.0f, outStorage[1], 1e-6f);
        assertEquals(64.0f, outStorage[2], 1e-6f);
        assertEquals(139.0f, outStorage[4], 1e-6f);
        assertEquals(154.0f, outStorage[5], 1e-6f);
    }

    @Test
    void f32MemorySegmentMatMulHonorsOffsetsAndStrides() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT32, new int[]{2, 3}, "a");
        Tensor b = tensor(DataType.FLOAT32, new int[]{3, 2}, "b");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        CountingPreparedMatMulExecutable executable = new CountingPreparedMatMulExecutable();
        float[] aStorage = new float[]{-1, 1, 2, 3, -1, 4, 5, 6};
        float[] bStorage = new float[]{-1, -1, 7, 8, -1, 9, 10, -1, 11, 12};
        float[] outStorage = new float[]{-1, -1, -1, -1, -1, -1};

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                segment(DataType.FLOAT32, aStorage, new int[]{2, 3}, new int[]{4, 1}, 1),
                segment(DataType.FLOAT32, bStorage, new int[]{3, 2}, new int[]{3, 1}, 2),
                segment(DataType.FLOAT32, outStorage, new int[]{2, 2}, new int[]{3, 1}, 1),
                executable
        ));

        assertEquals(0, executable.calls);
        assertEquals(58.0f, outStorage[1], 1e-6f);
        assertEquals(64.0f, outStorage[2], 1e-6f);
        assertEquals(139.0f, outStorage[4], 1e-6f);
        assertEquals(154.0f, outStorage[5], 1e-6f);
    }

    @Test
    void f32MemorySegmentBatchedMatMulBroadcastsBatchDimension() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT32, new int[]{2, 2, 3}, "a");
        Tensor b = tensor(DataType.FLOAT32, new int[]{1, 3, 2}, "b");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2, 2}, "out");
        float[] aStorage = new float[]{
                -1,
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
        };
        float[] bStorage = new float[]{-1, 1, 2, 3, 4, 5, 6};
        float[] outStorage = new float[9];

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                segment(DataType.FLOAT32, aStorage, new int[]{2, 2, 3}, new int[]{6, 3, 1}, 1),
                segment(DataType.FLOAT32, bStorage, new int[]{1, 3, 2}, new int[]{6, 2, 1}, 1),
                segment(DataType.FLOAT32, outStorage, new int[]{2, 2, 2}, new int[]{4, 2, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(22.0f, outStorage[1], 1e-6f);
        assertEquals(28.0f, outStorage[2], 1e-6f);
        assertEquals(49.0f, outStorage[3], 1e-6f);
        assertEquals(64.0f, outStorage[4], 1e-6f);
        assertEquals(76.0f, outStorage[5], 1e-6f);
        assertEquals(100.0f, outStorage[6], 1e-6f);
        assertEquals(103.0f, outStorage[7], 1e-6f);
        assertEquals(136.0f, outStorage[8], 1e-6f);
    }

    @Test
    void f64MemorySegmentMatMulMatchesDenseParity() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.FLOAT64, new int[]{2, 2}, "a");
        Tensor b = tensor(DataType.FLOAT64, new int[]{2, 2}, "b");
        Tensor out = tensor(DataType.FLOAT64, new int[]{2, 2}, "out");
        double[] aStorage = new double[]{-1, 1, 2, 3, 4};
        double[] bStorage = new double[]{-1, 5, 6, 7, 8};
        double[] outStorage = new double[5];

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                segment(DataType.FLOAT64, aStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                segment(DataType.FLOAT64, bStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                segment(DataType.FLOAT64, outStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(19.0d, outStorage[1], 1e-12);
        assertEquals(22.0d, outStorage[2], 1e-12);
        assertEquals(43.0d, outStorage[3], 1e-12);
        assertEquals(50.0d, outStorage[4], 1e-12);
    }

    @Test
    void bf16MemorySegmentMatMulMatchesBasicDenseParity() {
        Operation operation = new matmul();
        Tensor a = tensor(DataType.BFLOAT16, new int[]{2, 2}, "a");
        Tensor b = tensor(DataType.BFLOAT16, new int[]{2, 2}, "b");
        Tensor out = tensor(DataType.BFLOAT16, new int[]{2, 2}, "out");
        short[] aStorage = bf16(-1, 1, 2, 3, 4);
        short[] bStorage = bf16(-1, 5, 6, 7, 8);
        short[] outStorage = bf16(0, 0, 0, 0, 0);

        new CpuMatMulKernel().execute(call(
                operation,
                a,
                b,
                out,
                segment(DataType.BFLOAT16, aStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                segment(DataType.BFLOAT16, bStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                segment(DataType.BFLOAT16, outStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(19.0f, fromBF16(outStorage[1]), 0.0f);
        assertEquals(22.0f, fromBF16(outStorage[2]), 0.0f);
        assertEquals(43.0f, fromBF16(outStorage[3]), 0.0f);
        assertEquals(50.0f, fromBF16(outStorage[4]), 0.0f);
    }

    private static CpuKernelCall call(
            Operation operation,
            Tensor a,
            Tensor b,
            Tensor out,
            CpuStorageView aView,
            CpuStorageView bView,
            CpuStorageView outView,
            PreparedMatMulExecutable executable
    ) {
        CpuNodeExecutionPlan plan = plan(outView.dtype(), executable);
        CpuKernelContext context = context(operation, plan);
        return new CpuKernelCall(
                operation,
                List.of(a, b),
                out,
                List.of(aView, bView),
                outView,
                plan,
                context,
                null
        );
    }

    private static CpuNodeExecutionPlan plan(DataType dtype, PreparedMatMulExecutable executable) {
        CpuComputeDType computeDType = switch (dtype) {
            case FLOAT64 -> CpuComputeDType.F64;
            case FLOAT32 -> CpuComputeDType.F32;
            case BFLOAT16 -> CpuComputeDType.BF16_NATIVE;
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported dtype " + dtype);
        };
        return new CpuNodeExecutionPlan(
                new CpuLayoutPlan(StridedLayoutDecision.NONE, dtype, 0, null, null, List.of()),
                new ResolvedCpuComputeContract(
                        dtype,
                        computeDType,
                        CpuExecutionBackend.CPU_MATMUL_JAVA,
                        dtype == DataType.FLOAT64 ? CpuAccumulateDType.F64 : CpuAccumulateDType.F32
                ),
                false,
                1,
                0,
                null,
                null,
                new ResolvedMatMulHints(
                        false,
                        false,
                        MatMulExecutionRoute.JAVA_DIRECT,
                        false,
                        16,
                        16,
                        16,
                        1,
                        0,
                        CpuMatMulMicroKernel.AUTO
                ),
                executable,
                null
        );
    }

    private static CpuKernelContext context(Operation operation, CpuNodeExecutionPlan plan) {
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(0, 1),
                null
        );
        return new CpuKernelContext(
                2,
                List.of(0, 1),
                plan,
                new ExecutionContext(ExecutionMode.FORWARD, false, false),
                metadata,
                List.of(),
                operation
        );
    }

    private static Tensor tensor(DataType dtype, int[] shape, String label) {
        int size = TensorShape.checkedFlatSize(shape);
        return switch (dtype) {
            case FLOAT64 -> new Tensor(new double[size], shape, null, label, dtype);
            case FLOAT32 -> new Tensor(new float[size], shape, null, label, dtype);
            case BFLOAT16 -> new Tensor(new double[size], shape, null, label, dtype);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported dtype " + dtype);
        };
    }

    private static CpuStorageView array(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        return CpuStorageView.array(dtype, storage, shape, strides, storageOffset, TensorShape.checkedFlatSize(shape));
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        MemorySegment segment = switch (dtype) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) storage);
            case FLOAT32 -> MemorySegment.ofArray((float[]) storage);
            case BFLOAT16 -> MemorySegment.ofArray((short[]) storage);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException("Unsupported dtype " + dtype);
        };
        return CpuStorageView.segment(dtype, segment, shape, strides, storageOffset, TensorShape.checkedFlatSize(shape));
    }

    private static short[] bf16(float... values) {
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return out;
    }

    private static float fromBF16(short bits) {
        return TensorDTypeOps.fromBFloat16Bits(bits);
    }

    private static final class CountingPreparedMatMulExecutable implements PreparedMatMulExecutable {
        private int calls;

        @Override
        public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
            calls++;
        }
    }

    private static final class NativeCountingPreparedMatMulExecutable implements PreparedMatMulExecutable {
        private int calls;

        @Override
        public void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context) {
            calls++;
        }

        @Override
        public boolean acceptsNativeInputs() {
            return true;
        }
    }
}

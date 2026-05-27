package backend.cpu.kernels.linalg;

import backend.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageView;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.linalg.linear;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CpuLinearStorageViewKernelTest {
    @Test
    void denseZeroOffsetArrayInputsAndOutputUsePreparedExecutor() {
        Tensor input = tensor(DataType.FLOAT32, new int[]{2, 3}, "input");
        Tensor weight = tensor(DataType.FLOAT32, new int[]{3, 2}, "weight");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        CountingPreparedMatMulExecutable executable = new CountingPreparedMatMulExecutable();

        new CpuLinearKernel().execute(call(
                new linear(false),
                List.of(input, weight),
                out,
                List.of(
                        array(DataType.FLOAT32, new float[6], new int[]{2, 3}, new int[]{3, 1}, 0),
                        array(DataType.FLOAT32, new float[6], new int[]{3, 2}, new int[]{2, 1}, 0)
                ),
                array(DataType.FLOAT32, new float[4], new int[]{2, 2}, new int[]{2, 1}, 0),
                executable
        ));

        assertEquals(1, executable.calls);
    }

    @Test
    void f32MemorySegmentLinearWithoutBiasHonorsOffsetsAndStrides() {
        Tensor input = tensor(DataType.FLOAT32, new int[]{2, 3}, "input");
        Tensor weight = tensor(DataType.FLOAT32, new int[]{3, 2}, "weight");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        float[] inputStorage = new float[]{-1, 1, 2, 3, -1, 4, 5, 6};
        float[] weightStorage = new float[]{-1, -1, 7, 8, -1, 9, 10, -1, 11, 12};
        float[] outStorage = new float[]{-1, -1, -1, -1, -1, -1};

        CpuKernelResult result = new CpuLinearKernel().execute(call(
                new linear(false),
                List.of(input, weight),
                out,
                List.of(
                        segment(DataType.FLOAT32, inputStorage, new int[]{2, 3}, new int[]{4, 1}, 1),
                        segment(DataType.FLOAT32, weightStorage, new int[]{3, 2}, new int[]{3, 1}, 2)
                ),
                segment(DataType.FLOAT32, outStorage, new int[]{2, 2}, new int[]{3, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals("CPU_NATIVE", result.route());
        assertEquals(58.0f, outStorage[1], 1e-6f);
        assertEquals(64.0f, outStorage[2], 1e-6f);
        assertEquals(139.0f, outStorage[4], 1e-6f);
        assertEquals(154.0f, outStorage[5], 1e-6f);
    }

    @Test
    void f32MemorySegmentLinearWithBiasHonorsNonZeroOffsetsAndRowBiasShape() {
        Tensor input = tensor(DataType.FLOAT32, new int[]{2, 3}, "input");
        Tensor weight = tensor(DataType.FLOAT32, new int[]{3, 2}, "weight");
        Tensor bias = tensor(DataType.FLOAT32, new int[]{1, 2}, "bias");
        Tensor out = tensor(DataType.FLOAT32, new int[]{2, 2}, "out");
        float[] inputStorage = new float[]{-1, 1, 2, 3, -1, 4, 5, 6};
        float[] weightStorage = new float[]{-1, -1, 7, 8, -1, 9, 10, -1, 11, 12};
        float[] biasStorage = new float[]{-1, 0.5f, -1.5f};
        float[] outStorage = new float[]{-1, -1, -1, -1, -1, -1};

        new CpuLinearKernel().execute(call(
                new linear(true),
                List.of(input, weight, bias),
                out,
                List.of(
                        segment(DataType.FLOAT32, inputStorage, new int[]{2, 3}, new int[]{4, 1}, 1),
                        segment(DataType.FLOAT32, weightStorage, new int[]{3, 2}, new int[]{3, 1}, 2),
                        segment(DataType.FLOAT32, biasStorage, new int[]{1, 2}, new int[]{3, 1}, 1)
                ),
                segment(DataType.FLOAT32, outStorage, new int[]{2, 2}, new int[]{3, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(58.5f, outStorage[1], 1e-6f);
        assertEquals(62.5f, outStorage[2], 1e-6f);
        assertEquals(139.5f, outStorage[4], 1e-6f);
        assertEquals(152.5f, outStorage[5], 1e-6f);
    }

    @Test
    void f64MemorySegmentLinearWithVectorBiasMatchesDenseParity() {
        Tensor input = tensor(DataType.FLOAT64, new int[]{2, 2}, "input");
        Tensor weight = tensor(DataType.FLOAT64, new int[]{2, 2}, "weight");
        Tensor bias = tensor(DataType.FLOAT64, new int[]{2}, "bias");
        Tensor out = tensor(DataType.FLOAT64, new int[]{2, 2}, "out");
        double[] inputStorage = new double[]{-1, 1, 2, 3, 4};
        double[] weightStorage = new double[]{-1, 5, 6, 7, 8};
        double[] biasStorage = new double[]{-1, 1, 2};
        double[] outStorage = new double[5];

        new CpuLinearKernel().execute(call(
                new linear(true),
                List.of(input, weight, bias),
                out,
                List.of(
                        segment(DataType.FLOAT64, inputStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                        segment(DataType.FLOAT64, weightStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                        segment(DataType.FLOAT64, biasStorage, new int[]{2}, new int[]{1}, 1)
                ),
                segment(DataType.FLOAT64, outStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(20.0d, outStorage[1], 1e-12);
        assertEquals(24.0d, outStorage[2], 1e-12);
        assertEquals(44.0d, outStorage[3], 1e-12);
        assertEquals(52.0d, outStorage[4], 1e-12);
    }

    @Test
    void bf16MemorySegmentLinearWithBiasMatchesBasicDenseParity() {
        Tensor input = tensor(DataType.BFLOAT16, new int[]{2, 2}, "input");
        Tensor weight = tensor(DataType.BFLOAT16, new int[]{2, 2}, "weight");
        Tensor bias = tensor(DataType.BFLOAT16, new int[]{2}, "bias");
        Tensor out = tensor(DataType.BFLOAT16, new int[]{2, 2}, "out");
        short[] inputStorage = bf16(-1, 1, 2, 3, 4);
        short[] weightStorage = bf16(-1, 5, 6, 7, 8);
        short[] biasStorage = bf16(-1, 1, 2);
        short[] outStorage = bf16(0, 0, 0, 0, 0);

        new CpuLinearKernel().execute(call(
                new linear(true),
                List.of(input, weight, bias),
                out,
                List.of(
                        segment(DataType.BFLOAT16, inputStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                        segment(DataType.BFLOAT16, weightStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                        segment(DataType.BFLOAT16, biasStorage, new int[]{2}, new int[]{1}, 1)
                ),
                segment(DataType.BFLOAT16, outStorage, new int[]{2, 2}, new int[]{2, 1}, 1),
                new CountingPreparedMatMulExecutable()
        ));

        assertEquals(20.0f, fromBF16(outStorage[1]), 0.0f);
        assertEquals(24.0f, fromBF16(outStorage[2]), 0.0f);
        assertEquals(44.0f, fromBF16(outStorage[3]), 0.0f);
        assertEquals(52.0f, fromBF16(outStorage[4]), 0.0f);
    }

    private static CpuKernelCall call(
            linear operation,
            List<Tensor> inputs,
            Tensor out,
            List<CpuStorageView> inputViews,
            CpuStorageView outView,
            PreparedMatMulExecutable executable
    ) {
        CpuNodeExecutionPlan plan = plan(outView.dtype(), executable);
        CpuKernelContext context = context(operation, plan, inputs.size());
        return new CpuKernelCall(
                operation,
                inputs,
                out,
                inputViews,
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
                null,
                executable,
                null
        );
    }

    private static CpuKernelContext context(linear operation, CpuNodeExecutionPlan plan, int inputCount) {
        List<Integer> inputNodeIds = new ArrayList<>();
        for (int i = 0; i < inputCount; i++) {
            inputNodeIds.add(i);
        }
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                inputNodeIds,
                null
        );
        return new CpuKernelContext(
                inputCount,
                inputNodeIds,
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
}

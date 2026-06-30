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
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;
import operations.loss.crossEntropyLoss;
import operations.loss.nllLoss;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class LossReductionExecutorSegmentTest {
    @Test
    void nllLossF64ReadsDenseTargetSegmentsAndWritesScalarSegment() {
        int[] shape = {2, 3};
        double[] logProbs = {
                -2.0d, -1.0d, -0.1d,
                -0.7d, -0.2d, -1.4d
        };
        double[] targets = {
                0.0d, 0.0d, 1.0d,
                0.25d, 0.75d, 0.0d
        };
        double[] output = {0.0d};

        Tensor logProbTensor = tensor(DataType.FLOAT64, shape, "logProbs");
        Tensor targetTensor = tensor(DataType.FLOAT64, shape, "targets");
        Tensor outputTensor = tensor(DataType.FLOAT64, new int[]{1}, "loss");

        new CpuNllLossKernel().execute(call(
                new nllLoss(1),
                List.of(logProbTensor, targetTensor),
                outputTensor,
                List.of(segment(DataType.FLOAT64, MemorySegment.ofArray(logProbs), shape),
                        segment(DataType.FLOAT64, MemorySegment.ofArray(targets), shape)),
                segment(DataType.FLOAT64, MemorySegment.ofArray(output), new int[]{1})
        ));

        assertEquals(0.2125d, output[0], 1.0e-12);
        assertArrayEquals(new double[]{0.0d}, outputTensor.toDoubleArrayCopy(), 0.0d);
    }

    @Test
    void crossEntropyF32ReadsDenseTargetSegmentsAndWritesScalarSegment() {
        int[] shape = {2, 3};
        float[] logits = {
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        };
        float[] targets = {
                0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.0f
        };
        float[] output = {0.0f};

        Tensor logitsTensor = tensor(DataType.FLOAT32, shape, "logits");
        Tensor targetTensor = tensor(DataType.FLOAT32, shape, "targets");
        Tensor outputTensor = tensor(DataType.FLOAT32, new int[]{1}, "loss");

        new CpuCrossEntropyLossKernel().execute(call(
                new crossEntropyLoss(1),
                List.of(logitsTensor, targetTensor),
                outputTensor,
                List.of(segment(DataType.FLOAT32, MemorySegment.ofArray(logits), shape),
                        segment(DataType.FLOAT32, MemorySegment.ofArray(targets), shape)),
                segment(DataType.FLOAT32, MemorySegment.ofArray(output), new int[]{1})
        ));

        double first = 3.0d + Math.log(1.0d + Math.exp(-1.0d) + Math.exp(-2.0d)) - 3.0d;
        double second = Math.log(3.0d);
        assertEquals((float) ((first + second) * 0.5d), output[0], 1.0e-6f);
        assertArrayEquals(new double[]{0.0d}, outputTensor.toDoubleArrayCopy(), 0.0d);
    }

    @Test
    void nllLossBF16ReadsDenseTargetSegmentsAndWritesScalarSegment() {
        int[] shape = {2, 3};
        short[] logProbs = bf16(
                -1.0f, -2.0f, -3.0f,
                -0.5f, -1.5f, -2.5f
        );
        short[] targets = bf16(
                0.0f, 1.0f, 0.0f,
                0.5f, 0.5f, 0.0f
        );
        short[] output = {0};

        Tensor logProbTensor = tensor(DataType.BFLOAT16, shape, "logProbs");
        Tensor targetTensor = tensor(DataType.BFLOAT16, shape, "targets");
        Tensor outputTensor = tensor(DataType.BFLOAT16, new int[]{1}, "loss");

        new CpuNllLossKernel().execute(call(
                new nllLoss(1),
                List.of(logProbTensor, targetTensor),
                outputTensor,
                List.of(segment(DataType.BFLOAT16, MemorySegment.ofArray(logProbs), shape),
                        segment(DataType.BFLOAT16, MemorySegment.ofArray(targets), shape)),
                segment(DataType.BFLOAT16, MemorySegment.ofArray(output), new int[]{1})
        ));

        assertEquals(1.5f, TensorDTypeOps.fromBFloat16Bits(output[0]), 0.0f);
        assertArrayEquals(new double[]{0.0d}, outputTensor.toDoubleArrayCopy(), 0.0d);
    }

    @Test
    void crossEntropyBF16ContinuationReadsDenseTargetSegmentAndWritesScalarSegment() {
        int[] shape = {2, 3};
        float[] logitsContinuation = {
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        };
        short[] ignoredLogitStorage = bf16(
                9.0f, 9.0f, 9.0f,
                9.0f, 9.0f, 9.0f
        );
        short[] targets = bf16(
                0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.0f
        );
        short[] output = {0};
        CpuKernelContext context = context(new crossEntropyLoss(1), DataType.BFLOAT16);

        LossReductionExecutor.executeF32ToBF16(
                LossReduction.CROSS_ENTROPY,
                segment(DataType.BFLOAT16, MemorySegment.ofArray(ignoredLogitStorage), shape),
                logitsContinuation,
                segment(DataType.BFLOAT16, MemorySegment.ofArray(targets), shape),
                segment(DataType.BFLOAT16, MemorySegment.ofArray(output), new int[]{1}),
                1,
                context
        );

        double first = 3.0d + Math.log(1.0d + Math.exp(-1.0d) + Math.exp(-2.0d)) - 3.0d;
        double second = Math.log(3.0d);
        assertEquals((float) ((first + second) * 0.5d), TensorDTypeOps.fromBFloat16Bits(output[0]), 5.0e-3f);
    }

    private static CpuKernelCall call(
            Operation operation,
            List<Tensor> tensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        CpuKernelContext context = context(operation, output.dtype());
        return new CpuKernelCall(
                operation,
                tensors,
                outputTensor,
                inputs,
                output,
                context.nodePlan(),
                context,
                null
        );
    }

    private static CpuKernelContext context(Operation operation, DataType dtype) {
        CpuNodeExecutionPlan plan = new CpuNodeExecutionPlan(
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
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(1, 2),
                null
        );
        return new CpuKernelContext(
                3,
                List.of(1, 2),
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

    private static CpuStorageView segment(DataType dtype, MemorySegment memorySegment, int[] shape) {
        return CpuStorageView.segment(
                dtype,
                memorySegment,
                shape,
                TensorMetadata.computeStrides(shape),
                0,
                TensorShape.checkedFlatSize(shape)
        );
    }

    private static short[] bf16(float... values) {
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return out;
    }

}

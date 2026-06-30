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
import operations.loss.crossEntropyLossIndices;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;
import tensor.layout.TensorShape;
import tensor.loss.LossReduction;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CrossEntropyLossIndicesSegmentKernelTest {
    @Test
    void noneReductionReadsStridedF64LogitsAndInt64TargetsFromSegments() {
        crossEntropyLossIndices op = new crossEntropyLossIndices(1, LossReduction.NONE, -1);
        Tensor logitsTensor = tensor(DataType.FLOAT64, new int[]{2, 3}, new int[]{1, 2}, 1, "logits");
        Tensor targetTensor = tensor(DataType.INT64, new int[]{2}, new int[]{2}, 1, "target");
        Tensor outputTensor = tensor(DataType.FLOAT64, new int[]{2}, new int[]{2}, 1, "loss");
        double[] logits = {-9.0d, 1.0d, 0.0d, 2.0d, 0.0d, 3.0d, 0.0d};
        long[] targets = {-9L, 2L, -9L, -1L};
        double[] output = {-7.0d, -7.0d, -7.0d, -7.0d};

        new CpuCrossEntropyLossIndicesKernel().execute(call(
                op,
                List.of(logitsTensor, targetTensor),
                outputTensor,
                List.of(
                        segment(DataType.FLOAT64, logits, new int[]{2, 3}, new int[]{1, 2}, 1),
                        segment(DataType.INT64, targets, new int[]{2}, new int[]{2}, 1)
                ),
                segment(DataType.FLOAT64, output, new int[]{2}, new int[]{2}, 1)
        ));

        assertArrayEquals(new double[]{
                -7.0d,
                crossEntropy(new double[]{1.0d, 2.0d, 3.0d}, 2),
                -7.0d,
                0.0d
        }, output, 1.0e-12);
    }

    @Test
    void reducedF32SegmentPathAcceptsAllNumericTargetIndexDTypesWithIgnoreIndex() {
        for (DataType targetType : List.of(DataType.INT32, DataType.INT64, DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            crossEntropyLossIndices op = new crossEntropyLossIndices(1, LossReduction.MEAN, -1);
            Tensor logitsTensor = tensor(DataType.FLOAT32, new int[]{2, 3}, "logits");
            Tensor targetTensor = tensor(targetType, new int[]{2}, "target");
            Tensor outputTensor = tensor(DataType.FLOAT32, new int[]{1}, "loss");
            float[] logits = {
                    1.0f, 2.0f, 3.0f,
                    0.0f, 0.0f, 0.0f
            };
            Object targets = targetStorage(targetType, 2, -1);
            float[] output = {0.0f};

            new CpuCrossEntropyLossIndicesKernel().execute(call(
                    op,
                    List.of(logitsTensor, targetTensor),
                    outputTensor,
                    List.of(
                            segment(DataType.FLOAT32, logits, new int[]{2, 3}),
                            segment(targetType, targets, new int[]{2})
                    ),
                    segment(DataType.FLOAT32, output, new int[]{1})
            ));

            double expected = crossEntropy(new double[]{1.0d, 2.0d, 3.0d}, 2);
            assertEquals((float) expected, output[0], 1.0e-6f, "target dtype " + targetType);
        }
    }

    @Test
    void reducedBF16SegmentPathReadsBF16LogitsAndWritesBF16Output() {
        crossEntropyLossIndices op = new crossEntropyLossIndices(1, LossReduction.SUM, null);
        Tensor logitsTensor = tensor(DataType.BFLOAT16, new int[]{2, 3}, "logits");
        Tensor targetTensor = tensor(DataType.INT32, new int[]{2}, "target");
        Tensor outputTensor = tensor(DataType.BFLOAT16, new int[]{1}, "loss");
        short[] logits = bf16(
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        );
        int[] targets = {2, 0};
        short[] output = {0};

        new CpuCrossEntropyLossIndicesKernel().execute(call(
                op,
                List.of(logitsTensor, targetTensor),
                outputTensor,
                List.of(
                        segment(DataType.BFLOAT16, logits, new int[]{2, 3}),
                        segment(DataType.INT32, targets, new int[]{2})
                ),
                segment(DataType.BFLOAT16, output, new int[]{1})
        ));

        double expected = crossEntropy(new double[]{1.0d, 2.0d, 3.0d}, 2)
                + crossEntropy(new double[]{0.0d, 0.0d, 0.0d}, 0);
        assertEquals((float) expected, TensorDTypeOps.fromBFloat16Bits(output[0]), 5.0e-3f);
    }

    @Test
    void bf16FloatContinuationUsesContinuationLogitsWithSegmentTargetsAndOutput() {
        crossEntropyLossIndices op = new crossEntropyLossIndices(1, LossReduction.MEAN, null);
        short[] ignoredLogits = bf16(
                9.0f, 9.0f, 9.0f,
                9.0f, 9.0f, 9.0f
        );
        float[] continuation = {
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        };
        short[] targets = bf16(2.0f, 0.0f);
        short[] output = {0};

        CrossEntropyLossIndicesExecutor.executeF32ToBF16(
                op,
                segment(DataType.BFLOAT16, ignoredLogits, new int[]{2, 3}),
                continuation,
                segment(DataType.BFLOAT16, targets, new int[]{2}),
                segment(DataType.BFLOAT16, output, new int[]{1}),
                context(op, DataType.BFLOAT16)
        );

        double expected = (crossEntropy(new double[]{1.0d, 2.0d, 3.0d}, 2)
                + crossEntropy(new double[]{0.0d, 0.0d, 0.0d}, 0)) * 0.5d;
        assertEquals((float) expected, TensorDTypeOps.fromBFloat16Bits(output[0]), 5.0e-3f);
    }

    @Test
    void boolTargetIndexSegmentIsRejected() {
        crossEntropyLossIndices op = new crossEntropyLossIndices(1, LossReduction.MEAN, null);
        Tensor logitsTensor = tensor(DataType.FLOAT32, new int[]{2, 3}, "logits");
        Tensor targetTensor = tensor(DataType.BOOL, new int[]{2}, "target");
        Tensor outputTensor = tensor(DataType.FLOAT32, new int[]{1}, "loss");

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                new CpuCrossEntropyLossIndicesKernel().execute(call(
                        op,
                        List.of(logitsTensor, targetTensor),
                        outputTensor,
                        List.of(
                                segment(DataType.FLOAT32, new float[]{1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f}, new int[]{2, 3}),
                                segment(DataType.BOOL, new byte[]{1, 0}, new int[]{2})
                        ),
                        segment(DataType.FLOAT32, new float[]{0.0f}, new int[]{1})
                )));
        assertEquals("Target indices must be numeric integral values", error.getMessage());
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
        return tensor(dtype, shape, TensorMetadata.computeStrides(shape), 0, label);
    }

    private static Tensor tensor(DataType dtype, int[] shape, int[] strides, int storageOffset, String label) {
        return new Tensor(shape, strides, storageOffset, null, null, label, dtype);
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape) {
        return segment(dtype, storage, shape, TensorMetadata.computeStrides(shape), 0);
    }

    private static CpuStorageView segment(DataType dtype, Object storage, int[] shape, int[] strides, int storageOffset) {
        MemorySegment memorySegment;
        if (storage instanceof double[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else if (storage instanceof float[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else if (storage instanceof short[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else if (storage instanceof int[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else if (storage instanceof long[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else if (storage instanceof byte[] data) {
            memorySegment = MemorySegment.ofArray(data);
        } else {
            throw new IllegalArgumentException("Unsupported segment storage: " + storage.getClass());
        }
        return CpuStorageView.segment(
                dtype,
                memorySegment,
                shape,
                strides,
                storageOffset,
                TensorShape.checkedFlatSize(shape)
        );
    }

    private static Object targetStorage(DataType dtype, int... values) {
        return switch (dtype) {
            case INT32 -> values;
            case INT64 -> {
                long[] out = new long[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = values[i];
                }
                yield out;
            }
            case FLOAT64 -> {
                double[] out = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = values[i];
                }
                yield out;
            }
            case FLOAT32 -> {
                float[] out = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = values[i];
                }
                yield out;
            }
            case BFLOAT16 -> {
                short[] out = new short[values.length];
                for (int i = 0; i < values.length; i++) {
                    out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
                }
                yield out;
            }
            case BOOL -> throw new IllegalArgumentException("BOOL targets are rejected by the kernel");
        };
    }

    private static short[] bf16(float... values) {
        short[] out = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return out;
    }

    private static double crossEntropy(double[] logits, int targetIndex) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }
        double sumExp = 0.0d;
        for (double logit : logits) {
            sumExp += Math.exp(logit - max);
        }
        return max + Math.log(sumExp) - logits[targetIndex];
    }
}

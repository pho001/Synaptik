package backend.cpu1.kernels.matmul;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;

/**
 * Direct Java scalar matmul loops for dense cpu1 array storage.
 */
public final class Cpu1JavaScalarMatmulLoops {
    private Cpu1JavaScalarMatmulLoops() {
    }

    public static void matmulF32DenseScalar(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView output = outputView(unit, context);
        runF32(left.float32Array(), right.float32Array(), output.float32Array(), left.storageOffset(), right.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void matmulF64DenseScalar(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView output = outputView(unit, context);
        runF64(left.float64Array(), right.float64Array(), output.float64Array(), left.storageOffset(), right.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    public static void matmulBf16DenseScalar(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Cpu1TensorView left = inputView(unit.leftNodeId(), context);
        Cpu1TensorView right = inputView(unit.rightNodeId(), context);
        Cpu1TensorView output = outputView(unit, context);
        runBf16(left.bfloat16Array(), right.bfloat16Array(), output.bfloat16Array(), left.storageOffset(), right.storageOffset(), output.storageOffset(), unit);
        markOutputWritten(unit, output, context);
    }

    private static void runF32(
            float[] left,
            float[] right,
            float[] output,
            int leftStorageOffset,
            int rightStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int leftBatchBase = leftStorageOffset + unit.leftBatchOffset(batch);
            int rightBatchBase = rightStorageOffset + unit.rightBatchOffset(batch);
            int outputBatchBase = outputStorageOffset + unit.outputBatchOffset(batch);
            for (int row = 0; row < unit.m(); row++) {
                int leftRowBase = leftBatchBase + row * unit.leftRowStride();
                int outputRowBase = outputBatchBase + row * unit.outputRowStride();
                for (int col = 0; col < unit.n(); col++) {
                    float sum = 0.0f;
                    int rightColBase = rightBatchBase + col * unit.rightColStride();
                    for (int index = 0; index < unit.k(); index++) {
                        sum += left[leftRowBase + index * unit.leftColStride()]
                                * right[rightColBase + index * unit.rightRowStride()];
                    }
                    output[outputRowBase + col * unit.outputColStride()] = sum;
                }
            }
        }
    }

    private static void runF64(
            double[] left,
            double[] right,
            double[] output,
            int leftStorageOffset,
            int rightStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int leftBatchBase = leftStorageOffset + unit.leftBatchOffset(batch);
            int rightBatchBase = rightStorageOffset + unit.rightBatchOffset(batch);
            int outputBatchBase = outputStorageOffset + unit.outputBatchOffset(batch);
            for (int row = 0; row < unit.m(); row++) {
                int leftRowBase = leftBatchBase + row * unit.leftRowStride();
                int outputRowBase = outputBatchBase + row * unit.outputRowStride();
                for (int col = 0; col < unit.n(); col++) {
                    double sum = 0.0d;
                    int rightColBase = rightBatchBase + col * unit.rightColStride();
                    for (int index = 0; index < unit.k(); index++) {
                        sum += left[leftRowBase + index * unit.leftColStride()]
                                * right[rightColBase + index * unit.rightRowStride()];
                    }
                    output[outputRowBase + col * unit.outputColStride()] = sum;
                }
            }
        }
    }

    private static void runBf16(
            short[] left,
            short[] right,
            short[] output,
            int leftStorageOffset,
            int rightStorageOffset,
            int outputStorageOffset,
            Cpu1PreparedMatmulUnit unit
    ) {
        for (int batch = 0; batch < unit.batchCount(); batch++) {
            int leftBatchBase = leftStorageOffset + unit.leftBatchOffset(batch);
            int rightBatchBase = rightStorageOffset + unit.rightBatchOffset(batch);
            int outputBatchBase = outputStorageOffset + unit.outputBatchOffset(batch);
            for (int row = 0; row < unit.m(); row++) {
                int leftRowBase = leftBatchBase + row * unit.leftRowStride();
                int outputRowBase = outputBatchBase + row * unit.outputRowStride();
                for (int col = 0; col < unit.n(); col++) {
                    float sum = 0.0f;
                    int rightColBase = rightBatchBase + col * unit.rightColStride();
                    for (int index = 0; index < unit.k(); index++) {
                        float leftValue = TensorDTypeOps.fromBFloat16Bits(left[leftRowBase + index * unit.leftColStride()]);
                        float rightValue = TensorDTypeOps.fromBFloat16Bits(right[rightColBase + index * unit.rightRowStride()]);
                        sum += leftValue * rightValue;
                    }
                    output[outputRowBase + col * unit.outputColStride()] = TensorDTypeOps.toBFloat16Bits(sum);
                }
            }
        }
    }

    private static Cpu1TensorView inputView(int nodeId, ExecutionContext context) {
        context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        Tensor input = context.runtimeTensorForNodeId(nodeId);
        return Cpu1TensorView.fromTensor(input);
    }

    private static Cpu1TensorView outputView(Cpu1PreparedMatmulUnit unit, ExecutionContext context) {
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        return Cpu1TensorView.fromTensor(output);
    }

    private static void markOutputWritten(
            Cpu1PreparedMatmulUnit unit,
            Cpu1TensorView output,
            ExecutionContext context
    ) {
        output.markStorageModified();
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.route() + " matmul wrote CPU array");
    }
}

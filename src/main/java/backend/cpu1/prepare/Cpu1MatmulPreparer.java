package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the initial dense Java scalar matmul subset for cpu1.
 */
public final class Cpu1MatmulPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        if (operation.opType() != Operation.OpType.MATMUL) {
            throw new UnsupportedOperationException("cpu1 matmul preparer does not support " + operation.opType());
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 MATMUL requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL expects 2 inputs, got " + node.inputIds().size());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only JAVA_ARRAY storage.");
        }
        CompiledTensorDescriptor left = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor right = descriptorIndex.byNodeId(node.inputIds().get(1));
        requireDenseArrayContract(node, left, right);

        int[] leftShape = left.shape();
        int[] rightShape = right.shape();
        int[] outputShape = node.shape();
        int[] leftStrides = left.strides();
        int[] rightStrides = right.strides();
        int[] outputStrides = node.strides();
        validateShape(leftShape, rightShape, outputShape);

        int batchCount = batchCount(outputShape);
        Cpu1PreparedMatmulUnit unit = new Cpu1PreparedMatmulUnit(
                node.id(),
                node.inputIds().get(0),
                node.inputIds().get(1),
                node.dataType(),
                config.storageKind(),
                Cpu1MatmulRoute.JAVA_SCALAR,
                kernelId(node.dataType()),
                batchCount,
                outputShape[outputShape.length - 2],
                outputShape[outputShape.length - 1],
                leftShape[leftShape.length - 1],
                leftStrides[leftStrides.length - 2],
                leftStrides[leftStrides.length - 1],
                rightStrides[rightStrides.length - 2],
                rightStrides[rightStrides.length - 1],
                outputStrides[outputStrides.length - 2],
                outputStrides[outputStrides.length - 1],
                batchOffsets(leftShape, leftStrides, outputShape, true),
                batchOffsets(rightShape, rightStrides, outputShape, true),
                batchOffsets(outputShape, outputStrides, outputShape, false),
                Cpu1WorkspaceSpec.none()
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isMatmulOp(Operation.OpType opType) {
        return opType == Operation.OpType.MATMUL;
    }

    private static void requireDenseArrayContract(
            CompiledNode node,
            CompiledTensorDescriptor left,
            CompiledTensorDescriptor right
    ) {
        DataType dataType = node.dataType();
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            throw new UnsupportedOperationException("cpu1 MATMUL does not support output dtype " + dataType);
        }
        if (left.dataType() != dataType || right.dataType() != dataType) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL requires matching input/output dtype. left="
                    + left.dataType() + ", right=" + right.dataType() + ", output=" + dataType);
        }
        if (!left.denseContiguousWithoutOffset() || !right.denseContiguousWithoutOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous inputs without storage offset.");
        }
        if (!node.contiguous() || node.hasStorageOffset()) {
            throw new UnsupportedOperationException("cpu1 initial MATMUL supports only dense contiguous output without storage offset.");
        }
    }

    private static void validateShape(int[] leftShape, int[] rightShape, int[] outputShape) {
        if (leftShape.length < 2 || rightShape.length < 2 || outputShape.length < 2) {
            throw new UnsupportedOperationException("cpu1 MATMUL requires rank >= 2. left="
                    + Arrays.toString(leftShape) + ", right=" + Arrays.toString(rightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        int leftBatchRank = leftShape.length - 2;
        int rightBatchRank = rightShape.length - 2;
        int outputBatchRank = outputShape.length - 2;
        int expectedOutputRank = Math.max(leftBatchRank, rightBatchRank) + 2;
        if (outputShape.length != expectedOutputRank) {
            throw new UnsupportedOperationException("cpu1 MATMUL output rank mismatch. expected="
                    + expectedOutputRank + ", actual=" + outputShape.length);
        }
        int m = leftShape[leftShape.length - 2];
        int k = leftShape[leftShape.length - 1];
        int rightK = rightShape[rightShape.length - 2];
        int n = rightShape[rightShape.length - 1];
        if (rightK != k || outputShape[outputShape.length - 2] != m || outputShape[outputShape.length - 1] != n) {
            throw new UnsupportedOperationException("cpu1 MATMUL core dimensions mismatch. left="
                    + Arrays.toString(leftShape) + ", right=" + Arrays.toString(rightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        validateBroadcastBatch(leftShape, outputShape, outputBatchRank);
        validateBroadcastBatch(rightShape, outputShape, outputBatchRank);
    }

    private static void validateBroadcastBatch(int[] inputShape, int[] outputShape, int outputBatchRank) {
        int inputBatchRank = inputShape.length - 2;
        int shift = outputBatchRank - inputBatchRank;
        if (shift < 0) {
            throw new UnsupportedOperationException("cpu1 MATMUL input batch rank exceeds output batch rank.");
        }
        for (int dim = 0; dim < outputBatchRank; dim++) {
            int inputDim = dim < shift ? 1 : inputShape[dim - shift];
            int outputDim = outputShape[dim];
            if (inputDim != 1 && inputDim != outputDim) {
                throw new UnsupportedOperationException("cpu1 MATMUL batch dimensions are not broadcast-compatible. input="
                        + Arrays.toString(inputShape) + ", output=" + Arrays.toString(outputShape));
            }
        }
    }

    private static int[] batchOffsets(
            int[] shape,
            int[] strides,
            int[] outputShape,
            boolean allowBroadcast
    ) {
        int outputBatchRank = outputShape.length - 2;
        int inputBatchRank = shape.length - 2;
        int shift = outputBatchRank - inputBatchRank;
        int batchCount = batchCount(outputShape);
        int[] offsets = new int[batchCount];
        if (outputBatchRank == 0) {
            return offsets;
        }
        int[] outputBatchShape = Arrays.copyOf(outputShape, outputBatchRank);
        int[] outputBatchDenseStrides = denseStrides(outputBatchShape);
        for (int batch = 0; batch < batchCount; batch++) {
            int remaining = batch;
            int offset = 0;
            for (int dim = 0; dim < outputBatchRank; dim++) {
                int coordinate = remaining / outputBatchDenseStrides[dim];
                remaining %= outputBatchDenseStrides[dim];
                int inputDim = dim - shift;
                if (inputDim < 0) {
                    continue;
                }
                if (allowBroadcast && shape[inputDim] == 1) {
                    continue;
                }
                offset += coordinate * strides[inputDim];
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static int batchCount(int[] outputShape) {
        int count = 1;
        for (int dim = 0; dim < outputShape.length - 2; dim++) {
            count = Math.multiplyExact(count, outputShape[dim]);
        }
        return count;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride = Math.multiplyExact(stride, shape[dim]);
        }
        return strides;
    }

    private static Cpu1MatmulKernelId kernelId(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR;
            case FLOAT64 -> Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR;
            case BFLOAT16 -> Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 MATMUL does not support " + dataType);
        };
    }
}

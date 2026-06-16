package backend.cpu1.prepare;

import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.index.gather;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the initial dense array index subset for cpu1.
 */
public final class Cpu1IndexPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        Operation.OpType opType = operation.opType();
        if (!isIndexOp(opType)) {
            throw new UnsupportedOperationException("cpu1 index preparer does not support " + opType);
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 GATHER requires descriptors to resolve value/index dtype and layout.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 GATHER expects 2 inputs, got " + node.inputIds().size());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY) {
            throw new UnsupportedOperationException("cpu1 GATHER first slice supports only JAVA_ARRAY storage, got "
                    + config.storageKind());
        }
        if (!(operation instanceof gather gatherOp)) {
            throw new IllegalArgumentException("cpu1 GATHER operation must be operations.index.gather.");
        }

        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor indices = descriptorIndex.byNodeId(node.inputIds().get(1));
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan indexAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(indices);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);

        requireSupportedDTypes(node, input, indices);
        requireDenseContiguousNoOffset("input", inputAccessPlan);
        requireDenseContiguousNoOffset("indices", indexAccessPlan);
        requireDenseContiguousNoOffset("output", outputAccessPlan);

        int[] inputShape = input.shape();
        int dimension = normalizedDimension(gatherOp.getDimension(), inputShape.length);
        int[] expectedOutputShape = reducedShape(inputShape, dimension);
        if (!Arrays.equals(expectedOutputShape, indices.shape())) {
            throw new UnsupportedOperationException("cpu1 GATHER indices shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(indices.shape()));
        }
        if (!Arrays.equals(expectedOutputShape, node.shape())) {
            throw new UnsupportedOperationException("cpu1 GATHER output shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(node.shape()));
        }
        int axisSize = inputShape[dimension];
        int innerSize = product(inputShape, dimension + 1, inputShape.length);
        int outerSize = product(inputShape, 0, dimension);
        Cpu1PreparedIndexUnit unit = new Cpu1PreparedIndexUnit(
                node.id(),
                input.nodeId(),
                indices.nodeId(),
                opType,
                node.dataType(),
                indices.dataType(),
                config.storageKind(),
                kernelId(node.dataType(), indices.dataType()),
                dimension,
                axisSize,
                innerSize,
                outerSize,
                node.flatDataSize(),
                config.launchConfig(),
                launchPolicy(config.launchConfig()),
                inputAccessPlan,
                indexAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isIndexOp(Operation.OpType opType) {
        return opType == Operation.OpType.GATHER;
    }

    private static void requireSupportedDTypes(
            CompiledNode node,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor indices
    ) {
        if (input.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 GATHER requires matching input/output dtype. input="
                    + input.dataType() + ", output=" + node.dataType());
        }
        if (!isSupportedValueDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 GATHER does not support value dtype " + node.dataType());
        }
        if (indices.dataType() != DataType.INT32 && indices.dataType() != DataType.INT64) {
            throw new UnsupportedOperationException("cpu1 GATHER first slice supports only INT32/INT64 indices, got "
                    + indices.dataType());
        }
    }

    private static boolean isSupportedValueDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.INT32
                || dataType == DataType.INT64
                || dataType == DataType.BOOL;
    }

    private static void requireDenseContiguousNoOffset(String role, Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 GATHER first slice supports only dense contiguous no-offset "
                + role + " access; actual=" + accessPlan.kind() + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static int normalizedDimension(int dimension, int rank) {
        if (rank <= 0) {
            throw new UnsupportedOperationException("cpu1 GATHER requires rank > 0 input.");
        }
        int normalized = dimension < 0 ? dimension + rank : dimension;
        if (normalized < 0 || normalized >= rank) {
            throw new UnsupportedOperationException("cpu1 GATHER dimension out of bounds: dimension="
                    + dimension + ", rank=" + rank);
        }
        return normalized;
    }

    private static int[] reducedShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int inputDim = 0, outputDim = 0; inputDim < shape.length; inputDim++) {
            if (inputDim != axis) {
                reduced[outputDim++] = shape[inputDim];
            }
        }
        return reduced;
    }

    private static Cpu1IndexKernelId kernelId(DataType valueDataType, DataType indexDataType) {
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_BOOL_I64_DENSE_ARRAY;
        };
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }

    private static int product(int[] shape, int startInclusive, int endExclusive) {
        int product = 1;
        for (int i = startInclusive; i < endExclusive; i++) {
            product = Math.multiplyExact(product, shape[i]);
        }
        return product;
    }
}

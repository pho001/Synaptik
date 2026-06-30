package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.linalg.attention.Cpu1AttentionKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the dense direct scaled dot-product attention subset for cpu1.
 */
public final class Cpu1AttentionPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        if (operation.opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            return prepareAttention(node, descriptorIndex, config, operation);
        }
        if (operation.opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS) {
            return prepareWeights(node, descriptorIndex, config);
        }
        throw new UnsupportedOperationException("cpu1 attention preparer does not support " + operation.opType());
    }

    public static boolean isAttentionOp(Operation.OpType opType) {
        return opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                || opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS;
    }

    private Cpu1PreparedArtifact prepareAttention(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof scaledDotProductAttention attention)) {
            throw new IllegalArgumentException("cpu1 SCALED_DOT_PRODUCT_ATTENTION requires scaledDotProductAttention op");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION requires descriptors.");
        }
        int expectedInputs = attention.hasMask() ? 4 : 3;
        if (node.inputIds().size() != expectedInputs) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION expects "
                    + expectedInputs + " inputs, got " + node.inputIds().size());
        }
        CompiledTensorDescriptor query = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor key = descriptorIndex.byNodeId(node.inputIds().get(1));
        CompiledTensorDescriptor value = descriptorIndex.byNodeId(node.inputIds().get(2));
        CompiledTensorDescriptor mask = attention.hasMask()
                ? descriptorIndex.byNodeId(node.inputIds().get(3))
                : null;

        Cpu1StorageAccessPlan queryAccess = Cpu1StorageAccessPlan.fromDescriptor(query);
        Cpu1StorageAccessPlan keyAccess = Cpu1StorageAccessPlan.fromDescriptor(key);
        Cpu1StorageAccessPlan valueAccess = Cpu1StorageAccessPlan.fromDescriptor(value);
        Cpu1StorageAccessPlan maskAccess = mask == null ? null : Cpu1StorageAccessPlan.fromDescriptor(mask);
        Cpu1StorageAccessPlan outputAccess = Cpu1StorageAccessPlan.fromNode(node);

        int[] scoresShape = scoreShape(query.shape(), key.shape());
        int[] outputShape = outputShape(query.shape(), key.shape(), value.shape());
        requireAttentionContract(node, query, key, value, mask, scoresShape, outputShape);
        requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION", "query", queryAccess);
        requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION", "key", keyAccess);
        requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION", "value", valueAccess);
        if (maskAccess != null) {
            requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION", "mask", maskAccess);
        }
        requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION", "output", outputAccess);

        int batchCount = batchCount(outputShape);
        int queryLen = outputShape[outputShape.length - 2];
        int keyLen = key.shape()[key.rank() - 2];
        int depth = query.shape()[query.rank() - 1];
        int valueDim = outputShape[outputShape.length - 1];
        int totalRows = Math.multiplyExact(batchCount, queryLen);
        Cpu1LaunchConfig launchConfig = launchConfig(totalRows, config);
        int scratchSlotCount = Cpu1RangeLauncher.slotCount(totalRows, launchConfig);
        Cpu1VectorizationKind vectorizationKind = attentionVectorizationKind(node.dataType(), config);
        Cpu1PreparedAttentionUnit unit = new Cpu1PreparedAttentionUnit(
                node.id(),
                query.nodeId(),
                key.nodeId(),
                value.nodeId(),
                mask == null ? -1 : mask.nodeId(),
                -1,
                operation.opType(),
                node.dataType(),
                config.storageKind(),
                vectorizationKind,
                attentionKernelId(node.dataType(), config.storageKind(), vectorizationKind),
                query.shape(),
                key.shape(),
                value.shape(),
                mask == null ? null : mask.shape(),
                node.shape(),
                scoresShape,
                batchOffsets(query.shape(), outputShape),
                batchOffsets(key.shape(), outputShape),
                batchOffsets(value.shape(), outputShape),
                mask == null ? null : batchOffsets(mask.shape(), outputShape),
                batchCount,
                queryLen,
                keyLen,
                depth,
                valueDim,
                node.flatDataSize(),
                attention.getScale(),
                attention.hasMask(),
                config.useFastExpApprox(),
                launchConfig,
                launchPolicy(launchConfig),
                scratchSlotCount,
                queryAccess,
                keyAccess,
                valueAccess,
                maskAccess,
                outputAccess
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private Cpu1PreparedArtifact prepareWeights(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS requires descriptors.");
        }
        if (node.inputIds().size() != 1) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS expects 1 input, got "
                    + node.inputIds().size());
        }
        if (!isSupportedFloating(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS supports only "
                    + "FLOAT32/FLOAT64/BFLOAT16 output, got " + node.dataType());
        }
        CompiledTensorDescriptor attentionOutput = descriptorIndex.byNodeId(node.inputIds().getFirst());
        if (attentionOutput.opType() != Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS input must be a "
                    + "SCALED_DOT_PRODUCT_ATTENTION compiled node, got " + attentionOutput.opType());
        }
        if (!isSupportedFloating(attentionOutput.dataType())) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS requires floating "
                    + "attention output input, got " + attentionOutput.dataType());
        }
        if (attentionOutput.inputIds().size() < 3) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS input attention "
                    + "descriptor must expose q/k/v input ids, got " + attentionOutput.inputIds().size());
        }
        CompiledTensorDescriptor query = descriptorIndex.byNodeId(attentionOutput.inputIds().get(0));
        CompiledTensorDescriptor key = descriptorIndex.byNodeId(attentionOutput.inputIds().get(1));
        int[] expectedScoresShape = scoreShape(query.shape(), key.shape());
        if (!Arrays.equals(node.shape(), expectedScoresShape)) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS output shape must "
                    + "equal attention scores shape. expected=" + Arrays.toString(expectedScoresShape)
                    + ", actual=" + Arrays.toString(node.shape()));
        }
        Cpu1StorageAccessPlan outputAccess = Cpu1StorageAccessPlan.fromNode(node);
        requireDenseContiguousNoOffset("SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS", "output", outputAccess);
        Cpu1PreparedAttentionUnit unit = new Cpu1PreparedAttentionUnit(
                node.id(),
                -1,
                -1,
                -1,
                -1,
                attentionOutput.nodeId(),
                Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                node.dataType(),
                config.storageKind(),
                Cpu1VectorizationKind.SCALAR,
                weightsKernelId(node.dataType(), config.storageKind()),
                null,
                null,
                null,
                null,
                node.shape(),
                node.shape(),
                null,
                null,
                null,
                null,
                0,
                0,
                node.shape().length < 2 ? 0 : node.shape()[node.shape().length - 1],
                0,
                0,
                node.flatDataSize(),
                1.0d,
                false,
                config.useFastExpApprox(),
                Cpu1LaunchConfig.singleThread(),
                launchPolicy(Cpu1LaunchConfig.singleThread()),
                0,
                null,
                null,
                null,
                null,
                outputAccess
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private static void requireAttentionContract(
            CompiledNode node,
            CompiledTensorDescriptor query,
            CompiledTensorDescriptor key,
            CompiledTensorDescriptor value,
            CompiledTensorDescriptor mask,
            int[] scoresShape,
            int[] outputShape
    ) {
        if (!isSupportedFloating(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION supports only "
                    + "FLOAT32/FLOAT64/BFLOAT16 output, got " + node.dataType());
        }
        if (query.dataType() != node.dataType()
                || key.dataType() != node.dataType()
                || value.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION requires matching q/k/v/output "
                    + "dtype. q=" + query.dataType() + ", k=" + key.dataType()
                    + ", v=" + value.dataType() + ", output=" + node.dataType());
        }
        if (query.rank() < 2 || key.rank() < 2 || value.rank() < 2) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION requires q/k/v rank >= 2. "
                    + "qRank=" + query.rank() + ", kRank=" + key.rank() + ", vRank=" + value.rank());
        }
        if (query.shape()[query.rank() - 1] != key.shape()[key.rank() - 1]) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION query/key last dimension "
                    + "mismatch. q=" + Arrays.toString(query.shape()) + ", k=" + Arrays.toString(key.shape()));
        }
        if (key.shape()[key.rank() - 2] != value.shape()[value.rank() - 2]) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION key/value sequence dimension "
                    + "mismatch. k=" + Arrays.toString(key.shape()) + ", v=" + Arrays.toString(value.shape()));
        }
        if (!Arrays.equals(outputShape, node.shape())) {
            throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION output shape mismatch. expected="
                    + Arrays.toString(outputShape) + ", actual=" + Arrays.toString(node.shape()));
        }
        if (mask != null) {
            if (mask.dataType() != DataType.BOOL) {
                throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION mask must be BOOL, got "
                        + mask.dataType());
            }
            if (!Arrays.equals(mask.shape(), scoresShape)) {
                throw new UnsupportedOperationException("cpu1 SCALED_DOT_PRODUCT_ATTENTION direct mask shape must "
                        + "equal scores shape. scores=" + Arrays.toString(scoresShape)
                        + ", mask=" + Arrays.toString(mask.shape()));
            }
        }
        Math.toIntExact(query.logicalElementCount());
        Math.toIntExact(key.logicalElementCount());
        Math.toIntExact(value.logicalElementCount());
        Math.toIntExact(node.flatDataSize());
    }

    private static boolean isSupportedFloating(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16;
    }

    private static void requireDenseContiguousNoOffset(
            String opName,
            String role,
            Cpu1StorageAccessPlan accessPlan
    ) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 " + opName + " dense first slice supports only dense "
                + "contiguous no-offset " + role + " access; actual=" + accessPlan.kind()
                + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static int[] scoreShape(int[] queryShape, int[] keyShape) {
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] outBatch = broadcastLeadingShape(qBatch, kBatch);
        int[] out = Arrays.copyOf(outBatch, outBatch.length + 2);
        out[outBatch.length] = queryShape[queryShape.length - 2];
        out[outBatch.length + 1] = keyShape[keyShape.length - 2];
        return out;
    }

    private static int[] outputShape(int[] queryShape, int[] keyShape, int[] valueShape) {
        int[] scoresShape = scoreShape(queryShape, keyShape);
        int[] scoresBatch = Arrays.copyOf(scoresShape, scoresShape.length - 2);
        int[] valueBatch = Arrays.copyOf(valueShape, valueShape.length - 2);
        int[] outBatch = broadcastLeadingShape(scoresBatch, valueBatch);
        int[] out = Arrays.copyOf(outBatch, outBatch.length + 2);
        out[outBatch.length] = queryShape[queryShape.length - 2];
        out[outBatch.length + 1] = valueShape[valueShape.length - 1];
        return out;
    }

    private static int[] broadcastLeadingShape(int[] first, int[] second) {
        int rank = Math.max(first.length, second.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int firstDim = i < rank - first.length ? 1 : first[i - (rank - first.length)];
            int secondDim = i < rank - second.length ? 1 : second[i - (rank - second.length)];
            if (firstDim != secondDim && firstDim != 1 && secondDim != 1) {
                throw new UnsupportedOperationException("cpu1 attention batch dimensions are not broadcast-compatible: "
                        + Arrays.toString(first) + " vs " + Arrays.toString(second));
            }
            out[i] = Math.max(firstDim, secondDim);
        }
        return out;
    }

    private static int batchCount(int[] shape) {
        int count = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            count = Math.multiplyExact(count, shape[i]);
        }
        return count;
    }

    private static int[] batchOffsets(int[] sourceShape, int[] targetShape) {
        int targetBatchRank = targetShape.length - 2;
        int sourceBatchRank = sourceShape.length - 2;
        int count = batchCount(targetShape);
        int[] offsets = new int[count];
        int[] sourceBatchStrides = batchStrides(sourceShape);
        for (int batch = 0; batch < count; batch++) {
            int remainder = batch;
            int offset = 0;
            for (int targetDim = targetBatchRank - 1; targetDim >= 0; targetDim--) {
                int coordinate = remainder % targetShape[targetDim];
                remainder /= targetShape[targetDim];
                int sourceDim = targetDim - (targetBatchRank - sourceBatchRank);
                if (sourceDim >= 0) {
                    int sourceSize = sourceShape[sourceDim];
                    int sourceCoordinate = sourceSize == 1 ? 0 : coordinate;
                    if (sourceSize != 1 && sourceSize != targetShape[targetDim]) {
                        throw new UnsupportedOperationException("cpu1 attention cannot map broadcast batch dimension "
                                + "source=" + Arrays.toString(sourceShape)
                                + ", target=" + Arrays.toString(targetShape));
                    }
                    offset += sourceCoordinate * sourceBatchStrides[sourceDim];
                }
            }
            offsets[batch] = offset;
        }
        return offsets;
    }

    private static int[] batchStrides(int[] shape) {
        int batchRank = shape.length - 2;
        int[] strides = new int[batchRank];
        int stride = Math.multiplyExact(shape[shape.length - 2], shape[shape.length - 1]);
        for (int dim = batchRank - 1; dim >= 0; dim--) {
            strides[dim] = stride;
            stride = Math.multiplyExact(stride, shape[dim]);
        }
        return strides;
    }

    private static Cpu1VectorizationKind attentionVectorizationKind(DataType dataType, Cpu1PrepareConfig config) {
        if (config.vectorizationKind() == Cpu1VectorizationKind.VECTOR
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)) {
            return Cpu1VectorizationKind.VECTOR;
        }
        return Cpu1VectorizationKind.SCALAR;
    }

    private static Cpu1AttentionKernelId attentionKernelId(
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                    ? Cpu1AttentionKernelId.ATTENTION_F32_SEGMENT_DENSE_VECTOR
                    : Cpu1AttentionKernelId.ATTENTION_F32_SEGMENT_DENSE_SCALAR)
                    : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                    ? Cpu1AttentionKernelId.ATTENTION_F32_ARRAY_DENSE_VECTOR
                    : Cpu1AttentionKernelId.ATTENTION_F32_ARRAY_DENSE_SCALAR);
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                    ? Cpu1AttentionKernelId.ATTENTION_F64_SEGMENT_DENSE_VECTOR
                    : Cpu1AttentionKernelId.ATTENTION_F64_SEGMENT_DENSE_SCALAR)
                    : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                    ? Cpu1AttentionKernelId.ATTENTION_F64_ARRAY_DENSE_VECTOR
                    : Cpu1AttentionKernelId.ATTENTION_F64_ARRAY_DENSE_SCALAR);
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AttentionKernelId.ATTENTION_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1AttentionKernelId.ATTENTION_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 attention supports only floating dtypes.");
        };
    }

    private static Cpu1AttentionKernelId weightsKernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AttentionKernelId.ATTENTION_WEIGHTS_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1AttentionKernelId.ATTENTION_WEIGHTS_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AttentionKernelId.ATTENTION_WEIGHTS_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1AttentionKernelId.ATTENTION_WEIGHTS_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AttentionKernelId.ATTENTION_WEIGHTS_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1AttentionKernelId.ATTENTION_WEIGHTS_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 attention weights supports only floating dtypes.");
        };
    }

    private static Cpu1LaunchConfig launchConfig(int totalRows, Cpu1PrepareConfig config) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 attention dispatch requires CpuKernelConfig.");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1 || totalRows <= 1 || totalRows < cpuKernelConfig.attentionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, totalRows);
        int targets = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
        int candidate = (Math.max(1, totalRows) + targets - 1) / targets;
        return Cpu1LaunchConfig.parallel(
                plannedWorkers,
                Math.max(cpuKernelConfig.minReductionChunkSize(), candidate)
        );
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }
}

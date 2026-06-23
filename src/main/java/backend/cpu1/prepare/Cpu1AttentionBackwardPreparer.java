package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.linalg.attention.backward.Cpu1AttentionBackwardKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.region.specialization.RegionSpecializationCandidate;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import graph.compile.planning.region.specialization.SdpaBackwardOutputKind;
import graph.compile.planning.region.specialization.SdpaBackwardSpecializationPayload;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares dense cpu1 routes for canonical SDPA backward specialized regions.
 */
public final class Cpu1AttentionBackwardPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode outputNode,
            RegionSpecializationCandidate candidate,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(outputNode, "outputNode cannot be null");
        Objects.requireNonNull(candidate, "candidate cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        if (candidate.kind() != RegionSpecializationKind.SDPA_BACKWARD) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD preparer does not support " + candidate.kind());
        }
        if (!(candidate.payload() instanceof SdpaBackwardSpecializationPayload payload)) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD requires SdpaBackwardSpecializationPayload.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD requires descriptors.");
        }
        if (candidate.outputValueRef().nodeId() != outputNode.id()) {
            throw new IllegalStateException("SDPA_BACKWARD specialization output node mismatch. candidate="
                    + candidate.outputValueRef().nodeId() + ", outputNode=" + outputNode.id());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY
                && config.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD route supports only JAVA_ARRAY or "
                    + "MEMORY_SEGMENT storage, got " + config.storageKind());
        }
        if (outputNode.dataType() != DataType.FLOAT32 && outputNode.dataType() != DataType.FLOAT64) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD first route supports only FLOAT32/FLOAT64, got "
                    + outputNode.dataType() + ". Follow-up: add BF16 accumulation and output conversion.");
        }

        SdpaBackwardOutputKind outputKind = payload.outputKind();
        boolean needsQuery = outputKind == SdpaBackwardOutputKind.KEY;
        boolean needsKey = outputKind == SdpaBackwardOutputKind.QUERY;
        boolean needsValue = outputKind != SdpaBackwardOutputKind.VALUE;
        boolean needsMask = payload.hasMask() && outputKind != SdpaBackwardOutputKind.VALUE;
        int queryNodeId = needsQuery ? payload.queryNodeId() : -1;
        int keyNodeId = needsKey ? payload.keyNodeId() : -1;
        int valueNodeId = needsValue ? payload.valueNodeId() : -1;
        int maskNodeId = needsMask ? payload.maskNodeId() : -1;

        CompiledTensorDescriptor weights = descriptorIndex.byNodeId(payload.weightsNodeId());
        CompiledTensorDescriptor outGrad = descriptorIndex.byNodeId(payload.outGradNodeId());
        CompiledTensorDescriptor query = queryNodeId >= 0 ? descriptorIndex.byNodeId(queryNodeId) : null;
        CompiledTensorDescriptor key = keyNodeId >= 0 ? descriptorIndex.byNodeId(keyNodeId) : null;
        CompiledTensorDescriptor value = valueNodeId >= 0 ? descriptorIndex.byNodeId(valueNodeId) : null;
        CompiledTensorDescriptor mask = maskNodeId >= 0 ? descriptorIndex.byNodeId(maskNodeId) : null;

        requireDType("weights", weights, outputNode.dataType());
        requireDType("outGrad", outGrad, outputNode.dataType());
        if (query != null) {
            requireDType("query", query, outputNode.dataType());
        }
        if (key != null) {
            requireDType("key", key, outputNode.dataType());
        }
        if (value != null) {
            requireDType("value", value, outputNode.dataType());
        }
        if (mask != null && mask.dataType() != DataType.BOOL) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD mask must be BOOL, got " + mask.dataType());
        }

        ShapeContract shape = requireShapeContract(outputNode, payload, weights, outGrad, query, key, value, mask);
        Cpu1StorageAccessPlan weightsAccess = Cpu1StorageAccessPlan.fromDescriptor(weights);
        Cpu1StorageAccessPlan outGradAccess = Cpu1StorageAccessPlan.fromDescriptor(outGrad);
        Cpu1StorageAccessPlan queryAccess = query == null ? null : Cpu1StorageAccessPlan.fromDescriptor(query);
        Cpu1StorageAccessPlan keyAccess = key == null ? null : Cpu1StorageAccessPlan.fromDescriptor(key);
        Cpu1StorageAccessPlan valueAccess = value == null ? null : Cpu1StorageAccessPlan.fromDescriptor(value);
        Cpu1StorageAccessPlan maskAccess = mask == null ? null : Cpu1StorageAccessPlan.fromDescriptor(mask);
        Cpu1StorageAccessPlan outputAccess = Cpu1StorageAccessPlan.fromNode(outputNode);
        requireDenseContiguousNoOffset("weights", weightsAccess);
        requireDenseContiguousOrBroadcastNoOffset("outGrad", outGradAccess);
        if (queryAccess != null) {
            requireDenseContiguousNoOffset("query", queryAccess);
        }
        if (keyAccess != null) {
            requireDenseContiguousNoOffset("key", keyAccess);
        }
        if (valueAccess != null) {
            requireDenseContiguousNoOffset("value", valueAccess);
        }
        if (maskAccess != null) {
            requireDenseContiguousNoOffset("mask", maskAccess);
        }
        requireDenseContiguousNoOffset("output", outputAccess);

        int rowCount = switch (payload.outputKind()) {
            case QUERY -> Math.multiplyExact(shape.batchCount(), shape.queryLen());
            case KEY, VALUE -> Math.multiplyExact(shape.batchCount(), shape.keyLen());
        };
        Cpu1LaunchConfig launchConfig = launchConfig(outputKind, shape, config);
        int scratchSlotCount = payload.outputKind() == SdpaBackwardOutputKind.VALUE
                ? 0
                : Cpu1RangeLauncher.slotCount(rowCount, launchConfig);
        Cpu1VectorizationKind vectorizationKind = attentionBackwardVectorizationKind(
                outputNode.dataType(),
                config.storageKind(),
                config
        );
        Cpu1PreparedAttentionBackwardUnit unit = new Cpu1PreparedAttentionBackwardUnit(
                outputNode.id(),
                outputKind,
                outputNode.dataType(),
                config.storageKind(),
                vectorizationKind,
                kernelId(outputKind, outputNode.dataType(), config.storageKind(), vectorizationKind),
                payload.weightsNodeId(),
                payload.outGradNodeId(),
                queryNodeId,
                keyNodeId,
                valueNodeId,
                maskNodeId,
                weights.shape(),
                outGrad.shape(),
                query == null ? null : query.shape(),
                key == null ? null : key.shape(),
                value == null ? null : value.shape(),
                mask == null ? null : mask.shape(),
                outputNode.shape(),
                shape.batchCount(),
                shape.queryLen(),
                shape.keyLen(),
                shape.depth(),
                shape.valueDim(),
                outputNode.flatDataSize(),
                payload.scale(),
                needsMask,
                launchConfig,
                launchPolicy(launchConfig),
                scratchSlotCount,
                weightsAccess,
                outGradAccess,
                queryAccess,
                keyAccess,
                valueAccess,
                maskAccess,
                outputAccess
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private static ShapeContract requireShapeContract(
            CompiledNode outputNode,
            SdpaBackwardSpecializationPayload payload,
            CompiledTensorDescriptor weights,
            CompiledTensorDescriptor outGrad,
            CompiledTensorDescriptor query,
            CompiledTensorDescriptor key,
            CompiledTensorDescriptor value,
            CompiledTensorDescriptor mask
    ) {
        int[] weightsShape = weights.shape();
        int[] outGradShape = outGrad.shape();
        if (weightsShape.length < 2 || outGradShape.length < 2 || outputNode.shape().length < 2) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD requires rank >= 2 weights/outGrad/output. "
                    + "weights=" + Arrays.toString(weightsShape)
                    + ", outGrad=" + Arrays.toString(outGradShape)
                    + ", output=" + Arrays.toString(outputNode.shape()));
        }
        int queryLen = weightsShape[weightsShape.length - 2];
        int keyLen = weightsShape[weightsShape.length - 1];
        int valueDim = outGradShape[outGradShape.length - 1];
        int[] batchShape = batchShape(weightsShape);
        requireShape("outGrad", outGradShape, append(batchShape, queryLen, valueDim));
        if (mask != null) {
            requireShape("mask", mask.shape(), weightsShape);
        }
        int depth = switch (payload.outputKind()) {
            case QUERY -> {
                requirePresent("key", key);
                requirePresent("value", value);
                requireShape("key", key.shape(), append(batchShape, keyLen, key.shape()[key.rank() - 1]));
                requireShape("value", value.shape(), append(batchShape, keyLen, valueDim));
                int keyDepth = key.shape()[key.rank() - 1];
                requireShape("output dQ", outputNode.shape(), append(batchShape, queryLen, keyDepth));
                yield keyDepth;
            }
            case KEY -> {
                requirePresent("query", query);
                requirePresent("value", value);
                requireShape("query", query.shape(), append(batchShape, queryLen, query.shape()[query.rank() - 1]));
                requireShape("value", value.shape(), append(batchShape, keyLen, valueDim));
                int queryDepth = query.shape()[query.rank() - 1];
                requireShape("output dK", outputNode.shape(), append(batchShape, keyLen, queryDepth));
                yield queryDepth;
            }
            case VALUE -> {
                requireShape("output dV", outputNode.shape(), append(batchShape, keyLen, valueDim));
                yield 1;
            }
        };
        int batchCount = 1;
        for (int dim : batchShape) {
            batchCount = Math.multiplyExact(batchCount, dim);
        }
        return new ShapeContract(batchCount, queryLen, keyLen, depth, valueDim);
    }

    private static void requirePresent(String role, CompiledTensorDescriptor descriptor) {
        if (descriptor == null) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD missing required " + role + " descriptor.");
        }
    }

    private static void requireDType(String role, CompiledTensorDescriptor descriptor, DataType expected) {
        if (descriptor.dataType() != expected) {
            throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD " + role + " dtype mismatch. expected="
                    + expected + ", actual=" + descriptor.dataType());
        }
    }

    private static void requireShape(String role, int[] actual, int[] expected) {
        if (Arrays.equals(actual, expected)) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD " + role + " shape mismatch. expected="
                + Arrays.toString(expected) + ", actual=" + Arrays.toString(actual)
                + ". Follow-up: add explicit broadcast-aware SDPA backward lowering.");
    }

    private static int[] batchShape(int[] shape) {
        return Arrays.copyOf(shape, shape.length - 2);
    }

    private static int[] append(int[] batchShape, int rows, int cols) {
        int[] out = Arrays.copyOf(batchShape, batchShape.length + 2);
        out[out.length - 2] = rows;
        out[out.length - 1] = cols;
        return out;
    }

    private static void requireDenseContiguousNoOffset(String role, Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD first route supports only dense contiguous "
                + "no-offset " + role + " access; actual=" + accessPlan.kind()
                + rejectionSuffix(accessPlan)
                + ". Follow-up: add strided/broadcast access without hidden materialization.");
    }

    private static void requireDenseContiguousOrBroadcastNoOffset(String role, Cpu1StorageAccessPlan accessPlan) {
        if ((accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS
                || accessPlan.kind() == Cpu1StorageAccessKind.BROADCAST)
                && accessPlan.storageOffset() == 0) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 SDPA_BACKWARD first route supports only dense contiguous "
                + "or explicit broadcast no-offset " + role + " access; actual=" + accessPlan.kind()
                + ", storageOffset=" + accessPlan.storageOffset()
                + rejectionSuffix(accessPlan)
                + ". Follow-up: add strided/offset access without hidden materialization.");
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static Cpu1AttentionBackwardKernelId kernelId(
            SdpaBackwardOutputKind outputKind,
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return switch (dataType) {
            case FLOAT32 -> switch (outputKind) {
                case QUERY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F32_ARRAY_DENSE_SCALAR);
                case KEY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F32_ARRAY_DENSE_SCALAR);
                case VALUE -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F32_ARRAY_DENSE_SCALAR);
            };
            case FLOAT64 -> switch (outputKind) {
                case QUERY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DQ_F64_ARRAY_DENSE_SCALAR);
                case KEY -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DK_F64_ARRAY_DENSE_SCALAR);
                case VALUE -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_SEGMENT_DENSE_SCALAR)
                        : (vectorizationKind == Cpu1VectorizationKind.VECTOR
                        ? Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_ARRAY_DENSE_VECTOR
                        : Cpu1AttentionBackwardKernelId.SDPA_BACKWARD_DV_F64_ARRAY_DENSE_SCALAR);
            };
            case BFLOAT16, INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 SDPA_BACKWARD first route supports only FLOAT32/FLOAT64, got " + dataType);
        };
    }

    private static Cpu1VectorizationKind attentionBackwardVectorizationKind(
            DataType dataType,
            Cpu1StorageKind storageKind,
            Cpu1PrepareConfig config
    ) {
        if (config.vectorizationKind() == Cpu1VectorizationKind.VECTOR
                && (storageKind == Cpu1StorageKind.JAVA_ARRAY || storageKind == Cpu1StorageKind.MEMORY_SEGMENT)
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)) {
            return Cpu1VectorizationKind.VECTOR;
        }
        return Cpu1VectorizationKind.SCALAR;
    }

    private static Cpu1LaunchConfig launchConfig(
            SdpaBackwardOutputKind outputKind,
            ShapeContract shape,
            Cpu1PrepareConfig config
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 SDPA_BACKWARD dispatch requires CpuKernelConfig.");
        }
        int itemCount = launchItemCount(outputKind, shape);
        long workPerItem = estimatedWorkPerItem(outputKind, shape);
        long totalWork = Math.multiplyExact((long) itemCount, workPerItem);
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1 || itemCount <= 1 || totalWork < cpuKernelConfig.attentionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, itemCount);
        int targetTasks = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
        int candidateItemsPerChunk = (itemCount + targetTasks - 1) / targetTasks;
        int minItemsPerChunk = (int) Math.max(
                1L,
                cpuKernelConfig.minReductionChunkSize() / Math.max(1L, workPerItem)
        );
        return Cpu1LaunchConfig.parallel(
                plannedWorkers,
                Math.max(minItemsPerChunk, candidateItemsPerChunk)
        );
    }

    private static int launchItemCount(SdpaBackwardOutputKind outputKind, ShapeContract shape) {
        return switch (outputKind) {
            case QUERY -> Math.multiplyExact(shape.batchCount(), shape.queryLen());
            case KEY, VALUE -> Math.multiplyExact(shape.batchCount(), shape.keyLen());
        };
    }

    private static long estimatedWorkPerItem(SdpaBackwardOutputKind outputKind, ShapeContract shape) {
        return switch (outputKind) {
            case QUERY -> Math.addExact(
                    Math.multiplyExact((long) shape.keyLen(), shape.valueDim()),
                    Math.multiplyExact((long) shape.keyLen(), shape.depth())
            );
            case KEY -> Math.addExact(
                    Math.multiplyExact(
                            (long) shape.queryLen(),
                            Math.multiplyExact((long) shape.keyLen(), shape.valueDim())
                    ),
                    Math.multiplyExact((long) shape.queryLen(), shape.depth())
            );
            case VALUE -> Math.multiplyExact((long) shape.queryLen(), shape.valueDim());
        };
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }

    private record ShapeContract(
            int batchCount,
            int queryLen,
            int keyLen,
            int depth,
            int valueDim
    ) {
    }
}

package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1ScratchBufferSpec;
import backend.cpu1.kernels.loss.crossentropy.Cpu1DenseCrossEntropyKernelId;
import backend.cpu1.kernels.loss.crossentropy.Cpu1CrossEntropyKernelId;
import backend.cpu1.kernels.loss.nll.Cpu1NllLossKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.loss.crossEntropyLoss;
import operations.loss.crossEntropyLossIndices;
import operations.loss.nllLoss;
import tensor.DataType;
import tensor.loss.LossReduction;

import java.util.Arrays;
import java.util.Objects;

public final class Cpu1LossPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        return switch (operation.opType()) {
            case CROSS_ENTROPY_LOSS -> prepareDenseCrossEntropyLoss(node, descriptorIndex, config, operation);
            case CROSS_ENTROPY_LOSS_INDICES -> prepareCrossEntropyLossIndices(node, descriptorIndex, config, operation);
            case NLL_LOSS -> prepareNllLoss(node, descriptorIndex, config, operation);
            default -> throw new UnsupportedOperationException("cpu1 loss preparer does not support "
                    + operation.opType());
        };
    }

    private Cpu1PreparedArtifact prepareCrossEntropyLossIndices(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof crossEntropyLossIndices loss)) {
            throw new IllegalArgumentException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires crossEntropyLossIndices op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES expects 2 inputs, got "
                    + node.inputIds().size());
        }
        int logitsNodeId = node.inputIds().get(0);
        int targetsNodeId = node.inputIds().get(1);
        CompiledTensorDescriptor logits = descriptorIndex.byNodeId(logitsNodeId);
        CompiledTensorDescriptor targets = descriptorIndex.byNodeId(targetsNodeId);
        requireContract(node, logits, targets, loss);

        int classAxis = normalizedAxis("cpu1 CROSS_ENTROPY_LOSS_INDICES", loss.getClassDimension(), logits.rank());
        int axisSize = Math.toIntExact(logits.shape()[classAxis]);
        int axisStride = Math.toIntExact(logits.strides()[classAxis]);
        int groupCount = Math.toIntExact(targets.logicalElementCount());
        Cpu1LaunchConfig launchConfig = launchConfig(groupCount, config);
        Cpu1PreparedCrossEntropyLossUnit unit = new Cpu1PreparedCrossEntropyLossUnit(
                node.id(),
                logitsNodeId,
                targetsNodeId,
                operation.opType(),
                logits.dataType(),
                targets.dataType(),
                config.storageKind(),
                kernelId(logits.dataType(), targets.dataType(), config.storageKind()),
                classAxis,
                axisSize,
                axisStride,
                groupCount,
                logits.shape(),
                targets.shape(),
                loss.getReduction(),
                loss.getIgnoreIndex(),
                launchConfig,
                launchPolicy(launchConfig),
                scratchBufferSpec(loss.getReduction(), groupCount, launchConfig)
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isLossOp(Operation.OpType opType) {
        return opType == Operation.OpType.CROSS_ENTROPY_LOSS
                || opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES
                || opType == Operation.OpType.NLL_LOSS;
    }

    private Cpu1PreparedArtifact prepareDenseCrossEntropyLoss(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof crossEntropyLoss loss)) {
            throw new IllegalArgumentException("cpu1 CROSS_ENTROPY_LOSS requires crossEntropyLoss op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS expects 2 inputs, got "
                    + node.inputIds().size());
        }
        int logitsNodeId = node.inputIds().get(0);
        int targetsNodeId = node.inputIds().get(1);
        CompiledTensorDescriptor logits = descriptorIndex.byNodeId(logitsNodeId);
        CompiledTensorDescriptor targets = descriptorIndex.byNodeId(targetsNodeId);
        requireDenseCrossEntropyContract(node, logits, targets, loss);

        int classAxis = normalizedAxis("cpu1 CROSS_ENTROPY_LOSS", loss.getClassDimension(), logits.rank());
        int axisSize = Math.toIntExact(logits.shape()[classAxis]);
        int axisStride = Math.toIntExact(logits.strides()[classAxis]);
        int groupCount = Math.toIntExact(sampleCount(logits.shape(), classAxis));
        Cpu1LaunchConfig launchConfig = launchConfig(groupCount, config);
        Cpu1PreparedDenseCrossEntropyLossUnit unit = new Cpu1PreparedDenseCrossEntropyLossUnit(
                node.id(),
                logitsNodeId,
                targetsNodeId,
                operation.opType(),
                logits.dataType(),
                config.storageKind(),
                denseCrossEntropyKernelId(logits.dataType(), config.storageKind()),
                classAxis,
                axisSize,
                axisStride,
                groupCount,
                logits.shape(),
                launchConfig,
                launchPolicy(launchConfig),
                denseCrossEntropyScratchBufferSpec(groupCount, launchConfig)
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private Cpu1PreparedArtifact prepareNllLoss(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof nllLoss loss)) {
            throw new IllegalArgumentException("cpu1 NLL_LOSS requires nllLoss op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS expects 2 inputs, got " + node.inputIds().size());
        }
        int logProbsNodeId = node.inputIds().get(0);
        int targetsNodeId = node.inputIds().get(1);
        CompiledTensorDescriptor logProbs = descriptorIndex.byNodeId(logProbsNodeId);
        CompiledTensorDescriptor targets = descriptorIndex.byNodeId(targetsNodeId);
        requireNllContract(node, logProbs, targets, loss);

        int classAxis = normalizedAxis("cpu1 NLL_LOSS", loss.getClassDimension(), logProbs.rank());
        int axisSize = Math.toIntExact(logProbs.shape()[classAxis]);
        int axisStride = Math.toIntExact(logProbs.strides()[classAxis]);
        int groupCount = Math.toIntExact(sampleCount(logProbs.shape(), classAxis));
        Cpu1LaunchConfig launchConfig = launchConfig(groupCount, config);
        Cpu1PreparedNllLossUnit unit = new Cpu1PreparedNllLossUnit(
                node.id(),
                logProbsNodeId,
                targetsNodeId,
                operation.opType(),
                logProbs.dataType(),
                config.storageKind(),
                nllKernelId(logProbs.dataType(), config.storageKind()),
                classAxis,
                axisSize,
                axisStride,
                groupCount,
                logProbs.shape(),
                launchConfig,
                launchPolicy(launchConfig),
                nllScratchBufferSpec(groupCount, launchConfig)
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private static void requireContract(
            CompiledNode node,
            CompiledTensorDescriptor logits,
            CompiledTensorDescriptor targets,
            crossEntropyLossIndices loss
    ) {
        if (!isSupportedLogitsDType(logits.dataType()) || node.dataType() != logits.dataType()) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires FLOAT32/FLOAT64/BFLOAT16 "
                    + "logits and output with matching dtype, logits=" + logits.dataType()
                    + ", output=" + node.dataType());
        }
        if (targets.dataType() != DataType.INT32 && targets.dataType() != DataType.INT64) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires INT32/INT64 targets, got "
                    + targets.dataType());
        }
        if (!logits.denseContiguousWithoutOffset()
                || !targets.denseContiguousWithoutOffset()
                || node.storageOffset() != 0
                || !node.contiguous()) {
            throw new UnsupportedOperationException(
                    "cpu1 CROSS_ENTROPY_LOSS_INDICES first version requires dense contiguous logits/targets/output.");
        }
        if (logits.rank() <= 0) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires logits rank > 0.");
        }
        int classAxis = normalizedAxis("cpu1 CROSS_ENTROPY_LOSS_INDICES", loss.getClassDimension(), logits.rank());
        if (logits.shape()[classAxis] <= 0) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES requires non-empty class axis.");
        }
        int[] expectedTargetShape = reduceShape(logits.shape(), classAxis);
        if (!Arrays.equals(expectedTargetShape, targets.shape())) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES target shape mismatch. expected="
                    + Arrays.toString(expectedTargetShape) + ", actual=" + Arrays.toString(targets.shape()));
        }
        int[] expectedOutputShape = loss.getReduction() == LossReduction.NONE
                ? expectedTargetShape
                : new int[]{1};
        if (!Arrays.equals(expectedOutputShape, node.shape())) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES output shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(node.shape()));
        }
        if (targets.logicalElementCount() <= 0 || targets.logicalElementCount() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS_INDICES target element count is unsupported: "
                    + targets.logicalElementCount());
        }
    }

    private static void requireNllContract(
            CompiledNode node,
            CompiledTensorDescriptor logProbs,
            CompiledTensorDescriptor targets,
            nllLoss loss
    ) {
        if (!isSupportedFloatingDType(logProbs.dataType()) || node.dataType() != logProbs.dataType()) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS requires FLOAT32/FLOAT64/BFLOAT16 "
                    + "logProbs and output with matching dtype, logProbs=" + logProbs.dataType()
                    + ", output=" + node.dataType());
        }
        if (targets.dataType() != logProbs.dataType()) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS requires dense targets dtype to match logProbs, "
                    + "logProbs=" + logProbs.dataType() + ", targets=" + targets.dataType());
        }
        if (!logProbs.denseContiguousWithoutOffset()
                || !targets.denseContiguousWithoutOffset()
                || node.storageOffset() != 0
                || !node.contiguous()) {
            throw new UnsupportedOperationException(
                    "cpu1 NLL_LOSS first version requires dense contiguous logProbs/targets/output.");
        }
        if (logProbs.rank() <= 0) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS requires logProbs rank > 0.");
        }
        int classAxis = normalizedAxis("cpu1 NLL_LOSS", loss.getClassDimension(), logProbs.rank());
        if (!Arrays.equals(logProbs.shape(), targets.shape())) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS target shape mismatch. expected="
                    + Arrays.toString(logProbs.shape()) + ", actual=" + Arrays.toString(targets.shape()));
        }
        if (!Arrays.equals(new int[]{1}, node.shape())) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS output shape mismatch. expected=[1], actual="
                    + Arrays.toString(node.shape()));
        }
        if (logProbs.logicalElementCount() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS logProbs element count is unsupported: "
                    + logProbs.logicalElementCount());
        }
        long groupCount = sampleCount(logProbs.shape(), classAxis);
        if (groupCount > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 NLL_LOSS group count is unsupported: " + groupCount);
        }
    }

    private static void requireDenseCrossEntropyContract(
            CompiledNode node,
            CompiledTensorDescriptor logits,
            CompiledTensorDescriptor targets,
            crossEntropyLoss loss
    ) {
        if (!isSupportedFloatingDType(logits.dataType()) || node.dataType() != logits.dataType()) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS requires FLOAT32/FLOAT64/BFLOAT16 "
                    + "logits and output with matching dtype, logits=" + logits.dataType()
                    + ", output=" + node.dataType());
        }
        if (targets.dataType() != logits.dataType()) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS requires dense targets dtype to match logits, "
                    + "logits=" + logits.dataType() + ", targets=" + targets.dataType());
        }
        if (!logits.denseContiguousWithoutOffset()
                || !targets.denseContiguousWithoutOffset()
                || node.storageOffset() != 0
                || !node.contiguous()) {
            throw new UnsupportedOperationException(
                    "cpu1 CROSS_ENTROPY_LOSS first version requires dense contiguous logits/targets/output.");
        }
        if (logits.rank() <= 0) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS requires logits rank > 0.");
        }
        int classAxis = normalizedAxis("cpu1 CROSS_ENTROPY_LOSS", loss.getClassDimension(), logits.rank());
        if (logits.shape()[classAxis] <= 0) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS requires non-empty class axis.");
        }
        if (!Arrays.equals(logits.shape(), targets.shape())) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS target shape mismatch. expected="
                    + Arrays.toString(logits.shape()) + ", actual=" + Arrays.toString(targets.shape()));
        }
        if (!Arrays.equals(new int[]{1}, node.shape())) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS output shape mismatch. expected=[1], actual="
                    + Arrays.toString(node.shape()));
        }
        if (logits.logicalElementCount() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS logits element count is unsupported: "
                    + logits.logicalElementCount());
        }
        long groupCount = sampleCount(logits.shape(), classAxis);
        if (groupCount <= 0 || groupCount > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("cpu1 CROSS_ENTROPY_LOSS group count is unsupported: " + groupCount);
        }
    }

    private static boolean isSupportedLogitsDType(DataType dataType) {
        return isSupportedFloatingDType(dataType);
    }

    private static boolean isSupportedFloatingDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16;
    }

    private static Cpu1CrossEntropyKernelId kernelId(
            DataType logitsDataType,
            DataType targetDataType,
            Cpu1StorageKind storageKind
    ) {
        boolean segment = storageKind == Cpu1StorageKind.MEMORY_SEGMENT;
        return switch (logitsDataType) {
            case FLOAT32 -> targetDataType == DataType.INT32
                    ? (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I32_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I32_ARRAY_DENSE_SCALAR)
                    : (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I64_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F32_I64_ARRAY_DENSE_SCALAR);
            case FLOAT64 -> targetDataType == DataType.INT32
                    ? (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F64_I32_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F64_I32_ARRAY_DENSE_SCALAR)
                    : (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F64_I64_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_F64_I64_ARRAY_DENSE_SCALAR);
            case BFLOAT16 -> targetDataType == DataType.INT32
                    ? (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I32_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I32_ARRAY_DENSE_SCALAR)
                    : (segment
                            ? Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I64_SEGMENT_DENSE_SCALAR
                            : Cpu1CrossEntropyKernelId.CROSS_ENTROPY_INDICES_BF16_I64_ARRAY_DENSE_SCALAR);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 CROSS_ENTROPY_LOSS_INDICES requires floating logits dtype.");
        };
    }

    private static Cpu1NllLossKernelId nllKernelId(DataType dataType, Cpu1StorageKind storageKind) {
        boolean segment = storageKind == Cpu1StorageKind.MEMORY_SEGMENT;
        return switch (dataType) {
            case FLOAT32 -> segment
                    ? Cpu1NllLossKernelId.NLL_DENSE_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1NllLossKernelId.NLL_DENSE_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> segment
                    ? Cpu1NllLossKernelId.NLL_DENSE_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1NllLossKernelId.NLL_DENSE_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> segment
                    ? Cpu1NllLossKernelId.NLL_DENSE_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1NllLossKernelId.NLL_DENSE_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 NLL_LOSS requires floating dtype.");
        };
    }

    private static Cpu1DenseCrossEntropyKernelId denseCrossEntropyKernelId(
            DataType dataType,
            Cpu1StorageKind storageKind
    ) {
        boolean segment = storageKind == Cpu1StorageKind.MEMORY_SEGMENT;
        return switch (dataType) {
            case FLOAT32 -> segment
                    ? Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> segment
                    ? Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> segment
                    ? Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1DenseCrossEntropyKernelId.CROSS_ENTROPY_DENSE_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 CROSS_ENTROPY_LOSS requires floating dtype.");
        };
    }

    private static Cpu1LaunchConfig launchConfig(int groupCount, Cpu1PrepareConfig config) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 loss dispatch requires CpuKernelConfig.");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1 || groupCount < cpuKernelConfig.reductionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, Math.max(1, groupCount));
        return Cpu1LaunchConfig.parallel(
                plannedWorkers,
                groupChunkSize(groupCount, plannedWorkers, cpuKernelConfig)
        );
    }

    private static int groupChunkSize(
            int groupCount,
            int plannedWorkers,
            CpuKernelConfig cpuKernelConfig
    ) {
        int targets = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
        int candidate = (Math.max(1, groupCount) + targets - 1) / targets;
        return Math.max(cpuKernelConfig.minReductionChunkSize(), candidate);
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }

    private static Cpu1ScratchBufferSpec scratchBufferSpec(
            LossReduction reduction,
            int groupCount,
            Cpu1LaunchConfig launchConfig
    ) {
        if (reduction == LossReduction.NONE || launchConfig.workerCount() == 1) {
            return Cpu1ScratchBufferSpec.none();
        }
        int slots = Cpu1RangeLauncher.slotCount(groupCount, launchConfig);
        return Cpu1ScratchBufferSpec.arrays(0, slots, slots);
    }

    private static Cpu1ScratchBufferSpec nllScratchBufferSpec(
            int groupCount,
            Cpu1LaunchConfig launchConfig
    ) {
        if (groupCount == 0 || launchConfig.workerCount() == 1) {
            return Cpu1ScratchBufferSpec.none();
        }
        int slots = Cpu1RangeLauncher.slotCount(groupCount, launchConfig);
        return Cpu1ScratchBufferSpec.arrays(0, slots, 0);
    }

    private static Cpu1ScratchBufferSpec denseCrossEntropyScratchBufferSpec(
            int groupCount,
            Cpu1LaunchConfig launchConfig
    ) {
        if (launchConfig.workerCount() == 1) {
            return Cpu1ScratchBufferSpec.none();
        }
        int slots = Cpu1RangeLauncher.slotCount(groupCount, launchConfig);
        return Cpu1ScratchBufferSpec.arrays(0, slots, 0);
    }

    private static int normalizedAxis(String label, int axis, int rank) {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank) {
            throw new UnsupportedOperationException(label + " class axis out of bounds: axis=" + axis
                    + ", rank=" + rank);
        }
        return normalized;
    }

    private static long sampleCount(int[] shape, int classAxis) {
        long count = 1L;
        for (int dim = 0; dim < shape.length; dim++) {
            if (dim != classAxis) {
                count = Math.multiplyExact(count, shape[dim]);
            }
        }
        return count;
    }

    private static int[] reduceShape(int[] shape, int axis) {
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
}

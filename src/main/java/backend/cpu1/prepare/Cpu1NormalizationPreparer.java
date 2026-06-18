package backend.cpu1.prepare;

import backend.cpu1.kernels.nn.normalization.layernorm.Cpu1LayerNormKernelId;
import backend.cpu1.kernels.nn.normalization.rmsnorm.Cpu1RmsNormKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.normalization.layerNorm;
import operations.normalization.rmsNorm;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the dense normalization subset for cpu1.
 */
public final class Cpu1NormalizationPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        if (!isNormalizationOp(operation.opType())) {
            throw new UnsupportedOperationException("cpu1 normalization preparer does not support "
                    + operation.opType());
        }
        if (operation.opType() == Operation.OpType.RMS_NORM) {
            return prepareRmsNorm(node, descriptorIndex, config, operation);
        }
        return prepareLayerNorm(node, descriptorIndex, config, operation);
    }

    private Cpu1PreparedArtifact prepareLayerNorm(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof layerNorm norm)) {
            throw new IllegalArgumentException("cpu1 LAYER_NORM requires operations.normalization.layerNorm op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM requires descriptors.");
        }
        if (node.inputIds().size() != 3) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM expects 3 inputs, got "
                    + node.inputIds().size());
        }
        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor gamma = descriptorIndex.byNodeId(node.inputIds().get(1));
        CompiledTensorDescriptor beta = descriptorIndex.byNodeId(node.inputIds().get(2));
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan gammaAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(gamma);
        Cpu1StorageAccessPlan betaAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(beta);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);

        requireContract(node, input, gamma, beta, norm);
        requireDenseContiguousNoOffset("LAYER_NORM", "input", inputAccessPlan);
        requireDenseContiguousNoOffset("LAYER_NORM", "gamma", gammaAccessPlan);
        requireDenseContiguousNoOffset("LAYER_NORM", "beta", betaAccessPlan);
        requireDenseContiguousNoOffset("LAYER_NORM", "output", outputAccessPlan);

        int normalizedRank = norm.getNormalizedRank();
        int normalizedSize = normalizedSize("LAYER_NORM", input.shape(), normalizedRank);
        int outputElementCount = node.flatDataSize();
        if (outputElementCount % normalizedSize != 0) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM flatDataSize must be divisible by "
                    + "normalizedSize. elements=" + outputElementCount + ", normalizedSize=" + normalizedSize);
        }
        int groupCount = outputElementCount / normalizedSize;
        Cpu1LaunchConfig launchConfig = launchConfig("LayerNorm", groupCount, config);
        Cpu1PreparedLayerNormUnit unit = new Cpu1PreparedLayerNormUnit(
                node.id(),
                input.nodeId(),
                gamma.nodeId(),
                beta.nodeId(),
                operation.opType(),
                node.dataType(),
                config.storageKind(),
                kernelId(node.dataType(), config.storageKind()),
                input.shape(),
                normalizedRank,
                normalizedSize,
                groupCount,
                outputElementCount,
                norm.getEpsilon(),
                launchConfig,
                launchPolicy(launchConfig),
                inputAccessPlan,
                gammaAccessPlan,
                betaAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isNormalizationOp(Operation.OpType opType) {
        return opType == Operation.OpType.LAYER_NORM || opType == Operation.OpType.RMS_NORM;
    }

    private Cpu1PreparedArtifact prepareRmsNorm(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config,
            Operation operation
    ) {
        if (!(operation instanceof rmsNorm norm)) {
            throw new IllegalArgumentException("cpu1 RMS_NORM requires operations.normalization.rmsNorm op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM requires descriptors.");
        }
        if (node.inputIds().size() != 2) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM expects 2 inputs, got "
                    + node.inputIds().size());
        }
        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor gamma = descriptorIndex.byNodeId(node.inputIds().get(1));
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan gammaAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(gamma);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);

        requireContract(node, input, gamma, norm);
        requireDenseContiguousNoOffset("RMS_NORM", "input", inputAccessPlan);
        requireDenseContiguousNoOffset("RMS_NORM", "gamma", gammaAccessPlan);
        requireDenseContiguousNoOffset("RMS_NORM", "output", outputAccessPlan);

        int normalizedRank = norm.getNormalizedRank();
        int normalizedSize = normalizedSize("RMS_NORM", input.shape(), normalizedRank);
        int outputElementCount = node.flatDataSize();
        if (outputElementCount % normalizedSize != 0) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM flatDataSize must be divisible by "
                    + "normalizedSize. elements=" + outputElementCount + ", normalizedSize=" + normalizedSize);
        }
        int groupCount = outputElementCount / normalizedSize;
        Cpu1LaunchConfig launchConfig = launchConfig("RMSNorm", groupCount, config);
        Cpu1PreparedRmsNormUnit unit = new Cpu1PreparedRmsNormUnit(
                node.id(),
                input.nodeId(),
                gamma.nodeId(),
                operation.opType(),
                node.dataType(),
                config.storageKind(),
                rmsKernelId(node.dataType(), config.storageKind()),
                input.shape(),
                normalizedRank,
                normalizedSize,
                groupCount,
                outputElementCount,
                norm.getEpsilon(),
                launchConfig,
                launchPolicy(launchConfig),
                inputAccessPlan,
                gammaAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private static void requireContract(
            CompiledNode node,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor gamma,
            CompiledTensorDescriptor beta,
            layerNorm norm
    ) {
        if (!isSupportedDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM supports only FLOAT32/FLOAT64/BFLOAT16, got "
                    + node.dataType());
        }
        if (input.dataType() != node.dataType()
                || gamma.dataType() != node.dataType()
                || beta.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM requires matching input/gamma/beta/output dtype. "
                    + "input=" + input.dataType() + ", gamma=" + gamma.dataType()
                    + ", beta=" + beta.dataType() + ", output=" + node.dataType());
        }
        int normalizedRank = norm.getNormalizedRank();
        if (normalizedRank < 1 || normalizedRank > input.rank()) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM normalizedRank out of bounds: normalizedRank="
                    + normalizedRank + ", inputRank=" + input.rank());
        }
        if (!Arrays.equals(input.shape(), node.shape())) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM output shape must match input shape. input="
                    + Arrays.toString(input.shape()) + ", output=" + Arrays.toString(node.shape()));
        }
        if (gamma.rank() != normalizedRank || beta.rank() != normalizedRank) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM gamma/beta ranks must equal normalizedRank. "
                    + "normalizedRank=" + normalizedRank + ", gammaRank=" + gamma.rank()
                    + ", betaRank=" + beta.rank());
        }
        int[] inputShape = input.shape();
        int[] gammaShape = gamma.shape();
        int[] betaShape = beta.shape();
        int start = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            int expected = inputShape[start + i];
            if (gammaShape[i] != expected || betaShape[i] != expected) {
                throw new UnsupportedOperationException("cpu1 LAYER_NORM gamma/beta shapes must match trailing "
                        + "input dimensions. input=" + Arrays.toString(inputShape)
                        + ", gamma=" + Arrays.toString(gammaShape)
                        + ", beta=" + Arrays.toString(betaShape));
            }
        }
        int normalizedSize = normalizedSize("LAYER_NORM", inputShape, normalizedRank);
        if (gamma.logicalElementCount() != normalizedSize || beta.logicalElementCount() != normalizedSize) {
            throw new UnsupportedOperationException("cpu1 LAYER_NORM parameter element count mismatch. normalizedSize="
                    + normalizedSize + ", gammaElements=" + gamma.logicalElementCount()
                    + ", betaElements=" + beta.logicalElementCount());
        }
        Math.toIntExact(input.logicalElementCount());
    }

    private static void requireContract(
            CompiledNode node,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor gamma,
            rmsNorm norm
    ) {
        if (!isSupportedDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM supports only FLOAT32/FLOAT64/BFLOAT16, got "
                    + node.dataType());
        }
        if (input.dataType() != node.dataType() || gamma.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM requires matching input/gamma/output dtype. "
                    + "input=" + input.dataType() + ", gamma=" + gamma.dataType()
                    + ", output=" + node.dataType());
        }
        int normalizedRank = norm.getNormalizedRank();
        if (normalizedRank < 1 || normalizedRank > input.rank()) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM normalizedRank out of bounds: normalizedRank="
                    + normalizedRank + ", inputRank=" + input.rank());
        }
        if (!Arrays.equals(input.shape(), node.shape())) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM output shape must match input shape. input="
                    + Arrays.toString(input.shape()) + ", output=" + Arrays.toString(node.shape()));
        }
        if (gamma.rank() != normalizedRank) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM gamma rank must equal normalizedRank. "
                    + "normalizedRank=" + normalizedRank + ", gammaRank=" + gamma.rank());
        }
        int[] inputShape = input.shape();
        int[] gammaShape = gamma.shape();
        int start = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            int expected = inputShape[start + i];
            if (gammaShape[i] != expected) {
                throw new UnsupportedOperationException("cpu1 RMS_NORM gamma shape must match trailing "
                        + "input dimensions. input=" + Arrays.toString(inputShape)
                        + ", gamma=" + Arrays.toString(gammaShape));
            }
        }
        int normalizedSize = normalizedSize("RMS_NORM", inputShape, normalizedRank);
        if (gamma.logicalElementCount() != normalizedSize) {
            throw new UnsupportedOperationException("cpu1 RMS_NORM gamma element count mismatch. normalizedSize="
                    + normalizedSize + ", gammaElements=" + gamma.logicalElementCount());
        }
        Math.toIntExact(input.logicalElementCount());
    }

    private static boolean isSupportedDType(DataType dataType) {
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
        throw new UnsupportedOperationException("cpu1 " + opName + " dense first slice supports only dense contiguous "
                + "no-offset " + role + " access; actual=" + accessPlan.kind() + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static int normalizedSize(String opName, int[] inputShape, int normalizedRank) {
        int start = inputShape.length - normalizedRank;
        int product = 1;
        for (int i = start; i < inputShape.length; i++) {
            product = Math.multiplyExact(product, inputShape[i]);
        }
        if (product <= 0) {
            throw new UnsupportedOperationException("cpu1 " + opName + " requires positive normalizedSize, got "
                    + product);
        }
        return product;
    }

    private static Cpu1LayerNormKernelId kernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1LayerNormKernelId.LAYER_NORM_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1LayerNormKernelId.LAYER_NORM_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1LayerNormKernelId.LAYER_NORM_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1LayerNormKernelId.LAYER_NORM_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1LayerNormKernelId.LAYER_NORM_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1LayerNormKernelId.LAYER_NORM_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 LAYER_NORM supports only floating dtypes.");
        };
    }

    private static Cpu1RmsNormKernelId rmsKernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1RmsNormKernelId.RMS_NORM_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1RmsNormKernelId.RMS_NORM_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1RmsNormKernelId.RMS_NORM_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1RmsNormKernelId.RMS_NORM_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1RmsNormKernelId.RMS_NORM_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1RmsNormKernelId.RMS_NORM_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 RMS_NORM supports only floating dtypes.");
        };
    }

    private static Cpu1LaunchConfig launchConfig(String opName, int groupCount, Cpu1PrepareConfig config) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 " + opName + " dispatch requires CpuKernelConfig.");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1 || groupCount <= 1 || groupCount < cpuKernelConfig.reductionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, groupCount);
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
}

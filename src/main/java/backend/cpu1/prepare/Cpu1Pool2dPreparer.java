package backend.cpu1.prepare;

import backend.cpu1.kernels.nn.pool.avgpool.Cpu1AvgPool2dKernelId;
import backend.cpu1.kernels.nn.pool.maxpool.Cpu1MaxPool2dKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import config.backend.CpuKernelConfig;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptor;
import planning.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.DataType;
import tensor.options.Pool2dOptions;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the dense pool2d subset for cpu1.
 */
public final class Cpu1Pool2dPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        Operation.OpType opType = operation.opType();
        if (!isPool2dOp(opType)) {
            throw new UnsupportedOperationException("cpu1 pool2d preparer does not support " + opType);
        }
        Pool2dOptions options = pool2dOptions(operation);
        if (options == null) {
            throw new IllegalArgumentException("cpu1 pool2d requires operations.nn.pool maxPool2d/avgPool2d op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 " + opType + " requires descriptors.");
        }
        if (node.inputIds().size() != 1) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects 1 input, got "
                    + node.inputIds().size());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY
                && config.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            throw new UnsupportedOperationException("cpu1 " + opType + " supports only JAVA_ARRAY/MEMORY_SEGMENT storage, got "
                    + config.storageKind());
        }

        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().getFirst());
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);
        requireContract(node, input, options);
        requireDenseContiguousNoOffset(opType, "input", inputAccessPlan);
        requireDenseContiguousNoOffset(opType, "output", outputAccessPlan);

        Cpu1LaunchConfig launchConfig = launchConfig(node.flatDataSize(), config);
        if (opType == Operation.OpType.MAX_POOL2D) {
            Cpu1PreparedMaxPool2dUnit unit = new Cpu1PreparedMaxPool2dUnit(
                    node.id(),
                    input.nodeId(),
                    operation.opType(),
                    node.dataType(),
                    config.storageKind(),
                    maxPoolKernelId(node.dataType(), config.storageKind()),
                    input.shape(),
                    node.shape(),
                    options,
                    node.flatDataSize(),
                    launchConfig,
                    launchPolicy(launchConfig),
                    inputAccessPlan,
                    outputAccessPlan
            );
            return new Cpu1PreparedArtifact(unit);
        }
        Cpu1PreparedAvgPool2dUnit unit = new Cpu1PreparedAvgPool2dUnit(
                node.id(),
                input.nodeId(),
                operation.opType(),
                node.dataType(),
                config.storageKind(),
                avgPoolKernelId(node.dataType(), config.storageKind()),
                input.shape(),
                node.shape(),
                options,
                node.flatDataSize(),
                launchConfig,
                launchPolicy(launchConfig),
                inputAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isPool2dOp(Operation.OpType opType) {
        return opType == Operation.OpType.MAX_POOL2D
                || opType == Operation.OpType.AVG_POOL2D;
    }

    private static Pool2dOptions pool2dOptions(Operation operation) {
        if (operation instanceof maxPool2d pool) {
            return pool.getOptions();
        }
        if (operation instanceof avgPool2d pool) {
            return pool.getOptions();
        }
        return null;
    }

    private static void requireContract(
            CompiledNode node,
            CompiledTensorDescriptor input,
            Pool2dOptions options
    ) {
        if (!isSupportedDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 " + node.operation().opType()
                    + " supports only FLOAT32/FLOAT64/BFLOAT16, got "
                    + node.dataType());
        }
        if (input.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 " + node.operation().opType()
                    + " requires matching input/output dtype. input="
                    + input.dataType() + ", output=" + node.dataType());
        }
        int[] inputShape = input.shape();
        int[] outputShape = node.shape();
        if (inputShape.length != 4 || outputShape.length != 4) {
            throw new UnsupportedOperationException("cpu1 " + node.operation().opType()
                    + " requires rank-4 NCHW tensors. input="
                    + Arrays.toString(inputShape) + ", output=" + Arrays.toString(outputShape));
        }
        int expectedOutH = inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), options.ceilMode());
        int expectedOutW = inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), options.ceilMode());
        int[] expectedOutputShape = new int[]{inputShape[0], inputShape[1], expectedOutH, expectedOutW};
        if (!Arrays.equals(expectedOutputShape, outputShape)) {
            throw new UnsupportedOperationException("cpu1 " + node.operation().opType()
                    + " output shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(outputShape));
        }
        validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), expectedOutH, "height");
        validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), expectedOutW, "width");
        Math.toIntExact(input.logicalElementCount());
    }

    private static boolean isSupportedDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16;
    }

    private static void requireDenseContiguousNoOffset(
            Operation.OpType opType,
            String role,
            Cpu1StorageAccessPlan accessPlan
    ) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 " + opType
                + " dense direct route supports only dense contiguous "
                + "no-offset " + role + " access; actual=" + accessPlan.kind() + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static int inferOutputSize(int input, int kernel, int pad, int stride, boolean ceilMode) {
        int numerator = input + 2 * pad - kernel;
        if (numerator < 0) {
            throw new UnsupportedOperationException("cpu1 pool2d kernel does not fit input.");
        }
        return (ceilMode ? (numerator + stride - 1) / stride : numerator / stride) + 1;
    }

    private static void validateWindowCoverage(
            int inputSize,
            int kernel,
            int pad,
            int stride,
            int outputSize,
            String axisName
    ) {
        for (int outIndex = 0; outIndex < outputSize; outIndex++) {
            int start = outIndex * stride - pad;
            int end = start + kernel;
            if (Math.max(start, 0) >= Math.min(end, inputSize)) {
                throw new UnsupportedOperationException("cpu1 pool2d configuration creates an all-padding window on "
                        + axisName + " axis at output index " + outIndex + ".");
            }
        }
    }

    private static Cpu1MaxPool2dKernelId maxPoolKernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1MaxPool2dKernelId.MAX_POOL2D_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1MaxPool2dKernelId.MAX_POOL2D_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1MaxPool2dKernelId.MAX_POOL2D_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1MaxPool2dKernelId.MAX_POOL2D_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1MaxPool2dKernelId.MAX_POOL2D_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1MaxPool2dKernelId.MAX_POOL2D_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 MAX_POOL2D supports only floating dtypes.");
        };
    }

    private static Cpu1AvgPool2dKernelId avgPoolKernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AvgPool2dKernelId.AVG_POOL2D_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1AvgPool2dKernelId.AVG_POOL2D_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AvgPool2dKernelId.AVG_POOL2D_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1AvgPool2dKernelId.AVG_POOL2D_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1AvgPool2dKernelId.AVG_POOL2D_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1AvgPool2dKernelId.AVG_POOL2D_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 AVG_POOL2D supports only floating dtypes.");
        };
    }

    private static Cpu1LaunchConfig launchConfig(int outputElementCount, Cpu1PrepareConfig config) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 pool2d dispatch requires CpuKernelConfig.");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1
                || outputElementCount <= 1
                || outputElementCount < cpuKernelConfig.reductionParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, outputElementCount);
        return Cpu1LaunchConfig.parallel(
                plannedWorkers,
                chunkSize(outputElementCount, plannedWorkers, cpuKernelConfig)
        );
    }

    private static int chunkSize(
            int outputElementCount,
            int plannedWorkers,
            CpuKernelConfig cpuKernelConfig
    ) {
        int targets = Math.max(1, plannedWorkers * cpuKernelConfig.highCostTargetChunksPerWorker());
        int candidate = (Math.max(1, outputElementCount) + targets - 1) / targets;
        return Math.max(cpuKernelConfig.minReductionChunkSize(), candidate);
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }
}

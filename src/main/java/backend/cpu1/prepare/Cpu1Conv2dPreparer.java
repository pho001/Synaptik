package backend.cpu1.prepare;

import backend.cpu1.kernels.nn.conv.conv2d.Cpu1Conv2dKernelId;
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
import operations.nn.conv.conv2d;
import tensor.DataType;
import tensor.options.Conv2dOptions;

import java.util.Arrays;
import java.util.Objects;

/**
 * Prepares the dense direct CONV2D subset for cpu1.
 */
public final class Cpu1Conv2dPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        if (!isConv2dOp(operation.opType())) {
            throw new UnsupportedOperationException("cpu1 conv2d preparer does not support " + operation.opType());
        }
        if (!(operation instanceof conv2d conv)) {
            throw new IllegalArgumentException("cpu1 CONV2D requires operations.nn.conv.conv2d op.");
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 CONV2D requires descriptors.");
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY
                && config.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            throw new UnsupportedOperationException("cpu1 CONV2D supports only JAVA_ARRAY/MEMORY_SEGMENT storage, got "
                    + config.storageKind());
        }
        int expectedInputs = conv.hasBias() ? 3 : 2;
        if (node.inputIds().size() != expectedInputs) {
            throw new UnsupportedOperationException("cpu1 CONV2D expects " + expectedInputs + " inputs, got "
                    + node.inputIds().size());
        }

        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor weight = descriptorIndex.byNodeId(node.inputIds().get(1));
        CompiledTensorDescriptor bias = conv.hasBias() ? descriptorIndex.byNodeId(node.inputIds().get(2)) : null;
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan weightAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(weight);
        Cpu1StorageAccessPlan biasAccessPlan = bias == null ? null : Cpu1StorageAccessPlan.fromDescriptor(bias);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);
        requireContract(node, conv, input, weight, bias);
        requireDenseContiguousNoOffset("input", inputAccessPlan);
        requireDenseContiguousNoOffset("weight", weightAccessPlan);
        if (biasAccessPlan != null) {
            requireDenseContiguousNoOffset("bias", biasAccessPlan);
        }
        requireDenseContiguousNoOffset("output", outputAccessPlan);

        long work = estimateWork(node.flatDataSize(), weight.shape());
        Cpu1LaunchConfig launchConfig = launchConfig(node.flatDataSize(), work, config);
        Cpu1PreparedConv2dUnit unit = new Cpu1PreparedConv2dUnit(
                node.id(),
                input.nodeId(),
                weight.nodeId(),
                bias == null ? -1 : bias.nodeId(),
                operation.opType(),
                node.dataType(),
                config.storageKind(),
                kernelId(node.dataType(), config.storageKind()),
                input.shape(),
                weight.shape(),
                bias == null ? null : bias.shape(),
                node.shape(),
                conv.hasBias(),
                conv.getOptions(),
                work,
                node.flatDataSize(),
                launchConfig,
                launchPolicy(launchConfig),
                inputAccessPlan,
                weightAccessPlan,
                biasAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isConv2dOp(Operation.OpType opType) {
        return opType == Operation.OpType.CONV2D;
    }

    private static void requireContract(
            CompiledNode node,
            conv2d conv,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor weight,
            CompiledTensorDescriptor bias
    ) {
        DataType outputDataType = node.dataType();
        if (!isSupportedDType(outputDataType)) {
            throw new UnsupportedOperationException("cpu1 CONV2D supports only FLOAT32/FLOAT64/BFLOAT16, got "
                    + outputDataType);
        }
        if (input.dataType() != outputDataType || weight.dataType() != outputDataType
                || (bias != null && bias.dataType() != outputDataType)) {
            throw new UnsupportedOperationException("cpu1 CONV2D dense direct route requires matching "
                    + "input/weight/bias/output dtype. output=" + outputDataType + ", input="
                    + input.dataType() + ", weight=" + weight.dataType()
                    + ", bias=" + (bias == null ? "<none>" : bias.dataType()));
        }
        int[] inputShape = input.shape();
        int[] weightShape = weight.shape();
        int[] outputShape = node.shape();
        if (inputShape.length != 4 || weightShape.length != 4 || outputShape.length != 4) {
            throw new UnsupportedOperationException("cpu1 CONV2D requires rank-4 NCHW/OIHW tensors. input="
                    + Arrays.toString(inputShape) + ", weight=" + Arrays.toString(weightShape)
                    + ", output=" + Arrays.toString(outputShape));
        }
        Conv2dOptions options = conv.getOptions();
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        if (inChannels % options.groups() != 0) {
            throw new UnsupportedOperationException("cpu1 CONV2D input channels must be divisible by groups.");
        }
        if (outChannels % options.groups() != 0) {
            throw new UnsupportedOperationException("cpu1 CONV2D output channels must be divisible by groups.");
        }
        if (channelsPerGroup * options.groups() != inChannels) {
            throw new UnsupportedOperationException("cpu1 CONV2D weight shape is incompatible with input channels "
                    + "and groups. input=" + Arrays.toString(inputShape)
                    + ", weight=" + Arrays.toString(weightShape)
                    + ", groups=" + options.groups());
        }
        if (bias != null) {
            int[] biasShape = bias.shape();
            if (biasShape.length != 1 || biasShape[0] != outChannels) {
                throw new UnsupportedOperationException("cpu1 CONV2D bias must have shape [outChannels]. bias="
                        + Arrays.toString(biasShape) + ", outChannels=" + outChannels);
            }
        }
        int expectedOutH = inferOutputSize(inH, kernelH, options.padH(), options.strideH(), options.dilationH(), "height");
        int expectedOutW = inferOutputSize(inW, kernelW, options.padW(), options.strideW(), options.dilationW(), "width");
        int[] expectedOutputShape = new int[]{batch, outChannels, expectedOutH, expectedOutW};
        if (!Arrays.equals(expectedOutputShape, outputShape)) {
            throw new UnsupportedOperationException("cpu1 CONV2D output shape mismatch. expected="
                    + Arrays.toString(expectedOutputShape) + ", actual=" + Arrays.toString(outputShape));
        }
        Math.toIntExact(input.logicalElementCount());
        Math.toIntExact(weight.logicalElementCount());
        if (bias != null) {
            Math.toIntExact(bias.logicalElementCount());
        }
    }

    private static boolean isSupportedDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16;
    }

    private static void requireDenseContiguousNoOffset(
            String role,
            Cpu1StorageAccessPlan accessPlan
    ) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 CONV2D dense direct route supports only dense contiguous "
                + "no-offset " + role + " access; actual=" + accessPlan.kind() + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static int inferOutputSize(
            int inputSize,
            int kernelSize,
            int pad,
            int stride,
            int dilation,
            String axisName
    ) {
        int effectiveKernel = dilation * (kernelSize - 1) + 1;
        int numerator = inputSize + 2 * pad - effectiveKernel;
        if (numerator < 0) {
            throw new UnsupportedOperationException("cpu1 CONV2D effective kernel does not fit input " + axisName + ".");
        }
        return numerator / stride + 1;
    }

    private static Cpu1Conv2dKernelId kernelId(DataType dataType, Cpu1StorageKind storageKind) {
        return switch (dataType) {
            case FLOAT32 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1Conv2dKernelId.CONV2D_F32_SEGMENT_DENSE_SCALAR
                    : Cpu1Conv2dKernelId.CONV2D_F32_ARRAY_DENSE_SCALAR;
            case FLOAT64 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1Conv2dKernelId.CONV2D_F64_SEGMENT_DENSE_SCALAR
                    : Cpu1Conv2dKernelId.CONV2D_F64_ARRAY_DENSE_SCALAR;
            case BFLOAT16 -> storageKind == Cpu1StorageKind.MEMORY_SEGMENT
                    ? Cpu1Conv2dKernelId.CONV2D_BF16_SEGMENT_DENSE_SCALAR
                    : Cpu1Conv2dKernelId.CONV2D_BF16_ARRAY_DENSE_SCALAR;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException(
                    "cpu1 CONV2D supports only floating dtypes.");
        };
    }

    private static long estimateWork(int outputElementCount, int[] weightShape) {
        return (long) outputElementCount
                * weightShape[1]
                * weightShape[2]
                * weightShape[3];
    }

    private static Cpu1LaunchConfig launchConfig(
            int outputElementCount,
            long work,
            Cpu1PrepareConfig config
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 CONV2D dispatch requires CpuKernelConfig.");
        }
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1
                || outputElementCount <= 1
                || work < cpuKernelConfig.matMulParallelMinSize()) {
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
        int evenChunk = Math.max(1, (outputElementCount + plannedWorkers - 1) / plannedWorkers);
        return Math.max(cpuKernelConfig.minScalarChunkSize(), evenChunk);
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }
}

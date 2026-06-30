package backend.cpu1.prepare;

import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.dtype.Cpu1DTypeKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.dtype.cast;
import tensor.DataType;

import java.util.Arrays;

/**
 * Prepares cpu1 dtype conversion nodes outside layout and elementwise dispatch.
 */
public final class Cpu1DTypePreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        if (node == null) {
            throw new IllegalArgumentException("node cannot be null");
        }
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        Operation operation = node.operation();
        if (operation == null) {
            throw new IllegalArgumentException("node operation cannot be null");
        }
        Operation.OpType opType = operation.opType();
        if (!isDTypeOp(opType)) {
            throw new UnsupportedOperationException("cpu1 dtype preparer does not support " + opType);
        }
        if (node.inputIds().size() != 1) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects 1 input, got "
                    + node.inputIds().size());
        }
        if (!(operation instanceof cast castOp)) {
            throw new IllegalArgumentException("cpu1 CAST operation must be operations.dtype.cast.");
        }
        if (castOp.getTargetType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 CAST target dtype " + castOp.getTargetType()
                    + " does not match node dtype " + node.dataType());
        }
        if (descriptorIndex == null) {
            throw new UnsupportedOperationException("cpu1 CAST requires descriptors to resolve input dtype.");
        }
        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().getFirst());
        requireInputContract(node, input);
        if (!isSupportedDType(input.dataType()) || !isSupportedDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 CAST does not support input dtype " + input.dataType()
                    + " and output dtype " + node.dataType());
        }
        Cpu1DTypeKernelId kernelId = kernelId(opType, config.storageKind());
        Cpu1PreparedDTypeUnit unit = new Cpu1PreparedDTypeUnit(
                node.id(),
                input.nodeId(),
                opType,
                input.dataType(),
                node.dataType(),
                node.flatDataSize(),
                layoutKind(input, node),
                config.storageKind(),
                kernelId,
                config.launchConfig(),
                launchPolicy(config.launchConfig())
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isDTypeOp(Operation.OpType opType) {
        return opType == Operation.OpType.CAST;
    }

    private static void requireInputContract(CompiledNode node, CompiledTensorDescriptor input) {
        if (input.logicalElementCount() != node.flatDataSize()) {
            throw new UnsupportedOperationException("cpu1 CAST requires matching element count. input="
                    + input.logicalElementCount() + ", output=" + node.flatDataSize());
        }
        if (!Arrays.equals(input.shape(), node.shape())) {
            throw new UnsupportedOperationException("cpu1 CAST requires matching input/output shape. input="
                    + Arrays.toString(input.shape()) + ", output=" + Arrays.toString(node.shape()));
        }
    }

    private static boolean isSupportedDType(DataType dataType) {
        return dataType == DataType.FLOAT64
                || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16
                || dataType == DataType.INT32
                || dataType == DataType.INT64
                || dataType == DataType.BOOL;
    }

    private static Cpu1DTypeKernelId kernelId(Operation.OpType opType, Cpu1StorageKind storageKind) {
        if (opType != Operation.OpType.CAST) {
            throw new UnsupportedOperationException("cpu1 dtype preparer does not support " + opType);
        }
        return switch (storageKind) {
            case JAVA_ARRAY -> Cpu1DTypeKernelId.CAST_ARRAY_SCALAR;
            case MEMORY_SEGMENT -> Cpu1DTypeKernelId.CAST_SEGMENT_SCALAR;
        };
    }

    private static Cpu1LayoutKind layoutKind(CompiledTensorDescriptor input, CompiledNode node) {
        if (input.contiguous() && node.contiguous()) {
            return Cpu1LayoutKind.CONTIGUOUS;
        }
        return switch (node.shape().length) {
            case 2 -> Cpu1LayoutKind.STRIDED_RANK2;
            case 3 -> Cpu1LayoutKind.STRIDED_RANK3;
            case 4 -> Cpu1LayoutKind.STRIDED_RANK4;
            default -> Cpu1LayoutKind.STRIDED_GENERIC;
        };
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }
}

package backend.cpu1.prepare;

import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelId;
import backend.cpu1.kernels.Cpu1KernelRegistry;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.plan.Cpu1IterationPlan;
import backend.cpu1.prepare.dispatch.Cpu1DispatchDecision;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import tensor.DataType;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Prepares one compiled node for initial cpu1 execution.
 */
public final class Cpu1NodePreparer {
    private final Cpu1KernelRegistry kernelRegistry;
    private final Cpu1DispatchPolicy dispatchPolicy;

    public Cpu1NodePreparer() {
        this(new Cpu1KernelRegistry());
    }

    public Cpu1NodePreparer(Cpu1KernelRegistry kernelRegistry) {
        this.kernelRegistry = Objects.requireNonNull(kernelRegistry, "kernelRegistry cannot be null");
        this.dispatchPolicy = new Cpu1DispatchPolicy();
    }

    public Cpu1PreparedArtifact prepare(CompiledNode node) {
        return prepare(node, null, Cpu1PrepareConfig.scalarSingleThread());
    }

    public Cpu1PreparedArtifact prepare(CompiledNode node, CompiledTensorDescriptorIndex descriptorIndex) {
        return prepare(node, descriptorIndex, Cpu1PrepareConfig.scalarSingleThread());
    }

    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        Operation.OpType opType = operation.opType();
        if (Cpu1LayoutPreparer.isLayoutOp(opType)) {
            return new Cpu1LayoutPreparer().prepare(node, descriptorIndex, config);
        }
        if (Cpu1ReductionPreparer.isReductionOp(opType)) {
            return new Cpu1ReductionPreparer().prepare(node, descriptorIndex, config);
        }
        if (Cpu1MatmulPreparer.isMatmulOp(opType)) {
            return new Cpu1MatmulPreparer().prepare(node, descriptorIndex, config);
        }
        List<DataType> inputDataTypes = inputDataTypes(opType, node, descriptorIndex);
        requireSupported(opType, node, descriptorIndex, inputDataTypes);
        DataType kernelDataType = kernelDataType(opType, node.dataType(), inputDataTypes);
        Cpu1DispatchDecision dispatchDecision = dispatchPolicy.decideElementwise(
                opType,
                kernelDataType,
                node.flatDataSize(),
                config
        );
        Operation.OpType kernelOpType = dispatchDecision.kernelOpType();
        ScalarParameter scalarParameter = scalarParameter(operation);
        Cpu1LayoutKind layoutKind = layoutKind(
                node,
                descriptorIndex,
                dispatchDecision
        );
        Cpu1StorageKind storageKind = dispatchDecision.storageKind();
        Cpu1ElementwiseKernelId kernelId = kernelRegistry.resolve(
                kernelOpType,
                kernelDataType,
                inputDataTypes,
                layoutKind,
                storageKind,
                dispatchPolicy.kernelVectorizationKind(dispatchDecision, layoutKind)
        );
        Cpu1PreparedElementwiseUnit unit = new Cpu1PreparedElementwiseUnit(
                node.id(),
                node.inputIds(),
                node.id(),
                opType,
                node.dataType(),
                Cpu1IterationPlan.contiguous(node.flatDataSize(), node.shape()),
                layoutKind,
                storageKind,
                kernelId,
                launchPolicy(dispatchDecision.launchConfig()),
                scalarParameter.present(),
                scalarParameter.f32(),
                scalarParameter.f64(),
                inputDataTypes,
                dispatchDecision
        );
        return new Cpu1PreparedArtifact(unit);
    }

    private static Cpu1LaunchPolicy launchPolicy(Cpu1LaunchConfig launchConfig) {
        if (launchConfig.workerCount() == 1) {
            return new Cpu1SingleThreadLaunch(launchConfig);
        }
        return new Cpu1ParallelLaunch(launchConfig);
    }

    private static void requireSupported(
            Operation.OpType opType,
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            List<DataType> inputDataTypes
    ) {
        if (!isSupportedOutputDType(opType, node.dataType())) {
            throw new UnsupportedOperationException("cpu1 initial preparer does not support output dtype "
                    + node.dataType() + " for " + opType);
        }
        for (int i = 0; i < inputDataTypes.size(); i++) {
            DataType inputDataType = inputDataTypes.get(i);
            if (!isSupportedInputDType(opType, i, inputDataType)) {
                throw new UnsupportedOperationException("cpu1 initial preparer does not support input " + i
                        + " dtype " + inputDataType + " for " + opType);
            }
        }
        int expectedInputs = expectedInputCount(opType);
        if (node.inputIds().size() != expectedInputs) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects " + expectedInputs
                    + " inputs, got " + node.inputIds().size());
        }
        requireInputDTypeContract(opType, node.dataType(), inputDataTypes);
        if (descriptorIndex != null) {
            requireSupportedInputs(opType, node, descriptorIndex, inputDataTypes);
        }
    }

    private static void requireInputDTypeContract(
            Operation.OpType opType,
            DataType outputDataType,
            List<DataType> inputDataTypes
    ) {
        switch (opType) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> {
                for (DataType inputDataType : inputDataTypes) {
                    if (inputDataType != DataType.BOOL) {
                        throw new UnsupportedOperationException("cpu1 " + opType
                                + " requires BOOL inputs, got " + inputDataTypes);
                    }
                }
            }
            case WHERE -> {
                if (inputDataTypes.getFirst() != DataType.BOOL) {
                    throw new UnsupportedOperationException("cpu1 WHERE requires BOOL condition, got "
                            + inputDataTypes.getFirst());
                }
                DataType promoted = promoteFloating(inputDataTypes.get(1), inputDataTypes.get(2));
                if (promoted != outputDataType) {
                    throw new UnsupportedOperationException("cpu1 WHERE output dtype " + outputDataType
                            + " does not match promoted branch dtype " + promoted + " for inputs " + inputDataTypes);
                }
            }
            case GT, GE, LT, LE, EQ, NE -> {
                DataType compareDataType = inputDataTypes.getFirst();
                for (DataType inputDataType : inputDataTypes) {
                    if (inputDataType != compareDataType) {
                        throw new UnsupportedOperationException("cpu1 " + opType
                                + " requires matching compare input dtypes, got " + inputDataTypes);
                    }
                }
            }
            default -> {
                for (DataType inputDataType : inputDataTypes) {
                    if (inputDataType != outputDataType) {
                        throw new UnsupportedOperationException("cpu1 " + opType
                                + " requires input dtype " + outputDataType + ", got " + inputDataType);
                    }
                }
            }
        }
    }

    private static void requireSupportedInputs(
            Operation.OpType opType,
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            List<DataType> inputDataTypes
    ) {
        for (int i = 0; i < node.inputIds().size(); i++) {
            int inputNodeId = node.inputIds().get(i);
            CompiledTensorDescriptor input = descriptorIndex.byNodeId(inputNodeId);
            DataType expectedInputDataType = inputDataTypes.get(i);
            if (input.dataType() != expectedInputDataType) {
                throw new UnsupportedOperationException("cpu1 initial preparer requires matching input dtype for nodeId="
                        + node.id() + ", inputNodeId=" + inputNodeId + ", expected " + expectedInputDataType
                        + ", got " + input.dataType());
            }
            if (!isBroadcastCompatible(input.shape(), node.shape())) {
                throw new UnsupportedOperationException("cpu1 initial preparer input shape "
                        + Arrays.toString(input.shape()) + " is not broadcast-compatible with output shape "
                        + Arrays.toString(node.shape()));
            }
        }
    }

    private static boolean isBroadcastCompatible(int[] inputShape, int[] outputShape) {
        if (inputShape.length > outputShape.length) {
            return false;
        }
        int offset = outputShape.length - inputShape.length;
        for (int dim = 0; dim < outputShape.length; dim++) {
            int inputDim = dim < offset ? 1 : inputShape[dim - offset];
            int outputDim = outputShape[dim];
            if (inputDim != outputDim && inputDim != 1) {
                return false;
            }
        }
        return true;
    }

    private static int expectedInputCount(Operation.OpType opType) {
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX, POW_TENSOR, GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR -> 2;
            case WHERE -> 3;
            case RELU, NEG, ABS, INV, EXP, FAST_EXP, ERF, LOG, TANH, FAST_TANH, SIGMOID, SQRT, FLOOR, CEIL, SIGN,
                    POW, MUL_SCALAR, CLAMP_MIN, CLAMP_MAX, LOGICAL_NOT -> 1;
            default -> throw new UnsupportedOperationException("cpu1 initial preparer does not support " + opType);
        };
    }

    private static List<DataType> inputDataTypes(
            Operation.OpType opType,
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (descriptorIndex == null) {
            return inferredInputDataTypesWithoutDescriptors(opType, node);
        }
        List<DataType> inputDataTypes = new ArrayList<>(node.inputIds().size());
        for (int inputNodeId : node.inputIds()) {
            inputDataTypes.add(descriptorIndex.byNodeId(inputNodeId).dataType());
        }
        return List.copyOf(inputDataTypes);
    }

    private static List<DataType> inferredInputDataTypesWithoutDescriptors(Operation.OpType opType, CompiledNode node) {
        return switch (opType) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> repeatedInputDataTypes(node, DataType.BOOL);
            case WHERE -> List.of(DataType.BOOL, node.dataType(), node.dataType());
            default -> repeatedInputDataTypes(node, node.dataType());
        };
    }

    private static List<DataType> repeatedInputDataTypes(CompiledNode node, DataType dataType) {
        List<DataType> inputDataTypes = new ArrayList<>(node.inputIds().size());
        for (int ignored : node.inputIds()) {
            inputDataTypes.add(dataType);
        }
        return List.copyOf(inputDataTypes);
    }

    private static DataType kernelDataType(Operation.OpType opType, DataType outputDataType, List<DataType> inputDataTypes) {
        return switch (opType) {
            case GT, GE, LT, LE, EQ, NE -> inputDataTypes.getFirst();
            default -> outputDataType;
        };
    }

    private static DataType promoteFloating(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static boolean isSupportedOutputDType(Operation.OpType opType, DataType outputDataType) {
        return switch (opType) {
            case GT, GE, LT, LE, EQ, NE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> outputDataType == DataType.BOOL;
            default -> outputDataType == DataType.FLOAT32
                    || outputDataType == DataType.FLOAT64
                    || outputDataType == DataType.BFLOAT16;
        };
    }

    private static boolean isSupportedInputDType(Operation.OpType opType, int inputIndex, DataType inputDataType) {
        return switch (opType) {
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> inputDataType == DataType.BOOL;
            case WHERE -> inputIndex == 0
                    ? inputDataType == DataType.BOOL
                    : inputDataType == DataType.FLOAT32
                            || inputDataType == DataType.FLOAT64
                            || inputDataType == DataType.BFLOAT16;
            default -> inputDataType == DataType.FLOAT32
                    || inputDataType == DataType.FLOAT64
                    || inputDataType == DataType.BFLOAT16;
        };
    }

    private static ScalarParameter scalarParameter(Operation operation) {
        return switch (operation.opType()) {
            case POW -> {
                if (operation instanceof pow op) {
                    yield ScalarParameter.of(op.getExponentF32(), op.getExponent());
                }
                throw new IllegalArgumentException("cpu1 POW operation must be operations.elementwise.unary.pow.");
            }
            case MUL_SCALAR -> {
                if (operation instanceof mulScalar op) {
                    yield ScalarParameter.of(op.getScalarF32(), op.getScalar());
                }
                throw new IllegalArgumentException("cpu1 MUL_SCALAR operation must be operations.elementwise.unary.mulScalar.");
            }
            case CLAMP_MIN -> {
                if (operation instanceof clampMin op) {
                    yield ScalarParameter.of(op.getMinValueF32(), op.getMinValue());
                }
                throw new IllegalArgumentException("cpu1 CLAMP_MIN operation must be operations.elementwise.unary.clampMin.");
            }
            case CLAMP_MAX -> {
                if (operation instanceof clampMax op) {
                    yield ScalarParameter.of(op.getMaxValueF32(), op.getMaxValue());
                }
                throw new IllegalArgumentException("cpu1 CLAMP_MAX operation must be operations.elementwise.unary.clampMax.");
            }
            default -> ScalarParameter.NONE;
        };
    }

    private Cpu1LayoutKind layoutKind(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1DispatchDecision dispatchDecision
    ) {
        if (canUseBroadcastInnerVectorLayout(node, descriptorIndex, dispatchDecision)) {
            return Cpu1LayoutKind.BROADCAST_INNER;
        }
        boolean strided = !node.contiguous();
        int stridedRank = strided ? node.shape().length : -1;
        if (descriptorIndex != null) {
            for (int inputNodeId : node.inputIds()) {
                CompiledTensorDescriptor descriptor = descriptorIndex.byNodeId(inputNodeId);
                if (requiresStridedInputPath(descriptor, node)) {
                    strided = true;
                    stridedRank = node.shape().length;
                }
            }
        }
        if (!strided) {
            return Cpu1LayoutKind.CONTIGUOUS;
        }
        return switch (stridedRank) {
            case 2 -> Cpu1LayoutKind.STRIDED_RANK2;
            case 3 -> Cpu1LayoutKind.STRIDED_RANK3;
            case 4 -> Cpu1LayoutKind.STRIDED_RANK4;
            default -> Cpu1LayoutKind.STRIDED_GENERIC;
        };
    }

    private static boolean requiresStridedInputPath(CompiledTensorDescriptor descriptor, CompiledNode node) {
        return !descriptor.contiguous()
                || descriptor.hasZeroStride()
                || descriptor.rank() != node.shape().length
                || !Arrays.equals(descriptor.shape(), node.shape())
                || descriptor.logicalElementCount() != node.flatDataSize();
    }

    private boolean canUseBroadcastInnerVectorLayout(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1DispatchDecision dispatchDecision
    ) {
        if (descriptorIndex == null
                || !node.contiguous()
                || !dispatchPolicy.canUseBroadcastInnerVectorLayout(dispatchDecision)) {
            return false;
        }
        boolean hasBroadcastInput = false;
        for (int inputNodeId : node.inputIds()) {
            CompiledTensorDescriptor descriptor = descriptorIndex.byNodeId(inputNodeId);
            if (isFullContiguousInput(descriptor, node)) {
                continue;
            }
            if (isScalarBroadcastInput(descriptor, node) || isInnerBroadcastInput(descriptor, node)) {
                hasBroadcastInput = true;
                continue;
            }
            return false;
        }
        return hasBroadcastInput;
    }

    private static boolean isFullContiguousInput(CompiledTensorDescriptor descriptor, CompiledNode node) {
        return descriptor.contiguous()
                && descriptor.rank() == node.shape().length
                && Arrays.equals(descriptor.shape(), node.shape())
                && descriptor.logicalElementCount() == node.flatDataSize();
    }

    private static boolean isScalarBroadcastInput(CompiledTensorDescriptor descriptor, CompiledNode node) {
        return descriptor.logicalElementCount() == 1
                && isBroadcastCompatible(descriptor.shape(), node.shape());
    }

    private static boolean isInnerBroadcastInput(CompiledTensorDescriptor descriptor, CompiledNode node) {
        int[] outputShape = node.shape();
        if (outputShape.length == 0 || !isBroadcastCompatible(descriptor.shape(), outputShape)) {
            return false;
        }
        int[] inputShape = descriptor.shape();
        int[] inputStrides = descriptor.strides();
        int offset = outputShape.length - inputShape.length;
        for (int dim = 0; dim < outputShape.length - 1; dim++) {
            int inputDim = dim < offset ? 1 : inputShape[dim - offset];
            int inputStride = dim < offset ? 0 : inputStrides[dim - offset];
            if (inputDim != 1 && inputStride != 0) {
                return false;
            }
        }
        int lastDim = outputShape.length - 1;
        int inputLastDim = lastDim < offset ? 1 : inputShape[lastDim - offset];
        int inputLastStride = lastDim < offset ? 0 : inputStrides[lastDim - offset];
        return inputLastDim == outputShape[lastDim] && inputLastStride == 1;
    }

    private record ScalarParameter(boolean present, float f32, double f64) {
        private static final ScalarParameter NONE = new ScalarParameter(false, 0.0f, 0.0d);

        private static ScalarParameter of(float f32, double f64) {
            return new ScalarParameter(true, f32, f64);
        }
    }
}

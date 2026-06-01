package backend.cpu1.prepare;

import backend.cpu1.exec.Cpu1LayoutExecutableUnit;
import backend.cpu1.exec.Cpu1WorkspaceSpec;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1RangeLauncher;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.layout.concat;
import operations.layout.fold2d;
import operations.layout.pad;
import operations.layout.tile;
import operations.layout.unfoldAxis;
import operations.Operation;
import config.backend.CpuKernelConfig;
import config.runtime.RuntimeConfig;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import tensor.DataType;
import tensor.options.Window2dOptions;

import java.util.List;
import java.util.Objects;

/**
 * Prepares cpu1 layout/view nodes outside the elementwise dispatch path.
 */
public final class Cpu1LayoutPreparer {
    public Cpu1PreparedArtifact prepare(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex,
            Cpu1PrepareConfig config
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
        Operation.OpType opType = operation.opType();
        if (!isLayoutOp(opType)) {
            throw new UnsupportedOperationException("cpu1 layout preparer does not support " + opType);
        }
        requireInputCount(opType, node);
        if (!isSupportedDType(node.dataType())) {
            throw new UnsupportedOperationException("cpu1 layout preparer does not support dtype "
                    + node.dataType() + " for " + opType);
        }
        if (!isSupportedLayoutDType(opType, node.dataType())) {
            throw new UnsupportedOperationException("cpu1 layout preparer does not support dtype "
                    + node.dataType() + " for " + opType);
        }
        List<CompiledTensorDescriptor> inputs = descriptorIndex == null
                ? List.of()
                : node.inputIds().stream().map(descriptorIndex::byNodeId).toList();
        if (!inputs.isEmpty()) {
            requireInputContract(opType, operation, node, inputs);
        }
        int workElements = layoutWorkElements(opType, node, inputs);
        Cpu1VectorizationKind requestedVectorizationKind = resolveLayoutVectorization(node.dataType(), workElements, config);
        Cpu1LayoutKernelId kernelId = kernelId(operation, opType, node, inputs, requestedVectorizationKind);
        Cpu1VectorizationKind vectorizationKind = effectiveVectorizationKind(kernelId);
        Cpu1LaunchConfig launchConfig = resolveLayoutLaunch(workElements, vectorizationKind, node.dataType(), config);
        Cpu1PreparedLayoutUnit unit = new Cpu1PreparedLayoutUnit(
                node.id(),
                node.inputIds(),
                opType,
                node.dataType(),
                config.storageKind(),
                kernelId,
                materializeThreshold(node.dataType(), config),
                vectorizationKind,
                launchConfig,
                workspaceSpec(opType, kernelId, node, inputs, launchConfig)
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isLayoutOp(Operation.OpType opType) {
        return switch (opType) {
            case NOOP, RESHAPE, EXPAND, SELECT, SLICE, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS,
                    CONCAT, PAD, TILE, UNFOLD_AXIS, UNFOLD2D, FOLD2D -> true;
            default -> false;
        };
    }

    private static void requireInputCount(Operation.OpType opType, CompiledNode node) {
        if (opType == Operation.OpType.CONCAT) {
            if (node.inputIds().isEmpty()) {
                throw new UnsupportedOperationException("cpu1 CONCAT expects at least 1 input");
            }
            return;
        }
        if (node.inputIds().size() != 1) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects 1 input, got "
                    + node.inputIds().size());
        }
    }

    private static void requireInputContract(
            Operation.OpType opType,
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs
    ) {
        for (CompiledTensorDescriptor input : inputs) {
            if (input.dataType() != node.dataType()) {
                throw new UnsupportedOperationException("cpu1 " + opType + " requires matching input/output dtype. input="
                        + input.dataType() + ", output=" + node.dataType());
            }
        }
        CompiledTensorDescriptor input = inputs.getFirst();
        if ((opType == Operation.OpType.RESHAPE || opType == Operation.OpType.CONTIGUOUS)
                && input.logicalElementCount() != node.flatDataSize()) {
            throw new UnsupportedOperationException("cpu1 " + opType + " requires matching element count. input="
                    + input.logicalElementCount() + ", output=" + node.flatDataSize());
        }
        if (opType == Operation.OpType.CONCAT) {
            requireConcatContract(operation, node, inputs);
        }
    }

    private static Cpu1LayoutKernelId kernelId(
            Operation operation,
            Operation.OpType opType,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        return switch (opType) {
            case NOOP -> Cpu1LayoutKernelId.NOOP_ALIAS;
            case RESHAPE -> reshapeAliasesInput(node, inputs.isEmpty() ? null : inputs.getFirst())
                    ? Cpu1LayoutKernelId.RESHAPE_ALIAS
                    : Cpu1LayoutKernelId.RESHAPE_COPY_LINEARIZED_SCALAR;
            case EXPAND -> Cpu1LayoutKernelId.EXPAND_ALIAS;
            case SELECT -> Cpu1LayoutKernelId.SELECT_ALIAS;
            case SLICE -> Cpu1LayoutKernelId.SLICE_ALIAS;
            case PERMUTE -> Cpu1LayoutKernelId.PERMUTE_ALIAS;
            case EXPAND_DIMS -> Cpu1LayoutKernelId.EXPAND_DIMS_ALIAS;
            case SQUEEZE -> Cpu1LayoutKernelId.SQUEEZE_ALIAS;
            case CONTIGUOUS -> contiguousKernelId(node, inputs, vectorizationKind);
            case CONCAT -> concatKernelId(operation, node, inputs, vectorizationKind);
            case PAD -> padKernelId(operation, node, inputs, vectorizationKind);
            case TILE -> tileKernelId(operation, node, inputs, vectorizationKind);
            case UNFOLD_AXIS -> unfoldAxisKernelId(operation, node, inputs, vectorizationKind);
            case UNFOLD2D -> Cpu1LayoutKernelId.UNFOLD2D_COPY_SCALAR;
            case FOLD2D -> fold2dKernelId(operation, node, inputs);
            default -> throw new UnsupportedOperationException("cpu1 layout preparer does not support " + opType);
        };
    }

    private static boolean vectorized(Cpu1VectorizationKind vectorizationKind) {
        return vectorizationKind == Cpu1VectorizationKind.VECTOR;
    }

    private static Cpu1VectorizationKind effectiveVectorizationKind(Cpu1LayoutKernelId kernelId) {
        return switch (kernelId) {
            case CONTIGUOUS_COPY_VECTOR, CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_VECTOR,
                    CONCAT_AXIS0_BLOCK_COPY_VECTOR, CONCAT_INNER_AXIS_BLOCK_COPY_VECTOR, CONCAT_MIDDLE_AXIS_BLOCK_COPY_VECTOR,
                    PAD_COPY_VECTOR, PAD_DENSE_INNER_BLOCK_COPY_VECTOR, TILE_LAST_AXIS_BLOCK_COPY_VECTOR,
                    TILE_AXIS0_BLOCK_COPY_VECTOR, TILE_DENSE_BLOCK_REPEAT_VECTOR,
                    UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_VECTOR ->
                    Cpu1VectorizationKind.VECTOR;
            default -> Cpu1VectorizationKind.SCALAR;
        };
    }

    private static boolean reshapeAliasesInput(CompiledNode node, CompiledTensorDescriptor input) {
        if (input != null) {
            return input.contiguous();
        }
        return node.storageOwnerId() != node.id();
    }

    private static Cpu1LayoutKernelId contiguousKernelId(
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        if (!inputs.isEmpty()
                && denseNoOffset(node)
                && denseWithOffset(inputs.getFirst())) {
            return vectorized(vectorizationKind)
                    ? Cpu1LayoutKernelId.CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_VECTOR
                    : Cpu1LayoutKernelId.CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_SCALAR;
        }
        return vectorized(vectorizationKind)
                ? Cpu1LayoutKernelId.CONTIGUOUS_COPY_VECTOR
                : Cpu1LayoutKernelId.CONTIGUOUS_COPY_SCALAR;
    }

    private static Cpu1LayoutKernelId concatKernelId(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        if (!(operation instanceof concat concatOp) || inputs.isEmpty() || !denseNoOffset(node) || !allDenseNoOffset(inputs)) {
            return Cpu1LayoutKernelId.CONCAT_COPY_SCALAR;
        }
        int axis = concatOp.getAxis();
        if (axis == 0) {
            return vectorized(vectorizationKind)
                    ? Cpu1LayoutKernelId.CONCAT_AXIS0_BLOCK_COPY_VECTOR
                    : Cpu1LayoutKernelId.CONCAT_AXIS0_BLOCK_COPY_SCALAR;
        }
        if (axis == node.shape().length - 1) {
            return vectorized(vectorizationKind)
                    ? Cpu1LayoutKernelId.CONCAT_INNER_AXIS_BLOCK_COPY_VECTOR
                    : Cpu1LayoutKernelId.CONCAT_INNER_AXIS_BLOCK_COPY_SCALAR;
        }
        return vectorized(vectorizationKind)
                ? Cpu1LayoutKernelId.CONCAT_MIDDLE_AXIS_BLOCK_COPY_VECTOR
                : Cpu1LayoutKernelId.CONCAT_MIDDLE_AXIS_BLOCK_COPY_SCALAR;
    }

    private static Cpu1LayoutKernelId padKernelId(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        if (!(operation instanceof pad padOp) || inputs.isEmpty() || !denseNoOffset(node) || !denseNoOffset(inputs.getFirst())) {
            return vectorized(vectorizationKind) ? Cpu1LayoutKernelId.PAD_COPY_VECTOR : Cpu1LayoutKernelId.PAD_COPY_SCALAR;
        }
        int rank = node.shape().length;
        int[] before = padOp.getBefore();
        int[] after = padOp.getAfter();
        if (rank == 0 || before.length != rank || after.length != rank) {
            return vectorized(vectorizationKind) ? Cpu1LayoutKernelId.PAD_COPY_VECTOR : Cpu1LayoutKernelId.PAD_COPY_SCALAR;
        }
        return vectorized(vectorizationKind)
                ? Cpu1LayoutKernelId.PAD_DENSE_INNER_BLOCK_COPY_VECTOR
                : Cpu1LayoutKernelId.PAD_DENSE_INNER_BLOCK_COPY_SCALAR;
    }

    private static Cpu1LayoutKernelId tileKernelId(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        if (!(operation instanceof tile tileOp) || inputs.isEmpty() || !denseNoOffset(node) || !denseNoOffset(inputs.getFirst())) {
            return Cpu1LayoutKernelId.TILE_COPY_SCALAR;
        }
        int rank = node.shape().length;
        int[] repeats = tileOp.getRepeats();
        if (rank == 0 || repeats.length != rank) {
            return Cpu1LayoutKernelId.TILE_COPY_SCALAR;
        }
        if (repeatsOnlyAxis0(repeats)) {
            return vectorized(vectorizationKind)
                    ? Cpu1LayoutKernelId.TILE_AXIS0_BLOCK_COPY_VECTOR
                    : Cpu1LayoutKernelId.TILE_AXIS0_BLOCK_COPY_SCALAR;
        }
        int repeatedAxis = singleRepeatedAxis(repeats);
        if (repeatedAxis > 0 && repeatedAxis < rank - 1) {
            return vectorized(vectorizationKind)
                    ? Cpu1LayoutKernelId.TILE_DENSE_BLOCK_REPEAT_VECTOR
                    : Cpu1LayoutKernelId.TILE_DENSE_BLOCK_REPEAT_SCALAR;
        }
        for (int dim = 0; dim < rank - 1; dim++) {
            if (repeats[dim] != 1) {
                return Cpu1LayoutKernelId.TILE_COPY_SCALAR;
            }
        }
        return vectorized(vectorizationKind)
                ? Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_VECTOR
                : Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_SCALAR;
    }

    private static boolean repeatsOnlyAxis0(int[] repeats) {
        for (int dim = 1; dim < repeats.length; dim++) {
            if (repeats[dim] != 1) {
                return false;
            }
        }
        return true;
    }

    private static int singleRepeatedAxis(int[] repeats) {
        int axis = -1;
        for (int dim = 0; dim < repeats.length; dim++) {
            if (repeats[dim] == 1) {
                continue;
            }
            if (axis != -1) {
                return -1;
            }
            axis = dim;
        }
        return axis;
    }

    private static Cpu1LayoutKernelId unfoldAxisKernelId(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1VectorizationKind vectorizationKind
    ) {
        if (!(operation instanceof unfoldAxis unfoldOp)
                || inputs.isEmpty()
                || !denseNoOffset(node)
                || !denseNoOffset(inputs.getFirst())) {
            return Cpu1LayoutKernelId.UNFOLD_AXIS_COPY_SCALAR;
        }
        int inputRank = inputs.getFirst().rank();
        if (inputRank == 0 || unfoldOp.getAxis() != inputRank - 1) {
            return Cpu1LayoutKernelId.UNFOLD_AXIS_COPY_SCALAR;
        }
        return vectorized(vectorizationKind)
                ? Cpu1LayoutKernelId.UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_VECTOR
                : Cpu1LayoutKernelId.UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_SCALAR;
    }

    private static Cpu1LayoutKernelId fold2dKernelId(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs
    ) {
        if (!(operation instanceof fold2d foldOp)
                || inputs.isEmpty()
                || !denseNoOffset(node)
                || !denseNoOffset(inputs.getFirst())) {
            return Cpu1LayoutKernelId.FOLD2D_COPY_SCALAR;
        }
        return nonOverlappingWindow(foldOp.getOptions())
                ? Cpu1LayoutKernelId.FOLD2D_NON_OVERLAP_DIRECT_SCALAR
                : Cpu1LayoutKernelId.FOLD2D_COPY_SCALAR;
    }

    private static boolean nonOverlappingWindow(Window2dOptions options) {
        return options.dilationH() == 1
                && options.dilationW() == 1
                && options.strideH() >= options.kernelH()
                && options.strideW() >= options.kernelW();
    }

    private static boolean allDenseNoOffset(List<CompiledTensorDescriptor> inputs) {
        for (CompiledTensorDescriptor input : inputs) {
            if (!denseNoOffset(input)) {
                return false;
            }
        }
        return true;
    }

    private static boolean denseNoOffset(CompiledNode node) {
        return node.contiguous() && !node.hasStorageOffset();
    }

    private static boolean denseNoOffset(CompiledTensorDescriptor descriptor) {
        return descriptor.denseContiguousWithoutOffset();
    }

    private static boolean denseWithOffset(CompiledTensorDescriptor descriptor) {
        return descriptor.dense() && descriptor.hasStorageOffset();
    }

    private static int materializeThreshold(DataType dataType, Cpu1PrepareConfig config) {
        if (config.cpuKernelConfig() != null) {
            return config.cpuKernelConfig().contiguousMaterializeThreshold();
        }
        return RuntimeConfig.inferenceDefaults(dataType).cpuKernelConfig().contiguousMaterializeThreshold();
    }

    private static Cpu1WorkspaceSpec workspaceSpec(
            Operation.OpType opType,
            Cpu1LayoutKernelId kernelId,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs,
            Cpu1LaunchConfig launchConfig
    ) {
        if (opType == Operation.OpType.FOLD2D && kernelId != Cpu1LayoutKernelId.FOLD2D_NON_OVERLAP_DIRECT_SCALAR) {
            int inputElements = inputs.isEmpty()
                    ? node.flatDataSize()
                    : Math.toIntExact(inputs.getFirst().logicalElementCount());
            return Cpu1WorkspaceSpec.arrays(
                    0,
                    Math.multiplyExact(node.flatDataSize(), Cpu1RangeLauncher.slotCount(inputElements, launchConfig)),
                    0
            );
        }
        return Cpu1WorkspaceSpec.none();
    }

    private static boolean isSupportedDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.BOOL;
    }

    private static boolean isSupportedLayoutDType(Operation.OpType opType, DataType dataType) {
        if (opType == Operation.OpType.UNFOLD2D || opType == Operation.OpType.FOLD2D) {
            return dataType == DataType.FLOAT32
                    || dataType == DataType.FLOAT64
                    || dataType == DataType.BFLOAT16;
        }
        if (opType == Operation.OpType.UNFOLD_AXIS) {
            return dataType == DataType.FLOAT32
                    || dataType == DataType.FLOAT64
                    || dataType == DataType.BFLOAT16
                    || dataType == DataType.BOOL;
        }
        return isSupportedDType(dataType);
    }

    private static int layoutWorkElements(
            Operation.OpType opType,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs
    ) {
        if (opType == Operation.OpType.FOLD2D && !inputs.isEmpty()) {
            return Math.toIntExact(inputs.getFirst().logicalElementCount());
        }
        return node.flatDataSize();
    }

    private static Cpu1VectorizationKind resolveLayoutVectorization(
            DataType dataType,
            int workElements,
            Cpu1PrepareConfig config
    ) {
        if (!config.automaticVectorization()) {
            return config.vectorizationKind();
        }
        CpuKernelConfig cpuKernelConfig = requireCpuKernelConfig(config);
        int vectorWidth = preferredVectorWidth(dataType);
        if (vectorWidth <= 1 || workElements < cpuKernelConfig.cheapVectorMinSize()) {
            return Cpu1VectorizationKind.SCALAR;
        }
        return Cpu1VectorizationKind.VECTOR;
    }

    private static Cpu1LaunchConfig resolveLayoutLaunch(
            int workElements,
            Cpu1VectorizationKind vectorizationKind,
            DataType dataType,
            Cpu1PrepareConfig config
    ) {
        if (!config.automaticLaunch()) {
            return config.launchConfig();
        }
        CpuKernelConfig cpuKernelConfig = requireCpuKernelConfig(config);
        int maxWorkers = config.launchConfig().workerCount();
        if (maxWorkers <= 1 || workElements < cpuKernelConfig.cheapParallelMinSize()) {
            return Cpu1LaunchConfig.singleThread();
        }
        int plannedWorkers = Math.min(maxWorkers, Math.max(1, workElements));
        int alignment = vectorizationKind == Cpu1VectorizationKind.VECTOR ? preferredVectorWidth(dataType) : 1;
        int minChunk = vectorizationKind == Cpu1VectorizationKind.VECTOR
                ? cpuKernelConfig.minVectorChunkSize()
                : cpuKernelConfig.minScalarChunkSize();
        int chunk = computeChunkSize(
                workElements,
                alignment,
                cpuKernelConfig.lowCostTargetChunksPerWorker(),
                minChunk,
                plannedWorkers
        );
        return Cpu1LaunchConfig.parallel(plannedWorkers, chunk);
    }

    private static int computeChunkSize(
            int totalLength,
            int alignment,
            int targetChunksPerWorker,
            int minChunkSize,
            int plannedWorkers
    ) {
        int length = Math.max(1, totalLength);
        int workers = Math.max(1, plannedWorkers);
        int targets = Math.max(workers, workers * Math.max(1, targetChunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);
        int align = Math.max(1, alignment);
        int remainder = chunk % align;
        return remainder == 0 ? chunk : chunk + align - remainder;
    }

    private static int preferredVectorWidth(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case BFLOAT16 -> ShortVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            case INT32, INT64 -> 1;
        };
    }

    private static CpuKernelConfig requireCpuKernelConfig(Cpu1PrepareConfig config) {
        CpuKernelConfig cpuKernelConfig = config.cpuKernelConfig();
        if (cpuKernelConfig == null) {
            throw new IllegalArgumentException("Automatic cpu1 layout dispatch requires CpuKernelConfig-backed prepare config.");
        }
        return cpuKernelConfig;
    }

    private static void requireConcatContract(
            Operation operation,
            CompiledNode node,
            List<CompiledTensorDescriptor> inputs
    ) {
        if (!(operation instanceof concat concatOp)) {
            throw new IllegalArgumentException("cpu1 CONCAT operation must be operations.layout.concat.");
        }
        int axis = concatOp.getAxis();
        int[] outputShape = node.shape();
        if (axis < 0 || axis >= outputShape.length) {
            throw new UnsupportedOperationException("cpu1 CONCAT axis out of bounds: " + axis);
        }
        int axisSize = 0;
        for (CompiledTensorDescriptor input : inputs) {
            int[] inputShape = input.shape();
            if (inputShape.length != outputShape.length) {
                throw new UnsupportedOperationException("cpu1 CONCAT rank mismatch. inputRank="
                        + inputShape.length + ", outputRank=" + outputShape.length);
            }
            for (int dim = 0; dim < outputShape.length; dim++) {
                if (dim != axis && inputShape[dim] != outputShape[dim]) {
                    throw new UnsupportedOperationException("cpu1 CONCAT shape mismatch at dim=" + dim);
                }
            }
            axisSize = Math.addExact(axisSize, inputShape[axis]);
        }
        if (axisSize != outputShape[axis]) {
            throw new UnsupportedOperationException("cpu1 CONCAT output axis size mismatch. inputs="
                    + axisSize + ", output=" + outputShape[axis]);
        }
    }
}

package backend.cpu1.prepare;

import backend.cpu1.kernels.index.Cpu1IndexKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1LaunchPolicy;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageAccessPlan;
import backend.cpu1.storage.Cpu1StorageKind;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import operations.index.gather;
import operations.index.gatherAxis;
import operations.index.gatherNd;
import operations.index.ScatterReduction;
import operations.index.scatterAdd;
import operations.index.scatterAxisAdd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.index.takeAlongAxis;
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
            throw new UnsupportedOperationException("cpu1 " + opType
                    + " requires descriptors to resolve value/index dtype and layout.");
        }
        int expectedInputCount = isScatterOp(opType) ? 3 : 2;
        if (node.inputIds().size() != expectedInputCount) {
            throw new UnsupportedOperationException("cpu1 " + opType + " expects " + expectedInputCount + " inputs, got "
                    + node.inputIds().size());
        }
        if (config.storageKind() != Cpu1StorageKind.JAVA_ARRAY
                && config.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            throw new UnsupportedOperationException("cpu1 " + opType
                    + " dense index slice supports only JAVA_ARRAY/MEMORY_SEGMENT storage, got "
                    + config.storageKind());
        }

        CompiledTensorDescriptor input = descriptorIndex.byNodeId(node.inputIds().get(0));
        CompiledTensorDescriptor indices = descriptorIndex.byNodeId(node.inputIds().get(1));
        CompiledTensorDescriptor updates = isScatterOp(opType)
                ? descriptorIndex.byNodeId(node.inputIds().get(2))
                : null;
        Cpu1StorageAccessPlan inputAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(input);
        Cpu1StorageAccessPlan indexAccessPlan = Cpu1StorageAccessPlan.fromDescriptor(indices);
        Cpu1StorageAccessPlan updateAccessPlan = updates == null
                ? null
                : Cpu1StorageAccessPlan.fromDescriptor(updates);
        Cpu1StorageAccessPlan outputAccessPlan = Cpu1StorageAccessPlan.fromNode(node);

        ScatterReduction reduction = scatterReduction(operation, opType);
        requireSupportedDTypes(opType, reduction, node, input, indices, updates);
        requireDenseContiguousNoOffset("input", inputAccessPlan);
        requireDenseContiguousNoOffset("indices", indexAccessPlan);
        if (updateAccessPlan != null) {
            requireDenseContiguousNoOffset("updates", updateAccessPlan);
        }
        requireDenseContiguousNoOffset("output", outputAccessPlan);

        ShapeFacts shapeFacts = shapeFacts(
                operation,
                opType,
                input.shape(),
                indices.shape(),
                updates == null ? null : updates.shape(),
                node.shape()
        );
        Cpu1LaunchConfig launchConfig = isScatterOp(opType)
                ? Cpu1LaunchConfig.singleThread()
                : config.launchConfig();
        Cpu1PreparedIndexUnit unit = new Cpu1PreparedIndexUnit(
                node.id(),
                input.nodeId(),
                indices.nodeId(),
                updates == null ? -1 : updates.nodeId(),
                opType,
                reduction,
                node.dataType(),
                indices.dataType(),
                config.storageKind(),
                kernelId(opType, node.dataType(), indices.dataType(), config.storageKind()),
                shapeFacts.dimension(),
                shapeFacts.axisSize(),
                shapeFacts.innerSize(),
                shapeFacts.outerSize(),
                shapeFacts.indexElementCount(),
                shapeFacts.indexAxisSize(),
                shapeFacts.batchDims(),
                shapeFacts.tupleRank(),
                shapeFacts.prefixRank(),
                shapeFacts.tupleStride(),
                shapeFacts.updateElementCount(),
                shapeFacts.gatherNdInputShape(),
                shapeFacts.gatherNdInputStrides(),
                shapeFacts.gatherNdIndicesDenseStrides(),
                shapeFacts.gatherNdOutputShape(),
                shapeFacts.gatherNdOutputDenseStrides(),
                node.flatDataSize(),
                launchConfig,
                launchPolicy(launchConfig),
                inputAccessPlan,
                indexAccessPlan,
                updateAccessPlan,
                outputAccessPlan
        );
        return new Cpu1PreparedArtifact(unit);
    }

    public static boolean isIndexOp(Operation.OpType opType) {
        return opType == Operation.OpType.GATHER
                || opType == Operation.OpType.GATHER_AXIS
                || opType == Operation.OpType.GATHER_ND
                || opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.SCATTER_AXIS_ADD
                || opType == Operation.OpType.SCATTER_ELEMENTS
                || opType == Operation.OpType.SCATTER_ND
                || opType == Operation.OpType.TAKE_ALONG_AXIS;
    }

    private static boolean isScatterOp(Operation.OpType opType) {
        return opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.SCATTER_AXIS_ADD
                || opType == Operation.OpType.SCATTER_ELEMENTS
                || opType == Operation.OpType.SCATTER_ND;
    }

    private static void requireSupportedDTypes(
            Operation.OpType opType,
            ScatterReduction reduction,
            CompiledNode node,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor indices,
            CompiledTensorDescriptor updates
    ) {
        if (input.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 " + opType + " requires matching input/output dtype. input="
                    + input.dataType() + ", output=" + node.dataType());
        }
        if (updates != null && updates.dataType() != node.dataType()) {
            throw new UnsupportedOperationException("cpu1 " + opType + " requires matching input/updates/output dtype. input="
                    + input.dataType() + ", updates=" + updates.dataType() + ", output=" + node.dataType());
        }
        boolean supportedValue = isFloatingScatterAddOp(opType)
                ? isSupportedScatterAddValueDType(node.dataType())
                : isSupportedValueDType(node.dataType());
        if (!supportedValue) {
            throw new UnsupportedOperationException("cpu1 " + opType + " does not support value dtype "
                    + node.dataType());
        }
        if ((opType == Operation.OpType.SCATTER_ELEMENTS || opType == Operation.OpType.SCATTER_ND)
                && node.dataType() == DataType.BOOL
                && reduction != ScatterReduction.NONE) {
            throw new UnsupportedOperationException("cpu1 " + opType
                    + " BOOL tensors support only NONE reduction.");
        }
        if (indices.dataType() != DataType.INT32 && indices.dataType() != DataType.INT64) {
            throw new UnsupportedOperationException("cpu1 " + opType
                    + " dense index slice supports only INT32/INT64 indices, got " + indices.dataType());
        }
    }

    private static ScatterReduction scatterReduction(Operation operation, Operation.OpType opType) {
        return switch (opType) {
            case SCATTER_ADD, SCATTER_AXIS_ADD -> ScatterReduction.ADD;
            case SCATTER_ELEMENTS -> {
                if (!(operation instanceof scatterElements scatterElementsOp)) {
                    throw new IllegalArgumentException("cpu1 SCATTER_ELEMENTS operation must be operations.index.scatterElements.");
                }
                yield scatterElementsOp.getReduction();
            }
            case SCATTER_ND -> {
                if (!(operation instanceof scatterNd scatterNdOp)) {
                    throw new IllegalArgumentException("cpu1 SCATTER_ND operation must be operations.index.scatterNd.");
                }
                yield scatterNdOp.getReduction();
            }
            default -> ScatterReduction.NONE;
        };
    }

    private static boolean isFloatingScatterAddOp(Operation.OpType opType) {
        return opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.SCATTER_AXIS_ADD;
    }

    private static boolean isSupportedValueDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16
                || dataType == DataType.INT32
                || dataType == DataType.INT64
                || dataType == DataType.BOOL;
    }

    private static boolean isSupportedScatterAddValueDType(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.FLOAT64
                || dataType == DataType.BFLOAT16;
    }

    private static void requireDenseContiguousNoOffset(String role, Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.kind() == Cpu1StorageAccessKind.DENSE_CONTIGUOUS) {
            return;
        }
        throw new UnsupportedOperationException("cpu1 dense index slice supports only dense contiguous no-offset "
                + role + " access; actual=" + accessPlan.kind() + rejectionSuffix(accessPlan));
    }

    private static String rejectionSuffix(Cpu1StorageAccessPlan accessPlan) {
        if (accessPlan.rejectionReason() == null || accessPlan.rejectionReason().isBlank()) {
            return "";
        }
        return ", reason=" + accessPlan.rejectionReason();
    }

    private static ShapeFacts shapeFacts(
            Operation operation,
            Operation.OpType opType,
            int[] inputShape,
            int[] indicesShape,
            int[] updatesShape,
            int[] outputShape
    ) {
        return switch (opType) {
            case GATHER -> gatherShapeFacts(operation, inputShape, indicesShape, outputShape);
            case GATHER_AXIS -> gatherAxisShapeFacts(operation, inputShape, indicesShape, outputShape);
            case GATHER_ND -> gatherNdShapeFacts(operation, inputShape, indicesShape, outputShape);
            case SCATTER_ADD -> scatterAddShapeFacts(operation, inputShape, indicesShape, updatesShape, outputShape);
            case SCATTER_AXIS_ADD -> scatterAxisAddShapeFacts(operation, inputShape, indicesShape, updatesShape, outputShape);
            case SCATTER_ELEMENTS -> scatterElementsShapeFacts(operation, inputShape, indicesShape, updatesShape, outputShape);
            case SCATTER_ND -> scatterNdShapeFacts(operation, inputShape, indicesShape, updatesShape, outputShape);
            case TAKE_ALONG_AXIS -> takeAlongAxisShapeFacts(operation, inputShape, indicesShape, outputShape);
            default -> throw new UnsupportedOperationException("cpu1 index preparer does not support " + opType);
        };
    }

    private static ShapeFacts gatherShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof gather gatherOp)) {
            throw new IllegalArgumentException("cpu1 GATHER operation must be operations.index.gather.");
        }
        int dimension = normalizedDimension(gatherOp.getDimension(), inputShape.length, Operation.OpType.GATHER);
        int[] expectedOutputShape = reducedShape(inputShape, dimension);
        requireShapeEquals(
                "GATHER indices shape must equal input shape without gathered axis",
                indicesShape,
                expectedOutputShape
        );
        requireShapeEquals("GATHER output shape must equal indices shape", outputShape, expectedOutputShape);
        return commonShapeFacts(inputShape, indicesShape, dimension, 1);
    }

    private static ShapeFacts gatherAxisShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof gatherAxis gatherAxisOp)) {
            throw new IllegalArgumentException("cpu1 GATHER_AXIS operation must be operations.index.gatherAxis.");
        }
        int axis = normalizedDimension(gatherAxisOp.getAxis(), inputShape.length, Operation.OpType.GATHER_AXIS);
        requireShapeEquals(
                "GATHER_AXIS output shape mismatch",
                outputShape,
                gatherAxisOutputShape(inputShape, indicesShape, axis)
        );
        return commonShapeFacts(inputShape, indicesShape, axis, 1);
    }

    private static ShapeFacts takeAlongAxisShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof takeAlongAxis takeAlongAxisOp)) {
            throw new IllegalArgumentException("cpu1 TAKE_ALONG_AXIS operation must be operations.index.takeAlongAxis.");
        }
        int dimension = normalizedDimension(
                takeAlongAxisOp.getDimension(),
                inputShape.length,
                Operation.OpType.TAKE_ALONG_AXIS
        );
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("cpu1 TAKE_ALONG_AXIS indices rank must match input rank. inputRank="
                    + inputShape.length + ", indicesRank=" + indicesShape.length);
        }
        for (int i = 0; i < inputShape.length; i++) {
            if (i != dimension && indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("cpu1 TAKE_ALONG_AXIS indices must match input shape on all "
                        + "non-axis dimensions. input=" + Arrays.toString(inputShape)
                        + ", indices=" + Arrays.toString(indicesShape) + ", axis=" + dimension);
            }
        }
        requireShapeEquals("TAKE_ALONG_AXIS output shape must equal indices shape", outputShape, indicesShape);
        return commonShapeFacts(inputShape, indicesShape, dimension, indicesShape[dimension]);
    }

    private static ShapeFacts scatterAddShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] updatesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof scatterAdd scatterOp)) {
            throw new IllegalArgumentException("cpu1 SCATTER_ADD operation must be operations.index.scatterAdd.");
        }
        if (updatesShape == null) {
            throw new IllegalArgumentException("cpu1 SCATTER_ADD requires updates input shape.");
        }
        int dimension = normalizedDimension(scatterOp.getDimension(), inputShape.length, Operation.OpType.SCATTER_ADD);
        int[] expectedUpdatesShape = reducedShape(inputShape, dimension);
        requireShapeEquals(
                "SCATTER_ADD indices shape must equal input shape without scattered axis",
                indicesShape,
                expectedUpdatesShape
        );
        requireShapeEquals(
                "SCATTER_ADD updates shape must equal indices shape",
                updatesShape,
                expectedUpdatesShape
        );
        requireShapeEquals("SCATTER_ADD output shape must equal input shape", outputShape, inputShape);
        return commonShapeFacts(inputShape, indicesShape, dimension, 1, product(updatesShape, 0, updatesShape.length));
    }

    private static ShapeFacts scatterAxisAddShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] updatesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof scatterAxisAdd scatterAxisOp)) {
            throw new IllegalArgumentException("cpu1 SCATTER_AXIS_ADD operation must be operations.index.scatterAxisAdd.");
        }
        if (updatesShape == null) {
            throw new IllegalArgumentException("cpu1 SCATTER_AXIS_ADD requires updates input shape.");
        }
        int axis = normalizedDimension(scatterAxisOp.getAxis(), inputShape.length, Operation.OpType.SCATTER_AXIS_ADD);
        requireShapeEquals("SCATTER_AXIS_ADD output shape must equal input shape", outputShape, inputShape);
        requireShapeEquals(
                "SCATTER_AXIS_ADD updates shape must match GATHER_AXIS output shape",
                updatesShape,
                gatherAxisOutputShape(inputShape, indicesShape, axis)
        );
        return commonShapeFacts(inputShape, indicesShape, axis, 1, product(updatesShape, 0, updatesShape.length));
    }

    private static ShapeFacts scatterElementsShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] updatesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof scatterElements scatterElementsOp)) {
            throw new IllegalArgumentException("cpu1 SCATTER_ELEMENTS operation must be operations.index.scatterElements.");
        }
        if (updatesShape == null) {
            throw new IllegalArgumentException("cpu1 SCATTER_ELEMENTS requires updates input shape.");
        }
        int axis = normalizedDimension(scatterElementsOp.getAxis(), inputShape.length, Operation.OpType.SCATTER_ELEMENTS);
        requireShapeEquals("SCATTER_ELEMENTS output shape must equal input shape", outputShape, inputShape);
        if (indicesShape.length != inputShape.length) {
            throw new IllegalArgumentException("cpu1 SCATTER_ELEMENTS indices rank must match input rank. inputRank="
                    + inputShape.length + ", indicesRank=" + indicesShape.length);
        }
        requireShapeEquals("SCATTER_ELEMENTS updates shape must equal indices shape", updatesShape, indicesShape);
        for (int i = 0; i < indicesShape.length; i++) {
            if (i != axis && indicesShape[i] != inputShape[i]) {
                throw new IllegalArgumentException("cpu1 SCATTER_ELEMENTS indices must match input shape on all "
                        + "non-axis dimensions. input=" + Arrays.toString(inputShape)
                        + ", indices=" + Arrays.toString(indicesShape) + ", axis=" + axis);
            }
        }
        return new ShapeFacts(
                axis,
                inputShape[axis],
                product(inputShape, axis + 1, inputShape.length),
                product(inputShape, 0, axis),
                product(indicesShape, 0, indicesShape.length),
                indicesShape[axis],
                0,
                0,
                0,
                0,
                product(updatesShape, 0, updatesShape.length),
                inputShape,
                denseStrides(inputShape),
                denseStrides(indicesShape),
                updatesShape,
                denseStrides(updatesShape)
        );
    }

    private static ShapeFacts scatterNdShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] updatesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof scatterNd scatterNdOp)) {
            throw new IllegalArgumentException("cpu1 SCATTER_ND operation must be operations.index.scatterNd.");
        }
        if (updatesShape == null) {
            throw new IllegalArgumentException("cpu1 SCATTER_ND requires updates input shape.");
        }
        int batchDims = scatterNdOp.getBatchDims();
        validateGatherNdShape(inputShape, indicesShape, batchDims);
        requireShapeEquals("SCATTER_ND output shape must equal input shape", outputShape, inputShape);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int prefixRank = indicesShape.length - 1;
        int expectedRank = prefixRank + inputShape.length - batchDims - tupleRank;
        if (updatesShape.length != expectedRank) {
            if (!(expectedRank == 0 && updatesShape.length == 1 && updatesShape[0] == 1)) {
                throw new IllegalArgumentException("cpu1 SCATTER_ND updates shape must equal "
                        + "indices.shape[:-1] + input.shape[batchDims + indices.shape[-1]:]. expectedRank="
                        + expectedRank + ", updates=" + Arrays.toString(updatesShape));
            }
        } else {
            int p = 0;
            for (int i = 0; i < prefixRank; i++) {
                if (updatesShape[p++] != indicesShape[i]) {
                    throw new IllegalArgumentException("cpu1 SCATTER_ND updates prefix shape must match indices prefix shape.");
                }
            }
            for (int i = batchDims + tupleRank; i < inputShape.length; i++) {
                if (updatesShape[p++] != inputShape[i]) {
                    throw new IllegalArgumentException("cpu1 SCATTER_ND updates suffix shape must match indexed input slice shape.");
                }
            }
        }
        int[] indicesDense = denseStrides(indicesShape);
        return new ShapeFacts(
                batchDims,
                tupleRank,
                product(inputShape, batchDims + tupleRank, inputShape.length),
                product(indicesShape, 0, prefixRank),
                product(indicesShape, 0, indicesShape.length),
                tupleRank,
                batchDims,
                tupleRank,
                prefixRank,
                indicesDense[indicesShape.length - 1],
                product(updatesShape, 0, updatesShape.length),
                inputShape,
                denseStrides(inputShape),
                indicesDense,
                updatesShape,
                denseStrides(updatesShape)
        );
    }

    private static ShapeFacts commonShapeFacts(
            int[] inputShape,
            int[] indicesShape,
            int dimension,
            int indexAxisSize
    ) {
        return commonShapeFacts(inputShape, indicesShape, dimension, indexAxisSize, 0);
    }

    private static ShapeFacts commonShapeFacts(
            int[] inputShape,
            int[] indicesShape,
            int dimension,
            int indexAxisSize,
            int updateElementCount
    ) {
        return new ShapeFacts(
                dimension,
                inputShape[dimension],
                product(inputShape, dimension + 1, inputShape.length),
                product(inputShape, 0, dimension),
                product(indicesShape, 0, indicesShape.length),
                indexAxisSize,
                0,
                0,
                0,
                0,
                updateElementCount,
                new int[0],
                new int[0],
                new int[0],
                new int[0],
                new int[0]
        );
    }

    private static ShapeFacts gatherNdShapeFacts(
            Operation operation,
            int[] inputShape,
            int[] indicesShape,
            int[] outputShape
    ) {
        if (!(operation instanceof gatherNd gatherNdOp)) {
            throw new IllegalArgumentException("cpu1 GATHER_ND operation must be operations.index.gatherNd.");
        }
        int batchDims = gatherNdOp.getBatchDims();
        validateGatherNdShape(inputShape, indicesShape, batchDims);
        int tupleRank = indicesShape[indicesShape.length - 1];
        int prefixRank = indicesShape.length - 1;
        int[] expectedOutputShape = gatherNdOutputShape(inputShape, indicesShape, batchDims, tupleRank);
        requireShapeEquals("GATHER_ND output shape mismatch", outputShape, expectedOutputShape);
        int[] indicesDense = denseStrides(indicesShape);
        return new ShapeFacts(
                batchDims,
                tupleRank,
                product(inputShape, batchDims + tupleRank, inputShape.length),
                product(indicesShape, 0, prefixRank),
                product(indicesShape, 0, indicesShape.length),
                tupleRank,
                batchDims,
                tupleRank,
                prefixRank,
                indicesDense[indicesShape.length - 1],
                0,
                inputShape,
                denseStrides(inputShape),
                indicesDense,
                outputShape,
                denseStrides(outputShape)
        );
    }

    private static void validateGatherNdShape(int[] dataShape, int[] indicesShape, int batchDims) {
        if (indicesShape.length == 0) {
            throw new IllegalArgumentException("cpu1 gatherNd indices rank must be at least 1.");
        }
        if (batchDims < 0 || batchDims >= indicesShape.length) {
            throw new IllegalArgumentException("cpu1 gatherNd batchDims must be in [0, indices rank).");
        }
        if (batchDims > dataShape.length) {
            throw new IllegalArgumentException("cpu1 gatherNd batchDims cannot exceed data rank.");
        }
        for (int i = 0; i < batchDims; i++) {
            if (indicesShape[i] != dataShape[i]) {
                throw new IllegalArgumentException("cpu1 gatherNd batch dimensions must match data leading dimensions.");
            }
        }
        int tupleRank = indicesShape[indicesShape.length - 1];
        if (tupleRank <= 0 || batchDims + tupleRank > dataShape.length) {
            throw new IllegalArgumentException("cpu1 gatherNd final indices dimension must be in [1, data rank - batchDims].");
        }
    }

    private static int[] gatherNdOutputShape(
            int[] dataShape,
            int[] indicesShape,
            int batchDims,
            int tupleRank
    ) {
        int outputRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (outputRank == 0) {
            return new int[]{1};
        }
        int[] out = new int[outputRank];
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            out[p++] = indicesShape[i];
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            out[p++] = dataShape[i];
        }
        return out;
    }

    private static void requireShapeEquals(String message, int[] actual, int[] expected) {
        if (!Arrays.equals(actual, expected)) {
            throw new IllegalArgumentException("cpu1 " + message + ". expected=" + Arrays.toString(expected)
                    + ", actual=" + Arrays.toString(actual));
        }
    }

    private static int normalizedDimension(int dimension, int rank, Operation.OpType opType) {
        if (rank <= 0) {
            throw new IllegalArgumentException("cpu1 " + opType + " requires rank > 0 input.");
        }
        int normalized = dimension < 0 ? dimension + rank : dimension;
        if (normalized < 0 || normalized >= rank) {
            throw new IllegalArgumentException("cpu1 " + opType + " dimension out of bounds: dimension="
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

    private static int[] gatherAxisOutputShape(int[] dataShape, int[] indicesShape, int axis) {
        int[] output = new int[dataShape.length + indicesShape.length - 1];
        int p = 0;
        for (int i = 0; i < axis; i++) {
            output[p++] = dataShape[i];
        }
        for (int dim : indicesShape) {
            output[p++] = dim;
        }
        for (int i = axis + 1; i < dataShape.length; i++) {
            output[p++] = dataShape[i];
        }
        return output;
    }

    private static Cpu1IndexKernelId kernelId(
            Operation.OpType opType,
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        return switch (opType) {
            case GATHER -> gatherKernelId(valueDataType, indexDataType, storageKind);
            case GATHER_AXIS -> gatherAxisKernelId(valueDataType, indexDataType, storageKind);
            case GATHER_ND -> gatherNdKernelId(valueDataType, indexDataType, storageKind);
            case SCATTER_ADD -> scatterAddKernelId(valueDataType, indexDataType, storageKind);
            case SCATTER_AXIS_ADD -> scatterAxisAddKernelId(valueDataType, indexDataType, storageKind);
            case SCATTER_ELEMENTS -> scatterElementsKernelId(valueDataType, indexDataType, storageKind);
            case SCATTER_ND -> scatterNdKernelId(valueDataType, indexDataType, storageKind);
            case TAKE_ALONG_AXIS -> takeAlongAxisKernelId(valueDataType, indexDataType, storageKind);
            default -> throw new UnsupportedOperationException("cpu1 index preparer does not support " + opType);
        };
    }

    private static Cpu1IndexKernelId gatherKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_BOOL_I64_DENSE_SEGMENT;
            };
        }
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

    private static Cpu1IndexKernelId gatherAxisKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_AXIS_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_AXIS_BOOL_I64_DENSE_SEGMENT;
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_AXIS_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_AXIS_BOOL_I64_DENSE_ARRAY;
        };
    }

    private static Cpu1IndexKernelId gatherNdKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.GATHER_ND_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.GATHER_ND_BOOL_I64_DENSE_SEGMENT;
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.GATHER_ND_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.GATHER_ND_BOOL_I64_DENSE_ARRAY;
        };
    }

    private static Cpu1IndexKernelId takeAlongAxisKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.TAKE_ALONG_AXIS_BOOL_I64_DENSE_SEGMENT;
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.TAKE_ALONG_AXIS_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.TAKE_ALONG_AXIS_BOOL_I64_DENSE_ARRAY;
        };
    }

    private static Cpu1IndexKernelId scatterAddKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ADD_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ADD_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ADD_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ADD_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ADD_BF16_I64_DENSE_SEGMENT;
                case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 SCATTER_ADD supports only floating value dtypes, got " + valueDataType);
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ADD_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ADD_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ADD_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ADD_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ADD_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ADD_BF16_I64_DENSE_ARRAY;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 SCATTER_ADD supports only floating value dtypes, got " + valueDataType);
        };
    }

    private static Cpu1IndexKernelId scatterAxisAddKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_AXIS_ADD_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_AXIS_ADD_BF16_I64_DENSE_SEGMENT;
                case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 SCATTER_AXIS_ADD supports only floating value dtypes, got " + valueDataType);
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_AXIS_ADD_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_AXIS_ADD_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_AXIS_ADD_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_AXIS_ADD_BF16_I64_DENSE_ARRAY;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("cpu1 SCATTER_AXIS_ADD supports only floating value dtypes, got " + valueDataType);
        };
    }

    private static Cpu1IndexKernelId scatterElementsKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ELEMENTS_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ELEMENTS_BOOL_I64_DENSE_SEGMENT;
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ELEMENTS_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ELEMENTS_BOOL_I64_DENSE_ARRAY;
        };
    }

    private static Cpu1IndexKernelId scatterNdKernelId(
            DataType valueDataType,
            DataType indexDataType,
            Cpu1StorageKind storageKind
    ) {
        if (storageKind == Cpu1StorageKind.MEMORY_SEGMENT) {
            return switch (valueDataType) {
                case FLOAT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_F32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_F32_I64_DENSE_SEGMENT;
                case FLOAT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_F64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_F64_I64_DENSE_SEGMENT;
                case BFLOAT16 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_BF16_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_BF16_I64_DENSE_SEGMENT;
                case INT32 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_I32_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_I32_I64_DENSE_SEGMENT;
                case INT64 -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_I64_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_I64_I64_DENSE_SEGMENT;
                case BOOL -> indexDataType == DataType.INT32
                        ? Cpu1IndexKernelId.SCATTER_ND_BOOL_I32_DENSE_SEGMENT
                        : Cpu1IndexKernelId.SCATTER_ND_BOOL_I64_DENSE_SEGMENT;
            };
        }
        return switch (valueDataType) {
            case FLOAT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_F32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_F32_I64_DENSE_ARRAY;
            case FLOAT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_F64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_F64_I64_DENSE_ARRAY;
            case BFLOAT16 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_BF16_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_BF16_I64_DENSE_ARRAY;
            case INT32 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_I32_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_I32_I64_DENSE_ARRAY;
            case INT64 -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_I64_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_I64_I64_DENSE_ARRAY;
            case BOOL -> indexDataType == DataType.INT32
                    ? Cpu1IndexKernelId.SCATTER_ND_BOOL_I32_DENSE_ARRAY
                    : Cpu1IndexKernelId.SCATTER_ND_BOOL_I64_DENSE_ARRAY;
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

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = Math.multiplyExact(stride, shape[i]);
        }
        return strides;
    }

    private record ShapeFacts(
            int dimension,
            int axisSize,
            int innerSize,
            int outerSize,
            int indexElementCount,
            int indexAxisSize,
            int batchDims,
            int tupleRank,
            int prefixRank,
            int tupleStride,
            int updateElementCount,
            int[] gatherNdInputShape,
            int[] gatherNdInputStrides,
            int[] gatherNdIndicesDenseStrides,
            int[] gatherNdOutputShape,
            int[] gatherNdOutputDenseStrides
    ) {
        private ShapeFacts {
            gatherNdInputShape = gatherNdInputShape.clone();
            gatherNdInputStrides = gatherNdInputStrides.clone();
            gatherNdIndicesDenseStrides = gatherNdIndicesDenseStrides.clone();
            gatherNdOutputShape = gatherNdOutputShape.clone();
            gatherNdOutputDenseStrides = gatherNdOutputDenseStrides.clone();
        }
    }
}

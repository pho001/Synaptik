package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gatherAxisGrad;
import operations.index.gatherGrad;
import operations.index.ScatterReduction;
import operations.index.scatterAdd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.index.takeAlongAxisGrad;
import tensor.DataType;

import java.util.Arrays;

final class MetalIndexWriteSemantics {
    private MetalIndexWriteSemantics() {
    }

    static boolean isIndexWriteOrGradient(Operation.OpType opType) {
        return opType == Operation.OpType.SCATTER_ADD
                || opType == Operation.OpType.SCATTER_ELEMENTS
                || opType == Operation.OpType.SCATTER_ND
                || opType == Operation.OpType.GATHER_GRAD
                || opType == Operation.OpType.GATHER_AXIS_GRAD
                || opType == Operation.OpType.TAKE_ALONG_AXIS_GRAD;
    }

    static String unsupportedReason(CompiledNode node, PartitionPlanningContext context) {
        Operation.OpType opType = node.operation().opType();
        String common = commonReason(opType, node, context);
        if (!common.isBlank()) {
            return common;
        }
        return switch (opType) {
            case SCATTER_ADD -> scatterAddReason(node, context);
            case SCATTER_ELEMENTS -> scatterElementsReason(node, context);
            case SCATTER_ND -> scatterNdReason(node, context);
            case GATHER_GRAD -> gatherGradReason(node, context);
            case GATHER_AXIS_GRAD -> gatherAxisGradReason(node, context);
            case TAKE_ALONG_AXIS_GRAD -> takeAlongAxisGradReason(node, context);
            default -> "";
        };
    }

    private static String scatterAddReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scatterAdd op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD requires base, INT32 indices, and source inputs";
        }
        CompiledNode base = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        CompiledNode src = context.compiledNode(node.inputIds().get(2));
        if (base == null || indices == null || src == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("SCATTER_ADD", node, indices, base, src);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("SCATTER_ADD", base, indices, src);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] baseShape = base.shape();
        if (axis < 0 || axis >= baseShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD axis is outside base rank";
        }
        if (!Arrays.equals(node.shape(), baseShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD output shape must equal base shape";
        }
        int[] expected = reduceShape(baseShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(src.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ADD indices/source shape must equal base shape without scattered axis";
        }
        String boundsReason = staticBoundsReason("SCATTER_ADD", indices, baseShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String scatterElementsReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scatterElements op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS requires data, INT32 indices, and updates inputs";
        }
        CompiledNode data = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        CompiledNode updates = context.compiledNode(node.inputIds().get(2));
        if (data == null || indices == null || updates == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("SCATTER_ELEMENTS", node, indices, data, updates);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("SCATTER_ELEMENTS", data, indices, updates);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getAxis();
        int[] dataShape = data.shape();
        int[] indicesShape = indices.shape();
        int[] updatesShape = updates.shape();
        if (axis < 0 || axis >= dataShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS axis is outside data rank";
        }
        if (!Arrays.equals(node.shape(), dataShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS output shape must equal data shape";
        }
        if (indicesShape.length != dataShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS indices rank must match data rank";
        }
        if (!Arrays.equals(updatesShape, indicesShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS updates shape must equal indices shape";
        }
        for (int i = 0; i < indicesShape.length; i++) {
            if (i != axis && indicesShape[i] != dataShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ELEMENTS non-axis dimensions must match data shape";
            }
        }
        String boundsReason = staticBoundsReason("SCATTER_ELEMENTS", indices, dataShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return duplicateScatterElementsReason(node, indices, op.getReduction(), axis);
    }

    private static String scatterNdReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof scatterNd op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND descriptor is unavailable";
        }
        if (node.inputIds().size() != 3) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND requires data, INT32 indices, and updates inputs";
        }
        CompiledNode data = context.compiledNode(node.inputIds().get(0));
        CompiledNode indices = context.compiledNode(node.inputIds().get(1));
        CompiledNode updates = context.compiledNode(node.inputIds().get(2));
        if (data == null || indices == null || updates == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("SCATTER_ND", node, indices, data, updates);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("SCATTER_ND", data, indices, updates);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int[] dataShape = data.shape();
        int[] indicesShape = indices.shape();
        int[] updatesShape = updates.shape();
        int batchDims = op.getBatchDims();
        if (!Arrays.equals(node.shape(), dataShape)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND output shape must equal data shape";
        }
        if (indicesShape.length < 1) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND indices rank must be at least one";
        }
        if (batchDims < 0 || batchDims >= indicesShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND batchDims must be in [0, indices rank)";
        }
        if (batchDims > dataShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND batchDims cannot exceed data rank";
        }
        for (int i = 0; i < batchDims; i++) {
            if (indicesShape[i] != dataShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND batch dimensions must match data leading dimensions";
            }
        }
        int tupleRank = indicesShape[indicesShape.length - 1];
        if (tupleRank < 1 || tupleRank > dataShape.length - batchDims) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND final index dimension must be in [1, data rank - batchDims]";
        }
        String updatesShapeReason = scatterNdUpdatesShapeReason(dataShape, indicesShape, updatesShape, batchDims, tupleRank);
        if (!updatesShapeReason.isBlank()) {
            return updatesShapeReason;
        }
        String boundsReason = scatterNdBoundsReason(indices, dataShape, batchDims, tupleRank);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return duplicateScatterNdReason(node, indices, updates, op.getReduction(), batchDims, tupleRank);
    }

    private static String gatherGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof gatherGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("GATHER_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("GATHER_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] outputShape = node.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD axis is outside output rank";
        }
        int[] expected = reduceShape(outputShape, axis);
        if (!Arrays.equals(indices.shape(), expected) || !Arrays.equals(outGrad.shape(), expected)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_GRAD indices/outGrad shape must equal output shape without gathered axis";
        }
        String boundsReason = staticBoundsReason("GATHER_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String gatherAxisGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof gatherAxisGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("GATHER_AXIS_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("GATHER_AXIS_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getAxis();
        int[] outputShape = node.shape();
        int[] gradShape = outGrad.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD axis is outside output rank";
        }
        if (indices.shape().length != 1 || gradShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD supports 1-D index tensors that preserve value rank";
        }
        for (int i = 0; i < outputShape.length; i++) {
            int expected = i == axis ? indices.shape()[0] : outputShape[i];
            if (gradShape[i] != expected) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL GATHER_AXIS_GRAD outGrad shape must equal output shape with gathered axis replaced by index length";
            }
        }
        String boundsReason = staticBoundsReason("GATHER_AXIS_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String takeAlongAxisGradReason(CompiledNode node, PartitionPlanningContext context) {
        if (!(node.operation() instanceof takeAlongAxisGrad op)) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD descriptor is unavailable";
        }
        if (node.inputIds().size() != 2) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD requires INT32 indices and output-gradient inputs";
        }
        CompiledNode indices = context.compiledNode(node.inputIds().get(0));
        CompiledNode outGrad = context.compiledNode(node.inputIds().get(1));
        if (indices == null || outGrad == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD inputs are unavailable";
        }
        String dtypeReason = requireFloatingValueAndInt32Index("TAKE_ALONG_AXIS_GRAD", node, indices, outGrad);
        if (!dtypeReason.isBlank()) {
            return dtypeReason;
        }
        String layoutReason = requireDense("TAKE_ALONG_AXIS_GRAD", indices, outGrad);
        if (!layoutReason.isBlank()) {
            return layoutReason;
        }
        int axis = op.getDimension();
        int[] outputShape = node.shape();
        int[] gradShape = outGrad.shape();
        if (axis < 0 || axis >= outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD axis is outside output rank";
        }
        if (!Arrays.equals(indices.shape(), gradShape) || gradShape.length != outputShape.length) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD indices shape must equal outGrad shape and rank";
        }
        for (int i = 0; i < outputShape.length; i++) {
            if (i != axis && gradShape[i] != outputShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL TAKE_ALONG_AXIS_GRAD non-axis dimensions must match output shape";
            }
        }
        String boundsReason = staticBoundsReason("TAKE_ALONG_AXIS_GRAD", indices, outputShape[axis]);
        if (!boundsReason.isBlank()) {
            return boundsReason;
        }
        return "";
    }

    private static String commonReason(Operation.OpType opType, CompiledNode node, PartitionPlanningContext context) {
        if (node.backwardNode()) {
            return "BACKWARD_CONTEXT_UNSUPPORTED: " + opType + " nodes are not legal inside nested Metal backward regions";
        }
        if (context == null) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " requires planning context";
        }
        int[] shape = node.shape();
        if (shape.length < 1 || shape.length > 4) {
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL " + opType + " supports rank 1..4 tensors";
        }
        return "";
    }

    private static String scatterNdUpdatesShapeReason(
            int[] dataShape,
            int[] indicesShape,
            int[] updatesShape,
            int batchDims,
            int tupleRank
    ) {
        int expectedRank = indicesShape.length - 1 + dataShape.length - batchDims - tupleRank;
        if (updatesShape.length != expectedRank) {
            if (expectedRank == 0 && updatesShape.length == 1 && updatesShape[0] == 1) {
                return "";
            }
            return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND updates shape must equal indices.shape[:-1] plus data slice suffix";
        }
        int p = 0;
        for (int i = 0; i < indicesShape.length - 1; i++) {
            if (updatesShape[p++] != indicesShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND updates prefix shape must match indices prefix shape";
            }
        }
        for (int i = batchDims + tupleRank; i < dataShape.length; i++) {
            if (updatesShape[p++] != dataShape[i]) {
                return "UNSUPPORTED_RANK_OR_SHAPE: GPU_METAL SCATTER_ND updates suffix shape must match indexed data slice shape";
            }
        }
        return "";
    }

    private static String requireFloatingValueAndInt32Index(
            String opName,
            CompiledNode output,
            CompiledNode indices,
            CompiledNode... values
    ) {
        if (!isMetalFloatingDType(output.dataType())) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " output requires FLOAT32/BFLOAT16";
        }
        if (indices.dataType() != DataType.INT32) {
            return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " index input requires INT32";
        }
        for (CompiledNode value : values) {
            if (value != null && value.dataType() != output.dataType()) {
                return "UNSUPPORTED_DTYPE: GPU_METAL " + opName + " value inputs must match FLOAT32/BFLOAT16 output dtype";
            }
        }
        return "";
    }

    private static boolean isMetalFloatingDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.BFLOAT16;
    }

    private static String requireDense(String opName, CompiledNode... inputs) {
        for (CompiledNode input : inputs) {
            if (input != null && (!input.contiguous() || input.hasStorageOffset())) {
                return "UNSUPPORTED_LAYOUT: GPU_METAL " + opName + " inputs require dense layout";
            }
        }
        return "";
    }

    private static String scatterNdBoundsReason(CompiledNode indices, int[] dataShape, int batchDims, int tupleRank) {
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index bounds require a static INT32 leaf tensor";
        }
        int[] indexData = readableInt32Data("SCATTER_ND", indices);
        if (indexData == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index bounds require readable INT32 storage";
        }
        int[] indicesShape = indices.shape();
        int logicalElements = indices.flatDataSize();
        if (logicalElements < 0 || logicalElements > indexData.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index bounds cannot be proven from storage";
        }
        int tupleStride = denseStrides(indicesShape)[indicesShape.length - 1];
        int tupleCount = logicalElements / Math.max(1, indicesShape[indicesShape.length - 1]);
        for (int tuple = 0; tuple < tupleCount; tuple++) {
            int tupleBase = tuple * indicesShape[indicesShape.length - 1] * tupleStride;
            for (int d = 0; d < tupleRank; d++) {
                int dataDim = batchDims + d;
                int value = indexData[tupleBase + d * tupleStride];
                if (value < 0 || value >= dataShape[dataDim]) {
                    return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index " + value
                            + " is outside dimension " + dataDim + " size " + dataShape[dataDim];
                }
            }
        }
        return "";
    }

    private static String staticBoundsReason(String opName, CompiledNode indices, int axisSize) {
        if (axisSize <= 0) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " axis size must be positive";
        }
        if (!indices.leaf()) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds require a static INT32 leaf tensor";
        }
        int[] data = readableInt32Data(opName, indices);
        if (data == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds require readable INT32 storage";
        }
        int logicalElements = indices.flatDataSize();
        if (logicalElements < 0 || logicalElements > data.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index bounds cannot be proven from storage";
        }
        for (int i = 0; i < logicalElements; i++) {
            int value = data[i];
            if (value < 0 || value >= axisSize) {
                return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL " + opName + " index " + value + " is outside axis size " + axisSize;
            }
        }
        return "";
    }

    private static int[] readableInt32Data(String opName, CompiledNode indices) {
        if (!indices.leaf()) {
            return null;
        }
        try {
            return indices.semanticTensor().getInt32Data();
        } catch (RuntimeException ex) {
            return null;
        }
    }

    private static String duplicateScatterElementsReason(
            CompiledNode node,
            CompiledNode indices,
            ScatterReduction reduction,
            int axis
    ) {
        if (reduction != ScatterReduction.NONE) {
            return "";
        }
        int[] indexData = readableInt32Data("SCATTER_ELEMENTS", indices);
        if (indexData == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ELEMENTS index bounds require readable INT32 storage";
        }
        int[] dataShape = node.shape();
        int[] indicesShape = indices.shape();
        int[] indicesDense = denseStrides(indicesShape);
        int[] dataDense = denseStrides(dataShape);
        boolean[] seen = new boolean[flatSize(dataShape)];
        int total = indices.flatDataSize();
        if (total < 0 || total > indexData.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ELEMENTS index bounds cannot be proven from storage";
        }
        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int targetLogical = 0;
            for (int d = 0; d < indicesShape.length; d++) {
                int coord = rem / indicesDense[d];
                rem %= indicesDense[d];
                int targetCoord = d == axis ? indexData[logical] : coord;
                targetLogical += targetCoord * dataDense[d];
            }
            if (seen[targetLogical]) {
                return "UNSUPPORTED_DUPLICATE_INDEX: GPU_METAL SCATTER_ELEMENTS NONE reduction does not allow duplicate target indices";
            }
            seen[targetLogical] = true;
        }
        return "";
    }

    private static String duplicateScatterNdReason(
            CompiledNode node,
            CompiledNode indices,
            CompiledNode updates,
            ScatterReduction reduction,
            int batchDims,
            int tupleRank
    ) {
        if (reduction != ScatterReduction.NONE) {
            return "";
        }
        int[] indexData = readableInt32Data("SCATTER_ND", indices);
        if (indexData == null) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index bounds require readable INT32 storage";
        }
        int[] dataShape = node.shape();
        int[] dataDense = denseStrides(dataShape);
        int[] indicesShape = indices.shape();
        int[] indicesDense = denseStrides(indicesShape);
        int[] updatesShape = updates.shape();
        int[] updatesDense = denseStrides(updatesShape);
        int prefixRank = indicesShape.length - 1;
        int tupleStride = indicesDense[indicesShape.length - 1];
        boolean[] seen = new boolean[flatSize(dataShape)];
        int total = updates.flatDataSize();
        if (indices.flatDataSize() < 0 || indices.flatDataSize() > indexData.length) {
            return "UNSUPPORTED_BOUNDS_CHECK: GPU_METAL SCATTER_ND index bounds cannot be proven from storage";
        }
        for (int logical = 0; logical < total; logical++) {
            int rem = logical;
            int[] coords = new int[updatesShape.length];
            for (int d = 0; d < updatesShape.length; d++) {
                coords[d] = rem / updatesDense[d];
                rem %= updatesDense[d];
            }
            int indexBaseLogical = 0;
            for (int d = 0; d < prefixRank; d++) {
                indexBaseLogical += coords[d] * indicesDense[d];
            }
            int targetLogical = 0;
            for (int d = 0; d < batchDims; d++) {
                targetLogical += coords[d] * dataDense[d];
            }
            for (int d = 0; d < tupleRank; d++) {
                int targetCoord = indexData[indexBaseLogical + d * tupleStride];
                targetLogical += targetCoord * dataDense[batchDims + d];
            }
            for (int d = batchDims + tupleRank; d < dataShape.length; d++) {
                int coordIndex = prefixRank + d - batchDims - tupleRank;
                int updateCoord = coordIndex >= 0 && coordIndex < coords.length ? coords[coordIndex] : 0;
                targetLogical += updateCoord * dataDense[d];
            }
            if (seen[targetLogical]) {
                return "UNSUPPORTED_DUPLICATE_INDEX: GPU_METAL SCATTER_ND NONE reduction does not allow duplicate target indices";
            }
            seen[targetLogical] = true;
        }
        return "";
    }

    private static int[] reduceShape(int[] shape, int axis) {
        if (shape.length == 1) {
            return new int[]{1};
        }
        int[] reduced = new int[shape.length - 1];
        for (int i = 0, j = 0; i < shape.length; i++) {
            if (i != axis) {
                reduced[j++] = shape[i];
            }
        }
        return reduced;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static int flatSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
}

package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** Test-only conversion of shape fixtures into truthful schema-twenty logical descriptors. */
final class MetalTestProgram {
    private MetalTestProgram() {}

    static List<MetalMpsGraphProgram.ValueDescriptor> descriptors(
            int[] ranks, long[] dimensions, MetalMpsGraphProgram program) {
        if (dimensions.length != Math.multiplyExact(ranks.length, MetalMpsGraphProgram.MAX_RANK)) {
            throw new IllegalArgumentException("dimension fixture length");
        }
        DataType[] types = inferTypes(ranks.length, program);
        try {
            var result = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(ranks.length);
            for (int value = 0; value < ranks.length; value++) {
                long[] shape = new long[ranks[value]];
                System.arraycopy(dimensions, value * MetalMpsGraphProgram.MAX_RANK,
                        shape, 0, shape.length);
                result.add(new MetalMpsGraphProgram.ValueDescriptor(types[value], shape, false));
            }
            for (MetalMpsGraphProgram.Node node : program.nodes()) {
                boolean extraction = node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                        || node.kind() == MetalMpsGraphProgram.NodeKind.SLICE;
                if (!node.kind().isAffine() && !extraction) continue;
                var input = result.get(node.firstInputIndex());
                var output = result.get(node.outputIndex());
                LayoutDescriptor layout = viewLayout(node, input, output.dimensions());
                result.set(node.outputIndex(), new MetalMpsGraphProgram.ValueDescriptor(
                        output.dataType(), output.dimensions(), Optional.of(layout), false,
                        node.kind().isAffine() && layout.isView() || layout.hasZeroStride()));
            }
            return List.copyOf(result);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("descriptor fixture geometry overflows", overflow);
        }
    }

    private static LayoutDescriptor viewLayout(
            MetalMpsGraphProgram.Node node,
            MetalMpsGraphProgram.ValueDescriptor input,
            long[] outputShape) {
        LayoutDescriptor source = input.layout().orElseThrow();
        long[] sourceShape = input.dimensions();
        long[] strides = source.strides();
        long offset = source.storageOffset();
        long[] words = node.attributeWords();
        switch (node.kind()) {
            case RESHAPE -> {
                if (!source.isContiguous()) {
                    return LayoutDescriptor.contiguous(Shape.of(outputShape));
                }
                strides = LayoutDescriptor.contiguous(Shape.of(outputShape)).strides();
            }
            case EXPAND -> {
                long[] expanded = new long[outputShape.length];
                int padding = outputShape.length - sourceShape.length;
                for (int axis = 0; axis < sourceShape.length; axis++) {
                    expanded[axis + padding] =
                            sourceShape[axis] == 1L && outputShape[axis + padding] != 1L
                                    ? 0L : strides[axis];
                }
                strides = expanded;
            }
            case PERMUTE -> {
                long[] axes = node.attributeValues();
                long[] permuted = new long[axes.length];
                for (int axis = 0; axis < axes.length; axis++) {
                    permuted[axis] = strides[Math.toIntExact(axes[axis])];
                }
                strides = permuted;
            }
            case EXPAND_DIMS -> {
                int axis = node.axis();
                long[] expanded = new long[strides.length + 1];
                System.arraycopy(strides, 0, expanded, 0, axis);
                expanded[axis] = axis == strides.length ? 1L
                        : Math.multiplyExact(strides[axis], sourceShape[axis]);
                System.arraycopy(strides, axis, expanded, axis + 1, strides.length - axis);
                strides = expanded;
            }
            case SQUEEZE, SELECT -> {
                int axis = node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                        ? Math.toIntExact(words[0]) : node.axis();
                if (node.kind() == MetalMpsGraphProgram.NodeKind.SELECT) {
                    offset = Math.addExact(offset, Math.multiplyExact(words[1], strides[axis]));
                }
                long[] selected = new long[strides.length - 1];
                System.arraycopy(strides, 0, selected, 0, axis);
                System.arraycopy(strides, axis + 1, selected, axis, strides.length - axis - 1);
                strides = selected;
            }
            case SLICE -> {
                if (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.CROP_TO_SHAPE) {
                    int rank = Math.toIntExact(words[0]);
                    for (int axis = 0; axis < rank; axis++) {
                        offset = Math.addExact(offset,
                                Math.multiplyExact(words[2 + rank + axis], strides[axis]));
                    }
                } else {
                    int count = Math.toIntExact(words[0]);
                    for (int item = 0; item < count; item++) {
                        int axis = Math.toIntExact(words[1 + count * 2 + item]);
                        offset = Math.addExact(offset,
                                Math.multiplyExact(words[1 + item], strides[axis]));
                        strides[axis] = Math.multiplyExact(
                                strides[axis], words[1 + count * 3 + item]);
                    }
                }
            }
            default -> throw new IllegalArgumentException("not a logical view fixture");
        }
        return LayoutDescriptor.of(Shape.of(outputShape), strides, offset, true);
    }

    private static DataType[] inferTypes(int valueCount, MetalMpsGraphProgram program) {
        DataType[] result = new DataType[valueCount];
        int[] parent = new int[valueCount];
        for (int value = 0; value < valueCount; value++) parent[value] = value;
        for (MetalMpsGraphProgram.Node node : program.nodes()) {
            switch (node.kind()) {
                case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS ->
                        union(parent, node.firstInputIndex(), node.outputIndex());
                default -> {}
            }
        }
        for (MetalMpsGraphProgram.Node node : program.nodes()) {
            switch (node.kind()) {
                case INITIAL_STATE ->
                        require(result, parent, node.outputIndex(), DataType.INT64);
                case DROPOUT -> {
                    int[] inputs = node.inputs();
                    int[] outputs = node.outputs();
                    require(result, parent, inputs[0], DataType.FLOAT32);
                    require(result, parent, inputs[1], DataType.INT64);
                    require(result, parent, outputs[0], DataType.FLOAT32);
                    require(result, parent, outputs[1], DataType.BOOL);
                    require(result, parent, outputs[2], DataType.INT64);
                }
                case GATHER -> {
                    require(result, parent, node.firstInputIndex(), DataType.FLOAT32);
                    require(result, parent, node.secondInputIndex(), DataType.INT32);
                    require(result, parent, node.outputIndex(), DataType.FLOAT32);
                }
                case ONE_HOT -> {
                    require(result, parent, node.firstInputIndex(), DataType.INT32);
                    require(result, parent, node.outputIndex(), DataType.BOOL);
                }
                case SCATTER_ELEMENTS -> {
                    int[] inputs = node.inputs();
                    require(result, parent, inputs[0], DataType.FLOAT32);
                    require(result, parent, inputs[1], DataType.INT32);
                    require(result, parent, inputs[2], DataType.FLOAT32);
                    require(result, parent, node.outputIndex(), DataType.FLOAT32);
                }
                case GT, GE, LT, LE, EQ, NE -> {
                    require(result, parent, node.firstInputIndex(), DataType.FLOAT32);
                    require(result, parent, node.secondInputIndex(), DataType.FLOAT32);
                    require(result, parent, node.outputIndex(), DataType.BOOL);
                }
                case IS_FINITE, IS_NAN, IS_INF -> {
                    require(result, parent, node.firstInputIndex(), DataType.FLOAT32);
                    require(result, parent, node.outputIndex(), DataType.BOOL);
                }
                case LOGICAL_AND, LOGICAL_OR -> {
                    require(result, parent, node.firstInputIndex(), DataType.BOOL);
                    require(result, parent, node.secondInputIndex(), DataType.BOOL);
                    require(result, parent, node.outputIndex(), DataType.BOOL);
                }
                case LOGICAL_NOT -> {
                    require(result, parent, node.firstInputIndex(), DataType.BOOL);
                    require(result, parent, node.outputIndex(), DataType.BOOL);
                }
                case WHERE -> {
                    int[] inputs = node.inputs();
                    require(result, parent, inputs[0], DataType.BOOL);
                    require(result, parent, inputs[1], DataType.FLOAT32);
                    require(result, parent, inputs[2], DataType.FLOAT32);
                    require(result, parent, node.outputIndex(), DataType.FLOAT32);
                }
                default -> {
                    for (int input : node.inputs()) require(result, parent, input, DataType.FLOAT32);
                    for (int output : node.outputs()) require(result, parent, output, DataType.FLOAT32);
                }
            }
        }
        for (int value = 0; value < valueCount; value++) {
            int root = root(parent, value);
            if (result[root] == null) result[root] = DataType.FLOAT32;
            result[value] = result[root];
        }
        return result;
    }

    private static void require(DataType[] types, int[] parent, int value, DataType required) {
        if (value < 0 || value >= parent.length) {
            throw new IllegalArgumentException("fixture value index");
        }
        int root = root(parent, value);
        if (types[root] != null && types[root] != required) {
            throw new IllegalArgumentException("inconsistent fixture types");
        }
        types[root] = required;
    }

    private static void union(int[] parent, int left, int right) {
        if (left < 0 || left >= parent.length || right < 0 || right >= parent.length) {
            throw new IllegalArgumentException("fixture value index");
        }
        int leftRoot = root(parent, left);
        int rightRoot = root(parent, right);
        if (leftRoot != rightRoot) parent[rightRoot] = leftRoot;
    }

    private static int root(int[] parent, int value) {
        while (parent[value] != value) value = parent[value];
        return value;
    }
}

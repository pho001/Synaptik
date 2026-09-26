package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;

/** Test-only conversion of legacy shape fixtures into explicit schema-thirteen descriptors. */
final class MetalTestProgram {
    private MetalTestProgram() {}

    static List<MetalMpsGraphProgram.ValueDescriptor> descriptors(
            int[] ranks, long[] dimensions, MetalMpsGraphProgram program) {
        if (dimensions.length != Math.multiplyExact(ranks.length, MetalMpsGraphProgram.MAX_RANK)) {
            throw new IllegalArgumentException("dimension fixture length");
        }
        DataType[] types = inferTypes(ranks.length, program);
        var result = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(ranks.length);
        for (int value = 0; value < ranks.length; value++) {
            long[] shape = new long[ranks[value]];
            System.arraycopy(dimensions, value * MetalMpsGraphProgram.MAX_RANK,
                    shape, 0, shape.length);
            result.add(new MetalMpsGraphProgram.ValueDescriptor(types[value], shape, false));
        }
        return List.copyOf(result);
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

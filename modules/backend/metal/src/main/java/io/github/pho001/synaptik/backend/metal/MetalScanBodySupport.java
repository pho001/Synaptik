package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Optional;

/**
 * Physical qualification of genuine empty ordinary occurrences inside the bounded Scan binder.
 * This is not ordinary public empty/symbolic admission: original dimensions and all trailing
 * feature products are retained, no sample batch is substituted, and Model's ordinary descriptor
 * inference validates the complete family equations before backend carrier/geometry qualification.
 * Nonempty occurrences continue through the ordinary provider. Native construction independently
 * checks the same real dimensions and compiles nonempty empty-input identities through their math
 * owners; zero output cardinality alone never excuses an invalid index domain or sibling result.
 */
final class MetalScanBodySupport {
    private MetalScanBodySupport() {}

    /**
     * Qualifies the actual empty occurrence without erasing dimensions or reading values.
     * @param query ordinary local occurrence with at least one empty input or output domain
     * @return whether its complete static equations and native carrier/shape bounds are eligible
     */
    static boolean supports(OperationCapabilityQuery query) {
        try {
            if (query.scanBody().isPresent() || query.scanActivityPlan().isPresent()) return false;
            boolean empty = false;
            for (var descriptor : concat(query.inputs(), query.outputs())) {
                if (!descriptor.shape().isFullyStatic() || descriptor.shape().rank() > 16) return false;
                if (!carrier(descriptor.dataType())) return false;
                long trailing = 1;
                for (int a = descriptor.shape().rank() - 1; a >= 0; a--) {
                    long extent = descriptor.shape().dimension(a).staticSize().orElseThrow();
                    empty |= extent == 0;
                    trailing = Math.multiplyExact(trailing, extent);
                }
                Math.multiplyExact(trailing, descriptor.dataType().byteWidth());
            }
            if (!empty) return false;
            var inferred = ScanBody.describeOrdinary(query.operation(), query.inputs(), query.outputs().size(), Optional.empty());
            for (int p = 0; p < inferred.size(); p++) {
                var actual = query.outputs().get(p); var expected = inferred.get(p);
                if (actual.dataType() != expected.dataType() || !actual.shape().equals(expected.shape())
                        || actual.requiresGrad() != expected.requiresGrad()) return false;
            }
            int[] inputs = java.util.stream.IntStream.range(0, query.inputs().size()).toArray();
            int[] outputs = java.util.stream.IntStream.range(query.inputs().size(), query.inputs().size() + query.outputs().size()).toArray();
            var node = MetalNegPartitionPreparer.lower(query.operation(), inputs, outputs, query.inputs(), query.outputs());
            var in = query.inputs(); var out = query.outputs();
            return switch (node.kind()) {
                case NEG, ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT,
                        FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, GELU, GELU_TANH_APPROXIMATION, SILU,
                        SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW, SCALAR_MIN, SCALAR_MAX, CLAMP,
                        ADD, SUB, MUL, DIV, TENSOR_POW -> floats(in) && floats(out);
                case TENSOR_MIN, TENSOR_MAX, REDUCTION_MIN, REDUCTION_MAX -> numeric(in) && sameTypes(in, out);
                case SUM, MEAN, PROD, LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM ->
                        (floats(in) && floats(out)) || node.kind() == MetalMpsGraphProgram.NodeKind.PROD
                                && integers(in) && sameTypes(in, out);
                case CUM_SUM, CUM_PROD, MATMUL -> floats(in) && floats(out);
                case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS,
                        SELECT, PAD, SLICE, SLICE_UPDATE, CONCAT, STACK, TILE, UNFOLD_AXIS -> sameTypes(in, out);
                case FOLD_AXIS, UNFOLD2D, FOLD2D, UNFOLD3D, FOLD3D -> floats(in) && floats(out);
                case GT, GE, LT, LE, EQ, NE -> numeric(in) && bools(out);
                case IS_FINITE, IS_NAN, IS_INF -> floats(in) && bools(out);
                case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, ALL, ANY -> bools(in) && bools(out);
                case CAST -> in.size() == 1 && out.size() == 1;
                case WHERE -> in.getFirst().dataType() == DataType.BOOL
                        && floats(in.subList(1, in.size())) && floats(out);
                case GATHER, GATHER_ELEMENTS, GATHER_ND -> integer(in.get(1).dataType())
                        && in.getFirst().dataType() == out.getFirst().dataType();
                case ONE_HOT -> integers(in) && bools(out);
                case SCATTER_ADD, SCATTER_ELEMENTS, SCATTER_ND -> integer(in.get(1).dataType())
                        && in.getFirst().dataType() == in.get(2).dataType()
                        && in.getFirst().dataType() == out.getFirst().dataType()
                        && (node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD ? floats(out)
                                : node.attributeWords()[1] == 0 || floats(out));
                case SOFTMAX, LOG_SOFTMAX, LAYER_NORM, RMS_NORM, BATCH_NORM_INFERENCE, BATCH_NORM_TRAINING,
                        MEAN_SQUARED_ERROR, DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> floats(in) && floats(out);
                case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> floats(in.subList(0, 1))
                        && integer(in.get(1).dataType()) && floats(out);
                case SCALED_DOT_PRODUCT_ATTENTION -> floats(in.subList(0, 3))
                        && (in.size() == 3 || in.get(3).dataType() == DataType.BOOL) && floats(out);
                case CONV2D, CONV3D, MAX_POOL2D, AVERAGE_POOL2D, MAX_POOL3D, AVERAGE_POOL3D ->
                        floats(in) && floats(out) && uint32Geometry(in) && uint32Geometry(out);
                case SORT -> in.getFirst().dataType() == out.getFirst().dataType() && uint32Geometry(in);
                case ARGSORT, ARG_MAX, ARG_MIN -> out.getFirst().dataType() == DataType.INT64 && uint32Geometry(in);
                case TOP_K -> in.getFirst().dataType() == out.getFirst().dataType()
                        && out.get(1).dataType() == DataType.INT64 && uint32Geometry(in);
                default -> false;
            };
        } catch (IllegalArgumentException | ArithmeticException | java.util.NoSuchElementException unsupported) {
            return false;
        }
    }

    private static List<TensorDescriptor> concat(List<TensorDescriptor> a, List<TensorDescriptor> b) {
        var result = new java.util.ArrayList<>(a); result.addAll(b); return result;
    }
    private static boolean carrier(DataType type) { return type == DataType.FLOAT32 || type == DataType.BOOL || integer(type); }
    private static boolean integer(DataType type) { return type == DataType.INT32 || type == DataType.INT64; }
    private static boolean floats(List<TensorDescriptor> values) { return values.stream().allMatch(v -> v.dataType() == DataType.FLOAT32); }
    private static boolean bools(List<TensorDescriptor> values) { return values.stream().allMatch(v -> v.dataType() == DataType.BOOL); }
    private static boolean integers(List<TensorDescriptor> values) { return values.stream().allMatch(v -> integer(v.dataType())); }
    private static boolean numeric(List<TensorDescriptor> values) { return values.stream().allMatch(v -> v.dataType() == DataType.FLOAT32 || integer(v.dataType())); }
    private static boolean sameTypes(List<TensorDescriptor> in, List<TensorDescriptor> out) {
        DataType type = in.getFirst().dataType();
        return in.stream().allMatch(v -> v.dataType() == type) && out.stream().allMatch(v -> v.dataType() == type);
    }
    private static boolean uint32Geometry(List<TensorDescriptor> descriptors) {
        for (var descriptor : descriptors) for (var dimension : descriptor.shape().dimensions())
            if (dimension.staticSize().orElseThrow() > 0xffffffffL) return false;
        return true;
    }
}

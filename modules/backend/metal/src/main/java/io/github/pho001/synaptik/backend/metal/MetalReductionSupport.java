package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Arrays;
import java.util.List;

/**
 * Owns exact-carrier reduction admission and independent encoded ordinary-reduction validation.
 *
 * <p>Only stored-value extrema use FLOAT64: no FLOAT64 arithmetic or low arithmetic extension is
 * inferred from a FLOAT32 numerical reduction. Validation is cold and precedes resource creation.
 */
final class MetalReductionSupport {
    private static final long MAX_BUFFER_BYTES = 0xffff_ffffL;

    /** Prevents construction of the cold, stateless reduction proof owner. */
    private MetalReductionSupport() {}

    /**
     * Proves only the audit's additional exact-carrier Model occurrences.
     *
     * @param operation typed ordinary operation whose attributes determine contributor membership
     * @param inputs ordered input descriptors, exactly one for these unmasked forms
     * @param outputs output descriptors, exactly one canonical same-carrier result
     * @return whether integral SUM or integral/FLOAT64 MIN/MAX has a complete static physical proof
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (inputs.size() != 1 || outputs.size() != 1
                || !(operation.kind() instanceof AggregateReductionKind kind)) return false;
        var input = inputs.getFirst(); var output = outputs.getFirst();
        DataType type = input.dataType();
        boolean integral = type == DataType.INT32 || type == DataType.INT64;
        if (input.dataType() != output.dataType() || input.requiresGrad() || output.requiresGrad()
                || !((kind == AggregateReductionKind.SUM && integral)
                        || ((kind == AggregateReductionKind.MIN || kind == AggregateReductionKind.MAX)
                                && (integral || type == DataType.FLOAT64)))) return false;
        if (!input.shape().isFullyStatic() || !output.shape().isFullyStatic()) return false;
        long[] source = input.shape().toLongArray(), target = output.shape().toLongArray();
        try {
            if (!physical(input.layout().orElse(null), source, type)
                    || !physical(output.layout().orElse(null), target, type)
                    || !canonical(output.layout().orElse(null), target)) return false;
            if (operation.attrs() instanceof SumToShapeAttrs attrs) {
                return kind == AggregateReductionKind.SUM && attrs.targetShape().equals(output.shape())
                        && sumTarget(source, target);
            }
            boolean[] axes = new boolean[source.length]; boolean keep = false;
            if (operation.attrs() == NoOperationAttrs.INSTANCE) Arrays.fill(axes, true);
            else if (operation.attrs() instanceof AxisReductionAttrs attrs) {
                select(axes, attrs.axis()); keep = attrs.keepDimensions();
            } else if (operation.attrs() instanceof MultiAxisReductionAttrs attrs) {
                for (int axis : attrs.axes()) select(axes, axis);
                keep = attrs.keepDimensions();
            } else return false;
            return Arrays.equals(reduced(source, axes, keep), target);
        } catch (IllegalArgumentException | ArithmeticException invalid) { return false; }
    }

    /**
     * Identifies the new native reduction routes without claiming a descriptor is valid.
     *
     * @param node encoded node with already checked value references
     * @param values the entire encoded value descriptor table
     * @return whether this owner must independently validate the FLOAT32 or additional exact route
     */
    static boolean handles(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.firstInputIndex() == MetalMpsGraphProgram.NO_SECOND_INPUT) return false;
        DataType type = values.get(node.firstInputIndex()).dataType();
        return switch (node.kind()) {
            case SUM -> type == DataType.FLOAT32 || type == DataType.INT32 || type == DataType.INT64;
            case REDUCTION_MIN, REDUCTION_MAX -> type == DataType.FLOAT32 || type == DataType.FLOAT64
                    || type == DataType.INT32 || type == DataType.INT64;
            case MEAN, PROD, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM, LOG_SUM_EXP ->
                    type == DataType.FLOAT32;
            default -> false;
        };
    }

    /**
     * Independently proves encoded grammar, types, gradient roles, physical reads and result shape.
     *
     * @param node encoded reduction whose references have been range checked by the program owner
     * @param values complete descriptor table; no provider query or typed operation is consulted
     * @throws IllegalArgumentException if any grammar, carrier, count, mask or layout fact disagrees
     */
    static void validate(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        validate(node, values, false);
    }

    /**
     * Validates an authenticated Scan-local reduction without projecting empty axes to one.
     *
     * <p>Only canonical empty FLOAT32 SUM/MEAN or statistical local values use the empty-layout
     * exception. Every axis, shape, mask and correction check remains identical to ordinary
     * validation; a zero statistical contributor count still violates the required denominator.
     *
     * @param node Scan-body reduction with range-checked references
     * @param values complete Scan-local descriptor table with real resolved dimensions
     * @throws IllegalArgumentException if any reduction fact or empty-local restriction disagrees
     */
    static void validateScan(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        validate(node, values, true);
    }

    /**
     * Shares the entire independent reduction proof across ordinary and local-Scan boundaries.
     *
     * @param node encoded reduction with checked references
     * @param values actual descriptor table
     * @param scanLocal whether canonical empty local FLOAT32 reductions are authorized
     * @throws IllegalArgumentException if the encoded operation is not valid in the selected domain
     */
    private static void validate(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values, boolean scanLocal) {
        if (!handles(node, values) || node.outputs().length != 1) throw invalid();
        var input = values.get(node.firstInputIndex()); var output = values.get(node.outputIndex());
        DataType type = input.dataType(); long[] source = input.dimensions(), target = output.dimensions();
        boolean allowEmpty = scanLocal && type == DataType.FLOAT32
                && (node.kind() == MetalMpsGraphProgram.NodeKind.SUM
                        || node.kind() == MetalMpsGraphProgram.NodeKind.MEAN
                        || node.kind() == MetalMpsGraphProgram.NodeKind.VARIANCE
                        || node.kind() == MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION);
        boolean masked = node.attributeKind() == MetalMpsGraphProgram.AttributeKind.MASKED_REDUCTION;
        if (node.inputs().length != (masked ? 2 : 1) || output.dataType() != type
                || input.requiresGrad() != output.requiresGrad()
                || (type != DataType.FLOAT32 && (input.requiresGrad() || output.requiresGrad()))
                || !physical(input.layout().orElse(null), source, type, allowEmpty)
                || !physical(output.layout().orElse(null), target, type, allowEmpty)
                || !canonical(output.layout().orElse(null), target)) throw invalid();
        boolean[] axes = new boolean[source.length]; boolean keep = false;
        long correction = 0; long[] words = node.attributeWords();
        if (masked) {
            if ((node.kind() != MetalMpsGraphProgram.NodeKind.SUM
                    && node.kind() != MetalMpsGraphProgram.NodeKind.MEAN)
                    || type != DataType.FLOAT32 || words.length != 1) throw invalid();
            select(axes, Math.toIntExact(words[0]));
            var mask = values.get(node.secondInputIndex());
            if (mask.dataType() != DataType.BOOL || mask.requiresGrad()
                    || !physical(mask.layout().orElse(null), mask.dimensions(), DataType.BOOL, allowEmpty)
                    || !broadcastMask(source, mask.dimensions())) throw invalid();
        } else if (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.STATISTICAL_REDUCTION) {
            if ((node.kind() != MetalMpsGraphProgram.NodeKind.VARIANCE
                    && node.kind() != MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION)
                    || words.length < 3 || words[0] < 0 || words[0] > 16
                    || words.length != words[0] + 3) throw invalid();
            int count = (int) words[0];
            for (int index = 0; index < count; index++) select(axes, Math.toIntExact(words[index + 1]));
            keep = bool(words[count + 1]); correction = words[count + 2];
            if (correction < 0) throw invalid();
        } else if (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.REDUCTION) {
            if (words.length < 3 || words[2] < 0 || words[2] > 16 || words.length != words[2] + 3)
                throw invalid();
            keep = bool(words[1]); int count = (int) words[2];
            if (words[0] == 4) {
                if (node.kind() != MetalMpsGraphProgram.NodeKind.SUM || keep || count != target.length
                        || !Arrays.equals(Arrays.copyOfRange(words, 3, words.length), target)
                        || !sumTarget(source, target)) throw invalid();
                return;
            }
            if (words[0] == 1) {
                if (keep || count != 0) throw invalid();
                Arrays.fill(axes, true);
            } else {
                if ((words[0] != 2 && words[0] != 3) || (words[0] == 2 && count != 1)) throw invalid();
                for (int index = 0; index < count; index++) select(axes, Math.toIntExact(words[index + 3]));
            }
            if ((node.kind() == MetalMpsGraphProgram.NodeKind.L1_NORM
                    || node.kind() == MetalMpsGraphProgram.NodeKind.L2_NORM
                    || node.kind() == MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP) && words[0] != 3)
                throw invalid();
            if (node.kind() == MetalMpsGraphProgram.NodeKind.VARIANCE
                    || node.kind() == MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION) throw invalid();
        } else throw invalid();
        if (!Arrays.equals(reduced(source, axes, keep), target)) throw invalid();
        if (node.kind() == MetalMpsGraphProgram.NodeKind.VARIANCE
                || node.kind() == MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION) {
            long count = 1;
            for (int axis = 0; axis < source.length; axis++) if (axes[axis])
                count = Math.multiplyExact(count, source[axis]);
            if (count <= correction) throw invalid();
        }
    }

    /**
     * Extends the ordinary physical proof only for explicitly authorized canonical empty locals.
     *
     * @param layout optional concrete local layout
     * @param shape actual resolved extents, with no negative dimensions
     * @param type original carrier, FLOAT32 or BOOL for the empty-local case
     * @param allowEmpty whether the caller has established the Scan-local reduction exception
     * @return whether the ordinary span is valid or the exact canonical local storage is empty
     * @throws ArithmeticException if a nonempty physical span overflows
     */
    private static boolean physical(LayoutDescriptor layout, long[] shape, DataType type,
            boolean allowEmpty) {
        boolean empty = false;
        for (long extent : shape) {
            if (extent < 0) return false;
            empty |= extent == 0;
        }
        if (!empty) return physical(layout, shape, type);
        return allowEmpty && shape.length <= 16
                && (type == DataType.FLOAT32 || type == DataType.BOOL)
                && canonical(layout, shape) && layout.referencedElementSpan() == 0;
    }

    /**
     * Checks a resolved nonnegative affine span, including internal broadcast reads.
     *
     * @param layout optional concrete layout, rejected when absent
     * @param shape positive static logical extents
     * @param type original storage carrier, determining the native byte bound
     * @return whether the exact referenced span fits the existing custom-buffer bound
     * @throws ArithmeticException if a derived span or byte extent overflows
     */
    private static boolean physical(LayoutDescriptor layout, long[] shape, DataType type) {
        if (shape.length > 16 || layout == null || layout.rank() != shape.length
                || layout.storageOffset() < 0) return false;
        long span = Math.addExact(layout.storageOffset(), 1);
        long[] strides = layout.strides();
        for (int axis = 0; axis < shape.length; axis++) {
            if (shape[axis] <= 0 || strides[axis] < 0) return false;
            span = Math.addExact(span, Math.multiplyExact(shape[axis] - 1, strides[axis]));
        }
        return span == layout.referencedElementSpan()
                && Math.multiplyExact(span, type.byteWidth()) <= MAX_BUFFER_BYTES;
    }

    /**
     * Requires a canonical materializing target, never an affine alias.
     *
     * @param layout optional concrete output layout
     * @param shape static result extents
     * @return whether the layout exactly equals the ordinary contiguous non-view descriptor
     */
    private static boolean canonical(LayoutDescriptor layout, long[] shape) {
        return layout != null && layout.equals(LayoutDescriptor.contiguous(Shape.of(shape)));
    }

    /**
     * Proves right-aligned BOOL membership broadcasting without expanded storage.
     *
     * @param input logical data extents
     * @param mask logical mask extents
     * @return whether broadcasting produces exactly the data shape
     */
    private static boolean broadcastMask(long[] input, long[] mask) {
        if (mask.length > input.length) return false;
        int prefix = input.length - mask.length;
        for (int axis = 0; axis < mask.length; axis++)
            if (mask[axis] != 1 && mask[axis] != input[prefix + axis]) return false;
        return true;
    }

    /**
     * Proves sum-to-Shape's leading and singleton-axis membership.
     *
     * @param source static input extents
     * @param target exact requested output extents
     * @return whether every aligned extent is one or the corresponding input extent
     */
    private static boolean sumTarget(long[] source, long[] target) {
        if (target.length > source.length) return false;
        int prefix = source.length - target.length;
        for (int axis = 0; axis < target.length; axis++)
            if (target[axis] != 1 && target[axis] != source[prefix + axis]) return false;
        return true;
    }

    /**
     * Selects each normalized axis once.
     *
     * @param axes mutable cold selection bitmap
     * @param axis normalized input-axis ordinal
     * @throws IllegalArgumentException if the axis is repeated or outside the bitmap
     */
    private static void select(boolean[] axes, int axis) {
        if (axis < 0 || axis >= axes.length || axes[axis]) throw invalid();
        axes[axis] = true;
    }

    /**
     * Constructs cold expected geometry with kept axes in original order.
     *
     * @param input static input extents
     * @param axes exact selected-axis bitmap matching input rank
     * @param keep whether selected axes remain with extent one
     * @return the expected result extents
     */
    private static long[] reduced(long[] input, boolean[] axes, boolean keep) {
        int rank = input.length;
        if (!keep) for (boolean axis : axes) if (axis) rank--;
        long[] result = new long[rank];
        for (int axis = 0, output = 0; axis < input.length; axis++) {
            if (!axes[axis]) result[output++] = input[axis];
            else if (keep) result[output++] = 1;
        }
        return result;
    }

    /**
     * Decodes only authenticated Boolean words zero and one.
     *
     * @param word encoded attribute word
     * @return the exact Boolean value
     * @throws IllegalArgumentException if the word is neither zero nor one
     */
    private static boolean bool(long word) {
        if (word != 0 && word != 1) throw invalid();
        return word != 0;
    }

    /**
     * Creates the uniform cold malformed-reduction failure.
     *
     * @return a fresh exception with no backend resource side effects
     */
    private static IllegalArgumentException invalid() {
        return new IllegalArgumentException("invalid ordinary Metal reduction");
    }
}

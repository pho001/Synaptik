package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Owns original-carrier numerical reduction admission and independent encoded validation.
 *
 * <p>Only stored-value extrema use FLOAT64: no FLOAT64 arithmetic or low arithmetic extension is
 * inferred from a FLOAT32 numerical reduction. Validation is cold and precedes resource creation.
 */
final class MetalReductionSupport {
    private static final long MAX_BUFFER_BYTES = 0xffff_ffffL;

    /** Prevents construction of the cold, stateless reduction proof owner. */
    private MetalReductionSupport() {}

    /** Classifies numerical reductions without intercepting arg-extrema or logical reducers. */
    static boolean handles(Operation operation) {
        return operation.kind() instanceof AggregateReductionKind kind && switch (kind) {
            case SUM, MEAN, PROD, MIN, MAX, LOG_SUM_EXP, VARIANCE,
                    STANDARD_DEVIATION, L1_NORM, L2_NORM -> true;
            default -> false;
        };
    }

    /**
     * Proves original-carrier reduction occurrences through the same encoded geometry owner.
     *
     * <p>The ordinary lowerer preserves actual attributes; this method does not maintain a
     * second mask, correction, axis, byte-bound or storage-injectivity predicate.
     *
     * @param operation typed ordinary reduction
     * @param inputs ordered data and optional BOOL membership descriptors
     * @param outputs one canonical same-carrier result descriptor
     * @return whether the complete encoded occurrence belongs to this owner's supported domain
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (inputs.isEmpty() || outputs.size() != 1 || !handles(operation)) return false;
        try {
            var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(inputs.size() + 1);
            for (TensorDescriptor input : inputs) {
                if (!input.shape().isFullyStatic()) return false;
                values.add(MetalMpsGraphProgram.ValueDescriptor.from(input));
            }
            if (!outputs.getFirst().shape().isFullyStatic()) return false;
            values.add(MetalMpsGraphProgram.ValueDescriptor.from(outputs.getFirst()));
            var node = MetalNegPartitionPreparer.lowerScanBody(operation,
                    IntStream.range(0, inputs.size()).toArray(), new int[] {inputs.size()},
                    inputs, outputs);
            if (!handles(node, values)) return false;
            validate(node, values);
            return true;
        } catch (IllegalArgumentException | ArithmeticException invalid) {
            return false;
        }
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
     * <p>Canonical empty FLOAT32 local values use the empty-layout exception. Every axis,
     * shape, mask and correction check remains identical to ordinary validation. Empty SUM/norms
     * produce positive zero, PROD one, MEAN NaN, LSE negative infinity and extrema their signed
     * infinity identities; a zero statistical count still violates the required denominator.
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
        boolean allowEmpty = scanLocal && type == DataType.FLOAT32;
        boolean masked = node.attributeKind() == MetalMpsGraphProgram.AttributeKind.MASKED_REDUCTION;
        if (node.inputs().length != (masked ? 2 : 1) || output.dataType() != type
                || input.requiresGrad() != output.requiresGrad()
                || (type != DataType.FLOAT32 && (input.requiresGrad() || output.requiresGrad()))
                || !physical(input, allowEmpty)
                || !physical(output, allowEmpty)
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
                    || !physical(mask, allowEmpty)
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
     * Reuses the common broadcast-aware stored-read proof without allocating a second descriptor.
     *
     * <p>Nonbroadcast axes must be injective and zero strides require a truthful view descriptor.
     * Canonical empty values are admitted only by the explicit Scan-local caller flag.
     *
     * @param value encoded descriptor whose original carrier and layout remain authoritative
     * @param allowEmpty whether this caller has established the Scan-local empty exception
     * @return whether the common physical proof and existing custom byte bound both hold
     * @throws ArithmeticException if a descriptor byte extent overflows
     */
    private static boolean physical(MetalMpsGraphProgram.ValueDescriptor value, boolean allowEmpty) {
        if (value.elementCount() == 0) {
            return allowEmpty && value.rank() <= 16
                    && (value.dataType() == DataType.FLOAT32 || value.dataType() == DataType.BOOL)
                    && canonical(value.layout().orElse(null), value.dimensions())
                    && value.byteCount() == 0;
        }
        return value.supportsStorageLayout(0) && value.byteCount() <= MAX_BUFFER_BYTES;
    }

    /**
     * Adapts typed provider descriptors to the same common raw physical proof.
     *
     * @param layout optional concrete storage layout
     * @param shape positive static logical extents
     * @param type original storage carrier
     * @return whether the same broadcast-aware proof used by raw validation succeeds
     * @throws ArithmeticException if a derived descriptor byte extent overflows
     * @throws IllegalArgumentException if the descriptor geometry is invalid
     */
    private static boolean physical(LayoutDescriptor layout, long[] shape, DataType type) {
        return physical(new MetalMpsGraphProgram.ValueDescriptor(type, shape,
                java.util.Optional.ofNullable(layout), false, false), false);
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

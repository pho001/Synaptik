package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free multi-axis reduction expressions.
 *
 * <p>This package-private boundary owns ordered axis normalization, duplicate detection,
 * Shape derivation, static statistical-domain validation, exact attributes, descriptor, and
 * one-input provenance construction. Caller arrays are cloned before normalization and never
 * retained. An empty axis array selects a point domain at every input position; it is distinct
 * from an ordinary full reduction.</p>
 *
 * <p>Construction records mathematical requirements but reads no values or storage, evaluates no
 * reduction, chooses no numerical algorithm, creates no gradient rule, captures no graph, and
 * provides no compiler, backend, runtime, or execution behavior.</p>
 */
final class TensorMultiAxisReductionExpressions {
    /** Prevents instantiation because multi-axis construction is stateless. */
    private TensorMultiAxisReductionExpressions() {
    }

    /**
     * Creates one ordinary multi-axis reduction occurrence.
     *
     * <p>Validation checks input, kind, and axes for null; supported kind; and kind-specific input
     * type before cloning and normalizing axes. Normalized duplicates fail immediately in caller
     * order. The result preserves exact input type and gradient eligibility, derives removal or
     * retention Shape, leaves layout unresolved, and records one exact
     * {@link MultiAxisReductionAttrs} occurrence. Every pre-factory failure consumes no Tensor
     * identity.</p>
     *
     * @param input the non-null sole provenance input, retained by exact reference and not mutated
     * @param kind one of SUM, MEAN, PROD, MIN, MAX, ALL, or ANY
     * @param axes the non-null caller-owned positive or negative axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @return the non-null fresh, unlabeled, storage-free Tensor with derived Shape, exact input
     *     type/eligibility, unresolved layout, one-input provenance, and output index zero
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, checked
     *     in that order with the parameter name as message
     * @throws IllegalArgumentException if the kind or input type is unsupported or normalized axes
     *     repeat; kind/type failures precede axis handling
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyOrdinary(
            Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions) {
        return applyOrdinaryChecked(input, kind, axes, keepDimensions, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input the non-null sole provenance input, retained by exact reference and not mutated
     * @param kind one of SUM, MEAN, PROD, MIN, MAX, ALL, or ANY
     * @param axes the non-null caller-owned positive or negative axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, checked
     *     in that order with the parameter name as message
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if the kind or input type is unsupported or normalized axes
     *     repeat; kind/type failures precede axis handling
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyOrdinaryChecked(
            Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        return applyOrdinaryDescriptors(
                input.descriptor(), kind, axes, keepDimensions, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axes ordinary construction argument, validated in existing order
     * @param keepDimensions ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyOrdinaryDescriptors(
            TensorDescriptor input,
            AggregateReductionKind kind,
            int[] axes,
            boolean keepDimensions,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        validateOrdinaryKind(kind);
        validateOrdinaryInput(input, kind);
        int[] copiedAxes = axes.clone();
        List<Integer> normalizedAxes = normalizeAxes(input.shape(), copiedAxes);
        Shape resultShape = reduceShape(input.shape(), normalizedAxes, keepDimensions);
        return createDescriptors(input, kind, new MultiAxisReductionAttrs(normalizedAxes, keepDimensions), resultShape, descriptorCheck, inputTensor);
    }

    /**
     * Creates one floating log-sum-exp, L1-norm, or L2-norm multi-axis occurrence.
     *
     * @param input the non-null floating sole provenance input
     * @param kind LOG_SUM_EXP, L1_NORM, or L2_NORM
     * @param axes the non-null caller-owned axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @return the non-null fresh storage-free result preserving exact floating type and eligibility
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, in order
     * @throws IllegalArgumentException if {@code kind} is not supported, the input is not floating,
     *     or normalized axes repeat; these failures occur before identity allocation
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyAdvanced(
            Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions) {
        return applyAdvancedChecked(input, kind, axes, keepDimensions, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input the non-null floating sole provenance input
     * @param kind LOG_SUM_EXP, L1_NORM, or L2_NORM
     * @param axes the non-null caller-owned axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, in order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if {@code kind} is not supported, the input is not floating,
     *     or normalized axes repeat; these failures occur before identity allocation
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyAdvancedChecked(
            Tensor input, AggregateReductionKind kind, int[] axes, boolean keepDimensions,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        return applyAdvancedDescriptors(
                input.descriptor(), kind, axes, keepDimensions, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axes ordinary construction argument, validated in existing order
     * @param keepDimensions ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyAdvancedDescriptors(
            TensorDescriptor input,
            AggregateReductionKind kind,
            int[] axes,
            boolean keepDimensions,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        validateAdvancedKind(kind);
        validateFloating(input);
        int[] copiedAxes = axes.clone();
        List<Integer> normalizedAxes = normalizeAxes(input.shape(), copiedAxes);
        Shape resultShape = reduceShape(input.shape(), normalizedAxes, keepDimensions);
        return createDescriptors(input, kind, new MultiAxisReductionAttrs(normalizedAxes, keepDimensions), resultShape, descriptorCheck, inputTensor);
    }

    /**
     * Creates one corrected floating variance or standard-deviation occurrence.
     *
     * <p>After common null, kind, and floating-input validation, negative correction fails before
     * axis normalization. Static selected-domain count must be greater than correction; dynamic
     * count proof is deferred. Every local failure precedes attributes and identity allocation.</p>
     *
     * @param input the non-null floating sole provenance input
     * @param kind VARIANCE or STANDARD_DEVIATION
     * @param axes the non-null caller-owned axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @param correction non-negative value subtracted from selected-domain count
     * @return the non-null fresh storage-free result preserving exact floating type and eligibility
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, in order
     * @throws IllegalArgumentException if kind/type/correction/duplicate-axis validation fails, or
     *     if a static selected-domain count is not greater than correction
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyStatistical(
            Tensor input,
            AggregateReductionKind kind,
            int[] axes,
            boolean keepDimensions,
            long correction) {
        return applyStatisticalChecked(input, kind, axes, keepDimensions, correction, null);
    }

    /**
     * Shares ordinary semantic guards with descriptor-only scan validation. No Tensor is created
     * when the synchronous, unretained sink is non-null; null preserves ordinary construction.
     * Only the null-sink path consumes IDs or can encounter allocator exhaustion.
     * @param input the non-null floating sole provenance input
     * @param kind VARIANCE or STANDARD_DEVIATION
     * @param axes the non-null caller-owned axes; may be empty and is cloned
     * @param keepDimensions whether selected axes remain with extent one
     * @param correction non-negative value subtracted from selected-domain count
     * @param descriptorCheck optional consumer of validated semantics and ordered descriptors
     * @return null after descriptor-only completion, otherwise the ordinary canonical result
     * @throws NullPointerException if {@code input}, {@code kind}, or {@code axes} is null, in order
     * @throws IllegalArgumentException if the supplied descriptor check rejects metadata, or if kind/type/correction/duplicate-axis validation fails, or
     *     if a static selected-domain count is not greater than correction
     * @throws IndexOutOfBoundsException if a caller axis is invalid for the input Shape
     * @throws IllegalStateException if tensor identifier space is exhausted after valid metadata
     */
    static Tensor applyStatisticalChecked(
            Tensor input,
            AggregateReductionKind kind,
            int[] axes,
            boolean keepDimensions,
            long correction,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        return applyStatisticalDescriptors(
                input.descriptor(), kind, axes, keepDimensions, correction, descriptorCheck, input);
    }

    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param axes ordinary construction argument, validated in existing order
     * @param keepDimensions ordinary construction argument, validated in existing order
     * @param correction ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor applyStatisticalDescriptors(
            TensorDescriptor input,
            AggregateReductionKind kind,
            int[] axes,
            boolean keepDimensions,
            long correction,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(axes, "axes");
        validateStatisticalKind(kind);
        validateFloating(input);
        if (correction < 0) {
            throw new IllegalArgumentException("correction must be non-negative: " + correction);
        }
        int[] copiedAxes = axes.clone();
        Shape inputShape = input.shape();
        List<Integer> normalizedAxes = normalizeAxes(inputShape, copiedAxes);
        validateStaticDomainCount(inputShape, normalizedAxes, correction);
        Shape resultShape = reduceShape(inputShape, normalizedAxes, keepDimensions);
        return createDescriptors(input, kind, new StatisticalReductionAttrs(normalizedAxes, keepDimensions, correction), resultShape, descriptorCheck, inputTensor);
    }

    /**
     * Restricts ordinary construction to its seven kinds.
     *
     * @param kind non-null kind already checked by the entry
     * @throws IllegalArgumentException if {@code kind} is not ordinary, with the exact permitted
     *     kind list and rejected value in the message
     */
    private static void validateOrdinaryKind(AggregateReductionKind kind) {
        if (kind != AggregateReductionKind.SUM
                && kind != AggregateReductionKind.MEAN
                && kind != AggregateReductionKind.PROD
                && kind != AggregateReductionKind.MIN
                && kind != AggregateReductionKind.MAX
                && kind != AggregateReductionKind.ALL
                && kind != AggregateReductionKind.ANY) {
            throw new IllegalArgumentException(
                    "kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was " + kind);
        }
    }

    /**
     * Restricts uncorrected advanced construction to log-sum-exp and the two norms.
     *
     * @param kind non-null kind already checked by the entry
     * @throws IllegalArgumentException if unsupported, with the exact permitted list in the message
     */
    private static void validateAdvancedKind(AggregateReductionKind kind) {
        if (kind != AggregateReductionKind.LOG_SUM_EXP
                && kind != AggregateReductionKind.L1_NORM
                && kind != AggregateReductionKind.L2_NORM) {
            throw new IllegalArgumentException(
                    "kind must be LOG_SUM_EXP, L1_NORM, or L2_NORM, but was " + kind);
        }
    }

    /**
     * Restricts corrected construction to variance and standard deviation.
     *
     * @param kind non-null kind already checked by the entry
     * @throws IllegalArgumentException if unsupported, with both permitted kinds in the message
     */
    private static void validateStatisticalKind(AggregateReductionKind kind) {
        if (kind != AggregateReductionKind.VARIANCE
                && kind != AggregateReductionKind.STANDARD_DEVIATION) {
            throw new IllegalArgumentException(
                    "kind must be VARIANCE or STANDARD_DEVIATION, but was " + kind);
        }
    }

    /**
     * Enforces the existing kind-specific ordinary input domains without conversion.
     *
     * @param input non-null descriptor whose immutable data type is inspected
     * @param kind validated ordinary kind
     * @throws IllegalArgumentException if ALL/ANY is not BOOL, MEAN is not floating, or another
     *     ordinary numeric kind receives non-numeric input, with the existing exact messages
     */
    private static void validateOrdinaryInput(TensorDescriptor input, AggregateReductionKind kind) {
        DataType dataType = input.dataType();
        if (kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY) {
            if (dataType != DataType.BOOL) {
                throw new IllegalArgumentException(
                        "input must have BOOL data type for " + kind + ", but was " + dataType);
            }
        } else if (kind == AggregateReductionKind.MEAN) {
            validateFloating(input);
        } else if (!dataType.isFloating() && !dataType.isIntegral()) {
            throw new IllegalArgumentException(
                    "input must have a numeric data type for " + kind + ", but was " + dataType);
        }
    }

    /**
     * Requires exact floating input without cast or promotion.
     *
     * @param input non-null descriptor whose immutable data type is inspected
     * @throws IllegalArgumentException if the input is integral or BOOL, with the exact floating
     *     type message
     */
    private static void validateFloating(TensorDescriptor input) {
        DataType dataType = input.dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "input must have a floating data type, but was " + dataType);
        }
    }

    /**
     * Normalizes a caller-owned axis snapshot once per entry in caller order.
     *
     * @param shape non-null exact input Shape supplying rank and normalization
     * @param axes non-null private cloned axis array; may be empty
     * @return a non-null immutable ordered list of distinct normalized axes
     * @throws IndexOutOfBoundsException if an axis is invalid for {@code shape}
     * @throws IllegalArgumentException if a normalized axis repeats, reported at its raw index
     */
    private static List<Integer> normalizeAxes(Shape shape, int[] axes) {
        List<Integer> normalized = new ArrayList<>(axes.length);
        boolean[] seen = new boolean[shape.rank()];
        for (int index = 0; index < axes.length; index++) {
            int axis = shape.normalizeAxis(axes[index]);
            if (seen[axis]) {
                throw new IllegalArgumentException(
                        "axes contains duplicate axis " + axis + " at index " + index);
            }
            seen[axis] = true;
            normalized.add(axis);
        }
        return List.copyOf(normalized);
    }

    /**
     * Derives the multi-axis result Shape from selected-axis membership.
     *
     * @param inputShape non-null immutable input Shape
     * @param axes non-null distinct normalized axes
     * @param keepDimensions whether to replace selected positions with new extent-one Dimensions
     * @return the non-null derived Shape; every unselected Dimension is retained by exact reference,
     *     and removing all axes produces the canonical scalar Shape
     */
    private static Shape reduceShape(
            Shape inputShape, List<Integer> axes, boolean keepDimensions) {
        boolean[] selected = selectedAxes(inputShape.rank(), axes);
        Dimension[] dimensions = new Dimension[keepDimensions
                ? inputShape.rank()
                : inputShape.rank() - axes.size()];
        int outputIndex = 0;
        for (int axis = 0; axis < inputShape.rank(); axis++) {
            if (selected[axis]) {
                if (keepDimensions) {
                    dimensions[outputIndex++] = new StaticDimension(1);
                }
            } else {
                dimensions[outputIndex++] = inputShape.dimensions().get(axis);
            }
        }
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Rejects a statically provable non-positive corrected denominator.
     *
     * <p>A selected static zero proves count zero immediately. If every selected Dimension is
     * static, their checked product supplies {@code N}; multiplication overflow proves the count
     * exceeds every non-negative long correction. Any dynamic/expression Dimension defers proof.
     * Empty axes produce count one.</p>
     *
     * @param shape non-null input Shape
     * @param axes non-null normalized selected axes
     * @param correction validated non-negative correction
     * @throws IllegalArgumentException if static count is at most correction
     */
    private static void validateStaticDomainCount(
            Shape shape, List<Integer> axes, long correction) {
        for (int axis : axes) {
            Dimension dimension = shape.dimensions().get(axis);
            if (dimension instanceof StaticDimension staticDimension
                    && staticDimension.size() == 0) {
                rejectDomainCount(0, correction);
                return;
            }
        }
        long count = 1;
        for (int axis : axes) {
            Dimension dimension = shape.dimensions().get(axis);
            if (!(dimension instanceof StaticDimension staticDimension)) {
                return;
            }
            try {
                count = Math.multiplyExact(count, staticDimension.size());
            } catch (ArithmeticException overflow) {
                return;
            }
        }
        if (count <= correction) {
            rejectDomainCount(count, correction);
        }
    }

    /**
     * Throws the canonical invalid-statistical-domain failure.
     *
     * @param count statically known non-negative selected-domain count
     * @param correction validated non-negative correction at least {@code count}
     * @throws IllegalArgumentException always, with exact count and correction in the message
     */
    private static void rejectDomainCount(long count, long correction) {
        throw new IllegalArgumentException(
                "reduction domain count " + count
                        + " must be greater than correction " + correction);
    }

    /**
     * Creates a membership bitmap for normalized axes.
     *
     * @param rank non-negative input rank
     * @param axes non-null distinct valid normalized axes
     * @return a new non-null rank-sized bitmap whose true entries are exactly {@code axes}
     */
    private static boolean[] selectedAxes(int rank, List<Integer> axes) {
        boolean[] selected = new boolean[rank];
        for (int axis : axes) {
            selected[axis] = true;
        }
        return selected;
    }


    /**
     * Runs the shared descriptor guards and completes metadata before real provenance is used.
     * @param input immutable operand metadata
     * @param kind ordinary construction argument, validated in existing order
     * @param attrs ordinary construction argument, validated in existing order
     * @param shape ordinary construction argument, validated in existing order
     * @param descriptorCheck optional synchronous, unretained metadata completion; null constructs a Tensor
     * @param inputTensor real provenance, unused when metadata completion is supplied
     * @return null after metadata completion, otherwise the ordinary result
     */
    private static Tensor createDescriptors(
            TensorDescriptor input,
            AggregateReductionKind kind,
            OperationAttrs attrs,
            Shape shape,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> descriptorCheck,
            Tensor inputTensor) {
        TensorDescriptor descriptor = new TensorDescriptor(
                input.dataType(),
                shape,
                Optional.empty(),
                input.requiresGrad());
        Operation operation = new Operation(kind, attrs);
        if (descriptorCheck != null) {
            descriptorCheck.accept(operation, List.of(descriptor));
            return null;
        }
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(inputTensor));
    }

    /**
     * Completes one admitted ordinary occurrence synchronously using the real construction guards.
     * No Tensor or identity is created and the callback is not retained. The owning facade checks
     * occurrence cardinality and compares the returned canonical operation with the supplied one.
     *
     * @param operation non-null ordinary operation admitted to this family by the owning selector
     * @param inputs non-null immutable operand metadata in canonical input order
     * @param outputCount complete canonical output count, already validated by the owning facade
     * @param sink non-null synchronous consumer of the canonical operation and all output metadata
     * @throws IllegalArgumentException if ordinary semantic guards or the consumer reject metadata
     */
    static void describeChecked(Operation operation, List<TensorDescriptor> inputs, int outputCount,
            java.util.function.BiConsumer<Operation, List<TensorDescriptor>> sink) {
        Objects.requireNonNull(sink, "sink");
        AggregateReductionKind kind = (AggregateReductionKind) operation.kind();
        if (operation.attrs() instanceof StatisticalReductionAttrs attrs) {
            int[] axes = attrs.axes().stream().mapToInt(Integer::intValue).toArray();
            applyStatisticalDescriptors(inputs.get(0), kind, axes, attrs.keepDimensions(),
                    attrs.correction(), sink, null);
        } else {
            MultiAxisReductionAttrs attrs = (MultiAxisReductionAttrs) operation.attrs();
            int[] axes = attrs.axes().stream().mapToInt(Integer::intValue).toArray();
            if (kind == AggregateReductionKind.LOG_SUM_EXP
                    || kind == AggregateReductionKind.L1_NORM || kind == AggregateReductionKind.L2_NORM) {
                applyAdvancedDescriptors(inputs.get(0), kind, axes, attrs.keepDimensions(), sink, null);
            } else {
                applyOrdinaryDescriptors(inputs.get(0), kind, axes, attrs.keepDimensions(), sink, null);
            }
        }
    }
}

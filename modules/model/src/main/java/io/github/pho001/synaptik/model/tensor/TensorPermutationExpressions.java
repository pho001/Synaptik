package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free axis-permutation expressions for {@link Tensor}.
 *
 * <p>Raw axes are normalized once against the input rank and interpreted in output-to-input
 * order. Construction reorders the exact immutable Dimension references and, for any resolved
 * input layout kind, the exact element strides while preserving the element offset in one new
 * view-marked descriptor. Unresolved input geometry remains unresolved.</p>
 *
 * <p>This field-free package-private boundary creates semantic metadata only. It neither reads nor
 * copies values, observes or attaches storage, establishes a physical alias, constructs an inverse
 * permutation, canonicalizes expressions, defines gradients, captures a graph, plans
 * materialization, selects backend behavior, nor executes work.</p>
 */
final class TensorPermutationExpressions {
    /** Prevents instantiation because permutation-expression construction owns no state. */
    private TensorPermutationExpressions() {
    }

    /**
     * Validates a complete raw permutation and creates one fresh permutation expression.
     *
     * <p>Validation checks {@code input}, then {@code requestedAxes}, reads the exact input
     * descriptor and Shape once, and normalizes a private copy before deriving Shape and layout.
     * Failures before the final factory delegation consume no Tensor identity.</p>
     *
     * @param input non-null tensor retained as the exact sole provenance input; it is not mutated
     * @param requestedAxes non-null caller-owned complete output-to-input permutation; negative
     *     entries add the input rank once, and an empty array is valid for scalar input
     * @return the non-null fresh unlabeled, storage-free PERMUTE Tensor returned by the single
     *     derived-factory invocation
     * @throws NullPointerException if {@code input} or {@code requestedAxes} is null, with the
     *     corresponding parameter name as the message
     * @throws IllegalArgumentException if axis count differs from rank or a normalized axis is
     *     outside the rank or duplicated
     * @throws ArithmeticException if resolved result-layout classification or span arithmetic
     *     overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, int[] requestedAxes) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(requestedAxes, "requestedAxes");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        int[] normalizedAxes = normalizePermutation(inputShape.rank(), requestedAxes);
        Shape resultShape = permuteShape(inputShape, normalizedAxes);
        Optional<LayoutDescriptor> resultLayout =
                resolveViewLayout(inputDescriptor, resultShape, normalizedAxes);
        return create(
                input, inputDescriptor, resultShape, normalizedAxes, resultLayout);
    }

    /**
     * Creates the rank-two transpose convenience through the common permutation path.
     *
     * <p>Validation checks {@code input} before reading rank. Exactly rank two is accepted, then a
     * private {@code [1, 0]} array is passed to {@link #apply(Tensor, int[])}. The result therefore
     * uses ordinary PERMUTE semantics and attributes rather than another operation kind.</p>
     *
     * @param input non-null rank-two tensor retained as the exact sole provenance input
     * @return the non-null fresh unlabeled, storage-free PERMUTE Tensor with axes {@code [1, 0]}
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalStateException if rank is not two, with exact message
     *     {@code transpose() requires rank-2 tensor, got rank=<rank>}, or if identifier space is
     *     exhausted during final creation
     * @throws ArithmeticException if resolved result-layout classification or span arithmetic
     *     overflows
     */
    static Tensor transpose(Tensor input) {
        Objects.requireNonNull(input, "input");
        int rank = input.descriptor().shape().rank();
        if (rank != 2) {
            throw new IllegalStateException(
                    "transpose() requires rank-2 tensor, got rank=" + rank);
        }
        return apply(input, new int[] {1, 0});
    }

    /**
     * Copies, normalizes, and validates one complete permutation.
     *
     * <p>Count is checked before the single defensive clone. Copied entries are inspected in
     * ascending index order. A negative raw value adds {@code rank} once using {@code long}
     * arithmetic; range is checked before first-duplicate detection. Normalized values replace the
     * copied entries, and no caller-owned state is mutated or retained.</p>
     *
     * @param rank non-negative input rank defining valid normalized axes
     * @param requestedAxes non-null caller-owned raw permutation
     * @return the newly allocated normalized complete permutation
     * @throws IllegalArgumentException if count differs from rank, a normalized value is outside
     *     {@code [0, rank)}, or the first duplicate normalized value is encountered
     */
    private static int[] normalizePermutation(int rank, int[] requestedAxes) {
        if (requestedAxes.length != rank) {
            throw new IllegalArgumentException(
                    "permutation axis count "
                            + requestedAxes.length
                            + " must equal input rank "
                            + rank);
        }

        int[] normalizedAxes = requestedAxes.clone();
        boolean[] seen = new boolean[rank];
        for (int index = 0; index < normalizedAxes.length; index++) {
            int rawAxis = normalizedAxes[index];
            long normalizedAxis = rawAxis;
            if (normalizedAxis < 0) {
                normalizedAxis += rank;
            }
            if (normalizedAxis < 0 || normalizedAxis >= rank) {
                throw new IllegalArgumentException(
                        "permutation axis "
                                + rawAxis
                                + " at index "
                                + index
                                + " is outside rank "
                                + rank);
            }
            int axis = (int) normalizedAxis;
            if (seen[axis]) {
                throw new IllegalArgumentException(
                        "permutation contains duplicate normalized axis "
                                + axis
                                + " at index "
                                + index);
            }
            seen[axis] = true;
            normalizedAxes[index] = axis;
        }
        return normalizedAxes;
    }

    /**
     * Reorders exact immutable input Dimension references into one result Shape.
     *
     * @param inputShape non-null exact input Shape whose dimensions are read but not mutated
     * @param normalizedAxes non-null complete output-to-input permutation
     * @return non-null Shape containing the exact input Dimension references in requested order;
     *     scalar input may return the canonical scalar Shape
     */
    private static Shape permuteShape(Shape inputShape, int[] normalizedAxes) {
        Dimension[] resultDimensions = new Dimension[inputShape.rank()];
        for (int outputAxis = 0; outputAxis < normalizedAxes.length; outputAxis++) {
            resultDimensions[outputAxis] =
                    inputShape.dimensions().get(normalizedAxes[outputAxis]);
        }
        return Shape.ofDimensions(resultDimensions);
    }

    /**
     * Reorders resolved element strides while preserving offset in one new logical view.
     *
     * <p>The input layout optional is read exactly once. Absence returns unresolved state without
     * guessing geometry. Every resolved input layout kind is accepted; its exact strides are
     * reordered by the normalized output-to-input mapping and its exact element offset is retained.
     * {@link LayoutDescriptor} derives the new kind and referenced span. The new view metadata
     * attaches no storage and promises neither a physical alias nor zero-copy execution.</p>
     *
     * @param inputDescriptor non-null exact input descriptor supplying optional layout
     * @param resultShape non-null reordered result Shape used for layout derivation
     * @param normalizedAxes non-null complete normalized output-to-input permutation
     * @return non-null optional containing one new view-marked resolved layout, or empty when input
     *     geometry is unresolved
     * @throws ArithmeticException if checked classification or referenced-span arithmetic
     *     overflows
     */
    private static Optional<LayoutDescriptor> resolveViewLayout(
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            int[] normalizedAxes) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (inputLayout.isEmpty()) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] resultStrides = new long[resultShape.rank()];
        for (int outputAxis = 0; outputAxis < resultStrides.length; outputAxis++) {
            resultStrides[outputAxis] =
                    resolvedInputLayout.stride(normalizedAxes[outputAxis]);
        }
        return Optional.of(LayoutDescriptor.of(
                resultShape,
                resultStrides,
                resolvedInputLayout.storageOffset(),
                true));
    }

    /**
     * Creates exact permutation attributes, descriptor, operation, provenance, and Tensor once.
     *
     * <p>The normalized primitive axes are boxed in exact order into an immutable list. The result
     * descriptor retains input data type and gradient eligibility with the supplied Shape/layout.
     * The operation uses exact {@link AxisTransformKind#PERMUTE} and one
     * {@link PermutationAttrs}. The central derived factory is called exactly once with ordered
     * producer input {@code [input]} and no label or storage, creates index-zero provenance, and
     * produces a fresh identity even for identity, inverse, repeated, or nested requests.</p>
     *
     * @param input non-null exact sole provenance input; not inspected or mutated
     * @param inputDescriptor non-null descriptor supplying exact data type and gradient eligibility
     * @param resultShape non-null reordered Shape retained by the result descriptor
     * @param normalizedAxes non-null normalized complete output-to-input permutation
     * @param resultLayout non-null resolved-view or unresolved-layout value retained by descriptor
     * @return non-null fresh factory-derived Tensor with exact PERMUTE provenance and no label or
     *     storage
     * @throws IllegalArgumentException if completed semantic or descriptor invariants reject the
     *     supplied metadata
     * @throws ArithmeticException if descriptor layout reconstruction arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at the single factory
     *     delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape resultShape,
            int[] normalizedAxes,
            Optional<LayoutDescriptor> resultLayout) {
        List<Integer> axes = Arrays.stream(normalizedAxes).boxed().toList();
        PermutationAttrs attrs = new PermutationAttrs(axes);
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                resultShape,
                resultLayout,
                inputDescriptor.requiresGrad());
        Operation operation = new Operation(AxisTransformKind.PERMUTE, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}

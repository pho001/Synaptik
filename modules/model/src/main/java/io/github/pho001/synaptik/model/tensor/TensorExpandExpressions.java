package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated, storage-free expansion expressions for {@link Tensor}.
 *
 * <p>Expansion aligns input axes with target axes from the right. Equal dimensions are retained,
 * statically known input singletons may repeat to any target dimension, and target-only leading
 * axes behave as expansion from implicit singletons. The helper rejects compatibility that cannot
 * be proved from immutable Shape values; it never binds symbols or creates graph constraints.</p>
 *
 * <p>When the target is fully static and input layout is resolved, construction derives new
 * logical view geometry by preserving the exact input offset and aligned strides and inserting
 * zero strides for repeated axes. The resulting descriptor has no attached storage and makes no
 * claim about alias realization, materialization, backend lowering, gradients, or execution.</p>
 */
final class TensorExpandExpressions {
    /**
     * Prevents instantiation of this field-free construction helper.
     */
    private TensorExpandExpressions() {
    }

    /**
     * Normalizes raw literal dimensions and creates one fresh expansion expression.
     *
     * <p>Validation checks {@code input}, then {@code requestedShape}. The exact input descriptor
     * and its Shape are read once before {@link Shape#of(long...)} constructs the target. The
     * caller-owned array is neither retained nor mutated. An empty requested-dimension array
     * produces scalar Shape; every numeric value is literal, so negative values including
     * {@code -1} are rejected by the static-dimension contract.</p>
     *
     * @param input non-null tensor retained as the sole provenance input; it is not mutated
     * @param requestedShape non-null caller-owned literal dimensions; values must be non-negative
     *     and an empty array requests scalar Shape
     * @return a non-null fresh unlabeled, storage-free expansion expression
     * @throws NullPointerException if {@code input} or {@code requestedShape} is null, with the
     *     corresponding parameter name as the message
     * @throws IllegalArgumentException if a dimension is negative, target rank is below input
     *     rank, or aligned dimensions are incompatible
     * @throws ArithmeticException if resolved layout stride or referenced-span arithmetic
     *     overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, long[] requestedShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(requestedShape, "requestedShape");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        Shape targetShape = Shape.of(requestedShape);
        validateExpansion(inputShape, targetShape);
        Optional<LayoutDescriptor> resultLayout = resolveViewLayout(inputDescriptor, targetShape);
        return create(input, inputDescriptor, targetShape, resultLayout);
    }

    /**
     * Validates and retains an exact target Shape for one fresh expansion expression.
     *
     * <p>Validation checks {@code input}, then {@code targetShape}. The exact input descriptor and
     * Shape are read once. The supplied target reference is neither copied nor normalized and is
     * retained in both result descriptor and operation attributes.</p>
     *
     * @param input non-null tensor retained as the sole provenance input; it is not mutated
     * @param targetShape non-null exact target Shape retained by reference
     * @return a non-null fresh unlabeled, storage-free expansion expression
     * @throws NullPointerException if {@code input} or {@code targetShape} is null, with the
     *     corresponding parameter name as the message
     * @throws IllegalArgumentException if target rank is below input rank or aligned dimensions
     *     are incompatible
     * @throws ArithmeticException if resolved layout stride or referenced-span arithmetic
     *     overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, Shape targetShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        validateExpansion(inputShape, targetShape);
        Optional<LayoutDescriptor> resultLayout = resolveViewLayout(inputDescriptor, targetShape);
        return create(input, inputDescriptor, targetShape, resultLayout);
    }

    /**
     * Validates directional right-aligned expansion compatibility from current Shape metadata.
     *
     * <p>Target-only leading axes are valid. Aligned axes are inspected in ascending input-axis
     * order and are accepted only when their immutable dimensions are structurally equal or the
     * input is a static singleton. Thus equal dynamic symbols and singleton-to-dynamic targets are
     * valid, while unequal symbols and other unprovable dynamic pairs are rejected without
     * binding or constraint creation.</p>
     *
     * @param inputShape non-null exact input Shape to align from the right
     * @param targetShape non-null exact requested result Shape
     * @throws IllegalArgumentException if target rank is below input rank or the first aligned
     *     incompatible pair is encountered
     */
    private static void validateExpansion(Shape inputShape, Shape targetShape) {
        int inputRank = inputShape.rank();
        int targetRank = targetShape.rank();
        if (targetRank < inputRank) {
            throw new IllegalArgumentException(
                    "expand target rank " + targetRank
                            + " must be at least input rank " + inputRank);
        }

        int rankOffset = targetRank - inputRank;
        for (int inputAxis = 0; inputAxis < inputRank; inputAxis++) {
            int targetAxis = rankOffset + inputAxis;
            Dimension inputDimension = inputShape.dimension(inputAxis);
            Dimension targetDimension = targetShape.dimension(targetAxis);
            if (!inputDimension.equals(targetDimension)
                    && !(inputDimension instanceof StaticDimension staticDimension
                            && staticDimension.size() == 1)) {
                throw new IllegalArgumentException(
                        "cannot expand input shape " + inputShape + " to target shape "
                                + targetShape + " at target axis " + targetAxis);
            }
        }
    }

    /**
     * Resolves logical view geometry only when both required numeric facts are available.
     *
     * <p>The input layout optional is read exactly once. Dynamic target geometry or absent input
     * layout returns unresolved state. Otherwise one target-rank stride array is derived from
     * every resolved input layout kind, and a new view-marked descriptor preserves the exact input
     * element offset. This metadata neither attaches input storage nor proves that execution will
     * realize an alias.</p>
     *
     * @param inputDescriptor non-null exact input descriptor supplying optional layout and Shape
     * @param targetShape non-null validated target Shape
     * @return non-null optional containing one new resolved same-offset view layout, or empty when
     *     target or input layout geometry is unresolved
     * @throws ArithmeticException if checked referenced-span or layout classification arithmetic
     *     overflows
     */
    private static Optional<LayoutDescriptor> resolveViewLayout(
            TensorDescriptor inputDescriptor, Shape targetShape) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (!targetShape.isFullyStatic() || inputLayout.isEmpty()) {
            return Optional.empty();
        }

        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        long[] derivedStrides = deriveExpandedStrides(
                inputDescriptor.shape(), resolvedInputLayout, targetShape);
        return Optional.of(LayoutDescriptor.of(
                targetShape,
                derivedStrides,
                resolvedInputLayout.storageOffset(),
                true));
    }

    /**
     * Derives one target-rank stride vector for a validated static expansion.
     *
     * <p>New leading axes receive stride zero. For aligned axes, a static input singleton that is
     * changed to a different target extent also receives stride zero; otherwise the exact input
     * stride is preserved, including zero or non-canonical values from an existing view. The
     * returned array is newly allocated and owned by the caller.</p>
     *
     * @param inputShape non-null exact fully static input Shape
     * @param inputLayout non-null resolved input layout whose strides are read but not retained
     * @param targetShape non-null validated fully static target Shape
     * @return newly allocated target-rank element-stride array
     */
    private static long[] deriveExpandedStrides(
            Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape) {
        int rankOffset = targetShape.rank() - inputShape.rank();
        long[] derivedStrides = new long[targetShape.rank()];
        for (int inputAxis = 0; inputAxis < inputShape.rank(); inputAxis++) {
            int targetAxis = rankOffset + inputAxis;
            Dimension inputDimension = inputShape.dimension(inputAxis);
            Dimension targetDimension = targetShape.dimension(targetAxis);
            if (inputDimension instanceof StaticDimension staticDimension
                    && staticDimension.size() == 1
                    && !inputDimension.equals(targetDimension)) {
                derivedStrides[targetAxis] = 0;
            } else {
                derivedStrides[targetAxis] = inputLayout.stride(inputAxis);
            }
        }
        return derivedStrides;
    }

    /**
     * Creates the exact descriptor, expand semantics, provenance, and final derived Tensor once.
     *
     * @param input non-null exact sole provenance input; not inspected or mutated
     * @param inputDescriptor non-null descriptor supplying exact DataType and gradient eligibility
     * @param targetShape non-null exact target retained in descriptor and attributes
     * @param resultLayout non-null resolved-view or unresolved-layout value retained by descriptor
     * @return non-null fresh factory-derived Tensor with no label or host storage
     * @throws IllegalArgumentException if completed descriptor invariants reject supplied metadata
     * @throws ArithmeticException if descriptor layout reconstruction arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted at the single factory
     *     delegation
     */
    private static Tensor create(
            Tensor input,
            TensorDescriptor inputDescriptor,
            Shape targetShape,
            Optional<LayoutDescriptor> resultLayout) {
        TensorDescriptor descriptor = new TensorDescriptor(
                inputDescriptor.dataType(),
                targetShape,
                resultLayout,
                inputDescriptor.requiresGrad());
        TargetShapeAttrs attrs = new TargetShapeAttrs(targetShape);
        Operation operation = new Operation(ShapeTransformKind.EXPAND, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}

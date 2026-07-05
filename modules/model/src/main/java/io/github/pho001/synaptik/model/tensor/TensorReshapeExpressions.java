package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Constructs storage-free reshape expressions from local Tensor metadata.
 *
 * <p>This package-private stateless boundary normalizes raw numeric request syntax, validates
 * locally known element counts, derives alias-view geometry only when contiguity and a static
 * target prove it, and creates exact one-input reshape provenance. It never retains request
 * arrays, binds dynamic dimensions, accesses values or storage, inserts materialization, captures
 * a graph, or performs compiler, planning, backend, runtime, or gradient work.</p>
 */
final class TensorReshapeExpressions {
    /** Prevents instantiation because reshape-expression construction owns no state. */
    private TensorReshapeExpressions() {
    }

    /**
     * Normalizes a raw numeric request and creates one fresh reshape expression.
     *
     * <p>Validation checks {@code input}, then {@code requestedShape}. The exact input descriptor
     * and Shape are read once before normalization. The caller array is neither retained nor
     * mutated. Layout resolution and common construction occur only after normalization and known
     * count validation succeed.</p>
     *
     * @param input non-null tensor retained as the sole provenance input; it is not mutated
     * @param requestedShape non-null caller-owned dimensions containing non-negative values and at
     *     most one exact {@code -1}; empty means scalar Shape
     * @return a non-null fresh unlabeled and storage-free reshape expression
     * @throws NullPointerException if {@code input} or {@code requestedShape} is null, with the
     *     corresponding parameter name as the message
     * @throws IllegalArgumentException if raw dimensions, inference, divisibility, or known element
     *     counts violate the reshape contract
     * @throws ArithmeticException if checked product, count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, long[] requestedShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(requestedShape, "requestedShape");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        Shape targetShape = normalizeRequestedShape(inputShape, requestedShape);
        Optional<LayoutDescriptor> resultLayout = resolveViewLayout(inputDescriptor, targetShape);
        return create(input, inputDescriptor, targetShape, resultLayout);
    }

    /**
     * Validates an exact normalized target Shape and creates one fresh reshape expression.
     *
     * <p>Validation checks {@code input}, then {@code targetShape}. The exact input descriptor and
     * Shape are read once, and the exact target reference is retained after count validation.
     * Dynamic equality is deferred rather than solved or rejected.</p>
     *
     * @param input non-null tensor retained as the sole provenance input; it is not mutated
     * @param targetShape non-null normalized target Shape retained by exact reference
     * @return a non-null fresh unlabeled and storage-free reshape expression
     * @throws NullPointerException if {@code input} or {@code targetShape} is null, with the
     *     corresponding parameter name as the message
     * @throws IllegalArgumentException if both Shapes have known unequal element counts
     * @throws ArithmeticException if checked count, stride, or span arithmetic overflows
     * @throws IllegalStateException if tensor identifier space is exhausted during final creation
     */
    static Tensor apply(Tensor input, Shape targetShape) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetShape, "targetShape");
        TensorDescriptor inputDescriptor = input.descriptor();
        Shape inputShape = inputDescriptor.shape();
        validateTargetShape(inputShape, targetShape);
        Optional<LayoutDescriptor> resultLayout = resolveViewLayout(inputDescriptor, targetShape);
        return create(input, inputDescriptor, targetShape, resultLayout);
    }

    /**
     * Converts raw dimensions into a normalized Shape after deterministic request validation.
     *
     * <p>The complete array is scanned in ascending index order before inference. A no-sentinel
     * request delegates to {@link Shape#of(long...)} and common target validation. With one
     * sentinel, inference requires a known input count and a non-zero overflow-checked product of
     * the other dimensions. A zero elsewhere suppresses irrelevant multiplication overflow and
     * then fails as ambiguous. Only the inferred branch clones and replaces an array element;
     * Shape construction itself defensively copies the no-sentinel input.</p>
     *
     * @param inputShape non-null exact input Shape used only for known-count inference/validation
     * @param requestedShape non-null caller-owned raw dimensions; never retained or mutated
     * @return non-null normalized scalar or static target Shape
     * @throws IllegalArgumentException for an invalid negative value, duplicate sentinel,
     *     unavailable or ambiguous inference, failed divisibility, or known count mismatch
     * @throws ArithmeticException if checked non-zero product or element-count arithmetic overflows
     */
    private static Shape normalizeRequestedShape(Shape inputShape, long[] requestedShape) {
        int inferredAxis = -1;
        boolean hasZero = false;
        for (int index = 0; index < requestedShape.length; index++) {
            long dimension = requestedShape[index];
            if (dimension == -1) {
                if (inferredAxis >= 0) {
                    throw new IllegalArgumentException(
                            "requestedShape must contain at most one -1");
                }
                inferredAxis = index;
            } else if (dimension < -1) {
                throw new IllegalArgumentException(
                        "requestedShape[" + index + "] must be non-negative or -1: " + dimension);
            } else if (dimension == 0) {
                hasZero = true;
            }
        }

        if (inferredAxis < 0) {
            Shape targetShape = Shape.of(requestedShape);
            validateTargetShape(inputShape, targetShape);
            return targetShape;
        }

        OptionalLong inputElementCount = inputShape.knownElementCount();
        if (inputElementCount.isEmpty()) {
            throw new IllegalArgumentException(
                    "cannot infer -1 from dynamic input shape " + inputShape);
        }
        if (hasZero) {
            throw new IllegalArgumentException(
                    "cannot infer -1 when known requested dimensions have product zero");
        }

        long knownProduct = 1;
        for (int index = 0; index < requestedShape.length; index++) {
            if (index != inferredAxis) {
                knownProduct = Math.multiplyExact(knownProduct, requestedShape[index]);
            }
        }

        long count = inputElementCount.getAsLong();
        if (count % knownProduct != 0) {
            throw new IllegalArgumentException(
                    "cannot infer reshape dimension: input element count " + count
                            + " is not divisible by known requested product " + knownProduct);
        }

        long[] normalized = requestedShape.clone();
        normalized[inferredAxis] = count / knownProduct;
        Shape targetShape = Shape.of(normalized);
        validateTargetShape(inputShape, targetShape);
        return targetShape;
    }

    /**
     * Rejects unequal locally known element counts and defers every dynamic equality constraint.
     *
     * @param inputShape non-null input Shape whose known count is queried exactly once
     * @param targetShape non-null target Shape whose known count is queried exactly once
     * @throws IllegalArgumentException if both counts are present and unequal
     * @throws ArithmeticException if either fully static count overflows
     */
    private static void validateTargetShape(Shape inputShape, Shape targetShape) {
        OptionalLong inputElementCount = inputShape.knownElementCount();
        OptionalLong targetElementCount = targetShape.knownElementCount();
        if (inputElementCount.isPresent()
                && targetElementCount.isPresent()
                && inputElementCount.getAsLong() != targetElementCount.getAsLong()) {
            throw new IllegalArgumentException(
                    "reshape element count mismatch: input=" + inputElementCount.getAsLong()
                            + ", target=" + targetElementCount.getAsLong());
        }
    }

    /**
     * Derives locally proven alias-view geometry without choosing materialization.
     *
     * <p>The input layout optional is read exactly once. A dynamic target, unresolved layout, or
     * non-contiguous resolved layout yields unresolved result geometry. Otherwise a temporary
     * canonical target layout supplies copied row-major strides for one new view descriptor that
     * preserves the exact input element offset.</p>
     *
     * @param inputDescriptor non-null exact input descriptor; not retained or mutated
     * @param targetShape non-null validated target Shape used to derive static geometry
     * @return non-null optional containing one new same-offset canonical view layout when locally
     *     provable, or empty when geometry/materialization must be deferred
     * @throws ArithmeticException if checked canonical-stride or referenced-span arithmetic
     *     overflows
     */
    private static Optional<LayoutDescriptor> resolveViewLayout(
            TensorDescriptor inputDescriptor, Shape targetShape) {
        Optional<LayoutDescriptor> inputLayout = inputDescriptor.layout();
        if (!targetShape.isFullyStatic() || inputLayout.isEmpty()) {
            return Optional.empty();
        }
        LayoutDescriptor resolvedInputLayout = inputLayout.orElseThrow();
        if (!resolvedInputLayout.isContiguous()) {
            return Optional.empty();
        }

        LayoutDescriptor canonical = LayoutDescriptor.contiguous(targetShape);
        return Optional.of(LayoutDescriptor.of(
                targetShape,
                canonical.strides(),
                resolvedInputLayout.storageOffset(),
                true));
    }

    /**
     * Creates the exact descriptor, semantics, provenance, and final derived Tensor once.
     *
     * @param input non-null exact sole provenance input; not inspected or mutated
     * @param inputDescriptor non-null descriptor supplying exact DataType and gradient eligibility
     * @param targetShape non-null exact normalized target retained in descriptor and attributes
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
        Operation operation = new Operation(ShapeTransformKind.RESHAPE, attrs);
        TensorProvenance provenance = new TensorProvenance(operation, List.of(input));
        return TensorFactory.createDerived(descriptor, Optional.empty(), provenance);
    }
}

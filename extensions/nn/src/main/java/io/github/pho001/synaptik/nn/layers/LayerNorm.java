package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.util.Objects;

/**
 * A stateful affine layer normalization with mandatory scale and bias parameters.
 *
 * <p>The positive-rank, fully static Shape of {@code scale} identifies the trailing input axes
 * normalized by this layer. {@code bias} has the same structural Shape and exact floating data
 * type. The layer retains the selected exact Shape and one exact typed positive epsilon, without
 * exposing either as a second configuration surface, and declares the parameters under stable
 * local names {@code scale} then {@code bias}. Both affine parameters are mandatory because the
 * Model layer-normalization operation accepts either no affine state or the exact ordered pair.</p>
 *
 * <p>{@link #forward(Tensor)} reads both current parameter bindings once and delegates directly
 * to the affine Model {@link Tensor#layerNorm(Shape, Tensor, Tensor, ScalarValue)} expression.
 * Model owns trailing-Shape matching, floating promotion, result metadata, population-variance
 * semantics, and provenance. This layer adds no numerical algorithm, saved statistics,
 * compilation, backend behavior, storage, or execution. Forward construction is identical in
 * training and evaluation mode.</p>
 *
 * <p>A successful {@link Parameter#replace(Tensor)} becomes visible to the next forward call.
 * Earlier Tensor references and already constructed expressions retain their prior exact inputs.
 * Replacement and forward construction are not thread-safe as a combined operation; callers must
 * coordinate them when one scale-and-bias snapshot matters.</p>
 */
public final class LayerNorm extends UnaryTensorModule {
    private final Parameter scale;
    private final Parameter bias;
    private final Shape normalizedShape;
    private final ScalarValue epsilon;

    /**
     * Creates affine layer normalization from exact caller-supplied scale and bias Tensors.
     *
     * <p>The normalized Shape is the exact immutable Shape reference of {@code scale}. Both
     * parameters must be floating, gradient-eligible, structurally Shape-equal, and have the same
     * exact data type. The Shape must have positive rank, be fully static, and have a positive
     * extent on every axis. Epsilon must satisfy the Model finite-positive contract and have that
     * same exact data type. All validation completes before either parameter is declared. The
     * supplied Tensor references, scale Shape reference, and epsilon reference are retained
     * without copying, mutation, or evaluation.</p>
     *
     * @param scale non-null floating Tensor with {@code requiresGrad == true} and a positive,
     *     fully static, positive-rank normalized Shape; retained exactly
     * @param bias non-null floating Tensor with {@code requiresGrad == true}, the exact scale data
     *     type, and Shape structurally equal to the scale Shape; retained exactly
     * @param epsilon non-null finite strictly positive floating value with the exact parameter
     *     data type; retained exactly
     * @throws NullPointerException if {@code scale}, {@code bias}, or {@code epsilon} is null,
     *     checked in that order
     * @throws IllegalArgumentException if scale floating type, gradient eligibility, positive
     *     rank, static Shape, or positive extents fail; if bias floating type, gradient
     *     eligibility, exact data-type equality, or structural Shape equality fail; or if epsilon
     *     is not finite, strictly positive, floating, and exactly parameter-typed, checked in that
     *     order
     */
    public LayerNorm(Tensor scale, Tensor bias, ScalarValue epsilon) {
        Tensor suppliedScale = Objects.requireNonNull(scale, "scale");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        ScalarValue suppliedEpsilon = Objects.requireNonNull(epsilon, "epsilon");

        DataType parameterType = validateScale(suppliedScale);
        Shape parameterShape = suppliedScale.descriptor().shape();
        validateBias(suppliedBias, parameterType, parameterShape);
        validateIntrinsicEpsilon(parameterShape, suppliedEpsilon);
        validateEpsilonType(suppliedEpsilon, parameterType);

        this.normalizedShape = parameterShape;
        this.epsilon = suppliedEpsilon;
        this.scale = parameter("scale", suppliedScale);
        this.bias = parameter("bias", suppliedBias);
    }

    /**
     * Creates affine layer normalization with exact typed one scale and zero bias.
     *
     * <p>Caller-controlled Shape, type, and epsilon validation completes before Tensor creation
     * or identifier allocation. Scale is then created first through
     * {@link ParameterInitializers#ones(Shape, DataType)}, followed by bias through
     * {@link ParameterInitializers#zeros(Shape, DataType)}. Both are fresh gradient-eligible
     * eager leaves. No random source, default epsilon, implicit conversion, or configurable
     * initialization policy is used.</p>
     *
     * <p>If Model allocation or identifier creation fails, construction returns no partially
     * initialized layer. A failure after scale creation does not roll back its already consumed
     * Tensor identifier.</p>
     *
     * @param normalizedShape non-null positive-rank fully static Shape with every extent positive;
     *     retained exactly and used for both parameters
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param epsilon non-null finite strictly positive floating value with the exact parameter
     *     data type; retained exactly
     * @throws NullPointerException if {@code normalizedShape}, {@code dataType}, or
     *     {@code epsilon} is null, checked in that order
     * @throws IllegalArgumentException if the Shape has rank zero, is not fully static, or has a
     *     zero extent; if {@code dataType} is not floating; if epsilon is non-floating, non-finite,
     *     negative, or either signed zero; if epsilon has a different data type; or if the Shape's
     *     element count exceeds the Model Java-array limit
     * @throws ArithmeticException if checked Model element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public LayerNorm(Shape normalizedShape, DataType dataType, ScalarValue epsilon) {
        Shape parameterShape = Objects.requireNonNull(normalizedShape, "normalizedShape");
        DataType parameterType = Objects.requireNonNull(dataType, "dataType");
        ScalarValue suppliedEpsilon = Objects.requireNonNull(epsilon, "epsilon");

        validatePositiveStaticShape(parameterShape);
        if (!parameterType.isFloating()) {
            throw new IllegalArgumentException(
                    "layer normalization initialization requires floating data type: "
                            + parameterType);
        }
        validateIntrinsicEpsilon(parameterShape, suppliedEpsilon);
        validateEpsilonType(suppliedEpsilon, parameterType);

        Tensor initializedScale = ParameterInitializers.ones(parameterShape, parameterType);
        Tensor initializedBias = ParameterInitializers.zeros(parameterShape, parameterType);
        this.normalizedShape = parameterShape;
        this.epsilon = suppliedEpsilon;
        this.scale = parameter("scale", initializedScale);
        this.bias = parameter("bias", initializedBias);
    }

    /**
     * Returns the stable scale parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code scale}; its
     *     {@link Parameter#value()} is the current scale binding
     */
    public Parameter scale() {
        return scale;
    }

    /**
     * Returns the stable bias parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code bias}; its
     *     {@link Parameter#value()} is the current bias binding
     */
    public Parameter bias() {
        return bias;
    }

    /**
     * Builds one affine layer-normalization Tensor expression from the current bindings.
     *
     * <p>The input null check occurs before either binding read. Scale and bias are then read once
     * in declaration order and passed unchanged with the retained exact normalized Shape and
     * epsilon to Model. The inherited Model contract validates trailing Shape, floating input and
     * promotion, exact result-typed epsilon, metadata, and producer provenance. In particular, a
     * higher-precision input can promote the result beyond the stored epsilon type and fail; this
     * method inserts no cast or epsilon conversion. It is mode-insensitive and performs no value
     * evaluation, compilation, lowering, storage access, or execution.</p>
     *
     * @param input non-null Tensor accepted by the affine Model layer-normalization expression for
     *     the retained configuration and current parameter bindings
     * @return a non-null fresh, unlabeled, storage-free, unresolved-layout Model affine
     *     layer-normalization expression with the input Shape, promoted floating type, combined
     *     operand gradient eligibility, and exact ordered {@code [input, scale, bias]} provenance
     *     from the bindings observed by this call
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if inherited Model floating-type, trailing-Shape,
     *     promotion, affine-Shape, or exact epsilon-type validation fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor currentScale = scale.value();
        Tensor currentBias = bias.value();
        return suppliedInput.layerNorm(normalizedShape, currentScale, currentBias, epsilon);
    }

    private static DataType validateScale(Tensor scale) {
        DataType scaleType = scale.descriptor().dataType();
        if (!scaleType.isFloating()) {
            throw new IllegalArgumentException(
                    "layer normalization scale must have a floating data type: " + scaleType);
        }
        if (!scale.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "layer normalization scale must have requiresGrad == true");
        }
        validatePositiveStaticShape(scale.descriptor().shape());
        return scaleType;
    }

    private static void validatePositiveStaticShape(Shape shape) {
        if (shape.rank() == 0) {
            throw new IllegalArgumentException(
                    "layer normalization normalized Shape must have positive rank");
        }
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "layer normalization normalized Shape must be fully static: " + shape);
        }
        for (int axis = 0; axis < shape.rank(); axis++) {
            long extent = ((StaticDimension) shape.dimension(axis)).size();
            if (extent == 0) {
                throw new IllegalArgumentException(
                        "layer normalization normalized Shape must have positive extent at axis "
                                + axis + ": " + extent);
            }
        }
    }

    private static void validateBias(Tensor bias, DataType scaleType, Shape scaleShape) {
        DataType biasType = bias.descriptor().dataType();
        if (!biasType.isFloating()) {
            throw new IllegalArgumentException(
                    "layer normalization bias must have a floating data type: " + biasType);
        }
        if (!bias.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "layer normalization bias must have requiresGrad == true");
        }
        if (biasType != scaleType) {
            throw new IllegalArgumentException(
                    "layer normalization bias data type must equal scale data type: scale="
                            + scaleType + ", bias=" + biasType);
        }
        Shape biasShape = bias.descriptor().shape();
        if (!biasShape.equals(scaleShape)) {
            throw new IllegalArgumentException(
                    "layer normalization bias Shape must equal scale Shape: scale="
                            + scaleShape + ", bias=" + biasShape);
        }
    }

    private static void validateIntrinsicEpsilon(Shape normalizedShape, ScalarValue epsilon) {
        new AffineLayerNormAttrs(normalizedShape, epsilon);
    }

    private static void validateEpsilonType(ScalarValue epsilon, DataType parameterType) {
        if (epsilon.dataType() != parameterType) {
            throw new IllegalArgumentException(
                    "layer normalization epsilon data type must equal parameter data type: epsilon="
                            + epsilon.dataType() + ", parameter=" + parameterType);
        }
    }
}

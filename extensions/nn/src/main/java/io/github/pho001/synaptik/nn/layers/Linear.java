package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * A stateful linear projection with one rank-two weight and an optional rank-one bias.
 *
 * <p>The weight is oriented as {@code [outFeatures, inFeatures]}. A supplied bias is exactly
 * {@code [outFeatures]} and has the same floating data type as the weight. The layer declares
 * these trainable values under stable local names {@code weight} and, when present, {@code bias}.
 * Accessors return the exact stable {@link Parameter} wrappers used by direct and recursive module
 * discovery.</p>
 *
 * <p>{@link #forward(Tensor)} reads each current parameter binding once and delegates to the
 * matching Model {@link Tensor#linear(Tensor)} overload. The visible result is therefore the
 * existing PERMUTE-to-MATMUL or PERMUTE-to-MATMUL-to-ADD Tensor-expression chain; this class adds
 * no LINEAR operation, numerical algorithm, compiler behavior, storage, or execution. Forward
 * construction is identical in training and evaluation mode.</p>
 *
 * <p>A successful {@link Parameter#replace(Tensor)} becomes visible to the next forward call.
 * Tensor references read earlier and expressions already constructed from them remain unchanged.
 * Parameter replacement and forward construction are not thread-safe as a combined operation;
 * callers must coordinate them when a consistent multi-parameter snapshot matters.</p>
 */
public final class Linear extends Module {
    private final Parameter weight;
    private final Optional<Parameter> bias;

    /**
     * Creates a no-bias layer from one exact caller-supplied weight Tensor.
     *
     * <p>Validation completes before parameter declaration and does not copy, evaluate, allocate,
     * or otherwise change the supplied Tensor. The weight must be fully static, gradient-eligible,
     * floating, and shaped as a positive {@code [outFeatures, inFeatures]} matrix.</p>
     *
     * @param weight non-null floating Tensor with {@code requiresGrad == true} and fully static
     *     positive rank-two Shape {@code [outFeatures, inFeatures]}; retained exactly
     * @throws NullPointerException if {@code weight} is {@code null}
     * @throws IllegalArgumentException if the weight type is not floating, gradient eligibility
     *     is false, rank is not two, Shape is not fully static, or either feature extent is zero,
     *     checked in that order
     */
    public Linear(Tensor weight) {
        Tensor suppliedWeight = Objects.requireNonNull(weight, "weight");
        validateWeight(suppliedWeight);
        this.weight = parameter("weight", suppliedWeight);
        this.bias = Optional.empty();
    }

    /**
     * Creates a biased layer from exact caller-supplied weight and bias Tensors.
     *
     * <p>The weight is {@code [outFeatures, inFeatures]}; the bias is exactly rank one
     * {@code [outFeatures]}. Null bias never means absence. All state validation completes before
     * either parameter declaration and does not copy, evaluate, allocate, or otherwise change the
     * supplied Tensors. Weight is declared before bias.</p>
     *
     * @param weight non-null floating Tensor with {@code requiresGrad == true} and fully static
     *     positive rank-two Shape {@code [outFeatures, inFeatures]}; retained exactly
     * @param bias non-null floating Tensor with {@code requiresGrad == true}, the exact weight data
     *     type, and fully static rank-one Shape whose Dimension equals weight out-features;
     *     retained exactly
     * @throws NullPointerException if {@code weight} or {@code bias} is {@code null}, checked in
     *     that order
     * @throws IllegalArgumentException if weight floating type, gradient eligibility, rank,
     *     static Shape, or positive extents fail; or if bias floating type, gradient eligibility,
     *     rank, static Shape, exact data-type equality, or out-features equality fail, checked in
     *     that order
     */
    public Linear(Tensor weight, Tensor bias) {
        Tensor suppliedWeight = Objects.requireNonNull(weight, "weight");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        validateWeight(suppliedWeight);
        validateBias(suppliedWeight, suppliedBias);
        this.weight = parameter("weight", suppliedWeight);
        this.bias = Optional.of(parameter("bias", suppliedBias));
    }

    /**
     * Creates a layer with fixed Glorot-uniform weight initialization and optional zero bias.
     *
     * <p>Caller-controlled validation completes before a random draw or Tensor identifier
     * allocation. Weight is created first by exactly
     * {@link ParameterInitializers#glorotUniform(Shape, DataType, RandomGenerator)} with Shape
     * {@code [outFeatures, inFeatures]}. When requested, bias is then created by exactly
     * {@link ParameterInitializers#zeros(Shape, DataType)} with Shape {@code [outFeatures]}; it is
     * deterministic typed zero and consumes no random draw. The caller selects, owns, advances,
     * and coordinates the exact random source, which is never retained.</p>
     *
     * <p>A source failure leaves its completed draws consumed and creates no weight Tensor. A
     * later allocation or identifier failure does not roll back successful draws or an already
     * created weight Tensor identifier. Construction returns no partially initialized layer.</p>
     *
     * @param inFeatures strictly positive input-feature count
     * @param outFeatures strictly positive output-feature count
     * @param bias whether to create and declare a deterministic zero bias
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source used only for weight samples
     * @throws NullPointerException if {@code dataType} or {@code randomGenerator} is null, checked
     *     in that order
     * @throws IllegalArgumentException if {@code inFeatures} or {@code outFeatures} is not
     *     positive, checked in that order; if {@code dataType} is not floating; or if the
     *     initialized Shape exceeds the Model Java-array limit
     * @throws RuntimeException if the random source throws while sampling; completed draws remain
     *     consumed and no weight Tensor or identifier is created
     * @throws ArithmeticException if checked Model element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public Linear(
            long inFeatures,
            long outFeatures,
            boolean bias,
            DataType dataType,
            RandomGenerator randomGenerator) {
        DataType parameterType = Objects.requireNonNull(dataType, "dataType");
        RandomGenerator source = Objects.requireNonNull(randomGenerator, "randomGenerator");
        if (inFeatures <= 0) {
            throw new IllegalArgumentException(
                    "inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException(
                    "outFeatures must be positive: " + outFeatures);
        }
        if (!parameterType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear initialization requires floating data type: " + parameterType);
        }

        Shape weightShape = Shape.of(outFeatures, inFeatures);
        Tensor initializedWeight = ParameterInitializers.glorotUniform(
                weightShape, parameterType, source);
        this.weight = parameter("weight", initializedWeight);
        if (bias) {
            Shape biasShape = Shape.of(outFeatures);
            Tensor initializedBias = ParameterInitializers.zeros(biasShape, parameterType);
            this.bias = Optional.of(parameter("bias", initializedBias));
        } else {
            this.bias = Optional.empty();
        }
    }

    /**
     * Returns the stable weight parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code weight}; its
     *     {@link Parameter#value()} is the current weight binding
     */
    public Parameter weight() {
        return weight;
    }

    /**
     * Returns the optional stable bias parameter wrapper.
     *
     * @return a non-null empty Optional for a no-bias layer, or an Optional containing the exact
     *     wrapper declared under local name {@code bias}
     */
    public Optional<Parameter> bias() {
        return bias;
    }

    /**
     * Builds a linear Tensor expression from the input and current parameter bindings.
     *
     * <p>The input null check occurs before either binding read. Each current binding is then read
     * exactly once and passed unchanged to the matching {@link Tensor#linear(Tensor)} overload.
     * Model owns rank, promotion, contraction, Shape, allocation, gradient-eligibility, and
     * primitive provenance behavior. This method is mode-insensitive and performs no value
     * evaluation, compilation, lowering, storage access, or execution.</p>
     *
     * @param input non-null Tensor accepted by the matching Model linear convenience for the
     *     current weight and optional bias
     * @return the non-null fresh Model linear expression using the exact bindings observed by
     *     this call
     * @throws NullPointerException if {@code input} is null, with message {@code input}
     * @throws IllegalArgumentException if inherited Model linear type, rank, contraction, or bias
     *     validation fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public Tensor forward(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor currentWeight = weight.value();
        if (bias.isEmpty()) {
            return suppliedInput.linear(currentWeight);
        }
        Tensor currentBias = bias.orElseThrow().value();
        return suppliedInput.linear(currentWeight, currentBias);
    }

    private static void validateWeight(Tensor weight) {
        DataType weightType = weight.descriptor().dataType();
        if (!weightType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear weight must have a floating data type: " + weightType);
        }
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "linear weight must have requiresGrad == true");
        }
        Shape weightShape = weight.descriptor().shape();
        if (weightShape.rank() != 2) {
            throw new IllegalArgumentException(
                    "linear weight must have rank two: " + weightShape.rank());
        }
        if (!weightShape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "linear weight must have a fully static shape: " + weightShape);
        }
        long outFeatures = ((StaticDimension) weightShape.dimension(0)).size();
        if (outFeatures == 0) {
            throw new IllegalArgumentException(
                    "linear weight must have positive outFeatures: " + outFeatures);
        }
        long inFeatures = ((StaticDimension) weightShape.dimension(1)).size();
        if (inFeatures == 0) {
            throw new IllegalArgumentException(
                    "linear weight must have positive inFeatures: " + inFeatures);
        }
    }

    private static void validateBias(Tensor weight, Tensor bias) {
        DataType biasType = bias.descriptor().dataType();
        if (!biasType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear bias must have a floating data type: " + biasType);
        }
        if (!bias.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "linear bias must have requiresGrad == true");
        }
        Shape biasShape = bias.descriptor().shape();
        if (biasShape.rank() != 1) {
            throw new IllegalArgumentException(
                    "linear bias must have rank one: " + biasShape.rank());
        }
        if (!biasShape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "linear bias must have a fully static shape: " + biasShape);
        }
        DataType weightType = weight.descriptor().dataType();
        if (biasType != weightType) {
            throw new IllegalArgumentException(
                    "linear bias data type must equal weight data type: weight="
                            + weightType + ", bias=" + biasType);
        }
        if (!biasShape.dimension(0).equals(weight.descriptor().shape().dimension(0))) {
            throw new IllegalArgumentException(
                    "linear bias dimension must equal weight outFeatures: bias="
                            + biasShape.dimension(0)
                            + ", weight="
                            + weight.descriptor().shape().dimension(0));
        }
    }
}

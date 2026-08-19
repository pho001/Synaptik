package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * A stateful linear projection with one rank-two weight and an optional rank-one bias.
 *
 * <p>The weight is oriented as {@code [outFeatures, inFeatures]}. A supplied bias is exactly
 * {@code [outFeatures]} and has the same floating data type as the weight. The layer declares
 * these trainable values under stable local names {@code weight} and, when present, {@code bias}.
 * Accessors return the exact stable {@link Parameter} wrappers used by direct and recursive module
 * discovery.</p>
 *
 * <p>The constructor without {@code inFeatures} reserves those parameter names but creates no
 * parameter Tensor. Its first compatible {@link #forward(Tensor)} call infers the positive static
 * final input extent, initializes and publishes the complete direct parameter set, and only then
 * constructs the ordinary linear expression returned by that same call. Later calls reuse the
 * published wrappers and permit arbitrary compatible leading Dimensions. Strict state loading
 * may initialize the reservations instead without invoking the configured random source.</p>
 *
 * <p>Only {@code inFeatures} is inferred. The caller still chooses architectural facts such as
 * {@code outFeatures}, bias presence, data type, initialization policy, random-factory contract,
 * and seed. Before successful automatic initialization, parameter access, discovery, and state
 * export fail rather than expose partial state. A failed initialization attempt publishes no
 * wrapper and is retryable with a fresh generator from the same factory and seed; consumed random
 * draws, allocations, and opaque Tensor identifiers are not rolled back.</p>
 *
 * <p>{@link #forward(Tensor)} delegates to the matching Model {@link Tensor#linear(Tensor)}
 * overload. The visible result is therefore the existing PERMUTE-to-MATMUL or
 * PERMUTE-to-MATMUL-to-ADD Tensor-expression chain; this class adds no LINEAR operation,
 * numerical evaluation, compiler behavior, storage, or execution. Forward construction is
 * identical in training and evaluation mode.</p>
 */
public final class Linear extends UnaryTensorModule {
    private final boolean biasConfigured;
    private final AutomaticConfiguration automaticConfiguration;

    /**
     * Creates a no-bias layer from one exact caller-supplied weight Tensor.
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
        parameter("weight", suppliedWeight);
        this.biasConfigured = false;
        this.automaticConfiguration = null;
    }

    /**
     * Creates a biased layer from exact caller-supplied weight and bias Tensors.
     *
     * @param weight non-null floating Tensor with {@code requiresGrad == true} and fully static
     *     positive rank-two Shape {@code [outFeatures, inFeatures]}; retained exactly
     * @param bias non-null floating Tensor with {@code requiresGrad == true}, the exact weight data
     *     type, and fully static rank-one Shape whose Dimension equals weight out-features;
     *     retained exactly
     * @throws NullPointerException if {@code weight} or {@code bias} is {@code null}, checked in
     *     that order
     * @throws IllegalArgumentException if weight or bias schema validation fails
     */
    public Linear(Tensor weight, Tensor bias) {
        Tensor suppliedWeight = Objects.requireNonNull(weight, "weight");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        validateWeight(suppliedWeight);
        validateBias(suppliedWeight, suppliedBias);
        parameter("weight", suppliedWeight);
        parameter("bias", suppliedBias);
        this.biasConfigured = true;
        this.automaticConfiguration = null;
    }

    /**
     * Creates and immediately initializes a layer with Glorot-uniform weight and optional bias.
     *
     * @param inFeatures strictly positive input-feature count
     * @param outFeatures strictly positive output-feature count
     * @param bias whether to create and declare a deterministic zero bias
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source used only for weight samples
     * @throws NullPointerException if {@code dataType} or {@code randomGenerator} is null, checked
     *     in that order
     * @throws IllegalArgumentException if a feature count is not positive, the type is not
     *     floating, or the initialized Shape exceeds the Model Java-array limit
     * @throws RuntimeException if the random source throws while sampling
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
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        if (!parameterType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear initialization requires floating data type: " + parameterType);
        }

        Shape weightShape = Shape.of(outFeatures, inFeatures);
        Tensor initializedWeight = ParameterInitializers.glorotUniform(
                weightShape, parameterType, source);
        parameter("weight", initializedWeight);
        if (bias) {
            Tensor initializedBias = ParameterInitializers.zeros(
                    Shape.of(outFeatures), parameterType);
            parameter("bias", initializedBias);
        }
        this.biasConfigured = bias;
        this.automaticConfiguration = null;
    }

    /**
     * Creates a layer that initializes its parameters during its first compatible forward call.
     *
     * <p>Construction retains only immutable configuration and reserves {@code weight}, followed
     * by optional {@code bias}. It creates no generator, Tensor, Tensor identifier, or Parameter.
     * The supplied factory must be deterministic and is retained exactly. A sampling policy
     * creates one fresh generator from that factory and seed for each attempt; zero and one
     * policies never invoke the factory or create a generator. No created generator or caller
     * input is retained.</p>
     *
     * @param outFeatures strictly positive architectural output-feature count
     * @param bias whether the first compatible forward creates a zero bias
     * @param dataType non-null exact floating parameter and accepted input type
     * @param weightInitialization non-null closed weight policy applied to the complete weight
     *     Shape; it owns neither Shape, source, nor seed
     * @param randomGeneratorFactory non-null deterministic factory retained exactly; invoked only
     *     when the policy requires sampling
     * @param seed seed passed unchanged to the retained factory for each sampling attempt; any
     *     Java {@code long} value is accepted
     * @throws NullPointerException if {@code dataType}, {@code weightInitialization}, or
     *     {@code randomGeneratorFactory} is null, checked in that order after
     *     {@code outFeatures}
     * @throws IllegalArgumentException if {@code outFeatures} is not positive, the type is not
     *     floating, or the factory is stochastic, checked in that order
     */
    public Linear(
            long outFeatures,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            RandomGeneratorFactory<? extends RandomGenerator> randomGeneratorFactory,
            long seed) {
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        DataType parameterType = Objects.requireNonNull(dataType, "dataType");
        ParameterInitialization initialization =
                Objects.requireNonNull(weightInitialization, "weightInitialization");
        RandomGeneratorFactory<? extends RandomGenerator> factory =
                Objects.requireNonNull(randomGeneratorFactory, "randomGeneratorFactory");
        if (!parameterType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear initialization requires floating data type: " + parameterType);
        }
        if (factory.isStochastic()) {
            throw new IllegalArgumentException(
                    "linear automatic initialization requires a deterministic random generator factory: "
                            + factory.name());
        }

        this.biasConfigured = bias;
        this.automaticConfiguration = new AutomaticConfiguration(
                outFeatures, parameterType, initialization, factory, seed);
        reserveParameter("weight", this::validateAutomaticWeight);
        if (bias) {
            reserveParameter("bias", this::validateAutomaticBias);
        }
    }

    /**
     * Returns the stable weight parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code weight}
     * @throws IllegalStateException if automatic first-forward or strict-load initialization has
     *     not completed
     */
    public Parameter weight() {
        return boundParameter("weight");
    }

    /**
     * Returns the optional stable bias parameter wrapper.
     *
     * @return empty for a no-bias layer, otherwise the exact wrapper declared under {@code bias}
     * @throws IllegalStateException if bias is configured but automatic initialization has not
     *     completed
     */
    public Optional<Parameter> bias() {
        if (!biasConfigured) {
            return Optional.empty();
        }
        return Optional.of(boundParameter("bias"));
    }

    /**
     * Builds a linear Tensor expression from the input and current parameter bindings.
     *
     * <p>For the automatic constructor, the first compatible call initializes and publishes all
     * direct parameters before it constructs this call's expression. Concurrent first calls
     * serialize only that initialization phase. A later expression-construction failure does not
     * undo already published parameters. A failure before publication leaves the reservation
     * group unbound and retryable; completed random draws, local allocations, and opaque Tensor
     * identifiers are not rolled back. Initialization is local to this layer: a functional Model
     * may retain an earlier initialized layer when later user code or another layer fails.</p>
     *
     * <p>The automatic path validates every descriptor/count fact knowable before sampling,
     * creates one generator from the configured factory and seed only for a sampling policy,
     * creates weight before optional zero bias, validates and publishes the complete group, then
     * constructs the ordinary Model expression. Zero and one policies do not invoke the retained
     * factory. The method does not numerically execute that expression.</p>
     *
     * @param input non-null Tensor; the automatic constructor requires the exact configured type,
     *     rank at least one, and a positive static final feature Dimension
     * @return a non-null fresh Model linear expression using the exact bindings observed once by
     *     this call
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if automatic input compatibility or inherited Model linear
     *     validation fails
     * @throws RuntimeException if the configured random generator throws while the automatic
     *     weight is sampled; completed draws remain consumed and no wrapper is published
     * @throws ArithmeticException if checked Model count or layout arithmetic overflows during
     *     automatic initialization or expression construction
     * @throws IllegalStateException if Tensor identifier space is exhausted or automatic
     *     publication does not complete
     * @throws OutOfMemoryError if automatic parameter or expression allocation fails
     */
    @Override
    public Tensor forward(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        if (automaticConfiguration != null) {
            AutomaticInput automaticInput = validateAutomaticInput(suppliedInput);
            if (!parameterReservationsBound()) {
                synchronized (this) {
                    if (!parameterReservationsBound()) {
                        automaticInput = validateAutomaticInput(suppliedInput);
                        initializeAutomatically(automaticInput);
                        if (!parameterReservationsBound()) {
                            throw new IllegalStateException(
                                    "linear automatic parameter initialization did not publish complete state");
                        }
                    }
                }
            }
        }

        Parameter currentWeightParameter = weight();
        Tensor currentWeight = currentWeightParameter.value();
        if (automaticConfiguration != null) {
            validateAutomaticBoundInput(suppliedInput, currentWeight);
        }
        if (!biasConfigured) {
            return suppliedInput.linear(currentWeight);
        }
        Parameter currentBiasParameter = bias().orElseThrow();
        Tensor currentBias = currentBiasParameter.value();
        return suppliedInput.linear(currentWeight, currentBias);
    }

    private AutomaticInput validateAutomaticInput(Tensor input) {
        AutomaticConfiguration configuration = automaticConfiguration;
        DataType inputType = input.descriptor().dataType();
        if (inputType != configuration.dataType) {
            throw new IllegalArgumentException(
                    "linear automatic input data type must equal configured data type: expected="
                            + configuration.dataType + ", actual=" + inputType);
        }
        Shape inputShape = input.descriptor().shape();
        int rank = inputShape.rank();
        if (rank < 1) {
            throw new IllegalArgumentException(
                    "linear automatic input rank must be at least one: " + rank);
        }
        Dimension finalDimension = inputShape.dimension(rank - 1);
        if (!(finalDimension instanceof StaticDimension staticDimension)) {
            throw new IllegalArgumentException(
                    "linear automatic input final feature dimension must be static: "
                            + finalDimension);
        }
        long inFeatures = staticDimension.size();
        if (inFeatures <= 0) {
            throw new IllegalArgumentException(
                    "linear automatic input must have positive inFeatures: " + inFeatures);
        }

        Shape weightShape = Shape.of(configuration.outFeatures, inFeatures);
        validateJavaArrayCount(weightShape, "linear automatic weight");
        Shape biasShape = Shape.of(configuration.outFeatures);
        if (biasConfigured) {
            validateJavaArrayCount(biasShape, "linear automatic bias");
        }
        return new AutomaticInput(inFeatures, weightShape, biasShape);
    }

    private void initializeAutomatically(AutomaticInput input) {
        AutomaticConfiguration configuration = automaticConfiguration;
        Tensor initializedWeight;
        if (configuration.weightInitialization.requiresRandomGenerator()) {
            RandomGenerator generator =
                    configuration.randomGeneratorFactory.create(configuration.seed);
            initializedWeight = ParameterInitializers.initialize(
                    input.weightShape,
                    configuration.dataType,
                    configuration.weightInitialization,
                    generator);
        } else {
            initializedWeight = ParameterInitializers.initialize(
                    input.weightShape,
                    configuration.dataType,
                    configuration.weightInitialization);
        }
        if (biasConfigured) {
            Tensor initializedBias = ParameterInitializers.zeros(
                    input.biasShape, configuration.dataType);
            bindReservedParameters(List.of(initializedWeight, initializedBias));
        } else {
            bindReservedParameters(List.of(initializedWeight));
        }
    }

    private void validateAutomaticBoundInput(Tensor input, Tensor weight) {
        AutomaticConfiguration configuration = automaticConfiguration;
        if (input.descriptor().dataType() != configuration.dataType) {
            throw new IllegalArgumentException(
                    "linear automatic input data type must equal configured data type: expected="
                            + configuration.dataType + ", actual="
                            + input.descriptor().dataType());
        }
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 1) {
            throw new IllegalArgumentException(
                    "linear automatic input rank must be at least one: " + inputShape.rank());
        }
        Dimension inputFeatures = inputShape.dimension(inputShape.rank() - 1);
        if (!(inputFeatures instanceof StaticDimension inputStatic)) {
            throw new IllegalArgumentException(
                    "linear automatic input final feature dimension must be static: "
                            + inputFeatures);
        }
        long expected = ((StaticDimension) weight.descriptor().shape().dimension(1)).size();
        if (inputStatic.size() != expected) {
            throw new IllegalArgumentException(
                    "linear automatic input feature dimension must match initialized inFeatures: expected="
                            + expected + ", actual=" + inputStatic.size());
        }
    }

    private void validateAutomaticWeight(Tensor weight) {
        AutomaticConfiguration configuration = automaticConfiguration;
        if (weight.descriptor().dataType() != configuration.dataType) {
            throw new IllegalArgumentException(
                    "linear automatic weight data type must equal configured data type: expected="
                            + configuration.dataType + ", actual="
                            + weight.descriptor().dataType());
        }
        Shape weightShape = weight.descriptor().shape();
        if (weightShape.rank() != 2) {
            throw new IllegalArgumentException(
                    "linear automatic weight must have rank two: " + weightShape.rank());
        }
        if (!weightShape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "linear automatic weight must have a fully static shape: " + weightShape);
        }
        long actualOut = ((StaticDimension) weightShape.dimension(0)).size();
        if (actualOut != configuration.outFeatures) {
            throw new IllegalArgumentException(
                    "linear automatic weight outFeatures must equal configured outFeatures: expected="
                            + configuration.outFeatures + ", actual=" + actualOut);
        }
        long actualIn = ((StaticDimension) weightShape.dimension(1)).size();
        if (actualIn <= 0) {
            throw new IllegalArgumentException(
                    "linear automatic weight must have positive inFeatures: " + actualIn);
        }
        validateJavaArrayCount(weightShape, "linear automatic weight");
    }

    private void validateAutomaticBias(Tensor bias) {
        DataType biasType = bias.descriptor().dataType();
        AutomaticConfiguration configuration = automaticConfiguration;
        if (biasType != configuration.dataType) {
            throw new IllegalArgumentException(
                    "linear automatic bias data type must equal configured data type: expected="
                            + configuration.dataType + ", actual=" + biasType);
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
        long actualOut = ((StaticDimension) biasShape.dimension(0)).size();
        if (actualOut != configuration.outFeatures) {
            throw new IllegalArgumentException(
                    "linear automatic bias outFeatures must equal configured outFeatures: expected="
                            + configuration.outFeatures + ", actual=" + actualOut);
        }
        validateJavaArrayCount(biasShape, "linear automatic bias");
    }

    private static void validateJavaArrayCount(Shape shape, String role) {
        long count = shape.knownElementCount().orElseThrow();
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    role + " element count exceeds Java array limit: count=" + count
                            + ", maximum=" + Integer.MAX_VALUE);
        }
    }

    private static void validateWeight(Tensor weight) {
        DataType weightType = weight.descriptor().dataType();
        if (!weightType.isFloating()) {
            throw new IllegalArgumentException(
                    "linear weight must have a floating data type: " + weightType);
        }
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException("linear weight must have requiresGrad == true");
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
            throw new IllegalArgumentException("linear bias must have requiresGrad == true");
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
                            + biasShape.dimension(0) + ", weight="
                            + weight.descriptor().shape().dimension(0));
        }
    }

    private record AutomaticConfiguration(
            long outFeatures,
            DataType dataType,
            ParameterInitialization weightInitialization,
            RandomGeneratorFactory<? extends RandomGenerator> randomGeneratorFactory,
            long seed) {
    }

    private record AutomaticInput(long inFeatures, Shape weightShape, Shape biasShape) {
    }
}

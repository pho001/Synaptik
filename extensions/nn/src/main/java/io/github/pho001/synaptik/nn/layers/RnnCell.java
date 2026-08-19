package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Constructs one explicit-state application of a vanilla hyperbolic-tangent recurrent
 * neural-network cell.
 *
 * <p>The cell owns trainable {@code inputWeight}, {@code hiddenWeight}, and optional shared
 * {@code bias} parameters. Their respective Shapes are {@code [hiddenSize, inputSize]},
 * {@code [hiddenSize, hiddenSize]}, and {@code [hiddenSize]}. The caller supplies the current
 * hidden Tensor to every {@link #forward(Tensor, Tensor)} call and receives the next hidden Tensor
 * as the sole result. The cell never retains, initializes, updates, or registers hidden state.</p>
 *
 * <p>Forward construction is fixed to
 * {@code tanh((input @ inputWeight^T + bias?) + (hidden @ hiddenWeight^T))}. Leading input and
 * hidden Dimensions are ordinary right-broadcastable batch metadata; none is interpreted or
 * traversed as time. This two-input contract makes the cell a direct {@link Module}, not a
 * {@link io.github.pho001.synaptik.nn.module.UnaryTensorModule}, so it cannot participate in
 * {@link io.github.pho001.synaptik.nn.module.Sequential}.</p>
 *
 * <p>The automatic constructor reserves the complete parameter group without creating a Tensor
 * or {@link Parameter}. Its first compatible represented step infers only the positive static
 * final input extent, optionally creates one fresh standard {@code L64X128MixRandom} stream,
 * initializes input weight then hidden weight and optional zero bias, and publishes the group
 * before constructing that step. Zero/one policies create no generator. Strict state loading may
 * bind the reservations instead. A failed initialization is retryable and publishes no partial
 * group; concurrent compatible first calls serialize only this cell-local initialization.</p>
 *
 * <p>Each call snapshots current parameter bindings once in declaration order. Compatible
 * replacement affects later calls, while existing expressions retain their earlier exact
 * references. Replacement and forward construction are not thread-safe as one multi-parameter
 * snapshot; callers must coordinate them when consistency matters. Construction is identical in
 * training and evaluation mode and creates Model expression metadata only. It does not execute
 * values, define gradients, capture a graph, lower operations, or select a backend.</p>
 */
public final class RnnCell extends Module {

    private final long hiddenSize;
    private final DataType parameterType;
    private final boolean biasConfigured;
    private final AutomaticConfiguration automaticConfiguration;

    /**
     * Creates a no-bias cell from exact caller-supplied projection weights.
     *
     * <p>Complete schema validation precedes parameter declaration. The supplied Tensors are
     * retained without copying, evaluating, allocating, or changing them.</p>
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type and hidden-size Dimension, shaped
     *     {@code [hiddenSize, hiddenSize]}, and otherwise satisfying the input-weight contract;
     *     retained exactly
     * @throws NullPointerException if {@code inputWeight} or {@code hiddenWeight} is null, checked
     *     in that order
     * @throws IllegalArgumentException if either parameter violates its documented type,
     *     gradient, rank, static-Shape, positive-extent, exact-type, or hidden-size contract
     */
    public RnnCell(Tensor inputWeight, Tensor hiddenWeight) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        this.hiddenSize = extent(suppliedInputWeight.descriptor().shape(), 0);
        this.parameterType = suppliedInputWeight.descriptor().dataType();
        this.biasConfigured = false;
        this.automaticConfiguration = null;
        parameter("inputWeight", suppliedInputWeight);
        parameter("hiddenWeight", suppliedHiddenWeight);
    }

    /**
     * Creates a biased cell from exact caller-supplied projection weights and one shared bias.
     *
     * <p>Null bias never means absence. Complete validation of all three Tensors precedes
     * declaration in {@code inputWeight}, {@code hiddenWeight}, {@code bias} order. No supplied
     * Tensor is copied, evaluated, allocated, or changed.</p>
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type and hidden-size Dimension, shaped
     *     {@code [hiddenSize, hiddenSize]}, and otherwise satisfying the input-weight contract;
     *     retained exactly
     * @param bias non-null floating, gradient-eligible, fully static rank-one Tensor shaped
     *     {@code [hiddenSize]} with the exact common weight type; retained exactly
     * @throws NullPointerException if {@code inputWeight}, {@code hiddenWeight}, or {@code bias}
     *     is null, checked in that order
     * @throws IllegalArgumentException if a parameter violates its documented type, gradient,
     *     rank, static-Shape, positive-extent, exact-type, or hidden-size contract
     */
    public RnnCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        validateBias(suppliedInputWeight, suppliedBias);
        this.hiddenSize = extent(suppliedInputWeight.descriptor().shape(), 0);
        this.parameterType = suppliedInputWeight.descriptor().dataType();
        this.biasConfigured = true;
        this.automaticConfiguration = null;
        parameter("inputWeight", suppliedInputWeight);
        parameter("hiddenWeight", suppliedHiddenWeight);
        parameter("bias", suppliedBias);
    }

    /**
     * Creates a cell with two Glorot-uniform weights and an optional deterministic zero bias.
     *
     * <p>Caller-controlled null, size, floating-type, checked-count, and Java-array-limit
     * validation completes before the first random draw or Tensor identifier allocation. The
     * input weight is initialized first, followed by the hidden weight using the same now-advanced
     * caller-owned source. Requested bias is initialized afterward and consumes no draw. The
     * source is never retained, reset, synchronized, split, seeded, serialized, or closed.</p>
     *
     * <p>A source failure preserves completed draws. A failure during the hidden weight can leave
     * the already created input-weight Tensor and identifier, but no partially initialized cell
     * is returned. Later allocation or identifier failures are likewise not rolled back.</p>
     *
     * @param inputSize strictly positive input-feature count
     * @param hiddenSize strictly positive hidden-feature count
     * @param bias whether to create and declare one deterministic typed-zero shared bias
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source used by both weight
     *     initializers and never retained
     * @throws NullPointerException if {@code dataType} or {@code randomGenerator} is null, checked
     *     in that order
     * @throws IllegalArgumentException if {@code inputSize} or {@code hiddenSize} is not positive,
     *     checked in that order; if {@code dataType} is not floating; or if a requested parameter
     *     Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if checked Model element-count or layout arithmetic overflows
     * @throws RuntimeException if the random source fails while sampling; completed draws remain
     *     consumed
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public RnnCell(
            long inputSize,
            long hiddenSize,
            boolean bias,
            DataType dataType,
            RandomGenerator randomGenerator) {
        DataType parameterType = Objects.requireNonNull(dataType, "dataType");
        RandomGenerator source = Objects.requireNonNull(randomGenerator, "randomGenerator");
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        if (!parameterType.isFloating()) {
            throw new IllegalArgumentException(
                    "RNN cell initialization requires floating data type: " + parameterType);
        }

        Shape inputWeightShape = Shape.of(hiddenSize, inputSize);
        Shape hiddenWeightShape = Shape.of(hiddenSize, hiddenSize);
        Shape biasShape = bias ? Shape.of(hiddenSize) : null;
        validateJavaArrayLimit(inputWeightShape);
        validateJavaArrayLimit(hiddenWeightShape);
        if (biasShape != null) {
            validateJavaArrayLimit(biasShape);
        }

        Tensor initializedInputWeight = ParameterInitializers.glorotUniform(
                inputWeightShape, parameterType, source);
        Tensor initializedHiddenWeight = ParameterInitializers.glorotUniform(
                hiddenWeightShape, parameterType, source);
        Tensor initializedBias = bias
                ? ParameterInitializers.zeros(biasShape, parameterType)
                : null;

        this.hiddenSize = hiddenSize;
        this.parameterType = parameterType;
        this.biasConfigured = bias;
        this.automaticConfiguration = null;
        parameter("inputWeight", initializedInputWeight);
        parameter("hiddenWeight", initializedHiddenWeight);
        if (bias) {
            parameter("bias", initializedBias);
        }
    }

    /**
     * Creates a cell that infers only its input width on the first compatible forward call.
     *
     * <p>Construction retains immutable configuration and reserves {@code inputWeight},
     * {@code hiddenWeight}, then optional {@code bias}. It creates no random generator, Tensor,
     * Tensor identifier, Parameter, or recurrent state. The weight policy is applied independently
     * to the eventual complete {@code [hiddenSize, inputSize]} and
     * {@code [hiddenSize, hiddenSize]} Shapes. Optional bias is always layer-owned typed zero.</p>
     *
     * @param hiddenSize strictly positive recurrent hidden width
     * @param bias whether the cell will create a complete typed-zero bias after its two weights
     * @param dataType non-null floating parameter and default-sequence-state type
     * @param weightInitialization non-null closed policy applied independently to both matrices;
     *     it owns neither Shape, source, nor seed
     * @param seed seed for exact {@code L64X128MixRandom} sampling attempts; any Java
     *     {@code long} value is accepted and zero/one policies do not use it
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null,
     *     checked in that order after {@code hiddenSize}
     * @throws IllegalArgumentException if {@code hiddenSize} is not positive, the type is not
     *     floating, or a hidden-weight or requested-bias Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if checked hidden-weight or bias Shape/count arithmetic
     *     overflows
     */
    public RnnCell(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        DataType configuredType = Objects.requireNonNull(dataType, "dataType");
        ParameterInitialization initialization =
                Objects.requireNonNull(weightInitialization, "weightInitialization");
        if (!configuredType.isFloating()) {
            throw new IllegalArgumentException(
                    "RNN cell initialization requires floating data type: " + configuredType);
        }
        Shape hiddenWeightShape = Shape.of(hiddenSize, hiddenSize);
        validateJavaArrayLimit(hiddenWeightShape);
        if (bias) {
            validateJavaArrayLimit(Shape.of(hiddenSize));
        }

        this.hiddenSize = hiddenSize;
        this.parameterType = configuredType;
        this.biasConfigured = bias;
        this.automaticConfiguration = new AutomaticConfiguration(initialization, seed);
        reserveParameter("inputWeight", this::validateAutomaticInputWeight);
        reserveParameter("hiddenWeight", this::validateAutomaticHiddenWeight);
        if (bias) {
            reserveParameter("bias", this::validateAutomaticBias);
        }
    }

    /**
     * Returns the stable input-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code inputWeight}; its current value is
     *     shaped {@code [hiddenSize, inputSize]}
     * @throws IllegalStateException if automatic forward or strict-load initialization has not
     *     published the complete parameter group
     */
    public Parameter inputWeight() {
        return boundParameter("inputWeight");
    }

    /**
     * Returns the stable hidden-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code hiddenWeight}; its current value is
     *     shaped {@code [hiddenSize, hiddenSize]}
     * @throws IllegalStateException if automatic forward or strict-load initialization has not
     *     published the complete parameter group
     */
    public Parameter hiddenWeight() {
        return boundParameter("hiddenWeight");
    }

    /**
     * Returns the optional stable shared-bias parameter wrapper.
     *
     * @return a non-null empty Optional for a no-bias cell, or an Optional containing the exact
     *     wrapper declared under {@code bias} with Shape {@code [hiddenSize]}
     * @throws IllegalStateException if bias is configured but automatic forward or strict-load
     *     initialization has not published the complete parameter group
     */
    public Optional<Parameter> bias() {
        return biasConfigured ? Optional.of(boundParameter("bias")) : Optional.empty();
    }

    /**
     * Builds one vanilla tanh recurrent-cell step from explicit input and hidden Tensors.
     *
     * <p>The method rejects nulls, snapshots current parameters once in declaration order, and
     * prevalidates both linear projections plus their ordinary ADD broadcast before constructing
     * the first expression. It then delegates exactly to the existing Model linear, ADD, and TANH
     * methods in the fixed formula documented on this class. Caller-controlled local failure
     * therefore consumes no Tensor identifier and leaves no expression prefix.</p>
     *
     * <p>Both inputs must have rank at least one and floating types. Their final Dimensions must
     * equal the configured feature sizes when both sides are static; unresolved contraction
     * equality is deferred under current Model semantics. Complete leading prefixes may differ
     * in rank or singleton extents when ordinary right-aligned broadcasting can prove them
     * compatible. The returned Tensor is both the visible output and next hidden state; the cell
     * does not retain it, so callers explicitly pass it to a later call to express recurrence.</p>
     *
     * <p>For an unbound automatic cell, all caller-controlled descriptor and Shape checks complete
     * before initialization. Random policies restart one {@code L64X128MixRandom} stream from the
     * retained seed for every attempt and use it for input weight then hidden weight; zero/one use
     * no generator, and bias consumes no draw. Publication precedes formula construction. A later
     * formula failure retains the published parameters, while a pre-publication failure leaves the
     * complete group unbound. Tensor identifiers, allocations, and completed draws are not rolled
     * back.</p>
     *
     * @param input non-null floating rank-one-or-higher Tensor whose final Dimension is compatible
     *     with {@code inputSize}; not mutated or retained by the cell
     * @param hidden non-null floating rank-one-or-higher current hidden Tensor whose final
     *     Dimension is compatible with {@code hiddenSize}; not mutated or retained by the cell
     * @return the non-null fresh TANH Tensor expression that is both output and next hidden state,
     *     with ordinary broadcast leading Shape and final extent {@code hiddenSize}
     * @throws NullPointerException if {@code input} or {@code hidden} is null, checked in that
     *     order
     * @throws IllegalArgumentException if numeric promotion, rank, static feature contraction, or
     *     projection broadcast validation fails
     * @throws IllegalStateException if Tensor identifier space is exhausted during valid
     *     initialization or expression construction; an already created prefix is not rolled back
     * @throws RuntimeException if standard random sampling fails during automatic initialization;
     *     completed draws remain consumed and no parameter wrapper is published
     * @throws OutOfMemoryError if automatic parameter or expression allocation fails; completed
     *     effects are not rolled back
     */
    public Tensor forward(Tensor input, Tensor hidden) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedHidden = Objects.requireNonNull(hidden, "hidden");

        if (automaticConfiguration != null && !parameterReservationsBound()) {
            AutomaticInput automaticInput = validateAutomaticInput(suppliedInput, suppliedHidden);
            if (!parameterReservationsBound()) {
                synchronized (this) {
                    if (!parameterReservationsBound()) {
                        automaticInput = validateAutomaticInput(suppliedInput, suppliedHidden);
                        initializeAutomatically(automaticInput);
                    }
                }
            }
        }

        Tensor currentInputWeight = inputWeight().value();
        Tensor currentHiddenWeight = hiddenWeight().value();
        Optional<Tensor> currentBias = bias().map(Parameter::value);

        DataType inputProductType = validateProjection(
                suppliedInput, currentInputWeight, "input");
        DataType inputProjectionType = currentBias
                .map(value -> DataTypePromotion.promoteNumeric(
                        inputProductType, value.descriptor().dataType()))
                .orElse(inputProductType);
        DataType hiddenProjectionType = validateProjection(
                suppliedHidden, currentHiddenWeight, "hidden");
        Shape inputProjectionShape = projectionShape(suppliedInput, currentInputWeight);
        Shape hiddenProjectionShape = projectionShape(suppliedHidden, currentHiddenWeight);
        DataTypePromotion.promoteNumeric(inputProjectionType, hiddenProjectionType);
        ShapeBroadcast.broadcast(inputProjectionShape, hiddenProjectionShape);

        Tensor projectedInput = currentBias.isPresent()
                ? suppliedInput.linear(currentInputWeight, currentBias.orElseThrow())
                : suppliedInput.linear(currentInputWeight);
        Tensor projectedHidden = suppliedHidden.linear(currentHiddenWeight);
        return projectedInput.add(projectedHidden).tanh();
    }

    long configuredHiddenSize() {
        return hiddenSize;
    }

    DataType configuredDataType() {
        return parameterType;
    }

    boolean configuredBias() {
        return biasConfigured;
    }

    void validateConfiguredInputSize(long inputSize) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("input feature size must be positive: " + inputSize);
        }
        if (parameterReservationsBound()) {
            Tensor currentInputWeight = inputWeight().value();
            Tensor currentHiddenWeight = hiddenWeight().value();
            validateWeights(currentInputWeight, currentHiddenWeight);
            bias().ifPresent(value -> validateBias(currentInputWeight, value.value()));
            long expected = extent(currentInputWeight.descriptor().shape(), 1);
            if (inputSize != expected) {
                throw new IllegalArgumentException(
                        "input feature size must equal cell inputSize: input="
                                + inputSize + ", cell=" + expected);
            }
        }
    }

    private AutomaticInput validateAutomaticInput(Tensor input, Tensor hidden) {
        Shape inputShape = input.descriptor().shape();
        if (inputShape.rank() < 1) {
            throw new IllegalArgumentException("input rank must be at least 1: " + inputShape.rank());
        }
        Dimension feature = inputShape.dimension(inputShape.rank() - 1);
        if (!(feature instanceof StaticDimension staticFeature)) {
            throw new IllegalArgumentException(
                    "RNN automatic input final feature dimension must be static: " + feature);
        }
        if (staticFeature.size() <= 0) {
            throw new IllegalArgumentException(
                    "RNN automatic input feature size must be positive: " + staticFeature.size());
        }
        Shape hiddenShape = hidden.descriptor().shape();
        if (hiddenShape.rank() < 1) {
            throw new IllegalArgumentException("hidden rank must be at least 1: " + hiddenShape.rank());
        }
        Dimension hiddenFeature = hiddenShape.dimension(hiddenShape.rank() - 1);
        if (hiddenFeature instanceof StaticDimension staticHidden
                && staticHidden.size() != hiddenSize) {
            throw new IllegalArgumentException(
                    "hidden feature dimension must equal hidden size: hidden="
                            + staticHidden.size() + ", hiddenSize=" + hiddenSize);
        }
        DataType inputProjection = DataTypePromotion.promoteNumeric(
                input.descriptor().dataType(), parameterType);
        if (biasConfigured) {
            inputProjection = DataTypePromotion.promoteNumeric(inputProjection, parameterType);
        }
        DataType hiddenProjection = DataTypePromotion.promoteNumeric(
                hidden.descriptor().dataType(), parameterType);
        DataTypePromotion.promoteNumeric(inputProjection, hiddenProjection);
        ShapeBroadcast.broadcast(
                projectionShape(inputShape, Shape.of(hiddenSize, staticFeature.size())),
                projectionShape(hiddenShape, Shape.of(hiddenSize, hiddenSize)));
        Shape inputWeightShape = Shape.of(hiddenSize, staticFeature.size());
        validateJavaArrayLimit(inputWeightShape);
        return new AutomaticInput(inputWeightShape);
    }

    private void initializeAutomatically(AutomaticInput input) {
        ParameterInitialization initialization = automaticConfiguration.initialization;
        Tensor initializedInputWeight;
        Tensor initializedHiddenWeight;
        if (initialization.requiresRandomGenerator()) {
            RandomGenerator generator = RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom")
                    .create(automaticConfiguration.seed);
            initializedInputWeight = ParameterInitializers.initialize(
                    input.inputWeightShape, parameterType, initialization, generator);
            initializedHiddenWeight = ParameterInitializers.initialize(
                    Shape.of(hiddenSize, hiddenSize), parameterType, initialization, generator);
        } else {
            initializedInputWeight = ParameterInitializers.initialize(
                    input.inputWeightShape, parameterType, initialization);
            initializedHiddenWeight = ParameterInitializers.initialize(
                    Shape.of(hiddenSize, hiddenSize), parameterType, initialization);
        }
        if (biasConfigured) {
            Tensor initializedBias = ParameterInitializers.zeros(Shape.of(hiddenSize), parameterType);
            bindReservedParameters(List.of(
                    initializedInputWeight, initializedHiddenWeight, initializedBias));
        } else {
            bindReservedParameters(List.of(initializedInputWeight, initializedHiddenWeight));
        }
    }

    private void validateAutomaticInputWeight(Tensor weight) {
        validateInputWeight(weight);
        if (weight.descriptor().dataType() != parameterType) {
            throw new IllegalArgumentException("automatic inputWeight data type mismatch");
        }
        if (extent(weight.descriptor().shape(), 0) != hiddenSize) {
            throw new IllegalArgumentException("automatic inputWeight hidden size mismatch");
        }
    }

    private void validateAutomaticHiddenWeight(Tensor weight) {
        validateHiddenWeight(weight);
        Shape shape = weight.descriptor().shape();
        if (weight.descriptor().dataType() != parameterType
                || extent(shape, 0) != hiddenSize || extent(shape, 1) != hiddenSize) {
            throw new IllegalArgumentException("automatic hiddenWeight schema mismatch");
        }
    }

    private void validateAutomaticBias(Tensor value) {
        validateFloatingAndGradient(value, "bias");
        Shape shape = value.descriptor().shape();
        if (shape.rank() != 1 || !shape.isFullyStatic()
                || value.descriptor().dataType() != parameterType
                || extent(shape, 0) != hiddenSize) {
            throw new IllegalArgumentException("automatic bias schema mismatch");
        }
    }

    private static long extent(Shape shape, int axis) {
        return ((StaticDimension) shape.dimension(axis)).size();
    }

    private record AutomaticConfiguration(ParameterInitialization initialization, long seed) {
    }

    private record AutomaticInput(Shape inputWeightShape) {
    }

    private static void validateWeights(Tensor inputWeight, Tensor hiddenWeight) {
        validateInputWeight(inputWeight);
        validateHiddenWeight(hiddenWeight);
        DataType inputType = inputWeight.descriptor().dataType();
        DataType hiddenType = hiddenWeight.descriptor().dataType();
        if (hiddenType != inputType) {
            throw new IllegalArgumentException(
                    "hiddenWeight data type must equal inputWeight data type: inputWeight="
                            + inputType + ", hiddenWeight=" + hiddenType);
        }
        Dimension hiddenSize = inputWeight.descriptor().shape().dimension(0);
        Shape hiddenShape = hiddenWeight.descriptor().shape();
        if (!hiddenShape.dimension(0).equals(hiddenSize)) {
            throw new IllegalArgumentException(
                    "hiddenWeight axis zero must equal inputWeight hidden size: hiddenWeight="
                            + hiddenShape.dimension(0) + ", inputWeight=" + hiddenSize);
        }
        if (!hiddenShape.dimension(1).equals(hiddenSize)) {
            throw new IllegalArgumentException(
                    "hiddenWeight axis one must equal inputWeight hidden size: hiddenWeight="
                            + hiddenShape.dimension(1) + ", inputWeight=" + hiddenSize);
        }
    }

    private static void validateInputWeight(Tensor weight) {
        validateFloatingAndGradient(weight, "inputWeight");
        Shape shape = weight.descriptor().shape();
        validateRankAndStatic(shape, "inputWeight");
        validatePositiveExtent(shape, 0, "inputWeight hiddenSize");
        validatePositiveExtent(shape, 1, "inputWeight inputSize");
    }

    private static void validateHiddenWeight(Tensor weight) {
        validateFloatingAndGradient(weight, "hiddenWeight");
        Shape shape = weight.descriptor().shape();
        validateRankAndStatic(shape, "hiddenWeight");
        validatePositiveExtent(shape, 0, "hiddenWeight axis zero");
        validatePositiveExtent(shape, 1, "hiddenWeight axis one");
    }

    private static void validateFloatingAndGradient(Tensor tensor, String name) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    name + " must have a floating data type: " + dataType);
        }
        if (!tensor.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(name + " must have requiresGrad == true");
        }
    }

    private static void validateRankAndStatic(Shape shape, String name) {
        if (shape.rank() != 2) {
            throw new IllegalArgumentException(name + " must have rank two: " + shape.rank());
        }
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(name + " must have a fully static shape: " + shape);
        }
    }

    private static void validatePositiveExtent(Shape shape, int axis, String name) {
        long size = ((StaticDimension) shape.dimension(axis)).size();
        if (size == 0) {
            throw new IllegalArgumentException(name + " must be positive: " + size);
        }
    }

    private static void validateBias(Tensor inputWeight, Tensor bias) {
        validateFloatingAndGradient(bias, "bias");
        Shape biasShape = bias.descriptor().shape();
        if (biasShape.rank() != 1) {
            throw new IllegalArgumentException("bias must have rank one: " + biasShape.rank());
        }
        if (!biasShape.isFullyStatic()) {
            throw new IllegalArgumentException("bias must have a fully static shape: " + biasShape);
        }
        DataType inputType = inputWeight.descriptor().dataType();
        DataType biasType = bias.descriptor().dataType();
        if (biasType != inputType) {
            throw new IllegalArgumentException(
                    "bias data type must equal weight data type: weight="
                            + inputType + ", bias=" + biasType);
        }
        Dimension hiddenSize = inputWeight.descriptor().shape().dimension(0);
        if (!biasShape.dimension(0).equals(hiddenSize)) {
            throw new IllegalArgumentException(
                    "bias dimension must equal hidden size: bias="
                            + biasShape.dimension(0) + ", hiddenSize=" + hiddenSize);
        }
    }

    private static void validateJavaArrayLimit(Shape shape) {
        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "initialized parameter shape exceeds Java array limit: shape="
                            + shape + ", count=" + elementCount);
        }
    }

    private static DataType validateProjection(Tensor value, Tensor weight, String name) {
        DataType productType = DataTypePromotion.promoteNumeric(
                value.descriptor().dataType(), weight.descriptor().dataType());
        Shape valueShape = value.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        int rank = valueShape.rank();
        if (rank < 1) {
            throw new IllegalArgumentException(name + " rank must be at least 1: " + rank);
        }
        Dimension features = valueShape.dimension(rank - 1);
        Dimension expectedFeatures = weightShape.dimension(1);
        if (features instanceof StaticDimension staticFeatures
                && expectedFeatures instanceof StaticDimension staticExpected
                && staticFeatures.size() != staticExpected.size()) {
            throw new IllegalArgumentException(
                    name + " feature dimension must match weight in-features dimension: "
                            + name + "=" + features + ", weight=" + expectedFeatures);
        }

        return productType;
    }

    private static Shape projectionShape(Tensor value, Tensor weight) {
        return projectionShape(value.descriptor().shape(), weight.descriptor().shape());
    }

    private static Shape projectionShape(Shape valueShape, Shape weightShape) {
        int rank = valueShape.rank();
        Dimension[] resultDimensions = new Dimension[rank];
        for (int axis = 0; axis < rank - 1; axis++) {
            resultDimensions[axis] = valueShape.dimension(axis);
        }
        resultDimensions[rank - 1] = weightShape.dimension(0);
        return Shape.ofDimensions(resultDimensions);
    }
}

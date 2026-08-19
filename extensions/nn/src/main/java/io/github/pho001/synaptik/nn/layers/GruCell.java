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
 * Constructs one explicit-state application of a reset-after gated recurrent unit (GRU) cell.
 *
 * <p>The cell owns packed {@code inputWeight}, {@code hiddenWeight}, and optional input-side
 * {@code bias} parameters. Their Shapes are {@code [3 * hiddenSize, inputSize]},
 * {@code [3 * hiddenSize, hiddenSize]}, and {@code [3 * hiddenSize]}. Axis zero is packed in
 * reset, update, candidate order. The caller supplies the current hidden Tensor to every
 * {@link #forward(Tensor, Tensor)} call and receives the next hidden Tensor as the sole result;
 * the cell never retains recurrent state.</p>
 *
 * <p>For packed input projection {@code p_x} and packed hidden projection {@code p_h}, forward
 * slices reset ({@code r}), update ({@code z}), and candidate ({@code n}) lanes and uses
 * {@code r = sigmoid(x_r + h_r)}, {@code z = sigmoid(x_z + h_z)},
 * {@code n = tanh(x_n + r * h_n)}, and {@code n + z * (hidden - n)}. Reset is therefore applied
 * after the recurrent candidate projection, and update one retains the old hidden value. The
 * optional packed bias belongs only to the input projection. Leading Dimensions use ordinary
 * right-aligned broadcasting and are never interpreted as time.</p>
 *
 * <p>Each call snapshots current parameter bindings once in declaration order. Compatible
 * replacement affects later calls, while existing expressions retain earlier exact references.
 * Replacement and forward are not thread-safe as one multi-parameter snapshot. Forward is
 * mode-insensitive and constructs Model expression metadata only; it does not execute values,
 * define gradients, capture a graph, lower operations, or select a backend. Its two-input
 * contract makes this a direct {@link Module}, not a
 * {@link io.github.pho001.synaptik.nn.module.UnaryTensorModule} accepted by
 * {@link io.github.pho001.synaptik.nn.module.Sequential}.</p>
 *
 * <p>The automatic constructor reserves all packed parameters without creating a Tensor or
 * {@link Parameter}. The first compatible represented step infers only the positive static final
 * input extent. A sampling policy creates one fresh {@code L64X128MixRandom} stream and applies
 * it first to the complete packed input matrix and then to the complete packed hidden matrix;
 * zero/one policies create no generator. Optional packed bias is always typed zero. The complete
 * group is published before the step formula is built, or strict state loading may bind it
 * without initialization. Failed attempts publish no partial group and are retryable.</p>
 */
public final class GruCell extends Module {

    private final long hiddenSize;
    private final long packedHiddenSize;
    private final DataType parameterType;
    private final boolean biasConfigured;
    private final AutomaticConfiguration automaticConfiguration;

    /**
     * Creates a no-bias cell from exact caller-supplied packed projection weights.
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [3 * hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type, shaped
     *     {@code [3 * hiddenSize, hiddenSize]}, and otherwise satisfying the packed schema;
     *     retained exactly
     * @throws NullPointerException if {@code inputWeight} or {@code hiddenWeight} is null, checked
     *     in that order
     * @throws IllegalArgumentException if either parameter violates its documented type,
     *     gradient, rank, static-Shape, positive-extent, packing, exact-type, or hidden-size rule
     */
    public GruCell(Tensor inputWeight, Tensor hiddenWeight) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        this.packedHiddenSize = extent(suppliedInputWeight.descriptor().shape(), 0);
        this.hiddenSize = packedHiddenSize / 3L;
        this.parameterType = suppliedInputWeight.descriptor().dataType();
        this.biasConfigured = false;
        this.automaticConfiguration = null;
        parameter("inputWeight", suppliedInputWeight);
        parameter("hiddenWeight", suppliedHiddenWeight);
    }

    /**
     * Creates a biased cell from exact caller-supplied packed weights and input-side bias.
     *
     * <p>Null bias never means absence. Complete validation precedes declaration in
     * {@code inputWeight}, {@code hiddenWeight}, {@code bias} order.</p>
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [3 * hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type, shaped
     *     {@code [3 * hiddenSize, hiddenSize]}, and otherwise satisfying the packed schema;
     *     retained exactly
     * @param bias non-null floating, gradient-eligible, fully static rank-one Tensor shaped
     *     {@code [3 * hiddenSize]} with the exact common weight type; retained exactly
     * @throws NullPointerException if {@code inputWeight}, {@code hiddenWeight}, or {@code bias}
     *     is null, checked in that order
     * @throws IllegalArgumentException if a parameter violates its documented type, gradient,
     *     rank, static-Shape, positive-extent, packing, exact-type, or hidden-size rule
     */
    public GruCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        validateBias(suppliedInputWeight, suppliedBias);
        this.packedHiddenSize = extent(suppliedInputWeight.descriptor().shape(), 0);
        this.hiddenSize = packedHiddenSize / 3L;
        this.parameterType = suppliedInputWeight.descriptor().dataType();
        this.biasConfigured = true;
        this.automaticConfiguration = null;
        parameter("inputWeight", suppliedInputWeight);
        parameter("hiddenWeight", suppliedHiddenWeight);
        parameter("bias", suppliedBias);
    }

    /**
     * Creates a cell with two Glorot-uniform packed weights and optional typed-zero bias.
     *
     * <p>Null, size, floating-type, packed-size, checked-count, and Java-array-limit validation
     * completes before the first random draw or Tensor identifier allocation. Input weight is
     * initialized first and hidden weight second with the same now-advanced caller-owned source;
     * requested bias follows without a draw. The source is never retained or managed.</p>
     *
     * <p>Completed source calls and identifiers are not rolled back after a later source,
     * allocation, or identifier failure, and no partially initialized cell is returned.</p>
     *
     * @param inputSize strictly positive input-feature count
     * @param hiddenSize strictly positive hidden-feature count
     * @param bias whether to create one deterministic packed input-side bias
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source used by both weights and
     *     never retained
     * @throws NullPointerException if {@code dataType} or {@code randomGenerator} is null, checked
     *     in that order
     * @throws IllegalArgumentException if a size is not positive, the type is not floating, or a
     *     requested parameter Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if {@code 3 * hiddenSize} or checked Shape arithmetic overflows
     * @throws RuntimeException if the random source fails; completed calls remain consumed
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public GruCell(
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
                    "GRU cell initialization requires floating data type: " + parameterType);
        }

        long packedHiddenSize = Math.multiplyExact(hiddenSize, 3L);
        Shape inputWeightShape = Shape.of(packedHiddenSize, inputSize);
        Shape hiddenWeightShape = Shape.of(packedHiddenSize, hiddenSize);
        Shape biasShape = bias ? Shape.of(packedHiddenSize) : null;
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
        this.packedHiddenSize = packedHiddenSize;
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
     * Creates a GRU cell that infers only its input width on first represented forward use.
     *
     * <p>Construction retains immutable configuration and reserves {@code inputWeight},
     * {@code hiddenWeight}, then optional {@code bias}. It creates no source, Tensor, identifier,
     * Parameter, or recurrent state. The selected policy is applied independently to complete
     * packed Shapes, so fan presets derive their fan values separately for each matrix.</p>
     *
     * @param hiddenSize strictly positive hidden width
     * @param bias whether to create a complete packed typed-zero input-side bias
     * @param dataType non-null floating parameter and default-sequence-state type
     * @param weightInitialization non-null closed policy applied independently to both packed
     *     matrices; it owns neither Shape, gate order, source, nor seed
     * @param seed seed for exact {@code L64X128MixRandom} sampling attempts; any Java
     *     {@code long} value is accepted and zero/one policies do not use it
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null,
     *     checked in that order after {@code hiddenSize}
     * @throws IllegalArgumentException if {@code hiddenSize} is not positive, the type is not
     *     floating, or a hidden-weight or requested-bias Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if {@code 3 * hiddenSize} or checked Shape/count arithmetic
     *     overflows
     */
    public GruCell(
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
                    "GRU cell initialization requires floating data type: " + configuredType);
        }
        long packedSize = Math.multiplyExact(hiddenSize, 3L);
        validateJavaArrayLimit(Shape.of(packedSize, hiddenSize));
        if (bias) {
            validateJavaArrayLimit(Shape.of(packedSize));
        }

        this.hiddenSize = hiddenSize;
        this.packedHiddenSize = packedSize;
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
     * Returns the stable packed input-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code inputWeight}; its current value is
     *     shaped {@code [3 * hiddenSize, inputSize]} in reset, update, candidate order
     * @throws IllegalStateException if automatic forward or strict-load initialization has not
     *     published the complete parameter group
     */
    public Parameter inputWeight() {
        return boundParameter("inputWeight");
    }

    /**
     * Returns the stable packed hidden-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code hiddenWeight}; its current value is
     *     shaped {@code [3 * hiddenSize, hiddenSize]} in reset, update, candidate order
     * @throws IllegalStateException if automatic forward or strict-load initialization has not
     *     published the complete parameter group
     */
    public Parameter hiddenWeight() {
        return boundParameter("hiddenWeight");
    }

    /**
     * Returns the optional stable packed input-side bias wrapper.
     *
     * @return a non-null empty Optional for a no-bias cell, or the exact wrapper declared under
     *     {@code bias} whose current input-side value is shaped {@code [3 * hiddenSize]} in reset,
     *     update, candidate order
     * @throws IllegalStateException if bias is configured but automatic forward or strict-load
     *     initialization has not published the complete parameter group
     */
    public Optional<Parameter> bias() {
        return biasConfigured ? Optional.of(boundParameter("bias")) : Optional.empty();
    }

    /**
     * Builds one reset-after GRU step from explicit input and hidden Tensors.
     *
     * <p>The method rejects nulls, snapshots current parameters once, and prevalidates their
     * current schema, both packed projections, six independent final-axis gate slices, and every
     * gate and interpolation promotion/broadcast before creating the first expression. It then
     * constructs the input reset, update, and candidate slices followed by the corresponding
     * hidden slices and the exact equations documented on this class. Caller-controlled local
     * failure therefore consumes no Tensor identifier and leaves no expression prefix.</p>
     *
     * <p>Both inputs must be floating and have rank at least one. A statically known final
     * Dimension must equal its configured feature count; an unresolved contraction is retained
     * for later validation under Model rules. Leading prefixes may differ when conservative
     * right-aligned broadcasting proves compatibility. The result is both cell output and next
     * hidden state and is never retained by the cell.</p>
     *
     * <p>For an unbound automatic cell, all caller-controlled descriptor and Shape checks finish
     * before initialization. Each attempt restarts the retained seed: sampling policies use one
     * {@code L64X128MixRandom} stream for input weight then hidden weight, zero/one use no
     * generator, and bias consumes no draw. Complete publication precedes formula construction.
     * A later formula failure retains the published parameters; a pre-publication failure leaves
     * every reservation unbound and retryable. Allocations, identifiers, and completed draws are
     * not rolled back.</p>
     *
     * @param input non-null floating rank-one-or-higher Tensor whose final Dimension is compatible
     *     with {@code inputSize}; not mutated or retained
     * @param hidden non-null floating rank-one-or-higher current hidden Tensor whose final
     *     Dimension is compatible with {@code hiddenSize}; not mutated or retained
     * @return the non-null fresh final ADD expression, both output and next hidden state, with
     *     broadcast leading Shape and final extent {@code hiddenSize}
     * @throws NullPointerException if {@code input} or {@code hidden} is null, checked in that
     *     order
     * @throws IllegalArgumentException if current parameter schema, promotion, rank, static
     *     contraction, gate slicing, or any required broadcast validation fails
     * @throws ArithmeticException if checked packed-bound or Shape arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted during valid
     *     construction; an already created prefix is not rolled back
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
            synchronized (this) {
                if (!parameterReservationsBound()) {
                    automaticInput = validateAutomaticInput(suppliedInput, suppliedHidden);
                    initializeAutomatically(automaticInput);
                }
            }
        }

        Tensor currentInputWeight = inputWeight().value();
        Tensor currentHiddenWeight = hiddenWeight().value();
        Optional<Tensor> currentBias = bias().map(Parameter::value);

        long hiddenSize = validateWeights(currentInputWeight, currentHiddenWeight);
        currentBias.ifPresent(value -> validateBias(currentInputWeight, value));
        long twiceHiddenSize = Math.multiplyExact(hiddenSize, 2L);
        long packedHiddenSize = ((StaticDimension)
                currentInputWeight.descriptor().shape().dimension(0)).size();

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
        Shape inputGateShape = gateShape(inputProjectionShape, hiddenSize);
        Shape hiddenGateShape = gateShape(hiddenProjectionShape, hiddenSize);

        DataType resetType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape resetShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(resetType, "reset preactivation");
        DataType updateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape updateShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(updateType, "update preactivation");

        DataType resetProductType = DataTypePromotion.promoteNumeric(
                resetType, hiddenProjectionType);
        Shape resetProductShape = ShapeBroadcast.broadcast(resetShape, hiddenGateShape);
        DataType candidateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, resetProductType);
        Shape candidateShape = ShapeBroadcast.broadcast(inputGateShape, resetProductShape);
        requireFloating(candidateType, "candidate preactivation");

        DataType differenceType = DataTypePromotion.promoteNumeric(
                suppliedHidden.descriptor().dataType(), candidateType);
        Shape differenceShape = ShapeBroadcast.broadcast(
                suppliedHidden.descriptor().shape(), candidateShape);
        DataType weightedDifferenceType = DataTypePromotion.promoteNumeric(
                updateType, differenceType);
        Shape weightedDifferenceShape = ShapeBroadcast.broadcast(updateShape, differenceShape);
        DataTypePromotion.promoteNumeric(candidateType, weightedDifferenceType);
        ShapeBroadcast.broadcast(candidateShape, weightedDifferenceShape);

        Tensor inputProjection = currentBias.isPresent()
                ? suppliedInput.linear(currentInputWeight, currentBias.orElseThrow())
                : suppliedInput.linear(currentInputWeight);
        Tensor hiddenProjection = suppliedHidden.linear(currentHiddenWeight);

        Tensor inputReset = inputProjection.sliceAxis(-1, 0L, hiddenSize);
        Tensor inputUpdate = inputProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
        Tensor inputCandidate = inputProjection.sliceAxis(
                -1, twiceHiddenSize, packedHiddenSize);
        Tensor hiddenReset = hiddenProjection.sliceAxis(-1, 0L, hiddenSize);
        Tensor hiddenUpdate = hiddenProjection.sliceAxis(-1, hiddenSize, twiceHiddenSize);
        Tensor hiddenCandidate = hiddenProjection.sliceAxis(
                -1, twiceHiddenSize, packedHiddenSize);

        Tensor reset = inputReset.add(hiddenReset).sigmoid();
        Tensor update = inputUpdate.add(hiddenUpdate).sigmoid();
        Tensor candidate = inputCandidate.add(reset.mul(hiddenCandidate)).tanh();
        return candidate.add(update.mul(suppliedHidden.sub(candidate)));
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
                    "GRU automatic input final feature dimension must be static: " + feature);
        }
        long inputSize = staticFeature.size();
        if (inputSize <= 0) {
            throw new IllegalArgumentException(
                    "GRU automatic input feature size must be positive: " + inputSize);
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

        Shape inputWeightShape = Shape.of(packedHiddenSize, inputSize);
        Shape hiddenWeightShape = Shape.of(packedHiddenSize, hiddenSize);
        validateJavaArrayLimit(inputWeightShape);
        prevalidateFormula(
                input.descriptor().dataType(), inputShape,
                hidden.descriptor().dataType(), hiddenShape,
                inputWeightShape, hiddenWeightShape);
        return new AutomaticInput(inputWeightShape);
    }

    private void prevalidateFormula(
            DataType inputType,
            Shape inputShape,
            DataType hiddenType,
            Shape hiddenShape,
            Shape inputWeightShape,
            Shape hiddenWeightShape) {
        DataType inputProjectionType = DataTypePromotion.promoteNumeric(inputType, parameterType);
        if (biasConfigured) {
            inputProjectionType = DataTypePromotion.promoteNumeric(inputProjectionType, parameterType);
        }
        DataType hiddenProjectionType = DataTypePromotion.promoteNumeric(hiddenType, parameterType);
        Shape inputGateShape = gateShape(projectionShape(inputShape, inputWeightShape), hiddenSize);
        Shape hiddenGateShape = gateShape(projectionShape(hiddenShape, hiddenWeightShape), hiddenSize);
        DataType resetType = DataTypePromotion.promoteNumeric(inputProjectionType, hiddenProjectionType);
        Shape resetShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(resetType, "reset preactivation");
        DataType updateType = DataTypePromotion.promoteNumeric(inputProjectionType, hiddenProjectionType);
        Shape updateShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(updateType, "update preactivation");
        DataType resetProductType = DataTypePromotion.promoteNumeric(resetType, hiddenProjectionType);
        Shape resetProductShape = ShapeBroadcast.broadcast(resetShape, hiddenGateShape);
        DataType candidateType = DataTypePromotion.promoteNumeric(inputProjectionType, resetProductType);
        Shape candidateShape = ShapeBroadcast.broadcast(inputGateShape, resetProductShape);
        requireFloating(candidateType, "candidate preactivation");
        DataType differenceType = DataTypePromotion.promoteNumeric(hiddenType, candidateType);
        Shape differenceShape = ShapeBroadcast.broadcast(hiddenShape, candidateShape);
        DataType weightedType = DataTypePromotion.promoteNumeric(updateType, differenceType);
        Shape weightedShape = ShapeBroadcast.broadcast(updateShape, differenceShape);
        DataTypePromotion.promoteNumeric(candidateType, weightedType);
        ShapeBroadcast.broadcast(candidateShape, weightedShape);
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
                    Shape.of(packedHiddenSize, hiddenSize), parameterType, initialization, generator);
        } else {
            initializedInputWeight = ParameterInitializers.initialize(
                    input.inputWeightShape, parameterType, initialization);
            initializedHiddenWeight = ParameterInitializers.initialize(
                    Shape.of(packedHiddenSize, hiddenSize), parameterType, initialization);
        }
        if (biasConfigured) {
            Tensor initializedBias = ParameterInitializers.zeros(
                    Shape.of(packedHiddenSize), parameterType);
            bindReservedParameters(List.of(
                    initializedInputWeight, initializedHiddenWeight, initializedBias));
        } else {
            bindReservedParameters(List.of(initializedInputWeight, initializedHiddenWeight));
        }
    }

    private void validateAutomaticInputWeight(Tensor weight) {
        long actualHiddenSize = validateInputWeight(weight);
        Shape shape = weight.descriptor().shape();
        if (weight.descriptor().dataType() != parameterType
                || actualHiddenSize != hiddenSize
                || extent(shape, 0) != packedHiddenSize) {
            throw new IllegalArgumentException("automatic inputWeight schema mismatch");
        }
    }

    private void validateAutomaticHiddenWeight(Tensor weight) {
        validateHiddenWeight(weight);
        Shape shape = weight.descriptor().shape();
        if (weight.descriptor().dataType() != parameterType
                || extent(shape, 0) != packedHiddenSize
                || extent(shape, 1) != hiddenSize) {
            throw new IllegalArgumentException("automatic hiddenWeight schema mismatch");
        }
    }

    private void validateAutomaticBias(Tensor value) {
        validateFloatingAndGradient(value, "bias");
        Shape shape = value.descriptor().shape();
        if (shape.rank() != 1 || !shape.isFullyStatic()
                || value.descriptor().dataType() != parameterType
                || extent(shape, 0) != packedHiddenSize) {
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

    private static long validateWeights(Tensor inputWeight, Tensor hiddenWeight) {
        long hiddenSize = validateInputWeight(inputWeight);
        validateHiddenWeight(hiddenWeight);
        DataType inputType = inputWeight.descriptor().dataType();
        DataType hiddenType = hiddenWeight.descriptor().dataType();
        if (hiddenType != inputType) {
            throw new IllegalArgumentException(
                    "hiddenWeight data type must equal inputWeight data type: inputWeight="
                            + inputType + ", hiddenWeight=" + hiddenType);
        }
        Shape inputShape = inputWeight.descriptor().shape();
        Shape hiddenShape = hiddenWeight.descriptor().shape();
        if (!hiddenShape.dimension(0).equals(inputShape.dimension(0))) {
            throw new IllegalArgumentException(
                    "hiddenWeight axis zero must equal inputWeight packed extent: hiddenWeight="
                            + hiddenShape.dimension(0) + ", inputWeight=" + inputShape.dimension(0));
        }
        StaticDimension expectedHiddenSize = new StaticDimension(hiddenSize);
        if (!hiddenShape.dimension(1).equals(expectedHiddenSize)) {
            throw new IllegalArgumentException(
                    "hiddenWeight axis one must equal derived hidden size: hiddenWeight="
                            + hiddenShape.dimension(1) + ", hiddenSize=" + expectedHiddenSize);
        }
        return hiddenSize;
    }

    private static long validateInputWeight(Tensor weight) {
        validateFloatingAndGradient(weight, "inputWeight");
        Shape shape = weight.descriptor().shape();
        validateRankAndStatic(shape, "inputWeight");
        long packedHiddenSize = validatePositiveExtent(
                shape, 0, "inputWeight packed hidden size");
        if (packedHiddenSize % 3L != 0L) {
            throw new IllegalArgumentException(
                    "inputWeight packed hidden size must be divisible by three: "
                            + packedHiddenSize);
        }
        validatePositiveExtent(shape, 1, "inputWeight inputSize");
        return packedHiddenSize / 3L;
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

    private static long validatePositiveExtent(Shape shape, int axis, String name) {
        long size = ((StaticDimension) shape.dimension(axis)).size();
        if (size == 0) {
            throw new IllegalArgumentException(name + " must be positive: " + size);
        }
        return size;
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
        Dimension packedHiddenSize = inputWeight.descriptor().shape().dimension(0);
        if (!biasShape.dimension(0).equals(packedHiddenSize)) {
            throw new IllegalArgumentException(
                    "bias dimension must equal packed hidden size: bias="
                            + biasShape.dimension(0) + ", packedHiddenSize=" + packedHiddenSize);
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
        Dimension[] resultDimensions = new Dimension[valueShape.rank()];
        for (int axis = 0; axis < valueShape.rank() - 1; axis++) {
            resultDimensions[axis] = valueShape.dimension(axis);
        }
        resultDimensions[valueShape.rank() - 1] = weightShape.dimension(0);
        return Shape.ofDimensions(resultDimensions);
    }

    private static Shape gateShape(Shape projectionShape, long hiddenSize) {
        int rank = projectionShape.rank();
        Dimension[] resultDimensions = new Dimension[rank];
        for (int axis = 0; axis < rank - 1; axis++) {
            resultDimensions[axis] = projectionShape.dimension(axis);
        }
        resultDimensions[rank - 1] = new StaticDimension(hiddenSize);
        return Shape.ofDimensions(resultDimensions);
    }

    private static void requireFloating(DataType dataType, String name) {
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    name + " must have a floating data type: " + dataType);
        }
    }
}

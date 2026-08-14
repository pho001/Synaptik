package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * Constructs one explicit-state long short-term memory (LSTM) cell application.
 *
 * <p>The cell owns packed {@code inputWeight}, {@code hiddenWeight}, and optional input-side
 * {@code bias} parameters shaped {@code [4 * hiddenSize, inputSize]},
 * {@code [4 * hiddenSize, hiddenSize]}, and {@code [4 * hiddenSize]}. Their packed axis uses
 * input, forget, candidate, output gate order. There is no hidden-side bias. Initialized cells
 * draw the complete input matrix and then the complete hidden matrix with Glorot uniform from one
 * caller-owned source; the optional complete bias follows as typed zero, including its forget
 * interval, without a draw. This schema is not claimed to match frameworks with another packing,
 * bias association, equation, or initialization policy.</p>
 *
 * <p>For packed projections {@code p_x} and {@code p_h}, forward builds
 * {@code i = sigmoid(x_i + h_i)}, {@code f = sigmoid(x_f + h_f)},
 * {@code g = tanh(x_g + h_g)}, {@code o = sigmoid(x_o + h_o)},
 * {@code nextCell = f * cell + i * g}, and
 * {@code nextHidden = o * tanh(nextCell)}. The caller supplies and receives both recurrent states
 * explicitly; this module never retains, initializes, updates, registers, or discovers them.
 * Leading Dimensions use ordinary right-aligned broadcasting and are never traversed as time.</p>
 *
 * <p>Each call reads current parameter bindings once in declaration order. Replacement affects
 * later calls while existing expressions retain their exact earlier inputs. Replacement and
 * forward are not one thread-safe multi-parameter snapshot, so callers coordinate them when a
 * consistent view matters. Forward is mode-insensitive and creates Model expression metadata
 * only; it does not evaluate values, define gradients, capture a graph, lower operations, or
 * execute work. Its three-input contract makes this a direct {@link Module}, not a
 * {@link io.github.pho001.synaptik.nn.module.UnaryTensorModule} accepted by
 * {@link io.github.pho001.synaptik.nn.module.Sequential}.</p>
 */
public final class LstmCell extends Module {
    private final Parameter inputWeight;
    private final Parameter hiddenWeight;
    private final Optional<Parameter> bias;

    /**
     * Creates a no-bias cell from exact caller-supplied packed projection weights.
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [4 * hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type, shaped
     *     {@code [4 * hiddenSize, hiddenSize]}, and otherwise satisfying the packed schema;
     *     retained exactly
     * @throws NullPointerException if {@code inputWeight} or {@code hiddenWeight} is null, checked
     *     in that order
     * @throws IllegalArgumentException if either parameter violates its documented type,
     *     gradient, rank, static-Shape, positive-extent, packing, exact-type, or hidden-size rule
     */
    public LstmCell(Tensor inputWeight, Tensor hiddenWeight) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        this.inputWeight = parameter("inputWeight", suppliedInputWeight);
        this.hiddenWeight = parameter("hiddenWeight", suppliedHiddenWeight);
        this.bias = Optional.empty();
    }

    /**
     * Creates a biased cell from exact caller-supplied packed weights and input-side bias.
     *
     * <p>Null bias never means absence. Complete validation precedes declaration in
     * {@code inputWeight}, {@code hiddenWeight}, {@code bias} order.</p>
     *
     * @param inputWeight non-null floating, gradient-eligible, fully static positive rank-two
     *     Tensor shaped {@code [4 * hiddenSize, inputSize]}; retained exactly
     * @param hiddenWeight non-null Tensor with the same type, shaped
     *     {@code [4 * hiddenSize, hiddenSize]}, and otherwise satisfying the packed schema;
     *     retained exactly
     * @param bias non-null floating, gradient-eligible, fully static rank-one Tensor shaped
     *     {@code [4 * hiddenSize]} with the exact common weight type; retained exactly
     * @throws NullPointerException if {@code inputWeight}, {@code hiddenWeight}, or {@code bias}
     *     is null, checked in that order
     * @throws IllegalArgumentException if a parameter violates its documented type, gradient,
     *     rank, static-Shape, positive-extent, packing, exact-type, or hidden-size rule
     */
    public LstmCell(Tensor inputWeight, Tensor hiddenWeight, Tensor bias) {
        Tensor suppliedInputWeight = Objects.requireNonNull(inputWeight, "inputWeight");
        Tensor suppliedHiddenWeight = Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        validateWeights(suppliedInputWeight, suppliedHiddenWeight);
        validateBias(suppliedInputWeight, suppliedBias);
        this.inputWeight = parameter("inputWeight", suppliedInputWeight);
        this.hiddenWeight = parameter("hiddenWeight", suppliedHiddenWeight);
        this.bias = Optional.of(parameter("bias", suppliedBias));
    }

    /**
     * Creates a cell with two Glorot-uniform packed weights and optional typed-zero bias.
     *
     * <p>Null, size, floating-type, packed-size, checked-count, and Java-array-limit validation
     * completes before the first random draw or Tensor identifier allocation. Input weight is
     * initialized first and hidden weight second with the same advanced caller-owned source;
     * requested bias follows without a draw. The source is never retained, reset, synchronized,
     * split, seeded, serialized, or closed. Completed draws and identifiers are not rolled back
     * after a later source, allocation, or identifier failure.</p>
     *
     * @param inputSize strictly positive input-feature count
     * @param hiddenSize strictly positive hidden-feature and per-gate count
     * @param bias whether to create one deterministic all-zero packed input-side bias
     * @param dataType non-null floating parameter type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source used by both weights and
     *     never retained
     * @throws NullPointerException if {@code dataType} or {@code randomGenerator} is null, checked
     *     in that order
     * @throws IllegalArgumentException if a size is not positive, the type is not floating, or a
     *     requested parameter Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if {@code 4 * hiddenSize} or checked Shape arithmetic overflows
     * @throws RuntimeException if the random source fails; completed calls remain consumed
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public LstmCell(
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
                    "LSTM cell initialization requires floating data type: " + parameterType);
        }

        long packedHiddenSize = Math.multiplyExact(hiddenSize, 4L);
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

        this.inputWeight = parameter("inputWeight", initializedInputWeight);
        this.hiddenWeight = parameter("hiddenWeight", initializedHiddenWeight);
        this.bias = bias
                ? Optional.of(parameter("bias", initializedBias))
                : Optional.empty();
    }

    /**
     * Returns the stable packed input-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code inputWeight}; its current value is
     *     shaped {@code [4 * hiddenSize, inputSize]} in input, forget, candidate, output order
     */
    public Parameter inputWeight() {
        return inputWeight;
    }

    /**
     * Returns the stable packed hidden-projection parameter wrapper.
     *
     * @return the exact non-null wrapper declared under {@code hiddenWeight}; its current value is
     *     shaped {@code [4 * hiddenSize, hiddenSize]} in input, forget, candidate, output order
     */
    public Parameter hiddenWeight() {
        return hiddenWeight;
    }

    /**
     * Returns the optional stable packed input-side bias wrapper.
     *
     * @return a non-null empty Optional for a no-bias cell, or the exact wrapper declared under
     *     {@code bias} whose current value is shaped {@code [4 * hiddenSize]}
     */
    public Optional<Parameter> bias() {
        return bias;
    }

    /**
     * Builds one LSTM step from explicit input, hidden, and cell Tensors.
     *
     * <p>The method rejects nulls, reads current parameters once, and prevalidates their schema,
     * both projections, eight independent final-axis slices, every gate, and both state equations
     * before creating the first expression. Caller-controlled local failure therefore consumes no
     * Tensor identifier and leaves no expression prefix. Construction then follows the exact
     * input/forget/candidate/output slicing and equations documented on this class.</p>
     *
     * <p>All three inputs must be floating and rank one or higher. A statically known final
     * Dimension must equal its configured feature count. An unresolved input or hidden
     * contraction may remain deferred by current Model linear rules, but all later gate/state
     * broadcasts must be locally provable; in particular, an unresolved cell-feature Dimension
     * paired with the static gate width is rejected. Leading prefixes may differ when
     * conservative right-aligned broadcasting proves compatibility. The returned carrier
     * contains the exact final hidden MUL and cell ADD expressions and is never retained by the
     * module.</p>
     *
     * @param input non-null floating rank-one-or-higher Tensor whose final Dimension is compatible
     *     with {@code inputSize}; not mutated or retained
     * @param hidden non-null floating rank-one-or-higher current hidden Tensor whose final
     *     Dimension is compatible with {@code hiddenSize}; not mutated or retained
     * @param cell non-null floating rank-one-or-higher current cell Tensor whose final Dimension
     *     is compatible with {@code hiddenSize}; not mutated or retained
     * @return one non-null fresh carrier retaining the exact next-hidden and next-cell expressions
     * @throws NullPointerException if {@code input}, {@code hidden}, or {@code cell} is null,
     *     checked in that order
     * @throws IllegalArgumentException if current parameter schema, promotion, rank, static
     *     contraction, gate slicing, or any required broadcast validation fails
     * @throws ArithmeticException if checked packed-bound or Shape arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted during valid
     *     construction; an already created prefix is not rolled back
     */
    public LstmCellForwardResult forward(Tensor input, Tensor hidden, Tensor cell) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedHidden = Objects.requireNonNull(hidden, "hidden");
        Tensor suppliedCell = Objects.requireNonNull(cell, "cell");

        Tensor currentInputWeight = inputWeight.value();
        Tensor currentHiddenWeight = hiddenWeight.value();
        Optional<Tensor> currentBias = bias.map(Parameter::value);

        long hiddenSize = validateWeights(currentInputWeight, currentHiddenWeight);
        currentBias.ifPresent(value -> validateBias(currentInputWeight, value));
        long twiceHiddenSize = Math.multiplyExact(hiddenSize, 2L);
        long thriceHiddenSize = Math.multiplyExact(hiddenSize, 3L);
        long packedHiddenSize = Math.multiplyExact(hiddenSize, 4L);

        DataType inputProductType = validateProjection(
                suppliedInput, currentInputWeight, "input");
        DataType inputProjectionType = currentBias
                .map(value -> DataTypePromotion.promoteNumeric(
                        inputProductType, value.descriptor().dataType()))
                .orElse(inputProductType);
        DataType hiddenProjectionType = validateProjection(
                suppliedHidden, currentHiddenWeight, "hidden");
        validateState(suppliedCell, hiddenSize, "cell");

        Shape inputProjectionShape = projectionShape(suppliedInput, currentInputWeight);
        Shape hiddenProjectionShape = projectionShape(suppliedHidden, currentHiddenWeight);
        Shape inputGateShape = gateShape(inputProjectionShape, hiddenSize);
        Shape hiddenGateShape = gateShape(hiddenProjectionShape, hiddenSize);

        DataType inputGateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape inputGateResultShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(inputGateType, "input gate preactivation");
        DataType forgetGateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape forgetGateResultShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(forgetGateType, "forget gate preactivation");
        DataType candidateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape candidateResultShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(candidateType, "candidate preactivation");
        DataType outputGateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape outputGateResultShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(outputGateType, "output gate preactivation");

        DataType forgetProductType = DataTypePromotion.promoteNumeric(
                forgetGateType, suppliedCell.descriptor().dataType());
        Shape forgetProductShape = ShapeBroadcast.broadcast(
                forgetGateResultShape, suppliedCell.descriptor().shape());
        DataType inputProductGateType = DataTypePromotion.promoteNumeric(
                inputGateType, candidateType);
        Shape inputProductGateShape = ShapeBroadcast.broadcast(
                inputGateResultShape, candidateResultShape);
        DataType nextCellType = DataTypePromotion.promoteNumeric(
                forgetProductType, inputProductGateType);
        Shape nextCellShape = ShapeBroadcast.broadcast(forgetProductShape, inputProductGateShape);
        requireFloating(nextCellType, "next cell");
        DataTypePromotion.promoteNumeric(outputGateType, nextCellType);
        ShapeBroadcast.broadcast(outputGateResultShape, nextCellShape);

        Tensor inputProjection = currentBias.isPresent()
                ? suppliedInput.linear(currentInputWeight, currentBias.orElseThrow())
                : suppliedInput.linear(currentInputWeight);
        Tensor hiddenProjection = suppliedHidden.linear(currentHiddenWeight);

        Tensor inputGateProjection = inputProjection.sliceAxis(-1, 0L, hiddenSize);
        Tensor forgetGateProjection = inputProjection.sliceAxis(
                -1, hiddenSize, twiceHiddenSize);
        Tensor inputCandidate = inputProjection.sliceAxis(
                -1, twiceHiddenSize, thriceHiddenSize);
        Tensor outputGateProjection = inputProjection.sliceAxis(
                -1, thriceHiddenSize, packedHiddenSize);
        Tensor hiddenInputGate = hiddenProjection.sliceAxis(-1, 0L, hiddenSize);
        Tensor hiddenForgetGate = hiddenProjection.sliceAxis(
                -1, hiddenSize, twiceHiddenSize);
        Tensor hiddenCandidate = hiddenProjection.sliceAxis(
                -1, twiceHiddenSize, thriceHiddenSize);
        Tensor hiddenOutputGate = hiddenProjection.sliceAxis(
                -1, thriceHiddenSize, packedHiddenSize);

        Tensor activatedInputGate = inputGateProjection.add(hiddenInputGate).sigmoid();
        Tensor activatedForgetGate = forgetGateProjection.add(hiddenForgetGate).sigmoid();
        Tensor activatedCandidate = inputCandidate.add(hiddenCandidate).tanh();
        Tensor activatedOutputGate = outputGateProjection.add(hiddenOutputGate).sigmoid();
        Tensor nextCell = activatedForgetGate.mul(suppliedCell)
                .add(activatedInputGate.mul(activatedCandidate));
        Tensor nextHidden = activatedOutputGate.mul(nextCell.tanh());
        return new LstmCellForwardResult(nextHidden, nextCell);
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
        if (packedHiddenSize % 4L != 0L) {
            throw new IllegalArgumentException(
                    "inputWeight packed hidden size must be divisible by four: "
                            + packedHiddenSize);
        }
        validatePositiveExtent(shape, 1, "inputWeight inputSize");
        return packedHiddenSize / 4L;
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

    private static void validateState(Tensor state, long hiddenSize, String name) {
        Shape stateShape = state.descriptor().shape();
        int rank = stateShape.rank();
        if (rank < 1) {
            throw new IllegalArgumentException(name + " rank must be at least 1: " + rank);
        }
        Dimension features = stateShape.dimension(rank - 1);
        if (features instanceof StaticDimension staticFeatures
                && staticFeatures.size() != hiddenSize) {
            throw new IllegalArgumentException(
                    name + " feature dimension must equal hidden size: "
                            + features + ", hiddenSize=" + hiddenSize);
        }
    }

    private static Shape projectionShape(Tensor value, Tensor weight) {
        Shape valueShape = value.descriptor().shape();
        Dimension[] resultDimensions = new Dimension[valueShape.rank()];
        for (int axis = 0; axis < valueShape.rank() - 1; axis++) {
            resultDimensions[axis] = valueShape.dimension(axis);
        }
        resultDimensions[valueShape.rank() - 1] = weight.descriptor().shape().dimension(0);
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

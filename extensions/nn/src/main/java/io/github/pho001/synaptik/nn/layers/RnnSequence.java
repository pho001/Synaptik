package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Statically unrolls one time-major vanilla-RNN sequence over explicit construction-time lengths.
 *
 * <p>The container permanently owns one {@link RnnCell}. For input Shape
 * {@code [time, batch, inputSize]}, initial-hidden Shape {@code [batch, hiddenSize]}, and one
 * Java length per original batch row, step {@code t} gathers exactly the rows whose length is
 * greater than {@code t}. Those rows form the active batch for that step. The sequence invokes
 * the cell once on the compact active batch and exposes the exact returned Tensor. Original
 * relative row order is stable; no sorting or permutation is introduced. A numeric zero is
 * ordinary input data and is never interpreted as padding.</p>
 *
 * <p>The result contains compact outputs separated by time step plus final hidden rows restored
 * to original batch order. A zero-length row takes its final state from the caller's initial
 * hidden Tensor. If every row has length zero, construction returns an empty output list and the
 * exact initial-hidden reference without allocating a Tensor identity.</p>
 *
 * <p>Lengths are validated from the caller array and, when at least one step is represented,
 * cloned immediately before traversal; no array is retained. They are Java construction metadata,
 * not Tensor values. Callers must coordinate writes throughout validation and any snapshot.
 * Runtime-dependent lengths or masks would require a genuine Model recurrent scan/control-flow
 * contract and are not simulated with dense masking. Forward is mode-insensitive and retains no
 * input, state, index, length, or result. Parameter replacement and multi-step construction are
 * not one atomic snapshot; callers must coordinate them when a consistent binding set is
 * required.</p>
 *
 * <p>One Java cell and its exact Parameter leaf Tensors are shared across every represented time
 * step, while every select, gather, cell operation, and restored-state producer is fresh. Later
 * states retain temporal ancestry through those fresh producers. This static identity fan-out is
 * visible to the Compiler, whose existing exact-identity gradient contract combines repeated
 * contributions. This class itself only constructs eager index leaves and storage-free Model
 * expression metadata; it does not evaluate values, define numerical gradients, expose a public
 * training loop, capture or compile a graph, select a backend, or promise runtime work skipping.</p>
 */
public final class RnnSequence extends Module {
    private final RnnCell cell;

    /**
     * Creates a sequence container that permanently owns the exact supplied cell.
     *
     * @param cell non-null currently unowned vanilla-RNN cell to register under child name
     *     {@code cell}; retained exactly
     * @throws NullPointerException if {@code cell} is null
     * @throws IllegalStateException if {@code cell} is already owned by another module
     */
    public RnnSequence(RnnCell cell) {
        this.cell = child("cell", Objects.requireNonNull(cell, "cell"));
    }

    /**
     * Creates a sequence owning one automatic standard RNN cell.
     *
     * <p>Construction creates and owns exactly one unbound cell. It creates no random generator,
     * Tensor, Parameter, default state, or all-valid length array.</p>
     *
     * @param hiddenSize strictly positive hidden width
     * @param bias whether the cell will own a typed-zero shared bias
     * @param dataType non-null floating parameter and default-state type
     * @param weightInitialization non-null closed matrix policy applied independently by the cell
     * @param seed seed retained by the cell for random-policy initialization attempts; zero/one
     *     policies do not use it
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if {@code hiddenSize} is not positive, {@code dataType} is
     *     not floating, or a configured parameter Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if checked Shape/count arithmetic overflows
     */
    public RnnSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        this(new RnnCell(hiddenSize, bias, dataType, weightInitialization, seed));
    }

    /**
     * Returns the stable owned cell.
     *
     * @return the exact non-null child registered under {@code cell}
     */
    public RnnCell cell() {
        return cell;
    }

    /**
     * Builds the complete all-valid sequence from caller-supplied hidden state.
     *
     * <p>A fresh private Java array marks every row valid for all {@code time} steps. The explicit
     * state is preserved exactly when {@code time == 0}; no default state is created.</p>
     *
     * @param input non-null floating fully static rank-three input shaped
     *     {@code [time, batch, inputSize]}; not mutated or retained
     * @param initialHidden non-null compatible floating fully static state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained
     * @return the non-null static result with every input time step represented
     * @throws NullPointerException if {@code input} or {@code initialHidden} is null
     * @throws IllegalArgumentException if input, state, current cell schema, promotion, or static
     *     unroll validation fails
     * @throws ArithmeticException if checked Shape or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public RnnSequenceForwardResult forward(Tensor input, Tensor initialHidden) {
        SequenceSchema schema = validateInputAndStateForDefaults(input, initialHidden);
        return forward(input, initialHidden, allValidLengths(schema));
    }

    /**
     * Builds a statically packed sequence from a fresh typed zero hidden state.
     *
     * <p>After complete default-path validation, the method creates one fresh eager typed-zero
     * leaf shaped {@code [batch, hiddenSize]} with no name and no gradient requirement. It is not
     * retained. An all-zero length array returns that exact leaf as final state and leaves an
     * automatic cell unbound.</p>
     *
     * @param input non-null floating fully static rank-three time-major input; not retained
     * @param lengths non-null caller-owned valid length per original batch row; validated from the
     *     caller array, never mutated or retained, and cloned before traversal
     * @return the non-null packed result whose skipped rows originate from the fresh zero state
     * @throws NullPointerException if {@code input} or {@code lengths} is null
     * @throws IllegalArgumentException if input, lengths, current cell schema, promotion, default
     *     state, or static unroll validation fails
     * @throws ArithmeticException if checked state, Shape, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public RnnSequenceForwardResult forward(Tensor input, long[] lengths) {
        SequenceSchema schema = validateInputAndLengthsForDefaults(input, lengths);
        Tensor initialHidden = zeroState(schema);
        return forward(input, initialHidden, lengths);
    }

    /**
     * Builds an all-valid sequence from a fresh typed zero hidden state.
     *
     * <p>The method creates one private all-valid length array and one fresh eager typed-zero
     * non-gradient state leaf. For {@code time == 0}, the result contains no output and preserves
     * that exact leaf; the automatic cell remains unbound.</p>
     *
     * @param input non-null floating fully static rank-three time-major input; not retained
     * @return the non-null complete static result
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if input, current cell schema, promotion, default state, or
     *     static unroll validation fails
     * @throws ArithmeticException if checked state, Shape, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public RnnSequenceForwardResult forward(Tensor input) {
        SequenceSchema schema = validateInputForDefaults(input);
        validateDefaultStateCount(schema.batch, schema.hiddenSize);
        long[] lengths = allValidLengths(schema);
        Tensor initialHidden = zeroState(schema);
        return forward(input, initialHidden, lengths);
    }

    private SequenceSchema validateInputForDefaults(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Shape shape = suppliedInput.descriptor().shape();
        requireFloating(suppliedInput.descriptor().dataType(), "input");
        requireRank(shape, 3, "input");
        requireStatic(shape, "input");
        long time = extent(shape, 0);
        long batch = extent(shape, 1);
        long inputSize = extent(shape, 2);
        cell.validateConfiguredInputSize(inputSize);
        if (batch > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("batch exceeds Java array limit: " + batch);
        }
        if (time > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("time exceeds Java list limit: " + time);
        }
        DataTypePromotion.promoteNumeric(
                suppliedInput.descriptor().dataType(), cell.configuredDataType());
        return new SequenceSchema(time, batch, inputSize, cell.configuredHiddenSize());
    }

    private SequenceSchema validateInputAndStateForDefaults(Tensor input, Tensor state) {
        SequenceSchema schema = validateInputForDefaults(input);
        Tensor suppliedState = Objects.requireNonNull(state, "initialHidden");
        Shape stateShape = suppliedState.descriptor().shape();
        requireFloating(suppliedState.descriptor().dataType(), "initialHidden");
        requireRank(stateShape, 2, "initialHidden");
        requireStatic(stateShape, "initialHidden");
        if (extent(stateShape, 0) != schema.batch
                || extent(stateShape, 1) != schema.hiddenSize) {
            throw new IllegalArgumentException("initialHidden shape is incompatible with input/cell schema");
        }
        DataTypePromotion.promoteNumeric(
                suppliedState.descriptor().dataType(), cell.configuredDataType());
        return schema;
    }

    private SequenceSchema validateInputAndLengthsForDefaults(Tensor input, long[] lengths) {
        SequenceSchema schema = validateInputForDefaults(input);
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");
        if (suppliedLengths.length != schema.batch) {
            throw new IllegalArgumentException("length count must equal batch extent");
        }
        long maximumLength = 0;
        for (int index = 0; index < suppliedLengths.length; index++) {
            long length = suppliedLengths[index];
            if (length < 0 || length > schema.time) {
                throw new IllegalArgumentException("lengths[" + index + "] is outside [0,time]");
            }
            maximumLength = Math.max(maximumLength, length);
        }
        validateDefaultStateCount(schema.batch, schema.hiddenSize);
        DataType stateType = cell.configuredDataType();
        Shape stateShape = Shape.of(schema.batch, schema.hiddenSize);
        DataType outputType = prevalidateSteps(
                suppliedLengths,
                maximumLength,
                input.descriptor().dataType(),
                stateType,
                stateType,
                cell.configuredBias(),
                schema.inputSize,
                schema.hiddenSize);
        prevalidateFinalStack(
                suppliedLengths, stateType, stateShape, outputType,
                schema.hiddenSize, maximumLength);
        return schema;
    }

    private static long[] allValidLengths(SequenceSchema schema) {
        long[] lengths = new long[Math.toIntExact(schema.batch)];
        Arrays.fill(lengths, schema.time);
        return lengths;
    }

    private Tensor zeroState(SequenceSchema schema) {
        return TensorFactory.zeros(
                Shape.of(schema.batch, schema.hiddenSize),
                cell.configuredDataType(),
                Optional.empty(),
                false);
    }

    private static void validateDefaultStateCount(long batch, long hiddenSize) {
        long count = Math.multiplyExact(batch, hiddenSize);
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "default hidden state exceeds Java array limit: " + count);
        }
    }

    private record SequenceSchema(long time, long batch, long inputSize, long hiddenSize) {
    }

    /**
     * Builds a statically packed time-major recurrent expression and restores final hidden rows.
     *
     * <p>Caller-controlled null, cell-schema, type, rank, static-Shape, feature, batch, length,
     * promotion, broadcast, selection, gather, and stack checks finish before the first eager
     * index leaf or expression is created. After construction begins, successful index leaves and
     * expression prefixes are not rolled back if allocation, Tensor-identifier exhaustion, or an
     * unexpected delegated failure occurs later. No partial result is returned and this module
     * retains none of the prefix.</p>
     *
     * @param input non-null floating fully static rank-three time-major Tensor shaped
     *     {@code [time, batch, inputSize]}; neither mutated nor retained
     * @param initialHidden non-null floating fully static rank-two Tensor shaped
     *     {@code [batch, hiddenSize]}; neither mutated nor retained
     * @param lengths non-null caller-owned array with exactly one value per batch row, each in
     *     {@code [0, time]} and with maximum no greater than {@link Integer#MAX_VALUE}; cloned
     *     before traversal, never mutated or retained, and never inferred from Tensor values
     * @return a non-null immutable result containing exact compact cell outputs in increasing time
     *     order and exact final-hidden rows restored to original batch order
     * @throws NullPointerException if {@code input}, {@code initialHidden}, or {@code lengths} is
     *     null, checked in that order
     * @throws IllegalArgumentException if the current cell schema or any documented input, Shape,
     *     feature, batch, length, numeric-promotion, broadcast, gather, select, or stack contract
     *     is invalid
     * @throws ArithmeticException if checked Model Shape or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after valid
     *     construction begins; already constructed prefixes are not rolled back
     * @throws OutOfMemoryError if the length snapshot, a temporary Java collection or array, or
     *     eager index storage cannot be allocated; already completed construction effects are not
     *     rolled back
     */
    public RnnSequenceForwardResult forward(
            Tensor input, Tensor initialHidden, long[] lengths) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedInitialHidden = Objects.requireNonNull(initialHidden, "initialHidden");
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");

        Shape inputShape = suppliedInput.descriptor().shape();
        Shape hiddenShape = suppliedInitialHidden.descriptor().shape();
        requireFloating(suppliedInput.descriptor().dataType(), "input");
        requireRank(inputShape, 3, "input");
        requireStatic(inputShape, "input");
        requireFloating(suppliedInitialHidden.descriptor().dataType(), "initialHidden");
        requireRank(hiddenShape, 2, "initialHidden");
        requireStatic(hiddenShape, "initialHidden");

        long time = extent(inputShape, 0);
        long batch = extent(inputShape, 1);
        long inputSize = extent(inputShape, 2);
        cell.validateConfiguredInputSize(inputSize);
        long hiddenSize = cell.configuredHiddenSize();
        if (extent(hiddenShape, 1) != hiddenSize) {
            throw new IllegalArgumentException(
                    "initialHidden feature size must equal cell hiddenSize: initialHidden="
                            + extent(hiddenShape, 1) + ", cell=" + hiddenSize);
        }
        if (extent(hiddenShape, 0) != batch) {
            throw new IllegalArgumentException(
                    "input and initialHidden batch extents must match: input="
                            + batch + ", initialHidden=" + extent(hiddenShape, 0));
        }
        if (suppliedLengths.length != batch) {
            throw new IllegalArgumentException(
                    "length count must equal batch extent: lengths="
                            + suppliedLengths.length + ", batch=" + batch);
        }

        long maximumLength = 0;
        for (int index = 0; index < suppliedLengths.length; index++) {
            long length = suppliedLengths[index];
            if (length < 0) {
                throw new IllegalArgumentException(
                        "lengths[" + index + "] must be non-negative: " + length);
            }
            if (length > time) {
                throw new IllegalArgumentException(
                        "lengths[" + index + "] exceeds time extent " + time + ": " + length);
            }
            maximumLength = Math.max(maximumLength, length);
        }
        if (maximumLength > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "maximum length exceeds Java list indexing limit: " + maximumLength);
        }

        DataType outputType = prevalidateSteps(
                suppliedLengths,
                maximumLength,
                suppliedInput.descriptor().dataType(),
                suppliedInitialHidden.descriptor().dataType(),
                cell.configuredDataType(),
                cell.configuredBias(),
                inputSize,
                hiddenSize);
        prevalidateFinalStack(
                suppliedLengths,
                suppliedInitialHidden.descriptor().dataType(),
                hiddenShape,
                outputType,
                hiddenSize,
                maximumLength);

        if (maximumLength == 0) {
            return new RnnSequenceForwardResult(List.of(), suppliedInitialHidden);
        }

        long[] lengthSnapshot = suppliedLengths.clone();
        int steps = Math.toIntExact(maximumLength);
        List<Tensor> packedOutputs = new ArrayList<>(steps);
        int[] exitPositions = new int[lengthSnapshot.length];
        long[] previousActive = null;
        for (int timeIndex = 0; timeIndex < steps; timeIndex++) {
            Tensor timeSlice = suppliedInput.select(0, timeIndex);
            long[] active = activeOriginalRows(lengthSnapshot, timeIndex);
            for (int position = 0; position < active.length; position++) {
                int originalRow = Math.toIntExact(active[position]);
                if (lengthSnapshot[originalRow] == (long) timeIndex + 1) {
                    exitPositions[originalRow] = position;
                }
            }
            Tensor activeIndices = indexTensor(active);
            Tensor compactInput = timeSlice.gather(activeIndices, 0);
            Tensor compactHidden;
            if (timeIndex == 0) {
                compactHidden = suppliedInitialHidden.gather(activeIndices, 0);
            } else {
                long[] survivorPositions = survivorPositions(previousActive, active);
                compactHidden = packedOutputs
                        .get(timeIndex - 1)
                        .gather(indexTensor(survivorPositions), 0);
            }
            packedOutputs.add(cell.forward(compactInput, compactHidden));
            previousActive = active;
        }

        Tensor[] finalRows = new Tensor[lengthSnapshot.length];
        for (int batchIndex = 0; batchIndex < lengthSnapshot.length; batchIndex++) {
            long length = lengthSnapshot[batchIndex];
            if (length == 0) {
                finalRows[batchIndex] = suppliedInitialHidden.select(0, batchIndex);
            } else {
                finalRows[batchIndex] = packedOutputs
                        .get(Math.toIntExact(length - 1))
                        .select(0, exitPositions[batchIndex]);
            }
        }
        Tensor finalHidden = Tensor.stack(0, finalRows);
        return new RnnSequenceForwardResult(packedOutputs, finalHidden);
    }

    private static DataType prevalidateSteps(
            long[] lengths,
            long maximumLength,
            DataType inputType,
            DataType hiddenType,
            DataType parameterType,
            boolean bias,
            long inputSize,
            long hiddenSize) {
        DataType inputProduct = DataTypePromotion.promoteNumeric(
                inputType, parameterType);
        DataType inputProjection = bias
                ? DataTypePromotion.promoteNumeric(inputProduct, parameterType)
                : inputProduct;
        DataType hiddenProjection = DataTypePromotion.promoteNumeric(
                hiddenType, parameterType);
        DataType outputType = DataTypePromotion.promoteNumeric(inputProjection, hiddenProjection);

        long previousCount = -1;
        for (long timeIndex = 0; timeIndex < maximumLength; timeIndex++) {
            long activeCount = activeCount(lengths, timeIndex);
            if (activeCount != previousCount) {
                if (activeCount > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException(
                            "active batch exceeds Java index-array limit: " + activeCount);
                }
                Shape compactInputShape = Shape.of(activeCount, inputSize);
                Shape compactHiddenShape = Shape.of(activeCount, hiddenSize);
                Shape inputProjectionShape = projectionShape(
                        compactInputShape, Shape.of(hiddenSize, inputSize));
                Shape hiddenProjectionShape = projectionShape(
                        compactHiddenShape, Shape.of(hiddenSize, hiddenSize));
                Shape result = ShapeBroadcast.broadcast(inputProjectionShape, hiddenProjectionShape);
                if (!result.equals(Shape.of(activeCount, hiddenSize))) {
                    throw new IllegalArgumentException("cell step produced an unexpected shape: " + result);
                }
                previousCount = activeCount;
            }
        }
        return outputType;
    }

    private static void prevalidateFinalStack(
            long[] lengths,
            DataType initialHiddenType,
            Shape initialHiddenShape,
            DataType outputType,
            long hiddenSize,
            long maximumLength) {
        if (maximumLength == 0) {
            return;
        }
        Shape initialRowShape = Shape.of(extent(initialHiddenShape, 1));
        Shape recurrentRowShape = Shape.of(hiddenSize);
        if (!initialRowShape.equals(recurrentRowShape)) {
            throw new IllegalArgumentException(
                    "final hidden row shapes must match: initial="
                            + initialRowShape + ", recurrent=" + recurrentRowShape);
        }
        for (int index = 0; index < lengths.length; index++) {
            DataType rowType = lengths[index] == 0 ? initialHiddenType : outputType;
            if (rowType != outputType) {
                throw new IllegalArgumentException(
                        "final hidden rows must have one exact data type for stacking: row "
                                + index + "=" + rowType + ", recurrent=" + outputType);
            }
            if (lengths[index] > 0 && activeCount(lengths, lengths[index] - 1) == 0) {
                throw new IllegalArgumentException(
                        "final hidden source step has no active rows for batch index " + index);
            }
        }
    }

    private static Shape projectionShape(Shape inputShape, Shape weightShape) {
        Dimension[] dimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < inputShape.rank() - 1; axis++) {
            dimensions[axis] = inputShape.dimension(axis);
        }
        dimensions[dimensions.length - 1] = weightShape.dimension(0);
        return Shape.ofDimensions(dimensions);
    }

    private static void validateCellSchema(
            Tensor inputWeight, Tensor hiddenWeight, Optional<Tensor> bias) {
        validateWeight(inputWeight, "inputWeight");
        validateWeight(hiddenWeight, "hiddenWeight");
        DataType inputType = inputWeight.descriptor().dataType();
        DataType hiddenType = hiddenWeight.descriptor().dataType();
        if (hiddenType != inputType) {
            throw new IllegalArgumentException("cell weight data types must match");
        }
        Shape inputShape = inputWeight.descriptor().shape();
        Shape hiddenShape = hiddenWeight.descriptor().shape();
        long hiddenSize = extent(inputShape, 0);
        if (extent(hiddenShape, 0) != hiddenSize || extent(hiddenShape, 1) != hiddenSize) {
            throw new IllegalArgumentException("cell hiddenWeight must be square at hiddenSize");
        }
        if (bias.isPresent()) {
            Tensor biasValue = bias.orElseThrow();
            requireFloating(biasValue.descriptor().dataType(), "cell bias");
            if (!biasValue.descriptor().requiresGrad()) {
                throw new IllegalArgumentException("cell bias must have requiresGrad == true");
            }
            Shape biasShape = biasValue.descriptor().shape();
            requireRank(biasShape, 1, "cell bias");
            requireStatic(biasShape, "cell bias");
            if (biasValue.descriptor().dataType() != inputType
                    || extent(biasShape, 0) != hiddenSize) {
                throw new IllegalArgumentException("cell bias schema is incompatible with weights");
            }
        }
    }

    private static void validateWeight(Tensor weight, String name) {
        requireFloating(weight.descriptor().dataType(), "cell " + name);
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException("cell " + name + " must have requiresGrad == true");
        }
        Shape shape = weight.descriptor().shape();
        requireRank(shape, 2, "cell " + name);
        requireStatic(shape, "cell " + name);
        if (extent(shape, 0) <= 0 || extent(shape, 1) <= 0) {
            throw new IllegalArgumentException("cell " + name + " extents must be positive");
        }
    }

    private static void requireFloating(DataType type, String name) {
        if (!type.isFloating()) {
            throw new IllegalArgumentException(name + " must have a floating data type: " + type);
        }
    }

    private static void requireRank(Shape shape, int rank, String name) {
        if (shape.rank() != rank) {
            throw new IllegalArgumentException(name + " must have rank " + rank + ": " + shape.rank());
        }
    }

    private static void requireStatic(Shape shape, String name) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(name + " must have a fully static shape: " + shape);
        }
    }

    private static long extent(Shape shape, int axis) {
        Dimension dimension = shape.dimension(axis);
        return ((StaticDimension) dimension).size();
    }

    private static long activeCount(long[] lengths, long timeIndex) {
        long count = 0;
        for (long length : lengths) {
            if (length > timeIndex) {
                count++;
            }
        }
        return count;
    }

    private static long[] activeOriginalRows(long[] lengths, long timeIndex) {
        int count = Math.toIntExact(activeCount(lengths, timeIndex));
        long[] active = new long[count];
        int position = 0;
        for (int batchIndex = 0; batchIndex < lengths.length; batchIndex++) {
            if (lengths[batchIndex] > timeIndex) {
                active[position++] = batchIndex;
            }
        }
        return active;
    }

    private static long[] survivorPositions(long[] previousActive, long[] active) {
        long[] positions = new long[active.length];
        int previousPosition = 0;
        for (int index = 0; index < active.length; index++) {
            while (previousActive[previousPosition] != active[index]) {
                previousPosition++;
            }
            positions[index] = previousPosition;
            previousPosition++;
        }
        return positions;
    }

    private static Tensor indexTensor(long[] values) {
        Shape shape = Shape.of(values.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
    }
}

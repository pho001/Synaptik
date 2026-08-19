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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Statically unrolls one time-major long short-term memory (LSTM) sequence over explicit Java
 * lengths.
 *
 * <p>The container permanently owns one {@link LstmCell}. For input Shape
 * {@code [time, batch, inputSize]}, initial-state Shapes {@code [batch, hiddenSize]}, and one
 * length per original batch row, step {@code t} gathers only rows whose length is greater than
 * {@code t}. The same active-row index is used for both state gathers. Later steps likewise share
 * one survivor index between hidden and cell state. Each cell invocation contributes its exact
 * next-hidden Tensor to the public compact output list while its exact next-cell Tensor is carried
 * internally. Original relative order is stable; no sorting occurs, and numeric zero is ordinary
 * Tensor data rather than a padding marker.</p>
 *
 * <p>Final hidden and cell rows are independently restored to original batch order. A zero-length
 * row uses its matching caller-supplied initial row. If no row is active, the result contains an
 * empty output list and the exact two initial-state references without allocating a Tensor
 * identity or invoking the cell.</p>
 *
 * <p>Lengths are validated from the caller array and, when at least one step is represented,
 * cloned immediately before traversal; no array is retained. They are construction metadata, not
 * runtime Tensor values; callers must coordinate writes throughout validation and any snapshot.
 * Runtime-dependent masks or lengths require a future recurrent scan/control-flow contract.
 * Forward is mode-insensitive and retains no input, state, length,
 * index, or result. Parameter replacement and multi-step construction are not one atomic
 * snapshot, so callers coordinate them when consistent bindings matter.</p>
 *
 * <p>One Java cell and its exact Parameter leaf Tensors are shared across every represented time
 * step, while every select, gather, gate operation, and restored-state producer is fresh. Both
 * later states retain temporal ancestry through those fresh producers. This static identity
 * fan-out is visible to the Compiler, whose existing exact-identity gradient contract combines
 * repeated contributions. This class itself creates eager index leaves and storage-free Model
 * expression metadata; it does not evaluate values, define numerical gradients, expose a public
 * training loop, compile a graph, select a backend, or promise runtime work skipping.</p>
 */
public final class LstmSequence extends Module {
    private final LstmCell cell;

    /**
     * Creates a sequence container that permanently owns the exact supplied cell.
     *
     * @param cell non-null currently unowned LSTM cell to register under child name {@code cell};
     *     retained exactly
     * @throws NullPointerException if {@code cell} is null
     * @throws IllegalStateException if {@code cell} is already owned by another module
     */
    public LstmSequence(LstmCell cell) {
        this.cell = child("cell", Objects.requireNonNull(cell, "cell"));
    }

    /**
     * Creates a sequence owning one automatic standard LSTM cell.
     *
     * <p>Construction creates and owns exactly one unbound cell. It creates no random generator,
     * Tensor, Parameter, default state, or all-valid length array.</p>
     *
     * @param hiddenSize strictly positive hidden and cell-state width
     * @param bias whether the cell will own a complete packed typed-zero input-side bias
     * @param dataType non-null floating parameter and default-state type
     * @param weightInitialization non-null closed policy applied independently to both packed
     *     matrices by the cell
     * @param seed seed retained by the cell for random-policy initialization attempts; zero/one
     *     policies do not use it
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if {@code hiddenSize} is not positive, {@code dataType} is
     *     not floating, or a configured parameter Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if {@code 4 * hiddenSize} or checked Shape arithmetic overflows
     */
    public LstmSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        this(new LstmCell(hiddenSize, bias, dataType, weightInitialization, seed));
    }

    /**
     * Returns the stable owned LSTM cell.
     *
     * @return the exact non-null child registered under {@code cell}
     */
    public LstmCell cell() {
        return cell;
    }

    /**
     * Builds the complete all-valid sequence from explicit hidden and cell states.
     *
     * <p>A fresh private Java array marks every row valid for all {@code time} steps. Both
     * explicit state references are preserved exactly when {@code time == 0}; no default state is
     * created.</p>
     *
     * @param input non-null floating fully static rank-three input shaped
     *     {@code [time, batch, inputSize]}; not mutated or retained
     * @param initialHidden non-null compatible floating fully static hidden state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained
     * @param initialCell non-null compatible floating fully static cell state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained
     * @return the non-null static result with every input time step represented
     * @throws NullPointerException if {@code input}, {@code initialHidden}, or
     *     {@code initialCell} is null
     * @throws IllegalArgumentException if input, either state, current cell schema, promotion, or
     *     static unroll validation fails
     * @throws ArithmeticException if checked packed-size, Shape, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public LstmSequenceForwardResult forward(
            Tensor input, Tensor initialHidden, Tensor initialCell) {
        SequenceSchema schema = validateInputAndStatesForDefaults(
                input, initialHidden, initialCell);
        return forward(input, initialHidden, initialCell, allValidLengths(schema));
    }

    /**
     * Builds a statically packed sequence from fresh typed hidden and cell zero states.
     *
     * <p>After complete default-path validation, the method creates two distinct fresh eager
     * typed-zero leaves shaped {@code [batch, hiddenSize]}, each unnamed and not requiring a
     * gradient. Neither is retained. An all-zero length array returns those exact leaves as final
     * hidden and cell state and leaves an automatic cell unbound.</p>
     *
     * @param input non-null floating fully static rank-three time-major input; not retained
     * @param lengths non-null caller-owned valid length per original batch row; validated from the
     *     caller array, never mutated or retained, and cloned before traversal
     * @return the non-null packed result whose skipped rows originate from the fresh zero states
     * @throws NullPointerException if {@code input} or {@code lengths} is null
     * @throws IllegalArgumentException if input, lengths, current cell schema, promotion, default
     *     states, or static unroll validation fails
     * @throws ArithmeticException if checked packed-size, state, Shape, or layout arithmetic
     *     overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public LstmSequenceForwardResult forward(Tensor input, long[] lengths) {
        SequenceSchema schema = validateInputAndLengthsForDefaults(input, lengths);
        Tensor hidden = zeroState(schema);
        Tensor cellState = zeroState(schema);
        return forward(input, hidden, cellState, lengths);
    }

    /**
     * Builds an all-valid sequence from fresh typed hidden and cell zero states.
     *
     * <p>The method creates one private all-valid length array and two distinct fresh eager
     * typed-zero non-gradient state leaves. For {@code time == 0}, the result contains no output
     * and preserves those exact leaves; the automatic cell remains unbound.</p>
     *
     * @param input non-null floating fully static rank-three time-major input; not retained
     * @return the non-null complete static result
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if input, current cell schema, promotion, default states,
     *     or static unroll validation fails
     * @throws ArithmeticException if checked packed-size, state, Shape, or layout arithmetic
     *     overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic random-policy initialization fails
     * @throws OutOfMemoryError if an array, eager leaf, Tensor, or expression cannot be allocated
     */
    public LstmSequenceForwardResult forward(Tensor input) {
        SequenceSchema schema = validateInputForDefaults(input);
        validateDefaultStateCount(schema.batch, schema.hiddenSize);
        long[] lengths = allValidLengths(schema);
        Tensor hidden = zeroState(schema);
        Tensor cellState = zeroState(schema);
        return forward(input, hidden, cellState, lengths);
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
        if (batch > Integer.MAX_VALUE || time > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("time or batch exceeds Java collection limit");
        }
        DataTypePromotion.promoteNumeric(
                suppliedInput.descriptor().dataType(), cell.configuredDataType());
        return new SequenceSchema(time, batch, inputSize, cell.configuredHiddenSize());
    }

    private SequenceSchema validateInputAndStatesForDefaults(
            Tensor input, Tensor hidden, Tensor cellState) {
        SequenceSchema schema = validateInputForDefaults(input);
        validateDefaultState(hidden, "initialHidden", schema);
        validateDefaultState(cellState, "initialCell", schema);
        return schema;
    }

    private void validateDefaultState(Tensor state, String name, SequenceSchema schema) {
        Tensor suppliedState = Objects.requireNonNull(state, name);
        Shape shape = suppliedState.descriptor().shape();
        requireFloating(suppliedState.descriptor().dataType(), name);
        requireRank(shape, 2, name);
        requireStatic(shape, name);
        if (extent(shape, 0) != schema.batch || extent(shape, 1) != schema.hiddenSize) {
            throw new IllegalArgumentException(name + " shape is incompatible with input/cell schema");
        }
        DataTypePromotion.promoteNumeric(
                suppliedState.descriptor().dataType(), cell.configuredDataType());
    }

    private SequenceSchema validateInputAndLengthsForDefaults(Tensor input, long[] lengths) {
        SequenceSchema schema = validateInputForDefaults(input);
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");
        if (suppliedLengths.length != schema.batch) {
            throw new IllegalArgumentException("length count must equal batch extent");
        }
        long maximumLength = 0;
        for (int index = 0; index < suppliedLengths.length; index++) {
            if (suppliedLengths[index] < 0 || suppliedLengths[index] > schema.time) {
                throw new IllegalArgumentException("lengths[" + index + "] is outside [0,time]");
            }
            maximumLength = Math.max(maximumLength, suppliedLengths[index]);
        }
        validateDefaultStateCount(schema.batch, schema.hiddenSize);
        DataType stateType = cell.configuredDataType();
        Shape stateShape = Shape.of(schema.batch, schema.hiddenSize);
        DataType[] outputTypes = prevalidateSteps(
                suppliedLengths,
                maximumLength,
                input.descriptor().dataType(),
                stateType,
                stateType,
                stateType,
                cell.configuredBias(),
                schema.inputSize,
                schema.hiddenSize);
        prevalidateFinalStacks(
                suppliedLengths,
                stateType,
                stateShape,
                stateType,
                stateShape,
                outputTypes[0],
                outputTypes[1],
                schema.hiddenSize,
                maximumLength);
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
                cell.configuredDataType(), Optional.empty(), false);
    }

    private static void validateDefaultStateCount(long batch, long hiddenSize) {
        long count = Math.multiplyExact(batch, hiddenSize);
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "default recurrent state exceeds Java array limit: " + count);
        }
    }

    private record SequenceSchema(long time, long batch, long inputSize, long hiddenSize) {
    }

    /**
     * Builds a statically packed time-major LSTM expression and restores both final states.
     *
     * <p>Caller-controlled null, current-cell-schema, type, rank, fully static Shape, feature,
     * batch, length, projection, gate, state, gather, selection, and stack checks complete before
     * the first index leaf or expression is created. Once construction begins, successful leaves
     * and expression prefixes are not rolled back after allocation, identifier exhaustion, or an
     * unexpected delegated failure. No partial result is returned and this module retains none of
     * the prefix.</p>
     *
     * @param input non-null floating fully static rank-three time-major Tensor shaped
     *     {@code [time, batch, inputSize]}; neither mutated nor retained
     * @param initialHidden non-null floating fully static rank-two Tensor shaped
     *     {@code [batch, hiddenSize]}; neither mutated nor retained
     * @param initialCell non-null floating fully static rank-two Tensor shaped
     *     {@code [batch, hiddenSize]}; neither mutated nor retained
     * @param lengths non-null caller-owned array with exactly one value per batch row, each in
     *     {@code [0, time]} and with maximum no greater than {@link Integer#MAX_VALUE}; cloned
     *     before traversal, never mutated or retained, and never inferred from Tensor values
     * @return a non-null immutable result containing exact compact next-hidden outputs and exact
     *     final hidden and cell rows restored to original batch order
     * @throws NullPointerException if {@code input}, {@code initialHidden}, {@code initialCell},
     *     or {@code lengths} is null, checked in that order
     * @throws IllegalArgumentException if current cell schema or any documented input, Shape,
     *     feature, batch, length, promotion, broadcast, gather, select, or stack contract fails
     * @throws ArithmeticException if checked packed-bound, Shape, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after construction
     *     begins; completed prefixes are not rolled back
     * @throws OutOfMemoryError if a defensive snapshot, temporary Java value, eager index storage,
     *     or expression cannot be allocated; completed effects are not rolled back
     */
    public LstmSequenceForwardResult forward(
            Tensor input, Tensor initialHidden, Tensor initialCell, long[] lengths) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedInitialHidden = Objects.requireNonNull(initialHidden, "initialHidden");
        Tensor suppliedInitialCell = Objects.requireNonNull(initialCell, "initialCell");
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");
        long hiddenSize = cell.configuredHiddenSize();
        Math.multiplyExact(hiddenSize, 2L);
        Math.multiplyExact(hiddenSize, 3L);
        Math.multiplyExact(hiddenSize, 4L);

        Shape inputShape = suppliedInput.descriptor().shape();
        Shape hiddenShape = suppliedInitialHidden.descriptor().shape();
        Shape cellShape = suppliedInitialCell.descriptor().shape();
        requireFloating(suppliedInput.descriptor().dataType(), "input");
        requireRank(inputShape, 3, "input");
        requireStatic(inputShape, "input");
        requireFloating(suppliedInitialHidden.descriptor().dataType(), "initialHidden");
        requireRank(hiddenShape, 2, "initialHidden");
        requireStatic(hiddenShape, "initialHidden");
        requireFloating(suppliedInitialCell.descriptor().dataType(), "initialCell");
        requireRank(cellShape, 2, "initialCell");
        requireStatic(cellShape, "initialCell");

        long time = extent(inputShape, 0);
        long batch = extent(inputShape, 1);
        long inputSize = extent(inputShape, 2);
        cell.validateConfiguredInputSize(inputSize);
        if (extent(hiddenShape, 1) != hiddenSize) {
            throw new IllegalArgumentException(
                    "initialHidden feature size must equal cell hiddenSize: initialHidden="
                            + extent(hiddenShape, 1) + ", cell=" + hiddenSize);
        }
        if (extent(cellShape, 1) != hiddenSize) {
            throw new IllegalArgumentException(
                    "initialCell feature size must equal cell hiddenSize: initialCell="
                            + extent(cellShape, 1) + ", cell=" + hiddenSize);
        }
        if (extent(hiddenShape, 0) != batch) {
            throw new IllegalArgumentException(
                    "input and initialHidden batch extents must match: input="
                            + batch + ", initialHidden=" + extent(hiddenShape, 0));
        }
        if (extent(cellShape, 0) != batch) {
            throw new IllegalArgumentException(
                    "input and initialCell batch extents must match: input="
                            + batch + ", initialCell=" + extent(cellShape, 0));
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

        DataType[] outputTypes = prevalidateSteps(
                suppliedLengths,
                maximumLength,
                suppliedInput.descriptor().dataType(),
                suppliedInitialHidden.descriptor().dataType(),
                suppliedInitialCell.descriptor().dataType(),
                cell.configuredDataType(),
                cell.configuredBias(),
                inputSize,
                hiddenSize);
        prevalidateFinalStacks(
                suppliedLengths,
                suppliedInitialHidden.descriptor().dataType(),
                hiddenShape,
                suppliedInitialCell.descriptor().dataType(),
                cellShape,
                outputTypes[0],
                outputTypes[1],
                hiddenSize,
                maximumLength);

        if (maximumLength == 0) {
            return new LstmSequenceForwardResult(
                    List.of(), suppliedInitialHidden, suppliedInitialCell);
        }

        long[] lengthSnapshot = suppliedLengths.clone();
        int steps = Math.toIntExact(maximumLength);
        List<Tensor> packedOutputs = new ArrayList<>(steps);
        List<Tensor> carriedCells = new ArrayList<>(steps);
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
            Tensor compactCell;
            if (timeIndex == 0) {
                compactHidden = suppliedInitialHidden.gather(activeIndices, 0);
                compactCell = suppliedInitialCell.gather(activeIndices, 0);
            } else {
                long[] survivorPositions = survivorPositions(previousActive, active);
                Tensor survivorIndices = indexTensor(survivorPositions);
                compactHidden = packedOutputs.get(timeIndex - 1).gather(survivorIndices, 0);
                compactCell = carriedCells.get(timeIndex - 1).gather(survivorIndices, 0);
            }
            LstmCellForwardResult step = cell.forward(compactInput, compactHidden, compactCell);
            packedOutputs.add(step.nextHidden());
            carriedCells.add(step.nextCell());
            previousActive = active;
        }

        Tensor[] finalHiddenRows = new Tensor[lengthSnapshot.length];
        for (int batchIndex = 0; batchIndex < lengthSnapshot.length; batchIndex++) {
            long length = lengthSnapshot[batchIndex];
            finalHiddenRows[batchIndex] = length == 0
                    ? suppliedInitialHidden.select(0, batchIndex)
                    : packedOutputs.get(Math.toIntExact(length - 1))
                            .select(0, exitPositions[batchIndex]);
        }
        Tensor[] finalCellRows = new Tensor[lengthSnapshot.length];
        for (int batchIndex = 0; batchIndex < lengthSnapshot.length; batchIndex++) {
            long length = lengthSnapshot[batchIndex];
            finalCellRows[batchIndex] = length == 0
                    ? suppliedInitialCell.select(0, batchIndex)
                    : carriedCells.get(Math.toIntExact(length - 1))
                            .select(0, exitPositions[batchIndex]);
        }
        Tensor finalHidden = Tensor.stack(0, finalHiddenRows);
        Tensor finalCell = Tensor.stack(0, finalCellRows);
        return new LstmSequenceForwardResult(packedOutputs, finalHidden, finalCell);
    }

    private static DataType[] prevalidateSteps(
            long[] lengths,
            long maximumLength,
            DataType inputType,
            DataType initialHiddenType,
            DataType initialCellType,
            DataType parameterType,
            boolean bias,
            long inputSize,
            long hiddenSize) {
        if (maximumLength == 0) {
            return new DataType[] {initialHiddenType, initialCellType};
        }

        long firstActiveCount = activeCount(lengths, 0);
        DataType[] outputs = prevalidateCellStep(
                firstActiveCount,
                inputType,
                initialHiddenType,
                initialCellType,
                parameterType,
                bias,
                inputSize,
                hiddenSize);
        if (maximumLength == 1) {
            return outputs;
        }

        boolean[] checkedCounts = new boolean[lengths.length + 1];
        checkedCounts[Math.toIntExact(firstActiveCount)] = true;
        long secondActiveCount = activeCount(lengths, 1);
        prevalidateCellStep(
                secondActiveCount,
                inputType,
                outputs[0],
                outputs[1],
                parameterType,
                bias,
                inputSize,
                hiddenSize);
        checkedCounts[Math.toIntExact(secondActiveCount)] = true;

        for (long length : lengths) {
            if (length <= 1) {
                continue;
            }
            long activeCount = activeCount(lengths, length - 1);
            int countIndex = Math.toIntExact(activeCount);
            if (!checkedCounts[countIndex]) {
                prevalidateCellStep(
                        activeCount,
                        inputType,
                        outputs[0],
                        outputs[1],
                        parameterType,
                        bias,
                        inputSize,
                        hiddenSize);
                checkedCounts[countIndex] = true;
            }
        }
        return outputs;
    }

    private static DataType[] prevalidateCellStep(
            long activeCount,
            DataType inputType,
            DataType hiddenType,
            DataType cellType,
            DataType parameterType,
            boolean bias,
            long inputSize,
            long hiddenSize) {
        if (activeCount <= 0 || activeCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "active batch must fit a positive Java index array: " + activeCount);
        }
        long packedHiddenSize = Math.multiplyExact(hiddenSize, 4L);
        Shape compactInputShape = Shape.of(activeCount, inputSize);
        Shape compactHiddenShape = Shape.of(activeCount, hiddenSize);
        Shape compactCellShape = Shape.of(activeCount, hiddenSize);

        DataType inputProductType = DataTypePromotion.promoteNumeric(
                inputType, parameterType);
        Shape inputProjectionShape = projectionShape(
                compactInputShape, Shape.of(packedHiddenSize, inputSize));
        DataType inputProjectionType = inputProductType;
        if (bias) {
            inputProjectionType = DataTypePromotion.promoteNumeric(
                    inputProductType, parameterType);
            inputProjectionShape = ShapeBroadcast.broadcast(
                    inputProjectionShape, Shape.of(packedHiddenSize));
        }
        DataType hiddenProjectionType = DataTypePromotion.promoteNumeric(
                hiddenType, parameterType);
        Shape hiddenProjectionShape = projectionShape(
                compactHiddenShape, Shape.of(packedHiddenSize, hiddenSize));

        long twiceHiddenSize = Math.multiplyExact(hiddenSize, 2L);
        long thriceHiddenSize = Math.multiplyExact(hiddenSize, 3L);
        Shape inputGateShape = gateShape(
                inputProjectionShape, 0L, hiddenSize, packedHiddenSize);
        Shape inputForgetShape = gateShape(
                inputProjectionShape, hiddenSize, twiceHiddenSize, packedHiddenSize);
        Shape inputCandidateShape = gateShape(
                inputProjectionShape, twiceHiddenSize, thriceHiddenSize, packedHiddenSize);
        Shape inputOutputShape = gateShape(
                inputProjectionShape, thriceHiddenSize, packedHiddenSize, packedHiddenSize);
        Shape hiddenGateShape = gateShape(
                hiddenProjectionShape, 0L, hiddenSize, packedHiddenSize);
        Shape hiddenForgetShape = gateShape(
                hiddenProjectionShape, hiddenSize, twiceHiddenSize, packedHiddenSize);
        Shape hiddenCandidateShape = gateShape(
                hiddenProjectionShape, twiceHiddenSize, thriceHiddenSize, packedHiddenSize);
        Shape hiddenOutputShape = gateShape(
                hiddenProjectionShape, thriceHiddenSize, packedHiddenSize, packedHiddenSize);
        DataType gateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        Shape inputGateResultShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
        requireFloating(gateType, "input gate preactivation");
        Shape forgetGateResultShape = ShapeBroadcast.broadcast(inputForgetShape, hiddenForgetShape);
        requireFloating(gateType, "forget gate preactivation");
        Shape candidateResultShape = ShapeBroadcast.broadcast(
                inputCandidateShape, hiddenCandidateShape);
        requireFloating(gateType, "candidate preactivation");
        Shape outputGateResultShape = ShapeBroadcast.broadcast(inputOutputShape, hiddenOutputShape);
        requireFloating(gateType, "output gate preactivation");

        DataType forgetProductType = DataTypePromotion.promoteNumeric(gateType, cellType);
        Shape forgetProductShape = ShapeBroadcast.broadcast(
                forgetGateResultShape, compactCellShape);
        DataType inputCandidateProductType = DataTypePromotion.promoteNumeric(gateType, gateType);
        Shape inputProductShape = ShapeBroadcast.broadcast(
                inputGateResultShape, candidateResultShape);
        DataType nextCellType = DataTypePromotion.promoteNumeric(
                forgetProductType, inputCandidateProductType);
        Shape nextCellShape = ShapeBroadcast.broadcast(forgetProductShape, inputProductShape);
        requireFloating(nextCellType, "next cell");
        DataType nextHiddenType = DataTypePromotion.promoteNumeric(gateType, nextCellType);
        Shape nextHiddenShape = ShapeBroadcast.broadcast(outputGateResultShape, nextCellShape);

        Shape expected = Shape.of(activeCount, hiddenSize);
        if (!nextHiddenShape.equals(expected)) {
            throw new IllegalArgumentException(
                    "cell step produced an unexpected hidden shape: " + nextHiddenShape);
        }
        if (!nextCellShape.equals(expected)) {
            throw new IllegalArgumentException(
                    "cell step produced an unexpected cell shape: " + nextCellShape);
        }
        return new DataType[] {nextHiddenType, nextCellType};
    }

    private static Shape gateShape(
            Shape projectionShape, long fromInclusive, long toExclusive, long packedHiddenSize) {
        if (projectionShape.rank() < 1
                || extent(projectionShape, projectionShape.rank() - 1) != packedHiddenSize
                || fromInclusive < 0
                || fromInclusive > toExclusive
                || toExclusive > packedHiddenSize) {
            throw new IllegalArgumentException("packed gate slices are incompatible with projection");
        }
        Dimension[] dimensions = new Dimension[projectionShape.rank()];
        for (int axis = 0; axis < projectionShape.rank() - 1; axis++) {
            dimensions[axis] = projectionShape.dimension(axis);
        }
        dimensions[dimensions.length - 1] = new StaticDimension(toExclusive - fromInclusive);
        return Shape.ofDimensions(dimensions);
    }

    private static void prevalidateFinalStacks(
            long[] lengths,
            DataType initialHiddenType,
            Shape initialHiddenShape,
            DataType initialCellType,
            Shape initialCellShape,
            DataType outputHiddenType,
            DataType outputCellType,
            long hiddenSize,
            long maximumLength) {
        if (maximumLength == 0) {
            return;
        }
        prevalidateFinalStack(
                lengths,
                initialHiddenType,
                initialHiddenShape,
                outputHiddenType,
                hiddenSize,
                "hidden");
        prevalidateFinalStack(
                lengths,
                initialCellType,
                initialCellShape,
                outputCellType,
                hiddenSize,
                "cell");
    }

    private static void prevalidateFinalStack(
            long[] lengths,
            DataType initialType,
            Shape initialShape,
            DataType recurrentType,
            long hiddenSize,
            String name) {
        Shape initialRowShape = Shape.of(extent(initialShape, 1));
        Shape recurrentRowShape = Shape.of(hiddenSize);
        if (!initialRowShape.equals(recurrentRowShape)) {
            throw new IllegalArgumentException(
                    "final " + name + " row shapes must match: initial="
                            + initialRowShape + ", recurrent=" + recurrentRowShape);
        }
        for (int index = 0; index < lengths.length; index++) {
            DataType rowType = lengths[index] == 0 ? initialType : recurrentType;
            if (rowType != recurrentType) {
                throw new IllegalArgumentException(
                        "final " + name + " rows must have one exact data type for stacking: row "
                                + index + "=" + rowType + ", recurrent=" + recurrentType);
            }
            if (lengths[index] > 0) {
                long sourceCount = activeCount(lengths, lengths[index] - 1);
                long exitPosition = exitPosition(lengths, index, lengths[index] - 1);
                if (sourceCount <= exitPosition) {
                    throw new IllegalArgumentException(
                            "final " + name + " source selection is outside active batch at row "
                                    + index);
                }
            }
        }
    }

    private static long exitPosition(long[] lengths, int originalRow, long timeIndex) {
        long position = 0;
        for (int index = 0; index < originalRow; index++) {
            if (lengths[index] > timeIndex) {
                position++;
            }
        }
        return position;
    }

    private static Shape projectionShape(Shape inputShape, Shape weightShape) {
        Dimension[] dimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < inputShape.rank() - 1; axis++) {
            dimensions[axis] = inputShape.dimension(axis);
        }
        dimensions[dimensions.length - 1] = weightShape.dimension(0);
        return Shape.ofDimensions(dimensions);
    }

    private static long validateCellSchema(
            Tensor inputWeight, Tensor hiddenWeight, Optional<Tensor> bias) {
        validateWeight(inputWeight, "inputWeight");
        Shape inputShape = inputWeight.descriptor().shape();
        long packedHiddenSize = extent(inputShape, 0);
        if (packedHiddenSize % 4L != 0L) {
            throw new IllegalArgumentException(
                    "cell inputWeight packed hidden size must be divisible by four");
        }
        long hiddenSize = packedHiddenSize / 4L;
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("cell hiddenSize must be positive");
        }

        validateWeight(hiddenWeight, "hiddenWeight");
        if (hiddenWeight.descriptor().dataType() != inputWeight.descriptor().dataType()) {
            throw new IllegalArgumentException("cell weight data types must match");
        }
        Shape hiddenShape = hiddenWeight.descriptor().shape();
        if (extent(hiddenShape, 0) != packedHiddenSize
                || extent(hiddenShape, 1) != hiddenSize) {
            throw new IllegalArgumentException(
                    "cell hiddenWeight must have packed rows and hiddenSize columns");
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
            if (biasValue.descriptor().dataType() != inputWeight.descriptor().dataType()
                    || extent(biasShape, 0) != packedHiddenSize) {
                throw new IllegalArgumentException(
                        "cell bias schema is incompatible with packed weights");
            }
        }
        return hiddenSize;
    }

    private static void validateWeight(Tensor weight, String name) {
        requireFloating(weight.descriptor().dataType(), "cell " + name);
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "cell " + name + " must have requiresGrad == true");
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
            throw new IllegalArgumentException(
                    name + " must have a floating data type: " + type);
        }
    }

    private static void requireRank(Shape shape, int rank, String name) {
        if (shape.rank() != rank) {
            throw new IllegalArgumentException(
                    name + " must have rank " + rank + ": " + shape.rank());
        }
    }

    private static void requireStatic(Shape shape, String name) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    name + " must have a fully static shape: " + shape);
        }
    }

    private static long extent(Shape shape, int axis) {
        return ((StaticDimension) shape.dimension(axis)).size();
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

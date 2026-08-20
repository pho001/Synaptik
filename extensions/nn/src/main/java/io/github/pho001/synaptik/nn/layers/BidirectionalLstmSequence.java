package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.Module;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Statically composes independent forward and backward LSTM cells over valid prefixes.
 *
 * <p>The container permanently owns two identity-distinct cells under child names
 * {@code forward} and {@code backward}. Their packed parameters, automatic reservations, seeds,
 * mode, replacement bindings, and state-dictionary paths remain independent. The container
 * declares no direct parameter or buffer and retains no recurrent hidden or cell state, input,
 * length, index, output, or per-call result.</p>
 *
 * <p>Input is a fully static time-major Tensor shaped {@code [time, batch, inputSize]}. One Java
 * length in {@code [0, time]} per batch row specializes the expression structure. Forward depth
 * {@code t} visits original coordinate {@code t}; reverse depth {@code d} uses one Gather-ND
 * expression with coordinate {@code [length - 1 - d, batchRow]} for each active row. The reverse
 * traversal therefore never visits right padding. Its hidden outputs are gathered back into
 * original time and batch order before each visible output concatenates the exact forward hidden
 * features first and backward hidden features second on the final axis. Compact cell states are
 * carried internally and are not merged into the visible output list.</p>
 *
 * <p>The result keeps compact hidden outputs by original time plus separate forward hidden/cell
 * and backward hidden/cell final states in original batch order. A zero-length row takes each
 * final state from its corresponding initial Tensor. If every length is zero, neither cell is
 * invoked or initialized, the output list is empty, and all four exact initial-state references
 * are returned. Overloads that omit state create four distinct fresh typed-zero non-gradient
 * leaves in forward-hidden, forward-cell, backward-hidden, backward-cell order. Omitted lengths
 * mean every row is valid for the complete static time extent.</p>
 *
 * <p>Lengths are validated from the caller array and cloned immediately before non-empty
 * traversal; neither array is retained. Callers must coordinate array writes through validation
 * and the snapshot, and coordinate forward construction with replacement, strict loading, mode
 * changes, and discovery when one cross-direction view is required. The two automatic cells do
 * not form one initialization transaction, so completed publication or expression prefixes in
 * one direction are not rolled back after a later failure.</p>
 *
 * <p>This class creates eager index leaves and storage-free Model expressions only. It constructs
 * one compact batched call per represented step in each direction and no cell operand contains a
 * padded logical row, but it does not evaluate values, define a recurrent scan, accept runtime
 * Tensor lengths or masks, capture or compile a graph, select a backend, or promise physical work
 * skipping, graph reuse across different Java lengths, fusion, or a kernel count.</p>
 */
public final class BidirectionalLstmSequence extends Module {
    private final LstmCell forwardCell;
    private final LstmCell backwardCell;

    /**
     * Creates a sequence owning two exact, independent cells.
     *
     * <p>Complete pair and named-child validation precedes either parent link. Success retains the
     * exact cells under {@code forward} then {@code backward}; construction creates no Tensor,
     * Parameter, generator, state, or result.</p>
     *
     * @param forwardCell non-null unowned forward cell
     * @param backwardCell non-null unowned backward cell with distinct identity, equal hidden
     *     size, and exact matching parameter type
     * @throws NullPointerException if either cell is null, checked forward then backward
     * @throws IllegalArgumentException if identities, hidden sizes, or parameter types conflict
     * @throws IllegalStateException if either cell is already owned
     */
    public BidirectionalLstmSequence(LstmCell forwardCell, LstmCell backwardCell) {
        LstmCell suppliedForward = Objects.requireNonNull(forwardCell, "forwardCell");
        LstmCell suppliedBackward = Objects.requireNonNull(backwardCell, "backwardCell");
        validatePair(suppliedForward, suppliedBackward);
        Map<String, LstmCell> children = new LinkedHashMap<>();
        children.put("forward", suppliedForward);
        children.put("backward", suppliedBackward);
        registerNamedChildren(children);
        this.forwardCell = suppliedForward;
        this.backwardCell = suppliedBackward;
    }

    /**
     * Creates two independent automatically bound cells with explicit seeds.
     *
     * <p>Construction creates two unbound cells and atomically installs them only after both cell
     * constructors succeed. It creates no random generator, Tensor, Parameter, default state, or
     * length array. Equal numeric seeds are permitted and still describe separate cell-owned
     * initialization attempts and parameter identities.</p>
     *
     * @param hiddenSize positive common hidden width
     * @param bias whether each cell owns a bias
     * @param dataType non-null exact floating parameter/default-state type
     * @param weightInitialization non-null initialization policy for each cell's matrices
     * @param forwardSeed forward-cell seed
     * @param backwardSeed backward-cell seed
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if {@code hiddenSize} is not positive, {@code dataType} is
     *     not floating, or a configured packed parameter Shape exceeds the Model Java-array limit
     * @throws ArithmeticException if {@code 4 * hiddenSize} or checked Shape/count arithmetic
     *     overflows
     */
    public BidirectionalLstmSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long forwardSeed,
            long backwardSeed) {
        this(
                new LstmCell(hiddenSize, bias, dataType, weightInitialization, forwardSeed),
                new LstmCell(hiddenSize, bias, dataType, weightInitialization, backwardSeed));
    }

    /**
     * Returns the permanently owned forward-direction cell.
     *
     * @return the exact non-null forward cell
     */
    public LstmCell forwardCell() {
        return forwardCell;
    }

    /**
     * Returns the permanently owned backward-direction cell.
     *
     * @return the exact non-null backward cell
     */
    public LstmCell backwardCell() {
        return backwardCell;
    }

    /**
     * Builds the complete bidirectional static expression from explicit states and lengths.
     *
     * <p>The input and state Tensors are never mutated or retained by the module. The caller
     * array is cloned immediately before non-empty traversal. The returned list is structurally
     * immutable and retains exact merged hidden Tensor references. Entry {@code t} is shaped
     * {@code [activeCount(t), 2 * hiddenSize]} with forward hidden features before backward hidden
     * features.</p>
     *
     * @param input non-null floating fully static {@code [time,batch,inputSize]} Tensor
     * @param forwardInitialHidden non-null floating fully static forward hidden state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained by the module
     * @param forwardInitialCell non-null floating fully static forward cell state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained by the module
     * @param backwardInitialHidden non-null floating fully static backward hidden state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained by the module
     * @param backwardInitialCell non-null floating fully static backward cell state shaped
     *     {@code [batch, hiddenSize]}; not mutated or retained by the module
     * @param lengths non-null caller-owned valid-prefix lengths, one per original batch row and
     *     each in {@code [0, time]}; not mutated or retained
     * @return a non-null immutable compact hidden-output snapshot with four separate non-null
     *     directional final states in original batch order
     * @throws NullPointerException for null arguments in declaration order
     * @throws IllegalArgumentException if a static schema, length, promotion, gather, or concat
     *     contract is invalid
     * @throws ArithmeticException if checked size arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic parameter initialization fails after prevalidation
     * @throws OutOfMemoryError if an eager leaf, Tensor, expression, or supporting array cannot
     *     be allocated
     */
    public BidirectionalLstmSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor forwardInitialCell,
            Tensor backwardInitialHidden,
            Tensor backwardInitialCell,
            long[] lengths) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedForwardInitial = Objects.requireNonNull(
                forwardInitialHidden, "forwardInitialHidden");
        Tensor suppliedForwardInitialCell = Objects.requireNonNull(
                forwardInitialCell, "forwardInitialCell");
        Tensor suppliedBackwardInitial = Objects.requireNonNull(
                backwardInitialHidden, "backwardInitialHidden");
        Tensor suppliedBackwardInitialCell = Objects.requireNonNull(
                backwardInitialCell, "backwardInitialCell");
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");
        validatePair(forwardCell, backwardCell);

        Schema schema = validateInput(suppliedInput);
        validateState(suppliedForwardInitial, "forwardInitialHidden", schema);
        validateState(suppliedForwardInitialCell, "forwardInitialCell", schema);
        validateState(suppliedBackwardInitial, "backwardInitialHidden", schema);
        validateState(suppliedBackwardInitialCell, "backwardInitialCell", schema);
        long steps = validateLengths(suppliedLengths, schema);
        DataType[] forwardTypes = outputTypes(
                suppliedInput.descriptor().dataType(),
                suppliedForwardInitial.descriptor().dataType(),
                suppliedForwardInitialCell.descriptor().dataType(),
                forwardCell.configuredDataType(),
                forwardCell.configuredBias());
        DataType[] backwardTypes = outputTypes(
                suppliedInput.descriptor().dataType(),
                suppliedBackwardInitial.descriptor().dataType(),
                suppliedBackwardInitialCell.descriptor().dataType(),
                backwardCell.configuredDataType(),
                backwardCell.configuredBias());
        prevalidateComposition(
                suppliedLengths,
                schema,
                steps,
                suppliedForwardInitial.descriptor().dataType(),
                suppliedForwardInitialCell.descriptor().dataType(),
                suppliedBackwardInitial.descriptor().dataType(),
                suppliedBackwardInitialCell.descriptor().dataType(),
                forwardTypes,
                backwardTypes);

        if (steps == 0) {
            return new BidirectionalLstmSequenceForwardResult(
                    List.of(), suppliedForwardInitial, suppliedForwardInitialCell,
                    suppliedBackwardInitial, suppliedBackwardInitialCell);
        }
        long[] lengthSnapshot = suppliedLengths.clone();
        int stepCount = Math.toIntExact(steps);
        long[][] activeRows = new long[stepCount][];
        List<Tensor> forwardOutputs = new ArrayList<>(stepCount);
        List<Tensor> forwardCells = new ArrayList<>(stepCount);
        long[] previousActive = null;
        for (int timeIndex = 0; timeIndex < stepCount; timeIndex++) {
            Tensor timeSlice = suppliedInput.select(0, timeIndex);
            long[] active = activeOriginalRows(lengthSnapshot, timeIndex);
            activeRows[timeIndex] = active;
            Tensor activeIndices = indexTensor(active);
            Tensor compactInput = timeSlice.gather(activeIndices, 0);
            Tensor carryIndices = timeIndex == 0
                    ? activeIndices
                    : indexTensor(survivorPositions(previousActive, active));
            Tensor compactHidden = timeIndex == 0
                    ? suppliedForwardInitial.gather(carryIndices, 0)
                    : forwardOutputs.get(timeIndex - 1).gather(carryIndices, 0);
            Tensor compactCell = timeIndex == 0
                    ? suppliedForwardInitialCell.gather(carryIndices, 0)
                    : forwardCells.get(timeIndex - 1).gather(carryIndices, 0);
            LstmCellForwardResult step = forwardCell.forward(
                    compactInput, compactHidden, compactCell);
            forwardOutputs.add(step.nextHidden());
            forwardCells.add(step.nextCell());
            previousActive = active;
        }

        List<Tensor> reverseOutputs = new ArrayList<>(stepCount);
        List<Tensor> reverseCells = new ArrayList<>(stepCount);
        previousActive = null;
        for (int depth = 0; depth < stepCount; depth++) {
            long[] active = activeRows[depth];
            Tensor compactInput = suppliedInput.gatherNd(reverseCoordinates(
                    lengthSnapshot, active, depth));
            Tensor compactHidden;
            Tensor compactCell;
            if (depth == 0) {
                Tensor activeIndices = indexTensor(active);
                compactHidden = suppliedBackwardInitial.gather(activeIndices, 0);
                compactCell = suppliedBackwardInitialCell.gather(activeIndices, 0);
            } else {
                Tensor survivorIndices = indexTensor(survivorPositions(previousActive, active));
                compactHidden = reverseOutputs.get(depth - 1).gather(survivorIndices, 0);
                compactCell = reverseCells.get(depth - 1).gather(survivorIndices, 0);
            }
            LstmCellForwardResult step = backwardCell.forward(
                    compactInput, compactHidden, compactCell);
            reverseOutputs.add(step.nextHidden());
            reverseCells.add(step.nextCell());
            previousActive = active;
        }

        Tensor flattenedReverse = reverseOutputs.size() == 1
                ? reverseOutputs.getFirst()
                : Tensor.concat(0, reverseOutputs.toArray(Tensor[]::new));
        long[] offsets = offsets(activeRows);
        List<Tensor> packedOutputs = new ArrayList<>(stepCount);
        for (int timeIndex = 0; timeIndex < stepCount; timeIndex++) {
            long[] alignment = alignmentIndices(
                    lengthSnapshot, activeRows[timeIndex], timeIndex, offsets);
            Tensor alignedBackward = flattenedReverse.gather(indexTensor(alignment), 0);
            packedOutputs.add(Tensor.concat(
                    -1, forwardOutputs.get(timeIndex), alignedBackward));
        }

        Tensor finalIndices = indexTensor(finalStateIndices(lengthSnapshot, offsets));
        Tensor forwardSource = concatInitialAndSteps(suppliedForwardInitial, forwardOutputs);
        Tensor forwardFinal = forwardSource.gather(finalIndices, 0);
        Tensor forwardCellSource = concatInitialAndSteps(
                suppliedForwardInitialCell, forwardCells);
        Tensor forwardFinalCell = forwardCellSource.gather(finalIndices, 0);
        Tensor backwardSource = Tensor.concat(
                0, suppliedBackwardInitial, flattenedReverse);
        Tensor backwardFinal = backwardSource.gather(finalIndices, 0);
        Tensor backwardCellSource = concatInitialAndSteps(
                suppliedBackwardInitialCell, reverseCells);
        Tensor backwardFinalCell = backwardCellSource.gather(finalIndices, 0);
        return new BidirectionalLstmSequenceForwardResult(
                packedOutputs, forwardFinal, forwardFinalCell, backwardFinal, backwardFinalCell);
    }

    /**
     * Builds an all-valid expression from explicit directional hidden and cell states.
     *
     * <p>A fresh private Java length array marks every row valid for all {@code time} steps. The
     * explicit state references are not copied or retained by the module.</p>
     *
     * @param input non-null fully static time-major input
     * @param forwardInitialHidden non-null forward initial hidden state
     * @param forwardInitialCell non-null forward initial cell state
     * @param backwardInitialHidden non-null backward initial hidden state
     * @param backwardInitialCell non-null backward initial cell state
     * @return a non-null immutable compact hidden-output snapshot and four separate non-null
     *     directional final states
     * @throws NullPointerException if an argument is null in declaration order
     * @throws IllegalArgumentException if validation of the input, states, or composition fails
     * @throws ArithmeticException if checked Shape, count, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic parameter initialization fails after prevalidation
     * @throws OutOfMemoryError if a host array, eager leaf, Tensor, or expression cannot be
     *     allocated
     */
    public BidirectionalLstmSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor forwardInitialCell,
            Tensor backwardInitialHidden,
            Tensor backwardInitialCell) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedForward = Objects.requireNonNull(
                forwardInitialHidden, "forwardInitialHidden");
        Tensor suppliedForwardCell = Objects.requireNonNull(
                forwardInitialCell, "forwardInitialCell");
        Tensor suppliedBackward = Objects.requireNonNull(
                backwardInitialHidden, "backwardInitialHidden");
        Tensor suppliedBackwardCell = Objects.requireNonNull(
                backwardInitialCell, "backwardInitialCell");
        validatePair(forwardCell, backwardCell);
        Schema schema = validateInput(suppliedInput);
        validateState(suppliedForward, "forwardInitialHidden", schema);
        validateState(suppliedForwardCell, "forwardInitialCell", schema);
        validateState(suppliedBackward, "backwardInitialHidden", schema);
        validateState(suppliedBackwardCell, "backwardInitialCell", schema);
        return forward(
                suppliedInput, suppliedForward, suppliedForwardCell,
                suppliedBackward, suppliedBackwardCell, allValidLengths(schema));
    }

    /**
     * Builds from four fresh typed-zero states and explicit valid-prefix lengths.
     *
     * <p>Complete default-path validation precedes state creation in forward-hidden,
     * forward-cell, backward-hidden, backward-cell order. Every state uses the cells' exact
     * parameter type, has no label or gradient requirement, and is local to this call. An
     * all-zero request returns those exact leaves and leaves both automatic cells unbound.</p>
     *
     * @param input non-null fully static time-major input
     * @param lengths non-null valid-prefix lengths, one per batch row
     * @return a non-null immutable compact hidden-output snapshot and four separate non-null
     *     directional final states
     * @throws NullPointerException if an argument is null in declaration order
     * @throws IllegalArgumentException if validation or default-state construction fails
     * @throws ArithmeticException if checked state, Shape, count, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic parameter initialization fails after prevalidation
     * @throws OutOfMemoryError if a host array, eager leaf, Tensor, or expression cannot be
     *     allocated
     */
    public BidirectionalLstmSequenceForwardResult forward(Tensor input, long[] lengths) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        long[] suppliedLengths = Objects.requireNonNull(lengths, "lengths");
        validatePair(forwardCell, backwardCell);
        Schema schema = validateInput(suppliedInput);
        long steps = validateLengths(suppliedLengths, schema);
        prevalidateDefault(schema, suppliedLengths, steps);
        Tensor forwardInitial = zeroState(schema);
        Tensor forwardInitialCell = zeroState(schema);
        Tensor backwardInitial = zeroState(schema);
        Tensor backwardInitialCell = zeroState(schema);
        return forward(suppliedInput, forwardInitial, forwardInitialCell,
                backwardInitial, backwardInitialCell, suppliedLengths);
    }

    /**
     * Builds an all-valid expression from four fresh typed-zero directional states.
     *
     * <p>The method creates one private all-valid length array followed by the four default states
     * in documented direction/state order. None is retained by the module.</p>
     *
     * @param input non-null fully static time-major input
     * @return a non-null immutable compact hidden-output snapshot and four separate non-null
     *     directional final states
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if validation or default-state construction fails
     * @throws ArithmeticException if checked state, Shape, count, or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws RuntimeException if automatic parameter initialization fails after prevalidation
     * @throws OutOfMemoryError if a host array, eager leaf, Tensor, or expression cannot be
     *     allocated
     */
    public BidirectionalLstmSequenceForwardResult forward(Tensor input) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        validatePair(forwardCell, backwardCell);
        Schema schema = validateInput(suppliedInput);
        long[] lengths = allValidLengths(schema);
        prevalidateDefault(schema, lengths, schema.time);
        Tensor forwardInitial = zeroState(schema);
        Tensor forwardInitialCell = zeroState(schema);
        Tensor backwardInitial = zeroState(schema);
        Tensor backwardInitialCell = zeroState(schema);
        return forward(suppliedInput, forwardInitial, forwardInitialCell,
                backwardInitial, backwardInitialCell, lengths);
    }

    private static void validatePair(LstmCell forward, LstmCell backward) {
        if (forward == backward) {
            throw new IllegalArgumentException("forward and backward cells must be identity-distinct");
        }
        if (forward.configuredHiddenSize() != backward.configuredHiddenSize()) {
            throw new IllegalArgumentException("forward and backward hidden sizes must match");
        }
        if (forward.configuredDataType() != backward.configuredDataType()) {
            throw new IllegalArgumentException("forward and backward parameter data types must match");
        }
    }

    private Schema validateInput(Tensor input) {
        Shape shape = input.descriptor().shape();
        requireFloating(input.descriptor().dataType(), "input");
        requireRank(shape, 3, "input");
        requireStatic(shape, "input");
        long time = extent(shape, 0);
        long batch = extent(shape, 1);
        long inputSize = extent(shape, 2);
        if (time > Integer.MAX_VALUE || batch > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("time or batch exceeds Java collection limit");
        }
        forwardCell.validateConfiguredInputSize(inputSize);
        backwardCell.validateConfiguredInputSize(inputSize);
        Math.multiplyExact(forwardCell.configuredHiddenSize(), 2L);
        Math.multiplyExact(forwardCell.configuredHiddenSize(), 3L);
        Math.multiplyExact(forwardCell.configuredHiddenSize(), 4L);
        return new Schema(
                time, batch, inputSize, forwardCell.configuredHiddenSize(),
                input.descriptor().dataType());
    }

    private void validateState(Tensor state, String name, Schema schema) {
        Shape shape = state.descriptor().shape();
        requireFloating(state.descriptor().dataType(), name);
        requireRank(shape, 2, name);
        requireStatic(shape, name);
        if (extent(shape, 0) != schema.batch || extent(shape, 1) != schema.hiddenSize) {
            throw new IllegalArgumentException(name + " shape is incompatible with input/cell schema");
        }
        DataTypePromotion.promoteNumeric(
                state.descriptor().dataType(), forwardCell.configuredDataType());
    }

    private static long validateLengths(long[] lengths, Schema schema) {
        if (lengths.length != schema.batch) {
            throw new IllegalArgumentException("length count must equal batch extent");
        }
        long maximum = 0;
        for (int index = 0; index < lengths.length; index++) {
            long length = lengths[index];
            if (length < 0 || length > schema.time) {
                throw new IllegalArgumentException("lengths[" + index + "] is outside [0,time]");
            }
            maximum = Math.max(maximum, length);
        }
        return maximum;
    }

    private static DataType[] outputTypes(
            DataType inputType,
            DataType hiddenType,
            DataType cellType,
            DataType parameterType,
            boolean bias) {
        DataType inputProjection = DataTypePromotion.promoteNumeric(inputType, parameterType);
        if (bias) {
            inputProjection = DataTypePromotion.promoteNumeric(inputProjection, parameterType);
        }
        DataType hiddenProjection = DataTypePromotion.promoteNumeric(hiddenType, parameterType);
        DataType gateType = DataTypePromotion.promoteNumeric(inputProjection, hiddenProjection);
        DataType forgetProduct = DataTypePromotion.promoteNumeric(gateType, cellType);
        DataType inputProduct = DataTypePromotion.promoteNumeric(gateType, gateType);
        DataType nextCell = DataTypePromotion.promoteNumeric(forgetProduct, inputProduct);
        DataType nextHidden = DataTypePromotion.promoteNumeric(gateType, nextCell);
        return new DataType[] {nextHidden, nextCell};
    }

    private static void prevalidateComposition(
            long[] lengths,
            Schema schema,
            long steps,
            DataType forwardInitialType,
            DataType forwardInitialCellType,
            DataType backwardInitialType,
            DataType backwardInitialCellType,
            DataType[] forwardTypes,
            DataType[] backwardTypes) {
        if (steps == 0) {
            return;
        }
        if (forwardTypes[0] != backwardTypes[0]) {
            throw new IllegalArgumentException("directional hidden output types must match for concat");
        }
        if (forwardInitialType != forwardTypes[0]
                || forwardInitialCellType != forwardTypes[1]
                || backwardInitialType != backwardTypes[0]
                || backwardInitialCellType != backwardTypes[1]) {
            throw new IllegalArgumentException(
                    "directional initial and recurrent state types must match for restoration concat");
        }
        long representedRows = 0;
        long previousCount = -1;
        for (long step = 0; step < steps; step++) {
            long active = activeCount(lengths, step);
            representedRows = Math.addExact(representedRows, active);
            if (active > Integer.MAX_VALUE || Math.multiplyExact(active, 2L) > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("compact indices exceed Java array limit");
            }
            if (active != previousCount) {
                Shape.of(active, schema.inputSize);
                Shape.of(active, schema.hiddenSize);
                Shape.of(active, Math.multiplyExact(schema.hiddenSize, 2L));
                previousCount = active;
            }
        }
        if (representedRows > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("flattened recurrent rows exceed Java array limit");
        }
        Math.addExact(schema.batch, representedRows);
    }

    private void prevalidateDefault(Schema schema, long[] lengths, long steps) {
        long count = Math.multiplyExact(schema.batch, schema.hiddenSize);
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("default recurrent state exceeds Java array limit");
        }
        DataType type = forwardCell.configuredDataType();
        DataType[] output = outputTypes(
                schema.inputType, type, type, type, forwardCell.configuredBias());
        DataType[] backwardOutput = outputTypes(
                schema.inputType, type, type, type, backwardCell.configuredBias());
        prevalidateComposition(
                lengths, schema, steps, type, type, type, type, output, backwardOutput);
    }

    private Tensor zeroState(Schema schema) {
        return TensorFactory.zeros(
                Shape.of(schema.batch, schema.hiddenSize),
                forwardCell.configuredDataType(), Optional.empty(), false);
    }

    private static long[] allValidLengths(Schema schema) {
        long[] lengths = new long[Math.toIntExact(schema.batch)];
        Arrays.fill(lengths, schema.time);
        return lengths;
    }

    private static Tensor concatInitialAndSteps(Tensor initial, List<Tensor> steps) {
        Tensor[] inputs = new Tensor[steps.size() + 1];
        inputs[0] = initial;
        for (int index = 0; index < steps.size(); index++) {
            inputs[index + 1] = steps.get(index);
        }
        return Tensor.concat(0, inputs);
    }

    private static long[] offsets(long[][] activeRows) {
        long[] offsets = new long[activeRows.length];
        long offset = 0;
        for (int index = 0; index < activeRows.length; index++) {
            offsets[index] = offset;
            offset = Math.addExact(offset, activeRows[index].length);
        }
        return offsets;
    }

    private static long[] alignmentIndices(
            long[] lengths, long[] active, int time, long[] offsets) {
        long[] indices = new long[active.length];
        for (int position = 0; position < active.length; position++) {
            int row = Math.toIntExact(active[position]);
            int depth = Math.toIntExact(lengths[row] - 1L - time);
            indices[position] = offsets[depth] + activePosition(lengths, row, depth);
        }
        return indices;
    }

    private static long[] finalStateIndices(long[] lengths, long[] offsets) {
        long[] indices = new long[lengths.length];
        long batch = lengths.length;
        for (int row = 0; row < lengths.length; row++) {
            if (lengths[row] == 0) {
                indices[row] = row;
            } else {
                int step = Math.toIntExact(lengths[row] - 1L);
                indices[row] = batch + offsets[step] + activePosition(lengths, row, step);
            }
        }
        return indices;
    }

    private static long activePosition(long[] lengths, int row, long step) {
        long position = 0;
        for (int candidate = 0; candidate < row; candidate++) {
            if (lengths[candidate] > step) {
                position++;
            }
        }
        return position;
    }

    private static Tensor reverseCoordinates(long[] lengths, long[] active, int depth) {
        long[] coordinates = new long[Math.multiplyExact(active.length, 2)];
        for (int position = 0; position < active.length; position++) {
            int row = Math.toIntExact(active[position]);
            coordinates[2 * position] = lengths[row] - 1L - depth;
            coordinates[2 * position + 1] = row;
        }
        return indexTensor(coordinates, Shape.of(active.length, 2));
    }

    private static long activeCount(long[] lengths, long step) {
        long count = 0;
        for (long length : lengths) {
            if (length > step) {
                count++;
            }
        }
        return count;
    }

    private static long[] activeOriginalRows(long[] lengths, long step) {
        long[] active = new long[Math.toIntExact(activeCount(lengths, step))];
        int position = 0;
        for (int row = 0; row < lengths.length; row++) {
            if (lengths[row] > step) {
                active[position++] = row;
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
            positions[index] = previousPosition++;
        }
        return positions;
    }

    private static Tensor indexTensor(long[] values) {
        return indexTensor(values, Shape.of(values.length));
    }

    private static Tensor indexTensor(long[] values, Shape shape) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
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

    private record Schema(
            long time, long batch, long inputSize, long hiddenSize, DataType inputType) {
    }
}

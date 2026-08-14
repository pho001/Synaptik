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
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.ArrayList;
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
 * <p>Lengths are cloned before validation and never retained. They are construction metadata,
 * not runtime Tensor values; runtime-dependent masks or lengths require a future recurrent
 * scan/control-flow contract. Forward is mode-insensitive and retains no input, state, length,
 * index, or result. Parameter replacement and multi-step construction are not one atomic
 * snapshot, so callers coordinate them when consistent bindings matter.</p>
 *
 * <p>This class creates eager index leaves and storage-free Model expression metadata. It does
 * not evaluate values, define gradients, compile a graph, select a backend, or promise runtime
 * work skipping.</p>
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
     * Returns the stable owned LSTM cell.
     *
     * @return the exact non-null child registered under {@code cell}
     */
    public LstmCell cell() {
        return cell;
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
        long[] lengthSnapshot = Objects.requireNonNull(lengths, "lengths").clone();

        Tensor inputWeight = cell.inputWeight().value();
        Tensor hiddenWeight = cell.hiddenWeight().value();
        Optional<Tensor> bias = cell.bias().map(Parameter::value);
        long hiddenSize = validateCellSchema(inputWeight, hiddenWeight, bias);
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
        long inputSize = extent(inputWeight.descriptor().shape(), 1);
        if (extent(inputShape, 2) != inputSize) {
            throw new IllegalArgumentException(
                    "input feature size must equal cell inputSize: input="
                            + extent(inputShape, 2) + ", cell=" + inputSize);
        }
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
        if (lengthSnapshot.length != batch) {
            throw new IllegalArgumentException(
                    "length count must equal batch extent: lengths="
                            + lengthSnapshot.length + ", batch=" + batch);
        }

        long maximumLength = 0;
        for (int index = 0; index < lengthSnapshot.length; index++) {
            long length = lengthSnapshot[index];
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
                lengthSnapshot,
                maximumLength,
                suppliedInput.descriptor().dataType(),
                suppliedInitialHidden.descriptor().dataType(),
                suppliedInitialCell.descriptor().dataType(),
                inputWeight,
                hiddenWeight,
                bias,
                inputSize,
                hiddenSize);
        prevalidateFinalStacks(
                lengthSnapshot,
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
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias,
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
                inputWeight,
                hiddenWeight,
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
                inputWeight,
                hiddenWeight,
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
                        inputWeight,
                        hiddenWeight,
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
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias,
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
                inputType, inputWeight.descriptor().dataType());
        Shape inputProjectionShape = projectionShape(
                compactInputShape, inputWeight.descriptor().shape());
        DataType inputProjectionType = inputProductType;
        if (bias.isPresent()) {
            Tensor biasValue = bias.orElseThrow();
            inputProjectionType = DataTypePromotion.promoteNumeric(
                    inputProductType, biasValue.descriptor().dataType());
            inputProjectionShape = ShapeBroadcast.broadcast(
                    inputProjectionShape, biasValue.descriptor().shape());
        }
        DataType hiddenProjectionType = DataTypePromotion.promoteNumeric(
                hiddenType, hiddenWeight.descriptor().dataType());
        Shape hiddenProjectionShape = projectionShape(
                compactHiddenShape, hiddenWeight.descriptor().shape());

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

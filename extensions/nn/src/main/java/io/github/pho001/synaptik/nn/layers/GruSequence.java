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
 * Statically unrolls one time-major gated recurrent unit (GRU) sequence over explicit lengths.
 *
 * <p>The container permanently owns one {@link GruCell}. For input Shape
 * {@code [time, batch, inputSize]}, initial-hidden Shape {@code [batch, hiddenSize]}, and one
 * Java length per original batch row, step {@code t} gathers exactly the rows whose length is
 * greater than {@code t}. Those rows form the active batch for that step. The sequence invokes
 * the cell once on the compact active batch and exposes the exact returned Tensor. Original
 * relative row order is stable; no sorting or permutation is introduced. Numeric zero is
 * ordinary data and is never interpreted as padding.</p>
 *
 * <p>The result contains compact outputs separated by time step plus final hidden rows restored
 * to original batch order. A zero-length row takes its final state from the caller's initial
 * hidden Tensor. If every row has length zero, construction returns an empty output list and the
 * exact initial-hidden reference without allocating a Tensor identity.</p>
 *
 * <p>Lengths are cloned before validation and are never retained. They are Java construction
 * metadata, not Tensor values. Callers must coordinate writes that could race with the clone.
 * Runtime-dependent lengths or masks require a genuine Model recurrent scan/control-flow
 * contract and are not simulated with dense masking. Forward is mode-insensitive and retains no
 * input, state, index, length, or result. Parameter replacement and multi-step construction are
 * not one atomic snapshot; callers must coordinate them when a consistent binding set is
 * required.</p>
 *
 * <p>This class constructs eager index leaves and storage-free Model expression metadata. It does
 * not evaluate values, define gradients, capture or compile a graph, select a backend, or promise
 * that a runtime kernel skips work.</p>
 */
public final class GruSequence extends Module {
    private final GruCell cell;

    /**
     * Creates a sequence container that permanently owns the exact supplied cell.
     *
     * @param cell non-null currently unowned GRU cell to register under child name {@code cell};
     *     retained exactly
     * @throws NullPointerException if {@code cell} is null
     * @throws IllegalStateException if {@code cell} is already owned by another module
     */
    public GruSequence(GruCell cell) {
        this.cell = child("cell", Objects.requireNonNull(cell, "cell"));
    }

    /**
     * Returns the stable owned cell.
     *
     * @return the exact non-null child registered under {@code cell}
     */
    public GruCell cell() {
        return cell;
    }

    /**
     * Builds a statically packed time-major GRU expression and restores final hidden rows.
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
     * @return a non-null immutable result containing exact compact GRU outputs in increasing time
     *     order and exact final-hidden rows restored to original batch order
     * @throws NullPointerException if {@code input}, {@code initialHidden}, or {@code lengths} is
     *     null, checked in that order
     * @throws IllegalArgumentException if the current cell schema or any documented input, Shape,
     *     feature, batch, length, numeric-promotion, broadcast, gather, select, or stack contract
     *     is invalid
     * @throws ArithmeticException if checked packed-size, Model Shape, or layout arithmetic
     *     overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after valid
     *     construction begins; already constructed prefixes are not rolled back
     * @throws OutOfMemoryError if the length snapshot, a temporary Java collection or array, or
     *     eager index storage cannot be allocated; already completed construction effects are not
     *     rolled back
     */
    public GruSequenceForwardResult forward(
            Tensor input, Tensor initialHidden, long[] lengths) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        Tensor suppliedInitialHidden = Objects.requireNonNull(initialHidden, "initialHidden");
        long[] lengthSnapshot = Objects.requireNonNull(lengths, "lengths").clone();

        Tensor inputWeight = cell.inputWeight().value();
        Tensor hiddenWeight = cell.hiddenWeight().value();
        Optional<Tensor> bias = cell.bias().map(Parameter::value);
        long hiddenSize = validateCellSchema(inputWeight, hiddenWeight, bias);

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
        if (extent(hiddenShape, 0) != batch) {
            throw new IllegalArgumentException(
                    "input and initialHidden batch extents must match: input="
                            + batch + ", initialHidden=" + extent(hiddenShape, 0));
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

        DataType outputType = prevalidateSteps(
                lengthSnapshot,
                maximumLength,
                suppliedInput.descriptor().dataType(),
                suppliedInitialHidden.descriptor().dataType(),
                inputWeight,
                hiddenWeight,
                bias,
                inputSize,
                hiddenSize);
        prevalidateFinalStack(
                lengthSnapshot,
                suppliedInitialHidden.descriptor().dataType(),
                hiddenShape,
                outputType,
                hiddenSize,
                maximumLength);

        if (maximumLength == 0) {
            return new GruSequenceForwardResult(List.of(), suppliedInitialHidden);
        }

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
        return new GruSequenceForwardResult(packedOutputs, finalHidden);
    }

    private static DataType prevalidateSteps(
            long[] lengths,
            long maximumLength,
            DataType inputType,
            DataType hiddenType,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias,
            long inputSize,
            long hiddenSize) {
        DataType inputProductType = DataTypePromotion.promoteNumeric(
                inputType, inputWeight.descriptor().dataType());
        DataType inputProjectionType = bias
                .map(value -> DataTypePromotion.promoteNumeric(
                        inputProductType, value.descriptor().dataType()))
                .orElse(inputProductType);
        DataType hiddenProjectionType = DataTypePromotion.promoteNumeric(
                hiddenType, hiddenWeight.descriptor().dataType());

        DataType resetType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        requireFloating(resetType, "reset preactivation");
        DataType updateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, hiddenProjectionType);
        requireFloating(updateType, "update preactivation");
        DataType resetProductType = DataTypePromotion.promoteNumeric(
                resetType, hiddenProjectionType);
        DataType candidateType = DataTypePromotion.promoteNumeric(
                inputProjectionType, resetProductType);
        requireFloating(candidateType, "candidate preactivation");
        DataType differenceType = DataTypePromotion.promoteNumeric(hiddenType, candidateType);
        DataType weightedDifferenceType = DataTypePromotion.promoteNumeric(
                updateType, differenceType);
        DataType outputType = DataTypePromotion.promoteNumeric(
                candidateType, weightedDifferenceType);

        long packedHiddenSize = Math.multiplyExact(hiddenSize, 3L);
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
                Shape inputProjectionShape = projectionShape(compactInputShape, packedHiddenSize);
                Shape hiddenProjectionShape = projectionShape(compactHiddenShape, packedHiddenSize);
                Shape inputGateShape = gateShape(inputProjectionShape, hiddenSize);
                Shape hiddenGateShape = gateShape(hiddenProjectionShape, hiddenSize);

                Shape resetShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
                Shape updateShape = ShapeBroadcast.broadcast(inputGateShape, hiddenGateShape);
                Shape resetProductShape = ShapeBroadcast.broadcast(resetShape, hiddenGateShape);
                Shape candidateShape = ShapeBroadcast.broadcast(inputGateShape, resetProductShape);
                Shape differenceShape = ShapeBroadcast.broadcast(compactHiddenShape, candidateShape);
                Shape weightedDifferenceShape = ShapeBroadcast.broadcast(updateShape, differenceShape);
                Shape resultShape = ShapeBroadcast.broadcast(candidateShape, weightedDifferenceShape);
                Shape expectedShape = Shape.of(activeCount, hiddenSize);
                if (!resultShape.equals(expectedShape)) {
                    throw new IllegalArgumentException(
                            "cell step produced an unexpected shape: " + resultShape);
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

    private static Shape projectionShape(Shape inputShape, long outputSize) {
        Dimension[] dimensions = new Dimension[inputShape.rank()];
        for (int axis = 0; axis < inputShape.rank() - 1; axis++) {
            dimensions[axis] = inputShape.dimension(axis);
        }
        dimensions[dimensions.length - 1] = new StaticDimension(outputSize);
        return Shape.ofDimensions(dimensions);
    }

    private static Shape gateShape(Shape projectionShape, long hiddenSize) {
        return projectionShape(projectionShape, hiddenSize);
    }

    private static long validateCellSchema(
            Tensor inputWeight, Tensor hiddenWeight, Optional<Tensor> bias) {
        long hiddenSize = validateInputWeight(inputWeight);
        validateHiddenWeight(hiddenWeight);
        DataType inputType = inputWeight.descriptor().dataType();
        DataType hiddenType = hiddenWeight.descriptor().dataType();
        if (hiddenType != inputType) {
            throw new IllegalArgumentException("cell weight data types must match");
        }
        Shape inputShape = inputWeight.descriptor().shape();
        Shape hiddenShape = hiddenWeight.descriptor().shape();
        long packedHiddenSize = extent(inputShape, 0);
        if (extent(hiddenShape, 0) != packedHiddenSize
                || extent(hiddenShape, 1) != hiddenSize) {
            throw new IllegalArgumentException(
                    "cell hiddenWeight must match packed extent and hiddenSize");
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
                    || extent(biasShape, 0) != packedHiddenSize) {
                throw new IllegalArgumentException("cell bias schema is incompatible with weights");
            }
        }
        return hiddenSize;
    }

    private static long validateInputWeight(Tensor weight) {
        validateFloatingAndGradient(weight, "inputWeight");
        Shape shape = weight.descriptor().shape();
        requireRank(shape, 2, "cell inputWeight");
        requireStatic(shape, "cell inputWeight");
        long packedHiddenSize = extent(shape, 0);
        if (packedHiddenSize <= 0) {
            throw new IllegalArgumentException(
                    "cell inputWeight packed hidden size must be positive: " + packedHiddenSize);
        }
        if (packedHiddenSize % 3L != 0L) {
            throw new IllegalArgumentException(
                    "cell inputWeight packed hidden size must be divisible by three: "
                            + packedHiddenSize);
        }
        long inputSize = extent(shape, 1);
        if (inputSize <= 0) {
            throw new IllegalArgumentException(
                    "cell inputWeight inputSize must be positive: " + inputSize);
        }
        return packedHiddenSize / 3L;
    }

    private static void validateHiddenWeight(Tensor weight) {
        validateFloatingAndGradient(weight, "hiddenWeight");
        Shape shape = weight.descriptor().shape();
        requireRank(shape, 2, "cell hiddenWeight");
        requireStatic(shape, "cell hiddenWeight");
        if (extent(shape, 0) <= 0 || extent(shape, 1) <= 0) {
            throw new IllegalArgumentException("cell hiddenWeight extents must be positive");
        }
    }

    private static void validateFloatingAndGradient(Tensor weight, String name) {
        requireFloating(weight.descriptor().dataType(), "cell " + name);
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException("cell " + name + " must have requiresGrad == true");
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

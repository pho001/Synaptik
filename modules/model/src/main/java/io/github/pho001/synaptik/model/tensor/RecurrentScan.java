package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Advanced low-level static namespace for locally validated fixed recurrent-scan expressions.
 *
 * <p>The six {@link #rnn}-, {@link #gru}-, and {@link #lstm}-family overloads construct the
 * bias-free or explicitly biased fixed recurrent meanings from a time-major {@code input}, an
 * ordinary runtime valid-length Tensor, explicit initial states, packed weights, and one
 * direction. Every successful call creates one fresh flat multi-output producer and returns its
 * canonical output wrappers in the appropriate typed result record. The namespace is not a
 * neural-network layer or module, execution service, registry, or general scan-body API; ordinary
 * neural-network composition continues to use the NN {@code RnnSequence}, {@code GruSequence},
 * and {@code LstmSequence} types.</p>
 *
 * <p>This field-free namespace owns family-specific static descriptor checks, exact input and
 * output ordering, operation construction, and public result assembly. It never reads Tensor
 * storage or valid-length values, unrolls a transition, captures a body or region, constructs a
 * gradient, selects a backend, or executes recurrence. The existing NN sequence APIs still use
 * construction-time Java {@code long[]} lengths and static unrolling. Current Compiler inference
 * and autograd inventories remain fail-closed for the resulting operation family.</p>
 */
public final class RecurrentScan {
    private RecurrentScan() {
    }

    /**
     * Creates one bias-free fixed tanh-RNN scan.
     *
     * <p>The occurrence records ordered inputs
     * {@code [input, validLengths, initialHidden, inputWeight, hiddenWeight]} and the transition
     * {@code nextHidden = tanh((x @ inputWeight^T) + (hidden @ hiddenWeight^T))}. Its canonical
     * results are dense original-time-aligned {@code [time, batch, hiddenSize]} output and
     * {@code [batch, hiddenSize]} final hidden state. Direction selects traversal of each valid
     * prefix; padded positions are semantically positive zero and a zero-length row preserves its
     * initial state. Construction records that meaning without reading or executing values.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed weight shaped {@code [hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed weight shaped {@code [hiddenSize, hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output and final-hidden wrappers from one exact producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static RecurrentScanResult rnn(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.RNN_TANH,
                1,
                input,
                validLengths,
                initialHidden,
                null,
                inputWeight,
                hiddenWeight,
                null,
                direction);
        return new RecurrentScanResult(outputs.get(0), outputs.get(1));
    }

    /**
     * Creates one biased fixed tanh-RNN scan.
     *
     * <p>This variant records ordered inputs
     * {@code [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]}. It adds the
     * exact bias only to the input projection and otherwise retains the bias-free transition,
     * dense output, final-state, traversal, padding, provenance, and non-execution contracts.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed weight shaped {@code [hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed weight shaped {@code [hiddenSize, hiddenSize]}
     * @param bias non-null common-typed input-side bias shaped {@code [hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output and final-hidden wrappers from one exact producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static RecurrentScanResult rnn(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Tensor bias,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.RNN_TANH,
                1,
                input,
                validLengths,
                initialHidden,
                null,
                inputWeight,
                hiddenWeight,
                bias,
                direction);
        return new RecurrentScanResult(outputs.get(0), outputs.get(1));
    }

    /**
     * Creates one bias-free reset-after GRU scan with reset/update/candidate gate packing.
     *
     * <p>The occurrence records ordered inputs
     * {@code [input, validLengths, initialHidden, inputWeight, hiddenWeight]}. With packed reset,
     * update, and candidate intervals, reset is applied after the recurrent candidate projection
     * and the fixed update is
     * {@code candidate + update * (hidden - candidate)}. Canonical results are dense
     * original-time-aligned output and final hidden state with the shared traversal, padding, and
     * non-execution contracts.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed packed weight shaped
     *     {@code [3 * hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed packed weight shaped
     *     {@code [3 * hiddenSize, hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output and final-hidden wrappers from one exact producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static RecurrentScanResult gru(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.GRU_RESET_AFTER,
                3,
                input,
                validLengths,
                initialHidden,
                null,
                inputWeight,
                hiddenWeight,
                null,
                direction);
        return new RecurrentScanResult(outputs.get(0), outputs.get(1));
    }

    /**
     * Creates one biased reset-after GRU scan with reset/update/candidate gate packing.
     *
     * <p>This variant records ordered inputs
     * {@code [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]}. It adds the
     * exact packed bias only to the input projection and otherwise retains the bias-free
     * reset-after equations, dense output, final-state, traversal, padding, provenance, and
     * non-execution contracts.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed packed weight shaped
     *     {@code [3 * hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed packed weight shaped
     *     {@code [3 * hiddenSize, hiddenSize]}
     * @param bias non-null common-typed input-side packed bias shaped {@code [3 * hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output and final-hidden wrappers from one exact producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static RecurrentScanResult gru(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Tensor bias,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.GRU_RESET_AFTER,
                3,
                input,
                validLengths,
                initialHidden,
                null,
                inputWeight,
                hiddenWeight,
                bias,
                direction);
        return new RecurrentScanResult(outputs.get(0), outputs.get(1));
    }

    /**
     * Creates one bias-free LSTM scan with input/forget/candidate/output gate packing.
     *
     * <p>The occurrence records ordered inputs
     * {@code [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]}. With
     * packed input, forget, candidate, and output intervals, the fixed transition computes
     * {@code nextCell = forget * cell + inputGate * candidate} and
     * {@code nextHidden = outputGate * tanh(nextCell)}. Canonical results are dense
     * original-time-aligned output, final hidden state, and final cell state; zero-length rows
     * preserve both initial states. Construction records traversal and positive-zero padding
     * meaning without reading or executing values.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed hidden state shaped {@code [batch, hiddenSize]}
     * @param initialCell non-null common-typed cell state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed packed weight shaped
     *     {@code [4 * hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed packed weight shaped
     *     {@code [4 * hiddenSize, hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output, final-hidden, and final-cell wrappers from one producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static LstmRecurrentScanResult lstm(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor initialCell,
            Tensor inputWeight,
            Tensor hiddenWeight,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(initialCell, "initialCell");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.LSTM,
                4,
                input,
                validLengths,
                initialHidden,
                initialCell,
                inputWeight,
                hiddenWeight,
                null,
                direction);
        return new LstmRecurrentScanResult(outputs.get(0), outputs.get(1), outputs.get(2));
    }

    /**
     * Creates one biased LSTM scan with input/forget/candidate/output gate packing.
     *
     * <p>This variant records ordered inputs
     * {@code [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]}.
     * It adds the exact packed bias only to the input projection and otherwise retains the
     * bias-free LSTM equations, dense output, explicit final states, traversal, padding,
     * provenance, and non-execution contracts.</p>
     *
     * @param input non-null time-major floating input shaped {@code [time, batch, inputSize]}
     * @param validLengths non-null non-gradient INT64 vector shaped {@code [batch]}
     * @param initialHidden non-null common-typed hidden state shaped {@code [batch, hiddenSize]}
     * @param initialCell non-null common-typed cell state shaped {@code [batch, hiddenSize]}
     * @param inputWeight non-null common-typed packed weight shaped
     *     {@code [4 * hiddenSize, inputSize]}
     * @param hiddenWeight non-null common-typed packed weight shaped
     *     {@code [4 * hiddenSize, hiddenSize]}
     * @param bias non-null common-typed input-side packed bias shaped {@code [4 * hiddenSize]}
     * @param direction non-null traversal direction retained as the exact operation attributes
     * @return fresh canonical dense-output, final-hidden, and final-cell wrappers from one producer
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if a descriptor-visible recurrent requirement fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static LstmRecurrentScanResult lstm(
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor initialCell,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Tensor bias,
            RecurrentDirection direction) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(initialHidden, "initialHidden");
        Objects.requireNonNull(initialCell, "initialCell");
        Objects.requireNonNull(inputWeight, "inputWeight");
        Objects.requireNonNull(hiddenWeight, "hiddenWeight");
        Objects.requireNonNull(bias, "bias");
        Objects.requireNonNull(direction, "direction");
        List<Tensor> outputs = recurrent(
                RecurrentScanKind.LSTM,
                4,
                input,
                validLengths,
                initialHidden,
                initialCell,
                inputWeight,
                hiddenWeight,
                bias,
                direction);
        return new LstmRecurrentScanResult(outputs.get(0), outputs.get(1), outputs.get(2));
    }

    private static List<Tensor> recurrent(
            RecurrentScanKind kind,
            int gateCount,
            Tensor input,
            Tensor validLengths,
            Tensor initialHidden,
            Tensor initialCell,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Tensor bias,
            RecurrentDirection direction) {
        DataType commonType = input.descriptor().dataType();
        if (!commonType.isFloating()) {
            throw new IllegalArgumentException(
                    "recurrent scan input must have a floating data type, but was " + commonType);
        }
        Shape inputShape = requireStaticRank(input, "input", 3);
        StaticDimension time = staticDimension(inputShape, 0);
        StaticDimension batch = staticDimension(inputShape, 1);
        StaticDimension inputSize = staticDimension(inputShape, 2);
        if (inputSize.size() == 0) {
            throw new IllegalArgumentException("recurrent scan inputSize must be positive");
        }

        if (validLengths.descriptor().dataType() != DataType.INT64) {
            throw new IllegalArgumentException(
                    "recurrent scan validLengths data type must be INT64, but was "
                            + validLengths.descriptor().dataType());
        }
        if (validLengths.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "recurrent scan validLengths must not require gradients");
        }
        Shape lengthsShape = requireStaticRank(validLengths, "validLengths", 1);
        requireExtent(lengthsShape, 0, batch.size(), "validLengths", "batch");

        requireExactType(initialHidden, commonType, "initialHidden");
        Shape hiddenShape = requireStaticRank(initialHidden, "initialHidden", 2);
        requireExtent(hiddenShape, 0, batch.size(), "initialHidden", "batch");
        StaticDimension hiddenSize = staticDimension(hiddenShape, 1);
        if (hiddenSize.size() == 0) {
            throw new IllegalArgumentException("recurrent scan hiddenSize must be positive");
        }

        if (initialCell != null) {
            requireExactType(initialCell, commonType, "initialCell");
            Shape cellShape = requireStaticRank(initialCell, "initialCell", 2);
            requireExtent(cellShape, 0, batch.size(), "initialCell", "batch");
            requireExtent(cellShape, 1, hiddenSize.size(), "initialCell", "hiddenSize");
        }

        long packedSize;
        try {
            packedSize = Math.multiplyExact((long) gateCount, hiddenSize.size());
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException(
                    "recurrent scan packed gate extent overflow: gateCount=" + gateCount
                            + ", hiddenSize=" + hiddenSize.size(),
                    overflow);
        }

        requireExactType(inputWeight, commonType, "inputWeight");
        Shape inputWeightShape = requireStaticRank(inputWeight, "inputWeight", 2);
        requireExtent(inputWeightShape, 0, packedSize, "inputWeight", "packedHiddenSize");
        requireExtent(inputWeightShape, 1, inputSize.size(), "inputWeight", "inputSize");

        requireExactType(hiddenWeight, commonType, "hiddenWeight");
        Shape hiddenWeightShape = requireStaticRank(hiddenWeight, "hiddenWeight", 2);
        requireExtent(hiddenWeightShape, 0, packedSize, "hiddenWeight", "packedHiddenSize");
        requireExtent(hiddenWeightShape, 1, hiddenSize.size(), "hiddenWeight", "hiddenSize");

        if (bias != null) {
            requireExactType(bias, commonType, "bias");
            Shape biasShape = requireStaticRank(bias, "bias", 1);
            requireExtent(biasShape, 0, packedSize, "bias", "packedHiddenSize");
        }

        boolean requiresGrad = input.descriptor().requiresGrad()
                || initialHidden.descriptor().requiresGrad()
                || (initialCell != null && initialCell.descriptor().requiresGrad())
                || inputWeight.descriptor().requiresGrad()
                || hiddenWeight.descriptor().requiresGrad()
                || (bias != null && bias.descriptor().requiresGrad());
        Shape outputShape = Shape.ofDimensions(time, batch, hiddenSize);
        TensorDescriptor outputDescriptor = descriptor(commonType, outputShape, requiresGrad);
        TensorDescriptor finalHiddenDescriptor = descriptor(
                commonType, hiddenShape, requiresGrad);

        List<Tensor> inputs;
        List<TensorDescriptor> outputDescriptors;
        if (initialCell == null) {
            inputs = bias == null
                    ? List.of(input, validLengths, initialHidden, inputWeight, hiddenWeight)
                    : List.of(input, validLengths, initialHidden, inputWeight, hiddenWeight, bias);
            outputDescriptors = List.of(outputDescriptor, finalHiddenDescriptor);
        } else {
            inputs = bias == null
                    ? List.of(
                            input,
                            validLengths,
                            initialHidden,
                            initialCell,
                            inputWeight,
                            hiddenWeight)
                    : List.of(
                            input,
                            validLengths,
                            initialHidden,
                            initialCell,
                            inputWeight,
                            hiddenWeight,
                            bias);
            outputDescriptors = List.of(
                    outputDescriptor,
                    finalHiddenDescriptor,
                    descriptor(commonType, initialCell.descriptor().shape(), requiresGrad));
        }

        Operation operation = new Operation(kind, direction);
        return TensorFactory.createDerivedOutputs(operation, inputs, outputDescriptors);
    }

    private static Shape requireStaticRank(Tensor tensor, String role, int expectedRank) {
        Shape shape = tensor.descriptor().shape();
        if (shape.rank() != expectedRank) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " rank must be " + expectedRank
                            + ", but was " + shape.rank());
        }
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " must have a fully static shape: " + shape);
        }
        return shape;
    }

    private static StaticDimension staticDimension(Shape shape, int axis) {
        return (StaticDimension) shape.dimension(axis);
    }

    private static void requireExactType(Tensor tensor, DataType commonType, String role) {
        DataType actual = tensor.descriptor().dataType();
        if (actual != commonType) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " data type must match input: input="
                            + commonType + ", " + role + "=" + actual);
        }
    }

    private static void requireExtent(
            Shape shape, int axis, long expected, String role, String extentRole) {
        long actual = staticDimension(shape, axis).size();
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " " + extentRole + " extent mismatch: expected="
                            + expected + ", actual=" + actual);
        }
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }
}

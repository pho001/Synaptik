package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Independently derives the fully static descriptors of one fixed recurrent-scan occurrence.
 *
 * <p>The helper consumes only the immutable operation and ordered logical descriptors retained by
 * a captured node. It does not inspect Tensor storage or valid-length values, construct a loop or
 * nested graph, select a backend, allocate executable state, or provide a gradient rule.</p>
 */
final class RecurrentScanInference {
    private RecurrentScanInference() {}

    /**
     * Validates one closed RNN, GRU, or LSTM signature and derives every ordered output.
     *
     * @param operation non-null operation with an exact recurrent kind and direction attribute
     * @param inputs non-null ordered input descriptors from the captured occurrence
     * @param outputCount stored output cardinality, which must be two for RNN/GRU or three for
     *     LSTM
     * @return an immutable unconstrained result containing exact descriptors in recurrent output
     *     order; never {@code null}
     * @throws NullPointerException if {@code operation}, {@code inputs}, or an input descriptor is
     *     {@code null}
     * @throws IllegalArgumentException if the kind, attributes, cardinality, data types, gradient
     *     roles, or fully static Shapes violate the fixed recurrent contract
     */
    static CapturedGraphInference.InferenceResult infer(
            Operation operation, List<TensorDescriptor> inputs, int outputCount) {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(inputs, "inputs");
        for (int index = 0; index < inputs.size(); index++) {
            Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]");
        }

        if (!(operation.kind() instanceof RecurrentScanKind kind)) {
            throw new IllegalArgumentException("unsupported recurrent kind");
        }
        if (!(operation.attrs() instanceof RecurrentDirection)) {
            throw new IllegalArgumentException("recurrent attributes must be RecurrentDirection");
        }

        boolean lstm = kind == RecurrentScanKind.LSTM;
        int minimumInputs = lstm ? 6 : 5;
        int maximumInputs = minimumInputs + 1;
        int expectedOutputs = lstm ? 3 : 2;
        if (inputs.size() != minimumInputs && inputs.size() != maximumInputs) {
            throw new IllegalArgumentException(
                    "recurrent input count must be " + minimumInputs + " or " + maximumInputs
                            + ", but was " + inputs.size());
        }
        if (outputCount != expectedOutputs) {
            throw new IllegalArgumentException(
                    "recurrent output count must be " + expectedOutputs + ", but was "
                            + outputCount);
        }

        TensorDescriptor input = inputs.get(0);
        DataType commonType = input.dataType();
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

        TensorDescriptor validLengths = inputs.get(1);
        if (validLengths.dataType() != DataType.INT64) {
            throw new IllegalArgumentException(
                    "recurrent scan validLengths data type must be INT64, but was "
                            + validLengths.dataType());
        }
        if (validLengths.requiresGrad()) {
            throw new IllegalArgumentException(
                    "recurrent scan validLengths must not require gradients");
        }
        Shape lengthsShape = requireStaticRank(validLengths, "validLengths", 1);
        requireExtent(lengthsShape, 0, batch.size(), "validLengths", "batch");

        TensorDescriptor initialHidden = inputs.get(2);
        requireExactType(initialHidden, commonType, "initialHidden");
        Shape hiddenShape = requireStaticRank(initialHidden, "initialHidden", 2);
        requireExtent(hiddenShape, 0, batch.size(), "initialHidden", "batch");
        StaticDimension hiddenSize = staticDimension(hiddenShape, 1);
        if (hiddenSize.size() == 0) {
            throw new IllegalArgumentException("recurrent scan hiddenSize must be positive");
        }

        int inputWeightIndex;
        boolean requiresGrad = input.requiresGrad() || initialHidden.requiresGrad();
        if (lstm) {
            TensorDescriptor initialCell = inputs.get(3);
            requireExactType(initialCell, commonType, "initialCell");
            Shape cellShape = requireStaticRank(initialCell, "initialCell", 2);
            requireExtent(cellShape, 0, batch.size(), "initialCell", "batch");
            requireExtent(cellShape, 1, hiddenSize.size(), "initialCell", "hiddenSize");
            requiresGrad |= initialCell.requiresGrad();
            inputWeightIndex = 4;
        } else {
            inputWeightIndex = 3;
        }

        long gateCount = switch (kind) {
            case RNN_TANH -> 1;
            case GRU_RESET_AFTER -> 3;
            case LSTM -> 4;
        };
        long packedSize;
        try {
            packedSize = Math.multiplyExact(gateCount, hiddenSize.size());
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException(
                    "recurrent scan packed gate extent overflow: gateCount=" + gateCount
                            + ", hiddenSize=" + hiddenSize.size(),
                    overflow);
        }

        TensorDescriptor inputWeight = inputs.get(inputWeightIndex);
        requireExactType(inputWeight, commonType, "inputWeight");
        Shape inputWeightShape = requireStaticRank(inputWeight, "inputWeight", 2);
        requireExtent(inputWeightShape, 0, packedSize, "inputWeight", "packedHiddenSize");
        requireExtent(inputWeightShape, 1, inputSize.size(), "inputWeight", "inputSize");

        TensorDescriptor hiddenWeight = inputs.get(inputWeightIndex + 1);
        requireExactType(hiddenWeight, commonType, "hiddenWeight");
        Shape hiddenWeightShape = requireStaticRank(hiddenWeight, "hiddenWeight", 2);
        requireExtent(hiddenWeightShape, 0, packedSize, "hiddenWeight", "packedHiddenSize");
        requireExtent(hiddenWeightShape, 1, hiddenSize.size(), "hiddenWeight", "hiddenSize");
        requiresGrad |= inputWeight.requiresGrad() || hiddenWeight.requiresGrad();

        if (inputs.size() == maximumInputs) {
            TensorDescriptor bias = inputs.get(inputWeightIndex + 2);
            requireExactType(bias, commonType, "bias");
            Shape biasShape = requireStaticRank(bias, "bias", 1);
            requireExtent(biasShape, 0, packedSize, "bias", "packedHiddenSize");
            requiresGrad |= bias.requiresGrad();
        }

        TensorDescriptor outputs = descriptor(
                commonType,
                Shape.ofDimensions(time, batch, hiddenSize),
                requiresGrad);
        TensorDescriptor finalHidden = descriptor(commonType, hiddenShape, requiresGrad);
        if (lstm) {
            TensorDescriptor finalCell = descriptor(commonType, hiddenShape, requiresGrad);
            return CapturedGraphInference.InferenceResult.of(
                    outputs, finalHidden, finalCell);
        }
        return CapturedGraphInference.InferenceResult.of(outputs, finalHidden);
    }

    private static Shape requireStaticRank(
            TensorDescriptor descriptor, String role, int expectedRank) {
        Shape shape = descriptor.shape();
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

    private static void requireExactType(
            TensorDescriptor descriptor, DataType commonType, String role) {
        if (descriptor.dataType() != commonType) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " data type must match input: input="
                            + commonType + ", " + role + "=" + descriptor.dataType());
        }
    }

    private static void requireExtent(
            Shape shape, int axis, long expected, String role, String extentRole) {
        long actual = staticDimension(shape, axis).size();
        if (actual != expected) {
            throw new IllegalArgumentException(
                    "recurrent scan " + role + " " + extentRole
                            + " extent mismatch: expected=" + expected + ", actual=" + actual);
        }
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }
}

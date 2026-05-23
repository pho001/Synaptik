package backend.cpu.fused.ir;

import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Internal access plan for one runtime input of a fused expression.
 */
public record FusedExternalInputPlan(
        int inputIndex,
        DataType dataType,
        int[] inputShape,
        int[] inputStrides,
        int[] logicalOutputShape,
        int[] logicalOutputDenseStrides,
        int storageOffset,
        int[] effectiveStrides,
        FusedAccessKind accessKind
) {
    public FusedExternalInputPlan(
            int inputIndex,
            DataType dataType,
            int[] logicalOutputShape,
            int[] logicalOutputDenseStrides,
            int storageOffset,
            int[] effectiveStrides,
            FusedAccessKind accessKind
    ) {
        this(
                inputIndex,
                dataType,
                logicalOutputShape,
                effectiveStrides,
                logicalOutputShape,
                logicalOutputDenseStrides,
                storageOffset,
                effectiveStrides,
                accessKind
        );
    }

    public FusedExternalInputPlan {
        if (inputIndex < 0) {
            throw new IllegalArgumentException("inputIndex must be >= 0");
        }
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(inputShape, "inputShape cannot be null");
        Objects.requireNonNull(inputStrides, "inputStrides cannot be null");
        Objects.requireNonNull(logicalOutputShape, "logicalOutputShape cannot be null");
        Objects.requireNonNull(logicalOutputDenseStrides, "logicalOutputDenseStrides cannot be null");
        Objects.requireNonNull(effectiveStrides, "effectiveStrides cannot be null");
        Objects.requireNonNull(accessKind, "accessKind cannot be null");
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
        inputShape = inputShape.clone();
        inputStrides = inputStrides.clone();
        logicalOutputShape = logicalOutputShape.clone();
        logicalOutputDenseStrides = logicalOutputDenseStrides.clone();
        effectiveStrides = effectiveStrides.clone();
        if (inputShape.length != inputStrides.length) {
            throw new IllegalArgumentException(
                    "Fused source metadata rank mismatch for input " + inputIndex
                            + ": inputShape=" + Arrays.toString(inputShape)
            );
        }
        if (logicalOutputShape.length != logicalOutputDenseStrides.length
                || logicalOutputShape.length != effectiveStrides.length) {
            throw new IllegalArgumentException(
                    "Fused access metadata rank mismatch for input " + inputIndex
                            + ": shape=" + Arrays.toString(logicalOutputShape)
            );
        }
    }

    public boolean usesCursor() {
        return !isLinearAccess();
    }

    public boolean isLinearAccess() {
        return Arrays.equals(effectiveStrides, logicalOutputDenseStrides);
    }

    @Override
    public int[] inputShape() {
        return inputShape.clone();
    }

    @Override
    public int[] inputStrides() {
        return inputStrides.clone();
    }

    @Override
    public int[] logicalOutputShape() {
        return logicalOutputShape.clone();
    }

    @Override
    public int[] logicalOutputDenseStrides() {
        return logicalOutputDenseStrides.clone();
    }

    @Override
    public int[] effectiveStrides() {
        return effectiveStrides.clone();
    }
}

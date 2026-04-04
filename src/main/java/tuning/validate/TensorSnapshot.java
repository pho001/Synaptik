package tuning.validate;

import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;

public record TensorSnapshot(
        String label,
        DataType dataType,
        int[] shape,
        double[] numericValues,
        boolean[] boolValues
) {
    public TensorSnapshot {
        label = label == null ? "" : label;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        numericValues = numericValues == null ? null : numericValues.clone();
        boolValues = boolValues == null ? null : boolValues.clone();
    }

    public static TensorSnapshot capture(String label, Tensor tensor) {
        if (tensor == null) {
            throw new IllegalArgumentException("tensor cannot be null");
        }
        if (tensor.getDataType() == DataType.BOOL) {
            return new TensorSnapshot(
                    label == null ? tensor.getLabel() : label,
                    tensor.getDataType(),
                    tensor.getShapeUnsafe(),
                    null,
                    tensor.toBooleanArrayCopy()
            );
        }
        return new TensorSnapshot(
                label == null ? tensor.getLabel() : label,
                tensor.getDataType(),
                tensor.getShapeUnsafe(),
                tensor.toDoubleArrayCopy(),
                null
        );
    }

    public boolean shapeEquals(Tensor tensor) {
        return Arrays.equals(shape, tensor.getShapeUnsafe());
    }
}

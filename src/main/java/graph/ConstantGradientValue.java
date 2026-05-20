package graph;

import tensor.DataType;
import tensor.Tensor;

import java.util.Objects;

/**
 * Immutable logical tensor value used for constant gradient publication.
 */
public record ConstantGradientValue(
        DataType dataType,
        int[] shape,
        double[] float64Values,
        float[] float32Values,
        short[] bfloat16Values
) {
    public ConstantGradientValue {
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        float64Values = float64Values == null ? null : float64Values.clone();
        float32Values = float32Values == null ? null : float32Values.clone();
        bfloat16Values = bfloat16Values == null ? null : bfloat16Values.clone();
        int populated = (float64Values == null ? 0 : 1)
                + (float32Values == null ? 0 : 1)
                + (bfloat16Values == null ? 0 : 1);
        if (populated != 1) {
            throw new IllegalArgumentException("Exactly one gradient value payload must be populated.");
        }
    }

    public static ConstantGradientValue capture(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return switch (tensor.getDataType()) {
            case FLOAT64 -> new ConstantGradientValue(
                    DataType.FLOAT64,
                    tensor.getShape(),
                    tensor.toFloat64ArrayCopy(),
                    null,
                    null
            );
            case FLOAT32 -> new ConstantGradientValue(
                    DataType.FLOAT32,
                    tensor.getShape(),
                    null,
                    tensor.toFloat32ArrayCopy(),
                    null
            );
            case BFLOAT16 -> new ConstantGradientValue(
                    DataType.BFLOAT16,
                    tensor.getShape(),
                    null,
                    null,
                    tensor.toBFloat16BitsArrayCopy()
            );
            case INT32, INT64, BOOL -> throw GradientDTypePolicy.unsupportedGradientDType(
                    tensor.getDataType(),
                    "constant gradient snapshot"
            );
        };
    }

    public Tensor toTensor(String label) {
        return switch (dataType) {
            case FLOAT64 -> new Tensor(float64Values(), shape(), null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(float32Values(), shape(), null, label, DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(bfloat16Values(), shape(), null, label, DataType.BFLOAT16);
            case INT32, INT64, BOOL -> throw GradientDTypePolicy.unsupportedGradientDType(
                    dataType,
                    "constant gradient publication"
            );
        };
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public double[] float64Values() {
        return float64Values == null ? null : float64Values.clone();
    }

    @Override
    public float[] float32Values() {
        return float32Values == null ? null : float32Values.clone();
    }

    @Override
    public short[] bfloat16Values() {
        return bfloat16Values == null ? null : bfloat16Values.clone();
    }
}

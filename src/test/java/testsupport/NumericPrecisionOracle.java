package testsupport;

import tensor.dtype.TensorDTypeOps;
import tensor.DataType;
import utils.FastTranscendentals;

public final class NumericPrecisionOracle {
    private NumericPrecisionOracle() {}

    public static double add(double a, double b, DataType dataType) {
        return cast(cast(a, dataType) + cast(b, dataType), dataType);
    }

    public static double sub(double a, double b, DataType dataType) {
        return cast(cast(a, dataType) - cast(b, dataType), dataType);
    }

    public static double mul(double a, double b, DataType dataType) {
        return cast(cast(a, dataType) * cast(b, dataType), dataType);
    }

    public static double div(double a, double b, DataType dataType) {
        return cast(cast(a, dataType) / cast(b, dataType), dataType);
    }

    public static double min(double a, double b, DataType dataType) {
        return cast(Math.min(cast(a, dataType), cast(b, dataType)), dataType);
    }

    public static double max(double a, double b, DataType dataType) {
        return cast(Math.max(cast(a, dataType), cast(b, dataType)), dataType);
    }

    public static double neg(double a, DataType dataType) {
        return cast(-cast(a, dataType), dataType);
    }

    public static double inv(double a, DataType dataType) {
        return cast(1.0d / cast(a, dataType), dataType);
    }

    public static double log(double a, DataType dataType) {
        return cast(Math.log(cast(a, dataType)), dataType);
    }

    public static double exp(double a, DataType dataType) {
        return cast(Math.exp(cast(a, dataType)), dataType);
    }

    public static double fastExp(double a, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> FastTranscendentals.fastExpF64(a);
            case FLOAT32 -> FastTranscendentals.fastExpF32((float) a);
            case BFLOAT16 -> cast(FastTranscendentals.fastExpF32((float) a), dataType);
            case INT32, INT64, BOOL -> throw unsupported(dataType);
        };
    }

    public static double tanh(double a, DataType dataType) {
        return cast(Math.tanh(cast(a, dataType)), dataType);
    }

    public static double fastTanh(double a, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> FastTranscendentals.fastTanhF64(a);
            case FLOAT32 -> FastTranscendentals.fastTanhF32((float) a);
            case BFLOAT16 -> cast(FastTranscendentals.fastTanhF32((float) a), dataType);
            case INT32, INT64, BOOL -> throw unsupported(dataType);
        };
    }

    public static double sqrt(double a, DataType dataType) {
        return cast(Math.sqrt(cast(a, dataType)), dataType);
    }

    public static double abs(double a, DataType dataType) {
        return cast(Math.abs(cast(a, dataType)), dataType);
    }

    public static double pow(double a, double exponent, DataType dataType) {
        double x = cast(a, dataType);
        if (exponent == 0.0d) {
            return cast(1.0d, dataType);
        }
        if (exponent == 1.0d) {
            return x;
        }
        if (exponent == 2.0d) {
            return cast(x * x, dataType);
        }
        if (exponent == 0.5d) {
            return cast(Math.sqrt(x), dataType);
        }
        if (exponent == -1.0d) {
            return cast(1.0d / x, dataType);
        }
        return cast(Math.pow(x, exponent), dataType);
    }

    public static double mulScalar(double a, double scalar, DataType dataType) {
        return cast(cast(a, dataType) * cast(scalar, dataType), dataType);
    }

    public static double relu(double a, DataType dataType) {
        return cast(Math.max(cast(a, dataType), 0.0d), dataType);
    }

    public static double clampMin(double a, double minValue, DataType dataType) {
        return cast(Math.max(cast(a, dataType), cast(minValue, dataType)), dataType);
    }

    public static double clampMax(double a, double maxValue, DataType dataType) {
        return cast(Math.min(cast(a, dataType), cast(maxValue, dataType)), dataType);
    }

    public static double sigmoid(double a, DataType dataType) {
        double x = cast(a, dataType);
        return cast(1.0d / (1.0d + Math.exp(-x)), dataType);
    }

    public static double cast(double value, DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> value;
            case FLOAT32 -> (double) ((float) value);
            case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits((float) value));
            case INT32, INT64, BOOL -> throw unsupported(dataType);
        };
    }

    private static UnsupportedOperationException unsupported(DataType dataType) {
        return new UnsupportedOperationException(dataType + " is not a floating numeric oracle dtype.");
    }
}

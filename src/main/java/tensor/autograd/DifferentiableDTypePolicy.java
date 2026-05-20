package tensor.autograd;

import tensor.DataType;

/**
 * Central dtype contract for tensors that can participate in gradient computation.
 */
public final class DifferentiableDTypePolicy {
    private DifferentiableDTypePolicy() {
    }

    public static boolean supportsGradients(DataType dataType) {
        return switch (dataType) {
            case FLOAT64, FLOAT32, BFLOAT16 -> true;
            case INT32, INT64, BOOL -> false;
        };
    }

    public static void requireGradientSupported(DataType dataType, String context) {
        if (!supportsGradients(dataType)) {
            throw unsupportedGradientDType(dataType, context);
        }
    }

    public static UnsupportedOperationException unsupportedGradientDType(DataType dataType, String context) {
        String prefix = context == null || context.isBlank() ? "Gradients" : context;
        return new UnsupportedOperationException(prefix + " require floating tensors; got " + dataType + ".");
    }
}

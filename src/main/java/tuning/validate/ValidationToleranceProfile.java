package tuning.validate;

import tensor.DataType;

public enum ValidationToleranceProfile {
    FIXED,
    QUICK_DTYPE_AWARE,
    BALANCED_DTYPE_AWARE,
    THOROUGH_DTYPE_AWARE;

    public double absTolerance(DataType dataType, double fallback) {
        return switch (this) {
            case FIXED -> fallback;
            case QUICK_DTYPE_AWARE -> dtypeAwareAbs(dataType, 1e-8, 1e-5, 5e-3);
            case BALANCED_DTYPE_AWARE -> dtypeAwareAbs(dataType, 1e-9, 3e-6, 2e-3);
            case THOROUGH_DTYPE_AWARE -> dtypeAwareAbs(dataType, 1e-9, 5e-7, 1e-3);
        };
    }

    public double relTolerance(DataType dataType, double fallback) {
        return switch (this) {
            case FIXED -> fallback;
            case QUICK_DTYPE_AWARE -> dtypeAwareRel(dataType, 1e-8, 1e-5, 5e-3);
            case BALANCED_DTYPE_AWARE -> dtypeAwareRel(dataType, 1e-9, 3e-6, 2e-3);
            case THOROUGH_DTYPE_AWARE -> dtypeAwareRel(dataType, 1e-9, 5e-7, 1e-3);
        };
    }

    private static double dtypeAwareAbs(DataType dataType, double f64, double f32, double f16) {
        if (dataType == null) {
            return f64;
        }
        return switch (dataType) {
            case FLOAT64 -> f64;
            case FLOAT32 -> f32;
            case BFLOAT16 -> f16;
            case BOOL, INT32, INT64 -> 0.0d;
        };
    }

    private static double dtypeAwareRel(DataType dataType, double f64, double f32, double f16) {
        if (dataType == null) {
            return f64;
        }
        return switch (dataType) {
            case FLOAT64 -> f64;
            case FLOAT32 -> f32;
            case BFLOAT16 -> f16;
            case BOOL, INT32, INT64 -> 0.0d;
        };
    }
}

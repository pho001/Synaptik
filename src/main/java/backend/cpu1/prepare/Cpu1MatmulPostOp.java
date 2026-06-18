package backend.cpu1.prepare;

import backend.cpu1.provider.matmul.Cpu1MatmulRoute;

/**
 * Optional epilogue operation applied by cpu1 matmul kernels before publishing output.
 */
public enum Cpu1MatmulPostOp {
    NONE,
    RELU,
    ADD_BIAS,
    ADD_BIAS_RELU;

    public boolean supportedBy(Cpu1MatmulRoute route) {
        return switch (this) {
            case NONE -> true;
            case RELU -> route == Cpu1MatmulRoute.JAVA_SCALAR;
            case ADD_BIAS, ADD_BIAS_RELU -> route == Cpu1MatmulRoute.JAVA_SCALAR
                    || route == Cpu1MatmulRoute.OPENBLAS_NATIVE_SEGMENT;
        };
    }

    public boolean requiresBias() {
        return this == ADD_BIAS || this == ADD_BIAS_RELU;
    }

    public float apply(float value) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0f ? value : 0.0f;
            case ADD_BIAS -> throw new IllegalStateException("ADD_BIAS requires a bias value");
            case ADD_BIAS_RELU -> throw new IllegalStateException("ADD_BIAS_RELU requires a bias value");
        };
    }

    public double apply(double value) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0d ? value : 0.0d;
            case ADD_BIAS -> throw new IllegalStateException("ADD_BIAS requires a bias value");
            case ADD_BIAS_RELU -> throw new IllegalStateException("ADD_BIAS_RELU requires a bias value");
        };
    }

    public float apply(float value, float bias) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0f ? value : 0.0f;
            case ADD_BIAS -> value + bias;
            case ADD_BIAS_RELU -> {
                float biased = value + bias;
                yield biased > 0.0f ? biased : 0.0f;
            }
        };
    }

    public double apply(double value, double bias) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0d ? value : 0.0d;
            case ADD_BIAS -> value + bias;
            case ADD_BIAS_RELU -> {
                double biased = value + bias;
                yield biased > 0.0d ? biased : 0.0d;
            }
        };
    }
}

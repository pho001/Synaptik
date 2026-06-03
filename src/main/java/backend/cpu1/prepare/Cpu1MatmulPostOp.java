package backend.cpu1.prepare;

import backend.cpu1.provider.matmul.Cpu1MatmulRoute;

/**
 * Optional epilogue operation applied by Java cpu1 matmul kernels before storing each output element.
 */
public enum Cpu1MatmulPostOp {
    NONE,
    RELU,
    ADD_BIAS_RELU;

    public boolean supportedBy(Cpu1MatmulRoute route) {
        return this == NONE || route == Cpu1MatmulRoute.JAVA_SCALAR;
    }

    public boolean requiresBias() {
        return this == ADD_BIAS_RELU;
    }

    public float apply(float value) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0f ? value : 0.0f;
            case ADD_BIAS_RELU -> throw new IllegalStateException("ADD_BIAS_RELU requires a bias value");
        };
    }

    public double apply(double value) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0d ? value : 0.0d;
            case ADD_BIAS_RELU -> throw new IllegalStateException("ADD_BIAS_RELU requires a bias value");
        };
    }

    public float apply(float value, float bias) {
        return switch (this) {
            case NONE -> value;
            case RELU -> value > 0.0f ? value : 0.0f;
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
            case ADD_BIAS_RELU -> {
                double biased = value + bias;
                yield biased > 0.0d ? biased : 0.0d;
            }
        };
    }
}

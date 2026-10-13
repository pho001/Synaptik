package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.model.datatype.DataType;

/** Validated, dtype-narrowed coefficients shared by the two adaptive algorithms. */
record AdaptiveCoefficients(double learningRate, double beta1, double beta2,
        double epsilon, double weightDecay, boolean decoupled) {
    static void validate(double learningRate, double beta1, double beta2,
            double epsilon, double weightDecay) {
        if (!Double.isFinite(learningRate) || learningRate <= 0.0d
                || !Double.isFinite(epsilon) || epsilon <= 0.0d
                || !Double.isFinite(beta1) || beta1 < 0.0d || beta1 >= 1.0d
                || !Double.isFinite(beta2) || beta2 < 0.0d || beta2 >= 1.0d
                || !Double.isFinite(weightDecay) || weightDecay < 0.0d) {
            throw new IllegalArgumentException("invalid adaptive optimizer coefficients");
        }
    }

    static AdaptiveCoefficients forParameter(Optimizer optimizer, DataType dtype) {
        double lr;
        double b1;
        double b2;
        double eps;
        double decay;
        boolean decoupled;
        if (optimizer instanceof Adam adam) {
            lr = adam.learningRate(); b1 = adam.beta1(); b2 = adam.beta2();
            eps = adam.epsilon(); decay = adam.weightDecay(); decoupled = false;
        } else if (optimizer instanceof AdamW adamw) {
            lr = adamw.learningRate(); b1 = adamw.beta1(); b2 = adamw.beta2();
            eps = adamw.epsilon(); decay = adamw.weightDecay(); decoupled = true;
        } else {
            throw new IllegalArgumentException("not an adaptive optimizer: " + optimizer);
        }
        if (dtype != DataType.FLOAT64) {
            lr = (float) lr;
            b1 = (float) b1;
            b2 = (float) b2;
            eps = (float) eps;
            decay = (float) decay;
        }
        validate(lr, b1, b2, eps, decay);
        return new AdaptiveCoefficients(lr, b1, b2, eps, decay, decoupled);
    }

    static boolean equal(double left, double right) {
        return Double.doubleToLongBits(left) == Double.doubleToLongBits(right);
    }

    static int hash(double learningRate, double beta1, double beta2,
            double epsilon, double weightDecay) {
        int result = Double.hashCode(learningRate);
        result = 31 * result + Double.hashCode(beta1);
        result = 31 * result + Double.hashCode(beta2);
        result = 31 * result + Double.hashCode(epsilon);
        return 31 * result + Double.hashCode(weightDecay);
    }
}

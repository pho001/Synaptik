package training.optimizer;

import tensor.Tensor;

import java.util.Collection;

/**
 * Convenience factories for training optimizers.
 */
public final class Optimizers {
    private Optimizers() {
    }

    public static SgdOptimizer sgd(float learningRate) {
        return new SgdOptimizer(learningRate);
    }

    public static SgdOptimizer sgd(Collection<Tensor> parameters, float learningRate) {
        return new SgdOptimizer(parameters, learningRate);
    }

    public static AdamOptimizer adam(float learningRate) {
        return new AdamOptimizer(learningRate);
    }

    public static AdamOptimizer adam(Collection<Tensor> parameters, float learningRate) {
        return new AdamOptimizer(parameters, learningRate);
    }

    public static AdamOptimizer adam(
            Collection<Tensor> parameters,
            float learningRate,
            float beta1,
            float beta2,
            float epsilon
    ) {
        return new AdamOptimizer(parameters, learningRate, beta1, beta2, epsilon);
    }
}

package tensor.ops.loss;

import tensor.Tensor;
import tensor.loss.LossReduction;

final class LossReductionPolicy {
    private LossReductionPolicy() {
    }

    static Tensor apply(Tensor perSampleLoss, Tensor reductionWeights, LossReduction reduction) {
        return switch (reduction) {
            case NONE -> perSampleLoss;
            case SUM -> perSampleLoss.sum();
            case MEAN -> {
                if (reductionWeights == null) {
                    yield perSampleLoss.mean();
                }
                Tensor validCount = reductionWeights.sum();
                Tensor totalLoss = perSampleLoss.sum();
                yield totalLoss.div(validCount.clampMin(1.0));
            }
        };
    }
}

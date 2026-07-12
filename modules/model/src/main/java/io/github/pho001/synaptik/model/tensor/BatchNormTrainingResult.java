package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Public normalized output and explicit next running statistics of one training occurrence.
 *
 * <p>The result intentionally omits the producer's saved batch mean and inverse standard
 * deviation at slots three and four. All three exposed tensors retain the exact shared producer
 * at slots zero through two. The output has the input Shape; both next-statistic tensors share
 * the producer's rank-one channel Shape and promoted floating type. This shallowly immutable
 * record owns no statistic assignment,
 * cross-step state, training session, checkpoint, compiler lifetime, or execution behavior.</p>
 *
 * @param output non-null normalized affine output at producer slot zero
 * @param nextRunningMean non-null explicit next running mean at producer slot one
 * @param nextRunningVariance non-null explicit next running variance at producer slot two
 */
public record BatchNormTrainingResult(
        Tensor output, Tensor nextRunningMean, Tensor nextRunningVariance) {
    /**
     * Retains the three exact public output references.
     *
     * @param output non-null normalized affine output
     * @param nextRunningMean non-null explicit next running mean
     * @param nextRunningVariance non-null explicit next running variance
     * @throws NullPointerException if a component is null, checked in declaration order
     */
    public BatchNormTrainingResult {
        output = Objects.requireNonNull(output, "output");
        nextRunningMean = Objects.requireNonNull(nextRunningMean, "nextRunningMean");
        nextRunningVariance = Objects.requireNonNull(
                nextRunningVariance, "nextRunningVariance");
    }
}

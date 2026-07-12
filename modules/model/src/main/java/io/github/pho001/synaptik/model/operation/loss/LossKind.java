package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent loss semantics.
 *
 * <p>A loss occurrence combines ordered prediction and target inputs. This family currently owns
 * mean-squared error and dense-target categorical cross-entropy directly from logits. Executable
 * gradient, compiler, backend, runtime, and training behavior remain outside these model semantic
 * identities.</p>
 */
public enum LossKind implements OperationKind {
    /**
     * Computes squared differences between exact-shape prediction and target values and applies
     * the explicit reduction carried by {@link MeanSquaredErrorAttrs}.
     */
    MEAN_SQUARED_ERROR,

    /**
     * Computes target-weighted negative log-softmax from ordered logits and exact-shape dense
     * target inputs, then applies the explicit reduction and class axis carried by
     * {@link DenseCategoricalCrossEntropyWithLogitsAttrs}.
     *
     * <p>For non-class coordinate {@code g}, class coordinate {@code c}, logits {@code z}, and
     * dense target {@code t}, the stable meaning is
     * {@code m[g] = max_c(z[g,c])},
     * {@code lse[g] = m[g] + log(sum_c(exp(z[g,c] - m[g])))}, and
     * {@code loss[g] = sum_c(weightedContribution(t[g,c], lse[g] - z[g,c]))}.
     * {@code weightedContribution(0, q)} is exact positive zero; for positive {@code t} it is
     * {@code t * q}. Thus an absent class contributes positive zero even when its log probability
     * is negative infinity.</p>
     */
    DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;

    private static final List<OperationSignature> MEAN_SQUARED_ERROR_SIGNATURES =
            List.of(OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1));
    private static final List<OperationSignature> DENSE_CATEGORICAL_SIGNATURES = List.of(
            OperationSignature.fixed(
                    DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1));

    /**
     * Returns this loss kind's exact fixed two-input, one-output signature.
     *
     * @return the stable immutable singleton signature list accepting only this kind's exact
     *     attributes type
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case MEAN_SQUARED_ERROR -> MEAN_SQUARED_ERROR_SIGNATURES;
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    DENSE_CATEGORICAL_SIGNATURES;
        };
    }
}

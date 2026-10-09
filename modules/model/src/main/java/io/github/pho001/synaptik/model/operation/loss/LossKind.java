package io.github.pho001.synaptik.model.operation.loss;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent loss semantics.
 *
 * <p>A loss occurrence combines ordered prediction and target inputs. This family currently owns
 * mean-squared error plus dense-target and index-target categorical cross-entropy directly from
 * logits. Executable gradient, compiler, backend, runtime, and training behavior remain outside
 * these model semantic identities.</p>
 *
 * <p>Mean-squared error uses {@code (prediction - target)^2} at every exact logical position,
 * followed by its explicit reduction. The categorical kinds use the stable log-sum-exp formulas
 * documented below, with exact class axes, target membership, ignore-index guard, zero-target
 * guard, reduction membership, identities, and divisors. Their NaN, infinity, and signed-zero
 * classes follow those guards and named formula sites.</p>
 *
 * <p>MSE uses one subtraction and delta*delta per position, then its exact reduction and optional
 * divisor. Categorical losses retain exact maximum selection, target and ignore guards, score
 * subtraction, EXP, class sum, LOG, max addition, logit subtraction, declared target
 * multiplication, aggregation, negation, and optional divisor. Every contributor participates once;
 * only named floating arithmetic primitive inputs/results may use dtype-specific DAZ/FTZ for
 * FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64. Homogeneous low arithmetic works and accumulates
 * in FLOAT32 with one final low narrowing per declared output. Exact guards and special classes
 * remain; neither loss has a blanket output tolerance. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
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
    DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,

    /**
     * Computes selected-class negative log-softmax from ordered logits and class-index target
     * inputs, then applies the class axis, reduction, and optional exact ignore index carried by
     * {@link IndexCategoricalCrossEntropyWithLogitsAttrs}.
     *
     * <p>A target equal to the optional ignore index contributes positive zero before bounds or
     * logits evaluation. Every other target selects one class from its stable log-softmax slice;
     * mean reduction divides by the number of non-ignored targets. INT32 and INT64 targets are
     * exact indices, while floating computation and result type follow the logits: FLOAT16,
     * BFLOAT16, and FLOAT32 use at least FLOAT32 computation, and FLOAT64 uses FLOAT64.
     * NaN, infinity, empty, and all-ignored behavior is part of the semantic contract, not eager
     * evaluation.</p>
     */
    INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;

    private static final List<OperationSignature> MEAN_SQUARED_ERROR_SIGNATURES =
            List.of(OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1));
    private static final List<OperationSignature> DENSE_CATEGORICAL_SIGNATURES = List.of(
            OperationSignature.fixed(
                    DenseCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1));
    private static final List<OperationSignature> INDEX_CATEGORICAL_SIGNATURES = List.of(
            OperationSignature.fixed(
                    IndexCategoricalCrossEntropyWithLogitsAttrs.class, 2, 1));

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
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                    INDEX_CATEGORICAL_SIGNATURES;
        };
    }
}

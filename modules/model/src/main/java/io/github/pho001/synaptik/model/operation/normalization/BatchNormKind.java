package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent batch-normalization meanings with explicit inputs and outputs.
 *
 * <p>Each occurrence consumes ordered
 * {@code [input, scale, bias, runningMean, runningVariance]} tensors and produces exactly one
 * inference output or five training outputs. Their explicit channel axes are layout-neutral. The
 * training occurrence represents next statistics and saved batch values as ordinary producer
 * outputs; this kind owns no state across occurrences, gradient, compiler, backend, runtime, or
 * execution behavior.</p>
 *
 * <p>Inference computes
 * {@code ((input - runningMean) / sqrt(runningVariance + epsilon)) * scale + bias} per channel.
 * Training uses {@code mean = sum(input) / N}, population variance divided by {@code N}, saved
 * inverse standard deviation with epsilon inside the square root, the same affine formula, an
 * unbiased {@code N - 1} variance for the running update, and
 * {@code next = (1 - momentum) * old + momentum * batch}. Its exact domain is
 * {@code C == 0 || N >= 2}; an empty channel evaluates no values.</p>
 *
 * <p>Exact axes, memberships, count, epsilon, momentum, guards, output slots, saved statistics,
 * and running-state transition remain fixed. Inference uses subtraction, variance/epsilon addition,
 * SQRT, division, scale multiplication, and bias addition. Training also uses sum/count mean,
 * centered x*x sums with exact N and N-1 divisors, and fixed running-statistic multiply/add
 * transitions. Every contributor participates once; only named floating arithmetic primitive
 * inputs/results may use dtype-specific DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, never FLOAT64.
 * Homogeneous low arithmetic works and accumulates in FLOAT32, with one final low narrowing per
 * declared output; saved and state values are exact published results, not extra tolerance sites.
 * See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum BatchNormKind implements OperationKind {
    /** Requests explicit five-input per-channel batch-normalization inference. */
    BATCH_NORM_INFERENCE,

    /**
     * Requests five-input training normalization with explicit next and saved statistics.
     */
    BATCH_NORM_TRAINING;

    private static final List<OperationSignature> INFERENCE_SIGNATURES = List.of(
            OperationSignature.fixed(BatchNormInferenceAttrs.class, 5, 1));
    private static final List<OperationSignature> TRAINING_SIGNATURES = List.of(
            OperationSignature.fixed(BatchNormTrainingAttrs.class, 5, 5));

    /**
     * Returns the fixed occurrence signature for this batch-normalization meaning.
     *
     * @return immutable singleton signature accepting exactly five ordered inputs and either one
     *     inference output or five training outputs; never {@code null}
     */
    @Override
    public List<OperationSignature> signatures() {
        return switch (this) {
            case BATCH_NORM_INFERENCE -> INFERENCE_SIGNATURES;
            case BATCH_NORM_TRAINING -> TRAINING_SIGNATURES;
        };
    }
}

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
 * <p>Under the Model-owned numerical-profile contract, exact axes, memberships, {@code N},
 * epsilon, momentum, guards, output slots, saved statistics, and state transition remain
 * unchanged. For {@code ACCELERATOR FLOAT32}, aggregates include every contributor once,
 * arithmetic sites use DAZ/FTZ and one-round operations, and square root/reciprocal-root sites
 * use the five-ULP elementary ceiling. The composite gains no final-output envelope and saved
 * outputs are exact stored results; non-FLOAT32 behavior stays strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
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

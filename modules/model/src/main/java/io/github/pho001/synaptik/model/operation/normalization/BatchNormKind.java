package io.github.pho001.synaptik.model.operation.normalization;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, stateless batch-normalization inference.
 *
 * <p>The sole occurrence consumes ordered
 * {@code [input, scale, bias, runningMean, runningVariance]} tensors and produces exactly one
 * output. Its explicit channel axis is layout-neutral. The kind owns no training mode, statistic
 * update, saved value, gradient, compiler, backend, runtime, or execution behavior.</p>
 */
public enum BatchNormKind implements OperationKind {
    /** Requests explicit five-input per-channel batch-normalization inference. */
    BATCH_NORM_INFERENCE;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.fixed(BatchNormInferenceAttrs.class, 5, 1));

    /**
     * Returns the fixed inference occurrence signature.
     *
     * @return immutable singleton signature accepting exactly five ordered inputs and one output;
     *     never {@code null}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}

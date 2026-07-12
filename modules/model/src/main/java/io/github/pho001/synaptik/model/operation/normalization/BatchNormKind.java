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

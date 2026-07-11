package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Public output and next explicit RNG state of one training-dropout expression occurrence.
 *
 * <p>The result intentionally does not expose the producer's auxiliary keep-mask output. The
 * output and the private Tensor wrapped by {@code nextState} select producer slots zero and two
 * respectively, and both retain the same producer occurrence. This record is shallowly immutable,
 * retains both exact references, and uses record value equality over them.</p>
 *
 * @param output non-null dropped numerical Tensor expression; retained by exact reference
 * @param nextState non-null state occurrence positioned after this dropout's logical draws;
 *     retained by exact reference
 */
public record DropoutResult(Tensor output, GraphRngState nextState) {
    /**
     * Creates a dropout result retaining both exact references.
     *
     * @param output non-null dropped numerical Tensor expression
     * @param nextState non-null explicitly threaded next RNG state
     * @throws NullPointerException if {@code output} or {@code nextState} is null, checked in that
     *     order with the parameter name as the message
     */
    public DropoutResult {
        output = Objects.requireNonNull(output, "output");
        nextState = Objects.requireNonNull(nextState, "nextState");
    }
}

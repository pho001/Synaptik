package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * Output and explicit graph RNG state returned by one mode-selected {@link Dropout} forward call.
 *
 * <p>In training, the two components are the exact public output and next-state references from
 * one Model dropout occurrence. In evaluation, they are the exact caller-supplied input and state
 * references because no Model occurrence is created. This shallowly immutable record retains the
 * references without copying, mutation, inspection, sampling, or execution and uses ordinary
 * record value equality.</p>
 *
 * @param output non-null output selected by the forward branch; retained by exact reference
 * @param nextState non-null explicit graph RNG state selected by the forward branch; retained by
 *     exact reference
 */
public record DropoutForwardResult(Tensor output, GraphRngState nextState) {
    /**
     * Creates a forward result retaining both exact references.
     *
     * @param output non-null output Tensor selected by the forward branch
     * @param nextState non-null explicit graph RNG state selected by the forward branch
     * @throws NullPointerException if {@code output} or {@code nextState} is null, checked in that
     *     order with the component name as the message
     */
    public DropoutForwardResult {
        output = Objects.requireNonNull(output, "output");
        nextState = Objects.requireNonNull(nextState, "nextState");
    }
}

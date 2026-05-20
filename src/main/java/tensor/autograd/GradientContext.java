package tensor.autograd;

import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.Objects;

/**
 * Scoped API exposed to gradient rules while building a backward graph.
 */
public final class GradientContext {
    private final Tensor output;

    public GradientContext(Tensor output) {
        this.output = Objects.requireNonNull(output, "output cannot be null");
    }

    /**
     * Returns the forward output whose rule is being applied.
     *
     * @return output tensor
     */
    public Tensor output() {
        return output;
    }

    /**
     * Returns the upstream gradient currently accumulated for the output.
     *
     * @return gradient tensor, or null if no gradient reached the output
     */
    public Tensor upstreamGradient() {
        return output.getGradient();
    }

    /**
     * Accumulates a gradient delta for an input tensor.
     *
     * @param input input tensor receiving gradient
     * @param delta gradient contribution
     */
    public void accumulate(Tensor input, Tensor delta) {
        if (input == null || delta == null) {
            return;
        }
        Tensor current = input.getGradient();
        TensorInternalAccess.setGradient(input, current == null ? delta : current.add(delta));
    }
}

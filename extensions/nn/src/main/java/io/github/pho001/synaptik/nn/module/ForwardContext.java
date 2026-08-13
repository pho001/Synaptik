package io.github.pho001.synaptik.nn.module;

import java.util.Objects;

/**
 * Immutable snapshot of the local {@link ForwardMode} supplied to a concrete layer forward call.
 *
 * <p>A context records only NN composition mode. It is not a tensor, graph node, compile request,
 * runtime state, backend resource, or execution handle. A module creates a new snapshot when its
 * context is requested, so a previously obtained context remains unchanged after {@link Module#train()}
 * or {@link Module#eval()} changes the module's later local mode.</p>
 *
 * @param mode the non-null local mode captured by this immutable context
 */
public record ForwardContext(ForwardMode mode) {
    /**
     * Creates a context for one explicit forward mode.
     *
     * @param mode the non-null local mode snapshot
     * @throws NullPointerException if {@code mode} is {@code null}
     */
    public ForwardContext {
        Objects.requireNonNull(mode, "mode");
    }

    /**
     * Returns the mode captured by this context.
     *
     * @return the non-null immutable mode snapshot; it does not change when its originating
     *     module later changes mode
     */
    @Override
    public ForwardMode mode() {
        return mode;
    }
}

package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;
import java.util.Objects;

/**
 * An immutable ordered composition of unary Tensor modules.
 *
 * <p>Construction snapshots and permanently owns each exact child under its zero-based decimal
 * index. The inherited {@link #children()} snapshot is the public structural view, and inherited
 * recursive state operations therefore use stable paths such as {@code 0.weight} and
 * {@code 1.0.bias}. Caller mutation of the constructor list cannot alter membership or order.</p>
 *
 * <p>{@link #forward(Tensor)} passes the exact current Tensor reference through every child once
 * from left to right and returns the exact final result. An empty sequence returns its exact
 * input. If a child fails or returns null, completed prefix calls and expressions remain and no
 * suffix child is called; this class catches nothing and performs no rollback, prevalidation,
 * fusion, evaluation, compilation, or execution.</p>
 *
 * <p>Mode propagation, state traversal, and state-dictionary behavior are inherited from
 * {@link Module}. Construction and forward composition are not thread-safe or transactional with
 * mode changes or binding replacement; callers must coordinate concurrent access when one stable
 * view matters.</p>
 */
public final class Sequential extends UnaryTensorModule {
    private final List<UnaryTensorModule> modules;

    /**
     * Creates a sequence from one ordered list of exclusively owned unary modules.
     *
     * <p>The list may be empty. It is traversed once and retained only as an independent immutable
     * snapshot. Complete null, identity-duplicate, cycle, name, and existing-parent validation
     * precedes every ownership installation, so an ordinary construction failure leaves all
     * otherwise valid candidates available to another owner.</p>
     *
     * @param modules non-null ordered list whose non-null, identity-distinct exact elements will
     *     be permanently owned under names {@code 0}, {@code 1}, and so on; the caller retains no
     *     ability to mutate sequence structure through the list
     * @throws NullPointerException if {@code modules} or its first null element is null
     * @throws IllegalArgumentException if a module identity is repeated or a candidate would
     *     create a self or ancestor cycle
     * @throws IllegalStateException if a candidate is already owned by another module
     */
    public Sequential(List<? extends UnaryTensorModule> modules) {
        this.modules = registerIndexedChildren(modules);
    }

    /**
     * Passes one Tensor through every child exactly once in numeric order.
     *
     * <p>No whole-sequence compatibility check occurs. Each child receives the exact result of
     * its predecessor and applies its own contract at that point. A failure preserves completed
     * prefix work and suppresses every later call. This method neither reads nor synthesizes a
     * forward context and does not change module mode or state.</p>
     *
     * @param input non-null initial Tensor; returned exactly when this sequence is empty and
     *     otherwise supplied unchanged to child zero
     * @return the exact input for an empty sequence, or the non-null exact final child result
     * @throws NullPointerException if {@code input} is null or child {@code i} returns null; a
     *     null child-result message identifies that index
     * @throws RuntimeException if a child throws a runtime exception; the same exception is
     *     propagated after completed prefix effects
     */
    @Override
    public final Tensor forward(Tensor input) {
        Tensor current = Objects.requireNonNull(input, "input");
        for (int index = 0; index < modules.size(); index++) {
            current = Objects.requireNonNull(
                    modules.get(index).forward(current),
                    "modules[" + index + "] output");
        }
        return current;
    }
}

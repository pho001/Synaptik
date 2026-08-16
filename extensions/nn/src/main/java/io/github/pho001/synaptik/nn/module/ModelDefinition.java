package io.github.pho001.synaptik.nn.module;

/**
 * One-shot definition callback for a typed functional {@link Model}.
 *
 * <p>The callback registers every permanently owned child through the supplied short-lived
 * {@link Topology} and returns the exact forward function the model will retain. Registration is
 * collected without changing child ownership. After the callback finishes or fails, the topology
 * is sealed; successful definitions validate and install the complete child snapshot atomically.
 * The callback must treat the collector as invocation-scoped even if it can retain a Java
 * reference to it, because every later registration fails.</p>
 *
 * @param <I> non-null Java input type of the resulting model
 * @param <O> non-null Java output type of the resulting model
 */
@FunctionalInterface
public interface ModelDefinition<I, O> {
    /**
     * Declares named child ownership and returns the model's typed forward body.
     *
     * @param topology non-null definition-scoped collector; open only during this invocation and
     *     sealed before the factory returns or propagates a failure
     * @return non-null forward body to retain exactly after complete topology validation; a null
     *     result prevents every ownership installation
     */
    ModelForward<I, O> define(Topology topology);
}

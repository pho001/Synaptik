package io.github.pho001.synaptik.nn.module;

/**
 * One-shot definition callback for a typed {@link Model}.
 *
 * <p>The callback registers every permanently owned child through the supplied short-lived
 * {@link Topology} and returns a typed forward body. {@link Model#define(ModelDefinition)} retains
 * that exact body for later forward calls. {@link Model#defineTensor} instead invokes it once on
 * an internally created input prototype after selecting evaluation mode, then discards both this
 * definition callback and the forward body after acquiring the private source. Registration is
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
     * @return non-null forward body; Model.define retains it exactly after topology validation,
     *     while Model.defineTensor uses it only for one evaluation-mode prototype forward and
     *     discards it after private source acquisition; a null result prevents ownership installation
     */
    ModelForward<I, O> define(Topology topology);
}

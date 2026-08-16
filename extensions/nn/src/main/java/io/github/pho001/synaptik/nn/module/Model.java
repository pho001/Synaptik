package io.github.pho001.synaptik.nn.module;

import java.util.Map;
import java.util.Objects;

/**
 * Typed root module whose forward boundary may use arbitrary caller-selected Java types.
 *
 * <p>The generic input and output types describe Java composition only. A model may accept and
 * return Tensors directly or use caller-owned records for structured values; the functional
 * factory enforces non-null values at that boundary. These types add no Tensor semantics, tuple
 * type, graph representation, serialization schema, backward method, compiler, runtime,
 * optimizer, training session, or execution facade.</p>
 *
 * <p>{@link #define(ModelDefinition)} creates an immutable named child structure through a
 * short-lived sealed {@link Topology}. The resulting Model inherits recursive mode propagation,
 * parameter and buffer discovery, stable state-dictionary paths, strict state loading, and
 * compatible parameter replacement from {@link Module}. Structure is permanent after definition;
 * existing bindings and mode retain Module's caller-coordinated mutable lifecycle.</p>
 *
 * <p>Models are not thread-safe. Callers must coordinate forward construction with mode changes,
 * binding replacement, state loading, and other mutable Module operations when one consistent
 * view matters. Deferred parameter binding, input-dimension inference, tokenizer or batch
 * preparation, checkpoint persistence, graph capture, automatic differentiation, training,
 * compilation, and execution are outside this contract.</p>
 *
 * @param <I> non-null Java input type
 * @param <O> non-null Java output type
 */
public abstract class Model<I, O> extends Module {
    /**
     * Creates an ordinary empty training-mode Model base for an advanced subclass.
     *
     * <p>A subclass may use inherited protected declaration and child-registration operations.
     * Unlike the functional factory implementation, the abstract base cannot enforce the
     * documented non-null input and output contract around an arbitrary override.</p>
     */
    protected Model() {
        super();
    }

    /**
     * Applies this model's typed forward composition.
     *
     * @param input non-null caller input accepted by this model; ownership and mutation semantics
     *     are defined by the concrete model and input type
     * @return non-null caller-defined output; concrete implementations define exact identity and
     *     side effects
     */
    public abstract O forward(I input);

    /**
     * Defines one typed functional Model with an immutable named child topology.
     *
     * <p>The definition callback is invoked exactly once. Its topology collects modules without
     * ownership mutation and is sealed in a {@code finally} boundary. On success, a non-null
     * forward body and the complete encounter-ordered topology are retained only after Module has
     * preflighted every name, identity, cycle, and parent-ownership condition. The factory does
     * not inspect which local variables the body captures; an unused registered module is still
     * owned. Any callback, null-result, or validation failure leaves every previously unowned
     * candidate unattached and is propagated without publishing a partial Model.</p>
     *
     * <p>The functional implementation rejects null input before calling the retained body,
     * invokes that body exactly once, rejects a null result, and otherwise returns its exact
     * result reference. Body exceptions are propagated unchanged. Expressions and module-local
     * effects created by a failing or null-returning body prefix remain; forward performs no
     * rollback, lookup by module name, reflection, capture, compilation, or execution.</p>
     *
     * @param definition non-null one-shot definition callback
     * @param <I> inferred non-null Java input type of the resulting model
     * @param <O> inferred non-null Java output type of the resulting model
     * @return a non-null Model that permanently owns the declared modules and retains the exact
     *     forward body
     * @throws NullPointerException if {@code definition} or its returned forward body is null, or
     *     if the callback supplies a null topology name or module
     * @throws IllegalArgumentException if final topology validation rejects a name, repeated
     *     identity, or self/ancestor cycle
     * @throws IllegalStateException if final topology validation finds an already-owned module
     * @throws RuntimeException if the definition callback throws a runtime exception; the same
     *     exception is propagated after sealing and without attaching candidates
     */
    public static <I, O> Model<I, O> define(ModelDefinition<I, O> definition) {
        ModelDefinition<I, O> suppliedDefinition = Objects.requireNonNull(definition, "definition");
        Topology topology = new Topology();
        ModelForward<I, O> forward;
        try {
            forward = suppliedDefinition.define(topology);
        } finally {
            topology.seal();
        }
        ModelForward<I, O> suppliedForward = Objects.requireNonNull(forward, "model forward");
        return new FunctionalModel<>(suppliedForward, topology.snapshot());
    }

    private static final class FunctionalModel<I, O> extends Model<I, O> {
        private final ModelForward<I, O> forward;

        private FunctionalModel(ModelForward<I, O> forward, Map<String, Module> children) {
            this.forward = forward;
            registerNamedChildren(children);
        }

        @Override
        public O forward(I input) {
            I suppliedInput = Objects.requireNonNull(input, "input");
            return Objects.requireNonNull(forward.forward(suppliedInput), "model output");
        }
    }
}

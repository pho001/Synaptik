package io.github.pho001.synaptik.nn.module;

/**
 * Typed forward body supplied by a {@link ModelDefinition}.
 *
 * <p>{@link Model#define(ModelDefinition)} retains the exact body for subsequent forward calls.
 * {@link Model#defineTensor} invokes it once on an internally created prototype in evaluation mode,
 * then discards both the definition callback and this body after acquiring the private source.
 * Later owned-model symbolic forward calls reconstruct that source instead of invoking this body.</p>
 *
 * <p>The type parameters describe only the caller's Java composition boundary. This function may
 * construct ordinary Tensor expressions through registered modules or operate on caller-owned
 * structured values, but this contract adds no graph capture, differentiation, training,
 * compilation, execution, or rollback behavior. The body owns any additional mutation,
 * ownership, and thread-safety policy for its caller-defined values.</p>
 *
 * @param <I> non-null Java input type
 * @param <O> non-null Java output type
 */
@FunctionalInterface
public interface ModelForward<I, O> {
    /**
     * Applies the caller-defined forward body once.
     *
     * <p>Both Model factories supply a non-null input and require a non-null result. Model.define
     * invokes its retained body for each forward call; Model.defineTensor invokes the body only
     * for its one evaluation-mode prototype forward. If it throws or returns null, any prefix expressions or
     * module-local effects already created by the function remain; no rollback is performed and
     * the same exception is propagated.</p>
     *
     * @param input non-null caller input; ownership and mutation semantics are defined by the
     *     caller-owned input type and forward body
     * @return non-null caller-defined output; Model.define returns the exact result reference,
     *     while Model.defineTensor uses the prototype result to acquire a private source rather
     *     than retaining it as a future forward result
     */
    O forward(I input);
}

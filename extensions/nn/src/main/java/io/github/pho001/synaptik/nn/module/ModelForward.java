package io.github.pho001.synaptik.nn.module;

/**
 * Typed forward body retained by a functionally defined {@link Model}.
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
     * <p>A function retained by {@link Model#define(ModelDefinition)} receives a non-null input
     * and must return a non-null result. If it throws or returns null, any prefix expressions or
     * module-local effects already created by the function remain; no rollback is performed and
     * the same exception is propagated.</p>
     *
     * @param input non-null caller input; ownership and mutation semantics are defined by the
     *     caller-owned input type and forward body
     * @return non-null caller-defined output; exact identity is preserved by a functional Model
     */
    O forward(I input);
}

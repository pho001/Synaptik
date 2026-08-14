package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;

/**
 * A module whose complete forward contract accepts one Tensor and returns one Tensor.
 *
 * <p>This narrow base combines ordinary {@link Module} ownership with a type-safe unary forward
 * signature. It defines no common Shape, data-type, freshness, mode, state-transition, numerical,
 * compilation, or execution behavior. Each concrete module documents and validates its own
 * Tensor-expression contract.</p>
 */
public abstract class UnaryTensorModule extends Module {
    /**
     * Creates an empty training-mode module base with no declared state or children.
     */
    protected UnaryTensorModule() {
        super();
    }

    /**
     * Constructs this module's Tensor result from one input.
     *
     * @param input the non-null Tensor accepted by the concrete module's documented contract;
     *     never mutated by this contract
     * @return the concrete module's non-null Tensor result; identity, freshness, and expression
     *     semantics are defined by that module
     * @throws NullPointerException if {@code input} is {@code null}
     */
    public abstract Tensor forward(Tensor input);
}

package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * A module-owned, named trainable binding to one exact current {@link Tensor}.
 *
 * <p>This type marks its value as an eventual optimizer target without adding gradient, optimizer,
 * or mutable training state to {@code Tensor}. Its declaring {@link Module} can replace this
 * binding through its direct, protected replacement operation. The wrapper and its local
 * name remain stable; {@link #value()} returns the exact Tensor bound when it is called.</p>
 */
public final class Parameter {
    private final String name;
    private Tensor value;

    /**
     * Creates one module-declared parameter binding.
     *
     * <p>This constructor has package access so only the module declaration contract can create
     * a parameter. It retains both supplied references exactly and performs no Tensor copy or
     * evaluation.</p>
     *
     * @param name the non-null local declaration name
     * @param value the non-null Tensor reference to retain exactly
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     */
    Parameter(String name, Tensor value) {
        this.name = Objects.requireNonNull(name, "name");
        this.value = Objects.requireNonNull(value, "value");
    }

    /**
     * Returns this parameter's local module name.
     *
     * @return the non-null, declaration-time name; it is not a hierarchical path
     */
    public String name() {
        return name;
    }

    /**
     * Returns the exact tensor object currently bound to this parameter.
     *
     * <p>A direct or recursive module-discovery snapshot retains this wrapper object rather than
     * a historical tensor value. Consequently, reading this method through a wrapper obtained
     * from an earlier snapshot observes a later successful Module-owned replacement. A Tensor
     * returned before a replacement, and expressions constructed from that Tensor, remain
     * unchanged.</p>
     *
     * @return the non-null current tensor reference, without copying or evaluation
     */
    public Tensor value() {
        return value;
    }

    /**
     * Replaces this wrapper's current Tensor only through its declaring module's direct-state
     * contract.
     *
     * <p>This package-private method is not a general wrapper update API. {@link Module} invokes
     * it only after validating a non-null local name, a non-null value, and a direct parameter
     * target. It retains the exact supplied reference without validating descriptor, data type,
     * shape, layout, provenance, gradient eligibility, or storage compatibility because a module
     * declares no binding schema. It does not alter this wrapper's name or identity.</p>
     *
     * @param value the non-null exact Tensor reference to become the current binding
     * @throws NullPointerException if {@code value} is {@code null}
     */
    void replaceValue(Tensor value) {
        this.value = Objects.requireNonNull(value, "value");
    }
}

package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * A module-owned, named trainable binding to one exact current {@link Tensor}.
 *
 * <p>This type marks its value as an eventual optimizer target without adding gradient, optimizer,
 * or mutable training state to {@code Tensor}. The binding is established only during direct
 * {@link Module} declaration in this foundation: {@link #value()} always returns the exact object
 * supplied at declaration, and no replacement or update operation is exposed.</p>
 */
public final class Parameter {
    private final String name;
    private final Tensor value;

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
     * Returns the exact tensor object bound when this parameter was declared.
     *
     * @return the non-null declaration-time tensor reference, without copying, evaluation, or
     *     replacement
     */
    public Tensor value() {
        return value;
    }
}

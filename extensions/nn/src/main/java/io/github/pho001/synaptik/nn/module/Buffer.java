package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * A module-owned, named persistent binding to one exact current {@link Tensor} that is not trainable.
 *
 * <p>Buffers model layer state such as future running statistics while remaining outside optimizer
 * targets. The binding is established only during direct {@link Module} declaration in this
 * foundation: {@link #value()} always returns the exact object supplied at declaration, and no
 * replacement, update, checkpoint, or serialization operation is exposed.</p>
 */
public final class Buffer {
    private final String name;
    private final Tensor value;

    /**
     * Creates one module-declared buffer binding.
     *
     * <p>This constructor has package access so only the module declaration contract can create
     * a buffer. It retains both supplied references exactly and performs no Tensor copy or
     * evaluation.</p>
     *
     * @param name the non-null local declaration name
     * @param value the non-null Tensor reference to retain exactly
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     */
    Buffer(String name, Tensor value) {
        this.name = Objects.requireNonNull(name, "name");
        this.value = Objects.requireNonNull(value, "value");
    }

    /**
     * Returns this buffer's local module name.
     *
     * @return the non-null, declaration-time name; it is not a hierarchical path
     */
    public String name() {
        return name;
    }

    /**
     * Returns the exact tensor object bound when this buffer was declared.
     *
     * @return the non-null declaration-time tensor reference, without copying, evaluation, or
     *     replacement
     */
    public Tensor value() {
        return value;
    }
}

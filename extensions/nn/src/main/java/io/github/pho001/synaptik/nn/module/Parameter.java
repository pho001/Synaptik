package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * A module-owned, named trainable binding to one exact current {@link Tensor}.
 *
 * <p>This type marks its value as an eventual optimizer target without adding gradient, optimizer,
 * or mutable training state to {@code Tensor}. A generic downstream consumer can discover the
 * exact wrapper through a {@link Module} tree and install a schema-compatible Tensor through
 * {@link #replace(Tensor)} without knowing the concrete module or layer type. The wrapper, local
 * name, declaration-time data type, and declaration-time Shape remain stable; {@link #value()}
 * returns the exact Tensor bound when it is called.</p>
 *
 * <p>A parameter is mutable and is not thread-safe. Callers must coordinate replacement with
 * forward-expression construction and any other operation that requires a consistent binding
 * view. Replacement is individual: this type provides no version, transaction, rollback,
 * checkpoint, multi-parameter consistency, optimizer algorithm, or execution behavior.</p>
 */
public final class Parameter {
    private final String name;
    private final DataType dataType;
    private final Shape shape;
    private Tensor value;

    /**
     * Creates one module-declared parameter binding.
     *
     * <p>This constructor has package access so only the module declaration contract can create
     * a parameter. A parameter declaration requires a floating Tensor with gradient eligibility.
     * It retains the exact Tensor reference and privately captures that Tensor's exact data type
     * and immutable Shape as the permanent replacement schema. It performs no Tensor copy or
     * evaluation.</p>
     *
     * @param name the non-null local declaration name
     * @param value the non-null floating, gradient-eligible Tensor reference to retain exactly
     * @throws NullPointerException if {@code name} or {@code value} is {@code null}
     * @throws IllegalArgumentException if {@code value} is not floating or does not have
     *     {@code requiresGrad == true}, checked in that order
     */
    Parameter(String name, Tensor value) {
        this.name = Objects.requireNonNull(name, "name");
        Tensor declaredValue = Objects.requireNonNull(value, "value");
        DataType declaredDataType = declaredValue.descriptor().dataType();
        if (!declaredDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "parameter value must have a floating data type: " + declaredDataType);
        }
        if (!declaredValue.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "parameter value must have requiresGrad == true");
        }
        this.dataType = declaredDataType;
        this.shape = declaredValue.descriptor().shape();
        this.value = declaredValue;
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
     * from an earlier snapshot observes a later successful replacement. A Tensor
     * returned before a replacement, and expressions constructed from that Tensor, remain
     * unchanged.</p>
     *
     * @return the non-null current tensor reference, without copying or evaluation
     */
    public Tensor value() {
        return value;
    }

    /**
     * Replaces this parameter's current Tensor with one schema-compatible exact reference.
     *
     * <p>Validation rejects a null value first, then requires its exact data type and structural
     * Shape to equal the declaration-time schema, and finally requires
     * {@code requiresGrad == true}. A failure leaves the exact previous binding current. A
     * successful call retains the supplied Tensor without copying or evaluation and does not
     * alter this wrapper's name or identity.</p>
     *
     * <p>Tensor identity, layout, host storage, provenance, and label are deliberately not part of
     * compatibility and may differ from the declaration or previous binding. Tensors returned by
     * earlier {@link #value()} calls and expressions already built from them remain unchanged.
     * This individual mutable operation is not thread-safe and provides no version, transaction,
     * rollback, checkpoint, optimizer algorithm, update sequencing, or cross-parameter
     * consistency contract.</p>
     *
     * @param value the non-null exact Tensor reference whose data type and structural Shape equal
     *     the declaration-time schema and whose descriptor has {@code requiresGrad == true}
     * @throws NullPointerException if {@code value} is {@code null}
     * @throws IllegalArgumentException if the data type differs, the Shape differs structurally,
     *     or gradient eligibility is false, checked in that order
     */
    public void replace(Tensor value) {
        Tensor replacement = Objects.requireNonNull(value, "value");
        if (replacement.descriptor().dataType() != dataType) {
            throw new IllegalArgumentException(
                    "replacement data type must equal declaration data type: expected="
                            + dataType
                            + ", actual="
                            + replacement.descriptor().dataType());
        }
        if (!replacement.descriptor().shape().equals(shape)) {
            throw new IllegalArgumentException(
                    "replacement shape must equal declaration shape: expected="
                            + shape
                            + ", actual="
                            + replacement.descriptor().shape());
        }
        if (!replacement.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "replacement value must have requiresGrad == true");
        }
        this.value = replacement;
    }
}

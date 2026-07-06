package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the backend-independent meaning of one-input constant padding.
 *
 * <p>The kind composes with {@link PadAttrs} as an exact typed pair:</p>
 *
 * <pre>{@code
 * PadAttrs attrs = new PadAttrs(before, after, constantValue);
 * Operation padded = new Operation(PadKind.PAD, attrs);
 * }</pre>
 *
 * <p>For every input axis, the operation requests constant-filled logical positions before and
 * after the complete input extent without changing rank. For example, input {@code [10, 20]},
 * before width {@code 1}, after width {@code 2}, and constant {@code -1} semantically request
 * {@code [-1, 10, 20, -1, -1]}. This example states logical meaning only; this type does not
 * construct a Tensor or calculate values.</p>
 *
 * <p>The generic {@code Operation} descriptor does not enforce the family-specific pairing.
 * This enum defines no input-rank validation, result Shape or DataType, layout, storage,
 * materialization, provenance, gradient, compiler, backend, ONNX, or execution behavior. Its
 * inherited enum name is diagnostic text rather than a serialization or dispatch identifier.</p>
 */
public enum PadKind implements OperationKind {
    /**
     * Adds the constant-filled before and after widths described by {@link PadAttrs} around every
     * input axis.
     *
     * <p>The kind describes semantic intent only. It does not interpret the constant for an input
     * DataType, derive output extents, allocate storage, or execute padding.</p>
     */
    PAD
}

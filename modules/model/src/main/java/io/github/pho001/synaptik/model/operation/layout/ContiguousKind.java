package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the backend-independent, parameterless request for contiguous result geometry.
 *
 * <p>This vocabulary describes one logical input and does not store or validate arity. Its sole
 * kind composes explicitly with the generic operation descriptor and canonical no-attributes
 * value:</p>
 *
 * <pre>{@code
 * Operation operation = new Operation(
 *         ContiguousKind.CONTIGUOUS,
 *         NoOperationAttrs.INSTANCE);
 * }</pre>
 *
 * <p>The generic {@code Operation} contract checks only that its kind and attributes are non-null;
 * it does not validate this family-specific pairing. This enum stores no input, result descriptor,
 * layout state, materialization state, or other operation metadata.</p>
 *
 * <p>A contiguous request is computation semantics, whereas
 * {@code io.github.pho001.synaptik.model.layout.LayoutKind.DENSE_CONTIGUOUS} classifies already
 * resolved layout geometry. This type has no dependency on that geometric classification and
 * does not determine whether an existing representation can be reused or whether allocation,
 * copying, lowering, or execution is required. Compiler elimination, planning materialization,
 * backend preparation, runtime execution, gradient behavior, and backend availability remain
 * outside this semantic type.</p>
 *
 * <p>Inherited enum names provide stable diagnostic text only. They are not serialization tokens,
 * registry keys, dispatch keys, kernel names, or reflection identifiers.</p>
 */
public enum ContiguousKind implements OperationKind {
    /**
     * Requests logically equivalent canonical dense row-major result geometry with logical
     * storage offset zero.
     *
     * <p>The request preserves the one input's logical values, Shape, DataType, and row-major
     * element order. It describes the desired logical geometry without constructing a layout
     * descriptor, inspecting input geometry, enforcing one-input arity, or deciding whether the
     * eventual result aliases existing storage or requires a copy.</p>
     */
    CONTIGUOUS
}

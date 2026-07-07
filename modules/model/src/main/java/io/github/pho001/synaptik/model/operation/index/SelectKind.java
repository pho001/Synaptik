package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the backend-independent meaning of selecting one scalar coordinate on one tensor
 * axis.
 *
 * <p>{@link #SELECT} fixes one coordinate on an existing source axis and removes that axis from
 * the logical result. For a conceptual source shape {@code [2, 3, 4]}, selecting normalized axis
 * {@code 1} at normalized index {@code 2} therefore has conceptual result shape {@code [2, 4]}.
 * The kind pairs explicitly with {@link SelectAttrs}; the generic operation descriptor does not
 * enforce that family-specific pairing, one-input context, rank, bounds, or result shape.</p>
 *
 * <p>Scalar selection is distinct from elementwise conditional {@code WHERE}, which chooses
 * between branch values at corresponding positions; individually indexed {@code UNSTACK}, which
 * identifies one result of a logical multi-result request; and general {@code SLICE}, which
 * selects half-open coordinate intervals without removing an axis. It is also distinct from
 * gather operations whose indices are tensors rather than one intrinsic scalar coordinate.</p>
 *
 * <p>This enum stores no input, axis, index, shape, layout, result, provenance, graph state,
 * gradient, compiler policy, backend support, or execution state. Its inherited enum name is
 * diagnostic text rather than a serialization, parsing, registry, dispatch, reflection, route,
 * or kernel identifier.</p>
 */
public enum SelectKind implements OperationKind {
    /**
     * Fixes the normalized scalar coordinate in {@link SelectAttrs} on its normalized source axis
     * and conceptually removes that axis from the logical result.
     *
     * <p>The kind does not normalize caller input, inspect an input rank or axis extent, validate
     * bounds, construct a result shape or layout, select values, or execute work.</p>
     */
    SELECT
}

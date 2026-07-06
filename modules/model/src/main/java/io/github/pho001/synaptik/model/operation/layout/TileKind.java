package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;

/**
 * Identifies the backend-independent meaning of complete-pattern per-axis tiling.
 *
 * <p>The kind composes with {@link TileAttrs} as an exact typed pair:</p>
 *
 * <pre>{@code
 * TileAttrs attrs = new TileAttrs(repeats);
 * Operation tiled = new Operation(TileKind.TILE, attrs);
 * }</pre>
 *
 * <p>For conceptual input {@code [[1, 2], [3, 4]]} and repeats {@code [2, 3]}, axis zero
 * repeats the complete two-row input pattern twice and axis one repeats each complete row pattern
 * three times. The requested logical result is therefore
 * {@code [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2],
 * [3, 4, 3, 4, 3, 4]]}. This is complete-pattern tiling, not repetition of each scalar into a
 * consecutive run. The example states semantic meaning only and does not claim value execution.</p>
 *
 * <p>The generic {@code Operation} descriptor does not enforce the family-specific pairing.
 * This enum defines no input-rank validation, result Shape or DataType, layout, storage,
 * materialization, provenance, gradient, compiler, backend, ONNX, or execution behavior. Its
 * inherited enum name is diagnostic text rather than a serialization or dispatch identifier.</p>
 */
public enum TileKind implements OperationKind {
    /**
     * Repeats the complete input pattern by the positive per-axis counts in {@link TileAttrs}.
     *
     * <p>The kind describes logical meaning only. It does not multiply result extents, allocate
     * storage, repeat scalar values independently, or execute tiling.</p>
     */
    TILE
}

package io.github.pho001.synaptik.model;

/**
 * Classifies the resolved logical element geometry of a tensor layout.
 *
 * <p>The classification describes only the relationship between a static shape, element strides,
 * and an element offset. It does not prescribe storage ownership, materialization, backend support,
 * or an executable kernel route. View or alias status is recorded independently by
 * {@link LayoutDescriptor#isView()}.</p>
 */
public enum LayoutKind {
    /**
     * Canonical row-major element strides with storage offset zero.
     *
     * <p>This kind describes dense logical geometry and does not by itself prove that the tensor
     * owns its storage.</p>
     */
    DENSE_CONTIGUOUS,

    /**
     * Canonical row-major element strides with a non-zero storage offset.
     *
     * <p>The offset is measured in elements, not bytes. A descriptor of this kind may be marked as
     * either a view or a non-view because alias metadata is independent of geometry.</p>
     */
    DENSE_WITH_OFFSET,

    /**
     * Resolved non-canonical element strides that do not repeat a non-singleton dimension through
     * stride zero.
     *
     * <p>This kind includes ordinary positive-stride views and non-canonical zero strides on
     * singleton or empty dimensions.</p>
     */
    STRIDED,

    /**
     * A layout that repeats at least one static dimension larger than one through element stride
     * zero.
     *
     * <p>A descriptor with this kind must be marked as a view because repeated logical elements
     * alias the same storage element.</p>
     */
    BROADCAST_ZERO_STRIDE
}

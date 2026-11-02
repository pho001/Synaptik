package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.List;

/**
 * Immutable numeric code-shaping facts for controlled ordinary Scan helpers.
 *
 * <p>These facts contain no semantic graph, runtime segment, or derivative formula. In the
 * invocation-owned control segment, byte zero records complete-occurrence presence,
 * the long at byte eight holds canonical B,
 * and bytes {@code 16 + row} record original-row presence. Complete occurrence is independent
 * of contributor count, including empty tensors.
 * Row geometry is proved from the original local value, never inferred from equal extents.</p>
 *
 * @param boundaryRows exact ordinary ABI boundary order, including write boundaries
 * @param valueRows exact ordinary numerical value order when represented by a value inventory
 */
public record CpuScanControl(List<Row> boundaryRows, List<Row> valueRows) {
    /**
     * Snapshots generation-time row provenance.
     *
     * @param boundaryRows non-null ordered boundary facts
     * @param valueRows non-null ordered represented-value facts
     * @throws NullPointerException if either list or an element is null
     */
    public CpuScanControl {
        boundaryRows = List.copyOf(boundaryRows);
        valueRows = List.copyOf(valueRows);
    }

    /**
     * Proved mapping from a value's logical linear ordinal to its original row.
     *
     * <p>For ROW values, {@code row = ordinal / logicalRowStride % batch}. The fixed suffix
     * product may be zero for an empty feature domain; no element access is then permitted.
     * A SHARED value has axis minus one and stride zero and has occurrence presence, not an
     * invented row coordinate.</p>
     *
     * @param axis original row axis, or minus one for SHARED
     * @param logicalRowStride exact fixed product of dimensions following the row axis
     */
    public record Row(int axis, long logicalRowStride) {
        /**
         * Checks numeric provenance geometry without inspecting invocation values.
         *
         * @param axis row axis or minus one
         * @param logicalRowStride non-negative fixed suffix product; zero for SHARED
         * @throws IllegalArgumentException if geometry is invalid
         */
        public Row {
            if (axis < -1 || logicalRowStride < 0 || axis < 0 && logicalRowStride != 0) {
                throw new IllegalArgumentException("invalid Scan row provenance");
            }
        }

        /**
         * Returns occurrence-only provenance without a fabricated batch axis.
         *
         * @return immutable SHARED row fact
         */
        public static Row shared() {
            return new Row(-1, 0);
        }

        /**
         * Reports whether this value has an explicitly proved original row coordinate.
         *
         * @return true exactly for ROW values
         */
        public boolean rowIndexed() {
            return axis >= 0;
        }
    }
}

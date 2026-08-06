package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.List;
import java.util.Objects;

/**
 * Describes one route-independent input access form and its cold-bound instance geometry.
 * This unsupported internal contract currently admits only zero-offset canonical-dense access.
 *
 * @param regime non-null structural access regime
 */
public record CpuAccessPlan(Regime regime) {
    /** Structural access regimes implemented by the current CPU slice. */
    public enum Regime {
        /** Zero-offset dense row-major access with one linear element index. */
        CANONICAL_DENSE
    }

    /**
     * Creates one structural access plan.
     *
     * @throws NullPointerException if {@code regime} is {@code null}
     */
    public CpuAccessPlan { Objects.requireNonNull(regime, "regime"); }

    /**
     * Cold-bound geometry for one compatible invocation.
     *
     * @param extents immutable ordered non-negative extents
     * @param elementCount checked product of {@code extents}; scalar rank zero uses one
     * @param start inclusive primitive loop bound
     * @param end exclusive primitive loop bound
     */
    public record Binding(List<Long> extents, long elementCount, long start, long end) {
        /**
         * Validates exact static geometry and half-open bounds and snapshots the extent list.
         *
         * @throws NullPointerException if the list or an extent is {@code null}
         * @throws IllegalArgumentException if an extent is negative, the element count differs
         *     from the checked extent product, or the bounds are outside the element range
         * @throws ArithmeticException if the non-zero extent product overflows {@code long}
         */
        public Binding {
            Objects.requireNonNull(extents, "extents");
            var copy = new java.util.ArrayList<Long>(extents.size());
            boolean zero = false;
            for (int i = 0; i < extents.size(); i++) {
                long extent = Objects.requireNonNull(extents.get(i), "extents[" + i + "]");
                if (extent < 0) throw new IllegalArgumentException("extent must be non-negative");
                zero |= extent == 0;
                copy.add(extent);
            }
            extents = List.copyOf(copy);
            long product = zero ? 0 : 1;
            if (!zero) for (long extent : extents) product = Math.multiplyExact(product, extent);
            if (elementCount != product) throw new IllegalArgumentException(
                    "elementCount must equal the checked extent product");
            if (start < 0 || end < start || end > elementCount) throw new IllegalArgumentException(
                    "bounds must satisfy 0 <= start <= end <= elementCount");
        }
    }
}

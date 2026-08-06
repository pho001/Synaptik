package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Route-independent normalized access geometry for one value in a whole-partition CPU unit.
 * Structural axis roles and regime belong to generated identity; concrete geometry is held by
 * {@link Binding} and supplied only during cold invocation binding.
 *
 * @param accessKind whether the value is read or written
 * @param regime ordered scalar realization regime
 * @param iterationRank rank of the unit iteration space
 * @param axisRoles one normalized role per iteration axis
 * @param contiguousSuffix number of trailing contiguous axes
 */
public record CpuAccessPlan(AccessKind accessKind, Regime regime, int iterationRank,
        List<AxisRole> axisRoles, int contiguousSuffix) {
    /** Direction of access. */
    public enum AccessKind { /** Read-only input. */ READ, /** Write-only output. */ WRITE }
    /** Normalized role of one right-aligned iteration axis. */
    public enum AxisRole { /** Effective stride zero. */ BROADCAST,
        /** Member of the contiguous trailing suffix. */ CONTIGUOUS,
        /** Other positive effective stride. */ STRIDED }
    /** Ordered scalar access regimes, from most specialized to the complete fallback. */
    public enum Regime {
        /** Canonical linear addressing with one address increment. */ DENSE_LINEAR,
        /** Read-only scalar or all-zero-stride addressing with a constant address. */ SCALAR_ALL_ZERO,
        /** Broadcast leading axes plus one contiguous final-axis counter and reset. */ LAST_AXIS_BIAS,
        /** Contiguous trailing block plus outer carry and reset. */ BLOCK_OUTER,
        /** Complete primitive full-axis odometer for all remaining admitted geometry. */ GENERAL_ODOMETER
    }

    /**
     * Validates and snapshots one structural plan.
     *
     * @throws NullPointerException if a reference component or axis role is {@code null}
     * @throws IllegalArgumentException if rank, role count, or suffix length is invalid
     */
    public CpuAccessPlan {
        Objects.requireNonNull(accessKind, "accessKind");
        Objects.requireNonNull(regime, "regime");
        axisRoles = List.copyOf(axisRoles);
        if (iterationRank < 0 || axisRoles.size() != iterationRank
                || contiguousSuffix < 0 || contiguousSuffix > iterationRank) {
            throw new IllegalArgumentException("access-plan rank or suffix is invalid");
        }
    }

    /**
     * Immutable instance geometry for one value and exact half-open invocation range.
     *
     * @param plan structural plan this binding realizes
     * @param extents unit iteration extents
     * @param baseElementOffset resolved layout origin in elements
     * @param effectiveStrides right-aligned element strides, including broadcast zero strides
     * @param elementCount checked product of extents
     * @param start inclusive logical bound
     * @param end exclusive logical bound
     * @param referencedElementSpan exact selected storage span in elements
     * @param startCoordinates coordinates of {@code start}, or all zero for an empty range
     * @param startAddress physical element address corresponding to {@code start}
     * @param accessedElementStart inclusive lower address of the exact accessed range
     * @param accessedElementEnd exclusive upper address of the exact accessed range
     */
    public record Binding(CpuAccessPlan plan, List<Long> extents, long baseElementOffset,
            List<Long> effectiveStrides, long elementCount, long start, long end,
            long referencedElementSpan, List<Long> startCoordinates, long startAddress,
            long accessedElementStart, long accessedElementEnd) {
        /** Validates checked non-negative geometry and snapshots all lists. */
        public Binding {
            Objects.requireNonNull(plan, "plan");
            extents = checked(extents, "extents");
            effectiveStrides = checked(effectiveStrides, "effectiveStrides");
            startCoordinates = checked(startCoordinates, "startCoordinates");
            if (extents.size() != plan.iterationRank()
                    || effectiveStrides.size() != plan.iterationRank()
                    || startCoordinates.size() != plan.iterationRank()) {
                throw new IllegalArgumentException("binding rank differs from structural plan");
            }
            long product = extents.stream().anyMatch(value -> value == 0) ? 0 : 1;
            if (product != 0) for (long extent : extents) product = Math.multiplyExact(product, extent);
            if (elementCount != product || start < 0 || end < start || end > elementCount
                    || baseElementOffset < 0 || referencedElementSpan < 0 || startAddress < 0) {
                throw new IllegalArgumentException("binding geometry is invalid");
            }
            if (accessedElementStart < 0 || accessedElementEnd < accessedElementStart
                    || accessedElementEnd > referencedElementSpan) {
                throw new IllegalArgumentException("accessed element span is invalid");
            }
            for (int axis = 0; axis < extents.size(); axis++) {
                if (startCoordinates.get(axis) >= Math.max(1, extents.get(axis))) {
                    throw new IllegalArgumentException("starting coordinate exceeds its extent");
                }
            }
            if (elementCount != 0 && end > start && startAddress >= referencedElementSpan) {
                throw new IllegalArgumentException("starting address exceeds referenced span");
            }
        }

        /**
         * Creates an exact-range binding and computes its primitive initial odometer state and
         * conservative exact accessed address interval on the cold path.
         *
         * @param plan non-null structural access plan
         * @param extents non-null non-negative iteration extents; copied
         * @param baseElementOffset non-negative layout storage origin in elements
         * @param effectiveStrides non-null non-negative right-aligned element strides; copied
         * @param elementCount exact checked product of {@code extents}
         * @param start non-negative inclusive logical bound
         * @param end exclusive logical bound no greater than {@code elementCount}
         * @param referencedElementSpan non-negative selected storage span in elements
         * @return a new immutable binding with primitive start state and half-open accessed span
         * @throws NullPointerException if an array or {@code plan} is {@code null}
         * @throws IllegalArgumentException if rank, count, range, or span geometry is invalid
         * @throws ArithmeticException if exact geometry arithmetic overflows {@code long}
         */
        public static Binding create(CpuAccessPlan plan, long[] extents, long baseElementOffset,
                long[] effectiveStrides, long elementCount, long start, long end,
                long referencedElementSpan) {
            Objects.requireNonNull(extents, "extents");
            Objects.requireNonNull(effectiveStrides, "effectiveStrides");
            long[] coordinates = new long[extents.length];
            long remainder = start;
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                long extent = extents[axis];
                if (extent != 0) { coordinates[axis] = remainder % extent; remainder /= extent; }
            }
            long address = baseElementOffset;
            if (start < end) for (int axis = 0; axis < extents.length; axis++) {
                address = Math.addExact(address,
                        Math.multiplyExact(coordinates[axis], effectiveStrides[axis]));
            }
            Span span = start == end ? new Span(baseElementOffset, baseElementOffset)
                    : span(0, start, end, extents, effectiveStrides);
            long accessedStart = start == end ? baseElementOffset
                    : Math.addExact(baseElementOffset, span.minimum());
            long accessedEnd = start == end ? baseElementOffset
                    : Math.addExact(baseElementOffset, Math.addExact(span.maximum(), 1));
            return new Binding(plan, boxed(extents), baseElementOffset, boxed(effectiveStrides),
                    elementCount, start, end, referencedElementSpan, boxed(coordinates), address,
                    accessedStart, accessedEnd);
        }

        private static Span span(int axis, long start, long end, long[] extents, long[] strides) {
            if (axis == extents.length) return new Span(0, 0);
            long block = 1;
            for (int inner = axis + 1; inner < extents.length; inner++) {
                block = Math.multiplyExact(block, extents[inner]);
            }
            long first = start / block;
            long last = (end - 1) / block;
            if (first == last) {
                Span suffix = span(axis + 1, start % block, (end - 1) % block + 1,
                        extents, strides);
                long contribution = Math.multiplyExact(first, strides[axis]);
                return suffix.shift(contribution);
            }
            Span firstPart = span(axis + 1, start % block, block, extents, strides)
                    .shift(Math.multiplyExact(first, strides[axis]));
            Span lastPart = span(axis + 1, 0, (end - 1) % block + 1, extents, strides)
                    .shift(Math.multiplyExact(last, strides[axis]));
            long minimum = Math.min(firstPart.minimum(), lastPart.minimum());
            long maximum = Math.max(firstPart.maximum(), lastPart.maximum());
            if (last - first > 1) {
                Span fullSuffix = fullSpan(axis + 1, extents, strides);
                minimum = Math.min(minimum, fullSuffix.shift(
                        Math.multiplyExact(first + 1, strides[axis])).minimum());
                maximum = Math.max(maximum, fullSuffix.shift(
                        Math.multiplyExact(last - 1, strides[axis])).maximum());
            }
            return new Span(minimum, maximum);
        }

        private static Span fullSpan(int axis, long[] extents, long[] strides) {
            long maximum = 0;
            for (int current = axis; current < extents.length; current++) maximum = Math.addExact(
                    maximum, Math.multiplyExact(extents[current] - 1, strides[current]));
            return new Span(0, maximum);
        }

        private record Span(long minimum, long maximum) {
            Span shift(long amount) { return new Span(Math.addExact(minimum, amount),
                    Math.addExact(maximum, amount)); }
        }

        private static List<Long> checked(List<Long> values, String name) {
            Objects.requireNonNull(values, name);
            var copy = new ArrayList<Long>(values.size());
            for (int i = 0; i < values.size(); i++) {
                long value = Objects.requireNonNull(values.get(i), name + "[" + i + "]");
                if (value < 0) throw new IllegalArgumentException(name + " must be non-negative");
                copy.add(value);
            }
            return List.copyOf(copy);
        }

        private static List<Long> boxed(long[] values) {
            var result = new ArrayList<Long>(values.length);
            for (long value : values) result.add(value);
            return List.copyOf(result);
        }
    }
}

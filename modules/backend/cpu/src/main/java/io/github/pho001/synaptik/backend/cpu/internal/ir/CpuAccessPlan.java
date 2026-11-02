package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Route-independent normalized access geometry for one value in a whole-partition CPU unit.
 * Structural axis roles and regime belong to generated identity; concrete geometry is held by
 * {@link Binding} and supplied only during cold invocation binding.
 * A selected materialization retains the original source binding for its copy and supplies a
 * separate canonical dense, zero-offset read binding to the generated consumer.
 *
 * @param accessKind whether the value is read or written
 * @param regime ordered scalar realization regime
 * @param iterationRank rank of the unit iteration space
 * @param axisRoles one normalized role per iteration axis
 * @param contiguousSuffix number of trailing contiguous axes
 */
public record CpuAccessPlan(AccessKind accessKind, Regime regime, int iterationRank,
        List<AxisRole> axisRoles, int contiguousSuffix) {
    /**
     * Prepared numeric access coefficients in elements, with one B-bearing extent at most.
     * Nonoverlapping writes are proved structurally without enumerating invocation elements.
     * Reads retain transpose/expansion mappings. Explicit repeated writes require the affine
     * copy owner to check equal-source repeated destinations at numeric binding.
     * @param plan non-null immutable structural access plan
     * @param extentConstants non-null non-negative extent constants, copied
     * @param extentBatchFactors non-null non-negative extent coefficients, copied
     * @param baseElementOffset non-negative constant element origin
     * @param strideConstants non-null non-negative element-stride constants, copied
     * @param strideBatchFactors non-null non-negative element-stride coefficients, copied
     * @param familyProvedRepeatedWrites explicit affine-copy repeated-destination strategy
     */
    public record BatchBinding(CpuAccessPlan plan, List<Long> extentConstants,
            List<Long> extentBatchFactors, long baseElementOffset,
            List<Long> strideConstants, List<Long> strideBatchFactors,
            boolean familyProvedRepeatedWrites) {
        /**
         * Constructs the ordinary nonoverlapping access recipe.
         * @param plan structural direction and axes
         * @param extentConstants non-negative extent constants
         * @param extentBatchFactors non-negative extent coefficients
         * @param baseElementOffset non-negative fixed element origin
         * @param strideConstants non-negative stride constants
         * @param strideBatchFactors non-negative stride coefficients
         */
        public BatchBinding(CpuAccessPlan plan, List<Long> extentConstants,
                List<Long> extentBatchFactors, long baseElementOffset,
                List<Long> strideConstants, List<Long> strideBatchFactors) {
            this(plan, extentConstants, extentBatchFactors, baseElementOffset,
                    strideConstants, strideBatchFactors, false);
        }
        /**
         * Validates coefficients and proves writes, except explicit affine repeated writes
         * whose source/destination relationship must be checked by the owning copy route.
         * @param plan non-null structural plan
         * @param extentConstants non-null extent constants without nulls or negatives
         * @param extentBatchFactors non-null extent factors without nulls or negatives
         * @param baseElementOffset non-negative constant origin in elements
         * @param strideConstants non-null stride constants without nulls or negatives
         * @param strideBatchFactors non-null stride factors without nulls or negatives
         * @param familyProvedRepeatedWrites affine-copy repeated-destination binding strategy
         * @throws NullPointerException if a reference or list element is null
         * @throws IllegalArgumentException if rank, coefficients or canonical write proof disagree
         * @throws ArithmeticException if structural coefficient arithmetic overflows
         */
        public BatchBinding {
            Objects.requireNonNull(plan, "plan");
            extentConstants = Binding.checked(extentConstants, "extentConstants");
            extentBatchFactors = Binding.checked(extentBatchFactors, "extentBatchFactors");
            strideConstants = Binding.checked(strideConstants, "strideConstants");
            strideBatchFactors = Binding.checked(strideBatchFactors, "strideBatchFactors");
            int rank = plan.iterationRank();
            if (baseElementOffset < 0 || extentConstants.size() != rank
                    || extentBatchFactors.size() != rank || strideConstants.size() != rank
                    || strideBatchFactors.size() != rank) {
                throw new IllegalArgumentException("batch binding rank or origin disagrees");
            }
            int variableAxes = 0;
            long suffixConstant = 1, suffixFactor = 0;
            for (int axis = rank - 1; axis >= 0; axis--) {
                long constant = extentConstants.get(axis), factor = extentBatchFactors.get(axis);
                if (factor != 0) variableAxes++;
                if (factor != 0 && suffixFactor != 0) {
                    throw new IllegalArgumentException("non-affine batch product");
                }
                suffixFactor = Math.addExact(Math.multiplyExact(suffixFactor, constant),
                        Math.multiplyExact(suffixConstant, factor));
                suffixConstant = Math.multiplyExact(suffixConstant, constant);
            }
            if (variableAxes > 1) throw new IllegalArgumentException("multiple B-bearing axes");
            if (familyProvedRepeatedWrites && plan.accessKind() != AccessKind.WRITE)
                throw new IllegalArgumentException("repeated-write strategy requires a write plan");
            if (plan.accessKind() == AccessKind.WRITE && !familyProvedRepeatedWrites)
                proveWriteSeparation(extentConstants, extentBatchFactors, strideConstants, strideBatchFactors);
            Math.toIntExact(baseElementOffset);
            for (long extent : extentConstants) Math.toIntExact(extent);
            for (long stride : strideConstants) Math.toIntExact(stride);
        }

        /**
         * Proves mixed-radix separation for every positive B. Polynomial coefficients use
         * B-1, not a sampled descriptor; B=0 has no accesses when a variable axis is present.
         */
        private static void proveWriteSeparation(List<Long> ec, List<Long> ef,
                List<Long> sc, List<Long> sf) {
            for (int axis = 0; axis < ec.size(); axis++)
                if (ec.get(axis) == 0 && ef.get(axis) == 0) return;
            var axes = new ArrayList<Integer>();
            for (int axis = 0; axis < ec.size(); axis++)
                if (ef.get(axis) != 0 || ec.get(axis) > 1) axes.add(axis);
            axes.sort(java.util.Comparator.<Integer>comparingLong(axis -> sf.get(axis))
                    .thenComparingLong(axis -> Math.addExact(sc.get(axis), sf.get(axis))));
            long coveredConstant = 1, coveredFactor = 0;
            for (int axis : axes) {
                long strideConstant = Math.addExact(sc.get(axis), sf.get(axis));
                long strideFactor = sf.get(axis);
                if (strideConstant < coveredConstant || strideFactor < coveredFactor)
                    throw new IllegalArgumentException("batch write addresses are not proved disjoint");
                long extentConstant = Math.addExact(ec.get(axis), ef.get(axis)) - 1;
                long extentFactor = ef.get(axis);
                if (extentFactor != 0 && strideFactor != 0)
                    throw new IllegalArgumentException("non-affine batch write span");
                coveredConstant = Math.addExact(coveredConstant,
                        Math.multiplyExact(extentConstant, strideConstant));
                coveredFactor = Math.addExact(coveredFactor, Math.addExact(
                        Math.multiplyExact(extentConstant, strideFactor),
                        Math.multiplyExact(extentFactor, strideConstant)));
            }
        }

        /**
         * Preserves literal recipe text on every old nonoverlapping route.
         * @return deterministic structural recipe text, excluding invocation B
         */
        @Override public String toString() {
            return "BatchBinding[plan=" + plan + ", extentConstants=" + extentConstants
                    + ", extentBatchFactors=" + extentBatchFactors + ", baseElementOffset=" + baseElementOffset
                    + ", strideConstants=" + strideConstants + ", strideBatchFactors=" + strideBatchFactors
                    + (familyProvedRepeatedWrites ? ", familyProvedRepeatedWrites=true" : "") + "]";
        }

        /**
         * Resolves checked int-limited geometry once on the cold invocation boundary.
         * No Model lowerer or element-enumerating injectivity proof is called.
         * @param batchSize non-negative actual B, int-representable
         * @return fresh immutable numeric full-range binding
         * @throws IllegalArgumentException if B is negative
         * @throws ArithmeticException if extents, strides, counts or referenced span exceed limits
         */
        public Binding bind(long batchSize) {
            if (batchSize < 0) throw new IllegalArgumentException("batchSize must be non-negative");
            Math.toIntExact(batchSize);
            int rank = plan.iterationRank();
            long[] extents = new long[rank], strides = new long[rank];
            long count = 1;
            // Right-to-left multiplication checks trailing geometry even if a leading axis is zero.
            for (int axis = rank - 1; axis >= 0; axis--) {
                extents[axis] = Math.toIntExact(Math.addExact(extentConstants.get(axis),
                        Math.multiplyExact(extentBatchFactors.get(axis), batchSize)));
                strides[axis] = Math.toIntExact(Math.addExact(strideConstants.get(axis),
                        Math.multiplyExact(strideBatchFactors.get(axis), batchSize)));
                count = Math.toIntExact(Math.multiplyExact(count, extents[axis]));
            }
            Math.toIntExact(baseElementOffset);
            long span = baseElementOffset;
            if (count != 0) {
                for (int axis = 0; axis < rank; axis++) span = Math.addExact(span,
                        Math.multiplyExact(extents[axis] - 1, strides[axis]));
                span = Math.toIntExact(Math.addExact(span, 1));
            } else {
                span = baseElementOffset;
            }
            return Binding.create(plan, extents, baseElementOffset, strides, count, 0, count, span);
        }
    }
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

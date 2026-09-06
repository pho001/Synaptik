package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/**
 * CPU-private immutable recipe facts for one modular integral partial reduction.
 *
 * <p>Unlike {@link CpuAggregateIr}, this identity describes a split selected domain: a partial
 * body owns one ordinal quotient/remainder interval and writes only its aligned state slot.  A
 * later, invoking-thread combine consumes those slots in ascending ordinal order and is the sole
 * output writer.  The type is deliberately limited to ordinary {@code SUM}/{@code PROD} over
 * same-width primitive {@code INT32}/{@code INT64} values; it is not a reusable reduction state
 * abstraction and grants no floating-point reassociation.</p>
 *
 * @param kind non-null modular aggregate kind
 * @param dataType non-null same-width represented type
 * @param form non-null ordinary aggregate form
 * @param outputCount exact positive number of output cells
 * @param domainCount exact positive selected-domain cardinality per output cell
 * @param partialCount fixed two or four partial states per output cell
 */
public record CpuPartialReductionIr(Kind kind, DataType dataType, CpuAggregateIr.Form form,
        long outputCount, long domainCount, int partialCount) {
    /** One of the two associative same-width modular folds admitted by this private route. */
    public enum Kind {
        /** Two's-complement addition with wraparound at the represented width. */
        SUM,
        /** Two's-complement multiplication with wraparound at the represented width. */
        PROD
    }

    /** Every state slot has this fixed eight-byte aligned stride. */
    public static final long STATE_SLICE_BYTES = Long.BYTES;

    /**
     * Validates one fully prepared partial-reduction structural identity.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the facts exceed the deliberately narrow route
     */
    public CpuPartialReductionIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(form, "form");
        if ((dataType != DataType.INT32 && dataType != DataType.INT64)
                || form == CpuAggregateIr.Form.SUM_TO_SHAPE || outputCount <= 0
                || domainCount <= 0 || (partialCount != 2 && partialCount != 4)
                || domainCount < partialCount) {
            throw new IllegalArgumentException("partial-reduction facts are not admitted");
        }
    }

    /**
     * Returns the inclusive selected-domain ordinal at which one partial begins.
     *
     * @param outputOrdinal output-cell ordinal, validated for consistency but not used by the
     *     per-cell quotient/remainder geometry
     * @param partialOrdinal zero-based partial ordinal
     * @return the non-negative inclusive selected-domain ordinal
     * @throws IllegalArgumentException if either ordinal is outside this prepared recipe
     */
    public long begin(long outputOrdinal, int partialOrdinal) {
        checkOrdinals(outputOrdinal, partialOrdinal);
        return partialOrdinal * (domainCount / partialCount)
                + Math.min(partialOrdinal, (int) (domainCount % partialCount));
    }

    /**
     * Returns the exclusive selected-domain ordinal at which one partial ends.
     *
     * @param outputOrdinal output-cell ordinal, validated for consistency but not used by the
     *     per-cell quotient/remainder geometry
     * @param partialOrdinal zero-based partial ordinal
     * @return the positive exclusive selected-domain ordinal
     * @throws IllegalArgumentException if either ordinal is outside this prepared recipe
     */
    public long end(long outputOrdinal, int partialOrdinal) {
        checkOrdinals(outputOrdinal, partialOrdinal);
        return partialOrdinal + 1 == partialCount ? domainCount : begin(outputOrdinal,
                partialOrdinal + 1);
    }

    /**
     * Returns the byte offset of one aligned run-owned state slot.
     *
     * @param outputOrdinal zero-based output-cell ordinal
     * @param partialOrdinal zero-based partial ordinal
     * @return checked aligned offset into the route workspace
     * @throws IllegalArgumentException if an ordinal is outside this prepared recipe
     * @throws ArithmeticException if the checked offset cannot be represented
     */
    public long stateOffset(long outputOrdinal, int partialOrdinal) {
        checkOrdinals(outputOrdinal, partialOrdinal);
        return Math.multiplyExact(Math.addExact(Math.multiplyExact(outputOrdinal, partialCount),
                partialOrdinal), STATE_SLICE_BYTES);
    }

    /**
     * Returns the exact aligned workspace byte count for every partial state.
     *
     * @return checked positive multiple of eight bytes
     * @throws ArithmeticException if the state geometry overflows {@code long}
     */
    public long workspaceBytes() {
        return Math.multiplyExact(Math.multiplyExact(outputCount, partialCount), STATE_SLICE_BYTES);
    }

    private void checkOrdinals(long outputOrdinal, int partialOrdinal) {
        if (outputOrdinal < 0 || outputOrdinal >= outputCount || partialOrdinal < 0
                || partialOrdinal >= partialCount) {
            throw new IllegalArgumentException("partial-reduction ordinal is outside prepared geometry");
        }
    }
}

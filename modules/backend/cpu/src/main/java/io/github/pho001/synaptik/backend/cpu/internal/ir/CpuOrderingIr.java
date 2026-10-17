package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Records the CPU-private structural identity of one stable axis-ordering occurrence.
 *
 * <p>The identity fixes the family, represented value type, direction, TOP_K output-order flag,
 * ordered boundary access forms, and two-region INT64 merge-scratch policy. Concrete extents,
 * axis, selection count, offsets, stride magnitudes, carriers, assigned workspace, and execution
 * ranges remain cold prepared facts and therefore do not enter this value.</p>
 *
 * @param family non-null SORT, ARGSORT, or TOP_K family
 * @param representedType non-null input and values-output represented type; ARGSORT and TOP_K
 *     indices remain INT64 independently
 * @param descending whether non-NaN values use descending order; floating NaNs remain last
 * @param sorted for TOP_K, whether selected pairs retain stable value order; must be {@code true}
 *     for SORT and ARGSORT
 * @param boundaryAccess non-null ordered input then output access plans; copied defensively
 * @param scratchPolicy exact two-INT64-region merge policy
 */
public record CpuOrderingIr(Family family, DataType representedType, boolean descending,
        boolean sorted, List<CpuAccessPlan> boundaryAccess, int scratchPolicy)
        implements CpuPortableKernelIr {
    /** Current generated ordering families and their output roles. */
    public enum Family {
        /** Writes represented values in the complete stable order. */ SORT,
        /** Writes INT64 logical-axis indices in the complete stable order. */ ARGSORT,
        /** Writes represented selected values followed by their INT64 logical-axis indices. */ TOP_K
    }
    /** Two primitive INT64 merge-index regions per active range. */
    public static final int TWO_INDEX_MERGE_REGIONS = 1;

    /**
     * Validates and snapshots one exact structural ordering form.
     *
     * @param family non-null ordering family
     * @param representedType non-null represented input/value type
     * @param descending exact requested value direction
     * @param sorted exact TOP_K output-order flag; must be {@code true} otherwise
     * @param boundaryAccess ordered read input followed by one or two write outputs
     * @param scratchPolicy {@link #TWO_INDEX_MERGE_REGIONS}
     * @throws NullPointerException if a required reference or access-plan element is null
     * @throws IllegalArgumentException if boundary roles, output count, output order, or scratch
     *     policy disagrees with the family
     */
    public CpuOrderingIr {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(representedType, "representedType");
        boundaryAccess = List.copyOf(boundaryAccess);
        int outputs = family == Family.TOP_K ? 2 : 1;
        if (boundaryAccess.size() != outputs + 1
                || boundaryAccess.getFirst().accessKind() != CpuAccessPlan.AccessKind.READ
                || boundaryAccess.subList(1, boundaryAccess.size()).stream()
                    .anyMatch(plan -> plan.accessKind() != CpuAccessPlan.AccessKind.WRITE)
                || family != Family.TOP_K && !sorted
                || scratchPolicy != TWO_INDEX_MERGE_REGIONS) {
            throw new IllegalArgumentException("ordering structural facts disagree");
        }
    }

    /**
     * Returns the instruction-free cache-compatible kernel encoding.
     *
     * @return a fresh immutable kernel IR with ordered boundary values, one structural output
     *     store marker, and the complete ordering family identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        values.add(new CpuKernelIr.Value(0, representedType, CpuKernelIr.Value.Kind.INPUT,
                boundaryAccess.getFirst()));
        if (family == Family.ARGSORT) {
            values.add(new CpuKernelIr.Value(1, DataType.INT64, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.get(1)));
        } else {
            values.add(new CpuKernelIr.Value(1, representedType, CpuKernelIr.Value.Kind.OUTPUT,
                    boundaryAccess.get(1)));
            if (family == Family.TOP_K) values.add(new CpuKernelIr.Value(2, DataType.INT64,
                    CpuKernelIr.Value.Kind.OUTPUT, boundaryAccess.get(2)));
        }
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)), "ordering:" + family
                        + ":descending=" + descending + ":sorted=" + sorted
                        + ":scratch=" + scratchPolicy);
    }

    /**
     * Returns the deterministic cache key derived from the structural encoding.
     *
     * @return a non-null hexadecimal structural key that excludes cold geometry and resources
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

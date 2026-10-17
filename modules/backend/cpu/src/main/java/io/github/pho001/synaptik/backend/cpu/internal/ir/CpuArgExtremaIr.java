package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private generated-code identity for one one-axis arg-min or arg-max occurrence.
 *
 * <p>The identity fixes the represented numeric input type, logical-index result type, already
 * normalized axis, dimension-retention form, tie policy, and structural boundary accesses.
 * Concrete extents, offsets, stride magnitudes, carriers, slots, ranges, and invocation state
 * remain cold geometry.</p>
 *
 * @param kind non-null arg-min or arg-max selection
 * @param inputType non-null supported numeric represented input type
 * @param axis already-normalized non-negative logical input axis
 * @param keepDimensions whether the output retains the selected axis with extent one
 * @param tiePolicy non-null first- or last-logical-coordinate tie policy
 * @param narrowLogicalIndex whether the statically proved selected extent fits an {@code int}
 * @param narrowOutputIndex whether the statically proved output-cell count fits an {@code int}
 * @param inputAccess non-null read-only structural input access
 * @param outputAccess non-null write-only structural INT64 output access
 */
public record CpuArgExtremaIr(Kind kind, DataType inputType, int axis,
        boolean keepDimensions, ArgExtremaTiePolicy tiePolicy, boolean narrowLogicalIndex,
        boolean narrowOutputIndex,
        CpuAccessPlan inputAccess, CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /** Exact supported selection kind. */
    public enum Kind {
        /** Selects the least ordered represented value. */ ARG_MIN,
        /** Selects the greatest ordered represented value. */ ARG_MAX
    }

    /**
     * Validates and retains one exact structural identity.
     *
     * @param kind non-null arg-min or arg-max selection
     * @param inputType non-null supported numeric represented input type
     * @param axis already-normalized non-negative logical input axis
     * @param keepDimensions whether the output retains the selected axis with extent one
     * @param tiePolicy non-null first- or last-logical-coordinate tie policy
     * @param narrowLogicalIndex whether the selected extent is statically proved to fit an
     *     {@code int}
     * @param narrowOutputIndex whether the output-cell count is statically proved to fit an
     *     {@code int}
     * @param inputAccess non-null read-only structural input access
     * @param outputAccess non-null write-only structural INT64 output access
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if {@code inputType} is BOOL, {@code axis} is negative,
     *     or either access has the wrong read/write role
     */
    public CpuArgExtremaIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(inputType, "inputType");
        Objects.requireNonNull(tiePolicy, "tiePolicy");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (axis < 0 || inputType == DataType.BOOL
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("arg-extrema structural facts disagree");
        }
    }

    /**
     * Encodes this identity for generated-artifact compatibility.
     *
     * @return a new instruction-free canonical form with numeric input and INT64 output
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, inputType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, DataType.INT64, CpuKernelIr.Value.Kind.OUTPUT,
                        outputAccess)), List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "arg-extrema:" + kind + ":axis=" + axis + ":keep=" + keepDimensions
                        + ":narrow-index=" + narrowLogicalIndex
                        + ":narrow-output=" + narrowOutputIndex + ":tie=" + tiePolicy);
    }

    /** {@inheritDoc} */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

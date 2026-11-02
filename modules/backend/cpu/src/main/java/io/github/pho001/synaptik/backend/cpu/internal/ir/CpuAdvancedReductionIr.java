package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one logarithmic, statistical, or norm reduction.
 *
 * <p>Ordered axes remain diagnostic and compatibility identity, while {@code selectedAxes}
 * records canonical input-axis membership used by the generated row-major traversal. Concrete
 * extents, offsets, strides, carriers, ranges, and workspace slots remain cold geometry.</p>
 *
 * <p>Equality and hashing include every record component, comparing the defensively retained
 * axis arrays by content rather than array identity.</p>
 *
 * @param kind exact advanced reduction meaning
 * @param dataType identical floating input and output type
 * @param orderedAxes normalized ordered axes, copied defensively
 * @param selectedAxes input-rank membership flags, copied defensively
 * @param keepDimensions whether selected axes remain as extent-one output dimensions
 * @param correction statistical denominator correction, or zero for non-statistical kinds
 * @param algorithmVersion CPU-private numerical-algorithm compatibility version
 * @param passCount number of complete selected-domain passes
 * @param domainCount checked number of represented values in each selected domain
 * @param stateLimbCount exact-sum signed-limb count, or zero when no exact state is used
 * @param scratchSliceBytes exact per-range scratch bytes, or zero
 * @param inputAccess structural input read access
 * @param outputAccess structural output write access
 */
public record CpuAdvancedReductionIr(Kind kind, DataType dataType, int[] orderedAxes,
        boolean[] selectedAxes, boolean keepDimensions, long correction, int algorithmVersion,
        int passCount, long domainCount, int stateLimbCount, long scratchSliceBytes,
        CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /** Supported advanced meanings. */
    public enum Kind {
        /** Stable maximum-shift logarithm of the selected exponential sum. */ LOG_SUM_EXP,
        /** Corrected two-pass variance. */ VARIANCE,
        /** Non-negative principal square root of corrected variance. */ STANDARD_DEVIATION,
        /** Exact represented sum of absolute values. */ L1_NORM,
        /** Scaled sum-of-squares Euclidean norm. */ L2_NORM
    }

    /** Validates and snapshots one code-shaping identity. */
    public CpuAdvancedReductionIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        orderedAxes = Objects.requireNonNull(orderedAxes, "orderedAxes").clone();
        selectedAxes = Objects.requireNonNull(selectedAxes, "selectedAxes").clone();
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        boolean floating = dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16 || dataType == DataType.FLOAT16;
        boolean statistical = kind == Kind.VARIANCE || kind == Kind.STANDARD_DEVIATION;
        if (!floating || selectedAxes.length != inputAccess.iterationRank()
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || correction < 0 || !statistical && correction != 0
                || algorithmVersion != 1 || passCount != (kind == Kind.LOG_SUM_EXP
                    || statistical ? 2 : 1)
                || domainCount < 0 || stateLimbCount < 0 || scratchSliceBytes < 0
                || scratchSliceBytes % Long.BYTES != 0
                || stateLimbCount == 0 && scratchSliceBytes != 0
                || stateLimbCount > 0 && scratchSliceBytes
                    != Math.addExact(Long.BYTES, Math.multiplyExact(Long.BYTES, stateLimbCount))) {
            throw new IllegalArgumentException("advanced-reduction structural facts disagree");
        }
        boolean[] seen = new boolean[selectedAxes.length];
        for (int axis : orderedAxes) {
            if (axis < 0 || axis >= seen.length || seen[axis] || !selectedAxes[axis]) {
                throw new IllegalArgumentException("advanced-reduction axes disagree");
            }
            seen[axis] = true;
        }
        if (!Arrays.equals(seen, selectedAxes)) {
            throw new IllegalArgumentException("advanced-reduction membership disagrees");
        }
        int expectedRank = keepDimensions ? selectedAxes.length
                : selectedAxes.length - orderedAxes.length;
        if (outputAccess.iterationRank() != expectedRank) {
            throw new IllegalArgumentException("advanced-reduction output rank disagrees");
        }
    }

    /**
     * Returns the normalized axes in Model-owned order.
     * @return a new copy of the normalized ordered axes
     */
    @Override public int[] orderedAxes() { return orderedAxes.clone(); }
    /**
     * Returns canonical input-axis membership used by generated traversal.
     * @return a new copy of the canonical input-axis membership flags
     */
    @Override public boolean[] selectedAxes() { return selectedAxes.clone(); }

    /**
     * Compares every immutable component, including ordered axes and membership by content.
     *
     * @param other candidate value; may be {@code null}
     * @return whether the candidate is an advanced reduction with identical component values
     */
    @Override public boolean equals(Object other) {
        return this == other || other instanceof CpuAdvancedReductionIr ir
                && kind == ir.kind && dataType == ir.dataType
                && Arrays.equals(orderedAxes, ir.orderedAxes)
                && Arrays.equals(selectedAxes, ir.selectedAxes)
                && keepDimensions == ir.keepDimensions && correction == ir.correction
                && algorithmVersion == ir.algorithmVersion && passCount == ir.passCount
                && domainCount == ir.domainCount && stateLimbCount == ir.stateLimbCount
                && scratchSliceBytes == ir.scratchSliceBytes
                && inputAccess.equals(ir.inputAccess) && outputAccess.equals(ir.outputAccess);
    }

    /**
     * Hashes all immutable components with array-content semantics matching {@link #equals(Object)}.
     *
     * @return content hash without boxing components or exposing retained arrays
     */
    @Override public int hashCode() {
        int result = kind.hashCode();
        result = 31 * result + dataType.hashCode();
        result = 31 * result + Arrays.hashCode(orderedAxes);
        result = 31 * result + Arrays.hashCode(selectedAxes);
        result = 31 * result + Boolean.hashCode(keepDimensions);
        result = 31 * result + Long.hashCode(correction);
        result = 31 * result + Integer.hashCode(algorithmVersion);
        result = 31 * result + Integer.hashCode(passCount);
        result = 31 * result + Long.hashCode(domainCount);
        result = 31 * result + Integer.hashCode(stateLimbCount);
        result = 31 * result + Long.hashCode(scratchSliceBytes);
        result = 31 * result + inputAccess.hashCode();
        return 31 * result + outputAccess.hashCode();
    }

    /**
     * Encodes this family for generated-artifact compatibility.
     *
     * @return a new canonical instruction-free generated-kernel identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess)),
                List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "advanced-reduction:" + kind + ":type=" + dataType + ":axes="
                        + Arrays.toString(orderedAxes) + ":selected="
                        + Arrays.toString(selectedAxes) + ":keep=" + keepDimensions
                        + ":correction=" + correction + ":algorithm=" + algorithmVersion
                        + ":passes=" + passCount + ":domain=" + domainCount
                        + ":limbs=" + stateLimbCount + ":slice=" + scratchSliceBytes);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

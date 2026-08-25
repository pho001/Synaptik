package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one axis-removing masked floating sum or mean.
 *
 * <p>The identity fixes the normalized data axis, directional right-aligned mask topology,
 * structural access regimes, and exact-state shape. Concrete extents, offsets, stride
 * magnitudes, carrier objects, ranges, and workspace slots remain cold invocation geometry.</p>
 *
 * @param kind exact masked sum or mean meaning
 * @param dataType represented floating data and output type
 * @param axis normalized data axis removed from the result
 * @param maskRank resolved mask rank
 * @param singletonMaskAxes data-rank flags identifying aligned extent-one mask dimensions;
 *     omitted leading dimensions are represented by the rank difference; copied defensively
 * @param maximumDomainCount static selected-axis extent used to size exact state
 * @param stateLimbCount positive fixed exact-sum limb count
 * @param scratchSliceBytes exact bytes in one invocation-private state slice
 * @param dataAccess structural data read access
 * @param maskAccess structural canonical-BOOL mask read access
 * @param outputAccess structural result write access
 */
public record CpuMaskedReductionIr(Kind kind, DataType dataType, int axis, int maskRank,
        boolean[] singletonMaskAxes, long maximumDomainCount, int stateLimbCount,
        long scratchSliceBytes, CpuAccessPlan dataAccess, CpuAccessPlan maskAccess,
        CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /** Supported masked reduction meanings. */
    public enum Kind {
        /** Sums selected represented values exactly before one rounding. */ SUM,
        /** Divides the exact selected sum by the selected count before one rounding. */ MEAN
    }

    /**
     * Validates and snapshots one masked-reduction code-shaping identity.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if type, rank, axis, state, or access facts disagree
     */
    public CpuMaskedReductionIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        singletonMaskAxes = Objects.requireNonNull(singletonMaskAxes,
                "singletonMaskAxes").clone();
        Objects.requireNonNull(dataAccess, "dataAccess");
        Objects.requireNonNull(maskAccess, "maskAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        boolean floating = dataType == DataType.FLOAT64 || dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16;
        if (!floating || axis < 0 || axis >= dataAccess.iterationRank()
                || maskRank < 0 || maskRank > dataAccess.iterationRank()
                || singletonMaskAxes.length != dataAccess.iterationRank()
                || maximumDomainCount < 0 || stateLimbCount <= 0
                || scratchSliceBytes != Math.addExact(8L,
                        Math.multiplyExact(8L, stateLimbCount))
                || dataAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || maskAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || maskAccess.iterationRank() != maskRank
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || outputAccess.iterationRank() != dataAccess.iterationRank() - 1) {
            throw new IllegalArgumentException("masked-reduction structural facts disagree");
        }
        int omitted = dataAccess.iterationRank() - maskRank;
        for (int dataAxis = 0; dataAxis < omitted; dataAxis++) {
            if (singletonMaskAxes[dataAxis]) throw new IllegalArgumentException(
                    "omitted mask axes are not aligned singleton axes");
        }
    }

    /**
     * Returns which data axes have an aligned extent-one mask dimension.
     *
     * <p>The returned array is a defensive copy. Mutating it does not change this immutable
     * structural identity. Leading data axes omitted by the lower-rank mask are {@code false}.</p>
     *
     * @return a new data-rank array of aligned extent-one mask-axis flags; never {@code null}
     */
    @Override public boolean[] singletonMaskAxes() { return singletonMaskAxes.clone(); }

    /**
     * Encodes this family for generated-artifact compatibility.
     *
     * @return a new instruction-free three-boundary kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, dataAccess),
                new CpuKernelIr.Value(1, DataType.BOOL, CpuKernelIr.Value.Kind.INPUT, maskAccess),
                new CpuKernelIr.Value(2, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess)),
                List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(2, 0)),
                "masked-reduction:" + kind + ":type=" + dataType + ":axis=" + axis
                        + ":mask-rank=" + maskRank + ":singletons="
                        + Arrays.toString(singletonMaskAxes) + ":domain=" + maximumDomainCount
                        + ":limbs=" + stateLimbCount + ":slice=" + scratchSliceBytes);
    }

    /** {@inheritDoc} */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

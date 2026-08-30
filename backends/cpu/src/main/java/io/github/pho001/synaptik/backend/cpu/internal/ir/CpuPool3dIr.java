package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * Immutable code-shaping identity for one direct scalar NCDHW Pool3d body.
 *
 * @param kind exact maximum or fixed-divisor-average numerical family
 * @param dataType exact supported floating input and output representation
 * @param realization direct scalar output-cell realization
 * @param inputAccess immutable rank-five read-access identity
 * @param outputAccess immutable rank-five write-access identity
 */
public record CpuPool3dIr(
        Kind kind,
        DataType dataType,
        Realization realization,
        CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {
    /** Closed numerical-family vocabulary. */
    public enum Kind {
        /** Excluded-padding maximum. */
        MAX,
        /** Fixed-kernel-count average. */
        AVERAGE
    }

    /** Closed generated realization vocabulary. */
    public enum Realization {
        /** One complete depth-height-width window per output-cell work item. */
        DIRECT_SCALAR
    }

    /**
     * Validates one generated Pool3d identity.
     *
     * @param kind exact numerical family
     * @param dataType supported floating boundary type
     * @param realization direct scalar realization
     * @param inputAccess immutable read access identity
     * @param outputAccess immutable write access identity
     * @throws NullPointerException if any component is null
     * @throws IllegalArgumentException if the type or access directions are unsupported
     */
    public CpuPool3dIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(realization, "realization");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (dataType != DataType.BFLOAT16
                        && dataType != DataType.FLOAT32
                        && dataType != DataType.FLOAT64
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("Pool3d IR facts disagree");
        }
    }

    /**
     * Encodes every code-shaping fact while leaving concrete rank-five geometry cold.
     *
     * @return canonical instruction-free kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values = List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "pool3d:kind=" + kind + ":type=" + dataType + ":realization=" + realization);
    }

    /** Returns the deterministic structural key.
     * @return lowercase hexadecimal structural key
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

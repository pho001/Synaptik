package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * Immutable code-shaping identity for one direct scalar channels-first (NCHW) Pool2d body.
 *
 * <p>The identity deliberately excludes concrete extents, strides, offsets, and range bounds.
 * Those values remain cold invocation geometry, while this record retains only the numerical
 * family, represented type, generated realization, and input/output access regimes that can
 * change generated class bytes.
 *
 * @param kind exact pooling numerical family
 * @param dataType supported floating boundary type
 * @param realization direct generated realization
 * @param inputAccess immutable input access identity
 * @param outputAccess immutable output access identity
 */
public record CpuPool2dIr(
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
        /** One complete window per output-cell work item. */
        DIRECT_SCALAR
    }

    /**
     * Validates one generated pooling identity.
     *
     * @param kind exact pooling numerical family
     * @param dataType supported floating boundary type
     * @param realization direct generated realization
     * @param inputAccess immutable input access identity
     * @param outputAccess immutable output access identity
     * @throws NullPointerException if any identity component is {@code null}
     * @throws IllegalArgumentException if the type or access directions are unsupported
     */
    public CpuPool2dIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(realization, "realization");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (dataType != DataType.BFLOAT16
                        && dataType != DataType.FLOAT16
                        && dataType != DataType.FLOAT32
                        && dataType != DataType.FLOAT64
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE) {
            throw new IllegalArgumentException("Pool2d IR facts disagree");
        }
    }

    /**
     * Encodes all code-shaping facts while leaving concrete geometry cold.
     *
     * @return canonical instruction-free kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values =
                List.of(
                        new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                        new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(
                values,
                List.of(),
                new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "pool2d:kind=" + kind + ":type=" + dataType + ":realization=" + realization);
    }

    /**
     * Returns the deterministic structural key.
     *
     * @return lowercase hexadecimal structural key
     */
    @Override
    public String structuralKey() {
        return encodedKernelIr().structuralKey();
    }
    /**
     * Reads only the semantic discriminator consumed by emission, without decoding geometry.
     * @param ir non-null canonical pool2d envelope
     * @return the existing encoded semantic discriminator
     * @throws NullPointerException if the envelope is null
     * @throws IllegalArgumentException if the family or discriminator is malformed
     */
    public static Kind encodedKind(CpuKernelIr ir) {
        String identity = Objects.requireNonNull(ir, "ir").familyIdentity();
        if (!identity.startsWith("pool2d:kind=")) throw new IllegalArgumentException("wrong encoded family");
        int end = identity.indexOf(':', 12);
        if (end < 0) throw new IllegalArgumentException("missing encoded discriminator");
        return Kind.valueOf(identity.substring(12, end));
    }

}

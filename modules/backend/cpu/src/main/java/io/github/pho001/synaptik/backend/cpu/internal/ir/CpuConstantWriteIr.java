package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.List;
import java.util.Objects;

/**
 * Writes one cold-proved scalar's exact represented bits to every logical output cell.
 *
 * <p>This physical lowering is not a new graph constant source. It has no computational input,
 * contributor traversal, runtime conversion or scratch. The containing partition retains any
 * original constant-source initialization required by shared Prepare. Concrete output extents,
 * addresses and slot assignments remain separate cold facts; the literal is baked into both
 * existing generated compatibility identities.</p>
 *
 * @param immediate non-null exact represented output bits, including low floating storage bits
 * @param outputAccess non-null injective output-write access over the logical result domain
 */
public record CpuConstantWriteIr(CpuKernelIr.ScalarImmediate immediate,
        CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /**
     * Validates the narrow numeric write form; injectivity is proved by its lowerer.
     * @param immediate non-null exact numeric bits; BOOL is not a count result
     * @param outputAccess non-null structural write access
     * @throws NullPointerException if either component is null
     * @throws IllegalArgumentException if the type or access direction is unsupported
     */
    public CpuConstantWriteIr {
        Objects.requireNonNull(immediate, "immediate");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (immediate.dataType() == io.github.pho001.synaptik.model.datatype.DataType.BOOL
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE)
            throw new IllegalArgumentException("constant count requires one numeric write");
    }

    /**
     * Encodes an output-only compatibility envelope, not a pointwise computation. Both existing
     * specialization identity streams include its fingerprint, hence its baked literal bits.
     * @return immutable one-output envelope with no inputs or instructions; never null
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(List.of(new CpuKernelIr.Value(0, immediate.dataType(),
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess)), List.of(),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(0, 0)),
                "constant-write-v1:" + Long.toUnsignedString(immediate.bits(), 16));
    }

    /**
     * Identifies the generated meaning without including concrete extents or assigned slots.
     * @return deterministic structural identity including the exact typed literal and write access;
     *     never {@code null}
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}

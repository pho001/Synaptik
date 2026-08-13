package io.github.pho001.synaptik.backend.cpu.internal.cache;

/**
 * Defines the current generated-kernel compatibility schema and entry-point name.
 *
 * <p>The schema is current-only: changing a code-shaping fact, including scalar versus vector
 * compute, exact vector species, adjusted carrier/access pattern, selected materialized source,
 * opcode vocabulary, scalar-power realization, or exact two-bound clamp immediate, requires
 * compatible current-version metadata. Schema 12 extends schema 11 with the structural static
 * PAD/TILE/CONCAT/STACK family, ordered unique-boundary occurrence mapping, exact represented-bit
 * padding immediate, and scalar generated body. Schema 13 adds static window extraction,
 * schema 14 adds mixed-type gather/one-hot identity and output writers, and schema 15 adds
 * functional slice-update identity and its signed-sequence cursor body. Schema 16 adds functional
 * scatter family/reduction identity, typed direct output writers, and the optional exact-product
 * scratch entry signature. Schema 17 adds the distinct overlap-fold family, represented
 * sequential-addition policy, and workspace-free output-domain writer. Schema 18 adds stable
 * ordering family, represented type, direction/output-order flags, ordered one- or two-output
 * boundary structure, and the explicit scratch-bearing generated entry signature.
 * The optional persistent envelope stores this version and has no legacy reader, migration path,
 * or converter.
 */
public final class CpuGeneratorSchema {
    /**
     * Current schema version, including portable static axis and NCHW window extraction,
     * unequal-rank movement geometry, exact represented padding bits, and scalar generated
     * bodies. Schema 14 adds mixed-type gather/one-hot structural identity and generated output
     * writers; schema 15 adds structural functional slice-update identity and its generated
     * cursor body; schema 16 adds current functional scatter and its explicit scratch signature;
     * schema 17 adds current overlap fold and represented sequential addition; schema 18 adds
     * stable SORT/ARGSORT/TOP_K structural identity, multi-store shape, and merge scratch.
     * Envelopes written for earlier schemas are incompatible misses.
     */
    public static final int CURRENT_VERSION = 18;
    /** Generated entry name. */ public static final String ENTRY_NAME = "invoke";
    private CpuGeneratorSchema() { }

    /**
     * Returns a deterministic generated binary name.
     * @param specialization non-null exact structural specialization
     * @return a deterministic binary name in the CPU code-generation package; never {@code null}
     * @throws NullPointerException if {@code specialization} is {@code null}
     */
    public static String generatedBinaryName(CpuKernelSpecialization specialization) {
        return "io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.Generated_"
                + specialization.structuralKey();
    }
}

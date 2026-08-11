package io.github.pho001.synaptik.backend.cpu.internal.cache;

/**
 * Defines the current generated-kernel compatibility schema and entry-point name.
 *
 * <p>The schema is current-only: changing a code-shaping fact, including scalar versus vector
 * compute, exact vector species, adjusted carrier/access pattern, selected materialized source,
 * opcode vocabulary, scalar-power realization, or exact two-bound clamp immediate, requires
 * compatible current-version metadata. Schema 12 extends schema 11 with the structural static
 * PAD/TILE/CONCAT/STACK family, ordered unique-boundary occurrence mapping, exact represented-bit
 * padding immediate, and scalar generated body.
 * The optional persistent envelope stores this version and has no legacy reader, migration path,
 * or converter.
 */
public final class CpuGeneratorSchema {
    /**
     * Current schema version, including portable static axis and NCHW window extraction,
     * unequal-rank movement geometry, exact represented padding bits, and scalar generated
     * bodies. Schema 14 adds mixed-type gather/one-hot structural identity and generated output
     * writers; execution-time index validation remains outside generated artifacts. Envelopes
     * written for earlier schemas are incompatible misses.
     */
    public static final int CURRENT_VERSION = 14;
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

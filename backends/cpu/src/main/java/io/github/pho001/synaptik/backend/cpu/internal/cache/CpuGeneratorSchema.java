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
 * Schema 19 adds the CPU-private V1 explicit-state initializer/dropout mapping, uniform and
 * finite-precision policy, baked raw initializer/probability bits, one- or five-boundary entry
 * shape, three-output stores, and workspace-free state prologue.
 * Schema 20 adds cumulative sum/product identity, axis and mode roles, sequential typed rounding,
 * and the two-boundary workspace-free independent-slice entry shape.
 * Schema 21 adds ordinary MIN/MAX/ALL/ANY form, canonical selected-axis membership, deterministic
 * floating selection policy, complete-output-cell ranges, zero workspace, and its direct bridge.
 * Schema 22 embeds typed scan and aggregate bodies and adds proved dense heap-array int-address
 * scalar and single-bound Vector loop forms.
 * Schema 23 adds proved dense heap-array integer affine-copy and movement bodies with hoisted
 * invocation geometry while retaining the general long-address forms.
 * Schema 24 embeds carrier-, type-, and family-specialized indexing bodies, including proved
 * dense heap-array integer-address forms and typed general long-address forms.
 * Schema 25 embeds carrier-, type-, family-, reduction-, and access-specialized functional
 * scatter output and contribution bodies while retaining the optional exact-product entry.
 * Schema 26 embeds carrier-, type-, family-, access-, mapping-, and addition-specialized overlap
 * fold output and contribution bodies with dense integer and general long-address forms.
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
     * Schema 19 adds explicit-state initialization and FLOAT64/FLOAT32 dropout identity and code;
     * schema 20 adds the five-type cumulative-scan family and slice-domain execution; schema 21
     * adds ordinary extrema and Boolean output-cell reductions through a generated bridge;
     * schema 22 embeds typed family bodies and proved dense heap-array int-address loop forms;
     * schema 23 adds integer affine-copy and movement bodies with hoisted invariant geometry;
     * schema 24 embeds typed GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT bodies with proved
     * dense heap-array integer-address forms and typed general long-address forms.
     * Schema 25 embeds typed functional scatter output, matching, and reduction bodies; schema 26
     * embeds typed overlap-fold output, coordinate-matching, and sequential-addition bodies.
     * Envelopes written for earlier schemas are incompatible misses.
     */
    public static final int CURRENT_VERSION = 26;
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

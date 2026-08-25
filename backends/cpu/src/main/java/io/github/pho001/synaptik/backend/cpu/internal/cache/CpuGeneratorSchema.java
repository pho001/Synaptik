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
 * Schema 27 embeds carrier-, represented-type-, family-, direction-, output-, and access-
 * specialized stable ordering bodies with direct two-region merge scratch access.
 * Schema 28 embeds typed explicit-state initializer and FLOAT64/FLOAT32 dropout state, mapping,
 * threshold, value, canonical-mask, and loop bodies without a generic execution bridge.
 * Schema 29 adds ordinary numerical aggregate identity, exact floating-state shapes, typed
 * scratch-bearing entries, and direct exact sum, mean, and product limb bodies.
 * Schema 30 makes covered scalar activation, BFLOAT16 scan arithmetic, and extrema/Boolean
 * aggregate combination bodies self-contained with respect to Synaptik runtime members while
 * retaining the intentional chunk-level vector-math boundary.
 * Schema 31 replaces scratch-free scatter's output-per-update grouping with range-owned base copy
 * followed by one canonical update traversal, while retaining grouped exact floating-product
 * bodies and their existing scratch shape.
 * Schema 32 hoists each required native-order typed segment layout into one invocation-local
 * reference before repeated scalar carrier access.
 * Schema 33 adds cold-proved bounded movement geometry and invocation-local cursor loops for PAD,
 * CONCAT, UNFOLD_AXIS, and UNFOLD2D while retaining the typed general-long fallback.
 * Schema 34 adds the cold-proved occurrence-major canonical-BOOL STACK copy and zero-stride ANY
 * fold while retaining their typed general-long fallbacks.
 * Schema 35 adds two guarded forms for the frozen A1G shapes: one mixed-carrier padded/dilated
 * FLOAT32 FOLD2D output-cell loop and one mixed-carrier rank-one FLOAT32 dropout loop. Both
 * preserve the optimal clean Java semantic algorithm and hot-loop dataflow, accept arbitrary
 * legal subranges, and retain their typed general-long fallbacks when the complete cold proof
 * does not hold.
 * Schema 36 adds guarded exact-state cursor forms for the frozen mixed-carrier FLOAT32 axis-one
 * MEAN with domain {@code 2048} and BFLOAT16 axes-zero-and-two PROD shapes. The guarded forms
 * preserve the existing run-owned state, arbitrary legal output-cell subranges, and typed
 * general-long fallbacks.
 * Schema 37 adds guarded primitive cursor loops for the frozen mixed-carrier FLOAT64 GATHER and
 * FLOAT32 GATHER_ND geometries. These forms hoist index tuple reads outside contiguous suffix
 * writes; the completely guarded GATHER_ND body emits its fixed suffix length of 16 as one
 * generation-time straight-line sequence inside the existing artifact rather than as another
 * planner-visible unrolled variant. Both forms accept arbitrary legal output subranges and retain
 * the typed general-long fallbacks.
 * Schema 38 adds one completely guarded fixed {@code [1024,1024]} axis-one exclusive reverse
 * INT64 cumulative-product body over two {@code MemorySegment} carriers. The proved body uses
 * direct unaligned long access and descending element cursors for arbitrary legal complete-slice
 * subranges; every unproved geometry retains the typed general-long fallback.
 * Schema 39 adds one fully geometry-, range-, and sentinel-guarded raw-BFLOAT16
 * segment-to-short-array affine cursor body for the frozen {@code [256,32,32]} PERMUTE/SLICE
 * mapping. The proved body uses ordinal shifts and masks with direct typed loads/stores for any
 * legal half-open range; every unproved mapping, carrier, layout, or range retains the typed
 * general-long fallback.
 * Schema 40 adds one completely guarded frozen FLOAT32 pointwise body for the
 * {@code [512,512]} mixed-carrier {@code DIV -> SIGMOID -> MUL} topology. The proved loop derives
 * row and column from an integer ordinal, uses direct unaligned segment loads and strided heap
 * stores, preserves the stable binary64-exponential/sign-branch/one-final-narrowing sigmoid, and
 * retains the typed general-long state machine for every failed proof.
 * Schema 41 adds one completely guarded frozen INT64 {@code SCATTER_ND + MIN} body for the
 * {@code [16384,16]} output and {@code [4096,16]} update geometry. The proved body emits the
 * direct primitive linear copy and tuple/suffix update loops, preserves duplicate encounter
 * order and arbitrary legal output subranges, and retains the typed general-long implementation
 * for every failed proof.
 * Schema 42 adds one completely guarded frozen BFLOAT16 axes-zero-and-two MIN body for input
 * {@code [64,64,64]} and kept output {@code [1,64,1]}. The proved body follows the direct
 * primitive nested-loop oracle, preserves canonical factor order, first represented NaN and
 * negative-zero selection, accepts arbitrary legal complete-output-cell subranges, and retains
 * the typed general-long implementation for every failed proof.
 * Schema 43 adds binding-aware SUM-to-Shape alignment identity, selected leading/aligned-axis
 * traversal, direct represented-bit no-reduction copies, and their exact resource/entry shapes.
 * Schema 44 adds one-axis ARG_MIN/ARG_MAX direct typed loops. Schema 45 adds directional
 * right-aligned masked SUM/MEAN identity, three typed boundaries, early canonical-mask branching,
 * invocation-local selected counts, and exact-state generated entries.
 * Schema 46 adds the five-kind advanced floating reduction family, ordered selected-axis and
 * correction identity, operation-specific pass shape, exact-state resource shape, and direct
 * typed logarithmic, statistical, and norm bodies.
 * The optional persistent envelope stores this version and has no legacy reader, migration path,
 * or converter. Schema 47 adds first-class stable softmax/log-softmax. Schema 48 adds the four
 * trailing Layer/RMS forms, ordered mixed-type boundaries, exact typed epsilon identity,
 * complete-slice pass shape, Layer-only exact-state scratch, and frozen typed segment layouts.
 * Schema 49 adds first-class five-input batch-normalization inference, arbitrary channel-axis
 * geometry, direct running-variance arithmetic, and selected channel/non-channel range identity.
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
     * embeds typed overlap-fold output, coordinate-matching, and sequential-addition bodies;
     * schema 27 embeds typed SORT, ARGSORT, and TOP_K stable-merge and output bodies; schema 28
     * embeds typed INITIAL_STATE and FLOAT64/FLOAT32 DROPOUT bodies; schema 29 adds exact ordinary
     * numerical aggregates and their run-owned state shape; schema 30 directly embeds the
     * remaining scalar activation, BFLOAT16 scan, and extrema/Boolean aggregate formulas; schema
     * 31 adds range-owned copy-then-update scatter bodies and retains grouped floating products;
     * schema 32 hoists required native-order typed segment layouts into invocation locals; schema
     * 33 adds cold-proved bounded cursor loops for selected general-address movement families;
     * schema 34 adds direct canonical-BOOL STACK occurrence copies and zero-stride ANY folds;
     * schema 35 adds guarded bounded cursor forms for the frozen mixed-carrier padded/dilated
     * FLOAT32 FOLD2D and rank-one explicit-state FLOAT32 dropout shapes while retaining their
     * typed general-long fallbacks; schema 36 adds guarded exact-state cursor forms for the
     * frozen mixed-carrier FLOAT32 axis-one MEAN and BFLOAT16 axes-zero-and-two PROD shapes while
     * preserving exact run-owned state and typed general-long fallbacks; schema 37 adds guarded
     * primitive cursor loops for the frozen mixed-carrier FLOAT64 GATHER and FLOAT32 GATHER_ND
     * geometries, including one guarded generation-time fixed 16-element GATHER_ND suffix body,
     * while retaining arbitrary legal subranges and typed general-long fallbacks; schema 38 adds
     * one completely guarded fixed {@code [1024,1024]} axis-one exclusive reverse INT64
     * cumulative-product segment-cursor body with direct unaligned long access and arbitrary
     * legal complete-slice subranges, while retaining the typed general-long fallback; schema 39
     * adds the guarded frozen raw-BFLOAT16 affine PERMUTE/SLICE cursor body; schema 40 adds the
     * guarded frozen FLOAT32 mixed-carrier pointwise ordinal loop while retaining the typed
     * general-long fallback; schema 41 adds the guarded frozen INT64 SCATTER_ND MIN direct
     * copy-then-update loops with the same typed fallback; schema 42 adds the guarded frozen
     * BFLOAT16 axes-zero-and-two MIN primitive nested loops with the same typed fallback; schema
     * 43 adds right-aligned SUM-to-Shape identity, represented-copy entries, reduction cursors,
     * and guarded dense primitive nested loops; schema 44 adds one-axis ARG_MIN/ARG_MAX direct
     * typed loops, including guarded unit-stride and stride-two forms with arbitrary-stride
     * fallback; schema 45 adds direct typed masked SUM/MEAN output-cell bodies with directional
     * mask broadcasting, early false exclusion, runtime selected counts, and exact-state scratch.
     * Schema 46 adds direct typed LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, and L2_NORM
     * complete-output-cell bodies plus their ordered-axis, correction, pass, and resource identity.
     * Schema 47 adds direct typed SOFTMAX and LOG_SOFTMAX complete-slice bodies, including their
     * operation kind, selected axis, finite-input contract, pass structure, and exact layout
     * identity. Schema 48 adds the four trailing Layer/RMS forms, ordered floating promotion,
     * exact epsilon, unique-boundary mapping, normalized geometry, pass/resource identity, and
     * direct typed bodies. Schema 49 adds first-class batch-normalization inference with five
     * ordered input types, ordered promotion result, exact epsilon bits, arbitrary channel axis,
     * unique-boundary map, channel/non-channel range form, zero-resource identity, and direct
     * running-variance body.
     * Envelopes written for earlier schemas are incompatible misses.
     */
    public static final int CURRENT_VERSION = 49;
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

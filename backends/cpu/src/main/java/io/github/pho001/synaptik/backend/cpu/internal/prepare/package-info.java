/**
 * Owns the unsupported CPU analysis and post-assignment finalization lifecycle.
 *
 * <p>Analysis consumes one complete CPU-owned planned partition, validates one connected
 * one-to-eight-occurrence pointwise chain, forms one computation unit, retains normalized access
 * bindings and the derived ordered type/carrier pattern, compares direct access
 * with at most three one-input contiguous-copy candidates from explicit dimensionless cold
 * evidence, selects scalar or an exactly eligible FLOAT32, FLOAT64, INT32, INT64, canonical-BOOL,
 * or unit-private floating-mask preferred-species vector body plus single-thread or bounded
 * parallel orchestration, and
 * declares each boundary's exact referenced storage span plus at most one workspace before shared
 * assignment. Finalization verifies all assignments and any required borrowed worker group before
 * one artifact-store call, realizes the already-selected
 * scalar or vector artifact, and constructs one partition executable. It must not reinterpret
 * graph semantics, change fusion, route, strategy, or species selection, or introduce an
 * undeclared resource after shared Prepare has assigned slots. Ordinary analysis and finalization
 * never benchmark; optional persistence only performs bounded verified lookup/store work.
 * The current schema-45 pointwise artifact may enter the completely guarded frozen schema-42
 * {@code [512,512]} FLOAT32 mixed-carrier ordinal loop for arbitrary legal half-open ranges.
 * It preserves FLOAT32 division and multiplication around the stable binary64-exponential,
 * sign-branch, one-final-narrowing sigmoid. Failed topology, geometry, range, address, or
 * sentinel proofs retain the same typed general-long state machine without changing analysis,
 * declarations, route selection, or Runtime binding.
 * Static affine plans continue to declare exactly the source and distinct output, with no
 * workspace or additional materialization. Their one current schema-45 artifact may enter the
 * completely guarded raw-BFLOAT16 segment-to-short-array PERMUTE/SLICE cursor body for arbitrary
 * legal half-open ranges; every failed geometry, range, or sentinel proof retains the typed
 * general-long body. This body changes neither affine lowering nor declarations, route choice,
 * finalization ownership, or Runtime binding.
 * Indexing plans declare unique inputs followed by one output, select scalar or parallel-scalar
 * output execution, declare no workspace, and retain compact validation/write geometry. Their
 * one generated artifact contains only the output-writing pass. Current schema 45 may select
 * guarded primitive cursor bodies for the frozen GATHER and GATHER_ND geometries while retaining
 * the same declarations, arbitrary legal subranges, and typed fallback inside that artifact.
 * Functional-scatter plans likewise declare unique inputs followed by one output and select
 * scalar or parallel-scalar output ownership. Complete bounds validation and any replacement-
 * target uniqueness validation occur before generated work. Floating multiplication alone
 * declares one exact per-range-sliced workspace; it is mutually exclusive with materialization.
 * Finalization verifies the exact scratch assignment and the scratch-bearing signature introduced
 * by schema 16 before artifact realization. The current schema-45 artifact may enter the
 * completely guarded frozen INT64
 * SCATTER_ND MIN direct copy-then-update body without changing declarations, validation,
 * orchestration, or zero-workspace ownership; failed proofs retain the typed general-long body.
 * Fold plans declare exactly one input and one output buffer, select scalar or parallel-scalar
 * disjoint output ranges, retain compact geometry, and declare no workspace or materialization.
 * Schema 17 introduced their generated compatibility; finalization realizes the current
 * schema-42 artifact, including only guarded cold-proved forms admitted by that artifact.
 * Ordering plans declare one input followed by one SORT/ARGSORT output or ordered TOP_K values
 * and INT64-index outputs. They select scalar or complete-slice parallel-scalar execution and
 * declare one exact run-owned workspace with disjoint two-region INT64 merge scratch per selected
 * range. Finalization verifies all three TOP_K bindings, the workspace assignment, and the
 * current schema-45 signature before realizing one multi-store artifact; schema 18 introduced
 * the ordering-family compatibility facts.
 * Explicit-state random plans declare one initializer output or five dropout buffers in exact
 * boundary order and no workspace. Parallel dropout reuses one scalar artifact over disjoint
 * logical ranges after complete cold overlap validation.
 * Cumulative-scan plans declare input then output, retain the independent slice count as their
 * execution domain, and select scalar or whole-slice parallel-scalar orchestration. They declare
 * no workspace or materialization, and finalization realizes one current schema-45 artifact that
 * embeds the typed scan body introduced by schema 22, may use the completely guarded fixed
 * {@code [1024,1024]} axis-one exclusive reverse INT64 product segment-cursor form for arbitrary
 * legal complete-slice subranges, and retains the scan compatibility identity introduced by
 * schema 20. The guarded form changes neither the two-buffer declaration nor the zero-workspace,
 * zero-materialization contract, and every unproved geometry uses the typed general fallback.
 * Ordinary aggregate plans likewise declare input then output and select scalar or complete-
 * output-cell parallel-scalar orchestration. They retain canonical selected-axis membership and
 * realize one current schema-45 artifact. Floating numerical rows declare exact run-owned
 * per-range state before assignment; integral numerical, extrema, and Boolean rows declare no
 * workspace. The guarded frozen BFLOAT16 multi-axis MIN body preserves the same zero-workspace
 * declaration, complete-output-cell ranges, and scalar or parallel-scalar orchestration. Guarded
 * aggregate bodies change neither declared workspace size nor range slicing and retain the typed
 * general fallback. No aggregate plan selects
 * materialization or partial/combine state. Schema 43 extends the same aggregate lifecycle with
 * fully bound SUM-to-Shape geometry, direct general and guarded dense mapping bodies, and a raw
 * represented-copy mode. It preserves output-cell-only orchestration; floating reductions reuse
 * exact per-range state, while integral reductions and copies declare zero workspace.
 * Arg-extrema plans remain separate from aggregate plans. They declare one numeric input and one
 * INT64 output, retain complete-output-cell geometry, select scalar or parallel-scalar ownership,
 * and realize one schema-44 workspace-free artifact without materialization or combine state.
 * Masked SUM/MEAN plans declare ordered data, mask, and output buffers plus one eight-byte-aligned
 * exact-state workspace when output cells exist. Scalar or parallel-scalar orchestration owns
 * complete output cells and disjoint per-range slices. Analysis selects no mask materialization,
 * selected-count workspace, partial state, or combine state; schema 45 finalization realizes the
 * direct typed three-boundary entry after shared slot assignment.
 * Advanced-reduction plans declare one input and one output, select scalar or complete-output-cell
 * parallel-scalar orchestration, and never materialize or split a selected domain. L1 and
 * statistics declare one maximum exact-state slice per selected range; log-sum-exp and L2 declare
 * no workspace. Schema 46 finalization realizes the corresponding typed entry.
 * Softmax plans declare exactly one input and one output, select scalar or complete-slice
 * parallel-scalar orchestration, and declare no workspace or materialization. Schema 47
 * finalization realizes the first-class kind-specialized three-pass typed entry.
 *
 * <p>All work in this package is cold-path work. Runtime collaborates only through the resulting
 * prepared executable and never receives the canonical kernel intermediate representation.
 */
package io.github.pho001.synaptik.backend.cpu.internal.prepare;

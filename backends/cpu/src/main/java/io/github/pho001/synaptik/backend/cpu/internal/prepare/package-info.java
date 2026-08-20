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
 * Indexing plans declare unique inputs followed by one output, select scalar or parallel-scalar
 * output execution, declare no workspace, and retain compact validation/write geometry. Their
 * one generated artifact contains only the output-writing pass.
 * Functional-scatter plans likewise declare unique inputs followed by one output and select
 * scalar or parallel-scalar output ownership. Complete bounds validation and any replacement-
 * target uniqueness validation occur before generated work. Floating multiplication alone
 * declares one exact per-range-sliced workspace; it is mutually exclusive with materialization.
 * Finalization verifies the exact scratch assignment and schema-16 signature before artifact
 * realization.
 * Fold plans declare exactly one input and one output buffer, select scalar or parallel-scalar
 * disjoint output ranges, retain compact geometry, and declare no workspace or materialization.
 * Schema 17 introduced their generated compatibility; finalization realizes the current
 * schema-35 artifact, including only guarded cold-proved forms admitted by that artifact.
 * Ordering plans declare one input followed by one SORT/ARGSORT output or ordered TOP_K values
 * and INT64-index outputs. They select scalar or complete-slice parallel-scalar execution and
 * declare one exact run-owned workspace with disjoint two-region INT64 merge scratch per selected
 * range. Finalization verifies all three TOP_K bindings, the workspace assignment, and the
 * current schema-35 signature before realizing one multi-store artifact; schema 18 introduced
 * the ordering-family compatibility facts.
 * Explicit-state random plans declare one initializer output or five dropout buffers in exact
 * boundary order and no workspace. Parallel dropout reuses one scalar artifact over disjoint
 * logical ranges after complete cold overlap validation.
 * Cumulative-scan plans declare input then output, retain the independent slice count as their
 * execution domain, and select scalar or whole-slice parallel-scalar orchestration. They declare
 * no workspace or materialization, and finalization realizes one current schema-35 artifact that
 * embeds the typed scan body introduced by schema 22 while retaining the scan compatibility
 * identity introduced by schema 20.
 * Ordinary aggregate plans likewise declare input then output and select scalar or complete-
 * output-cell parallel-scalar orchestration. They retain canonical selected-axis membership and
 * realize one current schema-35 artifact. Floating numerical rows declare exact run-owned
 * per-range state before assignment; integral numerical, extrema, and Boolean rows declare no
 * workspace. No aggregate plan selects materialization or partial/combine state.
 *
 * <p>All work in this package is cold-path work. Runtime collaborates only through the resulting
 * prepared executable and never receives the canonical kernel intermediate representation.
 */
package io.github.pho001.synaptik.backend.cpu.internal.prepare;

/**
 * Supplies explicit capability, configuration and lifecycle integration for the Metal backend.
 *
 * <p>Capability is an occurrence contract over original carriers, attributes, Shapes, layouts and
 * gradient roles, not a family-wide promise. FLOAT32 elementary/activation math, shaped and masked
 * reductions, arithmetic scatters, normalization/losses, attention and fixed recurrent forward
 * have focused native components. Existing exact movement, casts, predicates, ordering, pooling,
 * convolution and modular integral contractions remain separately bounded. FLOAT64 represented-bit
 * unary operations, stored comparisons/extrema and exact movement do not imply FLOAT64 arithmetic.
 * There is no software-FLOAT64 implementation, CPU execution dependency or fallback.
 *
 * <p>Homogeneous BFLOAT16/FLOAT16 arithmetic retains its original frozen predicate, checked only
 * after authenticating the original descriptors and scalar carriers. New FLOAT32 support is never
 * consulted through that proxy. Low EXP/SIGMOID remain positive-rank, canonical and no-gradient;
 * FLOAT32 EXP/SIGMOID preserve matching gradient metadata through the same primal operation.
 * Direct mixed-low arithmetic remains excluded; explicit casts establish its FLOAT32 boundary.
 * Low operations retain one fixed custom route and one final integer round-to-nearest/ties-even
 * narrowing. FLOAT64/low image folds and FOLD_AXIS retain non-overlap restrictions; FLOAT32 image
 * folds and arithmetic scatter use one output writer rather than floating atomics.
 *
 * <p>Each partition fixes its complete route before resource declaration. Java lowering, executable
 * preflight and independent native decoding validate the actual grammar and resource geometry.
 * Ordinary Java ingress reconstructs the existing Model operation and independently infers every
 * canonical output's carrier, Shape and gradient metadata; completed physical layouts have separate
 * checks. Producer/index roles are supported without requiring a boundary feed. Replacement bounds
 * and destination uniqueness remain distinct from legal duplicate arithmetic contributors. Signed
 * recurrent lengths are checked against the actual time extent before writes. Fixed recurrent
 * forward does not introduce fixed-family BPTT or change Compiler differentiation ownership.
 *
 * <p>General Scan carries an authenticated body, demand/activity facts and explicit resource plan.
 * Actual zero extents are retained in its scoped ordinary grammar: no fictitious batch, denominator
 * or contraction extent is substituted. Independent Java/native row and terminal provenance checks
 * precede creation. Native traversal invokes controlled ordinary kernels with guards before inactive
 * value/index loads; canonical sibling outputs have independent row/scope facts. No Java child
 * execution loop, time unrolling, Runtime interpreter or new executor is involved. Compiler's existing
 * ordinary generated adjoints and neutral activity protocol retain differentiation ownership.
 *
 * <p>Only compact materialized slots receive run-owned buffers. Targets are direct assigned buffers;
 * generated pointwise interiors remain virtual. Output-only attention declares its score/weight
 * scratch explicitly, while canonical weights are reused when present. Scan workspaces and numerical
 * scratch are invocation-owned, included in admission/trace byte facts and released with their run.
 * No executable retains mutable numerical workspace. Authenticated static broadcast reads prove the
 * remaining nonbroadcast axes injective; a zero stride never bypasses that proof. Empty source
 * publications use zero-aware Model element counts and read no payload, including huge empty Shapes.
 *
 * <p>The ABI remains version seven with thirteen exports. Node schema twenty-one uses the fixed
 * 124-byte header, type wires {@code 1..7}, ordinary operation wires {@code 1..115}, General Scan
 * wire {@code 116}, and its separate body/activity attribute grammar. Native authenticates each
 * fixed source component and their exact dependency-ordered assembled corpus before compilation;
 * Java carries the corresponding bounded source-size facts. The fixed corpus is 188,030 UTF-8
 * bytes; the separately authenticated frozen low source is unchanged. Workload, exact-policy,
 * candidate, compatibility, route-policy and session-codec identities are forty-one. Predecessor
 * images and checksummed predecessor decisions have no compatibility reader or migration alias.
 *
 * <p>The package-private exhaustive route catalog is cold descriptive metadata, not admission or
 * execution authority. Existing pointwise/anchor fusion preserves its exact source order and
 * original eligibility; low arithmetic gains no generated-fusion path. Configuration names one
 * explicit native bridge. Integration owns one context and contributes checked transfer/source
 * binders, preparation, session-local tuning and schedule resources to Engine's explicit builder.
 * There is no discovery, hidden selection, execution-time retry, telemetry policy or backend-global
 * integration. Optional tracing reports bounded structural and aggregate resource facts, never
 * payloads, pointers or shader source, and preserves backend failures under observer failures.
 */
package io.github.pho001.synaptik.backend.metal;

# Task 0013: MPSGraph FLOAT32 Cumulative Scans

## Status

Ready

This is the sole authorized Metal frontier from clean `main` at
`7b34a7a9352d05d8930d4463b286209dff9a6228`. Metal 0005/0008 are Complete; 0006–0007 and
0009–0012 are independently Blocked. No implementation or probe change from 0012 remains.

## Change class

Class C — this extends Metal capability, the private typed native schema, candidate identity,
whole-partition lowering, persistent MPSGraph resources, and public Engine execution while
preserving the existing architecture, ABI exports, FLOAT32 storage, and transfer contract.

## Goal

Add bounded forward-only, non-gradient `FLOAT32` `CumulativeScanKind.CUM_SUM` and `CUM_PROD`
through MPSGraph's dedicated cumulative selectors. Support every inclusive/exclusive and
forward/reverse combination along one normalized axis without reopening blocked aggregate
SUM/MEAN, scalar arithmetic, BOOL storage, extrema, or custom kernels.

## Inventory and scope decision

The current Model candidates are first-class and distinct:

- masked reduction consists only of two-input axis-removing floating `SUM`/`MEAN` with a
  broadcast BOOL mask; it depends on exact aggregate arithmetic blocked by 0007 and a BOOL
  storage/input path not delivered by blocked 0010;
- arg extrema consists of single-axis `ARG_MIN`/`ARG_MAX`, fixed INT64 results, FIRST/LAST logical
  index ties, NaN preference, negative-zero-below-positive-zero ordering, and non-empty selected
  axes. Installed `reductionArgMinimum...`/`reductionArgMaximum...` selectors have no tie-policy
  parameter and expose INT32 symbolic results on this SDK, so exact tie selection, widening, INT64
  storage, and publication require a separate probe and task; and
- cumulative scan consists exactly of `CUM_SUM` and `CUM_PROD` with
  `CumulativeScanAttrs(axis, exclusive, reverse)`. The installed SDK exposes matching
  `cumulativeSumWithTensor:axis:exclusive:reverse:name:` and
  `cumulativeProductWithTensor:axis:exclusive:reverse:name:` selectors.

0013 therefore selects the complete bounded FLOAT32 cumulative-scan family. Scans produce values,
not ties or indices and add no tie policy, index width, or INT32/INT64 storage. Their represented
storage remains FLOAT32. The probe locks NaN classification, signed zeros, subnormals, infinities,
exact positive-zero/positive-one exclusive identities, and logical placement before capability.

## Exact capability domain

Admit one or more scan occurrences mixed with existing 0005/0008 nodes when every scan has one
input and one output, exact `CumulativeScanAttrs`, `requiresGrad == false`, identical FLOAT32 type
and Shape, fully static positive rank `1..16`, canonical dense-contiguous zero-offset non-view
layouts, a valid normalized axis, and checked element/byte geometry. All four exclusive/reverse
combinations are in scope. Scan results may feed later supported FLOAT32 nodes and supported nodes
may feed scans; every maximal supported topology must lower as one typed MPSGraph program.

Capability and analysis fail closed for masked, ordinary, arg-extrema, advanced, statistical,
Boolean, recurrent, or any other reduction/scan family; scalar or zero/dynamic extents; gradient
eligibility; non-FLOAT32 types; wrong attrs, arity, Shape, layout, or axis; unchecked geometry;
malformed topology; and any backward/training claim.

## Mandatory comprehensive real-device probe and fail-closed gate

Before production edits, compile and run a disposable Apple-silicon probe at MPSGraph optimization
levels `0` and `1`, with reduced-precision fast math set to none, at least three independently
compiled executables per axis length and level, and at least eight executions per case. Use the two
exact cumulative selectors, production-style FLOAT32 placeholders, caller-supplied canary-filled
direct targets, feed/target permutation checks, raw input identity controls, and no host fallback.

For both kinds and all four mode combinations, exhaust the ordered two-value cross-product of a
fixed raw-bit corpus containing both zeros; positive/negative minimum and ordinary subnormals;
minimum normals; adjacent, equal, and opposite-sign ordinary values; positive/negative one and
two; maximum finite values; both infinities; and quiet/signaling NaNs of both signs with multiple
payloads. Also exhaust every permutation of representative three- and four-value domains covering
NaN position, duplicate values, opposite zeros, subnormal/zero, underflow/overflow, zero/infinity,
opposing infinities, and sign parity.

The conservative oracle is one logical-slice sequential FLOAT32 fold in the declared traversal
direction, initialized with represented `+0.0f` for CUM_SUM or `+1.0f` for CUM_PROD, with one
ordinary same-type IEEE addition or multiplication per visited value and one rounding after each
operation. Exclusive mode stores before applying the current value; inclusive mode applies then
stores. Require NaN classification only, but require exact raw bits for every non-NaN output,
including zeros, subnormals, infinities, and exclusive identities. This deliberately strict route
gate does not relax or rewrite Model semantics.

Additionally prove axes at the first, middle, and last positions; rank-one, rank-two, and
rank-sixteen Shapes; axis lengths `1`, `2`, `3`, `4`, and representative longer lengths; exact
Shape/layout; complete writes; untouched surrounding canaries; chains of both scan kinds with
existing 0005 nodes; permutations, repetition, concurrent runs, and executable/session
independence. If any selector, mode, NaN, zero, subnormal, infinity, exact-bit, axis, Shape, target,
canary, binding, permutation, repetition, concurrency, or independence gate fails, remove the
probe and mark 0013 Blocked. Do not narrow the corpus, modes, or domain; substitute aggregate or
pointwise decompositions; add a custom kernel/CPU fallback; introduce tolerance; or relax Model.

## Native schema, candidates, and lifecycle

Keep native ABI version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte node
schema from version 2 to version 3; preserve operation wires `1..10`; assign `11` to CUM_SUM and
`12` to CUM_PROD; and assign attribute wire `4` to cumulative scan. A scan record has one input,
`second_input = UINT32_MAX`, `attribute_count = 2`, the normalized axis in `axis`,
`attribute_values[0] = exclusive ? 1 : 0`, `attribute_values[1] = reverse ? 1 : 0`, `reserved = 0`,
and zero remaining cells. Java and native validation independently prove exact same-Shape FLOAT32
geometry, axis range, Boolean flag encoding, topology, and output ownership.

Bump workload-signature, exact-policy, candidate, compatibility, route-policy, and backend-local
codec versions from 3 to 4. Candidate/workload identity includes scan kind, normalized axis, both
mode flags, descriptors, schema/ABI, topology, splats, and live target; every old or malformed
decision fails closed. Existing custom singleton NEG remains exact; any partition containing a
scan uses MPSGraph.

Analysis declares every buffer and address workspace before assignment. Finalization
transactionally creates one reusable Shape-specialized executable. Cold binding validates exact
FLOAT32 representations and fixed address order; hot execution performs one synchronous native
submission without route, graph, operation, slot, map, or type lookup. Preserve direct FLOAT32
resident storage, current canonical-only CPU/Metal transfer, rollback, autorelease pools,
close/run leases, repeated/concurrent run isolation, and independent sessions. Add no BOOL,
INT32, INT64, scalar, rank-zero, or new cross-owner storage/materialization path.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — Metal owns MPSGraph lowering, schema, resources, and materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — consume current forward capture only; no scan adjoint or training coverage is claimed.
- [Tensor API](../../../../api/tensor-api.md#cumulative-scan-expressions) — preserve scan kind,
  normalized axis, mode, Shape/type, identity, and represented floating semantics.

If implementation requires a shared contract, public API, new dependency, ABI export/version,
new data type/storage path, aggregate/pointwise decomposition, custom kernel, or numerical
relaxation, stop and mark 0013 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model cumulative-scan semantics and Compiler forward
  capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not Metal 0006,
  0007, or 0009–0012, and not Model 0026.
- **Conflicts with:** every concurrent Metal capability, preparation, native schema,
  candidate/codec, materialization, or Engine Metal integration change.
- **Parallel group:** None.
- **Common base revision:** N/A; serial frontier verified on clean `main` at `7b34a7a9352d05d8930d4463b286209dff9a6228`.
- **Integration order:** mandatory probe → schema/geometry → capability/topology →
  candidates/lifecycle → Engine proof → documentation and independent review.
- **Integration validation:** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer, and
  candidate/codec types — exact scan domain, schema v3, topology, and invalidation.
- Metal finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — direct cumulative selectors, reusable resources, and unchanged FLOAT32 hot lifecycle.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability, schema, lifecycle, and real behavior proofs.

## Acceptance criteria

- The disposable probe first passes the complete corpus, both kinds, all modes, exact oracle,
  Shape/axis, target, canary, binding, repetition, concurrency, and independence gates.
- Capability/preparation admit exactly the bounded no-gradient FLOAT32 scan domain and preserve
  every existing route, affine restriction, storage/materialization, and transfer rule.
- ABI v4 retains thirteen exports; schema v3 retains 160-byte records, uses only wires `11`, `12`,
  and attribute `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover both kinds, all modes, axis/Shape boundaries, wrong attrs/types,
  topology, rollback, close, reuse, concurrency, and independent sessions.
- A Metal-only public Engine run composes both scans with existing nodes, publishes exact outputs,
  repeats/concurrently runs across sessions, and proves no CPU owner.
- No excluded reduction/scan, BOOL/index storage, rank-zero path, custom kernel,
  backward/training, transfer widening, relaxed math, FLOAT16/BFLOAT16, or performance claim is
  introduced. Independent Class C review approves.

## Validation

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Scan*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Scan*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one Ready Metal frontier,
thirteen exports, unchanged contracts/dependencies, no custom scan kernel or new storage path, and
no excluded-family or backward claim.

## Documentation and review impact

Update Metal Javadocs, package documentation, backend/native guides, targeted public scope text,
and glossary status only after evidence passes. Model/Tensor semantics and architecture contracts
remain unchanged. Independent Class C review inspects capability truth, probe gates, schema and
compatibility invalidation, lifecycle, Engine evidence, and all exclusions.

## Result

Empty until execution.

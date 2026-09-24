# Task 0012: MPSGraph FLOAT32 Extrema Reductions

## Status

Blocked

Metal 0008 is Complete; 0006, 0007, and 0009–0011 are independently Blocked. The mandatory
real-device gate below also blocked this task before production changes. Metal 0013 is the sole
authorized Metal frontier.

## Change class

Class C — this extends Metal capability, the private typed native schema, candidate identity,
whole-partition lowering, scalar publication, persistent MPSGraph resources, and public Engine
execution while preserving the existing architecture, ABI exports, storage types, and transfer
contract.

## Goal

Add bounded forward `FLOAT32` ordinary `AggregateReductionKind.MIN` and `MAX` through MPSGraph's
NaN-propagating reduction selectors. Support full, single-axis, and ordered multi-axis Model forms,
including keep-dimensions, rank-zero outputs, and the empty-axis identity form. Do not reopen
blocked SUM, MEAN, or SUM_TO_SHAPE and do not introduce custom kernels.

## Inventory and scope decision

The current Model aggregate inventory is `SUM`, `MEAN`, `PROD`, `MIN`, `MAX`, `ALL`, `ANY`,
`ARG_MIN`, `ARG_MAX`, `LOG_SUM_EXP`, `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM`.
Ordinary extrema use `NoOperationAttrs`, `AxisReductionAttrs`, or `MultiAxisReductionAttrs` and
have a complete non-accumulating represented-value contract: NaN propagates without a payload or
signaling promise, infinities use numeric order, `MIN` selects negative zero, and `MAX` selects
positive zero. This is independent of 0007's exact-real SUM/MEAN requirement.

The installed macOS SDK declares
`reductionMinimumPropagateNaNWithTensor:axes:name:` and
`reductionMaximumPropagateNaNWithTensor:axes:name:`. It also declares product and non-propagating
extrema selectors; only the two propagating selectors match the requested NaN policy closely
enough to probe. Product remains separate after 0011 exposed MPSGraph scalar arithmetic,
subnormal, and division failures. `LOG_SUM_EXP`, variance, standard deviation, and L1/L2 norms
contain sum/mean or blocked unary mathematics in their Model targets, and the installed headers do
not expose a direct independent selector for all five. Boolean and arg extrema require BOOL or
INT64 storage and are not part of this numerical slice.

## Exact capability domain

Admit one or more existing 0005/0008 nodes plus `MIN`/`MAX` occurrences with one input and one
output, identical `FLOAT32` type and `requiresGrad`, fully static positive input Shape, rank
`1..16`, canonical dense-contiguous zero-offset non-view layouts, exact inferred output Shape, and
checked element/byte geometry. Ordinary full form reduces axes `0..rank-1`; single-axis and
multi-axis forms retain normalized axes and exact keep-dimensions meaning. Output rank is `0..16`.
An empty multi-axis list is a typed identity/copy and must not invoke an accidental reduction.

Capability and analysis fail closed for SUM, MEAN, SUM_TO_SHAPE, PROD, masked, advanced,
statistical, Boolean, arg-extrema, and scan reductions; zero/dynamic input extents; scalar input;
non-FLOAT32 types; wrong attrs, arity, result metadata, or axis sets; arbitrary views; unchecked
geometry; malformed topology; and any backward/training claim.

## Mandatory exhaustive real-device probe and fail-closed gate

Before production edits, compile and run a disposable Apple-silicon probe at MPSGraph optimization
levels `0` and `1`, with reduced-precision fast math disabled, at least three independently compiled
executables per level, and at least eight executions per case. Use production-style FLOAT32
placeholders, the two exact NaN-propagating selectors, caller-supplied canary-filled direct targets,
feed/target permutation checks, and raw identity controls.

Exhaust the ordered two-value cross-product of a fixed raw-bit corpus containing both zeros,
positive/negative minimum and ordinary subnormals, minimum normals, adjacent and equal ordinary
finite values, maximum finite values, both infinities, and quiet/signaling NaNs of both signs with
multiple payloads. Also exhaust every permutation of representative three- and four-value domains
covering NaN position, duplicate extrema, opposite zeros, subnormal versus zero, finite/infinity,
and mixed signs. Check every case for both MIN and MAX. NaN results require classification only;
every uniquely fixed non-NaN result requires exact raw bits.

The probe must additionally prove full, single-axis, and ordered multi-axis results; both
keep-dimensions choices; empty-axis identity; rank-zero, rank-one, and rank-sixteen outputs; exact
Shape and represented layout; complete direct writes; untouched outside canaries; stable mixed
chains with existing 0005 nodes; permutations, repetition, and executable independence. If any
selector, NaN propagation, signed-zero, subnormal ordering, infinity, exact-bit, Shape, target,
canary, permutation, repetition, or independence gate fails, remove the probe and mark 0012
Blocked. Do not narrow the corpus, substitute a non-propagating selector, custom kernel, host/CPU
fallback, tolerance, or Model relaxation.

## Native schema, candidates, and lifecycle

Keep native ABI version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte node
schema from version 2 to version 3; preserve operation wires `1..10`; assign `11` and `12` to
reduction MIN and MAX and attribute wire `4` to normalized reduction axes. A reduction record has
one input, `second_input = UINT32_MAX`, `attribute_count = 0..16`, `axis = UINT32_MAX`,
`reserved = 0`, ordered normalized axes in `attribute_values`, and zero remaining cells. Java and
native validation independently prove axis uniqueness/range and exact output geometry. Full forms
lower to all axes; empty-axis forms lower to identity; non-empty selectors reshape only when needed
to realize exact keep-dimensions Shape.

Bump candidate, compatibility/workload, route-policy, and backend-local codec versions. Identity
includes typed Model form, ordered axes, keep-dimensions, descriptors, schema/ABI, topology, splats,
and live target; old decisions fail closed. Existing custom singleton NEG remains exact; any
partition containing extrema uses MPSGraph.

Analysis declares all buffers and workspace before assignment. Finalization transactionally creates
one reusable Shape-specialized executable. Cold binding validates representations; hot execution
performs one synchronous native submission. Add only the backend-local four-byte rank-zero FLOAT32
publication path; do not widen rank-`1..16` CPU/Metal transfer. Preserve rollback, autorelease-pool,
closure, repeated/concurrent run, and independent-session rules.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — Metal owns MPSGraph lowering, native schema, resources, and materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — forward capture is consumed without claiming the extrema gradient graph, which needs blocked
  predicate/selection capability.
- [Tensor API](../../../../api/tensor-api.md) — ordinary extrema attributes, Shapes, and exact
  represented-value semantics remain Model-owned.

If implementation requires a shared contract, public API, new dependency, ABI export/version,
cross-owner scalar transfer, or numerical relaxation, stop and mark 0012 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model ordinary-extrema semantics and Compiler forward
  capture; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not Metal 0006,
  0007, or 0009–0011, and not Model 0026.
- **Conflicts with:** every concurrent Metal capability, preparation, native schema,
  candidate/codec, materialization, or Engine Metal integration change.
- **Parallel group:** None.
- **Common base revision:** N/A; serial frontier prepared from clean `ae3c74c4b97138bd6b0a196ff295654408f518ff`.
- **Integration order:** mandatory probe → schema/geometry → capability/topology →
  candidates/lifecycle/scalar publication → Engine proof → documentation and independent review.
- **Integration validation:** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer and candidate/codec
  types — exact extrema domain, schema v3, geometry, topology, and invalidation.
- Metal finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — propagating selectors, rank-zero publication, reusable resource, and unchanged hot lifecycle.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability, schema, lifecycle, and real behavior proofs.

## Acceptance criteria

- Capability/preparation admit exactly the bounded MIN/MAX domain and preserve every existing route,
  affine restriction, FLOAT32 storage/publication, and FLOAT32-only cross-owner transfer.
- The disposable probe passes the complete raw corpus, pair cross-product, required permutations,
  exact-bit/special-value, Shape, target, canary, repetition, and independence gates first.
- ABI v4 retains thirteen exports; schema v3 retains 160-byte records, uses only wires `11`, `12`,
  and attribute `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover both kinds/forms, axis/Shape boundaries, identity, wrong attrs/types,
  topology, rollback, close, reuse, concurrency, rank-zero publication, and independent sessions.
- A Metal-only public Engine run composes caller-input extrema with existing 0005 nodes, publishes
  direct rank-zero and positive-rank exact outputs, repeats across sessions, and proves no CPU owner.
- No excluded reduction, custom kernel, backward/training, transfer widening, relaxed math,
  FLOAT16/BFLOAT16, or performance claim is introduced. Independent Class C review approves.

## Validation

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Reduction*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Reduction*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one Ready Metal frontier,
thirteen exports, unchanged contracts/dependencies, no custom extrema kernel or scalar cross-owner
transfer, and no excluded-reduction or backward claim.

## Documentation and review impact

Update Metal Javadocs, package documentation, backend/native guides, targeted public scope text, and
glossary status only after evidence passes. Model/Tensor semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, numerical gates, schema and
compatibility invalidation, rank-zero lifecycle, Engine evidence, and all exclusions.

## Result

- The mandatory Apple M3 Max probe ran the exact NaN-propagating MIN/MAX selectors at optimization
  levels 0 and 1 with reduced-precision fast math disabled, three independently compiled
  executables per length and level, eight runs per case, and 43,824 total executions.
- For `[+0,-0]`, both selectors returned `-0`: MIN passed and MAX failed in all 48 observations.
  For `[-0,+0]`, both returned `+0`: MAX passed and MIN failed in all 48 observations. Three of
  four representative zero permutations were order-dependent.
- For `[+0,+minimum-subnormal]`, both selectors returned the subnormal, so MIN failed; reversing
  the pair made both return `+0`, so MAX failed. The matrix recorded 2,016 MIN mismatches, 2,016
  MAX mismatches, and 2,976 subnormal failures.
- NaN classification, minimum normals, identity controls, direct targets, canaries, permutations,
  repetition, and bindings passed, but they do not remove the signed-zero/subnormal blocker.
- Independent review returned `APPROVE-BLOCKER`. The disposable probe and all production, native,
  and test changes were removed. Unblocking requires an exact replacement route or explicit Model
  numerical-contract change; custom kernels and relaxation remain unauthorized here.
- Status: Incomplete
- Follow-up required: leave 0012 Blocked until an exact replacement or approved Model change exists.

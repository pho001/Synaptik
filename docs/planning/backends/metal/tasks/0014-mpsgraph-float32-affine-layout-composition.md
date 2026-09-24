# Task 0014: MPSGraph FLOAT32 Affine Layout Composition

## Status
Ready
This is the sole authorized Metal frontier from clean `main` at
`e554d9da5656e8fef24c681d45cda2eb1766a76e`. Metal 0005/0008 are Complete; 0006–0007 and
0009–0013 are independently Blocked. It does not depend on any blocked numerical operation.

## Change class
Class C — this extends Metal occurrence capability, whole-partition topology, the private native
schema and candidate identity, persistent MPSGraph resources, backend-private dense represented
order, and public Engine execution without changing shared architecture, public API, or transfer.

## Goal
Remove 0008's terminal-affine-leaf restriction for a bounded forward FLOAT32 domain. Compose the
five existing affine operations (`RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`) when
each logical input layout is exact and locally authenticated, and add the existing Model
`CONTIGUOUS` request as an explicit canonicalization barrier. Preserve logical descriptors while
MPSGraph carries dense represented-order symbolic tensors; do not claim source/result storage
aliasing.

## Contract and feasibility readout
Model `LayoutDescriptor` is logical geometry, not storage. Compiler `LayoutInference` propagates
exact strides/offset/view state through affine operations, resolves `RESHAPE` only from contiguous
input geometry, and gives static `CONTIGUOUS` exact canonical non-view geometry. The installed SDK
has the five documented selectors already used by 0008. `CONTIGUOUS` needs no undocumented
selector: `reshapeTensor:withShape:name:` to the unchanged exact Shape preserves logical row-major
values and yields a canonical direct target, subject to the mandatory real probe.

`OperationCapabilityQuery` intentionally contains one occurrence and descriptors but no graph
closure. Capability may therefore recognize an exact resolved view input; backend analysis must
then authenticate that every such input is an earlier value in the same maximal Metal partition.
Graph feeds remain canonical. This uses existing backend-analysis ownership and needs no shared
contract. If implementation cannot enforce that local-origin rule without changing Planning,
Compiler, Prepare, Runtime, or Engine contracts, mark 0014 Blocked and expose no Ready Metal
frontier.

## Exact capability and topology
All admitted values are FLOAT32, fully static, strictly positive, rank `1..16`, zero-offset, and
have checked logical element/byte geometry. Inputs and outputs retain one equal `requiresGrad`
flag. New execution evidence is forward-only; no Metal-only backward/training claim is added.

- Preserve the complete 0005 domain and 0008's five canonical-input terminal leaves.
- An affine node has the exact typed attributes and output Shape/layout produced by current Model
  and Compiler inference. Its input may be canonical or an exact resolved zero-offset view produced
  by an earlier admitted affine node in the same partition. Unresolved layouts reject. In
  particular, a non-contiguous view cannot feed `RESHAPE` unless `CONTIGUOUS` first makes its
  logical geometry canonical.
- `CONTIGUOUS` has exactly one input/output, `NoOperationAttrs.INSTANCE`, identical Shape/type and
  gradient eligibility, any exact canonical or locally produced resolved-view input, and exact
  `LayoutDescriptor.contiguous(shape)` output. It lowers to same-Shape MPSGraph reshape.
- A view value may feed another affine node or `CONTIGUOUS`; it may not feed existing NEG/binary
  nodes directly. A `CONTIGUOUS` result is canonical and may feed existing 0005 nodes or another
  affine node. Canonical graph feeds and existing canonical splats remain unchanged.
- One maximal supported topology lowers as one typed MPSGraph program. Stable fan-out, multiple
  publications, and a published intermediate that also feeds later supported nodes are allowed.
  Every noncanonical graph feed, foreign-partition view input, malformed order, unresolved result,
  or affine-to-elementwise edge lacking `CONTIGUOUS` fails before native resource creation.

Exclude slice/select/pad/tile/concat/stack/window transforms, arbitrary materialization, offset or
dynamic layouts, rank zero, zero extents, new types, blocked arithmetic/reductions/scans, custom
kernels, CPU fallback, implicit inserted nodes, and backward execution.

## Mandatory real-device probe
Before production edits, compile and run disposable Apple-silicon programs at MPSGraph optimization
levels `0` and `1`, reduced-precision fast math none, three independently compiled executables per
topology/level, and at least eight executions per executable. Bind caller-supplied canary-filled
direct buffers to every requested intermediate/final target, permute feed/target order, and retain
raw identity controls. No host rewrite or fallback is permitted.

Probe at least these exact topology families:
1. `[6] RESHAPE [2,1,3] -> EXPAND [2,4,3] -> PERMUTE [1,0,2] -> EXPAND_DIMS axis 2 -> SQUEEZE axis 2`, publishing every stage and the final zero-stride view;
2. `[2,3,4] PERMUTE [2,0,1] -> CONTIGUOUS -> NEG -> RESHAPE [4,6]`, proving the explicit view-to-canonical barrier before an existing operation;
3. identity/inverse reshape, permutation, and rank-edit chains; view fan-out into distinct affine
   and contiguous consumers; rank-1 and rank-16 boundaries; and intermediate-plus-final targets.

Exhaust a fixed raw-bit corpus in every logical position and repeated coordinate: both zeros;
positive/negative minimum and ordinary subnormals; minimum normals; adjacent and opposite ordinary
values; maximum finite values; both infinities; and quiet/signaling NaNs of both signs with multiple
payloads. Require exact 32-bit equality for every output, including NaN payload/sign, according to
an independent logical-coordinate oracle. Also require exact Shapes, direct-target dense canonical
represented order, full writes, untouched surrounding canaries, correct duplicate placement for
`EXPAND`, feed/target binding independence, repetition, concurrency, and independent sessions.

If any selector chain, same-Shape reshape used for `CONTIGUOUS`, raw bit, Shape/order, intermediate
target, canary, binding, repetition, concurrency, or independence gate fails, remove the probe and
mark 0014 Blocked. Do not narrow the topology/corpus or add an undocumented selector, arithmetic
identity, custom kernel, CPU copy, tolerance, or descriptor rewrite.
## Schema, identity, and representation
Keep native ABI version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte node
schema from version 2 to version 3, preserving wires `1..10`; assign operation wire `11` to
`CONTIGUOUS` with attribute kind `NONE`, one input, `second_input = UINT32_MAX`, zero attribute
count/payload/reserved cells, and exact same input/output Shape. Java and native validation use
explicit value states: canonical feed/output, affine view output, and unavailable. Affine nodes may
consume either prior state; `CONTIGUOUS` produces canonical state; existing elementwise nodes still
reject view state.

Bump workload-signature, exact-policy, candidate, compatibility, route-policy, and codec versions
from 3 to 4. Identity includes every kind and typed attr, exact descriptors/layouts, ordered edges,
value states, target set, dense byte geometry, schema/ABI, splats, and live target. Stale decisions
fail closed.

An affine publication receives full logical-size dense represented-order storage only when the
exact finalized MPSGraph plan authenticates its value ID, producer kind, complete validated
composition, descriptor, target position, byte extent, context, and executable. A canonical
`CONTIGUOUS` target uses the ordinary canonical materialization path. Intermediate symbolic tensors
make no allocation or alias promise. Do not widen canonical-only cross-owner transfer, accept a
view as caller ingress, rewrite logical view metadata, share Runtime representation objects, or
claim zero-copy execution.

## Lifecycle
Analysis validates the complete partition and local view provenance, fixes ordered values/nodes/
feeds/targets, declares every buffer and address workspace, and selects MPSGraph for every composed
partition. Finalization transactionally creates one reusable Shape-specialized executable. Cold
binding checks exact FLOAT32 representations and target authentication once; hot execution performs
one synchronous submission with no route, graph, node, map, layout, or type lookup. Preserve
rollback, context leases, autorelease pools, close rejection, repeated/concurrent run isolation,
and independent sessions.

## Contracts
- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal analysis owns lowering/resources and Runtime executes prepared work.
- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) — Metal
  owns MPSGraph lowering, storage, native integration, and Metal-specific materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  analysis precedes assignment/finalization; cold binding precedes direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — inspect current forward capture only; add no derivative rule or backward claim.
- [Tensor API — Contiguous expressions](../../../../api/tensor-api.md#contiguous-expressions),
  [Reshape expressions](../../../../api/tensor-api.md#reshape-expressions), and
  [Resolved layouts](../../../../api/tensor-api.md#resolved-layouts) — preserve current semantic and
  logical-layout meaning.

If implementation needs a shared contract/API, descriptor semantic change, new dependency, ABI
export/version, transfer widening, alias guarantee, or non-MPSGraph route, stop and mark 0014
Blocked.

## Dependencies and integration
- **Depends on:** Metal 0005/0008; current Model layout semantics and Compiler inference; Engine
  0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not Metal 0006–0007, 0009–0013, or
  Model 0026.
- **Conflicts with:** every concurrent Metal capability, preparation, native schema, candidate/
  codec, materialization, or Engine Metal integration change.
- **Parallel group:** None.
- **Common base revision:** N/A; serial frontier verified on clean `main` at
  `e554d9da5656e8fef24c681d45cda2eb1766a76e`.
- **Integration order:** mandatory probe -> capability/topology -> schema/state validation ->
  candidate invalidation -> lifecycle/authentication -> Engine proof -> docs/review.
- **Integration validation:** native export/schema audit; focused Metal, conformance, Compiler,
  real Engine, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols
- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  and native foundation — composed layouts, `CONTIGUOUS`, schema v3, and fail-closed topology.
- Preparation/finalization/executable/buffer/schedule and candidate/codec types — exact dense-target
  authentication, version invalidation, declarations, binding, and lifecycle.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides/Javadocs — exact
  layouts, raw bits, public execution, exclusions, and no transfer widening.

## Acceptance criteria
- The disposable probe passes every topology, full corpus, direct target, Shape/order, canary,
  binding, repetition, concurrency, and independence gate before capability changes.
- Capability and analysis admit only the stated local affine/contiguous compositions; view feeds,
  unresolved geometry, and affine-to-elementwise edges without `CONTIGUOUS` fail closed.
- ABI v4 retains thirteen exports; schema v3 retains 160-byte records and exact wire/state rules;
  all versioned decisions invalidate.
- Fake/native tests cover all positive topologies and malformed attrs, states, order, geometry,
  target authentication, rollback, close, reuse, concurrency, and independent sessions.
- A Metal-only public Engine run publishes intermediate/final views and canonical contiguous
  results with exact raw bits, then consumes `CONTIGUOUS` through existing NEG; no CPU owner exists.
- No shared contract, alias promise, transfer/ingress widening, implicit node, custom kernel,
  blocked numerical operation, backward/training, FLOAT16/BFLOAT16, or performance claim is added.
  Independent Class C review approves.

## Validation
```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Affine*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
./gradlew :modules:compiler:test --tests '*Layout*' --tests '*GraphCompiler*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one Ready Metal frontier,
exactly thirteen exports, unchanged dependencies/contracts, and no broadened transfer or alias text.

## Documentation and review impact
Update Metal/native Javadocs and guides plus targeted Tensor/Compiler status wording only after the
probe passes. Model semantics and architecture contracts remain unchanged. Independent Class C
review covers capability truth, local view authentication, exact bits/layouts, schema/candidates,
lifecycle, Engine evidence, and exclusions.

## Result
Empty until execution.

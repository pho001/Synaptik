# Task 0014: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) Affine [Layout](../../../../glossary.md#layout) Composition

## Reader context

This extended terminal view results into authenticated, partition-local compositions. A
LayoutDescriptor described logical indexing; Metal's published target still contained separate
dense represented-order values, not an alias of the source. “Local” meant that analysis proved
the view's producer inside this partition, not merely that a matching descriptor existed.

Occurrence capability checked eligible geometry; whole-partition analysis then checked those
producer relationships. The explicit CONTIGUOUS operation made canonical logical order available
before a consumer such as NEG. It was not an instruction for Runtime to insert a hidden copy or
choose a new route.
The [current lowering and source-geometry explanation](../../../../explanation/packages/metal.md#2-lower-once-declare-resources-then-compile)
shows the continuing distinction between logical descriptions and physically materialized buffers.

## Status
Complete

Implementation [commit](../../../../glossary.md#term-commit) `01e81be217fb89bdaeedb60a939bb8a734669a24` passed the task's
[validation](../../../../glossary.md#term-validation) and an independent Class C re-review with `APPROVE` and zero findings. [Metal](../../../../glossary.md#term-metal) 0005's
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) capability remains withdrawn, and 0006–0007 plus 0009–0013 remain independently Blocked.

## Change class
Class C — this extends [Metal](../../../../glossary.md#term-metal) [occurrence](../../../../glossary.md#term-occurrence) capability, whole-[partition](../../../../glossary.md#partition) [topology](../../../../glossary.md#term-topology), the private native
[schema](../../../../glossary.md#term-schema) and candidate [identity](../../../../glossary.md#term-identity-meanings), persistent [MPSGraph](../../../../glossary.md#term-mpsgraph) resources, [backend](../../../../glossary.md#backend)-private dense represented
order, and public [Engine](../../../../glossary.md#term-engine) execution without changing shared architecture, public [API](../../../../glossary.md#term-application-programming-interface-api), or transfer.

## Goal
Remove 0008's terminal-affine-[leaf](../../../../glossary.md#term-leaf) restriction for a bounded forward [FLOAT32](../../../../glossary.md#term-float32) domain. Compose the
five existing affine operations (`RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`) when
each logical input [layout](../../../../glossary.md#layout) is exact and locally authenticated, and add the existing [Model](../../../../glossary.md#term-model)
[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) request as an explicit [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) barrier. Preserve logical [descriptors](../../../../glossary.md#term-descriptor) while
[MPSGraph](../../../../glossary.md#term-mpsgraph) carries dense represented-order symbolic [tensors](../../../../glossary.md#tensor); do not claim source/result storage
[aliasing](../../../../glossary.md#term-aliasing).

## Contract and feasibility readout
[Model](../../../../glossary.md#term-model) `LayoutDescriptor` is logical geometry, not storage. [Compiler](../../../../glossary.md#term-compiler) `LayoutInference` propagates
exact [strides](../../../../glossary.md#term-stride)/offset/[view](../../../../glossary.md#term-view-meanings) state through affine operations, resolves `RESHAPE` only from [contiguous](../../../../glossary.md#term-contiguous)
input geometry, and gives static `CONTIGUOUS` exact [canonical](../../../../glossary.md#term-canonical-form) non-view geometry. The installed [SDK](../../../../glossary.md#term-sdk)
has the five documented selectors already used by 0008. `CONTIGUOUS` needs no undocumented
selector: `reshapeTensor:withShape:name:` to the unchanged exact [Shape](../../../../glossary.md#shape) preserves logical [row-major](../../../../glossary.md#term-row-major)
values and yields a canonical direct [target](../../../../glossary.md#term-target), subject to the mandatory real probe.

[`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) intentionally contains one [occurrence](../../../../glossary.md#term-occurrence) and [descriptors](../../../../glossary.md#term-descriptor) but no [graph](../../../../glossary.md#graph)
closure. Capability may therefore recognize an exact resolved view input; [backend](../../../../glossary.md#backend) analysis must
then authenticate that every such input is an earlier value in the same maximal [Metal](../../../../glossary.md#term-metal) [partition](../../../../glossary.md#partition).
Graph feeds remain canonical. This uses existing backend-analysis [ownership](../../../../glossary.md#term-ownership-meanings) and needs no shared
contract. If implementation cannot enforce that local-origin rule without changing [Planning](../../../../glossary.md#term-planning),
Compiler, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), or [Engine](../../../../glossary.md#term-engine) contracts, mark 0014 Blocked and expose no [Ready](../../../../glossary.md#term-ready) Metal
frontier.

## Exact capability and [topology](../../../../glossary.md#term-topology)
All admitted values are [FLOAT32](../../../../glossary.md#term-float32), fully static, strictly positive, [rank](../../../../glossary.md#term-rank) `1..16`, zero-offset, and
have checked logical [element](../../../../glossary.md#term-element)/byte geometry. Inputs and outputs retain one equal `requiresGrad`
flag. New execution evidence is forward-only; no [Metal](../../../../glossary.md#term-metal)-only backward/training claim is added.

- Preserve the retained NEG domain and 0008's five [canonical](../../../../glossary.md#term-canonical-form)-input terminal leaves.
- An affine node has the exact typed attributes and output Shape/layout produced by current Model
  and Compiler inference. Its input may be canonical or an exact resolved zero-offset view produced
  by an earlier admitted affine node in the same partition. Unresolved layouts reject. In
  particular, a non-contiguous view cannot feed <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [partition glossary](../../../../glossary.md#partition))</span>`RESHAPE` unless [`CONTIGUOUS`](../../../../glossary.md#term-contiguous) first makes its
  logical geometry canonical.
- `CONTIGUOUS` has exactly one input/output, `NoOperationAttrs.INSTANCE`, identical Shape/type and
  gradient eligibility, any exact canonical or locally produced resolved-view input, and exact<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
  `LayoutDescriptor.contiguous(shape)` output. It lowers to same-[Shape](../../../../glossary.md#shape) [MPSGraph](../../../../glossary.md#term-mpsgraph) reshape.
- A [view](../../../../glossary.md#term-view-meanings) value may feed another affine [node](../../../../glossary.md#node) or `CONTIGUOUS`; it may not feed `NEG` directly. A
  `CONTIGUOUS` result is canonical and may feed `NEG` or another affine node. Canonical graph
  feeds and existing canonical splats remain unchanged.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- One maximal supported topology lowers as one typed MPSGraph program. Stable fan-out, multiple
  publications, and a published intermediate that also feeds later supported nodes are allowed.
  Every noncanonical graph feed, foreign-partition view input, malformed order, unresolved result,
  or affine-to-NEG edge lacking <span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>`CONTIGUOUS` fails before native resource creation.

Exclude [slice](../../../../glossary.md#slice)/select/pad/tile/concat/stack/window transforms, arbitrary [materialization](../../../../glossary.md#materialization), offset or
dynamic [layouts](../../../../glossary.md#layout), rank zero, zero extents, new types, blocked arithmetic/reductions/scans, custom
[kernels](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu) fallback, implicit inserted nodes, and backward execution.

## Mandatory real-[device](../../../../glossary.md#term-device) probe
Before production edits, [compile](../../../../glossary.md#compile) and [run](../../../../glossary.md#term-run) disposable Apple-silicon programs at [MPSGraph](../../../../glossary.md#term-mpsgraph) optimization
levels `0` and `1`, reduced-[precision](../../../../glossary.md#term-precision) fast math none, three independently compiled executables per
[topology](../../../../glossary.md#term-topology)/level, and at least eight executions per executable. Bind caller-supplied [canary](../../../../glossary.md#term-canary)-filled
direct [buffers](../../../../glossary.md#term-buffer) to every requested intermediate/final [target](../../../../glossary.md#term-target), permute feed/target order, and retain
raw [identity](../../../../glossary.md#term-identity-meanings) controls. No [host](../../../../glossary.md#term-host) rewrite or fallback is permitted.

Probe at least these exact topology families:
1. `[6] RESHAPE [2,1,3] -> EXPAND [2,4,3] -> PERMUTE [1,0,2] -> EXPAND_DIMS axis 2 -> SQUEEZE axis 2`, publishing every stage and the final zero-[stride](../../../../glossary.md#term-stride) [view](../../../../glossary.md#term-view-meanings);
2. `[2,3,4] PERMUTE [2,0,1] -> CONTIGUOUS -> NEG -> RESHAPE [4,6]`, proving the explicit view-to-[canonical](../../../../glossary.md#term-canonical-form) barrier before an existing [operation](../../../../glossary.md#operation);
3. identity/inverse reshape, permutation, and rank-edit chains; view fan-out into distinct affine
   and contiguous consumers; rank-1 and rank-16 boundaries; and intermediate-plus-final targets.<span data-glossary-navigation="true"> ([permutation glossary](../../../../glossary.md#permutation); [rank glossary](../../../../glossary.md#term-rank); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>

No [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic operation is part of the probe or preservation claim.

Exhaust a fixed raw-bit corpus in every logical position and repeated [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix): both zeros;
positive/negative minimum and ordinary [subnormals](../../../../glossary.md#term-subnormal); minimum normals; adjacent and opposite ordinary
values; maximum [finite](../../../../glossary.md#term-finite) values; both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero); and quiet/signaling NaNs of both signs with multiple
payloads. Require exact 32-bit equality for every output, including [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign, according to
an independent logical-coordinate [oracle](../../../../glossary.md#term-test-oracle). Also require exact [Shapes](../../../../glossary.md#shape), direct-target dense canonical
represented order, full writes, untouched surrounding canaries, correct duplicate placement for
`EXPAND`, feed/target [binding](../../../../glossary.md#term-binding) independence, repetition, [concurrency](../../../../glossary.md#term-concurrency), and independent sessions.

If any selector chain, same-Shape reshape used for [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), raw bit, Shape/order, intermediate
target, canary, binding, repetition, concurrency, or independence gate fails, remove the probe and
mark 0014 Blocked. Do not narrow the topology/corpus or add an undocumented selector, arithmetic
identity, custom [kernel](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu) copy, [tolerance](../../../../glossary.md#term-tolerance), or [descriptor](../../../../glossary.md#term-descriptor) rewrite.
## [Schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and representation
Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte [node](../../../../glossary.md#node)
schema from remediated version 3 to version 4. Preserve live wires `NEG=1` and affine `6..10`;
withdrawn [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) wires `2..5` remain unassigned and rejected. Assign [operation](../../../../glossary.md#operation) wire `11` to
[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) with attribute kind `NONE`, one input, `second_input = UINT32_MAX`, zero attribute
count/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)/reserved cells, and exact same input/output [Shape](../../../../glossary.md#shape). Java and native [validation](../../../../glossary.md#term-validation) use
explicit value states: [canonical](../../../../glossary.md#term-canonical-form) feed/output, affine [view](../../../../glossary.md#term-view-meanings) output, and unavailable. Affine nodes may
consume either prior state; `CONTIGUOUS` produces canonical state; `NEG` still rejects view state.

Bump workload-[signature](../../../../glossary.md#term-signature), exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) versions
from 3 to 4. Identity includes every kind and typed attr, exact [descriptors](../../../../glossary.md#term-descriptor)/[layouts](../../../../glossary.md#layout), ordered edges,
value states, [target](../../../../glossary.md#term-target) set, dense byte geometry, schema/ABI, splats, and live target. Stale decisions
[fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

An affine [publication](../../../../glossary.md#term-publication) receives full logical-size dense represented-order storage only when the
exact finalized [MPSGraph](../../../../glossary.md#term-mpsgraph) plan authenticates its value ID, [producer](../../../../glossary.md#term-producer) kind, complete validated
composition, descriptor, target position, byte [extent](../../../../glossary.md#term-extent), context, and executable. A canonical
`CONTIGUOUS` target uses the ordinary canonical [materialization](../../../../glossary.md#materialization) path. Intermediate symbolic [tensors](../../../../glossary.md#tensor)
make no allocation or [alias](../../../../glossary.md#aliasing-and-arena) promise. Do not widen canonical-only cross-[owner](../../../../glossary.md#term-owner) transfer, accept a
view as caller ingress, rewrite logical view [metadata](../../../../glossary.md#term-metadata), share [Runtime](../../../../glossary.md#term-runtime) representation objects, or
claim zero-copy execution.

## [Lifecycle](../../../../glossary.md#lifecycle)
Analysis validates the complete [partition](../../../../glossary.md#partition) and local [view](../../../../glossary.md#term-view-meanings) [provenance](../../../../glossary.md#provenance), fixes ordered values/[nodes](../../../../glossary.md#node)/
feeds/[targets](../../../../glossary.md#term-target), declares every [buffer](../../../../glossary.md#term-buffer) and address workspace, and selects [MPSGraph](../../../../glossary.md#term-mpsgraph) for every composed
partition. Finalization transactionally creates one reusable [Shape](../../../../glossary.md#shape)-specialized executable. Cold
[binding](../../../../glossary.md#term-binding) checks exact [FLOAT32](../../../../glossary.md#term-float32) representations and target [authentication](../../../../glossary.md#term-authentication) once; hot execution performs
one [synchronous](../../../../glossary.md#term-synchronous) submission with no route, [graph](../../../../glossary.md#graph), node, map, [layout](../../../../glossary.md#layout), or type [lookup](../../../../glossary.md#term-lookup). Preserve
[rollback](../../../../glossary.md#term-rollback), context [leases](../../../../glossary.md#term-lease), autorelease pools, close rejection, repeated/concurrent [run](../../../../glossary.md#term-run) isolation,
and independent sessions.

## Contracts
- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal analysis owns lowering/resources and Runtime executes prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Metal glossary](../../../../glossary.md#term-metal); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span> — Metal
  owns MPSGraph lowering, storage, native integration, and Metal-specific materialization.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  analysis precedes assignment/finalization; cold binding precedes direct hot execution.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  — inspect current forward capture only; add no derivative rule or backward claim.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- [Tensor API — Contiguous expressions](../../../../api/tensor-api.md#contiguous-expressions)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Contiguous glossary](../../../../glossary.md#term-contiguous))</span>,
  [Reshape expressions](../../../../api/tensor-api.md#reshape-expressions), and
  [Resolved layouts](../../../../api/tensor-api.md#resolved-layouts)<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span> — preserve current semantic and
  logical-layout meaning.

If implementation needs a shared contract/[API](../../../../glossary.md#term-application-programming-interface-api), [descriptor](../../../../glossary.md#term-descriptor) semantic change, new [dependency](../../../../glossary.md#term-dependency), [ABI](../../../../glossary.md#term-application-binary-interface-abi)
export/version, transfer [widening](../../../../glossary.md#term-widening), [alias](../../../../glossary.md#aliasing-and-arena) guarantee, or non-[MPSGraph](../../../../glossary.md#term-mpsgraph) route, stop and mark 0014
Blocked.

## Dependencies and integration
- **Depends on:** Metal 0005's retained lifecycle/ABI foundation after binary withdrawal; Metal
  0008; current Model layout semantics and Compiler inference; Engine 0017; Compiler
  0006B7/0006B11; Prepare 0008; Runtime 0016. Not the withdrawn binary domain, Metal 0006–0007,
  0009–0013, or Model 0026.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [lifecycle glossary](../../../../glossary.md#lifecycle); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Model glossary](../../../../glossary.md#term-model); [layout glossary](../../../../glossary.md#layout); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** every concurrent Metal capability, preparation, native schema, candidate/
  codec, materialization, or Engine Metal integration change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** the urgent [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-withdrawal remediation based on clean parent
  `9fce9401614e3e3b4e9a01e6f2f663538fdb2994`; execution must start from clean integrated `main`.
- **Integration order:** mandatory probe -> capability/topology -> schema/state validation ->
  candidate invalidation -> lifecycle/authentication -> Engine proof -> docs/review.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native export/schema audit; focused Metal, conformance, Compiler,
  real Engine, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols
- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  and native foundation — composed layouts, <span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous), [schema](../../../../glossary.md#term-schema) v4, and [fail-closed](../../../../glossary.md#term-fail-closed) [topology](../../../../glossary.md#term-topology).
- Preparation/finalization/executable/buffer/schedule and candidate/codec types — exact dense-target
  authentication, version invalidation, declarations, binding, and lifecycle.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [codec glossary](../../../../glossary.md#term-codec); [target glossary](../../../../glossary.md#term-target); [authentication glossary](../../../../glossary.md#term-authentication); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides/Javadocs — exact
  layouts, raw bits, public execution, exclusions, and no transfer widening.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [widening glossary](../../../../glossary.md#term-widening))</span>

## Acceptance criteria
- The disposable probe passes every topology, full corpus, direct target, Shape/order, canary,
  binding, repetition, concurrency, and independence gate before capability changes.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [canary glossary](../../../../glossary.md#term-canary); [binding glossary](../../../../glossary.md#term-binding); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Capability and analysis admit only the stated local affine/contiguous compositions; view feeds,
  unresolved geometry, and affine-to-NEG edges without <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains thirteen exports; [schema](../../../../glossary.md#term-schema) v4 retains 160-byte records, keeps [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) wires `2..5`
  rejected, and enforces exact live wire/state rules; all versioned decisions invalidate.
- Fake/native tests cover all positive topologies and malformed attrs, states, order, geometry,
  target authentication, rollback, close, reuse, concurrency, and independent sessions.<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- A Metal-only public Engine run publishes intermediate/final views and canonical contiguous
  results with exact raw bits, then consumes <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run); [canonical glossary](../../../../glossary.md#term-canonical-form); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>`CONTIGUOUS` through existing NEG; no [CPU](../../../../glossary.md#term-cpu) [owner](../../../../glossary.md#term-owner) exists.
- No shared contract, alias promise, transfer/ingress widening, implicit node, custom kernel,
  blocked numerical operation, backward/training, FLOAT16/BFLOAT16, or performance claim is added.
  Independent Class C review approves.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [widening glossary](../../../../glossary.md#term-widening); [node glossary](../../../../glossary.md#node); [kernel glossary](../../../../glossary.md#kernel); [operation glossary](../../../../glossary.md#operation); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>

## [Validation](../../../../glossary.md#term-validation)
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

Also validate balanced Markdown fences, local links/anchors, exactly one [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier,
exactly thirteen exports, unchanged dependencies/contracts, and no broadened transfer or [alias](../../../../glossary.md#aliasing-and-arena) text.

## Documentation and review impact
Update [Metal](../../../../glossary.md#term-metal)/native Javadocs and guides plus targeted [Tensor](../../../../glossary.md#tensor)/[Compiler](../../../../glossary.md#term-compiler) status wording only after the
probe passes. [Model](../../../../glossary.md#term-model) semantics and architecture contracts remain unchanged. Independent Class C
review covers capability truth, local [view](../../../../glossary.md#term-view-meanings) [authentication](../../../../glossary.md#term-authentication), exact bits/[layouts](../../../../glossary.md#layout), [schema](../../../../glossary.md#term-schema)/candidates,
[lifecycle](../../../../glossary.md#lifecycle), [Engine](../../../../glossary.md#term-engine) evidence, and exclusions.

## Result
Implemented, validated, and independently approved on Apple M3 Max.

- Before production edits, a disposable Objective-C/MPSGraph probe passed two independent process
  sessions. Each session covered five required topologies at optimization levels 0 and 1 with
  three independently compiled executables per topology/level, 32 adversarial raw-bit corpus
  rotations per executable, and one concurrent run per executable: 960 sequential plus 30
  concurrent executions per session. Direct intermediate/final target buffers, canaries, Shapes,
  feed immutability, feed/target permutations, fan-out, same-Shape <span data-glossary-navigation="true"> ([Objective-C glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [topology glossary](../../../../glossary.md#term-topology); [run glossary](../../../../glossary.md#term-run); [target glossary](../../../../glossary.md#term-target); [buffers glossary](../../../../glossary.md#term-buffer); [Shapes glossary](../../../../glossary.md#shape); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous), and rank
  boundaries all passed exact raw-bit checks.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- [Metal](../../../../glossary.md#term-metal) now admits only exact [graph](../../../../glossary.md#graph)-local affine composition and explicit `CONTIGUOUS`, tracks
  unavailable/canonical/affine-view states in Java and native validation, rejects foreign view
  feeds and view-to-NEG edges, and retains binary-wire rejection.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings); [validation glossary](../../../../glossary.md#term-validation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- Native node schema, workload/exact-policy, candidate, compatibility, route, and codec identities
  are version 4. ABI 4 still exports exactly thirteen symbols, and node records remain 160 bytes.
  Dense affine target authentication binds the exact executable, plan, context, target position,
  value, producer kind, descriptor, byte extent, and live representation. Canonical targets and
  cross-owner transfer retain their existing paths.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [authentication glossary](../../../../glossary.md#term-authentication); [producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [extent glossary](../../../../glossary.md#term-extent); [owner glossary](../../../../glossary.md#term-owner))</span>
- Backend, conformance, Compiler, real-device public Engine, Javadoc, architecture, whitespace,
  Markdown fence, and local-link validations passed. The Engine proof uses Metal as the sole owner,
  publishes every composed view plus canonical <span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [device glossary](../../../../glossary.md#term-device); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`CONTIGUOUS`/NEG results, and preserves exact
  adversarial raw bits.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- No numerical-policy Variant B was incorporated. The strict bit-preserving policy remains
  current; its versioned workload/exact-policy fingerprint and candidate compatibility are the
  explicit touchpoint for any separately authorized future numerical profile. No backward,
  training, binary arithmetic, new type, transfer widening, alias promise, or custom kernel was
  added.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [widening glossary](../../../../glossary.md#term-widening); [alias glossary](../../../../glossary.md#aliasing-and-arena); [kernel glossary](../../../../glossary.md#kernel))</span>
- Independent Class C review of the integrated implementation returned `APPROVE` with zero
  findings. Task 0014 is Complete.

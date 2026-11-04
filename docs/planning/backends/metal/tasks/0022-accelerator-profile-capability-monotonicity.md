# Task 0022: [ACCELERATOR](../../../../glossary.md#term-accelerator) Profile Capability Monotonicity

## Reader context

Monotonicity here compared the same operation occurrence under two historical profiles:
support in the stricter profile had to remain support in the accelerator profile. It did not
enlarge shapes, types or layouts, and newly admitted movement and negation still retained their
exact result rules.

Planning queried occurrence capability; Metal's whole-partition analysis additionally
authenticated a view's local producer and the specific transpose edge consumed by MATMUL.
A descriptor could describe eligible geometry without proving that provenance. CONTIGUOUS
explicitly materialized canonical order; it did not authorize automatic canonicalization elsewhere.
The [current occurrence admission explanation](../../../../explanation/packages/metal.md#1-admit-an-occurrence-not-an-operation-name)
shows why these two checks remain distinct after profile removal. The original initial BLOCK,
documentation repair and final approval are preserved below.

## Status

Complete

Implementation `415175947efc5151b71e764ad941f91445342cd7` delivered the profile-monotonicity
correction. Documentation remediation `90cd5fd925a149d055d434fac2e5bb2ee38b130f` resolved the sole
P1 current-behavior drift found by independent Class C review; final independent review returned
`APPROVE` with zero findings. The retained worker [checkpoint](../../../../glossary.md#term-checkpoint) below is the complete executable
evidence. This [planning](../../../../glossary.md#term-planning)-only finalization reran no production or test execution.

## Change class

Class C — this corrects a [graph](../../../../glossary.md#graph)-wide numerical-profile capability [invariant](../../../../glossary.md#term-invariant) across [Planning](../../../../glossary.md#term-planning)
capability, [backend](../../../../glossary.md#backend) [preflight](../../../../glossary.md#term-preflight), whole-[partition](../../../../glossary.md#partition) state [topology](../../../../glossary.md#term-topology), native-route [authentication](../../../../glossary.md#term-authentication), public
[Engine](../../../../glossary.md#term-engine) behavior, [cache](../../../../glossary.md#term-cache) compatibility analysis, authoritative architecture status, and [conformance](../../../../glossary.md#term-conformance).
A clean implementation context and independent final review are mandatory.

## Goal and global [invariant](../../../../glossary.md#term-invariant)

Enforce the global monotonicity rule:

> For the same [operation](../../../../glossary.md#operation) [occurrence](../../../../glossary.md#term-occurrence), [descriptor](../../../../glossary.md#term-descriptor) domain, and [backend](../../../../glossary.md#backend) availability,
> `Metal.supports(STRICT_IEEE)` implies `Metal.supports(ACCELERATOR)`. ACCELERATOR behavior may use
> a wider Model-permitted result set for named relaxed families, but it may always execute the
> strict behavior and must never remove a strict capability.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

Add every currently supported strict [Metal](../../../../glossary.md#term-metal) operation and its exact existing domain to the
[`ACCELERATOR`](../../../../glossary.md#term-accelerator) [matrix](../../../../glossary.md#term-matrix-meanings):

- `UnaryElementwiseKind.NEG`;
- `ShapeTransformKind.RESHAPE` and `EXPAND`;
- `AxisTransformKind.PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`;
- `ContiguousKind.CONTIGUOUS`;
- existing exact [canonical](../../../../glossary.md#term-canonical-form) `ABS`, already shared by both profiles.

Retain accelerator-only [tensor](../../../../glossary.md#tensor) `ADD`/`SUB`/`MUL`/`DIV`, canonical
`SUM`/[`MEAN`](../../../../glossary.md#term-mean)/`SUM_TO_SHAPE`, and positive static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) with authenticated local
transposes. Preserve strict rejection of those relaxed accelerator-only families.

## Exact capability and state domain

Task 0022 changes profile [admission](../../../../glossary.md#term-admission) only. It does not widen any [operation](../../../../glossary.md#operation)'s existing [descriptor](../../../../glossary.md#term-descriptor),
[Shape](../../../../glossary.md#shape), type, [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient), or numerical domain.

- `NEG` and `ABS` retain the current [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32)
  input/output domain; fully static positive rank <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16`; equal Shape; equal `requiresGrad`; checked
  element/byte geometry; and exact existing result semantics.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` retain every strict attribute,
  normalized-axis/permutation, Shape, logical-layout, state-transition, positive-rank, checked-
  geometry, and <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [permutation glossary](../../../../glossary.md#permutation))</span>`requiresGrad` rule. Inputs may be canonical or the exact zero-offset affine view
  state already accepted under strict. Outputs retain their authenticated affine-view state.
- `CONTIGUOUS` retains its strict domain and converts an available canonical or authenticated local
  affine view to canonical state with identical Shape, type, and `requiresGrad`.
- No new type, scalar/rank-zero feed, zero/dynamic extent, offset/foreign/arbitrary view, malformed
  attribute, alias promise, transfer form, or operation kind is admitted.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

The corrected [accelerator](../../../../glossary.md#term-accelerator) [matrix](../../../../glossary.md#term-matrix-meanings) must support whole maximal partitions containing any valid mix of
shared exact operations and existing accelerator operations when state/type [topology](../../../../glossary.md#term-topology) permits:

- canonical NEG/ABS results may feed accelerator [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), reduction, or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [nodes](../../../../glossary.md#node);
- an affine view may feed another admitted [affine transform](../../../../glossary.md#affine-transform) or `CONTIGUOUS` under the same strict
  rules;
- `CONTIGUOUS` may bridge an affine view to any accelerator operation requiring canonical state;
- general accelerator `PERMUTE` may be published or consumed by valid affine/`CONTIGUOUS` topology;
- MATMUL may consume an affine operand only when that exact operand is authenticated as the local
  rank-two <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`PERMUTE [1,0]` of a canonical source, unchanged from [Metal](../../../../glossary.md#term-metal) 0021;
- an exact local rank-two transpose may also be published or have another valid local consumer; its
  MATMUL authentication does not turn every accelerator PERMUTE into MATMUL-only state;<span data-glossary-navigation="true"> ([authentication glossary](../../../../glossary.md#term-authentication))</span>
- locally produced rank-zero reduction targets remain direct targets only and cannot become feeds
  to the positive-rank exact operations.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>

Capability remains [occurrence](../../../../glossary.md#term-occurrence)-local and must not depend on whether a later consumer is MATMUL.
Complete-[partition](../../../../glossary.md#partition) analysis remains responsible for authenticating affine [provenance](../../../../glossary.md#provenance) and the
specific local-[transpose](../../../../glossary.md#term-transpose) edge used by MATMUL.

## Numerical contract and no-probe decision

[`ACCELERATOR`](../../../../glossary.md#term-accelerator) is a superset of the strict allowed-result set. The authoritative numerical-profile
table grants no relaxation to affine/[layout](../../../../glossary.md#layout) movement or [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), and no separate relaxation
to NEG or ABS. Therefore every newly admitted [occurrence](../../../../glossary.md#term-occurrence) must reuse its existing strict [lowering](../../../../glossary.md#lowering)
and produce exactly the same allowed result set:

- NEG preserves the current exact [FLOAT32](../../../../glossary.md#term-float32) sign inversion behavior;
- affine movement preserves the exact selected [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) and logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) mapping;
- CONTIGUOUS preserves exact represented values while materializing [canonical](../../../../glossary.md#term-canonical-form) order;
- surrounding accelerator arithmetic gains only its already authorized operation-local Model
  freedoms. No relaxation crosses a node boundary or rewrites an observable strict-operation
  result.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model); [node glossary](../../../../glossary.md#node))</span>

No fresh numerical/[device](../../../../glossary.md#term-device) probe is required or authorized for Task 0022 because it adds no selector,
wire, [data type](../../../../glossary.md#data-type--datatype), native lowering, numerical algorithm, or result-set freedom. Existing strict
selectors/lowering and strict raw-bit tests are reused unchanged under a second profile. Numerical
probes are required only when authoritative documentation is silent and a new floating numerical
capability claim depends on device behavior; the smallest deciding adversarial corpus is mandatory.
If implementation requires a new selector, algorithm, [schema](../../../../glossary.md#term-schema) wire, or numerical allowance, stop
and replan with a new evidence gate before production changes continue.

## Capability, [preflight](../../../../glossary.md#term-preflight), and route corrections

- Refactor `MetalCapabilityProvider.supportsOccurrence` so the common exact domain is evaluated for
  both profiles and accelerator-only kinds are added afterward. Do not duplicate the strict
  predicates or create a second affine implementation. A focused monotonicity test must prove that
  every enumerated strict-positive occurrence is also accelerator-positive, while strict remains
  false for accelerator-only arithmetic/reduction/MATMUL.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Extend `MetalNativeApi.MpsGraphExecutableAbi.profileAllows` so ACCELERATOR accepts every
  schema-eight node kind. STRICT_IEEE remains limited to <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [node glossary](../../../../glossary.md#node))</span>`NEG`, `ABS`, `RESHAPE`, `EXPAND`,
  `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, and [`CONTIGUOUS`](../../../../glossary.md#term-contiguous).
- Remove [accelerator](../../../../glossary.md#term-accelerator)-only global restrictions that require every PERMUTE to be [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[1,0]`,
  consumed only by MATMUL, and never targeted. Retain local-transpose recognition and enforce it
  only when MATMUL actually consumes an affine operand. General accelerator affine topology must
  use the existing strict state/provenance rules.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [topology glossary](../../../../glossary.md#term-topology); [provenance glossary](../../../../glossary.md#provenance))</span>
- Apply the same correction in `MetalNegPartitionPreparer`: admit common exact nodes, remove the
  accelerator-wide PERMUTE consumer/target prohibition, preserve state walking and MATMUL-edge
  authentication, and reject every unavailable/foreign affine state before resource creation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Preserve the route model. An eligible accelerator singleton NEG may use the already implemented
  exact <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [singleton glossary](../../../../glossary.md#term-singleton))</span>`CUSTOM_SINGLE_NEG` candidate and pipeline; an accelerator singleton/partition also retains
  MPSGraph. Any mixed or multi-node partition uses the existing typed whole-partition MPSGraph
  route. No fallback, late route choice, graph rewrite, or new resource exists.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [graph glossary](../../../../glossary.md#graph))</span>

## [ABI](../../../../glossary.md#term-application-binary-interface-abi), [schema](../../../../glossary.md#term-schema), and [identity](../../../../glossary.md#term-identity-meanings) non-change

Keep native ABI v4, exactly thirteen exports, all statuses/[signatures](../../../../glossary.md#term-signature), fixed 160-byte [node](../../../../glossary.md#node) schema 8,
[operation](../../../../glossary.md#operation) wires `1..15`, attribute wires, and workload/exact-policy/candidate/compatibility/route/
[codec](../../../../glossary.md#term-codec) identities at version 9.

This non-bump is deliberate and must be tested:

- the profile is already encoded in `MetalNegTuningBatch.Compatibility`, the workload digest, and
  codec data;<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest))</span>
- ordered node kinds, attributes, descriptors, states, topology, targets, ABI, schema, and splats
  already participate in the workload signature;<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- no valid version-nine ACCELERATOR decision exists for a formerly rejected NEG/affine/
  CONTIGUOUS workload;<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous))</span>
- a strict decision cannot match an accelerator workload because the profile participates in both
  compatibility and signature;
- existing strict and accelerator workload meanings, candidate wire identities, and safe heuristic
  order do not change.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>

Tests must prove cross-profile decisions still reject, newly admitted [accelerator](../../../../glossary.md#term-accelerator) baseline
workloads have deterministic distinct signatures, existing workload signatures/candidates remain
stable, and no stale or foreign decision becomes compatible. If current encoding omits any one of
those discriminating facts, stop and report the concrete collision; only then may a separately
justified identity bump be planned. Do not bump versions defensively.

The native [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) foundation already lowers every schema-eight wire without receiving a
numerical-profile [parameter](../../../../glossary.md#term-parameter). Source inspection during implementation found one [validation](../../../../glossary.md#term-validation)-only
mirror of the obsolete global [transpose](../../../../glossary.md#term-transpose) restriction: when a [partition](../../../../glossary.md#partition) contained [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), every
recognized local transpose was required to be MATMUL-only and non-[target](../../../../glossary.md#term-target). Remove only that
validation bookkeeping while retaining MATMUL's operand-edge [authentication](../../../../glossary.md#term-authentication). This changes no
selector, [lowering](../../../../glossary.md#lowering), status, signature, schema, or numerical behavior; native build/export and the
existing raw [topology](../../../../glossary.md#term-topology) test prove the correction and ABI stability.

## Non-goals

No new [operation](../../../../glossary.md#operation), selector, wire, attribute, [data type](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) domain, numerical freedom,
probe, custom [kernel](../../../../glossary.md#kernel), transfer, [materialization](../../../../glossary.md#materialization) form, [Compiler](../../../../glossary.md#term-compiler) rule, public Java [signature](../../../../glossary.md#term-signature),
[dependency](../../../../glossary.md#term-dependency), packaging/discovery, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) state, [cache](../../../../glossary.md#term-cache) version, or performance claim. No
strict [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/reduction/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) capability. No weakening of MATMUL affine-input [authentication](../../../../glossary.md#term-authentication) and
no implicit [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) of a [view](../../../../glossary.md#term-view-meanings) before a [canonical](../../../../glossary.md#term-canonical-form)-only [accelerator](../../../../glossary.md#term-accelerator) operation.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and
  [public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)

The implementation must update the authoritative numerical-profile/[backend capability](../../../../glossary.md#backend-capability) text to state
that, for each [backend](../../../../glossary.md#backend), [ACCELERATOR](../../../../glossary.md#term-accelerator) [occurrence](../../../../glossary.md#term-occurrence) capability is a behavioral superset of STRICT_IEEE
for the same domain. If that [invariant](../../../../glossary.md#term-invariant) conflicts with another authoritative contract, stop and
resolve the architecture text explicitly rather than preserving the current nonmonotone [matrix](../../../../glossary.md#term-matrix-meanings).

## Dependencies and integration

- Depends on: Metal 0021 Complete; Metal 0014–0015 and 0019–0020; Model 0027; Config 0006;
  Engine 0018<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native-preflight/
  candidate/codec/public Engine/architecture-status scope; Task 0023 implementation scope<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `5631d51f425bbc9f7cb579127a788dac550e0501` plus this planning-only
  authorization<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Integration order: Serial after Metal 0021; capability/common-domain refactor, partition,
  Java/native preflight correction, contract-minimal focused/Engine proof,
  authoritative/explanatory documentation, then independent Class C review<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- Integration [validation](../../../../glossary.md#term-validation): [Metal](../../../../glossary.md#term-metal) numerical-profile capability-monotonicity Class C [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider.supportsOccurrence` and class/method Javadocs — common exact domain plus
  accelerator-only extension, with strict-subset proof.<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- `MetalNegPartitionPreparer.analyze` and Javadocs — shared affine/state walk, general accelerator
  PERMUTE topology, retained MATMUL-edge authentication, and unchanged routes/declarations.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- `MetalNativeApi.MpsGraphExecutableAbi.profileAllows` and PERMUTE/local-transpose validation — all
  schema-eight kinds under accelerator, strict subset unchanged, no global accelerator transpose
  restriction.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and
  `MetalNegTuningDecision` — expected production no-change; focused tests prove version-nine
  profile/signature separation and accelerator singleton-NEG candidates.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- `MetalNegPreparationPlan`, `MetalNegPartitionFinalizer`, `MetalNegPreparedExecutable`,
  `MetalMpsGraphExecutableResource`, and `MetalNegKernelPipelineResource` — expected production
  no-change; lifecycle tests prove existing resources serve the newly admitted profiles.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — remove only the global
  transpose-consumer/target validation mirror; <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`MetalMpsGraphRawAbiNativeTest` flips the obsolete
  rejection to acceptance and native build/export proves unchanged schema-eight/thirteen-export ABI.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- `MetalCapabilityProviderTest`, `MetalMpsGraphAffineSchemaTest`,
  `MetalMpsGraphAffineNativeTest`, `MetalMpsGraphRawAbiNativeTest`,
  `MetalNegPreparedExecutionTest`, and `MetalNegRouteCandidateGeneratorTest`.
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest`.
- `ARCHITECTURE.md`, foundational numerical-profile and backend-execution contracts, affected Metal
  package/type Javadocs, Metal/native/backend/API/user/capability/preparer guides, targeted glossary,
  and synchronized planning after executable behavior passes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Contract-minimal test and public [Engine](../../../../glossary.md#term-engine) requirements

- One table-driven capability test covers one representative of every common exact operation plus
  the existing cheap rank and descriptor boundaries. Every strict-positive representative must be
  accelerator-positive; existing strict negatives and accelerator-only strict rejection remain.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- One focused Java [preflight](../../../../glossary.md#term-preflight) case covers a general affine chain, direct affine [publication](../../../../glossary.md#term-publication),
  [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), NEG/ABS, and accelerator binary composition. The existing focused MATMUL case must
  still reject a non-transpose affine input and accept canonical/authenticated-transpose inputs.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Existing route/codec coverage already proves cross-profile rejection and all version-nine
  constants. Add only the missing valid accelerator singleton-NEG candidate/signature check.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [constants glossary](../../../../glossary.md#term-constant); [singleton glossary](../../../../glossary.md#term-singleton); [signature glossary](../../../../glossary.md#term-signature))</span>
- Replace the obsolete raw-native assertion that a MATMUL-consumed transpose cannot also be a
  target with acceptance. Existing raw/native tests own all unchanged schema and malformed-input
  coverage; do not duplicate it.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [schema glossary](../../../../glossary.md#term-schema))</span>
- One real [CPU](../../../../glossary.md#term-cpu)-free Engine [smoke](../../../../glossary.md#term-smoke-test) under [`ACCELERATOR`](../../../../glossary.md#term-accelerator) composes affine publication,
  `CONTIGUOUS`, NEG/ABS, tensor binary arithmetic, and reduction in one maximal Metal partition,
  materializes exact raw bits, and proves there is no CPU owner or fallback.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [owner glossary](../../../../glossary.md#term-owner))</span>

Existing strict [lowering](../../../../glossary.md#lowering), direct-[operation](../../../../glossary.md#operation) [matrices](../../../../glossary.md#term-matrix-meanings), feed permutations, [fan-out](../../../../glossary.md#term-fan-out), reuse,
[concurrency](../../../../glossary.md#term-concurrency), [lifecycle](../../../../glossary.md#lifecycle), [rollback](../../../../glossary.md#term-rollback), sessions/contexts, cleanup, and close tests remain authoritative
for unchanged behavior. Task 0022 must not duplicate those matrices. Native build/export is
required only because implementation found and removed the [validation](../../../../glossary.md#term-validation)-only native restriction; no
numerical [device](../../../../glossary.md#term-device) probe is [run](../../../../glossary.md#term-run).

## Acceptance criteria

1. For the enumerated representative Metal domain, every STRICT_IEEE-supported occurrence is also
   ACCELERATOR-supported with identical descriptor boundaries. Strict accelerator-only families
   remain false and existing out-of-domain negatives remain false.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [occurrence glossary](../../../../glossary.md#term-occurrence); [ACCELERATOR glossary](../../../../glossary.md#term-accelerator); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
2. ACCELERATOR admits exact NEG, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS, and
   ABS semantics without a new result freedom, selector, lowering, or numerical probe.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [lowering glossary](../../../../glossary.md#lowering))</span>
3. Whole-partition analysis and Java/native preflight accept common-plus-accelerator composition,
   general affine publication/consumption, and the CONTIGUOUS state bridge, while retaining exact
   MATMUL local-transpose authentication and malformed/foreign-state rejection.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [preflight glossary](../../../../glossary.md#term-preflight); [publication glossary](../../../../glossary.md#term-publication); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [authentication glossary](../../../../glossary.md#term-authentication))</span>
4. [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains exactly thirteen exports; [schema](../../../../glossary.md#term-schema) 8 and wires `1..15` remain unchanged; workload,
   exact-policy, candidate, compatibility, route-policy, and codec versions remain 9. Existing
   codec coverage plus the focused accelerator NEG check prove profile/signature separation.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [signature glossary](../../../../glossary.md#term-signature))</span>
5. Existing custom singleton NEG and MPSGraph resources serve accelerator workloads without a new
   route or resource. Existing unchanged lifecycle tests remain authoritative.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
6. One real CPU-free Engine mixed maximal-partition smoke raw-bit-checks the corrected accelerator
   composition and proves no CPU owner, transfer, or host fallback.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [smoke glossary](../../../../glossary.md#term-smoke-test); [owner glossary](../../../../glossary.md#term-owner); [host glossary](../../../../glossary.md#term-host))</span>
7. Authoritative architecture and explanatory docs state the capability/behavior monotonicity
   invariant and exact current matrix; no stale current text says accelerator rejects common exact
   operations.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
8. Focused changed-path Metal/Javadoc, raw-native/export, conformance, real Engine, architecture,
   one full build, Markdown/status/frontier, and diff checks pass; independent Class C review has
   no unresolved finding.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation)

No numerical probe command belongs to this task. The worker runs the focused changed paths, then
one full build:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test --tests 'io.github.pho001.synaptik.backend.metal.MetalCapabilityProviderTest' --tests 'io.github.pho001.synaptik.backend.metal.MetalMpsGraphAffineSchemaTest' --tests 'io.github.pho001.synaptik.backend.metal.MetalMpsGraphRawAbiNativeTest' --tests 'io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGeneratorTest'
./gradlew :testing:backend-conformance:test --tests 'io.github.pho001.synaptik.testing.conformance.MetalNegCapabilityPartitionConformanceTest'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests 'io.github.pho001.synaptik.testing.integration.EngineExplicitCompositionMetalIntegrationTest.cpuFreeAcceleratorPublishesAffineViewsAndRunsMixedExactComposition'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Record the strict-subset capability representatives, focused Java/native [preflight](../../../../glossary.md#term-preflight) positive and
negative, general PERMUTE versus [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-authenticated edge, maximal mixed [partition](../../../../glossary.md#partition), [accelerator](../../../../glossary.md#term-accelerator)
[singleton](../../../../glossary.md#term-singleton) candidates and [cache](../../../../glossary.md#term-cache) profile/[signature](../../../../glossary.md#term-signature) separation, exact [Engine](../../../../glossary.md#term-engine) [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity)/[Shapes](../../../../glossary.md#shape),
[ABI](../../../../glossary.md#term-application-binary-interface-abi)/export/[schema](../../../../glossary.md#term-schema)/version non-change, test counts/skips, changed paths, and synchronized status.
Independent review reuses this evidence and reruns only for a concrete risk or relevant executable
change.

## Implementation result and worker evidence

The common capability path now evaluates exact NEG/ABS/affine/[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) before the
[accelerator](../../../../glossary.md#term-accelerator)-only extension. The table-driven proof covers NEG, ABS, RESHAPE, EXPAND, PERMUTE,
EXPAND_DIMS, SQUEEZE, CONTIGUOUS, and the [rank](../../../../glossary.md#term-rank)-16 NEG boundary under both profiles. Existing
negative [matrices](../../../../glossary.md#term-matrix-meanings) and strict rejection of [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/reduction/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) remain unchanged.

[Partition](../../../../glossary.md#partition) analysis and Java [preflight](../../../../glossary.md#term-preflight) now accept general accelerator affine [topology](../../../../glossary.md#term-topology), [targets](../../../../glossary.md#term-target), and
the CONTIGUOUS bridge. The focused preflight program is
RESHAPE -> EXPAND -> PERMUTE -> EXPAND_DIMS -> SQUEEZE -> CONTIGUOUS -> NEG -> ADD -> ABS with the
general PERMUTE published directly; the existing malformed RESHAPE-to-MATMUL case still rejects.
MATMUL recognizes an affine operand only when it is the exact local [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `PERMUTE [1,0]` of a
[canonical](../../../../glossary.md#term-canonical-form) source. The native source needed one [validation](../../../../glossary.md#term-validation)-only correction that [planning](../../../../glossary.md#term-planning) had not
identified: it independently enforced the obsolete MATMUL-wide [transpose](../../../../glossary.md#term-transpose) consumer/target rule.
Only that bookkeeping was removed; edge [authentication](../../../../glossary.md#term-authentication) and [lowering](../../../../glossary.md#lowering) are unchanged. The raw native
test now accepts one authenticated transpose as both MATMUL input and direct target.

An accelerator [singleton](../../../../glossary.md#term-singleton) NEG exposes `CUSTOM_SINGLE_NEG` then [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph); its workload [signature](../../../../glossary.md#term-signature)
differs from the otherwise identical strict workload. Existing [codec](../../../../glossary.md#term-codec) tests reject both directions
of cross-profile decision reuse. Candidate, compatibility, route-policy, exact-policy, workload,
and codec versions remain 9. [Node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) remains 8 with wires `1..15`; [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains 4.

The real [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test) owns one [`metal`](../../../../glossary.md#term-metal) partition and no CPU partition, transfer, or [host](../../../../glossary.md#term-host)
fallback. Starting from [Shape](../../../../glossary.md#shape) `[6]`, it publishes exact [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) through RESHAPE `[2,1,3]`, EXPAND
`[2,4,3]`, PERMUTE `[4,2,3]`, EXPAND_DIMS `[4,2,1,3]`, SQUEEZE/CONTIGUOUS/NEG/ABS/ADD
`[4,2,3]`, and [axis](../../../../glossary.md#term-axis)-2 SUM `[4,2]`. Input values `[1,-2,3,-4,5,-6]` preserve exact affine bits;
NEG flips only the sign bit, ABS is exact, ADD produces repeated `[2,4,6,8,10,12]`, and SUM
produces repeated `[12,30]`, all asserted through raw-bit [materialization](../../../../glossary.md#materialization).

Worker [checkpoint](../../../../glossary.md#term-checkpoint) evidence:

- Native build passed. `nm -gU` returned exactly the thirteen ABI-v4 exports:
  `foundation_abi_version`, context create/release, buffer create/release/upload/download,
  MPSGraph executable create/release/run, and NEG pipeline create/release/run.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [buffer glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run))</span>
- Focused Metal capability/preflight/raw/[identity](../../../../glossary.md#term-identity-meanings) command: 28 tests, 0 skipped, 0 failures.
- Focused obsolete-assertion correction command: 56 tests, 0 skipped, 0 failures.
- Focused Metal [conformance](../../../../glossary.md#term-conformance): 5 tests, 0 skipped, 0 failures.
- Focused real CPU-free Engine mixed smoke: 1 test, 0 skipped, 0 failures.
- Metal Javadocs passed; architecture tests passed 9 tests, 0 skipped, 0 failures.
- Markdown/link validation passed all 19 changed documents: 18 passed the duplicate-anchor validator;
  `tensor-api.md` passed its structural/link scan with only its pre-existing repeated template
  headings excluded.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- `git diff --check` passed; the authoritative-doc stale-profile search returned no matches; the
  change set contains exactly the 34 tracked paths below and no untracked path.
- The final full `./gradlew build` passed. Its XML evidence contains 3,497 tests across 533 suites,
  29 opt-in/performance skips, 0 failures, and 0 errors.
- No numerical/[device](../../../../glossary.md#term-device) probe ran.

Exact changed paths:

- `ARCHITECTURE.md`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparationPlan.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/package-info.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProviderTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalFoundationTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphAffineSchemaTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphBinaryNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphRawAbiNativeTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutionTest.java`
- `backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalNegRouteCandidateGeneratorTest.java`
- `docs/api/compile-api.md`
- `docs/api/public-api.md`
- `docs/api/tensor-api.md`
- `docs/architecture/contracts/backend-execution.md`
- `docs/architecture/contracts/foundational-modules.md`
- `docs/architecture/current-architecture-plan.md`
- `docs/architecture/module-boundaries.md`
- `docs/architecture/runtime-prepare-backend-boundary.md`
- `docs/backend-guide/capability-provider.md`
- `docs/backend-guide/metal-backend.md`
- `docs/backend-guide/partition-preparer.md`
- `docs/glossary.md`
- `docs/index.md`
- `docs/planning/backends/metal/master-plan.md`
- `docs/planning/backends/metal/tasks/0022-accelerator-profile-capability-monotonicity.md`
- `docs/planning/backends/metal/tasks/0023-exact-int32-gather-and-one-hot.md`
- `docs/planning/roadmap.md`
- `native/metal-macos-arm64/README.md`
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/MetalNegCapabilityPartitionConformanceTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineExplicitCompositionMetalIntegrationTest.java`

## Final independent review

Independent Class C review first returned `BLOCK` for one P1 documentation-only finding: six
current-behavior passages still described the pre-correction profile [matrix](../../../../glossary.md#term-matrix-meanings). Remediation
`90cd5fd925a149d055d434fac2e5bb2ee38b130f` synchronized those passages, preserved historical
evidence, passed changed-Markdown, baseline-aware [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), stale-wording, and diff checks, and
changed no production or test file. Final independent Class C review returned `APPROVE` with zero
findings. Implementation `415175947efc5151b71e764ad941f91445342cd7` plus that remediation is the
complete Task 0022 [commit](../../../../glossary.md#term-commit) chain.

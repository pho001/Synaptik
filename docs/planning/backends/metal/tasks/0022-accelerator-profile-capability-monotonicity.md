# Task 0022: ACCELERATOR Profile Capability Monotonicity

## Status

Ready

Readiness verification: Metal 0021 is Complete at implementation `ef2c6a1a`, worker-evidence
remediation `be5543f9`, and final evidence correction `5631d51f`; independent Class C review
returned `APPROVE` with zero findings. Model 0027, Config 0006, Engine 0018, and Metal 0014–0015
and 0019–0021 are Complete. Current source proves the defect is a profile gate, not a missing native
operation: schema-eight wires and lowering already exist for every strict operation, while
`MetalCapabilityProvider`, `MetalNegPartitionPreparer`, and
`MetalNativeApi.MpsGraphExecutableAbi` deliberately exclude or constrain them under
`ACCELERATOR`. This is the sole authorized Metal frontier.

## Change class

Class C — this corrects a graph-wide numerical-profile capability invariant across Planning
capability, backend preflight, whole-partition state topology, native-route authentication, public
Engine behavior, cache compatibility analysis, authoritative architecture status, and conformance.
A clean implementation context and independent final review are mandatory.

## Goal and global invariant

Enforce the global monotonicity rule:

> For the same operation occurrence, descriptor domain, and backend availability,
> `Metal.supports(STRICT_IEEE)` implies `Metal.supports(ACCELERATOR)`. ACCELERATOR behavior may use
> a wider Model-permitted result set for named relaxed families, but it may always execute the
> strict behavior and must never remove a strict capability.

Add every currently supported strict Metal operation and its exact existing domain to the
`ACCELERATOR` matrix:

- `UnaryElementwiseKind.NEG`;
- `ShapeTransformKind.RESHAPE` and `EXPAND`;
- `AxisTransformKind.PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE`;
- `ContiguousKind.CONTIGUOUS`;
- existing exact canonical `ABS`, already shared by both profiles.

Retain accelerator-only tensor `ADD`/`SUB`/`MUL`/`DIV`, canonical
`SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static rank-two MATMUL with authenticated local
transposes. Preserve strict rejection of those relaxed accelerator-only families.

## Exact capability and state domain

Task 0022 changes profile admission only. It does not widen any operation's existing descriptor,
Shape, type, layout, gradient, or numerical domain.

- `NEG` and `ABS` retain the current canonical dense-contiguous, zero-offset, non-view `FLOAT32`
  input/output domain; fully static positive rank `1..16`; equal Shape; equal `requiresGrad`; checked
  element/byte geometry; and exact existing result semantics.
- `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` retain every strict attribute,
  normalized-axis/permutation, Shape, logical-layout, state-transition, positive-rank, checked-
  geometry, and `requiresGrad` rule. Inputs may be canonical or the exact zero-offset affine view
  state already accepted under strict. Outputs retain their authenticated affine-view state.
- `CONTIGUOUS` retains its strict domain and converts an available canonical or authenticated local
  affine view to canonical state with identical Shape, type, and `requiresGrad`.
- No new type, scalar/rank-zero feed, zero/dynamic extent, offset/foreign/arbitrary view, malformed
  attribute, alias promise, transfer form, or operation kind is admitted.

The corrected accelerator matrix must support whole maximal partitions containing any valid mix of
shared exact operations and existing accelerator operations when state/type topology permits:

- canonical NEG/ABS results may feed accelerator binary, reduction, or MATMUL nodes;
- an affine view may feed another admitted affine transform or `CONTIGUOUS` under the same strict
  rules;
- `CONTIGUOUS` may bridge an affine view to any accelerator operation requiring canonical state;
- general accelerator `PERMUTE` may be published or consumed by valid affine/`CONTIGUOUS` topology;
- MATMUL may consume an affine operand only when that exact operand is authenticated as the local
  rank-two `PERMUTE [1,0]` of a canonical source, unchanged from Metal 0021;
- an exact local rank-two transpose may also be published or have another valid local consumer; its
  MATMUL authentication does not turn every accelerator PERMUTE into MATMUL-only state;
- locally produced rank-zero reduction targets remain direct targets only and cannot become feeds
  to the positive-rank exact operations.

Capability remains occurrence-local and must not depend on whether a later consumer is MATMUL.
Complete-partition analysis remains responsible for authenticating affine provenance and the
specific local-transpose edge used by MATMUL.

## Numerical contract and no-probe decision

`ACCELERATOR` is a superset of the strict allowed-result set. The authoritative numerical-profile
table grants no relaxation to affine/layout movement or `CONTIGUOUS`, and no separate relaxation
to NEG or ABS. Therefore every newly admitted occurrence must reuse its existing strict lowering
and produce exactly the same allowed result set:

- NEG preserves the current exact FLOAT32 sign inversion behavior;
- affine movement preserves the exact selected raw bits and logical coordinate mapping;
- CONTIGUOUS preserves exact represented values while materializing canonical order;
- surrounding accelerator arithmetic gains only its already authorized operation-local Model
  freedoms. No relaxation crosses a node boundary or rewrites an observable strict-operation
  result.

No fresh numerical/device probe is required or authorized for Task 0022 because it adds no selector,
wire, data type, native lowering, numerical algorithm, or result-set freedom. Existing strict
selectors/lowering and strict raw-bit tests are reused unchanged under a second profile. If
implementation requires a new selector, native algorithm, schema wire, numerical allowance, or
operation-specific lowering rather than removing profile exclusions, stop and replan with a new
evidence gate before production changes continue.

## Capability, preflight, and route corrections

- Refactor `MetalCapabilityProvider.supportsOccurrence` so the common exact domain is evaluated for
  both profiles and accelerator-only kinds are added afterward. Do not duplicate the strict
  predicates or create a second affine implementation. A focused monotonicity test must prove that
  every enumerated strict-positive occurrence is also accelerator-positive, while strict remains
  false for accelerator-only arithmetic/reduction/MATMUL.
- Extend `MetalNativeApi.MpsGraphExecutableAbi.profileAllows` so ACCELERATOR accepts every
  schema-eight node kind. STRICT_IEEE remains limited to `NEG`, `ABS`, `RESHAPE`, `EXPAND`,
  `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, and `CONTIGUOUS`.
- Remove accelerator-only global restrictions that require every PERMUTE to be rank-two `[1,0]`,
  consumed only by MATMUL, and never targeted. Retain local-transpose recognition and enforce it
  only when MATMUL actually consumes an affine operand. General accelerator affine topology must
  use the existing strict state/provenance rules.
- Apply the same correction in `MetalNegPartitionPreparer`: admit common exact nodes, remove the
  accelerator-wide PERMUTE consumer/target prohibition, preserve state walking and MATMUL-edge
  authentication, and reject every unavailable/foreign affine state before resource creation.
- Preserve the route model. An eligible accelerator singleton NEG may use the already implemented
  exact `CUSTOM_SINGLE_NEG` candidate and pipeline; an accelerator singleton/partition also retains
  MPSGraph. Any mixed or multi-node partition uses the existing typed whole-partition MPSGraph
  route. No fallback, late route choice, graph rewrite, or new resource exists.

## ABI, schema, and identity non-change

Keep native ABI v4, exactly thirteen exports, all statuses/signatures, fixed 160-byte node schema 8,
operation wires `1..15`, attribute wires, and workload/exact-policy/candidate/compatibility/route/
codec identities at version 9.

This non-bump is deliberate and must be tested:

- the profile is already encoded in `MetalNegTuningBatch.Compatibility`, the workload digest, and
  codec data;
- ordered node kinds, attributes, descriptors, states, topology, targets, ABI, schema, and splats
  already participate in the workload signature;
- no valid version-nine ACCELERATOR decision exists for a formerly rejected NEG/affine/
  CONTIGUOUS workload;
- a strict decision cannot match an accelerator workload because the profile participates in both
  compatibility and signature;
- existing strict and accelerator workload meanings, candidate wire identities, and safe heuristic
  order do not change.

Tests must prove cross-profile decisions still reject, newly admitted accelerator baseline
workloads have deterministic distinct signatures, existing workload signatures/candidates remain
stable, and no stale or foreign decision becomes compatible. If current encoding omits any one of
those discriminating facts, stop and report the concrete collision; only then may a separately
justified identity bump be planned. Do not bump versions defensively.

The native Objective-C foundation already validates and lowers schema-eight wires for the strict
operations without receiving a numerical-profile parameter. No production native source change is
expected. Raw-native tests must prove existing schema-eight programs and mixed exact/accelerator
programs remain accepted with unchanged Shape/state validation; native build/export checks prove
ABI stability.

## Non-goals

No new operation, selector, wire, attribute, data type, Shape/layout domain, numerical freedom,
probe, custom kernel, transfer, materialization form, Compiler rule, public Java signature,
dependency, packaging/discovery, Runtime/Trace state, cache version, or performance claim. No
strict binary/reduction/MATMUL capability. No weakening of MATMUL affine-input authentication and
no implicit canonicalization of a view before a canonical-only accelerator operation.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)

The implementation must update the authoritative numerical-profile/backend capability text to state
that, for each backend, ACCELERATOR occurrence capability is a behavioral superset of STRICT_IEEE
for the same domain. If that invariant conflicts with another authoritative contract, stop and
resolve the architecture text explicitly rather than preserving the current nonmonotone matrix.

## Dependencies and integration

- Depends on: Metal 0021 Complete; Metal 0014–0015 and 0019–0020; Model 0027; Config 0006;
  Engine 0018
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native-preflight/
  candidate/codec/public Engine/architecture-status scope; Draft Metal 0023
- Parallel group: None
- Common base revision: `5631d51f425bbc9f7cb579127a788dac550e0501` plus this planning-only
  authorization
- Integration order: Serial after Metal 0021; capability/common-domain refactor, partition and Java
  preflight correction, focused/raw-native/Engine proof, authoritative/explanatory documentation,
  then independent Class C review
- Integration validation: Metal numerical-profile capability-monotonicity Class C checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider.supportsOccurrence` and class/method Javadocs — common exact domain plus
  accelerator-only extension, with strict-subset proof.
- `MetalNegPartitionPreparer.analyze` and Javadocs — shared affine/state walk, general accelerator
  PERMUTE topology, retained MATMUL-edge authentication, and unchanged routes/declarations.
- `MetalNativeApi.MpsGraphExecutableAbi.profileAllows` and PERMUTE/local-transpose validation — all
  schema-eight kinds under accelerator, strict subset unchanged, no global accelerator transpose
  restriction.
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and
  `MetalNegTuningDecision` — expected production no-change; focused tests prove version-nine
  profile/signature separation and accelerator singleton-NEG candidates.
- `MetalNegPreparationPlan`, `MetalNegPartitionFinalizer`, `MetalNegPreparedExecutable`,
  `MetalMpsGraphExecutableResource`, and `MetalNegKernelPipelineResource` — expected production
  no-change; lifecycle tests prove existing resources serve the newly admitted profiles.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — no expected production edit;
  `MetalMpsGraphRawAbiNativeTest` and native integration prove schema-eight mixed programs and
  unchanged thirteen-export ABI.
- `MetalCapabilityProviderTest`, `MetalMpsGraphAffineSchemaTest`,
  `MetalMpsGraphAffineNativeTest`, `MetalMpsGraphRawAbiNativeTest`,
  `MetalNegPreparedExecutionTest`, and `MetalNegRouteCandidateGeneratorTest`.
- `testing/backend-conformance/.../MetalNegCapabilityPartitionConformanceTest` and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest`.
- `ARCHITECTURE.md`, foundational numerical-profile and backend-execution contracts, affected Metal
  package/type Javadocs, Metal/native/backend/API/user/capability/preparer guides, targeted glossary,
  and synchronized planning after executable behavior passes.

## Test and public Engine requirements

### Capability and preflight

- Convert every existing accelerator-false assertion for strict NEG/affine/CONTIGUOUS into a
  positive assertion over the identical descriptors. Add a table-driven monotonicity invariant:
  every representative strict-positive kind, rank boundary, affine chain, and canonicalization form
  is accelerator-positive with the same inputs/outputs.
- Retain negative matrices for wrong attrs, FLOAT64/other types, scalar/zero/dynamic/rank-17 Shapes,
  offset/foreign/arbitrary views, bad permutations/axes/target Shapes, gradient mismatch, malformed
  topology, and every unsupported kind. Monotonicity does not turn strict negatives into positives.
- Java preflight accepts general accelerator affine programs and mixed common/accelerator programs,
  while strict preflight still rejects ADD/SUB/MUL/DIV/SUM/MEAN/MATMUL. MATMUL must still reject an
  arbitrary affine input and accept only a canonical input or the authenticated local rank-two
  transpose edge.
- Prove a general accelerator PERMUTE may be a target or feed CONTIGUOUS/another affine operation;
  prove a rank-two transpose can feed MATMUL and also participate in another valid local use without
  losing authentication.

### Route, identity, native, and lifecycle

- Candidate tests prove an accelerator singleton NEG exposes the existing safe candidate order,
  custom and MPSGraph routes remain exact, profile remains in workload/compatibility/codec identity,
  cross-profile selection rejects, all version constants remain nine, and existing signature
  fixtures remain unchanged.
- Raw/native tests execute unchanged schema-eight NEG and every affine/CONTIGUOUS wire under the
  corrected Java accelerator preflight, plus mixed chains with ABS and accelerator arithmetic.
  Verify direct targets, stable/permuted feeds/targets, raw-bit preservation, canaries, input
  preservation, reuse, concurrency, sessions/contexts, rollback, cleanup, and close rejection.
- Assert ABI v4 and exactly thirteen exports. Assert schema 8, wires `1..15`, and version-nine
  identities. No test may rewrite expected constants to a new version.

### Real CPU-free Engine behavior

Under `ACCELERATOR`, the real dylib and CPU-free Engine must execute and raw-bit-check:

- direct singleton NEG, including the existing custom-route domain;
- direct publication of each admitted affine transform and CONTIGUOUS;
- a general affine chain whose view is canonicalized by CONTIGUOUS before NEG/ABS and accelerator
  binary arithmetic;
- composition from NEG/ABS into accelerator reduction and a valid MATMUL case;
- an exact local transpose consumed by MATMUL while general affine publication remains valid;
- session reuse, independent session, concurrency, close/rejection, and direct materialization.

Run the same strict baseline smoke to prove behavior is unchanged. Inspect ownership/partition
artifacts to prove each eligible mixed graph is one maximal Metal partition and contains no CPU
owner, inserted transfer, host fallback, or hidden canonicalization. Results must assert exact raw
bits and Shapes, not merely successful execution.

## Acceptance criteria

1. For the complete enumerated current Metal domain, every STRICT_IEEE-supported occurrence is also
   ACCELERATOR-supported with identical descriptor boundaries. Strict accelerator-only families
   remain false and every existing out-of-domain negative remains false.
2. ACCELERATOR admits exact NEG, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS, and
   ABS semantics without a new result freedom, selector, lowering, or numerical probe.
3. Whole-partition analysis and Java preflight accept every valid common-plus-accelerator
   composition, general affine publication/consumption, and CONTIGUOUS state bridge, while retaining
   exact MATMUL local-transpose authentication and all malformed/foreign-state rejection.
4. ABI v4 retains exactly thirteen exports; schema 8 and wires `1..15` remain unchanged; workload,
   exact-policy, candidate, compatibility, route-policy, and codec versions remain 9. Tests prove
   profile/signature separation and absence of a cache-identity collision.
5. Existing custom singleton NEG and MPSGraph resources execute under accelerator with unchanged
   direct binding, lifecycle, reuse, concurrency, rollback, cleanup, and hot-path behavior. No
   native production edit or new resource exists.
6. Real CPU-free Engine raw-bit-checks each newly admitted operation and mixed maximal-partition
   composition under accelerator, while strict baseline behavior and accelerator-only operations
   remain unchanged.
7. Authoritative architecture and explanatory docs state the capability/behavior monotonicity
   invariant and exact current matrix; no stale text says accelerator rejects strict operations.
8. Focused Metal/Javadoc, raw-native/export, conformance, real Engine, architecture, full-build,
   Markdown/status/frontier, and diff checks pass; independent Class C review has no unresolved
   finding.

## Validation

No numerical probe command belongs to this task. Worker validation:

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Record the strict-subset capability matrix, Java/native preflight positives and negatives, general
PERMUTE versus MATMUL-authenticated transpose evidence, maximal mixed partitions, custom/MPSGraph
routes, exact Engine raw bits/Shapes, lifecycle/concurrency, ABI/export/schema/version non-change,
cache profile/signature separation, test counts/skips, changed paths, and synchronized status.
The serial worker owns one full checkpoint. Independent review reruns executable checks only after
relevant executable changes or when evidence is concretely stale.

## Documentation and review impact

Independent Class C review inspects the global monotonicity invariant, absence of widened operation
domains or numerical freedoms, common capability implementation, complete topology/preflight
correction, general affine publication and CONTIGUOUS bridging, retained MATMUL authentication,
accelerator custom NEG, schema/ABI/identity non-change proof, real mixed Engine execution,
lifecycle/concurrency/cleanup, Runtime/Trace absence, no-probe rationale, changed scope, and all
authoritative/explanatory documentation and validation evidence.

# Task 0062: ACCELERATOR MSE and Normalization/Loss Proof Boundary

## Status

In review — implementation/proof `1cf8a6ef`, documentation `3e0e4ea4`, and cumulative-review
remediation `b12dbe73` are validated; zero-finding re-review is pending.

## Change class

Class C — this changes profile-qualified Metal capability, private lowering/native execution and
persistent route identities, public Engine behavior, and authoritative backend documentation. Use
one clean serial implementation context and an independent cumulative code/evidence/documentation
review.

## Goal

Implement the maximal current normalization/loss subset whose complete route is covered by the
proved positive-rank FLOAT32 Metal primitives: first-class `MEAN_SQUARED_ERROR` under `ACCELERATOR`.
Keep every normalization and categorical-loss occurrence fail-closed on its named elementary,
aggregate, guard, validation, or multi-output proof blocker. Select routes before resource creation;
run no timing, autotuning, runtime selection, retry, fallback, or host repair.

## Scope

### Audited 115-kind registry boundary

The registry contains exactly these nine normalization/loss wires; there is no group-normalization
or fourth loss kind:

| Wire/kind | Model contract that must remain exact | Task-0062 decision |
|---|---|---|
| 85 `MEAN_SQUARED_ERROR` | two floating exact-Shape inputs; promoted result; `NONE`, `SUM`, `MEAN` | implement bounded domain below |
| 86 dense categorical CE with logits | normalized class axis; stable max/EXP/sum/LOG; zero-target guard; sample reduction | blocked on EXP/LOG, guard and nested reductions |
| 87 index categorical CE with logits | INT32/INT64 target; optional exact ignore-before-bounds; stable logits path; nonignored mean | blocked on EXP/LOG, pre-write index/ignore proof and nested reductions |
| 88 `BATCH_NORM_INFERENCE` | five ordered floating inputs; channel axis; typed positive epsilon; SQRT formula | blocked on SQRT and opaque direct-selector subset proof |
| 89 `BATCH_NORM_TRAINING` | five inputs/five stored outputs; `C==0 || N>=2`; typed momentum/epsilon; `N`/`N-1` statistics | blocked on SQRT/statistics plus multi-output/saved-slot execution |
| 90 `LAYER_NORM` | trailing positive normalized Shape; one or three inputs; typed positive epsilon; mean/population variance | blocked on SQRT and complete composite proof |
| 91 `RMS_NORM` | trailing positive normalized Shape; one or two inputs; typed positive epsilon; mean square | blocked on SQRT and complete composite proof |
| 92 `SOFTMAX` | one normalized axis; same Shape/type; literal EXP/sum/division sites | blocked on EXP plus opaque direct-selector complete-subset proof |
| 93 `LOG_SOFTMAX` | one normalized axis; same Shape/type; literal EXP/sum/LOG/subtraction sites | blocked on EXP/LOG plus max-shift replacement complete-subset proof |

The blocked rows remain structurally cataloged only. Direct MPSGraph names, ordinary samples, a
final-output tolerance, or a degenerate extent never substitutes for complete-domain recursive
proof. Strict and every non-FLOAT32 result remain blocked because their sites retain the strict
contract.

### Exact production domain for wire 85

Admit exactly one `LossKind.MEAN_SQUARED_ERROR + MeanSquaredErrorAttrs` occurrence when:

- profile is `ACCELERATOR`; ordered inputs are `[prediction, target]`; output arity is one;
- input Shapes are exactly equal, fully static, positive, canonical dense zero-offset non-views of
  rank `1..16`; output is canonical and is that Shape for `NONE` or rank-zero scalar for
  `SUM`/`MEAN`; all element/byte/count/native conversions are checked;
- prediction, target and output are all `FLOAT32`; input gradient flags are arbitrary and output
  `requiresGrad` is their exact logical OR; and
- reduction is exactly `NONE`, `SUM`, or `MEAN`; `MEAN` uses the already-qualified full MPSGraph
  `MEAN` reduction over the squared values, including its mandatory positive-count division; and
- repeated input identity and fan-out are legal, while every input/output storage overlap and every
  malformed type/Shape/layout/gradient/result descriptor rejects before native creation.

Keep false: strict MSE; every BFLOAT16, FLOAT64 or mixed-type pair; rank-zero feeds;
dynamic/empty/view/broadcast Shapes; another reduction, weighting, mask, axis, implicit cast, or
value-dependent branch. Capability does not inspect buffer values.

### Fixed route and numerical proof

Use the existing wire `85`, attribute wire `MSE=25`, reduction values `1..3`, and one atomic Model
occurrence. Lower it through one fixed MPSGraph composition:

1. `difference = prediction - target`;
2. `squared = difference * difference` using the same tensor twice;
3. publish `squared` for `NONE`, apply the qualified full `SUM` reduction for `SUM`, or apply the
   qualified full `MEAN` reduction for `MEAN`.

This is a direct application of the Model recursive ledger: exact mapping/reduction membership,
one SUB and one MUL per position, and the qualified all-contributors-once aggregate plus mandatory
divisor. Reuse the approved positive-rank ACCELERATOR SUB/MUL and full SUM/MEAN DAZ/FTZ/root-zero
proofs; add no loss-wide error envelope, reciprocal substitution, algebraic expansion, dropped
term, hidden value, rank-zero binary primitive, or cross-node contraction. Special classes and
signed zero are accepted only by site-by-site result-set membership. MPSGraph is the sole
production candidate; a containing `CUSTOM_PROGRAM` may retain it only as the same nested
MPSGraph step. The custom catalog remains `PENDING`.

Add a source-derived recursive oracle over adversarial raw FLOAT32 words, all three reductions,
repeated roles and representative ranks. It corroborates the structural proof and rejects
out-of-set output; it is not a sampled authorization or performance gate.

### Compiler gradients and public callers

Model and Compiler remain unchanged. Audit `ReductionGradientRules`, `LossGradientRules`, and
`NormalizationGradientRules` plus public Tensor callers. Task 0062 supports MSE forward occurrences
with the metadata matrix above, but claims no generated loss/normalization backward route:
MSE pullbacks require gradient-bearing scalar MUL and reduced-loss restore/expand closure that Metal
does not currently own; categorical and normalization pullbacks additionally inherit their blocked
SOFTMAX/LOG_SOFTMAX, RSQRT, reduction, cast/layout, guard, or saved-output sites. Index targets remain
non-differentiable and BatchNorm-training saved slots `3..4` remain non-root auxiliaries.

Prove CPU-free public Engine forward execution for every admitted reduction and gradient-metadata
class, including rank-zero publication, reuse and independent sessions. Prove strict MSE, blocked
kinds, excluded metadata, and representative generated-gradient graphs fail ownership before
native creation when no CPU is registered.

### Catalog, schema, identity, and lifecycle

- Mark only wire 85 structurally executable: `87/28 -> 88/27`; production capability becomes
  `69/46 -> 70/45`.
- Keep MPSGraph catalog `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE` and custom catalog
  `47 AVAILABLE / 68 PENDING / 0 UNAVAILABLE-WITH-PROOF`; MSE remains `COMPOSED / MC-MSE` and
  `PENDING / CP-AGGREGATE`.
- Keep schema 15, ABI 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`,
  route wires `1..3`, the fixed image, and thirteen exports. Advance workload, exact-policy,
  candidate, compatibility, route-policy, and codec identities together `17 -> 18`; version 17
  fails closed with no alias or migration reader.
- Preserve pre-resource validation, stable value/feed/target order, cold declarations, transactional
  finalization, direct targets, synchronous invocation, leases, rollback/suppression, reuse,
  concurrency isolation, close ordering, trace identity, and profile-free Runtime/Trace.

## Non-goals

No Model/Compiler/public API change; no categorical loss or normalization capability; no SQRT,
LOG, EXP, RSQRT, power, or general elementary framework; no generated backward/training claim;
no empty/dynamic/arbitrary-view domain; no new custom kernel, export, schema field, public option,
backend fallback, host calculation, timing, benchmark, autotune, cache winner, or performance claim.
Task 0053 remains Blocked and owns EXP/SIGMOID proof work.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and Core invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns profile-indexed results; profile and route are cold; Runtime/Trace stay profile-free.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — exact/discrete, primitive, aggregate and composite-inheritance floors are the sole result oracle.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — Metal owns truthful capability, lowering, native state and fixed private identities.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

If implementation needs a new result freedom, opaque-selector assumption, operation/value-dependent
route, shared lifecycle change, ABI/schema field, or broader compiler primitive, stop for a new plan.

## Dependencies and integration

- Depends on: Task 0061 Complete at reviewed head `3913bac1`; Model 0030/0031 recursive profiles;
  current Model loss/normalization and Compiler gradient contracts; completed Metal FLOAT32
  primitive/reduction, route-catalog, schema-15, and lifecycle foundations.
- Conflicts with: every concurrent Metal capability/schema/native/custom/route/package/shared-doc
  scope and resumed Task-0053 production.
- Parallel group: None.
- Common base revision: `5bd268eed92494840c99ff8bc5d9f21351ca411f`.
- Integration order: serial proof/capability/lowering/native/tests, then package/docs/status.
- Integration validation: packaged-native Metal, conformance, CPU-free public Engine, Javadoc/docs,
  identity/count/export audits, and independent cumulative Class C review.
- Shared-document integration owner: Task-0062 coordinator.

## Files and symbols

- Metal capability, preparer/plan, `MetalMpsGraphProgram`, Java native preflight, candidate/batch/
  codec identities, catalog/count tests, and affected Javadocs.
- Native foundation MSE composition/preflight and native/package guide; no new kernel source.
- Focused Metal tests, `MetalNegCapabilityPartitionConformanceTest`, and
  `EngineExplicitCompositionMetalIntegrationTest` for exact positive/negative public behavior.
- Root/scoped contracts, Metal guide, API status, targeted glossary, master plan, roadmap and this
  brief; preserve historical blocked tasks as evidence.

## Acceptance criteria

1. An exhaustive nine-wire audit and tests retain exactly the decisions above; all blocked rows and
   every excluded profile/type/Shape/layout/reduction/gradient form reject pre-resource.
2. Java/native independently validate wire 85 arity, attrs, promotion, exact Shape/result, gradient
   metadata, positive checked geometry, scalar publication, aliases and overlap.
3. Every MSE output belongs to the recursive site oracle for ordinary, subnormal, signed-zero,
   overflow, infinity and NaN cases; every contributor occurs once and MEAN uses the mandatory count.
4. Whole partitions preserve stable topology, direct publication, nested-MPSGraph behavior, reuse,
   concurrency, sessions, rollback and close; there is one fixed route and no fallback/retry/timing.
5. Counts are `70/45`, `88/27`, `75/35/5`, and `47/68/0`; schema 15/ABI 5/thirteen exports remain;
   all identity-17 data rejects and identity 18 is deterministic.
6. CPU-free public Engine executes same-type FLOAT32 for all reductions and admitted gradient
   metadata; strict/excluded/blocked/gradient negative graphs fail without native invocation.
7. Native/package, focused Metal, conformance, Engine, Javadoc, authoritative/explanatory docs,
   Markdown/link/diff checks pass; independent Class C review has no unresolved P0/P1/P2.

## Validation

Worker validation after implementation:

```bash
./native/metal-macos-arm64/build.sh
codesign --force --sign - --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

No timing or benchmark command is authorized. No full repository build is planned absent a shared
API/dependency/build change; the packaged native/backend/conformance/public Engine gates are stronger.

## Documentation and review impact

Update every current capability/count/identity/MSE status claim and affected Metal Javadocs; explain
why eight neighboring kinds remain structural-only. Independently review the final code, recursive
proof, native validation/lifecycle, public negatives, authoritative/explanatory docs and targeted
glossary impact. Reuse passing execution evidence unless remediation changes executable behavior.

## Result

Implementation `1cf8a6ef` admits only ACCELERATOR same-type canonical positive-rank FLOAT32 MSE
for `NONE`, `SUM`, and `MEAN`. Capability enforces exact Shape, layout, carrier, and gradient-OR
metadata; lowering retains one typed wire-85 node and reduction value `1..3`; native construction
emits exactly `SUB(prediction,target)`, `MUL(delta,delta)`, and the qualified full SUM or MEAN.
Strict, rank-zero, dynamic, empty, view, mixed/other carriers, all eight neighboring
normalization/loss wires, and generated MSE backward remain fail-closed.

The source-derived oracle preserves the shared subtraction result at the self-multiplication site,
enumerates primitive DAZ/FTZ choices, all-contributors-once reduction trees, final-only signed-zero
freedom, mandatory mean division, and NaN class over rank-two/rank-sixteen ordinary and special
corpora. Java/native malformed-image parity covers reduction, attribute, type, Shape, rank, and
gradient metadata. Prepared, conformance, native, and public Engine proofs cover stable feed/target
order, direct scalar publication, repeated input, retained executable reuse, isolated sessions,
input preservation, both gradient-metadata classes, direct and nested-custom-program MSE, sole
Metal ownership, excluded domains, every neighboring normalization/loss family, and
generated-backward rejection. Review remediation `b12dbe73` preserved both nested MSE operands and
added the mixed custom-program regression plus complete public family negatives.

Native build, fixed ad-hoc signing, package publication, package verification, the complete
packaged Metal suite, Metal conformance tests, the complete public explicit-composition Metal Engine
class, Metal Javadoc, and architecture tests passed. Counts are `70/45` capability and `88/27`
structural execution; catalogs remain `75/35/5` and `47/68/0`; schema 15, ABI 5, and thirteen
exports remain fixed; every backend-local identity is 18 and identity 17 fails closed. No timing,
benchmark, fallback, retry, compatibility reader, schema field, ABI export, or public API was added.

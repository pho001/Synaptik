# Task 0062: [ACCELERATOR](../../../../glossary.md#term-accelerator) [MSE](../../../../glossary.md#mean-squared-error--mse) and [Normalization](../../../../glossary.md#term-normalization)/[Loss](../../../../glossary.md#loss) Proof Boundary

## Status

Complete — implementation/proof `1cf8a6ef`, documentation `3e0e4ea4`, native/custom remediation
`b12dbe73`, evidence [checkpoint](../../../../glossary.md#term-checkpoint) `e80a03f6`, and final documentation remediation `06c57844` passed
independent cumulative code, evidence, and security review with zero remaining P0/P1/P2 findings.

## Change class

Class C — this changes profile-qualified [Metal](../../../../glossary.md#term-metal) capability, private [lowering](../../../../glossary.md#lowering)/native execution and
persistent route [identities](../../../../glossary.md#term-identity-meanings), public [Engine](../../../../glossary.md#term-engine) behavior, and authoritative [backend](../../../../glossary.md#backend) documentation. Use
one clean serial implementation context and an independent cumulative code/evidence/documentation
review.

## Goal

Implement the maximal current [normalization](../../../../glossary.md#term-normalization)/[loss](../../../../glossary.md#loss) subset whose complete route is covered by the
proved positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32) [Metal](../../../../glossary.md#term-metal) primitives: first-class `MEAN_SQUARED_ERROR` under [`ACCELERATOR`](../../../../glossary.md#term-accelerator).
Keep every normalization and categorical-loss [occurrence](../../../../glossary.md#term-occurrence) [fail-closed](../../../../glossary.md#term-fail-closed) on its named elementary,
aggregate, [guard](../../../../glossary.md#term-guard), [validation](../../../../glossary.md#term-validation), or multi-output proof blocker. Select routes before resource creation;
[run](../../../../glossary.md#term-run) no timing, [autotuning](../../../../glossary.md#autotuning--model-autotuning), [runtime](../../../../glossary.md#term-runtime) selection, retry, fallback, or [host](../../../../glossary.md#term-host) repair.

## [Scope](../../../../glossary.md#term-scope)

### Audited 115-kind [registry](../../../../glossary.md#term-registry) boundary

The registry contains exactly these nine [normalization](../../../../glossary.md#term-normalization)/[loss](../../../../glossary.md#loss) wires; there is no group-normalization
or fourth loss kind:

| Wire/kind | [Model](../../../../glossary.md#term-model) contract that must remain exact | Task-0062 decision |
|---|---|---|
| 85 `MEAN_SQUARED_ERROR` | two floating exact-[Shape](../../../../glossary.md#shape) inputs; promoted result; `NONE`, `SUM`, [`MEAN`](../../../../glossary.md#term-mean) | implement bounded domain below |
| 86 dense categorical CE with logits | normalized [class axis](../../../../glossary.md#class-axis); stable max/[EXP](../../../../glossary.md#term-expx)/sum/[LOG](../../../../glossary.md#term-log); zero-[target](../../../../glossary.md#term-target) [guard](../../../../glossary.md#term-guard); [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) reduction | blocked on EXP/LOG, guard and nested reductions |
| 87 index categorical CE with logits | [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) target; optional exact ignore-before-bounds; stable logits path; nonignored mean | blocked on EXP/LOG, pre-write index/ignore proof and nested reductions |
| 88 `BATCH_NORM_INFERENCE` | five ordered floating inputs; [channel](../../../../glossary.md#term-channel) [axis](../../../../glossary.md#term-axis); typed positive [epsilon](../../../../glossary.md#epsilon); SQRT formula | blocked on SQRT and [opaque](../../../../glossary.md#term-opaque) direct-selector subset proof |
| 89 `BATCH_NORM_TRAINING` | five inputs/five stored outputs; `C==0 || N>=2`; typed momentum/epsilon; `N`/`N-1` statistics | blocked on SQRT/statistics plus multi-output/saved-slot execution |
| 90 `LAYER_NORM` | trailing positive [normalized Shape](../../../../glossary.md#normalized-shape); one or three inputs; typed positive epsilon; mean/[population variance](../../../../glossary.md#population-variance) | blocked on SQRT and complete composite proof |
| 91 `RMS_NORM` | trailing positive normalized Shape; one or two inputs; typed positive epsilon; mean square | blocked on SQRT and complete composite proof |
| 92 [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) | one [normalized axis](../../../../glossary.md#normalized-axis); same Shape/type; literal EXP/sum/division sites | blocked on EXP plus opaque direct-selector complete-subset proof |
| 93 `LOG_SOFTMAX` | one normalized axis; same Shape/type; literal EXP/sum/LOG/subtraction sites | blocked on EXP/LOG plus max-shift replacement complete-subset proof |

The blocked rows remain structurally cataloged only. Direct [MPSGraph](../../../../glossary.md#term-mpsgraph) names, ordinary samples, a
final-output [tolerance](../../../../glossary.md#term-tolerance), or a degenerate [extent](../../../../glossary.md#term-extent) never substitutes for complete-domain recursive
proof. Strict and every non-[FLOAT32](../../../../glossary.md#term-float32) result remain blocked because their sites retain the strict
contract.

### Exact production domain for wire 85

Admit exactly one `LossKind.MEAN_SQUARED_ERROR + MeanSquaredErrorAttrs` [occurrence](../../../../glossary.md#term-occurrence) when:

- profile is [`ACCELERATOR`](../../../../glossary.md#term-accelerator); ordered inputs are `[prediction, target]`; output [arity](../../../../glossary.md#term-arity) is one;
- input Shapes are exactly equal, fully static, positive, canonical dense zero-offset non-views of
  rank <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16`; output is [canonical](../../../../glossary.md#term-canonical-form) and is that [Shape](../../../../glossary.md#shape) for `NONE` or [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [scalar](../../../../glossary.md#term-scalar) for
  `SUM`/[`MEAN`](../../../../glossary.md#term-mean); all [element](../../../../glossary.md#term-element)/byte/count/native conversions are checked;
- [prediction](../../../../glossary.md#term-prediction), [target](../../../../glossary.md#term-target) and output are all [`FLOAT32`](../../../../glossary.md#term-float32); input [gradient](../../../../glossary.md#term-gradient) flags are arbitrary and output
  `requiresGrad` is their exact logical OR; and
- reduction is exactly `NONE`, `SUM`, or `MEAN`; `MEAN` uses the already-qualified full [MPSGraph](../../../../glossary.md#term-mpsgraph)
  `MEAN` reduction over the squared values, including its mandatory positive-count division; and
- repeated input identity and fan-out are legal, while every input/output storage overlap and every
  malformed type/Shape/layout/gradient/result descriptor rejects before native creation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [fan-out glossary](../../../../glossary.md#term-fan-out); [storage overlap glossary](../../../../glossary.md#aliasing-and-arena); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

Keep false: strict [MSE](../../../../glossary.md#mean-squared-error--mse); every [BFLOAT16](../../../../glossary.md#term-bfloat16), [FLOAT64](../../../../glossary.md#term-float64) or mixed-type pair; rank-zero feeds;
dynamic/empty/[view](../../../../glossary.md#term-view-meanings)/broadcast Shapes; another reduction, weighting, [mask](../../../../glossary.md#term-boolean-mask), [axis](../../../../glossary.md#term-axis), implicit cast, or
value-dependent branch. Capability does not inspect [buffer](../../../../glossary.md#term-buffer) values.

### Fixed route and numerical proof

Use the existing wire `85`, attribute wire `MSE=25`, reduction values `1..3`, and one [atomic](../../../../glossary.md#resource-transaction-and-rollback) [Model](../../../../glossary.md#term-model)
[occurrence](../../../../glossary.md#term-occurrence). Lower it through one fixed [MPSGraph](../../../../glossary.md#term-mpsgraph) composition:

1. `difference = prediction - target`;
2. `squared = difference * difference` using the same [tensor](../../../../glossary.md#tensor) twice;
3. publish `squared` for `NONE`, apply the qualified full `SUM` reduction for `SUM`, or apply the
   qualified full [`MEAN`](../../../../glossary.md#term-mean) reduction for `MEAN`.

This is a direct application of the Model recursive ledger: exact mapping/reduction membership,
one SUB and one MUL per position, and the qualified [all-contributors-once](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) aggregate plus mandatory
divisor. Reuse the approved positive-[rank](../../../../glossary.md#term-rank) [ACCELERATOR](../../../../glossary.md#term-accelerator) SUB/MUL and full SUM/MEAN [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz)/root-zero
proofs; add no [loss](../../../../glossary.md#loss)-wide [error envelope](../../../../glossary.md#term-error-envelope), reciprocal substitution, algebraic expansion, dropped
term, hidden value, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) primitive, or cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension). Special classes and
[signed zero](../../../../glossary.md#term-signed-zero) are accepted only by site-by-site result-set membership. MPSGraph is the sole
production candidate; a containing `CUSTOM_PROGRAM` may retain it only as the same nested
MPSGraph step. The custom catalog remains `PENDING`.

Add a source-derived recursive [oracle](../../../../glossary.md#term-test-oracle) over adversarial raw [FLOAT32](../../../../glossary.md#term-float32) words, all three reductions,
repeated roles and representative ranks. It corroborates the structural proof and rejects
out-of-set output; it is not a sampled authorization or performance gate.

### [Compiler](../../../../glossary.md#term-compiler) [gradients](../../../../glossary.md#term-gradient) and public callers

[Model](../../../../glossary.md#term-model) and Compiler remain unchanged. Audit `ReductionGradientRules`, `LossGradientRules`, and
`NormalizationGradientRules` plus public [Tensor](../../../../glossary.md#tensor) callers. Task 0062 supports [MSE](../../../../glossary.md#mean-squared-error--mse) forward occurrences
with the [metadata](../../../../glossary.md#term-metadata) [matrix](../../../../glossary.md#term-matrix-meanings) above, but claims no generated [loss](../../../../glossary.md#loss)/[normalization](../../../../glossary.md#term-normalization) backward route:
MSE pullbacks require gradient-bearing [scalar](../../../../glossary.md#term-scalar) MUL and reduced-loss restore/expand closure that [Metal](../../../../glossary.md#term-metal)
does not currently own; categorical and normalization pullbacks additionally inherit their blocked
[SOFTMAX](../../../../glossary.md#softmax--log-softmax)/LOG_SOFTMAX, RSQRT, reduction, cast/[layout](../../../../glossary.md#layout), [guard](../../../../glossary.md#term-guard), or saved-output sites. Index [targets](../../../../glossary.md#term-target) remain
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) and BatchNorm-training saved slots `3..4` remain non-root auxiliaries.

Prove [CPU](../../../../glossary.md#term-cpu)-free public [Engine](../../../../glossary.md#term-engine) forward execution for every admitted reduction and gradient-metadata
class, including [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [publication](../../../../glossary.md#term-publication), reuse and independent sessions. Prove strict MSE, blocked
kinds, excluded metadata, and representative generated-gradient [graphs](../../../../glossary.md#graph) fail [ownership](../../../../glossary.md#term-ownership-meanings) before
native creation when no CPU is registered.

### Catalog, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and [lifecycle](../../../../glossary.md#lifecycle)

- Mark only wire 85 structurally executable: `87/28 -> 88/27`; production capability becomes
  `69/46 -> 70/45`.
- Keep [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE` and custom catalog
  `47 AVAILABLE / 68 PENDING / 0 UNAVAILABLE-WITH-PROOF`; [MSE](../../../../glossary.md#mean-squared-error--mse) remains `COMPOSED / MC-MSE` and
  `PENDING / CP-AGGREGATE`.
- Keep schema 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, type wires `1..6`, [operation](../../../../glossary.md#operation) wires `1..115`, attribute wires `0..41`,
  route wires `1..3`, the fixed image, and thirteen exports. Advance workload, exact-policy,
  candidate, compatibility, route-policy, and codec identities together <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`17 -> 18`; version 17
  fails closed with no alias or migration reader.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve pre-resource validation, stable value/feed/target order, cold declarations, transactional
  finalization, direct targets, synchronous invocation, leases, rollback/suppression, reuse,
  concurrency isolation, close ordering, trace identity, and profile-free Runtime/Trace.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [target glossary](../../../../glossary.md#term-target); [synchronous glossary](../../../../glossary.md#term-synchronous); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Non-goals

No [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/public [API](../../../../glossary.md#term-application-programming-interface-api) change; no categorical [loss](../../../../glossary.md#loss) or [normalization](../../../../glossary.md#term-normalization) capability; no SQRT,
[LOG](../../../../glossary.md#term-log), [EXP](../../../../glossary.md#term-expx), RSQRT, power, or general elementary framework; no generated backward/training claim;
no empty/dynamic/arbitrary-[view](../../../../glossary.md#term-view-meanings) domain; no new custom [kernel](../../../../glossary.md#kernel), export, [schema](../../../../glossary.md#term-schema) field, public option,
[backend](../../../../glossary.md#backend) fallback, [host](../../../../glossary.md#term-host) calculation, timing, benchmark, autotune, [cache](../../../../glossary.md#term-cache) winner, or performance claim.
Task 0053 remains Blocked and owns EXP/[SIGMOID](../../../../glossary.md#term-sigmoid) proof work.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and Core invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns profile-indexed results; profile and route are cold; Runtime/Trace stay profile-free.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — exact/discrete, primitive, aggregate and composite-inheritance floors are the sole result oracle.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  — Metal owns truthful capability, lowering, native state and fixed private identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.

If implementation needs a new result freedom, [opaque](../../../../glossary.md#term-opaque)-selector assumption, [operation](../../../../glossary.md#operation)/value-dependent
route, shared [lifecycle](../../../../glossary.md#lifecycle) change, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/[schema](../../../../glossary.md#term-schema) field, or broader [compiler](../../../../glossary.md#term-compiler) primitive, stop for a new plan.

## Dependencies and integration

- Depends on: Task 0061 Complete at reviewed head `3913bac1`; Model 0030/0031 recursive profiles;
  current Model loss/normalization and Compiler gradient contracts; completed Metal FLOAT32
  primitive/reduction, route-catalog, schema-15, and lifecycle foundations.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [loss glossary](../../../../glossary.md#loss); [normalization glossary](../../../../glossary.md#term-normalization); [Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient); [Metal glossary](../../../../glossary.md#term-metal); [FLOAT32 glossary](../../../../glossary.md#term-float32); [schema glossary](../../../../glossary.md#term-schema); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Conflicts with: every concurrent Metal capability/schema/native/custom/route/package/shared-doc
  scope and resumed Task-0053 production.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None.
- Common base revision: `5bd268eed92494840c99ff8bc5d9f21351ca411f`.
- Integration order: serial proof/capability/[lowering](../../../../glossary.md#lowering)/native/tests, then package/docs/status.
- Integration validation: packaged-native Metal, conformance, CPU-free public Engine, Javadoc/docs,
  identity/count/export audits, and independent cumulative Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task-0062 coordinator.

## Files and symbols

- [Metal](../../../../glossary.md#term-metal) capability, preparer/plan, `MetalMpsGraphProgram`, Java native preflight, candidate/batch/
  codec identities, catalog/count tests, and affected Javadocs.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [codec glossary](../../../../glossary.md#term-codec))</span>
- Native foundation [MSE](../../../../glossary.md#mean-squared-error--mse) composition/[preflight](../../../../glossary.md#term-preflight) and native/package guide; no new [kernel](../../../../glossary.md#kernel) source.
- Focused Metal tests, `MetalNegCapabilityPartitionConformanceTest`, and
  `EngineExplicitCompositionMetalIntegrationTest` for exact positive/negative public behavior.
- Root/scoped contracts, Metal guide, API status, targeted glossary, master plan, roadmap and this
  brief; preserve historical blocked tasks as evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

1. An exhaustive nine-wire audit and tests retain exactly the decisions above; all blocked rows and
   every excluded profile/type/Shape/layout/reduction/gradient form reject pre-resource.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
2. Java/native independently validate wire 85 arity, attrs, promotion, exact Shape/result, gradient
   metadata, positive checked geometry, scalar publication, aliases and overlap.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [publication glossary](../../../../glossary.md#term-publication); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
3. Every MSE output belongs to the recursive site oracle for ordinary, subnormal, signed-zero,
   overflow, infinity and NaN cases; every contributor occurs once and MEAN uses the mandatory count.<span data-glossary-navigation="true"> ([MSE glossary](../../../../glossary.md#mean-squared-error--mse); [oracle glossary](../../../../glossary.md#term-test-oracle); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [MEAN glossary](../../../../glossary.md#term-mean))</span>
4. Whole partitions preserve stable topology, direct publication, nested-MPSGraph behavior, reuse,
   concurrency, sessions, rollback and close; there is one fixed route and no fallback/retry/timing.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [concurrency glossary](../../../../glossary.md#term-concurrency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
5. Counts are `70/45`, `88/27`, `75/35/5`, and `47/68/0`; schema 15/ABI 5/thirteen exports remain;
   all identity-17 data rejects and identity 18 is deterministic.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
6. CPU-free public Engine executes same-type FLOAT32 for all reductions and admitted gradient
   metadata; strict/excluded/blocked/gradient negative graphs fail without native invocation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [FLOAT32 glossary](../../../../glossary.md#term-float32); [graphs glossary](../../../../glossary.md#graph))</span>
7. Native/package, focused Metal, conformance, Engine, Javadoc, authoritative/explanatory docs,
   Markdown/link/diff checks pass; independent Class C review has no unresolved P0/P1/P2.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

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
[API](../../../../glossary.md#term-application-programming-interface-api)/[dependency](../../../../glossary.md#term-dependency)/build change; the packaged native/[backend](../../../../glossary.md#backend)/[conformance](../../../../glossary.md#term-conformance)/public [Engine](../../../../glossary.md#term-engine) gates are stronger.

## Documentation and review impact

Update every current capability/count/[identity](../../../../glossary.md#term-identity-meanings)/[MSE](../../../../glossary.md#mean-squared-error--mse) status claim and affected [Metal](../../../../glossary.md#term-metal) Javadocs; explain
why eight neighboring kinds remain structural-only. Independently review the final code, recursive
proof, native [validation](../../../../glossary.md#term-validation)/[lifecycle](../../../../glossary.md#lifecycle), public negatives, authoritative/explanatory docs and targeted
glossary impact. Reuse passing execution evidence unless remediation changes executable behavior.

## Result

Implementation `1cf8a6ef` admits only [ACCELERATOR](../../../../glossary.md#term-accelerator) same-type [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32) [MSE](../../../../glossary.md#mean-squared-error--mse)
for `NONE`, `SUM`, and [`MEAN`](../../../../glossary.md#term-mean). Capability enforces exact [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [carrier](../../../../glossary.md#term-storage-carrier), and [gradient](../../../../glossary.md#term-gradient)-OR
[metadata](../../../../glossary.md#term-metadata); [lowering](../../../../glossary.md#lowering) retains one typed wire-85 [node](../../../../glossary.md#node) and reduction value `1..3`; native construction
emits exactly `SUB(prediction,target)`, `MUL(delta,delta)`, and the qualified full SUM or MEAN.
Strict, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), dynamic, empty, [view](../../../../glossary.md#term-view-meanings), mixed/other carriers, all eight neighboring
[normalization](../../../../glossary.md#term-normalization)/[loss](../../../../glossary.md#loss) wires, and generated MSE backward remain [fail-closed](../../../../glossary.md#term-fail-closed).

The source-derived [oracle](../../../../glossary.md#term-test-oracle) preserves the shared subtraction result at the self-multiplication site,
enumerates primitive [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) choices, [all-contributors-once](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) reduction trees, final-only [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)
freedom, mandatory mean division, and [NaN](../../../../glossary.md#term-nan) class over [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/rank-sixteen ordinary and special
corpora. Java/native malformed-image parity covers reduction, attribute, type, Shape, rank, and
gradient metadata. Prepared, [conformance](../../../../glossary.md#term-conformance), native, and public [Engine](../../../../glossary.md#term-engine) proofs cover stable feed/[target](../../../../glossary.md#term-target)
order, direct [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication), repeated input, retained executable reuse, isolated sessions,
input preservation, both gradient-metadata classes, direct and nested-custom-program MSE, sole
[Metal](../../../../glossary.md#term-metal) [ownership](../../../../glossary.md#term-ownership-meanings), excluded domains, every neighboring normalization/loss family, and
generated-backward rejection. Review remediation `b12dbe73` preserved both nested MSE operands and
added the mixed custom-program [regression](../../../../glossary.md#term-regression) plus complete public family negatives.

Native build, fixed [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), package publication, package verification, the complete
packaged Metal suite, Metal conformance tests, the complete public explicit-composition Metal Engine
class, Metal [Javadoc](../../../../glossary.md#term-javadoc), and architecture tests passed. Counts are `70/45` capability and `88/27`
structural execution; catalogs remain `75/35/5` and `47/68/0`; [schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, and thirteen
exports remain fixed; every [backend](../../../../glossary.md#backend)-local [identity](../../../../glossary.md#term-identity-meanings) is 18 and identity 17 fails closed. No timing,
benchmark, fallback, retry, compatibility reader, schema field, ABI export, or public [API](../../../../glossary.md#term-application-programming-interface-api) was added.

The final forced executions recorded 193 complete Metal tests, 9 Metal conformance tests, 26 tests
in the complete public explicit-composition Metal Engine class, and 9 architecture tests, all with
zero failures, errors, or skips. Independent cumulative reviewers
`Task0062CodeRereview`, `Task0062EvidenceRereview`, and `Task0062SecurityRereview` approved the
code, proof, documentation, and native safety boundary with zero remaining P0/P1/P2 findings. The
evidence re-review covered `f5f46bfd..06c57844` after correcting the accepted identity decision and
both current backend-guide capability summaries.

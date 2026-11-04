# Task 0010: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32)/[BOOL](../../../../glossary.md#data-type--datatype) Predicates and Selection

## Reader context

This proposed Boolean storage and predicate extension stopped at its comparison gate. Shapes,
types and layout described a legal query; represented FLOAT32 input values still had to compare
correctly. Preserved input bits and canonical zero-or-one Boolean bytes ruled out some transport
errors, but did not repair the observed collapse of subnormal comparisons into zero comparisons.

Model owned those comparison rules. Metal's planned validation, storage and publication sections
did not become a delivered BOOL route through this task, and the controls were not proof that
every planned logic or selection case passed.
The [current exact comparison, Boolean and selection domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
connect to the later implementation; both historical status formulations below remain as recorded.

## Status

Blocked

The mandatory independent real-[device](../../../../glossary.md#term-device) probe completed [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). No production, test, or probe
changes remain. [Metal](../../../../glossary.md#term-metal) 0008 stays Complete, and 0006, 0007, and 0009 remain separate blockers.

## Blocking evidence

On an Apple M3 Max, real [MPSGraph](../../../../glossary.md#term-mpsgraph) [`FLOAT32`](../../../../glossary.md#term-float32) comparison executables ran with optimization levels
`0` and `1` and reduced-[precision](../../../../glossary.md#term-precision) fast math disabled. Three independently compiled executables per
level ran every case eight times against caller-supplied direct [`BOOL`](../../../../glossary.md#data-type--datatype) [targets](../../../../glossary.md#term-target).

For positive minimum [subnormal](../../../../glossary.md#term-subnormal) versus positive or [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), the observed
`[GT, GE, LT, LE, EQ, NE]` results were `[0, 1, 0, 1, 1, 0]`, not the required
`[1, 1, 0, 0, 0, 1]`. For negative minimum subnormal versus positive or negative zero, the
observed results were again `[0, 1, 0, 1, 1, 0]`, not the required
`[0, 0, 1, 1, 0, 1]`. MPSGraph therefore treated each minimum subnormal as numerically equal to
zero, contrary to [Model](../../../../glossary.md#term-model)'s exact represented-value relations.

Raw input [identity](../../../../glossary.md#term-identity-meanings) controls preserved the subnormal bits. Normal [finite](../../../../glossary.md#term-finite), ordinary finite,
[infinity](../../../../glossary.md#term-infinity), and [NaN](../../../../glossary.md#term-nan) comparison controls passed. Direct `BOOL` targets were [canonical](../../../../glossary.md#term-canonical-form), canaries
remained intact, and target/feed permutations passed. The failure is therefore an independent
comparison-semantic blocker rather than upload/download, target [binding](../../../../glossary.md#term-binding), BOOL representation, or
probe corruption.

Unblocking requires an exact replacement route or an explicit Model numerical-contract change.
This task does not authorize a custom [kernel](../../../../glossary.md#kernel), fallback, or relaxation.

## Change class

Class C. This adds a second represented [Metal](../../../../glossary.md#term-metal) [data type](../../../../glossary.md#data-type--datatype), new typed [MPSGraph](../../../../glossary.md#term-mpsgraph) [nodes](../../../../glossary.md#node), private native
[schema](../../../../glossary.md#term-schema) meaning, [host](../../../../glossary.md#term-host) ingress/[materialization](../../../../glossary.md#materialization) behavior, whole-[partition](../../../../glossary.md#partition) [topology](../../../../glossary.md#term-topology), candidate [identity](../../../../glossary.md#term-identity-meanings),
and real [Engine](../../../../glossary.md#term-engine) execution while preserving shared contracts, dependencies, and cross-[owner](../../../../glossary.md#term-owner) transfer.

## Goal

Add one bounded forward [MPSGraph](../../../../glossary.md#term-mpsgraph) capability for exact [FLOAT32](../../../../glossary.md#term-float32) predicates, [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) logic, and
BOOL-conditioned FLOAT32 selection. The capability makes predicate values useful as [graph](../../../../glossary.md#graph) inputs,
intermediates, and directly published outputs without relying on blocked [unary](../../../../glossary.md#term-unary), reduction, or [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)
semantics.

## Exact capability domain

Every admitted [descriptor](../../../../glossary.md#term-descriptor) is fully static, strictly positive, [rank](../../../../glossary.md#term-rank) `1..16`, zero-offset [canonical](../../../../glossary.md#term-canonical-form)
dense-[contiguous](../../../../glossary.md#term-contiguous), and passes checked [element](../../../../glossary.md#term-element) and byte geometry. Every [operation](../../../../glossary.md#operation) uses
`NoOperationAttrs.INSTANCE`, its exact [Model](../../../../glossary.md#term-model) [arity](../../../../glossary.md#term-arity), and one output:

| Family | Inputs | Output | [Shape](../../../../glossary.md#shape) rule |
|---|---|---|---|
| six `BinaryComparisonKind` values | two [`FLOAT32`](../../../../glossary.md#term-float32) | canonical [`BOOL`](../../../../glossary.md#data-type--datatype), `requiresGrad=false` | exact right-aligned broadcast |
| `IS_FINITE`, `IS_NAN`, `IS_INF` | one `FLOAT32` | canonical `BOOL`, `requiresGrad=false` | unchanged Shape |
| `AND`, `OR` | two canonical `BOOL` | canonical `BOOL`, all flags false | exact right-aligned broadcast |
| `NOT` | one canonical `BOOL` | canonical `BOOL`, all flags false | unchanged Shape |
| `WHERE` | canonical `BOOL`, `FLOAT32`, `FLOAT32` | canonical `FLOAT32` | branches broadcast first, then condition |

For `WHERE`, both branches share one `requiresGrad` flag and the output preserves it. This retains
the current [Metal](../../../../glossary.md#term-metal) common-flag boundary without claiming backward support. FLOAT32/BOOL boundary
feeds and [targets](../../../../glossary.md#term-target), canonical BOOL caller storage, exact BOOL splats, and internal typed values are
permitted. Existing 0005 operations remain unchanged; 0008 affine operations remain FLOAT32-only
and terminal.

Capability and analysis [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for noncanonical BOOL bytes, numeric truthiness, mixed numeric
types, integral comparison, `CAST`, [floating classification](../../../../glossary.md#floating-classification) outside FLOAT32, wrong attributes,
wrong result type/Shape/flags, arbitrary [views](../../../../glossary.md#term-view-meanings), [scalar](../../../../glossary.md#term-scalar)/zero/dynamic Shapes, rank above sixteen,
unchecked geometry, malformed [topology](../../../../glossary.md#term-topology), or unsupported [nodes](../../../../glossary.md#node).

## [Model](../../../../glossary.md#term-model) semantics and independence

Comparisons use Model's exact represented [FLOAT32](../../../../glossary.md#term-float32) numeric relations: every ordered relation is false
when either operand is [NaN](../../../../glossary.md#term-nan), `NOT_EQUAL` is the complement of numeric equality, and signed zeros
compare equal. Classification treats normal, [subnormal](../../../../glossary.md#term-subnormal), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) values as [finite](../../../../glossary.md#term-finite); only
[infinities](../../../../glossary.md#nan-infinity-and-signed-zero) satisfy `IS_INF`; every NaN sign, quiet/signaling state, and [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) satisfies `IS_NAN`.
[BOOL](../../../../glossary.md#data-type--datatype) has only [canonical](../../../../glossary.md#term-canonical-form) bytes `0` and `1`. `WHERE` chooses the corresponding branch value without
conversion; selected FLOAT32 bits must survive exactly.

No new [Compiler](../../../../glossary.md#term-compiler) rule is needed for forward [capture](../../../../glossary.md#term-capture-meanings). This task does not claim [automatic
differentiation](../../../../glossary.md#autograd): `WHERE` backward can require blocked `SUM_TO_SHAPE`; predicates are
non-[differentiable](../../../../glossary.md#derivative-and-chain-rule). [Metal](../../../../glossary.md#term-metal) 0006, 0007, and 0009 remain independently blocked.

`CAST` is excluded: [MPSGraph](../../../../glossary.md#term-mpsgraph) documentation does not establish Model's 36-pair [rounding](../../../../glossary.md#term-rounding),
[saturation](../../../../glossary.md#term-saturation), NaN, signed-zero, subnormal, modulo, or direct-[BFLOAT16](../../../../glossary.md#term-bfloat16) rules. This task is not
[FLOAT16](../../../../glossary.md#term-float16)/BFLOAT16; future Metal 0028/0029 remain reserved, and FLOAT16 also requires Model 0026.

## Mandatory [MPSGraph](../../../../glossary.md#term-mpsgraph) probe and [fail-closed](../../../../glossary.md#term-fail-closed) gate

Before production edits, [run](../../../../glossary.md#term-run) a disposable real-[device](../../../../glossary.md#term-device) probe using only the [SDK](../../../../glossary.md#term-sdk)'s six
comparison selectors; `isFinite`, `isNaN`, `isInfinite`; `logicalAND`, `logicalOR`, `not`; and
`selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:`.

Run optimization levels `0` and `1`, at least three independently compiled executables per level,
and at least eight executions per case. Use production-style typed placeholders, caller-supplied
[canary](../../../../glossary.md#term-canary)-filled direct [targets](../../../../glossary.md#term-target), exact [Shapes](../../../../glossary.md#shape), and target/feed [permutation](../../../../glossary.md#permutation) checks. Prove:

- all comparison truth tables over ordinary values, infinities, signed zeros, quiet/signaling NaNs
  of both signs, and positive/negative subnormals distinct from zero;<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal))</span>
- all classifications over normals, signed zeros, minimum/ordinary subnormals, infinities, and
  representative NaN signs/payloads;<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>
- exhaustive [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) truth tables, including direct BOOL feeds;
- branch-first then condition broadcasting, operand order, and exact selected FLOAT32 raw bits for
  normals, subnormals, signed zeros, infinities, and NaNs; and<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [FLOAT32 glossary](../../../../glossary.md#term-float32); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- exact BOOL Shapes and canonical bytes `0`/`1`, complete direct writes without touching outside
  canaries, stable permutations, mixed typed composition, repetition, and independence.

If any selector, typed [binding](../../../../glossary.md#term-binding), Shape, broadcast order, canonical output, semantic case, selected
raw bit, direct write, or permutation gate fails, remove the probe and mark 0010 Blocked. Do not
substitute a custom [kernel](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu)/[host](../../../../glossary.md#term-host) fallback, conversion, relaxation, or undocumented behavior.

## Native [schema](../../../../glossary.md#term-schema), candidates, and [lifecycle](../../../../glossary.md#lifecycle)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exports. Bump the pointed-to fixed
160-byte [node](../../../../glossary.md#node) schema from version 2 to version 3. Preserve [operation](../../../../glossary.md#operation) wires `1..10`, use `11..23` in
the table order above, and reinterpret only the former reserved header word as a third-input index:
absent for [unary](../../../../glossary.md#term-unary)/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) nodes and required only for `WHERE`. Derive and validate each value's
[FLOAT32](../../../../glossary.md#term-float32)/[BOOL](../../../../glossary.md#data-type--datatype) type from node roles before creating placeholders; every use must agree. Native
creation validates [arity](../../../../glossary.md#term-arity), sentinels, types, [Shapes](../../../../glossary.md#shape), broadcasts, [topology](../../../../glossary.md#term-topology), [targets](../../../../glossary.md#term-target), and byte widths.
[Runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) uses the matching [MPSGraph](../../../../glossary.md#term-mpsgraph) type for every feed and direct target.

Bump candidate, compatibility/workload, route-policy, and [backend](../../../../glossary.md#backend)-local [codec](../../../../glossary.md#term-codec) versions. [Identity](../../../../glossary.md#term-identity-meanings)
includes ordered typed nodes, [descriptors](../../../../glossary.md#term-descriptor), [canonical](../../../../glossary.md#term-canonical-form) BOOL and predicate/selection policies,
boundary roles, schema/ABI, topology, splat bits, and live target. Old decisions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). New
occurrences use MPSGraph; the custom [singleton](../../../../glossary.md#term-singleton)-NEG route is unchanged.

[Metal](../../../../glossary.md#term-metal) ingress, [run](../../../../glossary.md#term-run)-owned [buffers](../../../../glossary.md#term-buffer), splats, and direct [materialization](../../../../glossary.md#materialization) gain canonical one-byte BOOL.
Cold ingress rejects noncanonical bytes before native submission; BOOL [publication](../../../../glossary.md#term-publication) returns exact
canonical bytes. [CPU](../../../../glossary.md#term-cpu)/Metal cross-[owner](../../../../glossary.md#term-owner) transfer remains canonical positive [rank](../../../../glossary.md#term-rank) `1..16` FLOAT32,
so any mixed-owner BOOL edge fails before analysis.

Analysis declares typed buffers/workspace before assignment; finalization transactionally creates
one reusable Shape/type-specialized executable. Cold binding validates representations once; hot
execution makes one [synchronous](../../../../glossary.md#term-synchronous) submission. Existing [rollback](../../../../glossary.md#term-rollback), closure, repeated/concurrent run,
and independent-session [ownership](../../../../glossary.md#term-ownership-meanings) remains. No retry, fallback, hidden copy, per-node invocation,
async execution, [pooling](../../../../glossary.md#term-pooling), or tuning workflow is added.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime receives prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Concrete backends and Metal](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([backends glossary](../../../../glossary.md#backend))</span>
  — Metal owns MPSGraph, native schema, storage, and materialization; transfer stays FLOAT32.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [schema glossary](../../../../glossary.md#term-schema); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  resource requirements, route, and typed binding are fixed before hot execution.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Compiler autograd](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>
  — current forward capture is consumed without a backward claim.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and
  [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> — current [lifecycle](../../../../glossary.md#lifecycle) and predicate semantics.

If [BOOL](../../../../glossary.md#data-type--datatype) ingress/[materialization](../../../../glossary.md#materialization) or typed [node](../../../../glossary.md#node) [validation](../../../../glossary.md#term-validation) requires a shared representation, transfer,
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), or architecture-contract change, stop and mark 0010 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0008; current Model comparison/classification/BOOL/WHERE semantics; Compiler
  forward capture and layout closure; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime
  0016. Metal 0006/0007/0009 and Model 0026 are explicitly not dependencies.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [layout glossary](../../../../glossary.md#layout); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** Metal 0006/0007/0009 and any concurrent Metal capability, preparation, native
  ABI/schema, candidate/codec, storage/materialization, or Engine Metal integration work.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** `938a18cd1ab7e02e40fb6b585ec1bd6fcd5e40e2`.
- **Integration order:** disposable selector/type/value probe → schema/type validation → capability
  and topology → BOOL ingress/materialization → finalization/runtime → Engine proof → docs/review.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparation/candidate/[codec](../../../../glossary.md#term-codec) types — capability, [schema](../../../../glossary.md#term-schema) v3, typed [topology](../../../../glossary.md#term-topology), geometry, and invalidation.
- `MetalBackendRuntime`, storage, schedule, and finalization types — [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) [lifecycle](../../../../glossary.md#lifecycle).
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — typed schema and selectors.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides, Javadocs,
  glossary terms, task, master plan, and roadmap — exact proofs and status.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Acceptance criteria

- Capability/preparation admit exactly the listed FLOAT32/BOOL domain and preserve all existing
  routes, affine restrictions, and the FLOAT32-only cross-owner transfer contract.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [BOOL glossary](../../../../glossary.md#data-type--datatype); [owner glossary](../../../../glossary.md#term-owner))</span>
- The disposable real probe passes every selector, type, canonical-byte, special-value, raw-bit,
  direct-target, Shape, broadcast, permutation, repetition, and independence gate first.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [permutation glossary](../../../../glossary.md#permutation))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains exactly thirteen exports; [schema](../../../../glossary.md#term-schema) v3 retains 160-byte [nodes](../../../../glossary.md#node), adds only wire values
  `11..23` and typed third-input meaning, and rejects malformed or stale records/decisions.
- Fake and real tests cover wrong attrs/types/flags, noncanonical BOOL ingress, broadcast/Shape
  boundaries, overflow, topology, splats, direct publication, rollback, close, repetition,
  concurrency, and independent sessions.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [topology glossary](../../../../glossary.md#term-topology); [publication glossary](../../../../glossary.md#term-publication); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- A Metal-only public Engine run accepts FLOAT32 and canonical BOOL caller inputs, publishes direct
  BOOL predicate and FLOAT32 <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run))</span>`WHERE` outputs, composes comparison → logic → selection in one
  partition, and proves no CPU owner. A mixed-owner BOOL edge rejects before analysis.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- No CAST, integer, FLOAT64, BFLOAT16, FLOAT16, scalar/zero/dynamic Shape, view, blocked numerical
  family, custom kernel, backward/training, transfer widening, or performance claim is introduced.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [scalar glossary](../../../../glossary.md#term-scalar); [view glossary](../../../../glossary.md#term-view-meanings); [kernel glossary](../../../../glossary.md#kernel); [widening glossary](../../../../glossary.md#term-widening))</span>
- Independent Class C review returns APPROVE before completion.

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Predicate*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Elementwise*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, final newlines, repository-local links/anchors, exactly one
[Ready](../../../../glossary.md#term-ready) frontier, exactly thirteen exports, unchanged production dependencies/contracts, no custom
predicate [kernel](../../../../glossary.md#kernel), no [BOOL](../../../../glossary.md#data-type--datatype) cross-[owner](../../../../glossary.md#term-owner) transfer, and no [FLOAT16](../../../../glossary.md#term-float16)/[BFLOAT16](../../../../glossary.md#term-bfloat16)/CAST/backward claim.

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) Javadocs, package documentation, [backend](../../../../glossary.md#backend)/native guides, and targeted glossary status
after evidence passes. [Model](../../../../glossary.md#term-model)/[Tensor](../../../../glossary.md#tensor) semantics and architecture contracts remain unchanged.
Independent Class C review inspects capability truth, [BOOL](../../../../glossary.md#data-type--datatype) [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), special values,
[schema](../../../../glossary.md#term-schema)/types, compatibility invalidation, [lifecycle](../../../../glossary.md#lifecycle), [Engine](../../../../glossary.md#term-engine) evidence, transfer non-[widening](../../../../glossary.md#term-widening), and
all exclusions.

## Result

The mandatory probe established an independent blocker before production work. Across both
optimization levels, six independent executables and eight runs per case reproduced [subnormal](../../../../glossary.md#term-subnormal)
comparison collapse to zero while raw [identity](../../../../glossary.md#term-identity-meanings) and all named controls passed. The probe and every
other task-local executable change were removed; production remains unchanged.

Status: Incomplete
Follow-up required: provide an exact replacement route or explicitly change the [Model](../../../../glossary.md#term-model) numerical
contract before reconsidering these predicates.

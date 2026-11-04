# Task 0006: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) [Unary](../../../../glossary.md#term-unary) Algebra Route

## Status

Blocked

The implementation/probe pass is complete and [fail-closed](../../../../glossary.md#term-fail-closed). No production code, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture),
tests, or probe [artifacts](../../../../glossary.md#term-artifact) remain changed.

## Blocking evidence

The real [FLOAT32](../../../../glossary.md#term-float32) probe compared the [MPSGraph](../../../../glossary.md#term-mpsgraph) route with the [CPU](../../../../glossary.md#term-cpu) [oracle](../../../../glossary.md#term-test-oracle) and recorded these exact
results (maximum [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) where [finite](../../../../glossary.md#term-finite) comparison was applicable):

- Passed: `ABS` max ULP `0`; [`EXP`](../../../../glossary.md#term-expx) max ULP `1`; [`SIGMOID`](../../../../glossary.md#term-sigmoid) max ULP `1`.
- Failed: `RECIPROCAL` on `±FLT_MAX` produced [subnormal](../../../../glossary.md#term-subnormal) results that became [signed zero](../../../../glossary.md#term-signed-zero);
  [`LOG`](../../../../glossary.md#term-log) on the minimum positive subnormal produced finite output instead of `-infinity`;
  `SQRT` on the minimum positive subnormal produced finite output instead of `+0`;
  `RSQRT` on the minimum positive subnormal produced finite output instead of `+infinity`;
  `RELU` on [quiet NaN](../../../../glossary.md#nan-infinity-and-signed-zero) produced `+0` instead of the required [NaN](../../../../glossary.md#term-nan) result;
  [`TANH`](../../../../glossary.md#term-tanh) on `-0` produced `+0` instead of preserving `-0`, and on quiet NaN produced `+1`
  instead of NaN.

These failures are semantic, not [tolerance](../../../../glossary.md#term-tolerance) noise. The exact special-value contract in this brief
therefore cannot be satisfied by the observed MPSGraph selectors. The task has no production
change and remains blocked rather than weakening gates or silently adding a fallback. The exact
blocker is the absence of a bounded custom-[kernel](../../../../glossary.md#kernel) realization (or an explicitly relaxed numerical
contract) for all nine required semantics.

## Change class

Class C — this extends the typed [Metal](../../../../glossary.md#term-metal) native [ABI](../../../../glossary.md#term-application-binary-interface-abi), [backend capability](../../../../glossary.md#backend-capability) truth, whole-[partition](../../../../glossary.md#partition)
[lowering](../../../../glossary.md#lowering), prepared-resource [lifecycle](../../../../glossary.md#lifecycle), candidate [identity](../../../../glossary.md#term-identity-meanings), and real [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) evidence.
The [scope](../../../../glossary.md#term-scope) is deliberately one small typed ABI extension after Metal 0005 approval.

## Goal

Extend the [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 typed whole-[partition](../../../../glossary.md#partition) [MPSGraph](../../../../glossary.md#term-mpsgraph) route with static [canonical](../../../../glossary.md#term-canonical-form)-[contiguous](../../../../glossary.md#term-contiguous) [FLOAT32](../../../../glossary.md#term-float32)
[unary](../../../../glossary.md#term-unary) `ABS`, `RECIPROCAL`, [`EXP`](../../../../glossary.md#term-expx), [`LOG`](../../../../glossary.md#term-log), `SQRT`, `RSQRT`, `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), and [`TANH`](../../../../glossary.md#term-tanh). Preserve
`NEG`, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD/SUB/MUL/DIV`, the custom [singleton](../../../../glossary.md#term-singleton) NEG route, exact ABI-v4 compatibility, and
all existing [ownership](../../../../glossary.md#term-ownership-meanings)/[lifecycle](../../../../glossary.md#lifecycle) rules. A [scalar](../../../../glossary.md#term-scalar) `ADD/SUB/MUL/DIV` extension is explicitly
**deferred**: current [Model](../../../../glossary.md#term-model) semantics use `ScalarValueAttrs` [metadata](../../../../glossary.md#term-metadata) on a one-input [node](../../../../glossary.md#node), while
the ABI-v4 node record has no typed scalar field. Do not introduce a generic attribute bag or
claim scalar support without an accepted typed [schema](../../../../glossary.md#term-schema) and exact native semantics.

## [Scope](../../../../glossary.md#term-scope)

- Capability is true only for the nine listed parameterless `UnaryElementwiseKind` values plus
  the already-complete 0005 domain, with `NoOperationAttrs.INSTANCE`, positive static [rank](../../../../glossary.md#term-rank) `1..16`,
  FLOAT32, canonical dense-contiguous zero-offset non-view layouts, and equal <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [layouts glossary](../../../../glossary.md#layout))</span>`requiresGrad` flags.
- Analyze one complete maximal Metal-owned partition in stable node/value/feed/target order;
  preserve fan-out, repeated values, and unary ordering. No split, fusion, repartition, fallback,
  or per-node native invocation.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [target glossary](../../../../glossary.md#term-target); [fan-out glossary](../../../../glossary.md#term-fan-out); [unary glossary](../../../../glossary.md#term-unary); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Add typed operation enum values and an explicit unary second-input sentinel to the existing
  fixed-width node schema. Keep node schema versioned; bump the candidate/workload fingerprint
  schema and route-policy version because operation-domain meaning changes. Keep native ABI exactly
  version 4 and its thirteen-symbol export set unchanged.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- Map each [operation](../../../../glossary.md#operation) to its native [MPSGraph](../../../../glossary.md#term-mpsgraph) operation. Use [backend](../../../../glossary.md#backend)-local [constants](../../../../glossary.md#term-constant) only through
  `PrepareContext.constants()` when a test needs a typed [FLOAT32](../../../../glossary.md#term-float32) logical constant; public
  `GraphCompilationPort` remains unchanged and is not claimed as a splat ingress.
- Preserve transactional finalization, exact declarations, context child leases, reverse/attempt-all
  cleanup, close/run leases, isolated run-owned workspace/output buffers, direct assigned output
  publication, and one synchronous native call per prepared partition.<span data-glossary-navigation="true"> ([leases glossary](../../../../glossary.md#term-lease); [run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [synchronous glossary](../../../../glossary.md#term-synchronous); [prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition))</span>

## Explicit [scalar](../../../../glossary.md#term-scalar) decision

[Model](../../../../glossary.md#term-model) currently has typed `ScalarElementwiseKind.ADD/SUB/MUL/DIV` with `ScalarValueAttrs` and
one input/one output [signatures](../../../../glossary.md#term-signature). The [ABI](../../../../glossary.md#term-application-binary-interface-abi)-v4 [node](../../../../glossary.md#node) is exactly four [`uint32_t`](../../../../glossary.md#index-mappings-and-cost-dominance) cells and has no scalar
[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication). 0006 therefore does **not** advertise the scalar family and does not add a generic attrs
map, [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) hidden in [dimensions](../../../../glossary.md#dimension), string [dispatch](../../../../glossary.md#term-dispatch), or implicit scalar [Tensor](../../../../glossary.md#tensor) construction.
Scalar support is a separate follow-up only after a typed, bounded scalar [carrier](../../../../glossary.md#term-storage-carrier) (including exact
[FLOAT32](../../../../glossary.md#term-float32) conversion, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)/[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) policy, and native [validation](../../../../glossary.md#term-validation)) is accepted. If an
implementation experiment cannot prove those facts, leave scalar work deferred.

## Non-goals

No new [dtype](../../../../glossary.md#data-type--datatype), [layout](../../../../glossary.md#layout), dynamic/zero [shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank) outside `1..16`, attributes, [scalar](../../../../glossary.md#term-scalar) operations,
comparison/logical/cast/reduction/other [unary](../../../../glossary.md#term-unary) kinds, custom unary [kernels](../../../../glossary.md#kernel), [CPU](../../../../glossary.md#term-cpu) fallback, public [API](../../../../glossary.md#term-application-programming-interface-api),
partitioning policy, async execution, [pooling](../../../../glossary.md#term-pooling), [serialization](../../../../glossary.md#term-serialization), tuning, packaging, or [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 5.
Do not alter ABI-v4 symbols/[signatures](../../../../glossary.md#term-signature), existing NEG/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) mappings, or custom NEG [MSL](../../../../glossary.md#term-metal-shading-language-msl).

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — planning
  selects Metal; backend analysis owns route truth and Runtime executes prepared work.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`ARCHITECTURE.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  — Metal owns lowering, resources, and native execution.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- [`ARCHITECTURE.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — typed MPSGraph/native boundary remains backend-local.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — declare before assignment; finalization cannot add requirements or reselect.
- [`ARCHITECTURE.md` — Public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
  — use explicit Metal integration and no discovery/fallback.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>
  — cold bindings and direct typed hot execution.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and
  [native guide](../../../../../modules/backend/metal/macos-arm64/README.md) — current [ABI](../../../../glossary.md#term-application-binary-interface-abi) and [validation](../../../../glossary.md#term-validation).

If a typed [scalar](../../../../glossary.md#term-scalar) [carrier](../../../../glossary.md#term-storage-carrier) or another shared contract is required, stop and report it rather than
expanding this task.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0005; [Model](../../../../glossary.md#term-model) current `UnaryElementwiseKind`, `ScalarElementwiseKind`,
  `ScalarValueAttrs`; [Engine](../../../../glossary.md#term-engine) 0017; [Compiler](../../../../glossary.md#term-compiler) 0006B7; [Prepare](../../../../glossary.md#prepare) 0008; [Runtime](../../../../glossary.md#term-runtime) 0016.
- Conflicts with: Metal capability/preparation/native [ABI](../../../../glossary.md#term-application-binary-interface-abi)/Engine Metal scopes; [scalar](../../../../glossary.md#term-scalar) [schema](../../../../glossary.md#term-schema) work.
- Parallel group: None
- Common base revision: `9f3a264` (approved Metal 0005 remediation)
- Integration order: native typed operation/schema → capability/analysis → Prepare/finalization →
  focused fake-native tests → conformance → real CPU-free Engine evidence → docs and independent review.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Integration validation: native export/schema audit, focused Metal tests, Metal conformance,
  real Engine test with <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`SYNAPTIK_METAL_TEST_LIBRARY`, architecture checks, documentation checks,
  and `git diff --check`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphProgram.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparationPlan.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutable.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningBatch.java`,
  `MetalNegRouteCandidateGenerator.java` — versioned candidate/[fingerprint](../../../../glossary.md#term-fingerprint) bump.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and `native/metal-macos-arm64/README.md`
- Focused [Metal](../../../../glossary.md#term-metal) fake/real tests, Metal [conformance](../../../../glossary.md#term-conformance) test, and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java`
- `docs/backend-guide/metal-backend.md`, targeted glossary/Javadocs, this brief, master plan,
  and roadmap.

Exact implementation additions/renames must be recorded in the result; unrelated paths are forbidden.

## Numerical and special-value gates

[CPU](../../../../glossary.md#term-cpu) [oracle](../../../../glossary.md#term-test-oracle) compares [FLOAT32](../../../../glossary.md#term-float32) results with [operation](../../../../glossary.md#operation)-specific tolerances (ordinary [finite](../../../../glossary.md#term-finite) values:
`abs <= 2e-6 + 2e-5*abs(expected)`; transcendental/[activation](../../../../glossary.md#term-activation) chains may use `abs <= 5e-6 +
5e-5*abs(expected)`, with a documented maximum [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) gate). Gates must separately assert special
values and [signed zero](../../../../glossary.md#term-signed-zero) where specified, never hide them behind [tolerance](../../../../glossary.md#term-tolerance):

- `ABS`: `-0` becomes `+0`; [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) retain magnitude; [NaN](../../../../glossary.md#term-nan) remains NaN.
- `RECIPROCAL`: `±0 -> ±infinity`, `±infinity -> ±0` with sign, finite values preserve ordinary
  reciprocal sign, NaN remains NaN; division-by-zero behavior is required.
- [`EXP`](../../../../glossary.md#term-expx): `-infinity -> +0`, `+infinity -> +infinity`, NaN remains NaN, overflow/underflow agree
  with the FLOAT32 CPU oracle within the gate.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow))</span>
- [`LOG`](../../../../glossary.md#term-log): `+0 -> -infinity`, negative finite and `-infinity -> NaN`, `+infinity -> +infinity`,
  NaN remains NaN; positive finite values use the tolerance gate.
- `SQRT`: `+0/-0` preserve sign, [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) stays positive infinity, negative finite and
  `-infinity` produce NaN, NaN remains NaN.
- `RSQRT`: `±0 -> ±infinity`, `+infinity -> +0`, negative finite/`-infinity` -> NaN, NaN -> NaN.
- `RELU`: negative finite values and `-0` map to `+0`; positive values and infinities pass;
  NaN behavior must match the selected MPSGraph/CPU contract and be asserted explicitly.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- [`SIGMOID`](../../../../glossary.md#term-sigmoid): `-infinity -> +0`, `+infinity -> +1`, NaN remains NaN; test stability on large finite
  magnitudes and compare against a stable CPU oracle.
- [`TANH`](../../../../glossary.md#term-tanh): `-infinity -> -1`, `+infinity -> +1`, signed zero is preserved, NaN remains NaN.

Any native behavior that cannot satisfy these explicit gates is unsupported and must [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

## Acceptance criteria

- Capability and analysis admit exactly the stated unary plus 0005 domain; unsupported attrs,
  types, ranks, layouts, and scalar nodes fail closed.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [layouts glossary](../../../../glossary.md#layout); [nodes glossary](../../../../glossary.md#node))</span>
- Native schema remains typed and bounded, ABI remains exactly v4 with the exact thirteen exports,
  and Java/native validation agrees on unary sentinel, topology, dimensions, and FLOAT32 geometry.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [topology glossary](../../../../glossary.md#term-topology); [dimensions glossary](../../../../glossary.md#dimension); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- MPSGraph mappings preserve operation identity and ordered chain semantics; no generic attrs,
  string dispatch, hidden copy, fallback, or per-node invocation exists.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Candidate/workload [schema](../../../../glossary.md#term-schema) and [fingerprint](../../../../glossary.md#term-fingerprint) versions are bumped and stale decisions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).
- Fake-native tests cover every unary mapping, malformed records, special values, rollback,
  close/rejection, exact declarations, and reuse. CPU oracle tests cover each gate and tolerances.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback); [CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Real CPU-free Engine test runs a mixed unary chain with reuse, repeated runs, two independent
  prepared sessions, backend-local constants where needed, direct publication, and close/rejection;
  mixed CPU/Metal composition remains a separate existing test.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [Metal glossary](../../../../glossary.md#term-metal))</span>
- No public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, [scalar](../../../../glossary.md#term-scalar) support, or unrelated [scope](../../../../glossary.md#term-scope) changes.

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :backends:metal:javadoc
./gradlew :testing:architecture-tests:test
rg -n -i 'MPSGraph|unary|scalar|ABI|fingerprint|schema' docs/backend-guide/metal-backend.md docs/glossary.md docs/planning/backends/metal
awk '/^```/{n++} END{exit n%2}' docs/planning/backends/metal/tasks/0006-mpsgraph-float32-unary-algebra.md docs/planning/backends/metal/tasks/0005-mpsgraph-mixed-binary-whole-partition.md docs/planning/backends/metal/master-plan.md docs/planning/roadmap.md
git diff --check
```

## Documentation and review impact

Record the blocked evidence above and retain the exact numerical/special-value gates for any
future restart. Do not claim [ABI](../../../../glossary.md#term-application-binary-interface-abi), capability, [lifecycle](../../../../glossary.md#lifecycle), or [Engine](../../../../glossary.md#term-engine) completion from the reverted
probe. A future restart requires a Class C review of the exact replacement route, typed [schema](../../../../glossary.md#term-schema) and
[fingerprint](../../../../glossary.md#term-fingerprint) bump, numerical gates, ABI export audit, lifecycle [rollback](../../../../glossary.md#term-rollback), and real [CPU](../../../../glossary.md#term-cpu)-free Engine
evidence.

## Result

Blocked. The probe proved only `ABS` ULP0, [`EXP`](../../../../glossary.md#term-expx) ULP1, and [`SIGMOID`](../../../../glossary.md#term-sigmoid) ULP1; the listed reciprocal,
[log](../../../../glossary.md#term-log), square-root, reciprocal-square-root, [ReLU](../../../../glossary.md#term-relu), and [tanh](../../../../glossary.md#term-tanh) special-value cases failed. No production
change remains, and the repository is clean [fail-closed](../../../../glossary.md#term-fail-closed). Unblocking requires exact semantics via
custom [kernels](../../../../glossary.md#kernel) or an explicitly accepted relaxed contract.

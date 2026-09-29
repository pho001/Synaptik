# Task 0006: MPSGraph FLOAT32 Unary Algebra Route

## Status

Blocked

The implementation/probe pass is complete and fail-closed. No production code, native code,
tests, or probe artifacts remain changed.

## Blocking evidence

The real FLOAT32 probe compared the MPSGraph route with the CPU oracle and recorded these exact
results (maximum ULP where finite comparison was applicable):

- Passed: `ABS` max ULP `0`; `EXP` max ULP `1`; `SIGMOID` max ULP `1`.
- Failed: `RECIPROCAL` on `±FLT_MAX` produced subnormal results that became signed zero;
  `LOG` on the minimum positive subnormal produced finite output instead of `-infinity`;
  `SQRT` on the minimum positive subnormal produced finite output instead of `+0`;
  `RSQRT` on the minimum positive subnormal produced finite output instead of `+infinity`;
  `RELU` on quiet NaN produced `+0` instead of the required NaN result;
  `TANH` on `-0` produced `+0` instead of preserving `-0`, and on quiet NaN produced `+1`
  instead of NaN.

These failures are semantic, not tolerance noise. The exact special-value contract in this brief
therefore cannot be satisfied by the observed MPSGraph selectors. The task has no production
change and remains blocked rather than weakening gates or silently adding a fallback. The exact
blocker is the absence of a bounded custom-kernel realization (or an explicitly relaxed numerical
contract) for all nine required semantics.

## Change class

Class C — this extends the typed Metal native ABI, backend capability truth, whole-partition
lowering, prepared-resource lifecycle, candidate identity, and real CPU-free Engine evidence.
The scope is deliberately one small typed ABI extension after Metal 0005 approval.

## Goal

Extend the ABI-v4 typed whole-partition MPSGraph route with static canonical-contiguous FLOAT32
unary `ABS`, `RECIPROCAL`, `EXP`, `LOG`, `SQRT`, `RSQRT`, `RELU`, `SIGMOID`, and `TANH`. Preserve
`NEG`, binary `ADD/SUB/MUL/DIV`, the custom singleton NEG route, exact ABI-v4 compatibility, and
all existing ownership/lifecycle rules. A scalar `ADD/SUB/MUL/DIV` extension is explicitly
**deferred**: current Model semantics use `ScalarValueAttrs` metadata on a one-input node, while
the ABI-v4 node record has no typed scalar field. Do not introduce a generic attribute bag or
claim scalar support without an accepted typed schema and exact native semantics.

## Scope

- Capability is true only for the nine listed parameterless `UnaryElementwiseKind` values plus
  the already-complete 0005 domain, with `NoOperationAttrs.INSTANCE`, positive static rank `1..16`,
  FLOAT32, canonical dense-contiguous zero-offset non-view layouts, and equal `requiresGrad` flags.
- Analyze one complete maximal Metal-owned partition in stable node/value/feed/target order;
  preserve fan-out, repeated values, and unary ordering. No split, fusion, repartition, fallback,
  or per-node native invocation.
- Add typed operation enum values and an explicit unary second-input sentinel to the existing
  fixed-width node schema. Keep node schema versioned; bump the candidate/workload fingerprint
  schema and route-policy version because operation-domain meaning changes. Keep native ABI exactly
  version 4 and its thirteen-symbol export set unchanged.
- Map each operation to its native MPSGraph operation. Use backend-local constants only through
  `PrepareContext.constants()` when a test needs a typed FLOAT32 logical constant; public
  `GraphCompilationPort` remains unchanged and is not claimed as a splat ingress.
- Preserve transactional finalization, exact declarations, context child leases, reverse/attempt-all
  cleanup, close/run leases, isolated run-owned workspace/output buffers, direct assigned output
  publication, and one synchronous native call per prepared partition.

## Explicit scalar decision

Model currently has typed `ScalarElementwiseKind.ADD/SUB/MUL/DIV` with `ScalarValueAttrs` and
one input/one output signatures. The ABI-v4 node is exactly four `uint32_t` cells and has no scalar
payload. 0006 therefore does **not** advertise the scalar family and does not add a generic attrs
map, raw bits hidden in dimensions, string dispatch, or implicit scalar Tensor construction.
Scalar support is a separate follow-up only after a typed, bounded scalar carrier (including exact
FLOAT32 conversion, signed-zero/NaN payload policy, and native validation) is accepted. If an
implementation experiment cannot prove those facts, leave scalar work deferred.

## Non-goals

No new dtype, layout, dynamic/zero shape, rank outside `1..16`, attributes, scalar operations,
comparison/logical/cast/reduction/other unary kinds, custom unary kernels, CPU fallback, public API,
partitioning policy, async execution, pooling, serialization, tuning, packaging, or ABI version 5.
Do not alter ABI-v4 symbols/signatures, existing NEG/binary mappings, or custom NEG MSL.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — planning
  selects Metal; backend analysis owns route truth and Runtime executes prepared work.
- [`ARCHITECTURE.md` — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — Metal owns lowering, resources, and native execution.
- [`ARCHITECTURE.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — typed MPSGraph/native boundary remains backend-local.
- [`ARCHITECTURE.md` — Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — declare before assignment; finalization cannot add requirements or reselect.
- [`ARCHITECTURE.md` — Public explicit composition](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)
  — use explicit Metal integration and no discovery/fallback.
- [`ARCHITECTURE.md` — Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  — cold bindings and direct typed hot execution.
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and
  [native guide](../../../../../native/metal-macos-arm64/README.md) — current ABI and validation.

If a typed scalar carrier or another shared contract is required, stop and report it rather than
expanding this task.

## Dependencies and integration

- Depends on: Metal 0005; Model current `UnaryElementwiseKind`, `ScalarElementwiseKind`,
  `ScalarValueAttrs`; Engine 0017; Compiler 0006B7; Prepare 0008; Runtime 0016.
- Conflicts with: Metal capability/preparation/native ABI/Engine Metal scopes; scalar schema work.
- Parallel group: None
- Common base revision: `9f3a264` (approved Metal 0005 remediation)
- Integration order: native typed operation/schema → capability/analysis → Prepare/finalization →
  focused fake-native tests → conformance → real CPU-free Engine evidence → docs and independent review.
- Integration validation: native export/schema audit, focused Metal tests, Metal conformance,
  real Engine test with `SYNAPTIK_METAL_TEST_LIBRARY`, architecture checks, documentation checks,
  and `git diff --check`.
- Shared-document integration owner: Main planner.

## Files and symbols

- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalCapabilityProvider.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphProgram.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNativeApi.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPartitionPreparer.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparationPlan.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegPreparedExecutable.java`
- `backends/metal/src/main/java/io/github/pho001/synaptik/backend/metal/MetalNegTuningBatch.java`,
  `MetalNegRouteCandidateGenerator.java` — versioned candidate/fingerprint bump.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and `native/metal-macos-arm64/README.md`
- Focused Metal fake/real tests, Metal conformance test, and
  `testing/integration-tests/.../EngineExplicitCompositionMetalIntegrationTest.java`
- `docs/backend-guide/metal-backend.md`, targeted glossary/Javadocs, this brief, master plan,
  and roadmap.

Exact implementation additions/renames must be recorded in the result; unrelated paths are forbidden.

## Numerical and special-value gates

CPU oracle compares FLOAT32 results with operation-specific tolerances (ordinary finite values:
`abs <= 2e-6 + 2e-5*abs(expected)`; transcendental/activation chains may use `abs <= 5e-6 +
5e-5*abs(expected)`, with a documented maximum ULP gate). Gates must separately assert special
values and signed zero where specified, never hide them behind tolerance:

- `ABS`: `-0` becomes `+0`; infinities retain magnitude; NaN remains NaN.
- `RECIPROCAL`: `±0 -> ±infinity`, `±infinity -> ±0` with sign, finite values preserve ordinary
  reciprocal sign, NaN remains NaN; division-by-zero behavior is required.
- `EXP`: `-infinity -> +0`, `+infinity -> +infinity`, NaN remains NaN, overflow/underflow agree
  with the FLOAT32 CPU oracle within the gate.
- `LOG`: `+0 -> -infinity`, negative finite and `-infinity -> NaN`, `+infinity -> +infinity`,
  NaN remains NaN; positive finite values use the tolerance gate.
- `SQRT`: `+0/-0` preserve sign, positive infinity stays positive infinity, negative finite and
  `-infinity` produce NaN, NaN remains NaN.
- `RSQRT`: `±0 -> ±infinity`, `+infinity -> +0`, negative finite/`-infinity` -> NaN, NaN -> NaN.
- `RELU`: negative finite values and `-0` map to `+0`; positive values and infinities pass;
  NaN behavior must match the selected MPSGraph/CPU contract and be asserted explicitly.
- `SIGMOID`: `-infinity -> +0`, `+infinity -> +1`, NaN remains NaN; test stability on large finite
  magnitudes and compare against a stable CPU oracle.
- `TANH`: `-infinity -> -1`, `+infinity -> +1`, signed zero is preserved, NaN remains NaN.

Any native behavior that cannot satisfy these explicit gates is unsupported and must fail closed.

## Acceptance criteria

- Capability and analysis admit exactly the stated unary plus 0005 domain; unsupported attrs,
  types, ranks, layouts, and scalar nodes fail closed.
- Native schema remains typed and bounded, ABI remains exactly v4 with the exact thirteen exports,
  and Java/native validation agrees on unary sentinel, topology, dimensions, and FLOAT32 geometry.
- MPSGraph mappings preserve operation identity and ordered chain semantics; no generic attrs,
  string dispatch, hidden copy, fallback, or per-node invocation exists.
- Candidate/workload schema and fingerprint versions are bumped and stale decisions fail closed.
- Fake-native tests cover every unary mapping, malformed records, special values, rollback,
  close/rejection, exact declarations, and reuse. CPU oracle tests cover each gate and tolerances.
- Real CPU-free Engine test runs a mixed unary chain with reuse, repeated runs, two independent
  prepared sessions, backend-local constants where needed, direct publication, and close/rejection;
  mixed CPU/Metal composition remains a separate existing test.
- No public API, dependency direction, ABI export, scalar support, or unrelated scope changes.

## Validation

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
future restart. Do not claim ABI, capability, lifecycle, or Engine completion from the reverted
probe. A future restart requires a Class C review of the exact replacement route, typed schema and
fingerprint bump, numerical gates, ABI export audit, lifecycle rollback, and real CPU-free Engine
evidence.

## Result

Blocked. The probe proved only `ABS` ULP0, `EXP` ULP1, and `SIGMOID` ULP1; the listed reciprocal,
log, square-root, reciprocal-square-root, ReLU, and tanh special-value cases failed. No production
change remains, and the repository is clean fail-closed. Unblocking requires exact semantics via
custom kernels or an explicitly accepted relaxed contract.

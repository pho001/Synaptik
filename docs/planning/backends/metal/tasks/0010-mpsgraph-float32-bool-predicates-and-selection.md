# Task 0010: MPSGraph FLOAT32/BOOL Predicates and Selection

## Status

Blocked

The mandatory independent real-device probe completed fail closed. No production, test, or probe
changes remain. Metal 0008 stays Complete, and 0006, 0007, and 0009 remain separate blockers.

## Blocking evidence

On an Apple M3 Max, real MPSGraph `FLOAT32` comparison executables ran with optimization levels
`0` and `1` and reduced-precision fast math disabled. Three independently compiled executables per
level ran every case eight times against caller-supplied direct `BOOL` targets.

For positive minimum subnormal versus positive or negative zero, the observed
`[GT, GE, LT, LE, EQ, NE]` results were `[0, 1, 0, 1, 1, 0]`, not the required
`[1, 1, 0, 0, 0, 1]`. For negative minimum subnormal versus positive or negative zero, the
observed results were again `[0, 1, 0, 1, 1, 0]`, not the required
`[0, 0, 1, 1, 0, 1]`. MPSGraph therefore treated each minimum subnormal as numerically equal to
zero, contrary to Model's exact represented-value relations.

Raw input identity controls preserved the subnormal bits. Normal finite, ordinary finite,
infinity, and NaN comparison controls passed. Direct `BOOL` targets were canonical, canaries
remained intact, and target/feed permutations passed. The failure is therefore an independent
comparison-semantic blocker rather than upload/download, target binding, BOOL representation, or
probe corruption.

Unblocking requires an exact replacement route or an explicit Model numerical-contract change.
This task does not authorize a custom kernel, fallback, or relaxation.

## Change class

Class C. This adds a second represented Metal data type, new typed MPSGraph nodes, private native
schema meaning, host ingress/materialization behavior, whole-partition topology, candidate identity,
and real Engine execution while preserving shared contracts, dependencies, and cross-owner transfer.

## Goal

Add one bounded forward MPSGraph capability for exact FLOAT32 predicates, canonical BOOL logic, and
BOOL-conditioned FLOAT32 selection. The capability makes predicate values useful as graph inputs,
intermediates, and directly published outputs without relying on blocked unary, reduction, or MATMUL
semantics.

## Exact capability domain

Every admitted descriptor is fully static, strictly positive, rank `1..16`, zero-offset canonical
dense-contiguous, and passes checked element and byte geometry. Every operation uses
`NoOperationAttrs.INSTANCE`, its exact Model arity, and one output:

| Family | Inputs | Output | Shape rule |
|---|---|---|---|
| six `BinaryComparisonKind` values | two `FLOAT32` | canonical `BOOL`, `requiresGrad=false` | exact right-aligned broadcast |
| `IS_FINITE`, `IS_NAN`, `IS_INF` | one `FLOAT32` | canonical `BOOL`, `requiresGrad=false` | unchanged Shape |
| `AND`, `OR` | two canonical `BOOL` | canonical `BOOL`, all flags false | exact right-aligned broadcast |
| `NOT` | one canonical `BOOL` | canonical `BOOL`, all flags false | unchanged Shape |
| `WHERE` | canonical `BOOL`, `FLOAT32`, `FLOAT32` | canonical `FLOAT32` | branches broadcast first, then condition |

For `WHERE`, both branches share one `requiresGrad` flag and the output preserves it. This retains
the current Metal common-flag boundary without claiming backward support. FLOAT32/BOOL boundary
feeds and targets, canonical BOOL caller storage, exact BOOL splats, and internal typed values are
permitted. Existing 0005 operations remain unchanged; 0008 affine operations remain FLOAT32-only
and terminal.

Capability and analysis fail closed for noncanonical BOOL bytes, numeric truthiness, mixed numeric
types, integral comparison, `CAST`, floating classification outside FLOAT32, wrong attributes,
wrong result type/Shape/flags, arbitrary views, scalar/zero/dynamic Shapes, rank above sixteen,
unchecked geometry, malformed topology, or unsupported nodes.

## Model semantics and independence

Comparisons use Model's exact represented FLOAT32 numeric relations: every ordered relation is false
when either operand is NaN, `NOT_EQUAL` is the complement of numeric equality, and signed zeros
compare equal. Classification treats normal, subnormal, and signed-zero values as finite; only
infinities satisfy `IS_INF`; every NaN sign, quiet/signaling state, and payload satisfies `IS_NAN`.
BOOL has only canonical bytes `0` and `1`. `WHERE` chooses the corresponding branch value without
conversion; selected FLOAT32 bits must survive exactly.

No new Compiler rule is needed for forward capture. This task does not claim automatic
differentiation: `WHERE` backward can require blocked `SUM_TO_SHAPE`; predicates are
non-differentiable. Metal 0006, 0007, and 0009 remain independently blocked.

`CAST` is excluded: MPSGraph documentation does not establish Model's 36-pair rounding,
saturation, NaN, signed-zero, subnormal, modulo, or direct-BFLOAT16 rules. This task is not
FLOAT16/BFLOAT16; future Metal 0028/0029 remain reserved, and FLOAT16 also requires Model 0026.

## Mandatory MPSGraph probe and fail-closed gate

Before production edits, run a disposable real-device probe using only the SDK's six
comparison selectors; `isFinite`, `isNaN`, `isInfinite`; `logicalAND`, `logicalOR`, `not`; and
`selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:name:`.

Run optimization levels `0` and `1`, at least three independently compiled executables per level,
and at least eight executions per case. Use production-style typed placeholders, caller-supplied
canary-filled direct targets, exact Shapes, and target/feed permutation checks. Prove:

- all comparison truth tables over ordinary values, infinities, signed zeros, quiet/signaling NaNs
  of both signs, and positive/negative subnormals distinct from zero;
- all classifications over normals, signed zeros, minimum/ordinary subnormals, infinities, and
  representative NaN signs/payloads;
- exhaustive canonical BOOL truth tables, including direct BOOL feeds;
- branch-first then condition broadcasting, operand order, and exact selected FLOAT32 raw bits for
  normals, subnormals, signed zeros, infinities, and NaNs; and
- exact BOOL Shapes and canonical bytes `0`/`1`, complete direct writes without touching outside
  canaries, stable permutations, mixed typed composition, repetition, and independence.

If any selector, typed binding, Shape, broadcast order, canonical output, semantic case, selected
raw bit, direct write, or permutation gate fails, remove the probe and mark 0010 Blocked. Do not
substitute a custom kernel, CPU/host fallback, conversion, relaxation, or undocumented behavior.

## Native schema, candidates, and lifecycle

Keep native ABI version 4, statuses, and exactly thirteen exports. Bump the pointed-to fixed
160-byte node schema from version 2 to version 3. Preserve operation wires `1..10`, use `11..23` in
the table order above, and reinterpret only the former reserved header word as a third-input index:
absent for unary/binary nodes and required only for `WHERE`. Derive and validate each value's
FLOAT32/BOOL type from node roles before creating placeholders; every use must agree. Native
creation validates arity, sentinels, types, Shapes, broadcasts, topology, targets, and byte widths.
Runtime binding uses the matching MPSGraph type for every feed and direct target.

Bump candidate, compatibility/workload, route-policy, and backend-local codec versions. Identity
includes ordered typed nodes, descriptors, canonical BOOL and predicate/selection policies,
boundary roles, schema/ABI, topology, splat bits, and live target. Old decisions fail closed. New
occurrences use MPSGraph; the custom singleton-NEG route is unchanged.

Metal ingress, run-owned buffers, splats, and direct materialization gain canonical one-byte BOOL.
Cold ingress rejects noncanonical bytes before native submission; BOOL publication returns exact
canonical bytes. CPU/Metal cross-owner transfer remains canonical positive rank `1..16` FLOAT32,
so any mixed-owner BOOL edge fails before analysis.

Analysis declares typed buffers/workspace before assignment; finalization transactionally creates
one reusable Shape/type-specialized executable. Cold binding validates representations once; hot
execution makes one synchronous submission. Existing rollback, closure, repeated/concurrent run,
and independent-session ownership remains. No retry, fallback, hidden copy, per-node invocation,
async execution, pooling, or tuning workflow is added.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects Metal ownership; Metal owns truthful lowering and Runtime receives prepared work.
- [Concrete backends and Metal](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — Metal owns MPSGraph, native schema, storage, and materialization; transfer stays FLOAT32.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  resource requirements, route, and typed binding are fixed before hot execution.
- [Compiler autograd](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — current forward capture is consumed without a backward claim.
- [Metal backend guide](../../../../backend-guide/metal-backend.md) and
  [Tensor API](../../../../api/tensor-api.md) — current lifecycle and predicate semantics.

If BOOL ingress/materialization or typed node validation requires a shared representation, transfer,
Prepare, Runtime, Engine, or architecture-contract change, stop and mark 0010 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0008; current Model comparison/classification/BOOL/WHERE semantics; Compiler
  forward capture and layout closure; Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime
  0016. Metal 0006/0007/0009 and Model 0026 are explicitly not dependencies.
- **Conflicts with:** Metal 0006/0007/0009 and any concurrent Metal capability, preparation, native
  ABI/schema, candidate/codec, storage/materialization, or Engine Metal integration work.
- **Parallel group:** None.
- **Common base revision:** `938a18cd1ab7e02e40fb6b585ec1bd6fcd5e40e2`.
- **Integration order:** disposable selector/type/value probe → schema/type validation → capability
  and topology → BOOL ingress/materialization → finalization/runtime → Engine proof → docs/review.
- **Integration validation:** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparation/candidate/codec types — capability, schema v3, typed topology, geometry, and invalidation.
- `MetalBackendRuntime`, storage, schedule, and finalization types — canonical BOOL lifecycle.
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` — typed schema and selectors.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides, Javadocs,
  glossary terms, task, master plan, and roadmap — exact proofs and status.

## Acceptance criteria

- Capability/preparation admit exactly the listed FLOAT32/BOOL domain and preserve all existing
  routes, affine restrictions, and the FLOAT32-only cross-owner transfer contract.
- The disposable real probe passes every selector, type, canonical-byte, special-value, raw-bit,
  direct-target, Shape, broadcast, permutation, repetition, and independence gate first.
- ABI v4 retains exactly thirteen exports; schema v3 retains 160-byte nodes, adds only wire values
  `11..23` and typed third-input meaning, and rejects malformed or stale records/decisions.
- Fake and real tests cover wrong attrs/types/flags, noncanonical BOOL ingress, broadcast/Shape
  boundaries, overflow, topology, splats, direct publication, rollback, close, repetition,
  concurrency, and independent sessions.
- A Metal-only public Engine run accepts FLOAT32 and canonical BOOL caller inputs, publishes direct
  BOOL predicate and FLOAT32 `WHERE` outputs, composes comparison → logic → selection in one
  partition, and proves no CPU owner. A mixed-owner BOOL edge rejects before analysis.
- No CAST, integer, FLOAT64, BFLOAT16, FLOAT16, scalar/zero/dynamic Shape, view, blocked numerical
  family, custom kernel, backward/training, transfer widening, or performance claim is introduced.
- Independent Class C review returns APPROVE before completion.

## Validation

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
Ready frontier, exactly thirteen exports, unchanged production dependencies/contracts, no custom
predicate kernel, no BOOL cross-owner transfer, and no FLOAT16/BFLOAT16/CAST/backward claim.

## Documentation and review impact

Update Metal Javadocs, package documentation, backend/native guides, and targeted glossary status
after evidence passes. Model/Tensor semantics and architecture contracts remain unchanged.
Independent Class C review inspects capability truth, BOOL canonicalization, special values,
schema/types, compatibility invalidation, lifecycle, Engine evidence, transfer non-widening, and
all exclusions.

## Result

The mandatory probe established an independent blocker before production work. Across both
optimization levels, six independent executables and eight runs per case reproduced subnormal
comparison collapse to zero while raw identity and all named controls passed. The probe and every
other task-local executable change were removed; production remains unchanged.

Status: Incomplete
Follow-up required: provide an exact replacement route or explicitly change the Model numerical
contract before reconsidering these predicates.

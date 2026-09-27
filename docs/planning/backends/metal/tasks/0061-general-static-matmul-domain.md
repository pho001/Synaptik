# Task 0061: General Static MATMUL Domain

## Status

Complete at corrected approved plan `d81ec940` and reviewed implementation head `3913bac1`.
Independent cumulative code, evidence, and security reviews returned zero P0/P1/P2.

## Change class

Class C — this widens profile-qualified Metal capability, changes cold route/resource selection and
native custom execution, advances persistent backend-local identities, and changes public Engine
behavior while preserving the public API, native ABI, schema encoding, exports, and hot-run boundary.

## Goal

Admit the matrix below across every positive static Model rank/batch/broadcast form. Preserve the
reviewed rank-two FLOAT32 MPSGraph route; use one fixed custom whole-program route for new
occurrences. Fix every choice pre-resource, with no timing, autotuning, retry, fallback, or runtime
selector choice.

## Audited contract and bounded decision

Model MATMUL has no attrs, two inputs/one output, ranks `>= 1`, right-aligned leading-batch
broadcast, and vector promotion/removal (vector/vector returns rank zero). Floating promotion is
`BFLOAT16 < FLOAT32 < FLOAT64`; integral promotion is `INT32 < INT64`; BOOL/cross-category pairs
are invalid. Every floating pair sets output `requiresGrad` to the exact input OR; integral pairs
cannot require gradients. FLOAT32/BFLOAT16 results accumulate in FLOAT32, FLOAT64 in FLOAT64, and
integers modulo `2^32`/`2^64`; BFLOAT16 narrows once. Empty contraction is positive zero.

The implementation matrix is closed as follows:

| Left / right | Result | Production profiles/domain | Model output gradient metadata | Fixed route |
|---|---|---|---|---|
| `INT32 / INT32` | `INT32` | both | necessarily no-gradient | custom |
| `INT32 / INT64`, `INT64 / INT32`, `INT64 / INT64` | `INT64` | both | necessarily no-gradient | custom |
| `FLOAT32 / FLOAT32`, exact current rank-two geometry | `FLOAT32` | accelerator | output grad is exact input OR | existing MPSGraph |
| `FLOAT32 / FLOAT32`, every other admitted rank/batch geometry | `FLOAT32` | accelerator | output grad is exact input OR | custom |
| `BFLOAT16 / FLOAT32`, `FLOAT32 / BFLOAT16` | `FLOAT32` | accelerator, all descriptors no-gradient | output grad is exact input OR | custom |
| `BFLOAT16 / BFLOAT16` | `BFLOAT16` | none (Model-valid) | output grad is exact input OR | production-blocked |
| Every Model-valid pair promoted to `FLOAT64` | `FLOAT64` | none | output grad is exact input OR | production-blocked |

For both production-blocked floating families, Task 0061 grants neither Metal forward nor generated
gradient execution; capability rejection does not alter their Model descriptor metadata.

Admitted dimensions are resolved, static, positive, and product/address checked; input rank is
`1..16`, output rank `0..16`, and output is canonical. Zero extents remain false despite Model's
empty-contraction rule because current Metal storage rejects them; no second empty convention is
added. MATMUL has no transpose flag. Coverage means direct/left/right/both forms from an exact local
`PERMUTE` that fixes batch axes and swaps only the final two axes of a canonical rank-`>= 2`
source. Each operand retains its authenticated physical offset/strides; boundary/arbitrary/
overlapping/negative-stride views and rank-one pseudo-transposes stay false.

## Scope

1. Validate the exact Model shape/promotion independently in capability, partition preparation,
   Java ABI preflight, and native ABI preflight.
2. Generalize local affine authentication from `[1,0]` to `[0, ..., r-3, r-1, r-2]`, retaining
   independently checked per-operand physical offset/strides through custom-program execution.
3. Keep wire `15`, `AttributeKind.NONE`, schema 15, ABI 5, type wires `1..6`, operation wires
   `1..115`, attribute wires `0..41`, thirteen exports, and all existing route/image fields.
4. Preserve rank-two same-type FLOAT32 as `MPSGRAPH`, including nested custom-program execution.
   Extend the typed route predicate so only newly admitted MATMUL fixes `CUSTOM_PROGRAM`; Java and
   native must use the same descriptor/rank predicate.
5. Change custom-catalog MATMUL from `PENDING` to `AVAILABLE`. Capability stays `69/46`, structural
   execution `87/28`, MPSGraph catalog `75/35/5`; custom catalog becomes `47/68/0`.
6. Advance candidate, compatibility, route-policy, workload-signature, exact-default-policy, and
   codec identities from 16 to 17. Reject version 16; schema/ABI/exports remain unchanged.
7. Rebuild, sign, package, verify, and prove the dylib through backend, conformance, and public
   no-skip Engine surfaces.

## Fixed custom realization

Add one focused kernel-source owner, reusing custom-program buffers/grid/lifecycle. Cold preparation
selects one of seven ordered-type functions, avoiding per-contributor type branches. One thread owns
one output and visits `k = 0..K-1` once; no atomics, shared accumulators, races, partial outputs, or
hidden allocations.

Each operand metadata carries logical rank/dimensions, authenticated element offset, physical
strides, referenced span, and value state. Canonical state uses offset zero/canonical strides.
For affine state, Java and native independently authenticate the exact local last-two `PERMUTE`,
canonical source, source-buffer identity, permuted strides, offset, and span. After output/batch
decode, logical left coordinates are `[k]` or `[broadcastBatch..., m, k]`; right coordinates are
`[k]` or `[broadcastBatch..., k, n]`. Each broadcast extent one maps to coordinate zero, then each
load uses `offset + sum(logicalCoordinate[axis] * stride[axis])`, checked within the physical span.
Left and right are independent. A nested custom program preserves `AFFINE_VIEW` as an alias of its
source storage; it must never read that value canonically or silently treat a nested PERMUTE as
materialized. Only an explicit materialization node may create canonical state/storage.

- INT32 result uses unsigned 32-bit multiply/add modulo `2^32`; INT64 variants sign-extend INT32
  operands before unsigned 64-bit multiply/add modulo `2^64`.
- BFLOAT16 loads reconstruct FLOAT32 exactly by shifting their represented bits.
- FLOAT32 authorization requires a source/compiler-site certificate: every contributor maps to one
  multiply plus its corresponding add, emitted only as separate FLOAT32 sites or one permitted FMA,
  with no dropped/duplicated/invented/pretruncated term. At every multiply/add/FMA site it proves
  each finite subnormal operand is exact or read as same-signed zero (DAZ), each finite subnormal
  site result is exact or signed zero (FTZ), NaN and signed-infinity class behavior, signed zero,
  and infinity only from genuine overflow. It separately proves final nonempty-MATMUL exact-zero
  sign freedom. Retained safe-math options/emitted sites are evidence; fixtures only corroborate.

Route, layout, and checked geometry are authenticated before resource creation. New singleton
candidates contain only `CUSTOM_PROGRAM`; Run only binds retained resources, dispatches, waits,
and publishes.

## Gradients and callers

Same-type FLOAT32 preserves `requiresGrad(output) = requiresGrad(left) || requiresGrad(right)` and
must prove explicitly seeded vector/vector, vector/matrix, matrix/vector, matrix/matrix, batch
unbroadcast, and last-two-swap formulas. Every generated rank edit, MUL, MATMUL, SUM_TO_SHAPE, and
PERMUTE occurrence passes capability independently. Mixed BFLOAT16/FLOAT32 stays forward
no-gradient: vector-gradient promotion and normalization-to-BFLOAT16 lack complete Metal closure.

Public `Tensor.matmul`, batched biased/unbiased `Tensor.linear`, and seeded MATMUL gradients are in
scope. Compiler MATMUL/attention/convolution gradient rules are audited callers, not new owners;
their higher-rank nodes qualify occurrence by occurrence. Attention wire `35`, complete composite
backward, implicit seeding, scalar-loss training, generic training, and CPU/OpenBLAS changes are out.

## Non-goals and proof blockers

- Do not widen opaque MPSGraph beyond reviewed rank-two FLOAT32. The SDK documents broadcast, not
  complete rank-one, accumulation, rounding, DAZ/FTZ, special-class, or shape-dependent result
  sets; samples, selector names, and fast-math settings do not prove them.
- Strict floating stays false: safe MSL alone does not prove strict GPU subnormals.
  BFLOAT16/BFLOAT16 is a distinct Model-valid BFLOAT16-result family with FLOAT32 accumulation, one
  BFLOAT16 narrowing, and exact input-OR gradient metadata. Because non-FLOAT32 results remain
  strict under both profiles, its forward and gradient execution are blocked on complete strict
  accumulation/narrowing proof. FLOAT64-result forward and gradient execution separately remain
  blocked because Metal has no proved FLOAT64 arithmetic route.
- Do not add empty/dynamic tensors, arbitrary affine feeds, transpose/accumulation flags, fusion,
  tiling/autotuning, performance claims, fallback, or a parallel contract abstraction.

These boundaries delimit, but do not block implementation of, the admitted rows.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) and Core
  invariants — Planning selects; Prepare fixes resources; Run does no discovery/compile/allocation.
- [`foundational-modules.md` — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  and its four floors — exact mapping/contributors, monotonicity, and complete-domain proof.
- [`backend-execution.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — Metal owns routes/native state and profile-qualified identities.
- [`runtime-prepare-engine.md` — Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) lifecycles.

If implementation needs another Model result set, API, schema/ABI field, dependency, or ownership
rule, stop for an architecture decision.

## Dependencies and integration

- Depends on: Metal 0060 Complete through `a4fe4754`/review `9e6dd44d`; current Model MATMUL and
  Compiler gradient contracts.
- Conflicts with: concurrent Metal capability/schema/native/custom/route/package/shared-doc work
  and resumed 0053 production.
- Parallel group: None. Common base: `9e6dd44df17f6ee22e52dfb0cd33d719c5081b9f`.
- Integration order: serial capability/preflight/route/native/tests, then package/docs/status.
- Integration validation: real-native Metal, conformance, public Engine, package, Javadoc/docs, and
  independent cumulative Class C review.
- Shared-document integration owner: serial Task 0061 coordinator.

## Files and symbols

- Audit: Model MATMUL/promotion and Compiler linear/attention/convolution gradient owners/tests; no
  production edit absent a separately approved defect.
- Metal: capability, preparer/plan, graph/catalog, Java ABI, route candidate/batch/codec, Javadocs.
- Native: `synaptik_metal_foundation.m` and one focused MATMUL kernel header.
- Proof: focused Metal tests, backend conformance, public Engine integration.
- Docs: live root/contracts/guides/API/index/glossary/native guide/master/roadmap/task; completed
  predecessor tasks remain historical.

## Acceptance criteria

1. Truth tables prove the matrix, all rank/broadcast/transpose/output-rank boundaries, promotion,
   exact Model gradient metadata including both production-blocked floating families, separate
   forward/gradient capability rejection, zero/dynamic/malformed negatives, strict monotonicity,
   and unchanged `69/46`.
2. Java/native parity proves each operand state/source/offset/strides/span and exact physical
   mapping, including nested custom programs. Spoofed provenance/geometry/state/source/permutation
   fails pre-resource; poisoned affine canaries prove no silent canonical read.
3. The FLOAT32 certificate proves every DAZ/FTZ site, exact-or-flushed subnormal result, separate/
   FMA cases, NaN/infinity, signed zero, genuine overflow, contributors, and final nonempty zero-sign
   freedom; fixtures are corroboration only.
4. Other malformed images independently reject shape/type/overflow and identity 16 pre-resource.
5. Real native tests cover seven signatures, rank pairings/broadcast/rank 16, modular extremes,
   BFLOAT16 reconstruction, FLOAT32 special classes, affine canaries, immutable inputs, reuse,
   independent sessions, and concurrency.
6. Route proofs retain rank-two FLOAT32 MPSGraph, select custom for new singletons, and keep it as a
   nested MPSGraph step when another custom node fixes `CUSTOM_PROGRAM`.
7. Public Engine proves integers under both profiles; accelerator proves general FLOAT32, batched
   linear, transposes, and seeded gradients. BFLOAT16/BFLOAT16 and FLOAT64 forward/gradient graphs,
   strict floating, admitted-mixed gradient, and zero/type/layout negatives fail with no CPU owner.
8. Raw outputs use exact modular values or Model result sets, never tolerance; runs are deterministic.
9. Package proof keeps ABI 5/schema 15/thirteen exports/`87/28`/`75/35/5`, reaches custom `47/68/0`,
   rejects version 16, removes live rank-two claims, and receives zero P0/P1/P2 on final review.

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
./gradlew :backends:metal:javadoc
git diff --check
```

No full build is planned absent a dependency/build/API scope change; packaged-native backend,
conformance, Engine, Javadoc, and review are the stronger gates.

## Documentation and review impact

Update live root/contract/guides/API/glossary/native/Javadoc/master/roadmap claims. Add only a
current-domain ADR note; keep historical Task 0021. Clean documentation and independent cumulative
Class C reviews are mandatory.

## Result

- Implemented through `3cc49d94`, `79a7aaff`, `9bd06972`, `092ef7b2`, `17abb160`,
  `76ae5019`, `6ec0b359`, `d72e8317`, and final bounded capability remediation `3913bac1`.
- Both profiles execute every positive-static no-gradient INT32/INT64 ordered pair with
  INT64-dominant promotion and modular arithmetic. Accelerator executes the complete positive-static
  FLOAT32 rank/vector/batch/broadcast domain with generated gradients plus both no-gradient
  BFLOAT16/FLOAT32 orders. BFLOAT16/BFLOAT16, FLOAT64-result, strict floating, zero/dynamic,
  malformed, and disallowed-gradient forms remain closed.
- Existing rank-two same-type FLOAT32 direct and authenticated left/right/both-transpose MATMUL
  remains MPSGraph, including inside a custom program. Newly admitted geometry and carrier pairs
  use the seven fixed custom signatures with authenticated physical layout/provenance. Schema 15,
  ABI 5, thirteen exports, `69/46` capability, `87/28` structural execution, `75/35/5` MPSGraph,
  and `47/68/0` custom catalog are exact; all backend-local identities are 17 and reject 16.
- Compiler vector/vector gradients retain scalar-vector MUL and the Model numerical result set.
  The Metal-only closure is bounded to accelerator FLOAT32 rank-zero by rank-one MUL with rank-one
  output; every unrelated scalar, rank, operation, profile, and gradient combination remains
  closed. Public Engine proof covers both profiles' integers, general FLOAT32, biased and unbiased
  batched linear, every vector/matrix gradient pairing including an explicit dot seed, and the
  blocked floating families.
- The final signed local package verified. Packaged-native validation passed 187 Metal, 278
  Compiler, 8 Metal conformance, and 25 public Engine tests with zero failures or skips. Metal
  Javadoc, architecture checks, the source/compiler-site certificate, and `git diff --check`
  passed. Independent cumulative code, evidence, and security reviews of exact implementation
  head `3913bac1` each returned `APPROVE` with zero P0/P1/P2.

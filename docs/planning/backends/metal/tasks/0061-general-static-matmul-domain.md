# Task 0061: General Static [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Domain

## Reader context

Model determines vector promotion, batch broadcasting, result type and generated derivatives. Metal authenticates which physical buffer and strides each operand uses. A local PERMUTE can describe a transpose of existing storage; reading it as a canonical copy gives wrong coordinates. Native execution writes a separate output and visits contraction contributors.

This widened geometry without increasing true kind names. Its FLOAT32/mixed-carrier exclusions are the historical domain. A generated backward works only if every emitted rank edit, transpose, reduction and MATMUL is admitted—not merely because the forward Tensor carries a gradient flag.

- [Current operands and generated MATMUL gradients](../../../../backend-guide/metal-backend.md#general-static-matmul-and-generated-gradients) — Connects vector/batch semantics to physical addressing and later typed coverage.

## Status

Complete at corrected approved plan `d81ec940` and reviewed implementation head `3913bac1`.
Independent cumulative code, evidence, and security reviews returned zero P0/P1/P2.

## Change class

Class C — this widens profile-qualified [Metal](../../../../glossary.md#term-metal) capability, changes cold route/resource selection and
native custom execution, advances persistent [backend](../../../../glossary.md#backend)-local [identities](../../../../glossary.md#term-identity-meanings), and changes public [Engine](../../../../glossary.md#term-engine)
behavior while preserving the public [API](../../../../glossary.md#term-application-programming-interface-api), native [ABI](../../../../glossary.md#term-application-binary-interface-abi), [schema](../../../../glossary.md#term-schema) encoding, exports, and hot-[run](../../../../glossary.md#term-run) boundary.

## Goal

Admit the [matrix](../../../../glossary.md#term-matrix-meanings) below across every positive static [Model](../../../../glossary.md#term-model) [rank](../../../../glossary.md#term-rank)/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/broadcast form. Preserve the
reviewed [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../../../../glossary.md#term-float32) [MPSGraph](../../../../glossary.md#term-mpsgraph) route; use one fixed custom whole-program route for new
occurrences. Fix every choice pre-resource, with no timing, [autotuning](../../../../glossary.md#autotuning--model-autotuning), retry, fallback, or [runtime](../../../../glossary.md#term-runtime)
selector choice.

## Audited contract and bounded decision

[Model](../../../../glossary.md#term-model) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) has no attrs, two inputs/one output, ranks `>= 1`, right-aligned leading-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
broadcast, and [vector](../../../../glossary.md#term-vector) promotion/removal (vector/vector returns [rank](../../../../glossary.md#term-rank) zero). Floating promotion is
`BFLOAT16 < FLOAT32 < FLOAT64`; integral promotion is `INT32 < INT64`; [BOOL](../../../../glossary.md#data-type--datatype)/cross-category pairs
are invalid. Every floating pair sets output `requiresGrad` to the exact input OR; integral pairs
cannot require [gradients](../../../../glossary.md#term-gradient). [FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16) results accumulate in FLOAT32, [FLOAT64](../../../../glossary.md#term-float64) in FLOAT64, and
integers modulo `2^32`/`2^64`; BFLOAT16 narrows once. Empty [contraction](../../../../glossary.md#contraction-dimension) is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero).

The implementation [matrix](../../../../glossary.md#term-matrix-meanings) is closed as follows:

| Left / right | Result | Production profiles/domain | Model output gradient [metadata](../../../../glossary.md#term-metadata) | Fixed route |
|---|---|---|---|---|
| `INT32 / INT32` | [`INT32`](../../../../glossary.md#data-type--datatype) | both | necessarily no-gradient | custom |
| `INT32 / INT64`, `INT64 / INT32`, `INT64 / INT64` | [`INT64`](../../../../glossary.md#data-type--datatype) | both | necessarily no-gradient | custom |
| `FLOAT32 / FLOAT32`, exact current [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) geometry | `FLOAT32` | [accelerator](../../../../glossary.md#term-accelerator) | output grad is exact input OR | existing [MPSGraph](../../../../glossary.md#term-mpsgraph) |
| `FLOAT32 / FLOAT32`, every other admitted rank/batch geometry | `FLOAT32` | accelerator | output grad is exact input OR | custom |
| `BFLOAT16 / FLOAT32`, `FLOAT32 / BFLOAT16` | `FLOAT32` | accelerator, all [descriptors](../../../../glossary.md#term-descriptor) no-gradient | output grad is exact input OR | custom |
| `BFLOAT16 / BFLOAT16` | `BFLOAT16` | none (Model-valid) | output grad is exact input OR | production-blocked |
| Every Model-valid pair promoted to `FLOAT64` | `FLOAT64` | none | output grad is exact input OR | production-blocked |

For both production-blocked floating families, Task 0061 grants neither [Metal](../../../../glossary.md#term-metal) forward nor generated
gradient execution; capability rejection does not alter their Model descriptor metadata.

Admitted [dimensions](../../../../glossary.md#dimension) are resolved, static, positive, and product/address checked; input rank is
`1..16`, output rank `0..16`, and output is [canonical](../../../../glossary.md#term-canonical-form). Zero extents remain false despite Model's
empty-contraction rule because current Metal storage rejects them; no second empty convention is
added. MATMUL has no [transpose](../../../../glossary.md#term-transpose) flag. Coverage means direct/left/right/both forms from an exact local
`PERMUTE` that fixes batch [axes](../../../../glossary.md#term-axis) and swaps only the final two axes of a canonical rank-`>= 2`
source. Each operand retains its authenticated physical offset/[strides](../../../../glossary.md#term-stride); boundary/arbitrary/
overlapping/negative-stride [views](../../../../glossary.md#term-view-meanings) and [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) pseudo-transposes stay false.

## [Scope](../../../../glossary.md#term-scope)

1. Validate the exact Model shape/promotion independently in capability, partition preparation,
   Java ABI preflight, and native ABI preflight.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [shape glossary](../../../../glossary.md#shape); [partition glossary](../../../../glossary.md#partition); [preflight glossary](../../../../glossary.md#term-preflight))</span>
2. Generalize local affine [authentication](../../../../glossary.md#term-authentication) from `[1,0]` to `[0, ..., r-3, r-1, r-2]`, retaining
   independently checked per-operand physical offset/strides through custom-program execution.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>
3. Keep wire `15`, `AttributeKind.NONE`, [schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, type wires `1..6`, [operation](../../../../glossary.md#operation) wires
   `1..115`, attribute wires `0..41`, thirteen exports, and all existing route/image fields.
4. Preserve [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) same-type [FLOAT32](../../../../glossary.md#term-float32) as [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph), including nested custom-program execution.
   Extend the typed route predicate so only newly admitted MATMUL fixes `CUSTOM_PROGRAM`; Java and
   native must use the same descriptor/rank predicate.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [rank glossary](../../../../glossary.md#term-rank))</span>
5. Change custom-catalog [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) from `PENDING` to `AVAILABLE`. Capability stays `69/46`, structural
   execution `87/28`, MPSGraph catalog `75/35/5`; custom catalog becomes `47/68/0`.
6. Advance candidate, compatibility, route-policy, workload-signature, exact-default-policy, and
   codec identities from 16 to 17. Reject version 16; schema/ABI/exports remain unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>
7. Rebuild, sign, package, verify, and prove the dylib through backend, conformance, and public
   no-skip Engine surfaces.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Fixed custom realization

Add one focused [kernel](../../../../glossary.md#kernel)-source [owner](../../../../glossary.md#term-owner), reusing custom-program [buffers](../../../../glossary.md#term-buffer)/grid/[lifecycle](../../../../glossary.md#lifecycle). Cold preparation
selects one of seven ordered-type functions, avoiding per-contributor type branches. One thread owns
one output and visits `k = 0..K-1` once; no atomics, shared accumulators, races, partial outputs, or
hidden allocations.

Each operand [metadata](../../../../glossary.md#term-metadata) carries logical [rank](../../../../glossary.md#term-rank)/[dimensions](../../../../glossary.md#dimension), authenticated [element](../../../../glossary.md#term-element) offset, physical
[strides](../../../../glossary.md#term-stride), [referenced span](../../../../glossary.md#referenced-element-span), and value state. [Canonical](../../../../glossary.md#term-canonical-form) state uses offset zero/canonical strides.
For affine state, Java and native independently authenticate the exact local last-two `PERMUTE`,
canonical source, source-buffer [identity](../../../../glossary.md#term-identity-meanings), permuted strides, offset, and span. After output/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
decode, logical left [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) are `[k]` or `[broadcastBatch..., m, k]`; right coordinates are
`[k]` or `[broadcastBatch..., k, n]`. Each broadcast [extent](../../../../glossary.md#term-extent) one maps to coordinate zero, then each
load uses `offset + sum(logicalCoordinate[axis] * stride[axis])`, checked within the physical span.
Left and right are independent. A nested custom program preserves `AFFINE_VIEW` as an [alias](../../../../glossary.md#aliasing-and-arena) of its
source storage; it must never read that value canonically or silently treat a nested PERMUTE as
materialized. Only an explicit [materialization](../../../../glossary.md#materialization) [node](../../../../glossary.md#node) may create canonical state/storage.

- [INT32](../../../../glossary.md#data-type--datatype) result uses unsigned 32-bit multiply/add modulo `2^32`; INT64 variants sign-extend INT32
  operands before unsigned 64-bit multiply/add modulo <span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`2^64`.
- [BFLOAT16](../../../../glossary.md#term-bfloat16) loads reconstruct [FLOAT32](../../../../glossary.md#term-float32) exactly by shifting their [represented bits](../../../../glossary.md#term-represented-bits).
- FLOAT32 authorization requires a source/compiler-site certificate: every contributor maps to one
  multiply plus its corresponding add, emitted only as separate FLOAT32 sites or one permitted FMA,
  with no dropped/duplicated/invented/pretruncated term. At every multiply/add/FMA site it proves
  each finite subnormal operand is exact or read as same-signed zero (DAZ), each finite subnormal
  site result is exact or signed zero (FTZ), NaN and signed-infinity class behavior, signed zero,
  and infinity only from genuine overflow. It separately proves final nonempty-MATMUL exact-zero
  sign freedom. Retained safe-math options/emitted sites are evidence; fixtures only corroborate.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [finite glossary](../../../../glossary.md#term-finite); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed zero glossary](../../../../glossary.md#term-signed-zero); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [overflow glossary](../../../../glossary.md#term-overflow); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

Route, [layout](../../../../glossary.md#layout), and checked geometry are authenticated before resource creation. New [singleton](../../../../glossary.md#term-singleton)
candidates contain only `CUSTOM_PROGRAM`; [Run](../../../../glossary.md#term-run) only binds retained resources, dispatches, waits,
and publishes.

## [Gradients](../../../../glossary.md#term-gradient) and callers

Same-type [FLOAT32](../../../../glossary.md#term-float32) preserves `requiresGrad(output) = requiresGrad(left) || requiresGrad(right)` and
must prove explicitly seeded [vector](../../../../glossary.md#term-vector)/vector, vector/[matrix](../../../../glossary.md#term-matrix-meanings), matrix/vector, matrix/matrix, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
unbroadcast, and last-two-swap formulas. Every generated [rank](../../../../glossary.md#term-rank) edit, MUL, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), SUM_TO_SHAPE, and
PERMUTE [occurrence](../../../../glossary.md#term-occurrence) passes capability independently. Mixed [BFLOAT16](../../../../glossary.md#term-bfloat16)/FLOAT32 stays forward
no-gradient: vector-gradient promotion and [normalization](../../../../glossary.md#term-normalization)-to-BFLOAT16 lack complete [Metal](../../../../glossary.md#term-metal) closure.

Public `Tensor.matmul`, batched biased/unbiased `Tensor.linear`, and seeded MATMUL gradients are in
[scope](../../../../glossary.md#term-scope). [Compiler](../../../../glossary.md#term-compiler) MATMUL/attention/[convolution](../../../../glossary.md#term-channels-first-convolution-layers) gradient rules are audited callers, not new [owners](../../../../glossary.md#term-owner);
their higher-rank [nodes](../../../../glossary.md#node) qualify occurrence by occurrence. Attention wire `35`, complete composite
backward, implicit seeding, [scalar](../../../../glossary.md#term-scalar)-[loss](../../../../glossary.md#loss) training, generic training, and [CPU](../../../../glossary.md#term-cpu)/[OpenBLAS](../../../../glossary.md#term-openblas) changes are out.

## Non-goals and proof blockers

- Do not widen opaque MPSGraph beyond reviewed rank-two FLOAT32. The SDK documents broadcast, not
  complete rank-one, accumulation, rounding, DAZ/FTZ, special-class, or shape-dependent result
  sets; samples, selector names, and fast-math settings do not prove them.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [SDK glossary](../../../../glossary.md#term-sdk); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [shape glossary](../../../../glossary.md#shape); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Strict floating stays false: safe MSL alone does not prove strict GPU subnormals.
  BFLOAT16/BFLOAT16 is a distinct Model-valid BFLOAT16-result family with FLOAT32 accumulation, one
  BFLOAT16 narrowing, and exact input-OR gradient metadata. Because non-FLOAT32 results remain
  strict under both profiles, its forward and gradient execution are blocked on complete strict
  accumulation/narrowing proof. FLOAT64-result forward and gradient execution separately remain
  blocked because Metal has no proved FLOAT64 arithmetic route.<span data-glossary-navigation="true"> ([MSL glossary](../../../../glossary.md#term-metal-shading-language-msl); [GPU glossary](../../../../glossary.md#term-gpu); [subnormals glossary](../../../../glossary.md#term-subnormal); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Model glossary](../../../../glossary.md#term-model); [narrowing glossary](../../../../glossary.md#term-narrowing); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Do not add empty/dynamic tensors, arbitrary affine feeds, transpose/accumulation flags, fusion,
  tiling/autotuning, performance claims, fallback, or a parallel contract abstraction.<span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [transpose glossary](../../../../glossary.md#term-transpose); [fusion glossary](../../../../glossary.md#term-fusion); [tiling glossary](../../../../glossary.md#tiling); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>

These boundaries delimit, but do not block implementation of, the admitted rows.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> and Core
  invariants — Planning selects; Prepare fixes resources; Run does no discovery/compile/allocation.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Run glossary](../../../../glossary.md#term-run); [compile glossary](../../../../glossary.md#compile))</span>
- [`foundational-modules.md` — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  and its four floors — exact mapping/contributors, monotonicity, and complete-domain proof.
- [`backend-execution.md` — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span>
  and [backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  — Metal owns routes/native state and profile-qualified identities.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- [`runtime-prepare-engine.md` — Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) lifecycles.

If implementation needs another [Model](../../../../glossary.md#term-model) result set, [API](../../../../glossary.md#term-application-programming-interface-api), [schema](../../../../glossary.md#term-schema)/[ABI](../../../../glossary.md#term-application-binary-interface-abi) field, [dependency](../../../../glossary.md#term-dependency), or [ownership](../../../../glossary.md#term-ownership-meanings)
rule, stop for an architecture decision.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0060 Complete through `a4fe4754`/review `9e6dd44d`; current Model MATMUL and
  Compiler gradient contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Conflicts with: concurrent Metal capability/schema/native/custom/route/package/shared-doc work
  and resumed 0053 production.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Parallel group: None. Common base: `9e6dd44df17f6ee22e52dfb0cd33d719c5081b9f`.
- Integration order: serial capability/[preflight](../../../../glossary.md#term-preflight)/route/native/tests, then package/docs/status.
- Integration validation: real-native Metal, conformance, public Engine, package, Javadoc/docs, and
  independent cumulative Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): serial Task 0061 coordinator.

## Files and symbols

- Audit: Model MATMUL/promotion and Compiler linear/attention/convolution gradient owners/tests; no
  production edit absent a separately approved defect.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Metal](../../../../glossary.md#term-metal): capability, preparer/plan, [graph](../../../../glossary.md#graph)/catalog, Java [ABI](../../../../glossary.md#term-application-binary-interface-abi), route candidate/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[codec](../../../../glossary.md#term-codec), Javadocs.
- Native: `synaptik_metal_foundation.m` and one focused [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [kernel](../../../../glossary.md#kernel) header.
- Proof: focused Metal tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine) integration.
- Docs: live root/contracts/guides/API/index/glossary/native guide/master/roadmap/task; completed
  predecessor tasks remain historical.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

1. Truth tables prove the matrix, all rank/broadcast/transpose/output-rank boundaries, promotion,
   exact Model gradient metadata including both production-blocked floating families, separate
   forward/gradient capability rejection, zero/dynamic/malformed negatives, strict monotonicity,
   and unchanged <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [rank glossary](../../../../glossary.md#term-rank); [transpose glossary](../../../../glossary.md#term-transpose); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata))</span>`69/46`.
2. Java/native parity proves each operand state/source/offset/strides/span and exact physical
   mapping, including nested custom programs. Spoofed provenance/geometry/state/source/permutation
   fails pre-resource; poisoned affine canaries prove no silent canonical read.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [provenance glossary](../../../../glossary.md#provenance); [permutation glossary](../../../../glossary.md#permutation); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
3. The FLOAT32 certificate proves every DAZ/FTZ site, exact-or-flushed subnormal result, separate/
   FMA cases, NaN/infinity, signed zero, genuine overflow, contributors, and final nonempty zero-sign
   freedom; fixtures are corroboration only.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [subnormal glossary](../../../../glossary.md#term-subnormal); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
4. Other malformed images independently reject [shape](../../../../glossary.md#shape)/type/[overflow](../../../../glossary.md#term-overflow) and [identity](../../../../glossary.md#term-identity-meanings) 16 pre-resource.
5. Real native tests cover seven signatures, rank pairings/broadcast/rank 16, modular extremes,
   BFLOAT16 reconstruction, FLOAT32 special classes, affine canaries, immutable inputs, reuse,
   independent sessions, and concurrency.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [immutable glossary](../../../../glossary.md#term-immutable); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
6. Route proofs retain rank-two FLOAT32 MPSGraph, select custom for new singletons, and keep it as a
   nested MPSGraph step when another custom node fixes <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [node glossary](../../../../glossary.md#node))</span>`CUSTOM_PROGRAM`.
7. Public Engine proves integers under both profiles; accelerator proves general FLOAT32, batched
   linear, transposes, and seeded gradients. BFLOAT16/BFLOAT16 and FLOAT64 forward/gradient graphs,
   strict floating, admitted-mixed gradient, and zero/type/layout negatives fail with no CPU owner.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [accelerator glossary](../../../../glossary.md#term-accelerator); [linear glossary](../../../../glossary.md#term-linear-meanings); [FLOAT64 glossary](../../../../glossary.md#term-float64); [graphs glossary](../../../../glossary.md#graph); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner))</span>
8. Raw outputs use exact modular values or [Model](../../../../glossary.md#term-model) result sets, never [tolerance](../../../../glossary.md#term-tolerance); runs are deterministic.
9. Package proof keeps [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5/[schema](../../../../glossary.md#term-schema) 15/thirteen exports/`87/28`/`75/35/5`, reaches custom `47/68/0`,
   rejects version 16, removes live rank-two claims, and receives zero P0/P1/P2 on final review.

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
./gradlew :backends:metal:javadoc
git diff --check
```

No full build is planned absent a [dependency](../../../../glossary.md#term-dependency)/build/[API](../../../../glossary.md#term-application-programming-interface-api) [scope](../../../../glossary.md#term-scope) change; packaged-native [backend](../../../../glossary.md#backend),
[conformance](../../../../glossary.md#term-conformance), [Engine](../../../../glossary.md#term-engine), [Javadoc](../../../../glossary.md#term-javadoc), and review are the stronger gates.

## Documentation and review impact

Update live root/contract/guides/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary/native/[Javadoc](../../../../glossary.md#term-javadoc)/master/roadmap claims. Add only a
current-domain [ADR](../../../../glossary.md#term-architecture-decision-record-adr) note; keep historical Task 0021. Clean documentation and independent cumulative
Class C reviews are mandatory.

## Result

- Implemented through `3cc49d94`, `79a7aaff`, `9bd06972`, `092ef7b2`, `17abb160`,
  `76ae5019`, `6ec0b359`, `d72e8317`, and final bounded capability remediation `3913bac1`.
- Both profiles execute every positive-static no-gradient INT32/INT64 ordered pair with
  INT64-dominant promotion and modular arithmetic. Accelerator executes the complete positive-static
  FLOAT32 rank/vector/batch/broadcast domain with generated gradients plus both no-gradient
  BFLOAT16/FLOAT32 orders. BFLOAT16/BFLOAT16, FLOAT64-result, strict floating, zero/dynamic,
  malformed, and disallowed-gradient forms remain closed.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [modular arithmetic glossary](../../../../glossary.md#term-modular-arithmetic); [Accelerator glossary](../../../../glossary.md#term-accelerator); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank glossary](../../../../glossary.md#term-rank); [vector glossary](../../../../glossary.md#term-vector); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>
- Existing rank-two same-type FLOAT32 direct and authenticated left/right/both-transpose MATMUL
  remains MPSGraph, including inside a custom program. Newly admitted geometry and carrier pairs
  use the seven fixed custom signatures with authenticated physical layout/provenance. Schema 15,
  ABI 5, thirteen exports, <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [transpose glossary](../../../../glossary.md#term-transpose); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [carrier glossary](../../../../glossary.md#term-storage-carrier); [signatures glossary](../../../../glossary.md#term-signature); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [Schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>`69/46` capability, `87/28` structural execution, `75/35/5` MPSGraph,
  and `47/68/0` custom catalog are exact; all [backend](../../../../glossary.md#backend)-local [identities](../../../../glossary.md#term-identity-meanings) are 17 and reject 16.
- Compiler vector/vector gradients retain scalar-vector MUL and the Model numerical result set.
  The Metal-only closure is bounded to accelerator FLOAT32 rank-zero by rank-one MUL with rank-one
  output; every unrelated scalar, rank, operation, profile, and gradient combination remains
  closed. Public Engine proof covers both profiles' integers, general FLOAT32, biased and unbiased
  batched linear, every vector/matrix gradient pairing including an explicit dot seed, and the
  blocked floating families.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [scalar glossary](../../../../glossary.md#term-scalar); [Model glossary](../../../../glossary.md#term-model); [Metal glossary](../../../../glossary.md#term-metal); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine); [linear glossary](../../../../glossary.md#term-linear-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [seed glossary](../../../../glossary.md#term-seed))</span>
- The final signed local package verified. Packaged-native validation passed 187 Metal, 278
  Compiler, 8 Metal conformance, and 25 public Engine tests with zero failures or skips. Metal
  Javadoc, architecture checks, the source/compiler-site certificate, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>`git diff --check`
  passed. Independent cumulative code, evidence, and security reviews of exact implementation
  head `3913bac1` each returned `APPROVE` with zero P0/P1/P2.

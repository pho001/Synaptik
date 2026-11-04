# Task 0040: Profile-Common Exact [Canonical](../../../../glossary.md#term-canonical-form) [BFLOAT16](../../../../glossary.md#term-bfloat16) [Gather](../../../../glossary.md#gather)

## Reader context

Exact Gather meant copying the selected 16-bit BFLOAT16 representation, including its NaN
sign, payload and signaling state, not merely producing a numerically equivalent NaN.
Canonical layout describes address order; NaN canonicalization changes a stored value.
The direct selector failed that distinction at the first checked output.

Both input buffers were checked unchanged, but the early failure left later output words and
target guards unchecked. The implementation requirements below therefore remained conditional,
not delivered ingress, publication or allocated schema values. Model owned exact selection;
Metal could not repair the result on the host or silently substitute another type.
The [current original-carrier exact movement domain](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
provides the later route context without widening this sole failed execution's evidence.

## Status

Blocked

The mandatory one-execution gate failed from exact clean [planning](../../../../glossary.md#term-planning) revision
`c300582727ec568a7482dd974f9d1b2e2e13f82c` before production edits. The first reversed output
canonicalized `0xffa6` to `0x7fc0`, so the direct [MPSGraph](../../../../glossary.md#term-mpsgraph) [Gather](../../../../glossary.md#gather) selector cannot support the
required raw-bit contract. No [Metal](../../../../glossary.md#term-metal) task is [Ready](../../../../glossary.md#term-ready).

## Change class

Class C — implementation changes a [backend capability](../../../../glossary.md#backend-capability), private native [node](../../../../glossary.md#node)/type [schema](../../../../glossary.md#term-schema), caller
storage [admission](../../../../glossary.md#term-admission), local [materialization](../../../../glossary.md#materialization), and [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings) while preserving [ABI](../../../../glossary.md#term-application-binary-interface-abi) and cross-[owner](../../../../glossary.md#term-owner)
transfer boundaries. It requires one clean implementation context and independent lean Class C
review.

## Goal

Deliver one exact profile-common [Metal](../../../../glossary.md#term-metal) execution path for [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [BFLOAT16](../../../../glossary.md#term-bfloat16) data
[`GATHER`](../../../../glossary.md#gather) with canonical [INT32](../../../../glossary.md#data-type--datatype) indices. Preserve selected raw 16-bit representations from public
caller ingress through Metal-local chaining and [detached](../../../../glossary.md#detached-model-results) [host](../../../../glossary.md#term-host) [publication](../../../../glossary.md#term-publication), without BFLOAT16
arithmetic, conversion, [CPU](../../../../glossary.md#term-cpu) fallback, or CPU/Metal transfer [widening](../../../../glossary.md#term-widening).

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly `AxisGatherKind.GATHER` with `IndexAxisAttrs`, ordered `[data, indices]`, and one
  output. Data and output are BFLOAT16; indices are INT32.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype))</span>
- Require fully static Shapes with strictly positive extents and checked byte geometry. Data and
  indices ranks are <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>`1..16`; [normalized axis](../../../../glossary.md#normalized-axis) is in the data [rank](../../../../glossary.md#term-rank); output rank is `1..16` and its
  Shape is `data[:axis] + indices.shape + data[axis + 1:]`.
- Require canonical dense-contiguous, zero-offset, non-view layouts for both inputs and output.
  Data/output <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [layouts glossary](../../../../glossary.md#layout))</span>`requiresGrad` values remain equal [metadata](../../../../glossary.md#term-metadata); no backward execution is claimed.
- Copy the exact selected BFLOAT16 raw short. Signed zeros, subnormals, ordinary normals,
  infinities, and NaN sign/payload/signaling representations are never widened, rounded, decoded,
  canonicalized, or recreated numerically.<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Advertise the identical [occurrence](../../../../glossary.md#term-occurrence) and [represented-bit](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) behavior under `STRICT_IEEE` and
  [`ACCELERATOR`](../../../../glossary.md#term-accelerator); profile [identity](../../../../glossary.md#term-identity-meanings) remains distinct and cold.
- Admit caller-owned raw-short BFLOAT16 host storage through Metal upload without conversion.
  Admit a locally produced canonical BFLOAT16 Gather output as data for another BFLOAT16 Gather in
  the same Metal partition.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [Metal glossary](../../../../glossary.md#term-metal); [Gather glossary](../../../../glossary.md#gather); [partition glossary](../../../../glossary.md#partition))</span>
- Publish a Metal-local canonical BFLOAT16 target directly as canonical row-major big-endian
  two-byte elements through the owning Metal integration and existing <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [row-major glossary](../../../../glossary.md#term-row-major); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [elements glossary](../../../../glossary.md#term-element))</span>[`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) contract.
- Reuse current complete INT32 prevalidation, stable node/ordinal failure order, no-dispatch and
  no-target-write failure behavior, and exact public bounds exception reconstruction.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

### Native [schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Advance [node](../../../../glossary.md#node) schema `11 -> 12` and append only type-specific `BFLOAT16_GATHER=20` with two
  inputs, `AXIS=3`, and [canonical](../../../../glossary.md#term-canonical-form) output. Keep [FLOAT32](../../../../glossary.md#term-float32) `GATHER=16` unchanged.
- Add inferred local value type `BFLOAT16=4`, width two, mapped to `MPSDataTypeBFloat16` only on its
  documented available runtime. A type-specific operation wire is required because ABI v4 carries
  no per-value type table.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation))</span>
- Keep [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, the fixed 160-byte node record, exactly thirteen exports, attribute wires `0..6`,
  existing status/state/reduction wires, and every C signature unchanged. Add no attribute wire 7.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Advance workload, exact-policy, candidate, compatibility, route-policy, and codec identities
  from 12 to 13. Add BFLOAT16 workload type identity 4. Reject schema 11 and identity 12; add no
  dual reader, migration shim, alias, or fallback.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Infer BFLOAT16/INT32/BFLOAT16 for wire 20 in Java and native preflight, reject conflicting value
  reuse, and lower wire 20 through the existing batch-dimensions-zero Gather selector. Retain the
  existing MPSGRAPH route; custom NEG remains unrelated.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [preflight glossary](../../../../glossary.md#term-preflight); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [dimensions glossary](../../../../glossary.md#dimension); [Gather glossary](../../../../glossary.md#gather); [MPSGRAPH glossary](../../../../glossary.md#term-mpsgraph))</span>

### One mandatory lean [device](../../../../glossary.md#term-device) gate

The installed headers establish the [BFLOAT16](../../../../glossary.md#term-bfloat16) representation and generic [Gather](../../../../glossary.md#gather) structure, but not
the selector's concrete raw-bit movement. Before production edits, [run](../../../../glossary.md#term-run) exactly one disposable [graph](../../../../glossary.md#graph),
one [compile](../../../../glossary.md#compile), and one execution with one [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) BFLOAT16 feed and one reverse [INT32](../../../../glossary.md#data-type--datatype) [permutation](../../../../glossary.md#permutation).
The small raw-short feed is exactly:

```text
0000 8000 0001 8001 007f 807f 3f80 bf80 4120 c120
7f80 ff80 7fc1 7fe5 ffc2 ffe6 7f81 7fa5 ff82 ffa6
```

This covers positive/[negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), minimum/maximum [subnormal](../../../../glossary.md#term-subnormal), ordinary normals, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and
multiple positive/negative quiet-[NaN](../../../../glossary.md#term-nan) and signaling-NaN payloads. The single execution must produce
the exact reverse raw-short sequence, preserve both input [buffers](../../../../glossary.md#term-buffer) byte-for-byte, and preserve
prefilled [target](../../../../glossary.md#term-target) [guard](../../../../glossary.md#term-guard) bytes. Do not enumerate all 65,536 patterns, repeat contexts/runs, or add a
[Shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis)/[operation](../../../../glossary.md#operation) [matrix](../../../../glossary.md#term-matrix-meanings).

If [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), execution, any raw word, either input, or a guard fails, remove the probe, mark this
task Blocked, and make no production change. Do not narrow the corpus, compare numerically,
canonicalize NaNs, widen through [FLOAT32](../../../../glossary.md#term-float32), substitute [`CONTIGUOUS`](../../../../glossary.md#term-contiguous), or fall back to [host](../../../../glossary.md#term-host)/[CPU](../../../../glossary.md#term-cpu) work.
Remove every disposable probe [artifact](../../../../glossary.md#term-artifact) after either outcome.

## Blocker evidence

The sole gate ran once on the M3 Max from exact clean [planning](../../../../glossary.md#term-planning) revision
`c300582727ec568a7482dd974f9d1b2e2e13f82c`. One disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source compiled against
Foundation, [Metal](../../../../glossary.md#term-metal), and MetalPerformanceShadersGraph. Its executable created one Metal [device](../../../../glossary.md#term-device) and
queue, one [MPSGraph](../../../../glossary.md#term-mpsgraph) with one [BFLOAT16](../../../../glossary.md#term-bfloat16) data placeholder, one [INT32](../../../../glossary.md#data-type--datatype) index placeholder, and the direct
[axis](../../../../glossary.md#term-axis)-zero/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[dimensions](../../../../glossary.md#dimension)-zero [Gather](../../../../glossary.md#gather) [target](../../../../glossary.md#term-target). It performed one [graph](../../../../glossary.md#graph) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) with
`MPSGraphReducedPrecisionFastMathNone` and `MPSGraphOptionsNone`, then one [synchronous](../../../../glossary.md#term-synchronous) execution
over the exact 20-word corpus and reverse [permutation](../../../../glossary.md#permutation) above.

The probe compared both complete input [buffers](../../../../glossary.md#term-buffer) byte-for-byte and found them unchanged. It then
stopped at the first output mismatch: output ordinal zero selected input ordinal 19, required
negative signaling-[NaN](../../../../glossary.md#term-nan) word `0xffa6`, and returned positive [canonical](../../../../glossary.md#term-canonical-form) quiet-NaN word `0x7fc0`.
The required sign, payload, and signaling representation were therefore lost. Remaining output
words and the prefilled target [guard](../../../../glossary.md#term-guard) were not checked after that decisive failure and make no
success claim.

The source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) were removed immediately. No second execution, retry, alternate profile,
[Shape](../../../../glossary.md#shape)/axis/context/option [matrix](../../../../glossary.md#term-matrix-meanings), exhaustive corpus, production/test edit, native build, [schema](../../../../glossary.md#term-schema)/
[identity](../../../../glossary.md#term-identity-meanings)/capability change, [narrowing](../../../../glossary.md#term-narrowing), [host](../../../../glossary.md#term-host) repair, fallback, or full build exists. Task 0040 is
`Blocked`. Unblocking requires a separately planned exact replacement route; do not retry or weaken
this direct-selector contract.

## Non-goals

[BFLOAT16](../../../../glossary.md#term-bfloat16) arithmetic, [unary](../../../../glossary.md#term-unary)/reduction/[matrix](../../../../glossary.md#term-matrix-meanings) operations, affine/[view](../../../../glossary.md#term-view-meanings) input or [publication](../../../../glossary.md#term-publication),
[`CONTIGUOUS`](../../../../glossary.md#term-contiguous), window operations, casts, [mixed precision](../../../../glossary.md#term-mixed-precision), [scalar](../../../../glossary.md#term-scalar) or logical-splat [constants](../../../../glossary.md#term-constant),
[INT64](../../../../glossary.md#data-type--datatype)/BFLOAT16 indices, zero/dynamic/scalar [Shapes](../../../../glossary.md#shape), noncanonical [layouts](../../../../glossary.md#layout), in-place output,
backward/training, custom [kernels](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host) result repair, retry/fallback, [CPU](../../../../glossary.md#term-cpu) production changes,
BFLOAT16 CPU/[Metal](../../../../glossary.md#term-metal) transfer, public [API](../../../../glossary.md#term-application-programming-interface-api) additions, packaging/discovery, or performance claims.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and
  [compile ownership and mixed-owner routing](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

If implementation needs broader [Model](../../../../glossary.md#term-model) semantics, [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion/arithmetic, a new [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export,
shared type tables, noncanonical [publication](../../../../glossary.md#term-publication), cross-[owner](../../../../glossary.md#term-owner) BFLOAT16 transfer, [Compiler](../../../../glossary.md#term-compiler) rewriting,
[Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) profile state, or different invalid-index semantics, stop and report the conflict.

## Dependencies and integration

- Depends on: Metal 0023, 0025, and 0038 Complete; current Model BFLOAT16 representation and
  GATHER semantics; Compiler indexing capture; CPU 0006A2 raw-bit oracle; Config 0006; Engine 0018<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [GATHER glossary](../../../../glossary.md#gather); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [oracle glossary](../../../../glossary.md#term-test-oracle); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal capability/schema/native/type/ingress/publication/
  identity/transfer scope and any restart that would consume the same next schema values<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [publication glossary](../../../../glossary.md#term-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `b495e4e43ce5bdcd962256f9bfc726301f3d1549`
- Integration order: the sole mandatory gate failed; no implementation step is authorized
- Integration [validation](../../../../glossary.md#term-validation): [planning](../../../../glossary.md#term-planning) blocker evidence, status [synchronization](../../../../glossary.md#term-synchronization), Markdown, and diff
- Shared-document integration [owner](../../../../glossary.md#term-owner): None while Blocked

## Files and symbols

- [Metal](../../../../glossary.md#term-metal) capability/preparation/schedule/[runtime](../../../../glossary.md#term-runtime): `MetalCapabilityProvider`,
  `MetalNegPartitionPreparer`, `MetalNegPreparedScheduleAssembler`, `MetalBackendRuntime`, and
  `MetalBackendIntegration`.
- [Schema](../../../../glossary.md#term-schema)/native/error handling: `MetalMpsGraphProgram`, `MetalNativeApi.MpsGraphExecutableAbi`,
  `MetalNegPreparedExecutable`, and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`.
- [Identity](../../../../glossary.md#term-identity-meanings): `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, and `MetalNegTuningCodec`.
- Focused Metal capability, raw-ABI, prepared execution, candidate/codec, native indexing,
  conformance, and public Engine integration tests.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [codec glossary](../../../../glossary.md#term-codec); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- After behavior passes: affected Javadocs/package information, `ARCHITECTURE.md`, the owning
  scoped contracts, current Metal/backend/API guides, native README, glossary impact, this brief,
  Metal master plan, and repository roadmap.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

1. The one mandatory smoke uses only the listed representative vector and one graph/compile/run;
   exact reversed output, unchanged inputs, and unchanged guards pass, and artifacts are removed.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [vector glossary](../../../../glossary.md#term-vector); [graph glossary](../../../../glossary.md#graph); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
2. Both profiles admit exactly the canonical BFLOAT16+INT32 Gather domain with identical bit
   semantics; every named exclusion fails before native resource creation.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [Gather glossary](../../../../glossary.md#gather))</span>
3. Caller raw-short BFLOAT16 ingress, same-partition BFLOAT16 Gather chaining, and Metal-local
   big-endian BFLOAT16 publication preserve every tested raw word without conversion.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Metal glossary](../../../../glossary.md#term-metal); [big-endian glossary](../../../../glossary.md#text-encoding-and-file-boundaries); [publication glossary](../../../../glossary.md#term-publication))</span>
4. Existing complete INT32 validation still precedes tensor-data construction, dispatch, and target
   writes; one focused BFLOAT16 invalid-index test proves unchanged inputs/target and diagnostics.<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [dispatch glossary](../../../../glossary.md#term-dispatch); [target glossary](../../../../glossary.md#term-target))</span>
5. [Schema](../../../../glossary.md#term-schema) 12 adds only wire 20 and type 4/width two. [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, 160 bytes, thirteen exports, attributes
   `0..6`, and [FLOAT32](../../../../glossary.md#term-float32) wire 16 remain exact; Java/native [validation](../../../../glossary.md#term-validation) agree.
6. All six identity versions are 13, BFLOAT16 workload type is 4, and stale schema-11/version-12,
   cross-profile, cross-type, malformed, and foreign decisions reject without compatibility shims.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
7. CPU/Metal transfer remains positive-rank canonical FLOAT32-only; no BFLOAT16 mixed-owner graph,
   arithmetic, affine, constant, or backward route is advertised.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [rank glossary](../../../../glossary.md#term-rank); [owner glossary](../../../../glossary.md#term-owner); [constant glossary](../../../../glossary.md#term-constant))</span>
8. Focused Metal/native/conformance/public Engine/Javadoc/architecture checks, one final full build,
   Markdown/status/frontier/diff checks, and independent lean Class C review pass.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation)

The pre-production gate is the only executable evidence. Because it failed, no native build,
production test, [Javadoc](../../../../glossary.md#term-javadoc) task, [architecture test](../../../../glossary.md#term-architecture-test), or full build is authorized for this task.
Validate only blocker-evidence wording, links/anchors/fences/final newlines, status/frontier
consistency, documentation-only path [scope](../../../../glossary.md#term-scope), and `git diff --check`.

## Documentation and review impact

This blocker update owns only Task 0040, Task 0039's unconsumed-value correction, the [Metal](../../../../glossary.md#term-metal) master
plan, and the repository roadmap. Production architecture/capability/[API](../../../../glossary.md#term-application-programming-interface-api)/native/[Javadoc](../../../../glossary.md#term-javadoc) documents
remain unchanged because the gate authorized no behavior. No glossary term or independent Class C
review is needed for this [planning](../../../../glossary.md#term-planning)-only failure record.

## Result

Blocked at the sole pre-production execution. [MPSGraph](../../../../glossary.md#term-mpsgraph) changed selected [BFLOAT16](../../../../glossary.md#term-bfloat16) signaling-[NaN](../../../../glossary.md#term-nan)
word `0xffa6` to [canonical](../../../../glossary.md#term-canonical-form) quiet-NaN `0x7fc0`; all [artifacts](../../../../glossary.md#term-artifact) were removed and production remains
unchanged.

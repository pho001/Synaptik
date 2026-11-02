# Task CPU0050A: Repair low overlap-fold FP32 accumulation

## Status

Complete — MAIN accepted 2026-10-31 after Raman FINAL APPROVE, 0 open P1/P2, and qualified gate closure.
Historical Ready/launch verification retained. Frontier None; CPU0051/0052 Draft/held; no automatic successor.

## Change class

Class C — generated arithmetic, numerical results and cached-class compatibility. Clean implementation
and independent targeted code/documentation review are mandatory.

## Goal

Repair existing BFLOAT16/FLOAT16 FOLD_AXIS/FOLD2D: decode stored contributions exactly to FLOAT32
(FP32), accumulate in FP32, then narrow once to the low output with round-to-nearest, ties-to-even.
Existing Model repair, not CPU0050's F32/F64 compact-source optimization.

## Scope and non-goals

- Existing ordinary low fold emitter, cold scalar reference, selective identity and focused tests/docs.
  No added/moved production type/package, public API, lifecycle, dependency or architecture change.
- Preserve existing capability/admission, mappings, contributor order, positive-zero initialization,
  output ownership, ranges, checked addressing, binding and zero-workspace/resource topology.
- Do not extend low compact-input admission, absorb its expanded buffers or change the stored typed
  DIV/divisor recipe. CPU0050 compact F32/F64 and ordinary nonlow artifacts remain unchanged.
- No global schema/projection bump, new numerical mode, other family repair, Runtime/Compiler production
  change, native build, timing/benchmark/performance budget, broad campaign, stage/commit/push or successor launch.

## Contracts

Start at the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Required authority headings:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core invariants`,
  `Dependency rules`, `Testing requirements`.
- [Foundational modules](../../../../architecture/contracts/foundational-modules.md): `Arithmetic-site DAZ/FTZ`,
  `Exact and discrete floor`, `Floating arithmetic primitives`,
  [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor),
  `Composite inheritance`,
  [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- [Backend execution](../../../../architecture/contracts/backend-execution.md): `Concrete backend modules`,
  `CPU backend routes`, `Low-precision capability evidence`, `Profile-free backend capability and identity`.
- Compiler/autograd: [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  (read-only recipe, exact saved sites and actual generated gradients).
- [Runtime/Prepare/Engine](../../../../architecture/contracts/runtime-prepare-engine.md): `modules/prepare`,
  `Prepare lifecycle`, `modules/runtime`, `Run lifecycle` (preservation only).

## Dependencies and integration

- Depends on: [CPU0050 Complete](0050-average-pool2d-backward-broadcast-fold.md#result), MAIN accepted
  2026-10-31 after Dirac FINAL APPROVE, named join and filtered CPU/Javadoc PASS; predecessors retained.
- Conflicts with: CPU0051/0052 and every fold/emitter/reference/cache/consumer/shared-document writer.
- Parallel group: None; no parallel writers. MAIN's concurrent source/authority/protection preparation is read-only.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` PLUS current accepted dirty series,
  not HEAD-only source. Actual-before `/private/tmp/synaptik-cpu0050a.yqY321/before.json` records 2596 files,
  HEAD and empty index; `before-source.tar.gz` covers CPU/consumer sources and guide. Manifest SHA-256:
  `02f6392c3790655b0ec8d38c76daf0ad441125a165107502e4718fcacf0f17c0`.
- Integration order (actual): pre-edit controls → focused evidence/review → MAIN named join once →
  failed CPU gate → approved metadata/environment recovery → targeted recovery → final metadata/Javadoc
  closure → final read-only review/MAIN acceptance. Earlier consumer join changed order, not coverage/count.
- Integration validation: the two named gates below; no root campaign. Reuse successful evidence unless
  executable changes or a concrete relevant failure invalidate it.
- Shared-document integration owner: MAIN; exclusive serial delegation only for guide/plans/glossary.

Planning verified all 2596 hashes identical, no missing/added files, matching HEAD/empty index before
its three planning edits; dependency/frontier/conflicts verified. Raman reports no authority blocker.
MAIN's historical launch proof is `/private/tmp/synaptik-cpu0050a.yqY321/MAIN-LAUNCH.json`:
Ready/sole frontier/dependency/seven fields verified; only three planning paths differed, no competing writer.

## Files and symbols

Existing production owners under `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`:

- `internal.codegen.emit.CpuFoldEmitter`: dense/general bodies, accumulator/load/add/store/conversion.
- `internal.reference.CpuScalarReferenceKernel.execute(CpuFoldIr,...)`/`foldAdd`/store: independent cold repair.
- `internal.ir.CpuFoldIr.encodedKernelIr`: low-only discriminator; fingerprint reaches BOTH specialization
  streams, binary name/store/weak key. No global schema/nonlow change; affected Javadocs reviewed.

CPU tests cover generated/semantic folds, compatibility/IR/specialization/store and the test-only
`CpuLowFoldCleanJavaOracle`/`CpuLowFoldFp32AccumulationTest`; exact paths are in the final execution packet.
Consumers: `testing/backend-conformance/.../CpuBroadcastFoldConformanceTest.java`;
`testing/integration-tests/.../EngineCpuAveragePoolBackwardIntegrationTest.java`/`EngineCpuLowFoldIntegrationTest.java`.
`PoolingGradientRules.average`/`LayoutGradientRules` stay read-only. Guide: `docs/backend-guide/cpu-backend.md`,
`Current overlap fold family`/compact subsection; explain repair/identity boundaries, preserve CPU0050 history.
Approved metadata recovery adds nine test-pin files (ten Java pin sites including an existing test) and
three TSVs: `generated-coverage-inventory.tsv`, `generated-pointwise-semantic-closure.tsv`,
`specialized-semantic-closure-gaps.tsv`. Exact paths/diffs: final execution packet and `recovery/` artifacts.
Only derived hashes/provenance changed; no additional production owner, math or nonhash row change.

## Acceptance criteria

1. Both low types and both families, every existing scalar body/carrier route: exact stored decode,
   primitive float accumulator initialized +0.0f, one FP32 add per contributor, one final low narrowing
   per owned output. No low round-trip in the loop, double accumulator, term truncation or new scratch.
2. Preserve canonical logical input row-major contributor order. FOLD_AXIS maps selected position w and
   final-window offset k to w*step+k, retaining other coordinates. FOLD2D decodes q into channel/ky/kx
   and p into oh/ow, mapping y=oh*strideH-padH+ky*dilationH and analogously x; exclude out-of-bounds
   padding/ceil tails, never invent zeros as contributors. Every matching logical occurrence counts once,
   including repeated physical reads; gaps remain +0 and out-of-range output sentinels remain unchanged.
3. Retain the independent optimal Java oracle unchanged. The repaired arithmetic core must use exact low
   decode, sequential FP32 accumulation and one final narrowing. Existing generic geometry traversal remains
   unchanged: unlike the oracle's hoisted q-decode, it repeats decoding inside the input loop. MAIN's
   `MAIN-generic-fallback-evidence.json` records unchanged mapping/address/advance methods and the technical
   reason: avoid a low-only geometry fork or unrelated nonlow compatibility cutover. Whole-loop optimality,
   dataflow parity and performance parity are not established. Raman accepted this explicit retained-fallback
   deviation in `INDEPENDENT-PRE-FINAL-APPROVAL.json`, including eight identical 98-byte geometry blocks.
   Both artifacts are under `/private/tmp/synaptik-cpu0050a.yqY321/`. Do not weaken the oracle or expand scope;
   independently fix cold and test-local arithmetic reference paths, not merely their expectations.
4. Distinguish BF16 1+2^-8+2^-8 → 0x3f81 (not old 0x3f80) and F16 1+2^-11+2^-11 → 0x3c01
   (not old 0x3c00). Cover cancellation/order, even/odd ties, FP32 versus premature low overflow,
   final overflow, NaN class, both infinities/opposite-infinity NaN, signed zeros, low/FP32 subnormals/
   underflow. Exact decode is not DAZ; no new denormal policy or NaN-payload requirement.
5. Both types/families: dense int/general long, heap/native/mixed carriers, offsets/strided/zero-stride
   reads/injective writes, overlap/gaps/ranks/axes/N/C>1, nonsquare/stride/dilation/padding/ceil windows,
   admitted empty/partial ranges. Retain cold span/alignment/type/access/alias rejection before writes,
   isolated fresh reuse and existing parallel-scalar range bodies.
6. Authentic pre-edit identities/keys/ClassFiles/packs retain source/toolchain/hash provenance. CPU0050
   `/private/tmp/synaptik-cpu0050.ylFXyT/controls`: 160 payloads/32 forms; 16 F32/F64 forms literal,
   16 low forms change BOTH streams/class/key/name but keep pack/ABI/resources. Keep controls enabled/
   immutable; capture missing integer-axis and compact F32/F64 controls before edits and retain them literal.
7. Reject authentic old low classes/checksummed envelopes, including stale envelopes copied to current
   keys; regenerate safely. Prove isolated current weak/persisted reuse and distinct BF16/F16 keys.
   Retain envelope71/projection52, cold geometry/ranges, nonlow/compact admission/artifacts and other families.
8. ClassFiles: FP32 locals/adds, exact decode, final conversion outside contributor loop, one store;
   no hidden math helpers, boxing/dispatch/reflection/maps/callbacks/allocation. Audited JDK conversions/
   checked segment calls require oracle rationale; criterion 3's generic traversal qualification applies.
9. Actual low Compiler UNFOLD_AXIS/UNFOLD2D adjoints and AveragePool2d backward execute with nonuniform
   rounding/cancellation seeds and fresh runs against independent recipe-aware oracles. Retain stored DIV,
   split expanded low topology and compact F32/F64/resource/fail-before-write controls, not topology-only proof.

Stop for architectural ambiguity/conflict, a needed new production type/ABI/resource/dependency change,
unprovable selected-only identity isolation, missing authentic baseline, or required unrelated repair.
Report gaps to MAIN; defective old low behavior is not conformance.

## Validation

JDK26; planning ran no Java tests. Exact commands/environment/results are archived under evidence root
`/private/tmp/synaptik-cpu0050a.yqY321/`: `focused-03/`, `main-named-join/`, `final-gate/command.json`,
`targeted-recovery-gate/command.json`, `metadata-closure-gate/command.json`; final execution packet joins them.
All diagnostic/performance switches off. Original filtered `:backends:cpu:test :backends:cpu:javadoc`
uses `/private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle` (SHA-256
`7d5ecc10612e7550dfc73563877ace1c4bd6829291c59f8f38e42800b04310e6`), excluding ONLY
`CpuPartitionDagGeneratedEvidenceTest.fiveFixedHeapForksAndAggregateStayWithinGate` and
`.freshFusionEvidenceRetainsEverySampleAndBothRequiredComparisons`; controls stay enabled.
Named consumer classes: `CpuBroadcastFoldConformanceTest`, `CpuProfileFreeConformanceTest`,
`EngineCpuLowFoldIntegrationTest`, `EngineCpuAveragePoolBackwardIntegrationTest`,
`EngineCpuComposedValidationIntegrationTest`, `EngineCpuLongPartitionIntegrationTest`.
After failure, MAIN authorized only targeted metadata/environment recovery, using `--no-configuration-cache`
for correct environment delivery; no global Gradle-cause claim. Successful unchanged outcomes are reused,
not summed or relabeled a full PASS. No full repeat, consumer repeat, root suite or timing.
Check links/anchors, <=200 lines/15KB, whitespace and protection snapshot; record skips and Javadoc warnings.

## Documentation and review impact

Independent Class C review finalizes affected Javadocs and CPU guide;
apply documentation rules plus general/backend/API profiles, targeted glossary search and final Javadoc
generation. Explain repair versus CPU0050 optimization and affected-only invalidation. Existing glossary
terms suffice unless review finds a concrete gap. No authority/ADR change; reuse executable evidence.

## Result

MAIN accepted 2026-10-31; Raman FINAL APPROVE, 0 open P1/P2. Evidence root above;
`FINAL-EXECUTION-COMPLETION-PACKET.json` owns exact paths, commands, source/resource hashes and joins.

- Three existing production owners repair low FP32 accumulation/final narrowing and low-only identity;
  no added production type. Focused semantics, cold/test references, cache/structural tests and actual AD
  consumers complete; guide/Javadocs finalized. Frozen core 14 Java; 16 ClassFiles/unchanged optimal oracle inspected.
- Metadata-only recovery: 16 low inventory rows/64 derived fields; 41585 other inventory rows literal;
  ten Java SHA pins, one pointwise provenance header and 29 specialized row hashes mechanically refreshed.
  Algorithms/nonhash rows unchanged, independently approved; 25 then 27 source/resource freeze hashes retained.
- Focused-03: 79 total, 78 PASS/1 opt-in skip. MAIN join session49942 exit0/BUILD6s: 4 conformance +
  16 integration PASS, zero skips, once only. Six actual consumer XMLs archived; stale copied CPU XML excluded.
- Original final-gate session45942 exit1/5m27s remains FAILED: 1502 total, 1437 PASS/7 FAIL/58 SKIP;
  six stale inventory/hash failures plus one control-environment failure. Javadoc did not run in that attempt.
- Inventory recovery dump: two control tests PASS plus expected inventory failure. Targeted recovery FAILED:
  63 total, 62 PASS/1 FAIL/0 SKIP; all original seven closed, remaining pointwise provenance header failure retained.
- Metadata closure exit0/BUILD9s: 2 methods PASS/0 SKIP; closes header failure and proactively checks specialized
  source-hash join. CPU Javadoc PASS, 94 warnings: 92 in two unchanged-before-task Java files + 2 incubator warnings.
- Reuse unchanged successful core/consumer/full-run outcomes; overlapping runs are not summed. No full CPU
  rerun/full-green claim. Correct environment under no-configuration-cache, not proof of a global Gradle bug.
- Glossary unchanged: established terms suffice. No authority/ADR/API/lifecycle/dependency change; nonlow/
  compact controls retained. Generic geometry hoisting remains known follow-up; no whole-loop optimality,
  dataflow/performance parity, duplicate geometry loop, automatic task/timing/campaign or Git publication.
- MAIN and final review verify all 27 protected hashes match; HEAD unchanged/index empty. Frontier None;
  CPU0051/0052 Draft/held. No required follow-up for this qualified repair.

Status: Complete

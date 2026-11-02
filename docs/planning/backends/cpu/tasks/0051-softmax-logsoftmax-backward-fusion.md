# Task CPU0051: Specialized Softmax and LogSoftmax backward fusion

## Status

Complete — MAIN accepted 2026-10-31 after final independent APPROVE and qualified gate closure.
Performance parity remains unresolved. Historical launch: MAIN verified Ready/frontier, Complete CPU0050A/review closure, seven integration
fields, inactive conflicts, HEAD/empty index, retained 2600-file baseline (only three planning
paths changed), and filter/control artifacts on 2026-10-31. Independent plan review APPROVE,
0 P1/P2. Clean sole-writer implementation admitted; evidence: `/private/tmp/synaptik-cpu0051.AaUYCr`.

## Change class

Class C — generated hot-loop/dataflow, exact SUM and buffer removal; clean implementation and
independent targeted code/documentation review required.

## Goal and scope

Recognize the exact operation directed acyclic graphs (DAGs) built by Compiler
`ReductionGradientRules.apply`: `y * (g - sum(g * y, axis, true))` and
`g - exp(logY) * sum(g, axis, true)`. Here `g` is the incoming cotangent and `y`/`logY` is the
exact stored forward output. One CPU unit reduces each slice, then writes its gradient.
Remove only private unobserved buffers, not sites. Deliver code/tests/docs together.

### Selected matrix and fallback

All conditions apply; no capability widening.

| Concern | Specialized domain | Otherwise |
|---|---|---|
| Type | Homogeneous FLOAT32/FLOAT64 at all sites | BF16/F16/mixed/casts stay split |
| Shape/axis | Static rank 1 axis 0 or rank 2 axis 1; kept single-axis SUM | Other axes/ranks/forms/symbolic Shapes stay split |
| Reads | Canonical saved output; canonical cotangent or proved scalar/`[R,1]` zero-stride broadcast; nonnegative offsets | Other/unproved mappings stay split |
| Write | Canonical injective full-shape gradient, nonnegative offset; proved nonoverlap with reads | Split or existing binding rejection |
| Carriers | Typed float[]/double[] or checked MemorySegment independently at each boundary; mixed tuples included | No coercion/materialization/native route |
| Execution | SCALAR, one complete-slice range; int-bounded counts/spans/address arithmetic, including segments | No SIMD/parallel specialization; failed proof stays split |
| Empty/singleton | Zero rows/width, width one, legal slice subranges | No primitive-eliding identity shortcut |

Use `CpuAccessPlan.Binding`, not regime labels. Broadcast SUM visits the full logical
domain, not the smaller source. Select scalar/max1 even under vector-preference hints; incompatible
explicit candidates are ineligible. Failed proof retains existing decomposition/partial fusion, resources and errors.

## Non-goals

No public op/API, Compiler rewrite/dependency, forward change, Scan widening, SIMD/parallel redesign,
general framework, authority change, tuning, native build, historical campaign or Git publication.

## Contracts

Read `AGENTS.md`, root scope index and these headings.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Scope-indexed normative contracts`, `Core invariants`, `Dependency rules`, `Testing requirements`.
- [Foundational numerical owner](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics):
  `Profile-free numerical semantics`, `Arithmetic-site DAZ/FTZ`, `Exact and discrete floor`,
  `Floating arithmetic primitives`, `Aggregate floor`, `Composite inheritance`.
- [Backend owner](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  `Concrete backend modules`, `CPU backend routes`, `Profile-free backend capability and identity`,
  `Performance evidence and optimization tooling`.
- [Derivative owner](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation):
  `Training graph model`, `Compiler-owned automatic differentiation` — read-only formulas/saved boundaries.
- [Physical declaration owner](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare):
  `modules/prepare`, `Prepare lifecycle`, `modules/runtime` — declarations/slots and run isolation.

Preserve primitive rounding/exact SUM. STOP on authority conflict/ambiguity with evidence.

## Dependencies and integration

- Depends on: CPU0050A Complete.
- Conflicts with: CPU0052; CPU lowering/emitter/cache/resource/shared-document writers.
- Parallel group: None.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty 2600-file
  snapshot `/private/tmp/synaptik-cpu0051.AaUYCr/before.json`; predecessors are not all committed.
- Integration order: focused implementation evidence → independent code/docs review → short explicit
  diagnostic → final filtered CPU module/Javadoc → MAIN named conformance/integration join once → acceptance.
- Integration validation: `CPU0051 named join` commands below; no repository-wide gate without a new reason.
- Shared-document integration owner: MAIN (exclusive guide/Javadoc delegation allowed).

Verified 2026-10-31: user approval; CPU0050A Complete/final approval/qualified closure in master/roadmap;
matching HEAD and all 2600 hashes, empty index, no competing writer declared; current source/contracts
support this matrix. No predecessor evidence read/rerun; its failed full gate is not relabeled green.
CPU0051 alone is Ready; CPU0052 held. MAIN must record a launch receipt rechecking dependencies/review,
status/frontier, dirty hashes, HEAD/index, conflicts, filter/controls and seven integration fields.

## Files, symbols and proposed types

Production prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- **New `internal/ir/CpuSoftmaxBackwardIr.java`** implements `CpuPortableKernelIr`: immutable fixed
  variant, existing `CpuAggregateIr`, sites/bindings, reads/write and baked geometry.
  Unlike `CpuAggregateFusionIr`'s reduced-domain ADD/RELU tail, this rereads full-domain operands
  and supports independent EXP/SUM branches. Not a semantic operation or second math vocabulary.
- **New `internal/codegen/emit/CpuSoftmaxBackwardEmitter.java`** emits just those two-pass loops,
  composing `CpuAggregateEmitter.emitExactSum`/`CpuExactSumEmitter`, `CpuScalarEmitter.emit` and
  `CpuCarrierEmitter`; separates loops without duplicate math/execution. More types need MAIN approval.
- `internal/lowering/CpuAggregateLowering.java`: add bounded `fuseSoftmaxBackward` recognition/lowering;
  reuse `lower`, access/axis facts, `Geometry.workspaceBytes`/`packFusion`.
- `internal/lowering/CpuPartitionDagDecomposer.java`: select the fixed four-node patterns before
  generic aggregate candidates; reuse `partitionDag`, `aggregatePrivate`, convexity/reserved nodes,
  producer/use edges (not adjacency), original ordinals/dependencies.
- `internal/lowering/CpuPartitionLowering.java` (`LoweredPartition`) and
  `internal/ir/CpuPortableKernelIr.java`: carry/encode this exclusive physical family, reusing aggregate geometry.
- `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan,CpuPartitionFinalizer}.java`,
  `internal/route/portable/CpuPortableRoutePlan.java`, `internal/executable/CpuPreparedExecutable.java`:
  scalar range/virtual declarations, exact workspace, encoding/binding. Iterate slices, but final
  boundary spans the full gradient, not reduced SUM shape.
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`: exclusive validation/dispatch to the new emitter.
  `internal/cache/{CpuKernelSpecialization,CpuGeneratorSchema,CpuGeneratedKernelArtifactStore}.java`:
  exclusive family facts/selective compatibility; reuse artifact-store mechanism.
- Read-only formula/tests: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java#apply`
  and its test `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java#softmaxRulesConsumeTheExactSavedForwardOutput`.
  Read-only forward distinction: `CpuSoftmaxEmitter`, `CpuSoftmaxLowering`, `CpuSoftmaxInputValidator`.
- Focused CPU tests in matching `src/test/java/.../internal/` packages: new
  `lowering/CpuSoftmaxBackwardLoweringTest`, `codegen/emit/CpuSoftmaxBackwardGeneratedKernelTest`,
  `codegen/emit/CpuSoftmaxBackwardCleanJavaOracle`; extend existing decomposer, specialization/artifact,
  preparation/binding/package-inventory tests. Reuse `CpuAggregateFusionGeneratedKernelTest` and
  `CpuAggregateFusionCleanJavaOracle` fixture patterns, not their campaign.
- New `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuSoftmaxBackwardConformanceTest.java`
  and `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineCpuSoftmaxBackwardIntegrationTest.java`.

## Acceptance criteria

1. Match exact operands/sharing/type/kept axis/mappings of both four-node patterns, not names/IDs/ranges
   or algebraic equivalence. Keep saved output and `g`, including fan-out. Remove only expected-private-use
   intermediates: no publication, saved/higher-order/outside consumer. Prove convexity/no reserved overlap.
2. Softmax first rounds every `g*y` in its captured dtype, feeds it once to the current exact-limb
   SUM and rounds that SUM once; pass two rounds SUB then MUL. Log-softmax SUM consumes stored `g`
   once; pass two computes current scalar EXP (`StrictMath.exp`, with F32 result conversion), rounds
   MUL then SUB. No running-sum/Kahan substitute, FMA/contraction, high-precision product, dropped site
   or forward recomputation. Preserve limb width/flags, overflow/NaN/infinity/zero-sign finish and primitive behavior.
3. One unit, two direct passes/slice, three read/write roles (dedup identical reads), no intermediate
   tensor store/full-domain temporary. Hoist bases/strides/width/carriers; row/element cursors, no element
   coordinate decode. Reuse one flags-plus-limbs workspace slice across rows, declared before slots,
   isolated per run. Zero rows need no state/write; zero width no contributor/read/write; singleton retains sites.
4. Add selective `ordinary-softmax-backward-v1` canonical family and baked suffix in **both** existing
   specialization streams, under envelope 71/static class projection 52. Key variant/sites, dtype,
   axis, structural access and every baked extent/stride/offset/span/state fact; exclude graph IDs,
   slots/objects/addresses/run bases. Validate exclusivity, scratch ABI/geometry, cache reuse/misses.
   Forward, aggregate-fusion, low-fold, row/Scan and unrelated keys/ClassFiles remain unchanged.
5. Compare fused results with split execution and an independent optimal specialized clean-Java oracle
   with typed locals/direct loops/independent exact accumulation and rounding, not a production evaluator.
   Cover both dtypes/kinds/ranks, eight heap/segment tuples, offsets/broadcasts/empty/singleton/subranges,
   cancellation/subnormals/signed zero/NaN/infinities/overflow and product/SUM-rounding counterexamples.
   Class/sign checks are separate; NaN payload unspecified. No new F64 DAZ/FTZ.
6. Test rejection for altered operands/axis/keep flag, layouts/types, public/saved/fan-out intermediates,
   outside uses, nonconvex topology and resource/address limits; admit unrelated interleaved nodes.
   Actual Compiler AD selects both with nonuniform seeds; Engine reuse with fresh inputs/seeds preserves
   forward publications. Forward rejection still precedes writes; backward uses primitive special rules,
   not the forward finite-input validator.
7. Inspect generated ClassFiles against that Java algorithm/hot-loop/dataflow: typed MUL/SUB/EXP sites,
   two passes and rounded SUM before pass two; no hidden evaluator/helper, allocation/boxing/reflection/
   graph dispatch/intermediate store. JDK math/raw-bit/checked memory calls allowed; justify deviations.
   Run only a short explicitly opt-in matched generated/direct diagnostic after source review: one JVM,
   fixed `[32,128]` heap case per kind/dtype, five warmups and five alternating paired samples of five
   calls, 60-second cap/no retry. Report setup/entry separately, samples/limits and avoidable overhead;
   no universal speedup/steady-state parity claim.

## Validation

Use JDK26 and this setup for every Gradle phase; planning runs no tests.

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
export SYNAPTIK_CPU0050_CONTROLS=/private/tmp/synaptik-cpu0050.ylFXyT/controls
export SYNAPTIK_CPU0050A_CONTROLS=/private/tmp/synaptik-cpu0050a.yqY321/controls
unset SYNAPTIK_CPU0050_CAPTURE SYNAPTIK_CPU0050_DIAGNOSTIC SYNAPTIK_CPU0050_STRUCTURE
for n in 0022_DIAGNOSTIC 0033_DIAGNOSTIC ATTENTION_PERFORMANCE BFLOAT16_POINTWISE_PERFORMANCE LOSS_PERFORMANCE POOL2D_PERFORMANCE POOL3D_PERFORMANCE; do export "SYNAPTIK_CPU_${n}=false"; done
CPU_GATE=(--offline --no-daemon --no-configuration-cache --max-workers=1 --console=plain -I /private/tmp/synaptik-cpu0048.G18rM6/no-campaigns.gradle)
for n in generalScan.performance generalScan.pointwisePerformance dynamicBatch.performance loss.performance loss.performanceRepresentative loss.c2Evidence cast.performance concatStack.performance pointwiseMask.performance pointwiseMask.performanceDiagnostic vectorMse.performance vectorScalarPower.performance convSimd.performance pool3d.performance; do CPU_GATE+=("-Dsynaptik.cpu.${n}=false"); done
./gradlew "${CPU_GATE[@]}" :backends:cpu:test --tests '*CpuSoftmaxBackward*' --tests '*CpuAggregateFusionGeneratedKernelTest' --tests '*CpuPartitionDagDecomposerTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
# Final code/Javadoc: one filtered module gate.
./gradlew "${CPU_GATE[@]}" :backends:cpu:test :backends:cpu:javadoc
# CPU0051 named join: MAIN once after review.
./gradlew "${CPU_GATE[@]}" :testing:backend-conformance:test --tests '*CpuSoftmaxBackwardConformanceTest' --tests '*CpuSoftmaxFiniteConformanceTest' --tests '*CpuFloat16SoftmaxConformanceTest' --tests '*CpuProfileFreeConformanceTest' :testing:integration-tests:test --tests '*EngineCpuSoftmaxBackwardIntegrationTest' --tests '*EngineCpuAveragePoolBackwardIntegrationTest' --tests '*EngineCpuLowFoldIntegrationTest'
git diff --check
```

Verify filter/controls/worker environment; no configuration cache/control recapture.
Archive actual phase-specific XML from `modules/backend/cpu/build/test-results/test`,
`testing/backend-conformance/build/test-results/test` and `testing/integration-tests/build/test-results/test`
with command/environment/exit and executed/UP-TO-DATE provenance; no stale CPU XML for consumers.
Preserve inventory hash
`7e6faca8db36bb8700fde9dfe8e72c554bf51ee207693a22ef798bd991846f1b`; do not regenerate inventories or
re-pin evidence without actual ClassFile change and MAIN approval. No historical timing/Scan/prolonged
matrix or automatic diagnostic. No architecture/full-root gate without boundary/build changes.

## Documentation and review impact

Independent Class C review finalizes Javadocs and `docs/backend-guide/cpu-backend.md` (matrix/sites/
workspace/fallback), checks glossary softmax/fusion terms; no term/API/authority change expected.
Apply documentation-rules/general/backend/Javadoc profiles; reuse tests. MAIN owns master/roadmap.
Planning: general/planning profiles.

## Result

MAIN accepted 2026-10-31 after final independent result APPROVE, no blocking P1/P2. Evidence roots:
`/private/tmp/synaptik-cpu0051-impl.JgdND8` (implementation) and `/private/tmp/synaptik-cpu0051.AaUYCr` (MAIN).

- Implemented both bounded two-pass patterns with exact typed sites/SUM, private-buffer removal, scalar binding/workspace and selective dual-stream identity. Exactly two new production types; no API/authority/Scan/SIMD or global vector/thread-default change.
- Twenty implementation/test paths are in `freeze/paths.sha256`; guide plus three planning paths make 24 task-delta paths. Actual-before snapshots, not Git HEAD, delimit accepted dirty work.
- `focused-07`: 14 CPU0051 PASS; `retained-controls-08`: 2 PASS, no skips. Earlier mixed `focused-04` had 52 CPU PASS but exit1 from a test factory overload; corrected consumer compilation passed. Development failures remain in `HANDOFF.txt` and phase logs.
- `final-cpu-javadoc-10` exit0/6m18s: CPU 1518 total, 1461 PASS, 57 SKIP, zero failures/errors; Javadoc PASS with 94 warnings. Only the two specified historical timing methods were filtered; no full-root gate.
- MAIN `main-named-join` remains FAILED: 7 conformance, 6 PASS/1 invalid test expectation; integration did not execute. Softmax's nonfinite row requires one +Inf, not six NaNs; production was correct.
- Independent test-only repair review APPROVE: only that method/Javadoc changed, all other 20 reviewed hashes retained. `main-targeted-closure` exit0: corrected method 1 PASS and first Engine integration 6 PASS, zero skips. Reuse the six unchanged conformance successes; no full-join repeat or relabeling.
- Four retained ClassFiles/javap and the independent clean Java confirm two feature passes, rounded SUM before tail sites, direct cursors and no hidden Synaptik helper/allocation/boxing/dispatch/intermediate tensor store. Broadcast/shared-load corrections are structurally tested.
- Sole released diagnostic exit0 in 0.706s, all outputs verified; four synthetic [32,128] heap cases, five warmup groups and five alternating pairs of five calls. No retries or further budget.
- Generated/direct median ratios: F32 soft 14.648, F32 log 15.766, F64 soft 8.201, F64 log 3.546. F64 soft samples change sharply within the run. No JIT log/C2 causal proof, steady-state parity, profitability, end-to-end or universal-speed claim; performance parity remains unresolved.
- Independent source/docs/protocol and final result APPROVE, no blocking P1/P2; seven Javadocs and CPU guide finalized. Glossary/architecture/API/ADR impact none.
- Inventory SHA `7e6faca8db36bb8700fde9dfe8e72c554bf51ee207693a22ef798bd991846f1b`, HEAD and empty index retained; 2584 original files unchanged, eight added, sixteen changed, none removed. `git diff --check` PASS.
- CPU0052 remains held. No required implementation follow-up; performance investigation needs a separate authorization, not another historical campaign or automatic successor. No commit/push.

Size note: the atomic Class C brief exceeds 15 KB to retain the matrix, exact contracts/commands and qualified failed-gate closure together; splitting those would obscure this single capability's acceptance.

Status: Complete

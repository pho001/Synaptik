# Task 0044: Static and symbolic-batch length Scan source foundation

## Status

Complete — MAIN accepted the exact source-only scope on 2026-10-26 after independent source/docs
APPROVE and configured root test/Javadoc PASS. No downstream successor is authorized.

## Change class

Class C: bounded source-contract admission plus independent Compiler/Prepare fail-closed boundaries.

## Goal

Extend the existing length-aware `Scan.apply` to fully static shapes OR one named batch B,
with fixed T/features, homogeneous FLOAT32/FLOAT64 body and separate non-gradient INT64 `[B]`
lengths. Preserve true Dimension/ROW provenance and every old static/all-valid guard and behavior.

## Scope and non-goals

Source/tests/Javadoc/docs include bounded root/scan-owner wording BEFORE Java and exactly two
early rejection fixes: Compiler AD preflight and Prepare projection. Other consumer gates stay closed.
No new/moved production type, API, package, field, representation, dependency or Runtime handoff.
No ScanAttrs/ScanResult/CompiledScanBody representation/count change or narrowing of all-valid ScanBody.
No Compiler inference/AD/binding enablement, Prepare/CPU capability widening, emitter/codegen/perf,
outer F64/control prerequisites, persistence/cold/v5, NN/Training, variable T/features, higher-order,
low/mixed, vector/parallel/OpenBLAS/Metal, memo, new executor, native build, Git mutation/commit/push.
STOP for MAIN approval before another production file/type, supported signature/handoff or architecture change.
Private Scan helper changes and the approved test-only file are allowed.

## Contracts and scoped reading

Read AGENTS.md, this brief, affected source/focused tests and these scoped headings:

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Authority, incorporation, and precedence`, `Core lifecycle`, `Core invariants`,
  `Scope-indexed normative contracts`, `Dependency rules`, `Testing requirements`.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `General differentiable scan`, `Source body, attachments and scoped identities`;
  `Accepted symbolic-batch scan` and descendants; `Length-aware whole-batch scan` and descendants,
  especially `Header, bindings and source admission`, `Bounded row-role analysis`,
  `First-order length-mode reverse`, `Runtime preflight and implementation gates`.
- Downstream rejection only: [Compiler](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  `Compiler-owned automatic differentiation`, `Accepted symbolic-batch metadata binding`,
  `Bounded general-scan cold reconstruction`; [Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  `modules/prepare`, `Prepare lifecycle` (static length projection only);
  [CPU](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) `CPU backend routes`
  (separate static-length/all-valid-symbolic qualification);
  [Checkpoint](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
  `Generic graph source and values`, `Canonical graph-model version 5` (closed cold/wire gates).

Only root source/status pointers and recurrent-scan source wording change authority. Do not edit
other normative owners; report an actual contradiction to MAIN FIRST. Original-source AD restrictions,
arithmetic/activity/history/runtime/producer/resource rules and all identities remain unchanged.

## Dependencies and integration

- Depends on: Complete Model0043 (reviewed design), Model0042/0038, Compiler0013/0014,
  Prepare0011/0014, CPU0042; explicit user semantic/source/bounded-authority approval.
- Conflicts with: Scan source/ROW semantics, AD preflight, Prepare projection, shared contracts/docs;
  serialize writers; other module frontiers closed.
- Parallel group: None; sole serial writer, no concurrent-write worktree needed.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted dirty base; predecessors are accepted UNCOMMITTED work, not merges.
- Integration order: accepted pre-Java authority/Ready -> different clean implementation agent -> independent code/docs review
  -> source freeze -> MAIN's one configured root checkpoint and acceptance.
- Integration validation: `Model0044-source-rejection-root`, exact command below, includes architecture.
- Shared-document integration owner: MAIN; exclusive serial doc delegations; review reuses executable evidence.

### Admission verification — 2026-10-26

PlanningAuthorized=true; AuthorityAuthorized=true ONLY for bounded Model-source wording in root/scan owner.
ImplementationAuthorized=true ONLY for this source foundation and deliberate reject guards/tests.
Mencius pre-Java APPROVE: 0 P1/P2, no missing contracts/owners. MAIN accepted authority/Ready on
2026-10-26: dependencies Complete, conflicts inactive, sole Model frontier 0044, no other writer.
MAIN verified the frozen nine-doc authority packet and unchanged accepted dirty base/HEAD/origin/index.
Original baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/model0044-baseline.json` and `-files/`;
worker baseline: `model0044-implementation-baseline.json` and `-files/` in the same directory.
Preserve ALL preexisting dirty work; the three-production-file allowlist is unchanged by activation.
Model0043 stays Complete DESIGN DELIVERY ONLY; historical false authority/implementation flags unchanged.
MAIN launched a different clean worker only after authority/Ready verification; no downstream launch.
Evidence directory: `/private/tmp/synaptik-model0044-main.aFgru2`.

## Files and symbols / implementation allowlist

Java paths below are relative to `src/{main,test}/java/io/github/pho001/synaptik/` in each named module.
Only these production files may change; all others, including generators, stay hash-identical:

- `modules/model` main `model/tensor/Scan.java`: `validateRows`, `requireLengthStatic`, `checkRow`,
  `rowResult`, dimension/feature predicates and length-overload Javadoc. Prefer private
  Dimension-based predicates; no new proof type. ScanBody/family checks are read-only preserved guards.
- `modules/compiler` main `compiler/ScanGradientRules.java`: `preflight` ONLY early symbolic+length
  rejection before derivative IDs; Javadoc documents deliberate static-only length AD admission.
- `modules/prepare` main `prepare/analysis/PrepareContext.java`: `validateScan` ONLY early static-only
  length rejection before ANY numeric extent access/analyzer; update affected contract Javadoc.
- Model tests `model/tensor/{ScanSymbolicBatchSourceTest,ScanSemanticsTest,ScanExpressionTest,ScanBodyTest}.java`:
  source/row/attachment/last-port/storage/ID witnesses; old header/count tests remain compatible.
- Optional new TEST-ONLY `modules/model` test `model/tensor/ScanSymbolicLengthSourceTest.java` for focused coverage.
- Compiler tests `compiler/{GeneralScanCompilerTest,SymbolicScanInferenceTest,LengthAwareScanBodyInferenceTest,
  LengthAwareScanGradientRulesTest,SourceGraphReconstructionTest}.java`: public AD and raw IR/cold rejects.
- Prepare tests `prepare/analysis/{PrepareContextTest,BackendPartitionPreparerTest}.java`: direct symbolic
  length projection and late-partition rejection, IllegalArgumentException and zero analyzer calls.
- `modules/backend/cpu` tests `backend/cpu/CpuGeneralScanCapabilityTest.java`,
  `backend/cpu/internal/lowering/CpuGeneralScanLoweringTest.java`: raw query/descriptor rejection.
- `modules/checkpoint` tests `checkpoint/graph/{GraphCheckpointGeneralScanTest,GraphArtifactCodecTest}.java`:
  cold rejection and direct dead/late/T=0 codec bypass; no raw seven-field success.

Read-only: ScanBody family/containment; ScanAttrs old-seven defaults/counts; ScanResult/CompiledScanBody ordered ports;
ScanBodyInference.infer existing symbolic-length reject; CpuGeneralScanLowering static shape proof,
CpuGeneralScanIr.Stages mutual exclusion; SourceGraphReconstruction.preflight and GraphArtifactCodec
length guard. Preserve these guards; no later phases or guard framework.

## Acceptance criteria

1. Both F32/F64, both traversals, fixed T=0/nonzero; static B=0/1/ordinary and one canonical named B.
   Same normalized DynamicDimension name compares by VALUE, not identity/hard-coded spelling.
   Exact B only at identified ROW(0)/rank-two ROW(1) axes and lengths `[B]`; all features/SHARED
   declarations/values static, including unused ones. Reject unknown, independent/repeated/derived B,
   variable T/features, hidden shape-attribute symbols and SHARED-B; never narrow all-valid source.
2. Preserve bounded algebra, row broadcast/transpose/SUM/sumToShape, terminal taint and exact own
   `old.add(contribution)` rules. F=B/H=B, B=0/1 never infer/erase roles. Check fixed-feature products
   across ALL declarations/contained values/canonical siblings independently of empty B/T, not only
   reshape or symbolic knownElementCount. Preserve legitimate zero features; no sample B/extent read.
3. Exact actual/formal type/shape/flags, positional aliases and compatible outer tied roles; no length
   formal/gradient OR contribution. Old seven-field defaults/counts/overload unchanged; +1 length actual
   only. Static formals/ports canonical, symbolic formals/actuals/ports unresolved; shared histories
   `[T]+localShape` gain no B axis. Preserve full sibling inventory, canonical port identity and flags.
4. Public apply AND raw attachment validate every last/dead port/unused declaration and recheck body
   storage before output-ID supplier calls. No value/length read, storage/history allocation or execution;
   failures consume no output IDs. No host/free/effectful/nested/noncanonical containment regression.
5. Compiler public forward/raw inference reject symbolic lengths; AD preflight rejects BEFORE derivative
   IDs, including T=0 and an unrelated gradient target/source branch. Original-source all-carry/all-sequence
   ROW(0), residual-root/higher-order exclusions stay unchanged. No AD/binding implementation added.
6. Prepare direct/late-partition projection rejects symbolic lengths deliberately before Optional extent
   access and ANY analyzer/resource declaration. CPU provider false/direct lowering IllegalArgumentException;
   raw cold/codec bypasses reject independently, including dead/late/T=0. No incidental NoSuchElementException,
   IndexOutOfBoundsException, raw-wire success or hidden allocation stands in for a supported boundary.
7. Static lengths and old static/symbolic-all-valid tests remain green. Numeric/ABI/backend identities,
   generated producers/emitters and bytes are untouched; retain source hashes, not new timing evidence.

## Validation

Worker focused development:

```bash
./gradlew :modules:model:test --tests '*Scan*Test' :modules:compiler:test --tests '*GeneralScanCompilerTest' --tests '*LengthAwareScan*Test' :modules:prepare:test --tests '*PrepareContextTest' --tests '*BackendPartitionPreparerTest' --max-workers=2 --console=plain
```

After executable freeze, final affected-module tests/Javadoc ONCE; consumer test tasks also compile
their main/test sources and exercise independent rejects/static compatibility:

```bash
./gradlew :modules:model:test :modules:model:javadoc :modules:compiler:test :modules:compiler:javadoc :modules:prepare:test :modules:prepare:javadoc :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuGeneralScanLoweringTest' :extensions:checkpoint:test --tests '*GraphArtifactCodecTest' --tests '*GraphCheckpointGeneralScanTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --max-workers=2 --console=plain
```

MAIN ran `Model0044-source-rejection-root` after review/freeze, reusing worker/up-to-date evidence.
No worker/reviewer root or duplicate Java tests. Existing Vector module is configured.

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --max-workers=2 --console=plain
```

MAIN verified packaged dylib SHA256 `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`;
set SYNAPTIK_METAL_TEST_LIBRARY, no native rebuild or timing/ClassFile campaign: no emitter changed.
Validate owned links/anchors, whitespace/new-file guardrails (≤200 lines/15KB),
exact baseline-relative scope and unchanged protected hashes/HEAD/origin/index. No reviewer Java reruns.

## Documentation and review impact

Profiles: general + planning for this/master/roadmap/focused plans; architecture for root/scan owner
and training-graph; API/Javadoc for tensor-api/affected members. Follow documentation-rules.md.
Planning docs: this brief, Model master, roadmap, length-aware-general-scan, dynamic-batch-execution.
Authority/explanation: ARCHITECTURE.md, recurrent-scan.md, training-graph.md and tensor-api.md only.
Independent pre-Java review approved this bounded delta. MAIN's exclusive final-doc delegation adds
ONLY glossary `Symbolic Scan`/`General differentiable scan` source-status corrections and comment-only
Javadoc within the 15 frozen Java files. Review reused executable evidence; MAIN's root passed final
Javadoc validation after comment edits. Read-only ScanBody/ScanAttrs/ScanResult/CompiledScanBody
Javadocs remain accurate. No new term or ADR: existing scan decisions 0049/0051 and mechanisms suffice.

## Result

Scan.java source + ScanGradientRules.java/PrepareContext.java early rejects; 12 test files (one new).
Static/one-B F32/F64 source; Mencius independent APPROVE: 0 P1/P2;
MAIN accepted Complete 2026-10-26. Worker Model/Compiler/Prepare/CPU/Checkpoint: 1199/386/97/35/17;
1734 total, zero failures/errors/skips, three Javadocs passed. Evidence above:
`implementation.oF5NNj/worker-result.txt` (commands/fixes/XML/hashes) and `final-review.HNrw6X/`.
MAIN's exact root command above passed test/Javadoc, exit 0: BUILD SUCCESSFUL in 5m33s; 119 tasks
(41 executed, 78 up-to-date). `root-checkpoint.log` / `root-final-counts.json` record 707 suites /
5016 tests, zero failures/errors, 56 skips; includes UP-TO-DATE evidence, not all newly executed.
Ignored old `extensions/{training,checkpoint}` outputs excluded; included projects map to
`modules/{training,checkpoint}`. Skips: CPU 43, Metal 10, Engine 2, Integration 1.
Root source/HEAD/origin/index/native-package SHA unchanged; whitespace PASS. Closeout: ten owned
Markdown paths only, no executable changes/new authority/test reruns. All symbolic-length downstream
inference/AD/binding/projection/execution/cold/v5/NN/Training gates stay closed. Active Model frontiers:
None; all successors held, none authorized. Issues/follow-up: none.

Status: Complete

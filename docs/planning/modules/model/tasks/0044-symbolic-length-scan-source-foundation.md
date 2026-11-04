# Task 0044: Static and symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) length [Scan](../../../../glossary.md#term-cumulative-scan) source foundation

**Editorial reading key (current context, not a status update).**

This record closes a source-only admission, not symbolic-length execution: its Compiler and Prepare
rejections remain historical boundaries. The current [length-aware
reference](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
distinguishes later, independently scoped consumer support. Model semantics defines the recurrence;
providers realize qualified routes, not redefine it. Semantic responsibility is not buffer
ownership.

The caller supplies `Scan.apply` with actual Tensors matched positionally to `ScanBody` formals:
carry passes between steps, sequences supply slices, and invariants stay unchanged. A formal is a
placeholder leaf—no producing expression, not necessarily stored values. Here, descriptors cross the
construction boundary, not length values: separate non-gradient INT64 lengths have no body formal.
ROW identifies a proved batch coordinate; SHARED lacks one, regardless of equal sizes. For zero
steps and batch named `Rows`, row state `[Rows,2]` yields history `[0,Rows,2]`, while shared bias
`[2]` yields `[0,2]`. These storage-free expressions neither execute nor own result buffers. Follow
[canonical
results](../../../../reference/tensor/general-scan.md#canonical-results-residual-metadata-and-eligibility)
to distinguish port identity, residual metadata and gradient flags from allocated histories or
available derivatives.

## Status

Complete — MAIN accepted the exact source-only [scope](../../../../glossary.md#term-scope) on 2026-10-26 after independent source/docs
APPROVE and configured root test/[Javadoc](../../../../glossary.md#term-javadoc) PASS. No downstream successor is authorized.

## Change class

Class C: bounded source-contract [admission](../../../../glossary.md#term-admission) plus independent [Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare) [fail-closed](../../../../glossary.md#term-fail-closed) boundaries.

## Goal

Extend the existing length-aware `Scan.apply` to fully static [shapes](../../../../glossary.md#shape) OR one named [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) B,
with fixed T/[features](../../../../glossary.md#term-software-feature), [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) body and separate non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype) `[B]`
lengths. Preserve true [Dimension](../../../../glossary.md#dimension)/ROW [provenance](../../../../glossary.md#provenance) and every old static/all-valid [guard](../../../../glossary.md#term-guard) and behavior.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Source/tests/[Javadoc](../../../../glossary.md#term-javadoc)/docs include bounded root/[scan](../../../../glossary.md#term-cumulative-scan)-[owner](../../../../glossary.md#term-owner) wording BEFORE Java and exactly two
early rejection fixes: [Compiler](../../../../glossary.md#term-compiler) [AD](../../../../glossary.md#autograd) [preflight](../../../../glossary.md#term-preflight) and [Prepare](../../../../glossary.md#prepare) projection. Other consumer gates stay closed.
No new/moved production type, [API](../../../../glossary.md#term-application-programming-interface-api), package, field, representation, [dependency](../../../../glossary.md#term-dependency) or [Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff).
No ScanAttrs/ScanResult/CompiledScanBody representation/count change or [narrowing](../../../../glossary.md#term-narrowing) of all-valid ScanBody.
No Compiler [inference](../../../../glossary.md#term-inference)/AD/[binding](../../../../glossary.md#term-binding) enablement, Prepare/[CPU](../../../../glossary.md#term-cpu) capability [widening](../../../../glossary.md#term-widening), emitter/codegen/perf,
outer [F64](../../../../glossary.md#floating-point-precision-and-rounding)/control prerequisites, [persistence](../../../../glossary.md#term-persistence)/cold/v5, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training, variable T/[features](../../../../glossary.md#term-feature), higher-order,
low/mixed, [vector](../../../../glossary.md#term-vector)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas)/[Metal](../../../../glossary.md#term-metal), [memo](../../../../glossary.md#cache-and-memoization), new executor, native build, Git mutation/[commit](../../../../glossary.md#term-commit)/push.
STOP for MAIN approval before another production file/type, supported [signature](../../../../glossary.md#term-signature)/handoff or architecture change.
Private Scan helper changes and the approved test-only file are allowed.

## Contracts and scoped reading

Read AGENTS.md, this brief, affected source/focused tests and these scoped headings:

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  `Authority, incorporation, and precedence`, `Core lifecycle`, `Core invariants`,
  `Scope-indexed normative contracts`, `Dependency rules`, `Testing requirements`.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan), `Source body, attachments and scoped identities`;
  `Accepted symbolic-batch scan` and descendants; `Length-aware whole-batch scan` and descendants,
  especially `Header, bindings and source admission`, `Bounded row-role analysis`,
  `First-order length-mode reverse`, `Runtime preflight and implementation gates`.
- Downstream rejection only: [Compiler](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  `Compiler-owned automatic differentiation`, `Accepted symbolic-batch metadata binding`,
  `Bounded general-scan cold reconstruction`; [Prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  `modules/prepare`, `Prepare lifecycle` (static length projection only);
  [CPU](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> `CPU backend routes`
  (separate static-length/all-valid-symbolic qualification);<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
  [Checkpoint](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `Generic graph source and values`, `Canonical graph-model version 5` (closed cold/wire gates).

Only root source/status pointers and recurrent-[scan](../../../../glossary.md#term-cumulative-scan) source wording change authority. Do not edit
other [normative](../../../../glossary.md#term-normative) [owners](../../../../glossary.md#term-owner); report an actual contradiction to MAIN FIRST. Original-source [AD](../../../../glossary.md#autograd) restrictions,
arithmetic/activity/history/[runtime](../../../../glossary.md#term-runtime)/[producer](../../../../glossary.md#term-producer)/resource rules and all [identities](../../../../glossary.md#term-identity-meanings) remain unchanged.

## Dependencies and integration

- Depends on: Complete Model0043 (reviewed design), Model0042/0038, Compiler0013/0014,
  Prepare0011/0014, CPU0042; explicit user semantic/source/bounded-authority approval.
- Conflicts with: Scan source/ROW semantics, AD preflight, Prepare projection, shared contracts/docs;
  serialize writers; other module frontiers closed.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [AD glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [Prepare glossary](../../../../glossary.md#prepare); [module glossary](../../../../glossary.md#term-module))</span>
- Parallel group: None; sole serial writer, no concurrent-write worktree needed.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus accepted dirty base; predecessors are accepted UNCOMMITTED work, not merges.
- Integration order: accepted pre-Java authority/Ready -> different clean implementation agent -> independent code/docs review
  -> source freeze -> MAIN's one configured root checkpoint and acceptance.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): `Model0044-source-rejection-root`, exact command below, includes architecture.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive serial doc delegations; review reuses executable evidence.

### [Admission](../../../../glossary.md#term-admission) verification — 2026-10-26

PlanningAuthorized=true; AuthorityAuthorized=true ONLY for bounded [Model](../../../../glossary.md#term-model)-source wording in root/[scan](../../../../glossary.md#term-cumulative-scan) [owner](../../../../glossary.md#term-owner).
ImplementationAuthorized=true ONLY for this source foundation and deliberate reject guards/tests.
Mencius pre-Java APPROVE: 0 P1/P2, no missing contracts/owners. MAIN accepted authority/[Ready](../../../../glossary.md#term-ready) on
2026-10-26: dependencies Complete, conflicts inactive, sole Model frontier 0044, no other writer.
MAIN verified the frozen nine-doc authority packet and unchanged accepted dirty base/HEAD/origin/index.
Original baseline: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/model0044-baseline.json` and `-files/`;
worker baseline: `model0044-implementation-baseline.json` and `-files/` in the same directory.
Preserve ALL preexisting dirty work; the three-production-file allowlist is unchanged by [activation](../../../../glossary.md#term-activation).
Model0043 stays Complete DESIGN DELIVERY ONLY; historical false authority/implementation flags unchanged.
MAIN launched a different clean worker only after authority/Ready verification; no downstream launch.
Evidence directory: `/private/tmp/synaptik-model0044-main.aFgru2`.

## Files and symbols / implementation allowlist

Java paths below are relative to `src/{main,test}/java/io/github/pho001/synaptik/` in each named [module](../../../../glossary.md#term-module).
Only these production files may change; all others, including generators, stay [hash](../../../../glossary.md#term-hash)-identical:

- `modules/model` main `model/tensor/Scan.java`: `validateRows`, `requireLengthStatic`, `checkRow`,
  `rowResult`, dimension/feature predicates and length-overload Javadoc. Prefer private
  Dimension-based predicates; no new proof type. ScanBody/family checks are read-only preserved guards.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [feature glossary](../../../../glossary.md#term-feature); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `modules/compiler` main `compiler/ScanGradientRules.java`: [`preflight`](../../../../glossary.md#term-preflight) ONLY early symbolic+length
  rejection before derivative IDs; Javadoc documents deliberate static-only length AD admission.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [admission glossary](../../../../glossary.md#term-admission))</span>
- `modules/prepare` main `prepare/analysis/PrepareContext.java`: `validateScan` ONLY early static-only
  length rejection before ANY numeric extent access/analyzer; update affected contract Javadoc.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
- [Model](../../../../glossary.md#term-model) tests `model/tensor/{ScanSymbolicBatchSourceTest,ScanSemanticsTest,ScanExpressionTest,ScanBodyTest}.java`:
  source/row/attachment/last-port/storage/ID witnesses; old header/count tests remain compatible.
- Optional new TEST-ONLY `modules/model` test `model/tensor/ScanSymbolicLengthSourceTest.java` for focused coverage.
- [Compiler](../../../../glossary.md#term-compiler) tests `compiler/{GeneralScanCompilerTest,SymbolicScanInferenceTest,LengthAwareScanBodyInferenceTest,
  LengthAwareScanGradientRulesTest,SourceGraphReconstructionTest}.java`: public [AD](../../../../glossary.md#autograd) and raw [IR](../../../../glossary.md#term-intermediate-representation-ir)/cold rejects.
- [Prepare](../../../../glossary.md#prepare) tests `prepare/analysis/{PrepareContextTest,BackendPartitionPreparerTest}.java`: direct symbolic
  length projection and late-partition rejection, IllegalArgumentException and zero analyzer calls.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- `modules/backend/cpu` tests `backend/cpu/CpuGeneralScanCapabilityTest.java`,
  `backend/cpu/internal/lowering/CpuGeneralScanLoweringTest.java`: raw query/[descriptor](../../../../glossary.md#term-descriptor) rejection.
- `modules/checkpoint` tests `checkpoint/graph/{GraphCheckpointGeneralScanTest,GraphArtifactCodecTest}.java`:
  cold rejection and direct dead/late/T=0 codec bypass; no raw seven-field success.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>

Read-only: ScanBody family/containment; ScanAttrs old-seven defaults/counts; ScanResult/CompiledScanBody ordered ports;
ScanBodyInference.infer existing symbolic-length reject; CpuGeneralScanLowering static [shape](../../../../glossary.md#shape) proof,
CpuGeneralScanIr.Stages mutual exclusion; SourceGraphReconstruction.preflight and GraphArtifactCodec
length [guard](../../../../glossary.md#term-guard). Preserve these guards; no later phases or guard framework.

## Acceptance criteria

1. Both F32/F64, both traversals, fixed T=0/nonzero; static B=0/1/ordinary and one canonical named B.
   Same normalized DynamicDimension name compares by VALUE, not identity/hard-coded spelling.
   Exact B only at identified ROW(0)/rank-two ROW(1) axes and lengths <span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[B]`; all features/SHARED
   declarations/values static, including unused ones. Reject unknown, independent/repeated/derived B,
   variable T/features, hidden shape-attribute symbols and SHARED-B; never narrow all-valid source.<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-feature); [shape glossary](../../../../glossary.md#shape))</span>
2. Preserve bounded algebra, row broadcast/[transpose](../../../../glossary.md#term-transpose)/SUM/sumToShape, terminal taint and exact own
   `old.add(contribution)` rules. F=B/H=B, B=0/1 never infer/erase roles. Check fixed-feature products
   across ALL declarations/contained values/canonical siblings independently of empty B/T, not only
   reshape or symbolic knownElementCount. Preserve legitimate zero features; no sample B/extent read.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [extent glossary](../../../../glossary.md#term-extent))</span>
3. Exact actual/formal type/shape/flags, positional aliases and compatible outer tied roles; no length
   formal/gradient OR contribution. Old seven-field defaults/counts/overload unchanged; +1 length actual
   only. Static formals/ports canonical, symbolic formals/actuals/ports unresolved; shared histories<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient))</span>
   `[T]+localShape` gain no B [axis](../../../../glossary.md#term-axis). Preserve full sibling inventory, [canonical](../../../../glossary.md#term-canonical-form) port [identity](../../../../glossary.md#term-identity-meanings) and flags.
4. Public apply AND raw attachment validate every last/dead port/unused declaration and recheck body
   storage before output-ID supplier calls. No value/length read, storage/history allocation or execution;
   failures consume no output IDs. No host/free/effectful/nested/noncanonical containment regression.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [regression glossary](../../../../glossary.md#term-regression))</span>
5. Compiler public forward/raw inference reject symbolic lengths; AD preflight rejects BEFORE derivative
   IDs, including T=0 and an unrelated gradient target/source branch. Original-source all-carry/all-sequence
   ROW(0), residual-root/higher-order exclusions stay unchanged. No AD/binding implementation added.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [derivative glossary](../../../../glossary.md#term-derivative); [target glossary](../../../../glossary.md#term-target); [carry glossary](../../../../glossary.md#term-carry); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [binding glossary](../../../../glossary.md#term-binding))</span>
6. Prepare direct/late-partition projection rejects symbolic lengths deliberately before Optional extent
   access and ANY analyzer/resource declaration. CPU provider false/direct lowering IllegalArgumentException;
   raw cold/codec bypasses reject independently, including dead/late/T=0. No incidental NoSuchElementException,
   IndexOutOfBoundsException, raw-wire success or hidden allocation stands in for a supported boundary.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [codec glossary](../../../../glossary.md#term-codec))</span>
7. Static lengths and old static/symbolic-all-valid tests remain green. Numeric/ABI/backend identities,
   generated producers/emitters and bytes are untouched; retain source hashes, not new timing evidence.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer); [hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused development:

```bash
./gradlew :modules:model:test --tests '*Scan*Test' :modules:compiler:test --tests '*GeneralScanCompilerTest' --tests '*LengthAwareScan*Test' :modules:prepare:test --tests '*PrepareContextTest' --tests '*BackendPartitionPreparerTest' --max-workers=2 --console=plain
```

After executable freeze, final affected-[module](../../../../glossary.md#term-module) tests/[Javadoc](../../../../glossary.md#term-javadoc) ONCE; consumer test tasks also [compile](../../../../glossary.md#compile)
their main/test sources and exercise independent rejects/static compatibility:

```bash
./gradlew :modules:model:test :modules:model:javadoc :modules:compiler:test :modules:compiler:javadoc :modules:prepare:test :modules:prepare:javadoc :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuGeneralScanLoweringTest' :extensions:checkpoint:test --tests '*GraphArtifactCodecTest' --tests '*GraphCheckpointGeneralScanTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --max-workers=2 --console=plain
```

MAIN ran `Model0044-source-rejection-root` after review/freeze, reusing worker/up-to-date evidence.
No worker/reviewer root or duplicate Java tests. Existing [Vector](../../../../glossary.md#term-vector) module is configured.

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --max-workers=2 --console=plain
```

MAIN verified packaged [dylib](../../../../glossary.md#term-dylib) SHA256 `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`;
set SYNAPTIK_METAL_TEST_LIBRARY, no native rebuild or timing/ClassFile campaign: no emitter changed.
Validate owned links/anchors, whitespace/new-file guardrails (≤200 lines/15KB),
exact baseline-relative [scope](../../../../glossary.md#term-scope) and unchanged protected [hashes](../../../../glossary.md#term-hash)/HEAD/origin/index. No reviewer Java reruns.

## Documentation and review impact

Profiles: general + [planning](../../../../glossary.md#term-planning) for this/master/roadmap/focused plans; architecture for root/[scan](../../../../glossary.md#term-cumulative-scan) [owner](../../../../glossary.md#term-owner)
and training-[graph](../../../../glossary.md#graph); [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) for [tensor](../../../../glossary.md#tensor)-api/affected members. Follow documentation-rules.md.
Planning docs: this brief, [Model](../../../../glossary.md#term-model) master, roadmap, length-aware-general-scan, dynamic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-execution.
Authority/explanation: ARCHITECTURE.md, recurrent-scan.md, training-graph.md and tensor-api.md only.
Independent pre-Java review approved this bounded delta. MAIN's exclusive final-doc delegation adds
ONLY glossary [`Symbolic Scan`](../../../../glossary.md#symbolic-scan)/[`General differentiable scan`](../../../../glossary.md#general-differentiable-scan) source-status corrections and comment-only
Javadoc within the 15 frozen Java files. Review reused executable evidence; MAIN's root passed final
Javadoc [validation](../../../../glossary.md#term-validation) after comment edits. Read-only ScanBody/ScanAttrs/ScanResult/CompiledScanBody
Javadocs remain accurate. No new term or [ADR](../../../../glossary.md#term-architecture-decision-record-adr): existing scan decisions 0049/0051 and mechanisms suffice.

## Result

[Scan](../../../../glossary.md#term-cumulative-scan).java source + ScanGradientRules.java/[PrepareContext](../../../../glossary.md#prepare-context-preparecontext).java early rejects; 12 test files (one new).
Static/one-B [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) source; Mencius independent APPROVE: 0 P1/P2;
MAIN accepted Complete 2026-10-26. Worker [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Checkpoint](../../../../glossary.md#term-checkpoint): 1199/386/97/35/17;
1734 total, zero failures/errors/skips, three Javadocs passed. Evidence above:
`implementation.oF5NNj/worker-result.txt` (commands/fixes/XML/[hashes](../../../../glossary.md#term-hash)) and `final-review.HNrw6X/`.
MAIN's exact root command above passed test/[Javadoc](../../../../glossary.md#term-javadoc), exit 0: BUILD SUCCESSFUL in 5m33s; 119 tasks
(41 executed, 78 up-to-date). `root-checkpoint.log` / `root-final-counts.json` record 707 suites /
5016 tests, zero failures/errors, 56 skips; includes UP-TO-DATE evidence, not all newly executed.
Ignored old `extensions/{training,checkpoint}` outputs excluded; included projects map to
`modules/{training,checkpoint}`. Skips: CPU 43, [Metal](../../../../glossary.md#term-metal) 10, [Engine](../../../../glossary.md#term-engine) 2, Integration 1.
Root source/HEAD/origin/index/native-package SHA unchanged; whitespace PASS. Closeout: ten owned
Markdown paths only, no executable changes/new authority/test reruns. All symbolic-length downstream
[inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd)/[binding](../../../../glossary.md#term-binding)/projection/execution/cold/v5/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training gates stay closed. Active Model frontiers:
None; all successors held, none authorized. Issues/follow-up: none.

Status: Complete

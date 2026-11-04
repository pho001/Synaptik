# Task Model0038: Length-aware [Scan](../../../../glossary.md#term-cumulative-scan) source/[IR](../../../../glossary.md#term-intermediate-representation-ir) foundation

**Editorial reading key (current context, not a status update).**

This record covers Model0038’s source/IR foundation and rejection seams, not today’s downstream
scope. The caller supplies actual Tensors to `Scan.apply`; `ScanBody` supplies positional formals
and expression roots. A formal is a storage-free leaf placeholder: “leaf” means no producing
expression, not stored data. Carry passes state between steps; sequences supply slices; invariants
stay unchanged. Lengths is an extra actual with no body formal or gradient contribution. Binding
checks descriptions without reading length values; storage-free expressions neither execute nor own
result buffers.

ROW identifies a proved independent batch coordinate; SHARED lacks that coordinate, rather than
meaning scalar. For example, a singleton shared bias can broadcast into a row value, but reducing
its row axis cannot feed row feedback even when shape stays unchanged. Follow [current length-aware
construction](../../../../reference/tensor/general-scan.md#length-aware-source-construction-and-qualified-execution)
for today’s role and qualification boundaries. Model semantics defines meaning; providers realize
supported routes, not redefine it. Canonical port ownership concerns wrapper identity, not storage
ownership; [residual
metadata](../../../../reference/tensor/general-scan.md#canonical-results-residual-metadata-and-eligibility)
explains why declaring histories allocates no buffers.

## Status

Complete — MAIN accepted 2026-10-24 after independent Class C/[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc)/docs APPROVE,
zero open P1/P2, and successful combined [compile](../../../../glossary.md#compile)/architecture/root test/Javadoc retry.

## Change class

Class C — public record/[API](../../../../glossary.md#term-application-programming-interface-api) and scoped actual/formal boundary with bounded consumer rejection guards.
Clean implementation and independent targeted API/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review are mandatory.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver one cohesive [Model](../../../../glossary.md#term-model) source/[IR](../../../../glossary.md#term-intermediate-representation-ir) capability: approved header and overload, explicit boundary
row roles, bounded body checks, [canonical](../../../../glossary.md#term-canonical-form) ports and attachment/[cardinality](../../../../glossary.md#term-cardinality) [validation](../../../../glossary.md#term-validation). Small named
downstream guards below only reject the new mode; they are safety seams, not capability work.
No new production type/package/enum, [dependency](../../../../glossary.md#term-dependency), [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings)/[inference](../../../../glossary.md#term-inference)/[AD](../../../../glossary.md#autograd) support, query
[widening](../../../../glossary.md#term-widening), [CPU](../../../../glossary.md#term-cpu) route, [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding), cold copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) or v5 reader/writer implementation.
Integration-only Class A repair: one existing capability-ledger test serializer preserves the old
seven-field default projection and distinguishes new roles; no production/[fixture](../../../../glossary.md#term-fixture) capability change.

## Contracts

Read AGENTS.md and root [scope](../../../../glossary.md#term-scope) index, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
[Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
[Documentation authority and architecture updates](../../../../../ARCHITECTURE.md#documentation-authority-and-architecture-updates),
[Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
Read recurrent-[scan](../../../../glossary.md#term-cumulative-scan) [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities)<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>,
[Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
and [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
with every subheading: Header, Bounded row-role analysis, Activity/histories, First-order reverse,
[Runtime](../../../../glossary.md#term-runtime) [preflight](../../../../glossary.md#term-preflight)/gates. Do not implement the later runtime/[AD](../../../../glossary.md#autograd) duties.
Boundary headings: foundational [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [model glossary](../../../../glossary.md#term-model))</span>;
[compiler](../../../../glossary.md#term-compiler) [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
[Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>;
runtime [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>;
[backend](../../../../glossary.md#backend) [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>;
extensions [Canonical graph-model version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
and [version 5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5).
Stop on a concrete conflict; do not reinterpret the accepted design or invent architecture.

## Dependencies and integration

- Depends on: Model0037 Complete after independent Class C approval and MAIN acceptance;
  accepted Model0035 and its downstream baseline remain Complete.
- Conflicts with: scan API/roles/AD/inference/cold/Prepare/CPU/codec writers and NN0032 migration;
  all shared authority/planning/explanation edits. Serialize; MAIN verified no other writer at Ready.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [AD glossary](../../../../glossary.md#autograd); [inference glossary](../../../../glossary.md#term-inference); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [codec glossary](../../../../glossary.md#term-codec); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  297 accepted dirty baseline paths / 2425 Git-listed files plus 15 accepted authority deltas
  (12 modifications, 3 new files), all uncommitted, not merged predecessor commits.<span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Integration order: Model0037 review/MAIN acceptance → MAIN Ready verification → clean Model0038
  implementation → independent review → MAIN compatibility checkpoint/acceptance.
- Integration [validation](../../../../glossary.md#term-validation): combined named [compile](../../../../glossary.md#compile)/architecture/root [checkpoint](../../../../glossary.md#term-checkpoint) below; cached evidence reused.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

[Ready](../../../../glossary.md#term-ready) verification by MAIN, 2026-10-24: user authorized this sole [Model](../../../../glossary.md#term-model) source frontier after all
dependencies, including independently approved/MAIN-accepted Model0037, were Complete. MAIN
verified the recorded base/accepted dirty baseline, empty index, no competing writer/conflict,
zero outside/missing paths and unchanged native [SHA-256](../../../../glossary.md#term-sha-256)
`03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
Backup/[handoff](../../../../glossary.md#term-handoff): `/private/tmp/synaptik-length-scan-implementation.Dma3c9/`; preserve accepted work.
Source is frozen/independently approved and MAIN-accepted; frontier None, no successor Ready.

## Files and symbols

Paths below are relative to repo; Java suffixes share `src/main/java/io/github/pho001/synaptik/`.

- `modules/model/` + `model/operation/scan/ScanAttrs.java`: record, compact/old constructor,
  inputCount/outputCount. Append rowCarryCount, sequenceRowAxes after reverse; snapshot list.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `modules/model/` + `model/tensor/Scan.java`: new eight-argument apply, validateBindings,
  validateAttachment, descriptors; bounded Tensor-ID-allocation-free row checks in this existing owner.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [descriptors glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [owner glossary](../../../../glossary.md#term-owner))</span>
  `model/tensor/ScanBody.java`: only directly affected containment/[Javadoc](../../../../glossary.md#term-javadoc) if needed; keep inventory.
- `modules/model/` + `model/graph/CompiledNode.java`: structural local count `c+s+i`, outer
  attrs.inputCount; `model/graph/CompiledScanBody.java`: representation unchanged, Javadoc review.
- Reuse `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/scan/ScanSemanticsTest.java`,
  `model/tensor/{ScanBodyTest,ScanExpressionTest,ScanSemanticsTest}.java` and
  `model/graph/CompiledScanBodyTest.java` under that same test prefix.
- Integration repair only: `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/LowPrecisionCapabilityLedgerTest.java`; no [fixture](../../../../glossary.md#term-fixture) edits.

Only these downstream production files may change, for explicit rejection/Javadoc, no reimplementation:

- `modules/compiler/` + `compiler/ScanBodyInference.java#infer`: reject rowCarryCount>0 before
  indexing local inputs. `compiler/ScanGradientRules.java#preflight`: reject before derivative IDs;
  existing whole-original-inventory caller covers unrelated branches. Leave old AD formulas alone.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [AD glossary](../../../../glossary.md#autograd))</span>
- `modules/compiler/` + `compiler/SourceGraphReconstruction.java#preflight`: reject length mode
  in complete outer/body preflight, including late/dead/T=0, before any formal/producer IDs.
  Never use reconstructScan's old overload or old trailing-invariant slice for this mode.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [producer glossary](../../../../glossary.md#term-producer); [invariant glossary](../../../../glossary.md#term-invariant); [slice glossary](../../../../glossary.md#slice))</span>
- `modules/prepare/` + `prepare/analysis/PrepareContext.java#validateScan`: reject before
  actual/formal indexing/analyzers. No projection/resource widening.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening))</span>
- `modules/backend/cpu/` + `backend/cpu/internal/lowering/CpuGeneralScanLowering.java#analyze`:
  common reject before pointwise/shaped dispatch, or equivalent guards in BOTH paths. Provider
  supports remains false; shaped fallback must not bypass rejection. No IR/emitter/identity edits.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [dispatch glossary](../../../../glossary.md#term-dispatch); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `modules/checkpoint/` + `checkpoint/graph/GraphArtifactCodec.java#opcode`: reject length mode
  before opcode23/version selection/dry write/storage admission. Explicit guard protects direct
  codec use even if cold Compiler preflight is bypassed; no body-count workaround/v5 serialization.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [codec glossary](../../../../glossary.md#term-codec); [Compiler glossary](../../../../glossary.md#term-compiler); [serialization glossary](../../../../glossary.md#term-serialization))</span>

[Guard](../../../../glossary.md#term-guard) [regression](../../../../glossary.md#term-regression) files only: `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/`
`{GeneralScanCompilerTest,ScanGradientRulesTest,SourceGraphReconstructionTest}.java`;
`modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/analysis/PrepareContextTest.java`;
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuGeneralScanCapabilityTest.java`;
`modules/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/graph/`
`{GraphArtifactCodecTest,GraphArtifactV3Test,GraphArtifactV4Test}.java`.

## Acceptance criteria

- Old seven-[argument](../../../../glossary.md#term-argument) ScanAttrs and five-argument [Scan](../../../../glossary.md#term-cumulative-scan).apply select `(0, empty)` all-valid, with
  old counts/descriptors/canonical wrappers/flags/PURE/type inventory intact. New record equality,
  hash/text/reflection shape is not promised unchanged; no key may derive from changed record text.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [PURE glossary](../../../../glossary.md#term-pure); [hash glossary](../../../../glossary.md#term-hash); [shape glossary](../../../../glossary.md#shape))</span>
- New attrs validates positive bounded rowCarryCount, exact immutable/non-null ordered axes,
  ROW0 or rank-two ROW1 binding, checked <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [binding glossary](../../../../glossary.md#term-binding))</span>`c+s+i+1`; malformed all-valid/length-mode combinations
  reject. Local formal count is always `c+s+i`, lengths has no formal or [gradient](../../../../glossary.md#term-gradient) contribution.
- New apply uses the exact approved [signature](../../../../glossary.md#term-signature). Split invariants at `c+s+i`, not inputs.size();
  validate final non-gradient INT64[B], common T/B, matching formals, all roles/siblings/storage
  and static arithmetic before IDs. No reading lengths/storage values or execution.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- New source bodies are homogeneous F32/F64 over only tensor/scalar ADD/SUB/MUL/DIV,
  NEG/TANH/SIGMOID, rank-two MATMUL/[1,0] PERMUTE, SUM/sumToShape, EXPAND/RESHAPE/rank edits,
  CONTIGUOUS and same-dtype CAST with existing family guards. No backend predicate in Model.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [rank glossary](../../../../glossary.md#term-rank); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [dtype glossary](../../../../glossary.md#data-type--datatype); [backend glossary](../../../../glossary.md#backend); [Model glossary](../../../../glossary.md#term-model))</span>
- Apply the owner's exact right-aligned row broadcasting (SHARED missing row axis or singleton),
  rank-two products/transposes and normalized SUM axes. ROW0 reshape preserves B and checked
  trailing feature product; expand/rank edits affect trailing features only; no ROW1 shape edits.
  Never infer a role from rank, requiresGrad or equal extents, including B=F/1/0. Explicitly test
  ROW0 <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [broadcasting glossary](../../../../glossary.md#broadcasting); [feature glossary](../../../../glossary.md#term-software-feature))</span>`[1,H]` + SHARED `[1,H]` accepted; non-[singleton](../../../../glossary.md#term-singleton) shared row-[axis](../../../../glossary.md#term-axis) broadcast rejects.
- Row-reduced dependencies are terminal shared contributions only, never broadcast back into
  row feedback/emissions. Shared suffix formal occurs solely as exact old/left operand of its
  own matching old.add(contribution) root; contribution is free of every shared carry formal.
  Row roots/emissions are ROW0. Reject unsupported/unproved paths before canonical output IDs.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry))</span>
- Preserve residual order, previous shared/row carries, aliases, no-extra-B shared geometry,
  independent scopes and conservative OR flags. Test explicit ROW0/ROW1/SHARED sequence bindings
  even at equal sizes and unused declarations. CompiledNode structural acceptance is not execution.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Test nulls/mutation snapshots/count overflow; both directions; T/B=0; template reuse/tied
  positions; invalid row broadcasts/reshape/row reduction feedback/shared carry coupling;
  direct producer attachment failures with ID supplier probes. Old construction tests stay green.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [producer glossary](../../../../glossary.md#term-producer))</span>
- Each named downstream gate explicitly rejects the entire mode; test no IndexOutOfBounds fallback,
  no derivative/formal/producer IDs or copy on relevant preflight failures, late/dead/T=0 cold
  rejection, CPU pointwise and shaped examples false, direct codec pre-write failure, old v1–v4
  fixtures unchanged. Do not implement runtime skipping, actual AD, role pruning or v5 to pass.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [preflight glossary](../../../../glossary.md#term-preflight); [CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [codec glossary](../../../../glossary.md#term-codec); [runtime glossary](../../../../glossary.md#term-runtime); [AD glossary](../../../../glossary.md#autograd))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused iteration then one final [Model module](../../../../glossary.md#model-module-buffer-and-scan-distinctions) gate; consumer guards have focused regressions:

```bash
./gradlew :modules:model:test :modules:model:javadoc
./gradlew :modules:compiler:test --tests '*GeneralScanCompilerTest' --tests '*ScanGradientRulesTest' --tests '*SourceGraphReconstructionTest' :modules:compiler:javadoc
./gradlew :modules:prepare:test --tests '*PrepareContextTest' :modules:prepare:javadoc
./gradlew :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' :backends:cpu:javadoc
./gradlew :extensions:checkpoint:test --tests '*GraphArtifactCodecTest' --tests '*GraphArtifactV3Test' --tests '*GraphArtifactV4Test' :extensions:checkpoint:javadoc
```

MAIN combined named [compile](../../../../glossary.md#compile)/architecture and existing-native root test/[Javadoc](../../../../glossary.md#term-javadoc) [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew :modules:planning:compileJava :modules:compiler:compileTestJava :modules:prepare:compileTestJava :backends:cpu:compileTestJava :extensions:checkpoint:compileTestJava :extensions:nn:compileTestJava :testing:backend-conformance:compileTestJava :testing:integration-tests:compileTestJava :testing:architecture-tests:test test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
git diff --check
```

Reuse source-matched successful/cached evidence and the existing [native library](../../../../glossary.md#term-native-library); no native build,
forced rerun or performance campaign. Both attempts and the bounded test-only repair are below.

## Documentation and review impact

Review/finalize source-only Java/[API](../../../../glossary.md#term-application-programming-interface-api) contracts, examples/glossary/status under documentation profiles.

## Result

Source/[IR](../../../../glossary.md#term-intermediate-representation-ir) and guards implemented; old all-valid behavior/[identities](../../../../glossary.md#term-identity-meanings) preserved. Worker: 1271 tests,
zero skips/failures/errors, five Javadocs PASS. Source/[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc)/docs and ledger reviews APPROVE,
zero open P1/P2; [singleton](../../../../glossary.md#term-singleton) sumToShape, exact contribution [shape](../../../../glossary.md#shape) and zero-[feature](../../../../glossary.md#term-feature) P2s fixed/tested.
Combined attempt 1 FAILED in 5m31s (89 tasks: 35 executed/54 UP-TO-DATE), solely ledger reflection
including new default record fields. Class A repair touched only LowPrecisionCapabilityLedgerTest:
+74 lines/two regressions, explicit old-seven projection and distinct OptionalInt [metadata](../../../../glossary.md#term-metadata); v1/v2
fixtures unchanged. Its nine tests now PASS, zero skips. No provider/[descriptor](../../../../glossary.md#term-descriptor)/[kernel](../../../../glossary.md#kernel) change.
Same combined command retry PASS in 21s (119 tasks: 18 executed/101 UP-TO-DATE), reusing successful
source-matched work, no forced/performance campaign. Current-project XML: 684 suites, 4838 reported
= 4789 non-skipped + 49 skips, zero failures/errors; 13 old relocated XML excluded. Baseline-identical
skips: 38 [CPU](../../../../glossary.md#term-cpu) opt-in/strategy/diagnostic, ten [Metal](../../../../glossary.md#term-metal) [observer](../../../../glossary.md#term-observer), one EngineModelAutotuning assumption.
Architecture 25/0 skips, [conformance](../../../../glossary.md#term-conformance) 63/0, integration 212/1; CPU 1237/38; [Model](../../../../glossary.md#term-model) 1177, [Compiler](../../../../glossary.md#term-compiler) 350,
[Prepare](../../../../glossary.md#prepare) 80, [Checkpoint](../../../../glossary.md#term-checkpoint) 63 all zero skips. Seven native Train/[EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid) cases freshly PASS unskipped.
All Javadocs PASS: 94 CPU/46 Metal warnings in unchanged files plus incubator/[FFM](../../../../glossary.md#term-ffm)/reflection notices.
Evidence in backup directory: `integration-summary.json` (exact commands/data),
`integration-attempt-{1,2}.log`, `worker-handoff.md`. Docs finalized under selected profiles;
Javadoc accurate without edits after worker freeze. Historical 0036/0037 briefs untouched;
downstream length support still rejects. Checks: `check-docs.mjs all-paths.json` PASS (21 files,
982 links/590 fragments), diff/newlines/sizes PASS. Preservation: 2428 files, 40 existing + three new,
zero outside/extra/missing; HEAD/origin/index/native unchanged. All 22 worker Java [hashes](../../../../glossary.md#term-hash) unchanged;
plus only the approved ledger-test repair. MAIN accepted 2026-10-24; frontier None, no successor [Ready](../../../../glossary.md#term-ready).
Status: Complete

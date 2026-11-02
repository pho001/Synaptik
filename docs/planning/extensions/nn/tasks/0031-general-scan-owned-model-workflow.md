# Task NN0031: General-scan owned Model workflow

## Status

Complete — main accepted implementation, independent review, final module validation and the
single configured native-enabled program join on 2026-10-23. All program frontiers None.

## Change class

Class C — qualify a cross-owner NN acquisition/training workflow and its prepared-resource,
private-state and native-storage lifetime boundaries; no new architecture decision is selected.

## Goal

Enable existing `Model.graph`/`defineTensor` acquisition of bounded general Scan sources, then
compute and configured first-order training through the existing Engine/Training owners.
Acquire private state once → prepare once per execution role → bind fresh batches repeatedly.

## Scope

Remove the temporary NN GENERAL-acquisition protection introduced for CP0012 only as part of
this qualified cutover. Delegate constructor-backed source admission/copy/rebind to Checkpoint
and Compiler; never replace it with CPU dtype, operation or layout predicates. Preserve Source
container/null snapshots and one definition/prototype-forward attempt before acquisition.
Qualify all-valid canonical static-T homogeneous FLOAT32/FLOAT64 CPU compute, gradients and
training/weight updates, using existing list and single-Tensor Model overloads and owners.
Cold `forward` remains explicit reconstruction, never an execution prerequisite or per-batch path.

## Non-goals

No new production type (including nested), package, public API, dependency, runner/session,
graph representation, AD rule, numerical/backend/generated-code change or build configuration.
No body persistence/wire/opcode, save/load widening, padding/lengths/row skipping, fixed builder
redirection, fixed `rnnScan` training/BPTT, low-type execution qualification or performance campaign.
No native build/code/test-source/gating/skip correction, commits or pushes. If an upstream change,
policy/API decision or architecture ambiguity is needed, stop with evidence; do not widen scope.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules`,
  `Testing requirements` — retain compile/prepare/run separation and existing public owner edges.
- [Extensions/training](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `extensions/nn`, `Owned tensor Models`, `modules/training`, `Generic graph source and values`,
  `Optimizer/training lifecycle` — staged general admission, private state, fixed-only rejection,
  frozen binding, native transition, transactional updates, invalidation and cleanup.
- [Scan semantics](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `General differentiable scan` and its `Source body, attachments and scoped identities`,
  `Canonical residual ports and demand`, `Reverse scan and exact primal bindings`,
  `Traversal, empty domains and declared qualification`, `Qualification, performance and migration gates`.
- [Compiler cold boundary](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  `Compile lifecycle`, including `Bounded general-scan cold reconstruction` — unchanged outer
  inventory plus exact body inventory; source/dtypes are independent of CPU/AD/wire admission.
- [Engine owner](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution):
  `Public reusable prepared execution` — fresh exact-descriptor values reuse one prepared owner.

The approved direction is now qualified within this scope, not automatic train/save/load support
for arbitrary bodies. Fixed-only `containsScan`/`rejectScanTraining` keep their meaning.

## Dependencies and integration

- Depends on: NN0030 direction; Compiler0012; CP0012; CPU0030–0033 and accepted low-level join;
  existing NN0028/0029/0022B, Engine0021, Training0007/0008 and Checkpoint0008–0011 — all Complete.
  Accepted CPU0034–0036 changes are part of the frozen baseline, not new work here.
- Conflicts with: NN source/API/lifecycle and shared authority/docs; Compiler/Checkpoint cold
  boundary, Training/Engine fresh-value owners and CPU scan semantics. None active; serial writers.
- Parallel group: None. Serial implementation, independent review and integration are complete.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all 284
  accepted dirty paths, exactly CP0012 final, plus four verified planning deltas.
  Baseline: `/private/tmp/synaptik-nn0031-baseline.yieMqy/{accepted,snapshot.json}`.
- Integration order: main Ready review/reverification → clean NN implementation/yield → independent
  targeted Class C review/final docs and Javadoc under exclusive delegation → worker final
  NN test/Javadoc → main single native-enabled program join → main acceptance. No automatic successor.
- Integration validation: final native-enabled root `test javadoc` below PASS, including architecture,
  new workflow integration and reused conformance/module/low-level successes without forced reruns.
- Shared-document integration owner: main. Worker owns scoped Java/tests and draft Javadoc;
  after source writer yields, main may exclusively delegate final docs/Javadoc writes to the clean
  independent reviewer. No overlapping writers; main retains integration ownership and acceptance.

Historical launch verification (2026-10-23): main confirmed sole Ready frontier, Complete dependencies/
reviews/gates, stable contracts/base/hashes, empty index and inactive conflicts/no competing writer.

## Files and symbols

Java paths below are rooted beneath `io/github/pho001/synaptik/` in their named source trees.

- `extensions/nn/src/main/java/.../nn/module/Model.java` — graph/defineTensor acquisition and Javadoc;
  `OwnedModelSource.java` — acquisition, inventory/adopt/forward, temporary `rejectGeneralScan`,
  fixed-only `containsScan`; `ModelExecutionState.java` — computeOpen/configureOpen/trainOpen,
  bindings/values and rejectScanTraining.
- NN focused tests: `OwnedModelSourceTest`, `OwnedModelExecutionTest`, `OwnedModelTrainingTest`,
  `OwnedModelLifecycleTest`, `RuntimeLengthScanModelTest`; added `GeneralScanOwnedModelTest`.
  Temporary GENERAL-rejection expectations were replaced, not fixed guards.
- Added `testing/integration-tests/src/test/java/.../testing/integration/GeneralScanOwnedModelIntegrationTest.java`.
- Read-only upstream seams: GraphCheckpoint copy/rebind/save, Engine prepare/runValues and Training
  session/native admission. No upstream, CPU, native or build edits.

## Acceptance criteria

1. Graph and definition acquisition accept the exact existing cold source/body inventory, not all
   PURE operations and not just CPU-supported signatures. Include cold low-type TANH and F32 RELU
   without claiming execution; cold-unsupported bodies still fail through the existing owner.
   Test both traversal flags, T=0/positive T, final-carry/emission/residual-only roots and ordinary
   heads within the unchanged outer inventory. Residual-root differentiation remains rejected.
2. Each external state identity is acquired once, preserving full descriptors, canonical port order,
   occurrences, repeated actual positions, role aliases and deterministic wrapper paths. Inputs
   become storage-free prototypes; constants stay outside stateDictionary. Cold forward with produced
   exact-descriptor substitutions preserves upstream provenance and private state; no Engine is created.
   Definition/prototype callbacks run once and are discarded; caller Tensors/stores/bytes stay unchanged.
3. Both F32/F64 graph/list and definition/single workflows compute on fresh batches. Compare final
   carry and emissions to an independent unrolled Tensor/Engine or represented-arithmetic oracle,
   including reverse original-index order, T=0 and representative empty step domains. Reuse low-level
   algebra coverage; add representative pointwise and broadcast/SUM or non-square MATMUL training cases,
   not another full backend qualification matrix.
4. Configure once, build one scalar objective per explicit attempt, and train at least two distinct
   fresh batches/targets with Sgd. Check pre-update losses, actual first-order parameter gradients
   against an independent unrolled oracle and expected weight deltas (loss decrease alone is insufficient).
   Include T>=3 with non-gradient initial carry and a shared trainable invariant, both traversal flags,
   and an objective consuming final-carry plus emissions through the list boundary. T=0 checks obey
   existing Training connected/nonempty parameter admission; do not invent a waiver for disconnected state.
5. Repeated compute reuses the same inference PreparedExecution; repeated train reuses the same
   TrainingSession/prepared graph, positional plans and admitted native stores. Assert stable owned
   Tensor/wrapper identities, one native transition, closed old inference before that transition,
   post-update compute observing those same weights, zero retained RunResults and no prototype/storage
   mutation or callback/reconstruction/compile/prepare/provider lookup per batch. Use existing test seams,
   not a production instrumentation API or timing campaign.
6. Preserve full-signature validation (including unused slots), aliases in inference and Training's
   alias/overlap rejection, native-copy failure/failed-session retry, frozen Engine binding on setup
   failure, live mode/wrapper/Tensor/store invalidation, busy/closed precedence and close wait/cleanup.
   Extend representative scan cases and reuse unchanged lifecycle tests; detached values/TrainingStep
   outlive caller batches and Model close, and borrowed Engines are never closed by Model.
7. Fixed RNN/GRU/LSTM detection, inference, all configure/train rejection overloads and their no-effect
   precedence remain unchanged, including heads/final-state-only roots and after compute/load.
   Mixed fixed/general outer sources still reject fixed training; static unrolled training still works.
8. General-body save still rejects before file effects without freezing binding or disabling compute/
   train; cover an existing sentinel file and subsequent usable workflow. v1/v2/v3 load/wire rejection
   and old fixed/static paths remain unchanged. No new persistence implementation or CP source edit.

## Validation

Recorded final affected-module command after independent documentation/Javadoc finalization:

```bash
./gradlew :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain
```

Recorded main's single final program join, with explicit existing-library environment:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

Both commands passed; fresh/cached evidence is distinguished in Result. Existing library SHA256
`03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a` stayed unchanged,
matching packaged ABI7/schema20; preflight/source identity is in `native-preflight.json` under the
evidence directory. No native build, gating/skip edit, exclusion or performance opt-in occurred.
The successful configured invocation does not remove the explicit native-library environment requirement.
Unchanged Compiler/Checkpoint/CPU gates were reused, not repeated; CPU root results were cached.

## Documentation and review impact

General/planning profiles govern this closure; main retains integration ownership. After source-writer
yield, independent review under exclusive delegation finalized ten documents/three Java Javadocs with
matching profiles. No new terminology/ADR/policy; source/execution/wire and fixed-only training remain
distinct. This closure changes six granted plans only, without repeating executable validation.

## Result

Status: Complete — implementation, independent review, final module and native program join accepted.

- Removed only the temporary GENERAL acquisition guard; one private state set and cached public owners
  now support the bounded all-valid workflow. No new production type/API/edge or backend/numerical change.
- Java: Model, OwnedModelSource, ModelExecutionState Javadoc; OwnedModelSourceTest updated;
  GeneralScanOwnedModelTest and GeneralScanOwnedModelIntegrationTest added. Ten documents finalized.
- Six plans closed: this brief, NN master, roadmap, general-scan program, plus targeted current status in
  `docs/planning/extensions/checkpoint/master-plan.md` and `docs/planning/modules/compiler/master-plan.md`.
  Those two final planning additions bring authorized task paths from 20 to 22; predecessor briefs unchanged.
- Independent review PASS, zero findings; 580 links/anchors checked, three Java Javadocs finalized.
- Focused 22 fresh tests passed, zero failures/errors/skips; includes four new workflow integration cases.
- Final module command above PASS: NN 542 fresh, zero failures/errors/skips; Javadoc zero warnings.
- Single root command above PASS, exit 0/20s, no exclusions or duplicate attempts; root NN/CPU cached.
- Current reports: 4782 reported = 4733 passed + 49 skipped, zero failures/errors; not all newly run.
  Root fresh: Metal 356 reported/10 skipped, integration 210/1 skipped, architecture 25/0 skipped, all pass;
  four new workflow integrations fresh/pass. Thirteen stale pre-relocation reports excluded from counts.
- Both required TrainingSessionCpuIntegrationTest Metal cases and four MetalExp/Sigmoid native cases
  executed unskipped/pass. Earlier 19 environment failures resolved by explicit existing-library env,
  not code/gating changes or a native build; library hash above unchanged. Both performance flags false.
- Main verified source/docs freeze unchanged after gates, all 284 baseline paths preserved outside
  authorized deltas, HEAD=origin at launch base and index empty; final planning preservation check is main-owned.
- Evidence: `/private/tmp/synaptik-nn0031-evidence.ZuAAT1/` final-validation.json,
  root-test-results.json and root-verified-summary.json. No remaining acceptance issue/follow-up.
  Fixed-scan training, padding, body wire and low-type execution gates remain closed; frontiers None, no successor.

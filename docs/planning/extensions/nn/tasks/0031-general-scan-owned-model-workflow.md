# Task NN0031: General-[scan](../../../../glossary.md#term-cumulative-scan) [owned Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) workflow

> **Editorial reading context — not a status update.** The application supplies source once;
> NN acquires private parameter state through Checkpoint/Compiler, then prepares execution for
> repeated fresh batches. An optional cold forward reconstructs expressions but is not the
> per-batch compute/train path. General-scan constructor support, backend numerical support,
> first-order differentiation and disk persistence are separate gates.
> Today's [owned execution](../../../../reference/nn/owned-model.md#owned-tensor-model-capability)
> and [general-scan copy/rebind](../../../../reference/checkpoint/graph.md#general-scan-in-memory-copy-and-rebind)
> explain those boundaries. Later wire/length support in the references does not change this
> task's all-valid qualification or its recorded native-environment integration evidence.

## Status

Complete — main accepted implementation, independent review, final [module](../../../../glossary.md#term-module) [validation](../../../../glossary.md#term-validation) and the
single configured native-enabled program join on 2026-10-23. All program frontiers None.

## Change class

Class C — qualify a cross-[owner](../../../../glossary.md#term-owner) [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition/training workflow and its prepared-resource,
private-state and native-storage [lifetime](../../../../glossary.md#term-lifetime) boundaries; no new architecture decision is selected.

## Goal

Enable existing `Model.graph`/`defineTensor` acquisition of bounded [general Scan](../../../../glossary.md#general-differentiable-scan) sources, then
compute and configured first-order training through the existing [Engine](../../../../glossary.md#term-engine)/Training [owners](../../../../glossary.md#term-owner).
Acquire private state once → [prepare](../../../../glossary.md#prepare) once per execution role → bind fresh [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) repeatedly.

## [Scope](../../../../glossary.md#term-scope)

Remove the temporary [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) GENERAL-acquisition protection introduced for CP0012 only as part of
this qualified cutover. Delegate constructor-backed source [admission](../../../../glossary.md#term-admission)/copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) to [Checkpoint](../../../../glossary.md#term-checkpoint)
and [Compiler](../../../../glossary.md#term-compiler); never replace it with [CPU](../../../../glossary.md#term-cpu) [dtype](../../../../glossary.md#data-type--datatype), [operation](../../../../glossary.md#operation) or [layout](../../../../glossary.md#layout) predicates. Preserve Source
container/null snapshots and one definition/[prototype](../../../../glossary.md#term-prototype)-forward attempt before acquisition.
Qualify all-valid [canonical](../../../../glossary.md#term-canonical-form) static-T [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) CPU compute, [gradients](../../../../glossary.md#term-gradient) and
training/[weight](../../../../glossary.md#term-weight) updates, using existing list and single-[Tensor](../../../../glossary.md#tensor) [Model](../../../../glossary.md#term-model) overloads and [owners](../../../../glossary.md#term-owner).
Cold `forward` remains explicit reconstruction, never an execution prerequisite or per-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) path.

## Non-goals

No new production type (including nested), package, public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), runner/session,
[graph](../../../../glossary.md#graph) representation, [AD](../../../../glossary.md#autograd) rule, numerical/[backend](../../../../glossary.md#backend)/generated-code change or build configuration.
No body [persistence](../../../../glossary.md#term-persistence)/wire/[opcode](../../../../glossary.md#term-opcode), save/load [widening](../../../../glossary.md#term-widening), [padding](../../../../glossary.md#padding)/lengths/row skipping, fixed builder
redirection, fixed `rnnScan` training/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), low-type execution [qualification](../../../../glossary.md#term-qualification) or performance campaign.
No native build/code/test-source/gating/skip correction, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) or pushes. If an upstream change,
policy/API decision or architecture ambiguity is needed, stop with evidence; do not widen [scope](../../../../glossary.md#term-scope).

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules`,
  `Testing requirements` — retain [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) separation and existing public [owner](../../../../glossary.md#term-owner) edges.
- [Extensions/training](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `extensions/nn`, `Owned tensor Models`, `modules/training`, `Generic graph source and values`,
  `Optimizer/training lifecycle` — staged general admission, private state, fixed-only rejection,
  frozen binding, native transition, transactional updates, invalidation and cleanup.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [binding glossary](../../../../glossary.md#term-binding))</span>
- [Scan semantics](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>:
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan) and its `Source body, attachments and scoped identities`,
  `Canonical residual ports and demand`, `Reverse scan and exact primal bindings`,
  `Traversal, empty domains and declared qualification`, `Qualification, performance and migration gates`.
- [Compiler cold boundary](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>:
  `Compile lifecycle`, including `Bounded general-scan cold reconstruction` — unchanged outer
  inventory plus exact body inventory; source/dtypes are independent of CPU/AD/wire admission.<span data-glossary-navigation="true"> ([dtypes glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu); [AD glossary](../../../../glossary.md#autograd))</span>
- [Engine owner](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>:
  `Public reusable prepared execution` — fresh exact-[descriptor](../../../../glossary.md#term-descriptor) values reuse one prepared owner.

The approved direction is now qualified within this [scope](../../../../glossary.md#term-scope), not automatic train/save/load support
for arbitrary bodies. Fixed-only `containsScan`/`rejectScanTraining` keep their meaning.

## Dependencies and integration

- Depends on: NN0030 direction; Compiler0012; CP0012; CPU0030–0033 and accepted low-level join;
  existing NN0028/0029/0022B, Engine0021, Training0007/0008 and Checkpoint0008–0011 — all Complete.
  Accepted CPU0034–0036 changes are part of the frozen baseline, not new work here.
- Conflicts with: NN source/API/lifecycle and shared authority/docs; Compiler/Checkpoint cold
  boundary, Training/Engine fresh-value owners and CPU scan semantics. None active; serial writers.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compiler glossary](../../../../glossary.md#term-compiler); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Parallel group: None. Serial implementation, independent review and integration are complete.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all 284
  accepted dirty paths, exactly CP0012 final, plus four verified planning deltas.
  Baseline: <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>`/private/tmp/synaptik-nn0031-baseline.yieMqy/{accepted,snapshot.json}`.
- Integration order: main Ready review/reverification → clean NN implementation/yield → independent
  targeted Class C review/final docs and Javadoc under exclusive delegation → worker final
  NN test/Javadoc → main single native-enabled program join → main acceptance. No automatic successor.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Integration [validation](../../../../glossary.md#term-validation): final native-enabled root `test javadoc` below PASS, including architecture,
  new workflow integration and reused conformance/module/low-level successes without forced reruns.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration owner: main. Worker owns scoped Java/tests and draft Javadoc;
  after source writer yields, main may exclusively delegate final docs/Javadoc writes to the clean
  independent reviewer. No overlapping writers; main retains integration ownership and acceptance.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Historical launch verification (2026-10-23): main confirmed sole [Ready](../../../../glossary.md#term-ready) frontier, Complete dependencies/
reviews/gates, stable contracts/base/[hashes](../../../../glossary.md#term-hash), empty index and inactive conflicts/no competing writer.

## Files and symbols

Java paths below are rooted beneath `io/github/pho001/synaptik/` in their named source trees.

- `extensions/nn/src/main/java/.../nn/module/Model.java` — [graph](../../../../glossary.md#graph)/defineTensor acquisition and [Javadoc](../../../../glossary.md#term-javadoc);
  `OwnedModelSource.java` — acquisition, inventory/adopt/forward, temporary `rejectGeneralScan`,
  fixed-only `containsScan`; `ModelExecutionState.java` — computeOpen/configureOpen/trainOpen,
  bindings/values and rejectScanTraining.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) focused tests: `OwnedModelSourceTest`, `OwnedModelExecutionTest`, `OwnedModelTrainingTest`,
  `OwnedModelLifecycleTest`, `RuntimeLengthScanModelTest`; added `GeneralScanOwnedModelTest`.
  Temporary GENERAL-rejection expectations were replaced, not fixed guards.
- Added `testing/integration-tests/src/test/java/.../testing/integration/GeneralScanOwnedModelIntegrationTest.java`.
- Read-only upstream seams: GraphCheckpoint copy/rebind/save, Engine prepare/runValues and Training
  session/native admission. No upstream, CPU, native or build edits.<span data-glossary-navigation="true"> ([rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Engine glossary](../../../../glossary.md#term-engine); [prepare glossary](../../../../glossary.md#prepare); [Training session glossary](../../../../glossary.md#training-session--trainingsession); [admission glossary](../../../../glossary.md#term-admission); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Acceptance criteria

1. Graph and definition acquisition accept the exact existing cold source/body inventory, not all
   PURE operations and not just CPU-supported signatures. Include cold low-type TANH and F32 RELU
   without claiming execution; cold-unsupported bodies still fail through the existing owner.
   Test both traversal flags, T=0/positive T, final-carry/emission/residual-only roots and ordinary
   heads within the unchanged outer inventory. Residual-root differentiation remains rejected.<span data-glossary-navigation="true"> ([Graph glossary](../../../../glossary.md#graph); [PURE glossary](../../../../glossary.md#term-pure); [CPU glossary](../../../../glossary.md#term-cpu); [signatures glossary](../../../../glossary.md#term-signature); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [owner glossary](../../../../glossary.md#term-owner); [carry glossary](../../../../glossary.md#term-carry); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
2. Each external state identity is acquired once, preserving full descriptors, canonical port order,
   occurrences, repeated actual positions, role aliases and deterministic wrapper paths. Inputs
   become storage-free prototypes; constants stay outside stateDictionary. Cold forward with produced
   exact-descriptor substitutions preserves upstream provenance and private state; no Engine is created.
   Definition/prototype callbacks run once and are discarded; caller Tensors/stores/bytes stay unchanged.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [prototypes glossary](../../../../glossary.md#term-prototype); [constants glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run); [Tensors glossary](../../../../glossary.md#tensor))</span>
3. Both F32/F64 graph/list and definition/single workflows compute on fresh batches. Compare final
   carry and emissions to an independent unrolled Tensor/Engine or represented-arithmetic oracle,
   including reverse original-index order, T=0 and representative empty step domains. Reuse low-level
   algebra coverage; add representative pointwise and broadcast/SUM or non-square MATMUL training cases,
   not another full backend qualification matrix.<span data-glossary-navigation="true"> ([qualification matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [oracle glossary](../../../../glossary.md#term-test-oracle); [pointwise glossary](../../../../glossary.md#term-pointwise); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [backend glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification); [matrix glossary](../../../../glossary.md#term-review-matrix))</span>
4. Configure once, build one scalar objective per explicit attempt, and train at least two distinct
   fresh batches/targets with Sgd. Check pre-update losses, actual first-order parameter gradients
   against an independent unrolled oracle and expected weight deltas (loss decrease alone is insufficient).
   Include T>=3 with non-gradient initial carry and a shared trainable invariant, both traversal flags,
   and an objective consuming final-carry plus emissions through the list boundary. T=0 checks obey
   existing Training connected/nonempty parameter admission; do not invent a waiver for disconnected state.<span data-glossary-navigation="true"> ([scalar objective glossary](../../../../glossary.md#loss); [targets glossary](../../../../glossary.md#term-target); [losses glossary](../../../../glossary.md#loss); [parameter glossary](../../../../glossary.md#term-parameter); [gradients glossary](../../../../glossary.md#term-gradient); [weight glossary](../../../../glossary.md#term-weight); [invariant glossary](../../../../glossary.md#term-invariant); [objective glossary](../../../../glossary.md#term-objective); [admission glossary](../../../../glossary.md#term-admission))</span>
5. Repeated compute reuses the same inference PreparedExecution; repeated train reuses the same
   TrainingSession/prepared graph, positional plans and admitted native stores. Assert stable owned
   Tensor/wrapper identities, one native transition, closed old inference before that transition,
   post-update compute observing those same weights, zero retained RunResults and no prototype/storage
   mutation or callback/reconstruction/compile/prepare/provider lookup per batch. Use existing test seams,
   not a production instrumentation API or timing campaign.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [PreparedExecution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [TrainingSession glossary](../../../../glossary.md#training-session--trainingsession); [callback glossary](../../../../glossary.md#term-callback); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [lookup glossary](../../../../glossary.md#term-lookup); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
6. Preserve full-signature validation (including unused slots), aliases in inference and Training's
   alias/overlap rejection, native-copy failure/failed-session retry, frozen Engine binding on setup
   failure, live mode/wrapper/Tensor/store invalidation, busy/closed precedence and close wait/cleanup.
   Extend representative scan cases and reuse unchanged lifecycle tests; detached values/TrainingStep
   outlive caller batches and Model close, and borrowed Engines are never closed by Model.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [scan glossary](../../../../glossary.md#term-cumulative-scan); [lifecycle glossary](../../../../glossary.md#lifecycle); [detached glossary](../../../../glossary.md#detached-model-results); [Model glossary](../../../../glossary.md#term-model); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
7. Fixed RNN/GRU/LSTM detection, inference, all configure/train rejection overloads and their no-effect
   precedence remain unchanged, including heads/final-state-only roots and after compute/load.
   Mixed fixed/general outer sources still reject fixed training; static unrolled training still works.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
8. General-body save still rejects before file effects without freezing binding or disabling compute/
   train; cover an existing sentinel file and subsequent usable workflow. v1/v2/v3 load/wire rejection
   and old fixed/static paths remain unchanged. No new persistence implementation or CP source edit.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>

## [Validation](../../../../glossary.md#term-validation)

Recorded final affected-[module](../../../../glossary.md#term-module) command after independent documentation/[Javadoc](../../../../glossary.md#term-javadoc) finalization:

```bash
./gradlew :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain
```

Recorded main's single final program join, with explicit existing-library environment:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

Both commands passed; fresh/cached evidence is distinguished in Result. Existing library SHA256
`03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a` stayed unchanged,
matching packaged ABI7/schema20; [preflight](../../../../glossary.md#term-preflight)/source [identity](../../../../glossary.md#term-identity-meanings) is in `native-preflight.json` under the
evidence directory. No native build, gating/skip edit, exclusion or performance opt-in occurred.
The successful configured invocation does not remove the explicit native-library environment requirement.
Unchanged [Compiler](../../../../glossary.md#term-compiler)/[Checkpoint](../../../../glossary.md#term-checkpoint)/[CPU](../../../../glossary.md#term-cpu) gates were reused, not repeated; CPU root results were cached.

## Documentation and review impact

General/[planning](../../../../glossary.md#term-planning) profiles govern this closure; main retains integration [ownership](../../../../glossary.md#term-ownership-meanings). After source-writer
yield, independent review under exclusive delegation finalized ten documents/three Java Javadocs with
matching profiles. No new terminology/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/policy; source/execution/wire and fixed-only training remain
distinct. This closure changes six granted plans only, without repeating executable [validation](../../../../glossary.md#term-validation).

## Result

Status: Complete — implementation, independent review, final [module](../../../../glossary.md#term-module) and native program join accepted.

- Removed only the temporary GENERAL acquisition guard; one private state set and cached public owners
  now support the bounded all-valid workflow. No new production type/API/edge or backend/numerical change.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend))</span>
- Java: Model, OwnedModelSource, ModelExecutionState Javadoc; OwnedModelSourceTest updated;
  GeneralScanOwnedModelTest and GeneralScanOwnedModelIntegrationTest added. Ten documents finalized.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Six plans closed: this brief, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master, roadmap, general-[scan](../../../../glossary.md#term-cumulative-scan) program, plus targeted current status in
  `docs/planning/extensions/checkpoint/master-plan.md` and `docs/planning/modules/compiler/master-plan.md`.
  Those two final planning additions bring authorized task paths from 20 to 22; predecessor briefs unchanged.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Independent review PASS, zero findings; 580 links/anchors checked, three Java Javadocs finalized.
- Focused 22 fresh tests passed, zero failures/errors/skips; includes four new workflow integration cases.
- Final module command above PASS: NN 542 fresh, zero failures/errors/skips; [Javadoc](../../../../glossary.md#term-javadoc) zero warnings.
- Single root command above PASS, exit 0/20s, no exclusions or duplicate attempts; root NN/[CPU](../../../../glossary.md#term-cpu) cached.
- Current reports: 4782 reported = 4733 passed + 49 skipped, zero failures/errors; not all newly run.
  Root fresh: Metal 356 reported/10 skipped, integration 210/1 skipped, architecture 25/0 skipped, all pass;
  four new workflow integrations fresh/pass. Thirteen stale pre-relocation reports excluded from counts.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Both required TrainingSessionCpuIntegrationTest Metal cases and four MetalExp/Sigmoid native cases
  executed unskipped/pass. Earlier 19 environment failures resolved by explicit existing-library env,
  not code/gating changes or a native build; library hash above unchanged. Both performance flags false.<span data-glossary-navigation="true"> ([Sigmoid glossary](../../../../glossary.md#term-sigmoid); [hash glossary](../../../../glossary.md#term-hash))</span>
- Main verified source/docs freeze unchanged after gates, all 284 baseline paths preserved outside
  authorized deltas, HEAD=origin at launch base and index empty; final planning preservation check is main-owned.
- Evidence: `/private/tmp/synaptik-nn0031-evidence.ZuAAT1/` final-validation.json,
  root-test-results.json and root-verified-summary.json. No remaining acceptance issue/follow-up.
  Fixed-scan training, padding, body wire and low-type execution gates remain closed; frontiers None, no successor.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [padding glossary](../../../../glossary.md#padding))</span>

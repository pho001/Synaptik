# Task NN0030: General-[scan](../../../../glossary.md#term-cumulative-scan) [owned Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) [admission](../../../../glossary.md#term-admission) decision

> **Editorial reading context — decision-stage findings, not a new implementation claim.**
> Cold admission asks whether supported source can be reconstructed, not whether CPU can
> numerically execute or differentiate it. Model acquisition must privately copy named state;
> retaining a definition callback would leave an external-weight path around that ownership
> boundary. Checkpoint preserves the contained body, Compiler reconstructs its semantics,
> and NN integrates the acquired source with Engine/Training.
> Today's [general-scan cold boundary](../../../../reference/checkpoint/graph.md#general-scan-in-memory-copy-and-rebind)
> and [owned Model consumer](../../../../reference/checkpoint/graph.md#owned-nn-model-consumer)
> explain the completed surrounding concepts; predecessor findings below retain their decision-time meaning.

## Status

Complete — approved limited direction/[planning](../../../../glossary.md#term-planning) decision only, not [module](../../../../glossary.md#term-module) implementation.
Explicit user approval “ok, pust se do toho” is recorded 2026-10-23 for the recommended existing-[owner](../../../../glossary.md#term-owner)
direction and FIRST [Compiler](../../../../glossary.md#term-compiler) prerequisite. Prior independent approval was Draft planning only.
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) implementation frontier: None. [Compiler0012](../../../modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md)
is Complete after independent code/docs APPROVE and main's final gates (2026-10-23); its brief retains
historical pre-code APPROVE/[Ready](../../../../glossary.md#term-ready) verification. Next is separately briefed [Checkpoint](../../../../glossary.md#term-checkpoint) copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction),
then NN [qualification](../../../../glossary.md#term-qualification); Compiler/NN/Checkpoint frontiers None, no successor launch.

## Change class

The approved cold-source/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings) cutover is Class C; this completed decision is not an
implementation grant. Clean implementation and independent targeted review remain mandatory.

## Goal and recommendation

Let an application use an ordinary [Tensor](../../../../glossary.md#tensor) `ScanBody` (a closed pure step expression) through
existing in-memory `Model.graph` / `Model.defineTensor`, then `compute` and
`configureTraining` / `train`. Initial tested compute/train [scope](../../../../glossary.md#term-scope): all-valid, [canonical](../../../../glossary.md#term-canonical-form), static-T
[homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) GENERAL (`ScanKind.SCAN`) scans in the qualified [CPU](../../../../glossary.md#term-cpu) domain; first-order [automatic
differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) remains [Compiler](../../../../glossary.md#term-compiler)-owned. Source/cold-copy [admission](../../../../glossary.md#term-admission) is separate, as specified below.
Keep one private state set and the existing public [Engine](../../../../glossary.md#term-engine), [TrainingSession](../../../../glossary.md#training-session--trainingsession) and GraphCheckpoint [owners](../../../../glossary.md#term-owner).
No new public method/class or production type is presently justified by the source inventory.

## [Scope](../../../../glossary.md#term-scope) and source-backed findings

Paths below are relative to the repository root; findings record the decision-time baseline, not
new contracts. Compiler0012's completed Result supersedes its former cold-reconstruction barrier;
[Checkpoint](../../../../glossary.md#term-checkpoint) attachment preservation remains the next separate gate, not implemented [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) support.

- [NN Model](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)<span data-glossary-navigation="true"> ([NN Model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions))</span>:
  [`graph`](../../../../glossary.md#graph) calls `GraphCheckpoint.copy`; `defineTensor` invokes one eval [prototype](../../../../glossary.md#term-prototype) [callback](../../../../glossary.md#term-callback) and
  `OwnedModelSource.acquireDefinition`, then discards the callback. It is not safe to retain it
  to bypass copying: a closure can still refer to external weights after wrapper installation.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
- [OwnedModelSource](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/OwnedModelSource.java):
  `acquireDefinition` inventories state/[constants](../../../../glossary.md#term-constant), copies once and installs wrappers;
  `forward` delegates to `GraphCheckpoint.rebind` unless this is existing builder composition.
  `acquireComposition` uses `Source` only as a borrowed Tensor-list carrier, without codec/capture.
  Its inventory walks outer operands; body formals must never become external state/constants.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Tensor glossary](../../../../glossary.md#tensor); [carrier glossary](../../../../glossary.md#term-storage-carrier); [codec glossary](../../../../glossary.md#term-codec); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [GraphCheckpoint](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)
  `copy` / [`rebind`](../../../../glossary.md#capture-graph-ports-and-reconstruction) use `GraphSource.capture`, NOT [artifact](../../../../glossary.md#term-artifact) encoding.
  [GraphSource](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphSource.java)
  calls `GraphCompilationPort.captureSource` once, normalizes IDs and reconstructs through Compiler.
  Its <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`new CompiledNode(..., outputs)` currently omits `scanBody`; merely admitting GENERAL in
  Compiler would therefore be insufficient. Preserve the attachment and local IDs during outer remap.
- [Compiler cold owner](../../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/SourceGraphReconstruction.java)<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
  [`preflight`](../../../../glossary.md#term-preflight) admits a closed flat family inventory; `additional` admits fixed `RecurrentScanKind`,
  not `ScanKind.SCAN`. `reconstruct` has no GENERAL branch and lacks some declared body algebra
  (e.g. DIV, SUM and scalar forms). Existing<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
  [GraphCapture.captureSource](../../../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java)
  already captures attachments, but `GraphCompilationPort.SourceCapture` calls this cold preflight.
  Low-level capture/AD success does not prove cold copy/reconstruction support.<span data-glossary-navigation="true"> ([AD glossary](../../../../glossary.md#autograd))</span>
- [ModelExecutionState](../../../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelExecutionState.java)
  `rejectScanTraining` uses `OwnedModelSource.containsScan`, which detects ONLY fixed recurrent kinds.
  Do not remove this guard to admit GENERAL. <span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>`configureOpen` already builds one objective, admits
  native parameter storage once and delegates to <span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [parameter glossary](../../../../glossary.md#term-parameter))</span>`TrainingSession.open`; compute/train cache
  prepared bindings and consume fresh values without per-batch reconstruction.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [bindings glossary](../../../../glossary.md#term-binding); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- [GraphArtifactCodec](../../../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java)
  `encode` / [`opcode`](../../../../glossary.md#term-opcode) independently enforce opcodes 1..22 before storage admission and file publication.
  There is no GENERAL wire. After copy/rebind widening, save must still explicitly reject body
  attachments, including malformed attachments on otherwise encodable kinds, before value/file effects.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [publication glossary](../../../../glossary.md#term-publication); [widening glossary](../../../../glossary.md#term-widening))</span>

Compiler0012 and its inventory audit record focused source/test evidence; no new tests ran here.

## Approved decision and alternatives

Approved: **separate source semantics, [cold reconstruction](../../../../glossary.md#term-cold-reconstruction), execution and wire eligibility**.
Model0035 still admits existing known [PURE](../../../../glossary.md#term-pure) operations at their applicable [Model](../../../../glossary.md#term-model) [dtypes](../../../../glossary.md#data-type--datatype); never use
CPU0033 support as a source-validity/copy predicate. A valid [FLOAT16](../../../../glossary.md#term-float16) [TANH](../../../../glossary.md#term-tanh) or [F32](../../../../glossary.md#floating-point-precision-and-rounding) RELU body remains
valid without an executable [CPU](../../../../glossary.md#term-cpu) route. Cold support must preserve all Model-valid dtypes of each
implemented reconstruction [signature](../../../../glossary.md#term-signature), not filter to F32/[F64](../../../../glossary.md#floating-point-precision-and-rounding). The initial source capability should
reuse existing exact cold-family mappings and add missing DIV/[scalar](../../../../glossary.md#term-scalar)/SUM/[layout](../../../../glossary.md#layout) mappings needed
for the declared algebra. Any narrower cold inventory/attribute [guard](../../../../glossary.md#term-guard) must be justified by missing
constructor/reconstruction support (as above), explicitly listed in the first [Ready](../../../../glossary.md#term-ready) brief and
reviewed independently of providers; unsupported reconstruction does not invalidate the Model body.
Do not promise all PURE-body copying merely because Model construction accepts it.
Alternative: leave owned GENERAL acquisition rejected and keep direct [Engine](../../../../glossary.md#term-engine) use; smaller change,
but it does not deliver the requested ordinary [owned Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions) workflow. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-local cloning or a new
runner/session/source wrapper would duplicate [Compiler](../../../../glossary.md#term-compiler) reconstruction/[ownership](../../../../glossary.md#term-ownership-meanings) and conflicts with
the project's single ordinary [Tensor](../../../../glossary.md#tensor) [graph](../../../../glossary.md#graph) and execution philosophy; do not select it.

Approved [scope](../../../../glossary.md#term-scope): **a Class C cutover extending Compiler cold source to body reconstruction and
[Checkpoint](../../../../glossary.md#term-checkpoint) copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) based on exact reconstructible Model semantics, independent of CPU/dtype
execution support, then owned Model.graph/defineTensor with initially tested all-valid static-T
[canonical](../../../../glossary.md#term-canonical-form) F32/F64 compute/first-order training, keeping body save/load unsupported and fixed
[runtime](../../../../glossary.md#term-runtime)-length scans [inference](../../../../glossary.md#term-inference)-only.**
Explicit approval is recorded above. Lovelace [binding](../../../../glossary.md#term-binding) pre-code APPROVE and main verification are
complete and retained in Compiler0012, now Complete for that Compiler capability only. Scope is the
declared compute/[AD](../../../../glossary.md#autograd) algebra/closure; cold inventory needs its own constructor-backed review,
not an all-PURE or blanket outer-source [widening](../../../../glossary.md#term-widening).

## Smallest serial upstream frontier and packages

Approved serial order: reviewed authority cutover → **[Compiler](../../../../glossary.md#term-compiler) cold-source capability** → [Checkpoint](../../../../glossary.md#term-checkpoint)
attachment-preserving in-memory copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) with independent wire rejection → [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition/workflow
verification → main's integration join. Compiler0012 is Complete; the next separately briefed gate
is Checkpoint copy/rebind. No future per-[module](../../../../glossary.md#term-module) brief or [Ready](../../../../glossary.md#term-ready) status is created here.

First module: existing `io.github.pho001.synaptik.compiler`, principally `SourceGraphReconstruction`,
`GraphCompilationPort`, `GraphCapture` and `ScanBodyInference`. Use existing `CompiledScanBody` /
[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) to [preflight](../../../../glossary.md#term-preflight) the complete outer [graph](../../../../glossary.md#graph) AND every scoped body before constructing
any [producer](../../../../glossary.md#term-producer); reconstruct fresh distinct [canonical](../../../../glossary.md#term-canonical-form) formals with TensorFactory, ordinary body algebra,
one `ScanBody` and one `Scan.apply` per [occurrence](../../../../glossary.md#term-occurrence). Map every canonical output once, including latent
residuals and aliased roles; retain unused formals, repeated actual positions and independent scopes.
Rebind substitutes outer inputs only; keep produced/repeated substitutions' upstream [provenance](../../../../glossary.md#provenance) [opaque](../../../../glossary.md#term-opaque).
Do not flatten body IDs into outer IDs or let private copies retain external actual leaves/templates.
The same [immutable](../../../../glossary.md#term-immutable) source template may be reused without merging [scan](../../../../glossary.md#term-cumulative-scan) occurrences or leaking formals.
[Cold reconstruction](../../../../glossary.md#term-cold-reconstruction) is call-local and requested explicitly, never per train/compute invocation.

No added/moved production type selected. Existing records contain body graph, formal order and role
[bindings](../../../../glossary.md#term-binding), so no missing representation has been demonstrated. If implementation proves another type
necessary, stop with its exact name/package, missing responsibility, existing-[owner](../../../../glossary.md#term-owner) alternatives and
the duplication/[ownership](../../../../glossary.md#term-ownership-meanings) cost; this decision does not authorize an abstraction or public [API](../../../../glossary.md#term-application-programming-interface-api) addition.

## Non-goals and code disposition

No [padding](../../../../glossary.md#padding), row skipping, dense [masks](../../../../glossary.md#term-boolean-mask), body migration, fixed-[scan](../../../../glossary.md#term-cumulative-scan) [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), `.rnn()`/`.gru()`/`.lstm()`
redirection, new builder recipes, source wire/version, low-type execution, [vector](../../../../glossary.md#term-vector)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas) [widening](../../../../glossary.md#term-widening),
new executor, [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag), [AD](../../../../glossary.md#autograd) layer, numerical policy or performance-parity claim. T=0 source/compute follows
existing scan semantics; it does not relax Training's empty/nonfinite/[alias](../../../../glossary.md#aliasing-and-arena)/storage [admission](../../../../glossary.md#term-admission).

No production code is redundant/deletable as a result of this [planning](../../../../glossary.md#term-planning) task. After an approved
implementation, only blanket GENERAL cold/acquisition exclusions become obsolete within its exact
[scope](../../../../glossary.md#term-scope). Fixed `containsScan` / `rejectScanTraining`, fixed emitters, current cells/Sequences, static
builder recurrence, composition adapters, private copy/alias guards, Training native admission and
v1/v2/v3 readers/[codec](../../../../glossary.md#term-codec) MUST remain. Do not cancel held NN0027 or unrelated Draft gates.

## Coordinated authority and documentation cutover

The root/owning contracts now distinguish the approved [target](../../../../glossary.md#term-target) from current implementation, with
bounded [Compiler](../../../../glossary.md#term-compiler)-only [admission](../../../../glossary.md#term-admission) and unchanged consumer/wire rejection. Their cutover has [binding](../../../../glossary.md#term-binding)
independent pre-code APPROVE/main acceptance; this plan is not authority. Exact headings:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules` (preserve all edges).
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [owner glossary](../../../../glossary.md#term-owner))</span>:
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan), especially `Qualification, performance and migration gates`;
  clarify independent all-valid NN admission versus later padding, without changing scan semantics.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [padding glossary](../../../../glossary.md#padding))</span>
- [Compiler owner](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  `Compile lifecycle` — separate bounded cold body support from [persistence](../../../../glossary.md#term-persistence), retaining full [preflight](../../../../glossary.md#term-preflight).
- [Extension owner](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `Owned tensor Models`, `Generic graph source and values`, `Canonical graph-model version 3`;
  preserve `modules/training`, v1/v2/v3 closed wires and fixed early rejection.

The cutover includes [ADR0046](../../../../design/decisions/0046-general-scan-cold-source-and-owned-model-direction.md), targeted
`docs/architecture/module-boundaries.md` headings `Owned tensor Model composition` and
`modules/checkpoint` ([Gradle](../../../../glossary.md#term-gradle) `:extensions:checkpoint`), `docs/architecture/training-graph.md`
`Owned Model consumer`, and `docs/architecture/general-differentiable-scan.md`
`Source expressions and independent scopes`. Final [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc)/user explanation review covers
`docs/api/graph-checkpoint-api.md`, `docs/user-guide/running-models.md#owned-static-model-workflow`
and glossary [`Owned Model source`](../../../../glossary.md#owned-model-source), [`Generic graph checkpoint source`](../../../../glossary.md#generic-graph-checkpoint-source), `General differentiable scan`.
This task changes authority/[planning](../../../../glossary.md#term-planning)/explanations/glossary only, no Java/Javadoc. General/planning,
architecture and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) profiles apply. Class C implementation still needs independent code/docs review.

## Dependencies and integration

- Depends on: Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0033 and low-level
  root join; NN0028/0029, NN0022B, Compiler0008, Checkpoint0008–0011, Training0007/0008 — Complete.
- Conflicts with: cold source/reconstruction, NN admission/lifecycle, Checkpoint wire and shared
  authority/document edits; serialized under main. Curie is read-only and exclusively owns Gradle.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [admission glossary](../../../../glossary.md#term-admission); [lifecycle glossary](../../../../glossary.md#lifecycle); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Parallel group: None; one repository writer for this [planning](../../../../glossary.md#term-planning) packet.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  dirty baseline (257 paths, empty index); task-local archive records pre-edit content, not HEAD diff.
- Integration order: approved decision → independent authority/inventory review → serial module
  frontier above → independent review → main join; no executor or automatic successor launch.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Integration validation: main's planning link/anchor/metadata/size/stale-fact checks and task-only
  diff review now; later approved capability checkpoint assigned in the first Ready brief.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [metadata glossary](../../../../glossary.md#term-metadata); [Ready glossary](../../../../glossary.md#term-ready))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; one writer, shared authority and explanations included.

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

[Planning](../../../../glossary.md#term-planning) acceptance: one source-backed approved decision, all seven fields, Compiler0012 [Ready](../../../../glossary.md#term-ready) after
[binding](../../../../glossary.md#term-binding) pre-code APPROVE/main verification, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) implementation frontier None, no [API](../../../../glossary.md#term-application-programming-interface-api) invention, and preserved
low-level completion/performance limits. Validate local Markdown links/anchors, targeted stale facts,
brief <=200 lines / <=15 KB and `git diff --check`; archive before/after [SHA-256](../../../../glossary.md#term-sha-256) and exact task-only diff.
No Java/[Gradle](../../../../glossary.md#term-gradle)/native/benchmark execution, source/test edits, [commit](../../../../glossary.md#term-commit) or push in this planning pass.

Compiler0012 owns exact constructor/[scope](../../../../glossary.md#term-scope)/consumer acceptance. Later NN [qualification](../../../../glossary.md#term-qualification) must prove
private [aliases](../../../../glossary.md#aliasing-and-arena), one-shot definition, same owned [weights](../../../../glossary.md#term-weight)/cached [owners](../../../../glossary.md#term-owner), [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) compute/train against
ordinary [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)/direct-session oracles, [lifecycle](../../../../glossary.md#lifecycle)/retry and unchanged mixed fixed-[scan](../../../../glossary.md#term-cumulative-scan) rejection.
Body save still rejects before storage/file effects; old wires stay compatible. Reuse accepted
[CPU](../../../../glossary.md#term-cpu) evidence; main runs one final native root test/[Javadoc](../../../../glossary.md#term-javadoc) at the future NN join, no timing campaign.

## Result

Status: Complete — limited decision/[planning](../../../../glossary.md#term-planning) only. User approval recorded; one compact Compiler0012
prerequisite prepared with constructor-backed inventory and test-only consumer rejection acceptance.
Root/three owning scopes, ADR0046, focused explanations/glossary and frontiers coordinated. [Module](../../../../glossary.md#term-module)
implementation was not launched by this decision; historical pre-code APPROVE/main [Ready](../../../../glossary.md#term-ready) verification
remains recorded. Compiler0012 is now Complete; its Result owns review/gate evidence, not [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) completion.
No new public [API](../../../../glossary.md#term-application-programming-interface-api)/type/[dependency](../../../../glossary.md#term-dependency) or Java edit. Planning link/size/whitespace and baseline-preservation
evidence is archived in the task-local [handoff](../../../../glossary.md#term-handoff); main accepted the [Compiler](../../../../glossary.md#term-compiler) consumer [checkpoint](../../../../glossary.md#term-checkpoint).
Size exception (>15 KB): preserve accepted decision/acceptance and historical Ready facts with current status.

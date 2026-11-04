# Task 0006: Supported training recipe and owned reusable [run](../../../../glossary.md#term-run)

> **Editorial reading context — proposed convenience, not an implemented TrainingRun.**
> A recipe would describe objective and input semantics; an owned run would coordinate model
> transfer, native Arena lifetime and an ordinary TrainingSession. That proposal would hide
> routine setup, not replace Engine execution or optimizer authority. Optimizer/group history
> would still come from TrainingState, while the application's excluded cursor/random state
> would remain outside automatic restoration. Compare today's implemented
> [owned Model training convenience](../../../../reference/training/sessions.md#owned-model-training-convenience)
> and [advanced native-copy setup](../../../../reference/training/parameters.md#explicit-native-parameter-copy).
> Those APIs are related explanations, not evidence that this Draft recipe/run surface exists.

## Status

Draft — necessary owning capability for [Checkpoint](../../../../glossary.md#term-checkpoint) 0006, not a new [persistence](../../../../glossary.md#term-persistence) implementation.
Recipe [loss](../../../../glossary.md#loss)/[signature](../../../../glossary.md#term-signature) [matrix](../../../../glossary.md#term-matrix-meanings) and resource/[capture](../../../../glossary.md#term-capture-meanings) bridge need contract approval before [Ready](../../../../glossary.md#term-ready).

## Change class

Class C implementation — [model](../../../../glossary.md#term-model) [ownership](../../../../glossary.md#term-ownership-meanings) transfer, native [lifetime](../../../../glossary.md#term-lifetime) and reusable [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) orchestration.

## Goal

Provide a Training-owned recipe and [run](../../../../glossary.md#term-run) that build a supported [objective](../../../../glossary.md#term-objective) without a user lambda,
hide routine native [Arena](../../../../glossary.md#term-arena) work and reuse ordinary [TrainingSession](../../../../glossary.md#training-session--trainingsession) with fresh typed [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) values.
Deliver recipe/run semantics, tests, [Javadoc](../../../../glossary.md#term-javadoc) and documentation together; no file I/O.

## [Scope](../../../../glossary.md#term-scope)

Proposed `TrainingRecipe` and `TrainingRun implements AutoCloseable` in
`io.github.pho001.synaptik.training`, plus narrowly scoped nested [immutable](../../../../glossary.md#term-immutable) recipe values and
`TrainingRun.CheckpointCapture` if needed for the concrete downstream [capture](../../../../glossary.md#term-capture-meanings) boundary.
No generic manager/[facade](../../../../glossary.md#term-facade)/[transaction](../../../../glossary.md#term-transaction)/participant framework, new runner or private [Engine](../../../../glossary.md#term-engine) access.

Recipe records supported [loss](../../../../glossary.md#loss) ID/options, named input/[target](../../../../glossary.md#term-target) static [dtype](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape) and dense offset-zero
non-[view](../../../../glossary.md#term-view-meanings) [binding](../../../../glossary.md#term-binding) contract, [module](../../../../glossary.md#term-module) modes/effective contexts and GradientMode.
Propose initial [mean](../../../../glossary.md#term-mean)-square [scalar](../../../../glossary.md#term-scalar) loss with SUM/MEAN reduction and [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [objective](../../../../glossary.md#term-objective);
low [predictions](../../../../glossary.md#term-prediction) explicitly cast to FLOAT32 when required. [Embedding](../../../../glossary.md#embedding) inputs may be integral;
targets/prediction types follow loss semantics, not a blanket [homogeneous](../../../../glossary.md#term-homogeneous)-input rule.
[Optimizer](../../../../glossary.md#term-optimizer)/groups come separately at new [run](../../../../glossary.md#term-run) creation and from saved TrainingState at recovery:
TrainingState is their sole [persistence](../../../../glossary.md#term-persistence) authority. Freeze additional losses/options before [Ready](../../../../glossary.md#term-ready).

Conceptual [owner](../../../../glossary.md#term-owner) [API](../../../../glossary.md#term-application-programming-interface-api), not existing:

```text
TrainingRun.openOwned(DeclarativeModel, TrainingRecipe, Engine, Optimizer, List<ParameterGroup>): TrainingRun
TrainingRun.trainBatch(Tensor input, Tensor target): TrainingStep
TrainingRun.checkpointCapture(): CheckpointCapture
TrainingRun.close(): void
```

The factory takes exclusive [ownership](../../../../glossary.md#term-ownership-meanings) of a private [model](../../../../glossary.md#term-model) at the documented transfer point,
copies [parameters](../../../../glossary.md#term-parameter) with [TrainingParameterStorage](../../../../glossary.md#native-parameter-copy--trainingparameterstorage).copyToSharedNative into its own shared [Arena](../../../../glossary.md#term-arena),
applies recipe contexts, builds the objective from final [Tensors](../../../../glossary.md#tensor), then opens ordinary [TrainingSession](../../../../glossary.md#training-session--trainingsession).
Restoration calls ordinary state/restore through the unpublished owned run under exclusive [admission](../../../../glossary.md#term-admission);
freeze the narrow restore bridge before Ready. It must not expose a general [mutable](../../../../glossary.md#term-mutable) session.

## Non-goals

No [Checkpoint](../../../../glossary.md#term-checkpoint) [dependency](../../../../glossary.md#term-dependency), byte codecs, persistent versions/file I/O, automatic [backend](../../../../glossary.md#backend) selection,
new [compiler](../../../../glossary.md#term-compiler)/preparer/runner, hidden [inference](../../../../glossary.md#term-inference) [autograd](../../../../glossary.md#autograd) or per-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) recompilation.
No cursor/[RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/scheduler/scaler restoration or advancing [dropout](../../../../glossary.md#dropout--inverted-dropout) stream.
No shared [parameter](../../../../glossary.md#term-parameter)-path update workaround, BatchNorm training-[buffer](../../../../glossary.md#term-buffer) [alignment](../../../../glossary.md#term-alignment), [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) or [Conv3d](../../../../glossary.md#conv3d--ncdhw)
[gradient](../../../../glossary.md#term-gradient) support. Existing explicit [TrainingSession](../../../../glossary.md#training-session--trainingsession) and native-copy APIs remain unchanged.

## Contracts

- [Root authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning scoped headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [extensions/training](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>,
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
- [Proposed ADR 0038](../../../../design/decisions/0038-declarative-model-and-training-resume.md).

The current contracts do not authorize the new product. Before [Ready](../../../../glossary.md#term-ready), coordinate root + owning
extensions contract + affected architecture explanations + [ADR](../../../../glossary.md#term-architecture-decision-record-adr) + architecture tests as applicable.
Stop on missing/ambiguous contracts; this proposal does not override them.

## Dependencies and integration

- Depends on: NN0026 Complete; CP0005B; NN0027 semantic recurrence; CP0005C versioned persistence;
  completed Training 0004 and Checkpoint 0005 defensive reconstruction;
  approved TrainingRecipe/TrainingRun ownership and support matrix under ADR 0038.<span data-glossary-navigation="true"> ([support matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [matrix glossary](../../../../glossary.md#term-review-matrix); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Conflicts with: [Training session](../../../../glossary.md#training-session--trainingsession)/[lifecycle](../../../../glossary.md#lifecycle) contracts and shared [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Checkpoint](../../../../glossary.md#term-checkpoint) documents.
- Parallel group: None; separate clean context with serialized shared contract edits.
- Common base revision: [planning](../../../../glossary.md#term-planning) base `52733fcfd32d278b651302e308487e0b6c6afa81`; reverify before [Ready](../../../../glossary.md#term-ready).
- Integration order: NN0026 → CP0005B → NN0027 → CP0005C → Training0006 → optional CP0006.
- Integration [validation](../../../../glossary.md#term-validation): one source-matched combined repository test/[Javadoc](../../../../glossary.md#term-javadoc) checkpoint.
- Shared-document integration [owner](../../../../glossary.md#term-owner): NN/Training/Checkpoint coordinator.
- [Authorized frontier](../../../../glossary.md#term-authorized-frontier): None while Draft; exact dependencies/contracts reverified before launch.

## Files and symbols

- `extensions/training/.../training/{TrainingSession,TrainingParameterStorage,TrainingState}.java`.
- New TrainingRecipe/TrainingRun and focused external-[API](../../../../glossary.md#term-application-programming-interface-api), [binding](../../../../glossary.md#term-binding) and [lifecycle](../../../../glossary.md#lifecycle) tests.
- Public [Tensor](../../../../glossary.md#tensor) `replaceHostStorage`, `clearHostStorage`, `hostStorage` and compiled input [metadata](../../../../glossary.md#term-metadata).
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected architecture explanations.

## Acceptance criteria

- Recipe owner validates complete loss/options, named roles/static signatures/contexts and scalar
  objective capability before returning a run. Unsupported model/loss/gradient route fails closed.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [loss glossary](../../../../glossary.md#loss); [signatures glossary](../../../../glossary.md#term-signature); [scalar objective glossary](../../../../glossary.md#loss); [run glossary](../../../../glossary.md#term-run); [model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Stable input/target prototype Tensors are built once; compiled metadata and Tensor identity resolve
  roles once. Batch values' fresh opaque IDs are NOT passed directly to TrainingSession.run.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [prototype glossary](../../../../glossary.md#term-prototype); [Tensors glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings); [Batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [opaque glossary](../../../../glossary.md#term-opaque); [TrainingSession glossary](../../../../glossary.md#training-session--trainingsession))</span>
- trainBatch validates actual dtype/Shape/layout/offset/view and host capacity/liveness/accessibility;
  dense offset-zero v1 refuses strided/views rather than interpreting their bytes as dense.
  Pure input roles are separate from caller requiresGrad flags and parameter gradient targets.
  Borrow actual host storage with prototype.replaceHostStorage, run using fixed prototype IDs,
  clearHostStorage in finally. No default batch copy/materialization or hot-path name lookup.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [host glossary](../../../../glossary.md#term-host); [liveness glossary](../../../../glossary.md#term-liveness); [parameter glossary](../../../../glossary.md#term-parameter); [host storage glossary](../../../../glossary.md#host-storage); [materialization glossary](../../../../glossary.md#materialization); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Borrow user input storage only through the synchronous step; never close user owners. Preserve
  fixed parameter/Tensor/storage identities. Reuse one prepared session; shape changes reject.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Run admission excludes trainBatch, restore, coherent capture and close. Restore delegates
  TrainingSession.restore under private ownership; no duplicate optimizer/session validator.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [capture glossary](../../../../glossary.md#term-capture-meanings); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- CheckpointCapture is a narrowly scoped exclusive borrow held across session.state and downstream
  model capture, supplying detached Training state + recipe and a borrowed quiescent model/Engine.
  It is not a trusted-state certificate or generic transaction. Closing releases exclusion.
  No native-backed model is returned for unrestricted use; advanced capture borrows end with lease.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [quiescent glossary](../../../../glossary.md#term-quiescent); [Engine glossary](../../../../glossary.md#term-engine); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [transaction glossary](../../../../glossary.md#term-transaction); [releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [lease glossary](../../../../glossary.md#term-lease))</span>
- Repeated parameter Tensor identities at multiple paths and overlapping parameter regions fail.
  Reusing a uniquely owned layer at multiple execution sites is not itself that alias condition.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Run owns its private model, native parameter Arena and placeholder resources, and session;
  Engine is borrowed. Close session before all arenas; attempt all cleanup, preserve failures/
  suppressed cleanup failures and idempotent close outcome. Failure at every construction/restore
  stage closes acquired resources and returns no owner. Cleanup never closes the Engine or batch.<span data-glossary-navigation="true"> ([Arena glossary](../../../../glossary.md#term-arena); [idempotent glossary](../../../../glossary.md#term-idempotent))</span>
- Model access is absent or explicitly run-lifetime-bound; only independently detached inference
  models/snapshots may outlive close. No stale references masquerading as usable models.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [inference glossary](../../../../glossary.md#term-inference))</span>
- Existing session atomic update/counter failure semantics remain the authority; failed batch binding
  mutates neither optimizer/counters nor parameter state. Test partial binding and cleanup failures.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding))</span>

## [Validation](../../../../glossary.md#term-validation)

Future worker: focused recipe/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[capture](../../../../glossary.md#term-capture-meanings)/failure/close tests, then
`./gradlew :extensions:training:test :extensions:training:javadoc` once after final changes.
Coordinator runs one source-matched `./gradlew test javadoc --max-workers=2 --console=plain`
after full program integration, with [dependency](../../../../glossary.md#term-dependency) and durable end-to-end tests. Review reuses evidence.
[Planning](../../../../glossary.md#term-planning) now: links/anchors, ≤200 lines/15KB and `git diff --check`; no Java/build tests.

## Documentation and review impact

Clean implementation + independent targeted Class C review/docs. Finalize transfer point, [admission](../../../../glossary.md#term-admission),
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [lifetime](../../../../glossary.md#term-lifetime), exact recipe [matrix](../../../../glossary.md#term-matrix-meanings), stable-placeholder mapping and close order in [Javadoc](../../../../glossary.md#term-javadoc).
Targeted glossary and Training guides; coordinator owns shared contract/explanation updates before [Ready](../../../../glossary.md#term-ready).

## Result

Empty until implementation; required summary includes changes/files/tests/docs/limitations/follow-up
and `Status: Complete` or `Status: Incomplete` with specific follow-up.

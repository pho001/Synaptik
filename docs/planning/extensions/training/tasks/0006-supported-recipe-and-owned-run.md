# Task 0006: Supported training recipe and owned reusable run

## Status

Draft — necessary owning capability for Checkpoint 0006, not a new persistence implementation.
Recipe loss/signature matrix and resource/capture bridge need contract approval before Ready.

## Change class

Class C implementation — model ownership transfer, native lifetime and reusable batch orchestration.

## Goal

Provide a Training-owned recipe and run that build a supported objective without a user lambda,
hide routine native Arena work and reuse ordinary TrainingSession with fresh typed batch values.
Deliver recipe/run semantics, tests, Javadoc and documentation together; no file I/O.

## Scope

Proposed `TrainingRecipe` and `TrainingRun implements AutoCloseable` in
`io.github.pho001.synaptik.training`, plus narrowly scoped nested immutable recipe values and
`TrainingRun.CheckpointCapture` if needed for the concrete downstream capture boundary.
No generic manager/facade/transaction/participant framework, new runner or private Engine access.

Recipe records supported loss ID/options, named input/target static dtype/Shape and dense offset-zero
non-view binding contract, module modes/effective contexts and GradientMode.
Propose initial mean-square scalar loss with SUM/MEAN reduction and FLOAT32/FLOAT64 objective;
low predictions explicitly cast to FLOAT32 when required. Embedding inputs may be integral;
targets/prediction types follow loss semantics, not a blanket homogeneous-input rule.
Optimizer/groups come separately at new run creation and from saved TrainingState at recovery:
TrainingState is their sole persistence authority. Freeze additional losses/options before Ready.

Conceptual owner API, not existing:

```text
TrainingRun.openOwned(DeclarativeModel, TrainingRecipe, Engine, Optimizer, List<ParameterGroup>): TrainingRun
TrainingRun.trainBatch(Tensor input, Tensor target): TrainingStep
TrainingRun.checkpointCapture(): CheckpointCapture
TrainingRun.close(): void
```

The factory takes exclusive ownership of a private model at the documented transfer point,
copies parameters with TrainingParameterStorage.copyToSharedNative into its own shared Arena,
applies recipe contexts, builds the objective from final Tensors, then opens ordinary TrainingSession.
Restoration calls ordinary state/restore through the unpublished owned run under exclusive admission;
freeze the narrow restore bridge before Ready. It must not expose a general mutable session.

## Non-goals

No Checkpoint dependency, byte codecs, persistent versions/file I/O, automatic backend selection,
new compiler/preparer/runner, hidden inference autograd or per-batch recompilation.
No cursor/RNG/scheduler/scaler restoration or advancing dropout stream.
No shared parameter-path update workaround, BatchNorm training-buffer alignment, BPTT or Conv3d
gradient support. Existing explicit TrainingSession and native-copy APIs remain unchanged.

## Contracts

- [Root authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning scoped headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [extensions/training](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- [Proposed ADR 0038](../../../../design/decisions/0038-declarative-model-and-training-resume.md).

The current contracts do not authorize the new product. Before Ready, coordinate root + owning
extensions contract + affected architecture explanations + ADR + architecture tests as applicable.
Stop on missing/ambiguous contracts; this proposal does not override them.

## Dependencies and integration

- Depends on: NN 0026; completed Training 0004 and Checkpoint 0005 defensive reconstruction;
  approved TrainingRecipe/TrainingRun ownership and support matrix under ADR 0038.
- Conflicts with: Training session/lifecycle contracts and shared NN/Training/Checkpoint documents.
- Parallel group: None; separate clean context with serialized shared contract edits.
- Common base revision: planning base `52733fcfd32d278b651302e308487e0b6c6afa81`; reverify before Ready.
- Integration order: NN 0026 first; Training 0006 and Checkpoint 0005B join before Checkpoint 0006.
- Integration validation: one source-matched combined repository test/Javadoc checkpoint.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Authorized frontier: None while Draft; exact dependencies/contracts reverified before launch.

## Files and symbols

- `extensions/training/.../training/{TrainingSession,TrainingParameterStorage,TrainingState}.java`.
- New TrainingRecipe/TrainingRun and focused external-API, binding and lifecycle tests.
- Public Tensor `replaceHostStorage`, `clearHostStorage`, `hostStorage` and compiled input metadata.
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected architecture explanations.

## Acceptance criteria

- Recipe owner validates complete loss/options, named roles/static signatures/contexts and scalar
  objective capability before returning a run. Unsupported model/loss/gradient route fails closed.
- Stable input/target prototype Tensors are built once; compiled metadata and Tensor identity resolve
  roles once. Batch values' fresh opaque IDs are NOT passed directly to TrainingSession.run.
- trainBatch validates actual dtype/Shape/layout/offset/view and host capacity/liveness/accessibility;
  dense offset-zero v1 refuses strided/views rather than interpreting their bytes as dense.
  Pure input roles are separate from caller requiresGrad flags and parameter gradient targets.
  Borrow actual host storage with prototype.replaceHostStorage, run using fixed prototype IDs,
  clearHostStorage in finally. No default batch copy/materialization or hot-path name lookup.
- Borrow user input storage only through the synchronous step; never close user owners. Preserve
  fixed parameter/Tensor/storage identities. Reuse one prepared session; shape changes reject.
- Run admission excludes trainBatch, restore, coherent capture and close. Restore delegates
  TrainingSession.restore under private ownership; no duplicate optimizer/session validator.
- CheckpointCapture is a narrowly scoped exclusive borrow held across session.state and downstream
  model capture, supplying detached Training state + recipe and a borrowed quiescent model/Engine.
  It is not a trusted-state certificate or generic transaction. Closing releases exclusion.
  No native-backed model is returned for unrestricted use; advanced capture borrows end with lease.
- Repeated parameter Tensor identities at multiple paths and overlapping parameter regions fail.
  Reusing a uniquely owned layer at multiple execution sites is not itself that alias condition.
- Run owns its private model, native parameter Arena and placeholder resources, and session;
  Engine is borrowed. Close session before all arenas; attempt all cleanup, preserve failures/
  suppressed cleanup failures and idempotent close outcome. Failure at every construction/restore
  stage closes acquired resources and returns no owner. Cleanup never closes the Engine or batch.
- Model access is absent or explicitly run-lifetime-bound; only independently detached inference
  models/snapshots may outlive close. No stale references masquerading as usable models.
- Existing session atomic update/counter failure semantics remain the authority; failed batch binding
  mutates neither optimizer/counters nor parameter state. Test partial binding and cleanup failures.

## Validation

Future worker: focused recipe/batch/capture/failure/close tests, then
`./gradlew :extensions:training:test :extensions:training:javadoc` once after final changes.
Coordinator runs one source-matched `./gradlew test javadoc --max-workers=2 --console=plain`
after full program integration, with dependency and durable end-to-end tests. Review reuses evidence.
Planning now: links/anchors, ≤200 lines/15KB and `git diff --check`; no Java/build tests.

## Documentation and review impact

Clean implementation + independent targeted Class C review/docs. Finalize transfer point, admission,
borrowed batch lifetime, exact recipe matrix, stable-placeholder mapping and close order in Javadoc.
Targeted glossary and Training guides; coordinator owns shared contract/explanation updates before Ready.

## Result

Empty until implementation; required summary includes changes/files/tests/docs/limitations/follow-up
and `Status: Complete` or `Status: Incomplete` with specific follow-up.

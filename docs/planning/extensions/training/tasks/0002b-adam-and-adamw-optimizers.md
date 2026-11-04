# Task 0002B: [Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw) and [AdamW](../../../../glossary.md#adaptive-optimizer--adam-adamw) [optimizer state](../../../../glossary.md#term-optimizer-state)

> **Editorial reading context — not a status update.** Adam/AdamW are immutable algorithm
> configurations; TrainingSession owns per-parameter moment history, staging and commit.
> A first moment tracks gradients and a second moment tracks squared gradients. Adam couples
> decay into the gradient; AdamW applies it separately to the parameter. These slots and
> the global step are not restored by a model-only file. Today's
> [optimizer formulas/modes](../../../../reference/training/optimizers.md#optimizers-and-gradient-modes)
> and [state ownership](../../../../reference/training/optimizers.md#atomicity-state-concurrency-and-close)
> explain the boundary.
>
> **Mathematical reading example.** Let a finite real gradient be $g=2$, initial moments
> $m_0=v_0=0$, and decay coefficients $\beta_1=\beta_2=1/2$. At the first step,
> $m_1=1$ and $v_1=2$; bias correction gives $\widehat m_1=2$ and $\widehat v_1=4$.
> Thus the adaptive subtraction has magnitude $2\eta/(2+\epsilon)$ for real learning rate
> $\eta>0$ and stabilizer $\epsilon>0$, before any separate decay. This illustrates history
> and correction, not permission to reorder the recorded rounded arithmetic.

## Status

Complete

## Change class

Class B — extends the public [optimizer](../../../../glossary.md#term-optimizer) configurations, [module](../../../../glossary.md#term-module)-local update behavior, and exact
in-memory state/restore contract. It does not change [Engine](../../../../glossary.md#term-engine)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools),
[backend capability](../../../../glossary.md#backend-capability), or the [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution) path. The owning scoped [lifecycle](../../../../glossary.md#lifecycle) contract must
describe the added algorithms and payloads in the same change.

## Goal

Support [Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw) and [AdamW](../../../../glossary.md#adaptive-optimizer--adam-adamw) as default or per-group [optimizers](../../../../glossary.md#term-optimizer), including mixed [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd)/Adam/AdamW groups,
without another execution pipeline, per-[element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch), or weaker [failure atomicity](../../../../glossary.md#term-failure-atomicity).

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public [`Adam`](../../../../glossary.md#adaptive-optimizer--adam-adamw) and [`AdamW`](../../../../glossary.md#adaptive-optimizer--adam-adamw) configurations, each constructed as
  `(double learningRate, double beta1, double beta2, double epsilon, double weightDecay)`.
  Require all five values finite, learning rate and epsilon positive,<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [learning rate glossary](../../../../glossary.md#term-learning-rate); [epsilon glossary](../../../../glossary.md#epsilon))</span>
  `beta1` and `beta2` in `[0, 1)`, and weight decay non-negative. Keep exact configuration
  equality, including algorithm kind, for state compatibility; extend sealed <span data-glossary-navigation="true"> ([weight decay glossary](../../../../glossary.md#term-weight-decay); [sealed glossary](../../../../glossary.md#term-sealed))</span>[`Optimizer`](../../../../glossary.md#term-optimizer).
- Retain one global optimizer-step counter. On a successful update with one-based step `t`,
  compute `m = beta1 * oldM + (1 - beta1) * g` and
  `v = beta2 * oldV + (1 - beta2) * g * g`; use [bias](../../../../glossary.md#term-bias)-corrected
  `m / (1 - beta1^t)` and `v / (1 - beta2^t)`, then subtract
  `learningRate * correctedM / (sqrt(correctedV) + epsilon)`.
  Adam first adds coupled `weightDecay * masterParameter` to `g`; AdamW leaves `g` unchanged
  and applies decoupled `masterParameter * (1 - learningRate * weightDecay)` before the adaptive
  subtraction. `ACCUMULATE_AND_STEP` uses the existing pending [gradient](../../../../glossary.md#term-gradient) **sum**, not its [mean](../../../../glossary.md#term-mean).
- Preserve the current FLOAT64 arithmetic domain and FLOAT32 working/master domain for FLOAT32,
  BFLOAT16, and FLOAT16 parameters. Narrow configuration once per assigned dtype; reject a
  learning rate or epsilon that becomes zero, a beta that becomes one, or any coefficient that
  becomes non-finite or leaves its permitted range. Other valid narrowed coefficients retain
  the existing SGD-style narrowing policy. Compute and validate bias corrections outside the
  element loop. Check every intermediate, both moments, master candidate, and final
  low-precision narrowing for finiteness.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [dtype glossary](../../../../glossary.md#data-type--datatype); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [narrowing glossary](../../../../glossary.md#term-narrowing); [element glossary](../../../../glossary.md#term-element); [precision glossary](../../../../glossary.md#term-precision))</span>
- Resolve one immutable algorithm/configuration per [parameter](../../../../glossary.md#term-parameter) at `open` from the default and
  ordered groups. All parameters still update on every successful global step. Preserve SGD
  arithmetic and existing group/path identity. Allow mixed algorithms in one session without
  backend identity checks or an algorithm/map lookup per element.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [backend identity glossary](../../../../glossary.md#backend-identity--backendid); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Retain the existing first-moment/momentum arrays. Allocate an active second-moment array only
  for Adam/AdamW parameters and its staging array only before an update or restore that needs it.
  Stage all optimizer slots and parameter candidates before any parameter write; commit slots and
  counters only after every candidate and complete binding view pass validation.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [commit glossary](../../../../glossary.md#term-commit); [binding glossary](../../../../glossary.md#term-binding); [view glossary](../../../../glossary.md#term-view-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
- Generalize `TrainingState.optimizer()` to return the exact default `Optimizer`, including
  Adam and AdamW; do not add a second default-optimizer accessor. This intentionally changes
  the method's source and binary signature from <span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [signature glossary](../../../../glossary.md#term-signature))</span>`Sgd` to `Optimizer`; update all project
  call sites, tests, Javadocs, and examples together. Retain exact group configurations. Preserve
  `ParameterState.momentumBytes()` as the existing first-slot payload (Adam's first moment);
  add a neutral <span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`firstMomentBytes(): ByteBuffer` [alias](../../../../glossary.md#aliasing-and-arena) and
  `secondMomentBytes(): Optional<ByteBuffer>` for Adam/AdamW. Returned views stay fresh and
  read-only.
  No second-moment payload exists for SGD. Snapshot remains detached, canonical, and immutable.<span data-glossary-navigation="true"> ([Snapshot glossary](../../../../glossary.md#term-snapshot); [detached glossary](../../../../glossary.md#detached-model-results); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `restore()` first validates default/group algorithm and configuration, ordered paths and
  schemas, counters, exact slot presence/width, finite values, and non-negative Adam second
  moments. Only then stage/install all parameter and slot bytes. Preserve old SGD snapshots and
  the existing <span data-glossary-navigation="true"> ([schemas glossary](../../../../glossary.md#term-schema))</span>`TrainingSession.open(...)` overloads unchanged.
- Update the owning scoped optimizer/training lifecycle contract, Training API/user guide,
  architecture explanation, affected Javadocs, and targeted glossary terms. Include one
  mixed-group example and explicit accessor compatibility notes.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

- AMSGrad, sparse gradients, maximize mode, per-parameter step counters, per-group gradient modes,
  learning-rate schedules, dynamic regrouping, optimizer-state persistence schema, checkpoint
  files, backend optimizer kernels, cross-backend numerical identity, or a rollback-policy switch.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [learning-rate glossary](../../../../glossary.md#gradient-and-learning-rate); [optimizer glossary](../../../../glossary.md#term-optimizer); [persistence schema glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [identity glossary](../../../../glossary.md#term-identity-meanings); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- New Tensor operations, new gradient rules, expanded Metal low-precision graph capability, or
  an optimizer update inside the compiled graph.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Metal glossary](../../../../glossary.md#term-metal); [precision glossary](../../../../glossary.md#term-precision); [graph glossary](../../../../glossary.md#graph); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>, and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> keep optimizer policy in
  Training and preserve the NN/Engine edges.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns deterministic recursive parameter paths, not optimizer algorithms.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  permits Adam/AdamW and backend-neutral state; no backend bridge is introduced.<span data-glossary-navigation="true"> ([Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw))</span>
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  governs one global lifecycle, finite state, exact restore, and ordinary-failure atomicity.
  Its SGD-specific text must be extended for the additional moments and algorithms without
  weakening existing SGD guarantees.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd))</span>

If these contracts require a changed [ownership](../../../../glossary.md#term-ownership-meanings) or [Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend) boundary, stop for a separate
architecture decision rather than silently editing the contract.

## Dependencies and integration

- Depends on: Training 0002A (Complete, validated, reviewed, and pushed at `19b7785c`).
- Conflicts with: concurrent Training public [optimizer](../../../../glossary.md#term-optimizer) [API](../../../../glossary.md#term-application-programming-interface-api), [parameter](../../../../glossary.md#term-parameter)-group assignment,
  `TrainingState`/restore, optimizer-slot, or training-lifecycle work; Training 0003 until this
  state contract is integrated.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None.
- Common base revision: `19b7785c` on `main`.
- Integration order: implementation and focused validation; independent public API/state and
  documentation review; then integration-owner repository checkpoint.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): configured source-matched [Metal](../../../../glossary.md#term-metal) test library with
  `./gradlew test javadoc`, plus `git diff --check`, once after integration because Training,
  integration tests, and the owning scoped contract change.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

Launch verification (2026-10-13): `main` and `origin/main` are at `19b7785c` with only this
[planning](../../../../glossary.md#term-planning) update pending. Training 0002A has its complete result, validation, documentation, and
independent approval; 0002B is its sole next Training frontier, and no conflicting writer is
active. Implementation starts only after this plan is committed and pushed.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/Adam.java` and
  `AdamW.java` — new [immutable](../../../../glossary.md#term-immutable) public configurations; `Optimizer.java` [sealed](../../../../glossary.md#term-sealed) set.
- `ParameterGroup.java`, `TrainingSession.java`, `TrainingState.java`, and
  `TrainingParameter.java` in the same package — mixed assignment, update, slots, snapshots,
  and restore. Preserve existing `Sgd.java` semantics.
- Focused Training tests under `extensions/training/src/test/java/`; public [CPU](../../../../glossary.md#term-cpu) scenario in
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`.
- `docs/architecture/contracts/extensions-training.md`, `docs/architecture/training-graph.md`,
  `docs/api/training-api.md`, `docs/user-guide/training.md`, and targeted `docs/glossary.md`.

## Acceptance criteria

- Existing SGD-only construction, update bytes, snapshot payloads, and restore behavior remain
  compatible, except for the intentional <span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>`TrainingState.optimizer()` return-type change. Adam and
  AdamW each have an independent multi-step scalar oracle with beta bias
  correction, coupled versus decoupled decay, all three gradient modes, and <span data-glossary-navigation="true"> ([Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [bias correction glossary](../../../../glossary.md#term-bias-correction); [gradient glossary](../../../../glossary.md#term-gradient))</span>`zeroGrad` behavior.
- Mixed default/group configurations (including Adam default with an SGD group) share one global
  step and one atomic commit. Group order, path order, algorithm kind, and every coefficient are
  exact restore identity; no per-element algorithm selection or transcendental calculation occurs.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [commit glossary](../../../../glossary.md#term-commit); [identity glossary](../../../../glossary.md#term-identity-meanings); [element glossary](../../../../glossary.md#term-element))</span>
- FLOAT64, FLOAT32, BFLOAT16, and FLOAT16 parameter cases cover coefficient narrowing, one final
  low-value conversion, moment widths, and non-finite/overflow rejection. SGD-only parameters
  acquire no second-moment allocation or snapshot payload.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [parameter glossary](../../../../glossary.md#term-parameter); [narrowing glossary](../../../../glossary.md#term-narrowing); [finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Detached Adam states round-trip raw represented bytes, including signed zeros where valid.
  Missing, extra, wrong-width, negative, or non-finite second moments fail before any parameter,
  slot, accumulation, or counter mutation. A later-parameter injected failure leaves all mixed
  groups unchanged.<span data-glossary-navigation="true"> ([Detached glossary](../../../../glossary.md#detached-model-results); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- Public CPU integration exercises mixed algorithms and restore across a successful next step;
  no Engine, NN, backend, native ABI, or checkpoint-codec change is needed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [backend glossary](../../../../glossary.md#backend); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [codec glossary](../../../../glossary.md#term-codec))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker uses a clean isolated worktree, runs focused new/existing Training tests and the targeted
[CPU](../../../../glossary.md#term-cpu) [integration test](../../../../glossary.md#term-integration-test), then one
final affected-[module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) [run](../../../../glossary.md#term-run) after executable code stabilizes:

```bash
./gradlew :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test
git diff --check
```

Integration [owner](../../../../glossary.md#term-owner) reuses worker evidence, obtains independent targeted Class B review, and runs
the recorded repository [checkpoint](../../../../glossary.md#term-checkpoint) once. Configure `SYNAPTIK_METAL_TEST_LIBRARY` to a
source-matched local library for native tests; do not substitute skipped [device](../../../../glossary.md#term-device) tests for a pass.

## Documentation and review impact

Independent review finalizes the public/[Javadoc](../../../../glossary.md#term-javadoc) contract, two-algorithm mathematics and mixed
example, [snapshot](../../../../glossary.md#term-snapshot) [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) compatibility and intentional accessor [signature](../../../../glossary.md#term-signature) change, glossary
terms, and scoped [lifecycle](../../../../glossary.md#lifecycle) wording. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr)
or root [dependency](../../../../glossary.md#term-dependency) change is expected because [Adam](../../../../glossary.md#adaptive-optimizer--adam-adamw)/[AdamW](../../../../glossary.md#adaptive-optimizer--adam-adamw) and Training [ownership](../../../../glossary.md#term-ownership-meanings) are already
allowed; stop if implementation discovers an architectural conflict.

## Result

- Implemented at `10d507a3`, with independent documentation/[Javadoc](../../../../glossary.md#term-javadoc) review fix `23d1e39d`.
  [`Adam`](../../../../glossary.md#adaptive-optimizer--adam-adamw), [`AdamW`](../../../../glossary.md#adaptive-optimizer--adam-adamw), and dtype-narrowed coefficients support mixed path groups, one global
  bias-corrected step, finite staged updates, exact optional second-moment snapshots, and
  validate-before-install restore. <span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [bias glossary](../../../../glossary.md#term-bias); [finite glossary](../../../../glossary.md#term-finite))</span>`TrainingState.optimizer()` now returns [`Optimizer`](../../../../glossary.md#term-optimizer).
- Changed Training optimizer/session/state sources and tests, CPU integration coverage, the
  owning scoped contract, architecture explanation and module-boundary overview, API and user
  guides, glossary, and affected Javadocs. SGD update behavior and first-slot payload remain.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Worker: focused `AdamOptimizerTest` and mixed CPU integration passed; source-matched Metal
  library built and signature-checked; configured<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [signature glossary](../../../../glossary.md#term-signature))</span>
  `./gradlew :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test`
  passed (64 Training tests and 9 Training CPU/Metal integration tests, zero skips in those suites).
- Independent Class B review: `APPROVE`, no blocking findings. Reviewer validated Javadoc and
  380 local Markdown links/anchors; no executable code changed in review.
- Integration [owner](../../../../glossary.md#term-owner): configured `./gradlew test javadoc` passed (`100` actionable tasks: 4 executed,
  4 from cache, 92 up-to-date); <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`git diff --check f07ab9b5..HEAD` passed. One unrelated
  autotuning integration fixture skipped because the host lacked eligible alternatives.<span data-glossary-navigation="true"> ([autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [fixture glossary](../../../../glossary.md#term-fixture); [host glossary](../../../../glossary.md#term-host))</span>
- No unresolved implementation issue. Later exact-[resume](../../../../glossary.md#schema-serialization-and-checkpoint) and [checkpoint](../../../../glossary.md#term-checkpoint) work stays Draft.

Status: Complete

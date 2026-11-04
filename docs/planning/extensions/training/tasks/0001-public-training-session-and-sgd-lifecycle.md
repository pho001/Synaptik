# Task 0001: Public [Training Session](../../../../glossary.md#training-session--trainingsession) and [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) [Lifecycle](../../../../glossary.md#lifecycle)

## Status

Complete

## Change class

Class C — adds a public cross-[module](../../../../glossary.md#term-module) session, an intentional Training-to-[Engine](../../../../glossary.md#term-engine) [dependency](../../../../glossary.md#term-dependency),
[optimizer](../../../../glossary.md#term-optimizer) mutation and [snapshot](../../../../glossary.md#term-snapshot) state, [failure atomicity](../../../../glossary.md#term-failure-atomicity), and close/in-flight [concurrency](../../../../glossary.md#term-concurrency).

## Goal

Deliver one complete public [CPU](../../../../glossary.md#term-cpu)-capable training [lifecycle](../../../../glossary.md#lifecycle) that compiles and prepares a [scalar](../../../../glossary.md#term-scalar)
forward/[backward graph](../../../../glossary.md#backward-graph) once, maps deterministic [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [parameter](../../../../glossary.md#term-parameter) paths to [Compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient)
occurrences, applies persistent [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) state over repeated runs, and exposes exact lifecycle and
in-memory restore behavior without another execution pipeline.

## [Scope](../../../../glossary.md#term-scope)

- Public [`Optimizer`](../../../../glossary.md#term-optimizer), `Sgd`, `GradientMode`, [`TrainingSession`](../../../../glossary.md#training-session--trainingsession), `TrainingStep`, and `TrainingState`.
- One borrowed Engine and Module, one owned reusable inference session, stable recursive parameter
  path/wrapper/Tensor/storage capture, and external input metadata/binding.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Engine glossary](../../../../glossary.md#term-engine); [Module glossary](../../../../glossary.md#term-module); [inference glossary](../../../../glossary.md#term-inference); [parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [capture glossary](../../../../glossary.md#term-capture-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding))</span>
- FLOAT32/FLOAT64 dense writable shareable-native parameters, SGD momentum/dampening/coupled weight
  decay/compatible Nesterov, reset/accumulate semantics, persistent counters, immutable state
  snapshot, strict restore, non-finite/ordinary-failure atomicity, and deterministic
  close/in-flight behavior.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [momentum glossary](../../../../glossary.md#term-momentum); [dampening glossary](../../../../glossary.md#term-dampening); [weight decay glossary](../../../../glossary.md#term-weight-decay); [Nesterov glossary](../../../../glossary.md#term-nesterov); [immutable glossary](../../../../glossary.md#term-immutable); [snapshot glossary](../../../../glossary.md#term-snapshot); [finite glossary](../../../../glossary.md#term-finite); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
- Public documentation, Javadocs, architecture decision, architecture dependency enforcement,
  numerical/state/lifecycle tests, and an end-to-end convergence smoke.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [smoke glossary](../../../../glossary.md#term-smoke-test))</span>

## Non-goals

- Parameter groups, Adam/AdamW, schedulers, mixed precision, BFLOAT16 updates, dynamic or
  strided/view layouts, JVM-heap, read-only, thread-confined, or device-only parameter storage,
  asynchronous runs, or data loaders.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [mixed precision glossary](../../../../glossary.md#term-mixed-precision); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [view glossary](../../../../glossary.md#term-view-meanings); [layouts glossary](../../../../glossary.md#layout); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [heap glossary](../../../../glossary.md#term-heap); [device glossary](../../../../glossary.md#term-device); [asynchronous glossary](../../../../glossary.md#term-asynchronous))</span>
- Durable [checkpoint](../../../../glossary.md#term-checkpoint) bytes/files/codecs or [atomic](../../../../glossary.md#resource-transaction-and-rollback) filesystem [publication](../../../../glossary.md#term-publication).
- Compiled/fused optimizer graph operations or dependencies on Runtime, Prepare, tuning, CPU,
  Metal, or CUDA.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [optimizer glossary](../../../../glossary.md#term-optimizer); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda))</span>
- Changing [Tensor](../../../../glossary.md#tensor) [autograd](../../../../glossary.md#autograd) state or replacing the existing one-shot [Engine](../../../../glossary.md#term-engine) backward convenience.

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#core-lifecycle) — one Engine execution path and
  the explicit public-Engine dependency.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md#dependency-rules) — Training has no direct
  Runtime, Prepare, tuning, or concrete-backend edge.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — session, optimizer, mapping, atomicity, state, and close authority.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- [Compiler/autograd — training graph model](../../../../architecture/contracts/compiler-autograd.md#training-graph-model)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [training graph glossary](../../../../glossary.md#training-graph); [model glossary](../../../../glossary.md#term-model))</span>
  — compiler owns combined graph and ordered gradient roles.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- [Runtime/Prepare/Engine — run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  — the session reuses the existing prepared runner and result lifecycle.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) stable recursive [parameters](../../../../glossary.md#term-parameter); [Compiler](../../../../glossary.md#term-compiler) first-order publications; [Engine](../../../../glossary.md#term-engine) 0017
- Conflicts with: Training public [API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle) contract, and shared training documentation
- Parallel group: None
- Common base revision: `2e110c7688ead480faa4215ff67f9814fa81f4ad`
- Integration order: contract/[ADR](../../../../glossary.md#term-architecture-decision-record-adr), implementation/tests, documentation, independent Class C review
- Integration [validation](../../../../glossary.md#term-validation): full [Gradle](../../../../glossary.md#term-gradle) test and [Javadoc](../../../../glossary.md#term-javadoc) lifecycle plus architecture/docs checks
- Shared-document integration [owner](../../../../glossary.md#term-owner): this task

The user explicitly authorized this frontier from the recorded common base. Its prerequisite NN,
Compiler, and Engine capabilities are current; no other Training write frontier is active.

## Files and symbols

- `extensions/training` — public [API](../../../../glossary.md#term-application-programming-interface-api), [optimizer](../../../../glossary.md#term-optimizer)/session implementation, focused tests and Javadocs.
- `extensions/nn/.../Parameter.java`, `Module.java` — stable-storage/exclusive-session wording only.
- `testing/architecture-tests` — [dependency](../../../../glossary.md#term-dependency)-edge enforcement.
- `testing/integration-tests` — actual public [CPU](../../../../glossary.md#term-cpu) convergence [smoke](../../../../glossary.md#term-smoke-test).
- `ARCHITECTURE.md`, `docs/architecture/contracts/extensions-training.md`, ADR 0018, lifecycle and
  training-graph explanations — Class C contract and rationale.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [graph glossary](../../../../glossary.md#graph))</span>
- Training API/user guide/glossary and Training [planning](../../../../glossary.md#term-planning) records — public workflow and status.

## Acceptance criteria

- Opening a session snapshots non-empty recursive parameter paths, rejects empty/non-finite
  parameter state before seed allocation or compile, compiles one scalar objective and its first
  gradients once, prepares once, and exposes only non-parameter bindable inputs.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [finite glossary](../../../../glossary.md#term-finite); [seed glossary](../../../../glossary.md#term-seed); [compile glossary](../../../../glossary.md#compile); [scalar objective glossary](../../../../glossary.md#loss); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Repeated real CPU runs converge on a deterministic learnable problem without recompilation or
  re-preparation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- SGD numerical updates, first/subsequent dampening, momentum state, weight decay, compatible
  Nesterov behavior, FLOAT32 narrowing, update order, execution numbers, optimizer step numbers,
  zeroing, accumulation, snapshots, and restore are exact.<span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [dampening glossary](../../../../glossary.md#term-dampening); [momentum glossary](../../../../glossary.md#term-momentum); [weight decay glossary](../../../../glossary.md#term-weight-decay); [Nesterov glossary](../../../../glossary.md#term-nesterov); [FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [optimizer step glossary](../../../../glossary.md#optimizer-and-stochastic-gradient-descent))</span>
- Multiple parameter paths map to exact target-index gradient occurrences; aliases are not guessed
  or deduplicated.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Missing, duplicate, foreign, descriptor-incompatible, absent, dead, inaccessible, read-only,
  overlapping, replaced, unsupported, or non-finite parameter/input-derived optimizer state fails
  before optimizer mutation.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>
- A failed run, including represented overflow or post-execution/materialization/late-validation
  failure, leaves parameter bytes, optimizer state, counters, and pending gradients unchanged.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [overflow glossary](../../../../glossary.md#term-overflow); [materialization glossary](../../../../glossary.md#materialization); [validation glossary](../../../../glossary.md#term-validation))</span>
- Operation admission uses Engine-closed, session-closed, then busy precedence before argument
  validation. Close waits for an admitted operation, attempts all cleanup, and retains failure.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [admission glossary](../../../../glossary.md#term-admission); [Engine glossary](../../../../glossary.md#term-engine); [argument glossary](../../../../glossary.md#term-argument))</span>
- Detached step results and immutable finite state snapshots remain readable after session/Engine
  close; restore rejects mismatched sessions/configurations and non-finite payloads.<span data-glossary-navigation="true"> ([Detached glossary](../../../../glossary.md#detached-model-results); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- CPU and a known configured real Metal/mixed route succeed. A separately classified unsupported
  Metal-only route fails before optimizer mutation; Training contains no backend branch/import.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [backend glossary](../../../../glossary.md#backend))</span>
- Training publishes exact NN, Model, and Engine API dependencies, and a consumer declaring only
  Training can compile every leaked public signature type.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [signature glossary](../../../../glossary.md#term-signature))</span>
- The [optimizer](../../../../glossary.md#term-optimizer) [element](../../../../glossary.md#term-element) loop reuses primitive staging [buffers](../../../../glossary.md#term-buffer) and creates no per-element object.

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :extensions:training:test :testing:architecture-tests:test :testing:integration-tests:test
./gradlew :extensions:training:javadoc :extensions:nn:javadoc
./gradlew test javadoc
```

Also [run](../../../../glossary.md#term-run) the repository documentation/architecture checks named by the root build or maintained
scripts, `git diff --check`, and one executable public [CPU](../../../../glossary.md#term-cpu) convergence [smoke](../../../../glossary.md#term-smoke-test).

Integration/repository validation: required because the task changes dependencies, public [API](../../../../glossary.md#term-application-programming-interface-api),
architecture, [concurrency](../../../../glossary.md#term-concurrency), and multiple [modules](../../../../glossary.md#term-module).

## Documentation and review impact

- Update authoritative root/scoped contract, ADR, architecture explanations, API reference, user
  training guide, glossary, task/master/roadmap, package/type Javadocs, and NN cross-links.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Mandatory independent targeted Class C review after implementation and validation; leave the
  review-ready change uncommitted.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ready glossary](../../../../glossary.md#term-ready))</span>

## Result

- Added the public sealed optimizer configuration boundary, SGD with persistent momentum,
  dampening, coupled weight decay and compatible Nesterov, explicit gradient modes, detached
  step/state values, and one reusable <span data-glossary-navigation="true"> ([sealed glossary](../../../../glossary.md#term-sealed); [optimizer glossary](../../../../glossary.md#term-optimizer); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [momentum glossary](../../../../glossary.md#term-momentum); [dampening glossary](../../../../glossary.md#term-dampening); [weight decay glossary](../../../../glossary.md#term-weight-decay); [Nesterov glossary](../../../../glossary.md#term-nesterov); [gradient glossary](../../../../glossary.md#term-gradient); [detached glossary](../../../../glossary.md#detached-model-results))</span>[`TrainingSession`](../../../../glossary.md#training-session--trainingsession) over exact NN recursive parameter paths
  and Engine gradient publications.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Added finite-value/overflow precommit validation, reusable primitive staging/rollback buffers,
  strict snapshot/restore, lifecycle-first single-operation admission, wait-on-close ownership,
  retained attempt-all cleanup, private seed cleanup, and the bounded shareable-native
  non-empty <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [overflow glossary](../../../../glossary.md#term-overflow); [precommit glossary](../../../../glossary.md#term-precommit); [rollback glossary](../../../../glossary.md#term-rollback); [buffers glossary](../../../../glossary.md#term-buffer); [snapshot glossary](../../../../glossary.md#term-snapshot); [lifecycle glossary](../../../../glossary.md#lifecycle); [operation glossary](../../../../glossary.md#operation); [admission glossary](../../../../glossary.md#term-admission); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [seed glossary](../../../../glossary.md#term-seed))</span>[`FLOAT32`](../../../../glossary.md#term-float32)/[`FLOAT64`](../../../../glossary.md#term-float64) update domain.
- Published exact NN, Model, and Engine Gradle API edges plus a compile-only consumer that declares
  only Training.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [compile glossary](../../../../glossary.md#compile))</span>
- Added numerical, mapping, non-finite/overflow failure atomicity, storage, accumulation, restore,
  close/busy/admitted-operation, rollback, retained-cleanup-failure, detached-lifetime,
  architecture dependency, CPU convergence, configured real Metal/mixed success, and precisely
  classified unsupported Metal-only coverage.<span data-glossary-navigation="true"> ([failure atomicity glossary](../../../../glossary.md#term-failure-atomicity); [lifetime glossary](../../../../glossary.md#term-lifetime); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>
- The initial implementation [validation](../../../../glossary.md#term-validation) passed `./gradlew test javadoc` and `git diff --check`.
  Mandatory external Class C review returned `BLOCK` with six P1 and one P2 findings. All findings
  were remediated in the uncommitted worktree. Focused training/architecture/integration/Javadoc
  validation passed; both configured real Metal cases passed with the built dylib; full<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [dylib glossary](../../../../glossary.md#term-dylib))</span>
  `./gradlew test javadoc` passed with 98 actionable tasks; and `git diff --check` passed.
- Exact external review trail: initial verdict `BLOCK` with six P1 and one P2 findings; external
  Class C re-review verdict `APPROVE`, findings `0`.

# Task 0002A: [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) [parameter](../../../../glossary.md#term-parameter) groups

## Status

Complete

## Change class

Class B — adds a public Training capability and extends its [module](../../../../glossary.md#term-module)-local [optimizer](../../../../glossary.md#term-optimizer)-state and
restore contract. It changes no module [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or [backend route](../../../../glossary.md#backend-route).

## Goal

Let one [`TrainingSession`](../../../../glossary.md#training-session--trainingsession) apply different [immutable](../../../../glossary.md#term-immutable) [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) configurations to disjoint named
[parameter](../../../../glossary.md#term-parameter) paths while retaining one [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel), one global [run](../../../../glossary.md#term-run)/step [lifecycle](../../../../glossary.md#lifecycle), and the
existing ungrouped [API](../../../../glossary.md#term-application-programming-interface-api).

## [Scope](../../../../glossary.md#term-scope)

- Add [immutable](../../../../glossary.md#term-immutable) public `io.github.pho001.synaptik.training.ParameterGroup` containing a nonempty
  ordered list of stable recursive parameter paths and one <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>[`Optimizer`](../../../../glossary.md#term-optimizer) configuration (currently
  only `Sgd` is permitted). Its ordered path list is its identity; no separate group name or index
  namespace is introduced.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add `TrainingSession.open(engine, module, objective, Optimizer defaultOptimizer,
  List<ParameterGroup> groups)`. Every discovered parameter belongs to at most one explicit group;
  omitted paths use `defaultOptimizer`. Empty groups list is equivalent to the existing overload.
  Groups do not freeze or exclude parameters from gradient targets.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [targets glossary](../../../../glossary.md#term-target))</span>
- Snapshot the ordered group list at open. Reject a null list or entry, null/blank/duplicate
  paths, empty groups, repeated membership across groups, and paths absent from the captured
  Module before seed allocation, compile, or prepare. The list order and each group's path order
  are part of the exact session/state configuration identity, even where reordering would not
  change arithmetic.<span data-glossary-navigation="true"> ([Snapshot glossary](../../../../glossary.md#term-snapshot); [Module glossary](../../../../glossary.md#term-module); [seed glossary](../../../../glossary.md#term-seed); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare))</span>
- Resolve one `Sgd` per parameter in Module traversal order at open. Validate coefficient
  narrowing for each assigned parameter dtype, then use that indexed configuration for update,
  momentum staging, and restore validation without a per-element map lookup.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [dtype glossary](../../../../glossary.md#data-type--datatype); [momentum glossary](../../../../glossary.md#term-momentum); [validation glossary](../../../../glossary.md#term-validation); [element glossary](../../../../glossary.md#term-element); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Preserve the old `TrainingSession.open(..., Optimizer)` behavior and the public
  `TrainingState.optimizer()` result as the default `Sgd`. Add immutable
  `parameterGroups()` accessors on session and state. `state()` [captures](../../../../glossary.md#term-capture-meanings) group configuration;
  `restore()` compares the exact default and ordered groups before any state installation.
- Keep one global execution counter, optimizer-step counter, accumulation count, gradient mode,
  and atomic commit across all parameters. Document the API with a two-rate example.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [commit glossary](../../../../glossary.md#term-commit); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

- Adam/AdamW, mixed optimizer algorithms, per-group counters or gradient modes, parameter
  freezing, dynamic regrouping after open, persistence codec/schema, backend optimizer kernels,
  an optional rollback policy, or changes to the native parameter-storage domain.<span data-glossary-navigation="true"> ([Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [persistence glossary](../../../../glossary.md#term-persistence); [codec glossary](../../../../glossary.md#term-codec); [schema glossary](../../../../glossary.md#term-schema); [backend glossary](../../../../glossary.md#backend); [kernels glossary](../../../../glossary.md#kernel); [rollback glossary](../../../../glossary.md#term-rollback))</span>

## Contracts

- [`ARCHITECTURE.md` scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [module ownership](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>, and
  [dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span> keep optimizer policy in
  Training and the NN/Engine edges unchanged.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Extensions/training — `extensions/nn`](../../../../architecture/contracts/extensions-training.md#extensionsnn)
  owns Module parameter declarations and deterministic recursive paths.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- [Extensions/training — `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  allows [`ParameterGroup`](../../../../glossary.md#parameter-group--parametergroup) and [backend](../../../../glossary.md#backend)-neutral [optimizer](../../../../glossary.md#term-optimizer) orchestration.
- [Extensions/training — optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  requires all parameter targets, one session-owned state set, finite updates, complete precommit
  validation, ordinary-failure atomicity, and exact state/restore compatibility.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [finite glossary](../../../../glossary.md#term-finite); [precommit glossary](../../../../glossary.md#term-precommit); [validation glossary](../../../../glossary.md#term-validation); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>

If a selected contract conflicts with the scoped design, stop and request a coordinated
architecture decision rather than changing the contract implicitly.

## Dependencies and integration

- Depends on: Training 0001E (Complete and pushed at `cbd14f3b`).
- Conflicts with: concurrent Training public optimizer API, state/restore, parameter mapping, or
  lifecycle changes; Training 0002B until this contract is integrated.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None.
- Common base revision: `cbd14f3b` on `main`.
- Integration order: implementation, focused Training/integration validation, independent public
  API and state-contract review, then integration-owner repository validation.
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew test javadoc` and `git diff --check` once after integration,
  because the public behavior also changes an end-to-end integration test module.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordination context.

Training 0001E has final validation, documentation, and independent-review results. The pushed
base is clean, no conflicting writer is active, and 0002A is the sole authorized Training write
frontier. No implementation is launched by this [planning](../../../../glossary.md#term-planning) update.

Launch verification (2026-10-12): `main` and `origin/main` both point to `cbd14f3b` before
publishing this brief. Training 0001E is Complete with its required validation and review;
the master plan and roadmap expose only 0002A as [Ready](../../../../glossary.md#term-ready), and 0002B remains Draft. No conflicting
Training writer is active. Implementation starts only after this planning update is published.

## Files and symbols

- `extensions/training/src/main/java/io/github/pho001/synaptik/training/ParameterGroup.java` —
  new immutable path/configuration value.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java` —
  grouped open, path resolution, run dispatch, state/restore, and public accessor.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingState.java` —
  detached ordered groups and exact restore identity, retaining ungrouped API behavior.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `extensions/training/src/main/java/io/github/pho001/synaptik/training/TrainingParameter.java` —
  assigned-SGD update and momentum validation without changing its per-element algorithm.<span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [momentum glossary](../../../../glossary.md#term-momentum); [validation glossary](../../../../glossary.md#term-validation); [element glossary](../../../../glossary.md#term-element))</span>
- Focused tests in `extensions/training/src/test/java/`; public [CPU](../../../../glossary.md#term-cpu) scenario in
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`.
- `docs/api/training-api.md`, `docs/user-guide/training.md`, affected Javadocs, and targeted
  Training terminology in `docs/glossary.md`.

## Acceptance criteria

- Existing four-[argument](../../../../glossary.md#term-argument) `open`, ungrouped state snapshots, arithmetic, and restore remain
  source-compatible and behaviorally unchanged. Empty explicit groups behave identically.
- Two disjoint groups can use distinct learning rates/momentum and produce independently expected
  parameter bytes in one atomic run. Ungrouped parameters use the default SGD.<span data-glossary-navigation="true"> ([momentum glossary](../../../../glossary.md#term-momentum); [parameter glossary](../../../../glossary.md#term-parameter); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [run glossary](../../../../glossary.md#term-run); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd))</span>
- Invalid or overlapping path membership fails before compile/prepare or parameter mutation;
  group mutation through caller-owned collections cannot change an open session or detached state.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- State round-trips exact default/group configurations and parameter bytes. Restore rejects any
  changed group order, path order, membership, or SGD configuration before writing parameters,
  slots, accumulation, or counters.
- `RESET_AND_STEP`, `ACCUMULATE`, `ACCUMULATE_AND_STEP`, and `zeroGrad` retain global semantics.
  Per-parameter coefficient narrowing and zero-momentum slot rules hold for FLOAT64, FLOAT32,
  BFLOAT16, and FLOAT16; an injected later-parameter failure rolls back all groups.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- Group resolution occurs once at open; update loops have no per-element path/map dispatch,
  boxing, or new allocation. A public CPU integration test exercises grouped training and restore.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch); [CPU glossary](../../../../glossary.md#term-cpu); [integration test glossary](../../../../glossary.md#term-integration-test))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: focused new and existing Training tests and the targeted [CPU](../../../../glossary.md#term-cpu) [integration test](../../../../glossary.md#term-integration-test), then one
final affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) after executable code stabilizes:

```bash
./gradlew :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test
git diff --check
```

Integration [owner](../../../../glossary.md#term-owner): reuse successful worker evidence, obtain independent targeted Class B review,
then run the recorded repository [checkpoint](../../../../glossary.md#term-checkpoint) once because both Training and integration-test
modules change.

## Documentation and review impact

Independent review finalizes public Javadocs, the [API](../../../../glossary.md#term-application-programming-interface-api) and user-guide example, and glossary
terminology. Explain that group paths come from `Module.parametersRecursively()`, that unlisted
[parameters](../../../../glossary.md#term-parameter) use the default [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd), and that group ordering affects restore compatibility. No
architecture or decision record is expected unless implementation uncovers a contract conflict.

## Result

- Implemented in isolated worktree [commit](../../../../glossary.md#term-commit) `92494077` and integrated from [planning](../../../../glossary.md#term-planning) base
  `9ba848a9`: [immutable](../../../../glossary.md#term-immutable) [`ParameterGroup`](../../../../glossary.md#parameter-group--parametergroup), per-parameter SGD selection, exact ordered group
  snapshot/restore identity, preserved ungrouped API, focused tests, and public CPU integration.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [snapshot glossary](../../../../glossary.md#term-snapshot); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Changed six Training production files (including new `ParameterGroup`), two Training test
  files, one CPU integration test, and the Training API guide, user guide, and glossary.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Worker [validation](../../../../glossary.md#term-validation) passed with
  `SYNAPTIK_METAL_TEST_LIBRARY=<source-matched local dylib> ./gradlew
  :extensions:training:test :extensions:training:javadoc :testing:integration-tests:test`:
  57 Training tests and 8 CPU integration tests, zero failures/skips in those suites.
  `git diff --check` passed.
- Independent Class B public [API](../../../../glossary.md#term-application-programming-interface-api)/state-contract and documentation/[Javadoc](../../../../glossary.md#term-javadoc) review returned
  `APPROVE`, zero findings. The two-rate examples and ordered restore [identity](../../../../glossary.md#term-identity-meanings) were reviewed.
- Integration-[owner](../../../../glossary.md#term-owner) [checkpoint](../../../../glossary.md#term-checkpoint) passed with the same [Metal](../../../../glossary.md#term-metal) library and `./gradlew test javadoc`
  (`BUILD SUCCESSFUL`; 100 actionable tasks: 2 executed, 6 from cache, 92 up-to-date).
  Integrated <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`git diff --check` passed. No architecture or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) update was needed.
- Limitations: only SGD groups are supported; Adam/AdamW remain for a later task. No unresolved
  issue or required follow-up within 0002A.<span data-glossary-navigation="true"> ([Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw))</span>

Status: Complete

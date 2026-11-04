# Task Training0008: Fresh-value [TrainingSession](../../../../glossary.md#training-session--trainingsession) runs

> **Editorial reading context — not a status update.** The application passes fresh stored
> batch/target Tensors in session.inputs order, not replacement expressions to capture.
> Training inserts its fixed parameter and cotangent-seed bindings into final compiled
> positions and delegates to Engine.runValues on the existing preparation. This does not
> replace prototype storage or recopy parameters each run. Fresh input storage is borrowed
> through the synchronous operation and internal result closure; the session retains its
> optimizer history, not those batches. Today's [positional input mapping](../../../../reference/training/sessions.md#session-construction-and-input-mapping)
> and [shared optimizer transaction](../../../../reference/training/optimizers.md#atomicity-state-concurrency-and-close)
> explain why fresh identities are allowed but Training's stronger overlap rejection still applies.

## Status

Complete — main accepted after Euclid's definitive code/docs APPROVE, no findings, and the
source-matched root gate (2026-10-19). Authorized Training frontier: None.
NN0028 is the next [planning](../../../../glossary.md#term-planning) candidate, still Draft; [Model](../../../../glossary.md#term-model)/withEngine remains [TARGET](../../../../glossary.md#term-target).

## Change class

Class C — public positional input [admission](../../../../glossary.md#term-admission) crosses the Training/[Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) boundary while
preserving a stateful [optimizer](../../../../glossary.md#term-optimizer) [transaction](../../../../glossary.md#term-transaction), [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) storage, and concurrent close admission.

## Goal

Add `public TrainingStep runValues(List<Tensor> values, GradientMode mode)` to the existing
[`TrainingSession`](../../../../glossary.md#training-session--trainingsession): fresh stored [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) and [targets](../../../../glossary.md#term-target) reuse one compiled/prepared [objective](../../../../glossary.md#term-objective) and the
same fixed [parameters](../../../../glossary.md#term-parameter) and [optimizer state](../../../../glossary.md#term-optimizer-state). Existing [`run`](../../../../glossary.md#term-run) retains exact-ID, arbitrary-order [binding](../../../../glossary.md#term-binding).

## [Scope](../../../../glossary.md#term-scope)

- `values` supplies exactly `session.inputs()` positions in that returned compiled encounter order,
  excluding captured parameters AND the private cotangent seed; targets and every other external
  bindable leaf are included. Fresh TensorIds are accepted with exact full static descriptors.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [cotangent seed glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed); [targets glossary](../../../../glossary.md#term-target); [leaf glossary](../../../../glossary.md#term-leaf); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Reuse `fixedCompiledInputs`, `externalIndexByCompiledInput`, `nonParameterCompiledIndices` and
  diagnostic roles. Insert fixed parameters/seed into final compiled positions and call public<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
  [`Engine.runValues`](../../../../glossary.md#fresh-value-binding--enginerunvalues) on the existing prepared owner. Values are stored values, not expressions to
  recapture or execute. Do not rebind or copy prototype storage.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [prototype glossary](../../../../glossary.md#term-prototype))</span>
- Share the existing [run](../../../../glossary.md#term-run) [transaction](../../../../glossary.md#term-transaction) privately; preserve old `run` validation/diagnostics and
  Engine.run dispatch. No second optimizer implementation or per-element dispatch is needed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Engine glossary](../../../../glossary.md#term-engine); [dispatch glossary](../../../../glossary.md#term-dispatch); [optimizer glossary](../../../../glossary.md#term-optimizer); [element glossary](../../../../glossary.md#term-element))</span>
- Keep Training's stronger nonempty overlap rejection, including a repeated fresh Tensor in two
  positions, external/external and external/parameter overlap, and seed checks. Generic Engine's
  repeated-value permission does not relax Training; zero-byte overlap behavior stays unchanged.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Fresh caller storage is borrowed only through the synchronous call and internal result closure,
  never retained for the next run. Parameters and seed remain captured fixed bindings.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [synchronous glossary](../../../../glossary.md#term-synchronous); [bindings glossary](../../../../glossary.md#term-binding))</span>

## Non-goals

No NN0028 [Model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), [checkpoint](../../../../glossary.md#term-checkpoint)/[codec](../../../../glossary.md#term-codec) work, new production types, project edges or build changes;
no per-[run](../../../../glossary.md#term-run) [graph](../../../../glossary.md#graph) mutation/reconstruction/[compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare), [parameter](../../../../glossary.md#term-parameter) copying, state/counter [widening](../../../../glossary.md#term-widening),
[optimizer](../../../../glossary.md#term-optimizer) arithmetic, [rollback](../../../../glossary.md#term-rollback) policy or memory redesign. No advancing graph [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [backend
capability](../../../../glossary.md#backend-capability), [fusion](../../../../glossary.md#term-fusion), native/performance campaign or broader Training0005 gate. No Model/Training
[autotuning](../../../../glossary.md#autotuning--model-autotuning) integration; existing `Engine.prepareTuned` stays separate. Report any concrete need
for a new abstraction or expanded contract before proceeding.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning extensions-training contract: [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining)
  and [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer))</span>.
- Read-only [Engine](../../../../glossary.md#term-engine) contract: [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  and [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>,
  especially exact descriptors, positional geometry, borrowing, and unchanged identity-run checks.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [borrowing glossary](../../../../glossary.md#term-borrowing); [identity glossary](../../../../glossary.md#term-identity-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- Accepted [TARGET](../../../../glossary.md#term-target) rationale, not authority: ADR0039
  [Source, shape, state and ownership](../../../../design/decisions/0039-portable-graph-model-persistence.md#source-shape-state-and-ownership)<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
  and [Coordinated authority cutovers and validation](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation)<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>.

At launch Training authority explicitly excluded fresh positional [admission](../../../../glossary.md#term-admission). Before code relies on
it, [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) the bounded root Core [lifecycle](../../../../glossary.md#lifecycle) and owning Training lifecycle update, explanations
and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) reconciliation in this same capability. Clarify fixed [parameter](../../../../glossary.md#term-parameter)/[seed](../../../../glossary.md#term-seed) identities versus
call-local external values; preserve all other lifecycle rules. Root [ownership](../../../../glossary.md#term-ownership-meanings)/dependencies/testing
are reviewed, not permission for new edges. Engine authority stays unchanged. This is the accepted
TARGET cutover, not permission from this brief or ADR alone. Stop on any further contract ambiguity.

## Dependencies and integration

- Depends on: Engine0021 + Training0007 (capabilities); completed Checkpoint0008 is the serial
  shared-document join, not an algorithm dependency. Compiler0008 is already integrated.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: NN0028 and any shared Training API/lifecycle/root/scoped/documentation writer;
  Training0006/adapter and broader Training0005 remain held outside this capability.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None; serial exclusive writer delegation, separate planning/implementation/docs
  contexts. No concurrent shared-document writes.
- Common base revision: main `7386249479ccda991b96a7e27bd5ed0279ca1ae3`.
- Integration order: completed Engine0021 → Training0007 → Compiler0008 → Checkpoint0008;
  Training0008 authority cutover → implementation → independent code/docs review → main gate;
  only then may main advance NN0028 planning/readiness.
- Integration validation: focused architecture/API-consumer and CPU regression below, then main's
  one recorded source-matched root <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [regression glossary](../../../../glossary.md#term-regression))</span>`test javadoc` [checkpoint](../../../../glossary.md#term-checkpoint) for this authority/integration change.
- Shared-document integration owner: main model-execution coordinator; delegates bounded exclusive
  writes to the clean implementation/docs contexts and integrates active-status updates.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [model glossary](../../../../glossary.md#term-model))</span>

Launch verification (2026-10-19): HEAD/common base unchanged; the approved 2267-file [snapshot](../../../../glossary.md#term-snapshot)
has no drift, with only five authorized [planning](../../../../glossary.md#term-planning) deltas from baseline; all source and both protected
files unchanged. Engine0021, Training0007, Compiler0008 and Checkpoint0008 final Complete reviews/
gates verified. Hilbert's independent review of the frozen five documents returned APPROVE with no
findings. Training0008 is the sole authorized Training frontier; NN0028/Training0006/adapter are
inactive, no conflicting writer. Main authorized [Ready](../../../../glossary.md#term-ready) and will delegate exclusive implementation.
Never edit/stage/revert `THIRD_PARTY_NOTICES.md` or
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`.

## Files and symbols

- `modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`:
  [`run`](../../../../glossary.md#term-run), new `runValues`, `executeAndDecode`, `bindInputs`, `validateInputStorage`, `bindingPlan`,
  `inputs`, [precommit](../../../../glossary.md#term-precommit)/[admission](../../../../glossary.md#term-admission)/cleanup. Read adjacent `TrainingParameter` and [`TrainingParameters`](../../../../glossary.md#live-parameter-source--trainingparameters)
  for fixed live/native admission and state invariants; no optimizer algorithm changes expected.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Existing training tests in the matching test package: `TrainingSessionTest`,
  `TrainingSessionFailureContractTest`, `AdamOptimizerTest`, and [fixture](../../../../glossary.md#term-fixture) `TestParameterSource`.
  Reuse hooks, byte-state comparisons and private-field inspection already used there.
- Read-only `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java#runValues`
  and matching `EngineTypedLifecycleTest`
  establish positional binding and unresolved-layout capacity behavior; no Engine change expected.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [layout glossary](../../../../glossary.md#layout); [Engine glossary](../../../../glossary.md#term-engine))</span>
- `testing/architecture-tests/src/trainingApiConsumer/java/io/github/pho001/synaptik/testing/architecture/TrainingApiConsumer.java`;
  existing `NnTrainingDependencyContractTest` in that [module](../../../../glossary.md#term-module)'s test package.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`:
  one focused real-CPU public workflow regression. Production package stays<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [regression glossary](../../../../glossary.md#term-regression))</span>
  `io.github.pho001.synaptik.training`; no added/moved production type or public test accessor.

## Acceptance criteria

1. Repeated fresh batches AND targets produce independently expected, observably different scalar
   objective, gradient and update outcomes. Observe gradients through accumulation/state or the
   existing oracle, not a new API. Parameter Tensor/storage identities, <span data-glossary-navigation="true"> ([batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [targets glossary](../../../../glossary.md#term-target); [scalar objective glossary](../../../../glossary.md#loss); [gradient glossary](../../../../glossary.md#term-gradient); [oracle glossary](../../../../glossary.md#term-test-oracle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor))</span>`compiledGraph()` and the
   single private prepared owner stay fixed; source/private-field evidence and existing seams
   establish reuse without a public prepared accessor. Prototypes remain unchanged; closing a
   completed batch's storage does not affect the next fresh run or detached results.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Prototypes glossary](../../../../glossary.md#term-prototype); [detached glossary](../../../../glossary.md#detached-model-results))</span>
2. `inputs()` excludes parameters/seed and preserves relative final compiled order. Check correct
   positions and swapped incompatible descriptors; same-descriptor positions are positional, not
   identity-remapped. Reject wrong count, null list/element/mode, dtype/shape/layout presence or
   geometry/requiresGrad mismatch, absent/short/dead/inaccessible storage and storage dtype mismatch
   where constructible. Reuse Engine's checked static geometry for unresolved layouts; do NOT
   strengthen old <span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings); [element glossary](../../../../glossary.md#term-element); [dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [Engine glossary](../../../../glossary.md#term-engine))</span>[`run`](../../../../glossary.md#term-run)'s [layout](../../../../glossary.md#layout)-span-or-zero storage checks, diagnostics, [admission](../../../../glossary.md#term-admission) or precedence.
3. Reject repeated nonempty values, overlapping distinct external slices and parameter aliases
   before execution/update; preserve compiled-order overlap diagnostics and zero-byte treatment.
   Keep fixed live source/path/Tensor/storage validation and native-only parameter admission.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation))</span>
4. Malformed fresh input, non-finite decoded gradient, post-execution/precommit live-binding failure
   and injected later parameter-write failure preserve all parameter/master bytes, slots, pending
   accumulation and counters. Reuse hooks to prove result cleanup and rollback; retry succeeds
   after repairing the test fault. Target-indexed gradient publication mapping stays unchanged.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precommit glossary](../../../../glossary.md#term-precommit); [binding glossary](../../../../glossary.md#term-binding); [rollback glossary](../../../../glossary.md#term-rollback); [publication glossary](../../../../glossary.md#term-publication))</span>
5. Cover RESET_AND_STEP, ACCUMULATE and ACCUMULATE_AND_STEP with pending-state behavior, plus one
   stateful optimizer continuation that mixes old <span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>`run`, fresh `runValues`, `state` and `restore`.
   Preserve default/group SGD/Adam/AdamW and low-precision master/narrowing semantics through the
   shared transaction and existing regressions, not a new dtype × optimizer × mode test matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [Adam glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [precision glossary](../../../../glossary.md#term-precision); [narrowing glossary](../../../../glossary.md#term-narrowing); [transaction glossary](../../../../glossary.md#term-transaction))</span>
6. Old `run` still accepts exact IDs in arbitrary order and rejects foreign fresh IDs, duplicates
   and missing inputs. New entry point keeps Engine-closed → session-closed → busy → arguments
   admission; deterministic hooks show an admitted fresh run finishes before close cleanup.
   Result-before-prepared-owner-before-seed cleanup, failure replay and detached lifetimes remain.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument))</span>
7. Isolated Training-only consumer compiles the new method with no NN/back-edge. Focused CPU
   integration proves fresh batches/targets update the same parameters through public Engine;
   no new test framework, backend route or native rebuild is required.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [CPU glossary](../../../../glossary.md#term-cpu); [backend route glossary](../../../../glossary.md#backend-route))</span>

## [Validation](../../../../glossary.md#term-validation)

Implementation worker: focused development tests, then one final affected Training suite after
executable edits stabilize; final [Javadoc](../../../../glossary.md#term-javadoc) after the independent docs pass. [Run](../../../../glossary.md#term-run) from repository root:

```bash
./gradlew :extensions:training:test --tests '*TrainingSessionTest' --tests '*TrainingSessionFailureContractTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionValuesTest' --tests '*TrainingSessionValueAdmissionTest' --tests '*TrainingSessionValueFailureTest'
./gradlew :extensions:training:test :extensions:training:javadoc :testing:architecture-tests:test --tests '*NnTrainingDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest.freshValuesTrainOnCpu'
git diff --check
```

Architecture `test` includes `compileTrainingApiConsumerJava`. Euclid reused successful worker
evidence. Main's accepted source-matched root gate reused the existing [Metal](../../../../glossary.md#term-metal) [dylib](../../../../glossary.md#term-dylib), without rebuilding:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```

## Documentation and review impact

Same-change [scope](../../../../glossary.md#term-scope): `ARCHITECTURE.md` Core [lifecycle](../../../../glossary.md#lifecycle); owning
`docs/architecture/contracts/extensions-training.md` [modules](../../../../glossary.md#term-module)/training and [Optimizer](../../../../glossary.md#term-optimizer)/training
lifecycle; `docs/architecture/lifecycle.md` Public training lifecycle and
`docs/architecture/training-graph.md` Optimizer as a [backend](../../../../glossary.md#backend)-agnostic step / Continuing at the next
training input; `docs/api/training-api.md` construction/input mapping, [atomicity](../../../../glossary.md#term-atomicity) and [Engine](../../../../glossary.md#term-engine)
coordination; `docs/user-guide/training.md` runnable [scalar](../../../../glossary.md#term-scalar) problem, lifetimes and common errors;
ADR0039 Source/[shape](../../../../glossary.md#shape)/state/[ownership](../../../../glossary.md#term-ownership-meanings) and cutover status; targeted glossary [Training session](../../../../glossary.md#training-session--trainingsession) and
[Fresh-value binding](../../../../glossary.md#fresh-value-binding--enginerunvalues) entries. Review `docs/architecture/module-boundaries.md` Training heading and
`modules/training/src/main/java/io/github/pho001/synaptik/training/package-info.java`;
update only affected claims or record no impact.
Finalize [TrainingSession](../../../../glossary.md#training-session--trainingsession) class/open/inputs/[run](../../../../glossary.md#term-run)/runValues and changed private-contract Javadocs,
including every [parameter](../../../../glossary.md#term-parameter)/result/failure and [synchronous](../../../../glossary.md#term-synchronous) borrow; no new term/framework is needed.
Use the general and matching documentation profiles and an executable fresh-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[target](../../../../glossary.md#term-target) example.
Separate clean implementation and independent code/docs contexts are mandatory. Main owns concise
status/link updates in this brief, Training master, program and roadmap after review/acceptance;
preserve completed histories and keep NN0028 TARGET/Draft. No unrelated guides or authority changes.

## Result

- `runValues` shares [transaction](../../../../glossary.md#term-transaction)/[captures](../../../../glossary.md#term-capture-meanings); old checks retained; no new production type/edge/policy.
- Java: [TrainingSession](../../../../glossary.md#training-session--trainingsession), three Value* tests/[fixture](../../../../glossary.md#term-fixture), [API](../../../../glossary.md#term-application-programming-interface-api) consumer, [CPU](../../../../glossary.md#term-cpu) [regression](../../../../glossary.md#term-regression); docs/[Javadoc](../../../../glossary.md#term-javadoc) updated above.
- Commands above PASS: focused 64; Training 93; architecture 2 + consumer [compile](../../../../glossary.md#compile); CPU 1; skips/failures 0; Javadoc/diff clean.
- Fixture fixes: [Model](../../../../glossary.md#term-model) forbids short resolved stores; CPU forbids unresolved compile. [Metadata](../../../../glossary.md#term-metadata)-only test preserves old checks.
- Guide CPU-tested; links checked; package/[module](../../../../glossary.md#term-module)-boundary claims accurate.
- Euclid definitive APPROVE/no findings; main accepted. Gate above: exit 0, BUILD SUCCESSFUL 5m10s; 107 tasks, 8 executed/99 up-to-date.
- Fresh architecture 12/0 skips; integration 133/1 skip (CPU fixture lacks local tuning alternatives); both actual [Metal](../../../../glossary.md#term-metal) training cases PASS.
- Final 19-project reports: 4199 cases, 0 failures/errors, 41 skips (CPU 30/Metal 10 opt-ins/integration 1); not all fresh; Training 93/Javadoc reused.
- HEAD/base unchanged; 2271 files byte-identical through gate; protected changes unchanged; no unresolved issue/follow-up.
Status: Complete

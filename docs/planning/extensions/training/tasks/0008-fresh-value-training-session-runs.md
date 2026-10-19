# Task Training0008: Fresh-value TrainingSession runs

## Status

Complete — main accepted after Euclid's definitive code/docs APPROVE, no findings, and the
source-matched root gate (2026-10-19). Authorized Training frontier: None.
NN0028 is the next planning candidate, still Draft; Model/withEngine remains TARGET.

## Change class

Class C — public positional input admission crosses the Training/Engine lifecycle boundary while
preserving a stateful optimizer transaction, borrowed storage, and concurrent close admission.

## Goal

Add `public TrainingStep runValues(List<Tensor> values, GradientMode mode)` to the existing
`TrainingSession`: fresh stored batches and targets reuse one compiled/prepared objective and the
same fixed parameters and optimizer state. Existing `run` retains exact-ID, arbitrary-order binding.

## Scope

- `values` supplies exactly `session.inputs()` positions in that returned compiled encounter order,
  excluding captured parameters AND the private cotangent seed; targets and every other external
  bindable leaf are included. Fresh TensorIds are accepted with exact full static descriptors.
- Reuse `fixedCompiledInputs`, `externalIndexByCompiledInput`, `nonParameterCompiledIndices` and
  diagnostic roles. Insert fixed parameters/seed into final compiled positions and call public
  `Engine.runValues` on the existing prepared owner. Values are stored values, not expressions to
  recapture or execute. Do not rebind or copy prototype storage.
- Share the existing run transaction privately; preserve old `run` validation/diagnostics and
  Engine.run dispatch. No second optimizer implementation or per-element dispatch is needed.
- Keep Training's stronger nonempty overlap rejection, including a repeated fresh Tensor in two
  positions, external/external and external/parameter overlap, and seed checks. Generic Engine's
  repeated-value permission does not relax Training; zero-byte overlap behavior stays unchanged.
- Fresh caller storage is borrowed only through the synchronous call and internal result closure,
  never retained for the next run. Parameters and seed remain captured fixed bindings.

## Non-goals

No NN0028 Model API, checkpoint/codec work, new production types, project edges or build changes;
no per-run graph mutation/reconstruction/compile/prepare, parameter copying, state/counter widening,
optimizer arithmetic, rollback policy or memory redesign. No advancing graph RNG, BPTT, backend
capability, fusion, native/performance campaign or broader Training0005 gate. No Model/Training
autotuning integration; existing `Engine.prepareTuned` stays separate. Report any concrete need
for a new abstraction or expanded contract before proceeding.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning extensions-training contract: [`modules/training`](../../../../architecture/contracts/extensions-training.md#modulestraining)
  and [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- Read-only Engine contract: [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  and [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution),
  especially exact descriptors, positional geometry, borrowing, and unchanged identity-run checks.
- Accepted TARGET rationale, not authority: ADR0039
  [Source, shape, state and ownership](../../../../design/decisions/0039-portable-graph-model-persistence.md#source-shape-state-and-ownership)
  and [Coordinated authority cutovers and validation](../../../../design/decisions/0039-portable-graph-model-persistence.md#coordinated-authority-cutovers-and-validation).

At launch Training authority explicitly excluded fresh positional admission. Before code relies on
it, coordinate the bounded root Core lifecycle and owning Training lifecycle update, explanations
and ADR reconciliation in this same capability. Clarify fixed parameter/seed identities versus
call-local external values; preserve all other lifecycle rules. Root ownership/dependencies/testing
are reviewed, not permission for new edges. Engine authority stays unchanged. This is the accepted
TARGET cutover, not permission from this brief or ADR alone. Stop on any further contract ambiguity.

## Dependencies and integration

- Depends on: Engine0021 + Training0007 (capabilities); completed Checkpoint0008 is the serial
  shared-document join, not an algorithm dependency. Compiler0008 is already integrated.
- Conflicts with: NN0028 and any shared Training API/lifecycle/root/scoped/documentation writer;
  Training0006/adapter and broader Training0005 remain held outside this capability.
- Parallel group: None; serial exclusive writer delegation, separate planning/implementation/docs
  contexts. No concurrent shared-document writes.
- Common base revision: main `7386249479ccda991b96a7e27bd5ed0279ca1ae3`.
- Integration order: completed Engine0021 → Training0007 → Compiler0008 → Checkpoint0008;
  Training0008 authority cutover → implementation → independent code/docs review → main gate;
  only then may main advance NN0028 planning/readiness.
- Integration validation: focused architecture/API-consumer and CPU regression below, then main's
  one recorded source-matched root `test javadoc` checkpoint for this authority/integration change.
- Shared-document integration owner: main model-execution coordinator; delegates bounded exclusive
  writes to the clean implementation/docs contexts and integrates active-status updates.

Launch verification (2026-10-19): HEAD/common base unchanged; the approved 2267-file snapshot
has no drift, with only five authorized planning deltas from baseline; all source and both protected
files unchanged. Engine0021, Training0007, Compiler0008 and Checkpoint0008 final Complete reviews/
gates verified. Hilbert's independent review of the frozen five documents returned APPROVE with no
findings. Training0008 is the sole authorized Training frontier; NN0028/Training0006/adapter are
inactive, no conflicting writer. Main authorized Ready and will delegate exclusive implementation.
Never edit/stage/revert `THIRD_PARTY_NOTICES.md` or
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`.

## Files and symbols

- `modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java`:
  `run`, new `runValues`, `executeAndDecode`, `bindInputs`, `validateInputStorage`, `bindingPlan`,
  `inputs`, precommit/admission/cleanup. Read adjacent `TrainingParameter` and `TrainingParameters`
  for fixed live/native admission and state invariants; no optimizer algorithm changes expected.
- Existing training tests in the matching test package: `TrainingSessionTest`,
  `TrainingSessionFailureContractTest`, `AdamOptimizerTest`, and fixture `TestParameterSource`.
  Reuse hooks, byte-state comparisons and private-field inspection already used there.
- Read-only `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java#runValues`
  and matching `EngineTypedLifecycleTest`
  establish positional binding and unresolved-layout capacity behavior; no Engine change expected.
- `testing/architecture-tests/src/trainingApiConsumer/java/io/github/pho001/synaptik/testing/architecture/TrainingApiConsumer.java`;
  existing `NnTrainingDependencyContractTest` in that module's test package.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java`:
  one focused real-CPU public workflow regression. Production package stays
  `io.github.pho001.synaptik.training`; no added/moved production type or public test accessor.

## Acceptance criteria

1. Repeated fresh batches AND targets produce independently expected, observably different scalar
   objective, gradient and update outcomes. Observe gradients through accumulation/state or the
   existing oracle, not a new API. Parameter Tensor/storage identities, `compiledGraph()` and the
   single private prepared owner stay fixed; source/private-field evidence and existing seams
   establish reuse without a public prepared accessor. Prototypes remain unchanged; closing a
   completed batch's storage does not affect the next fresh run or detached results.
2. `inputs()` excludes parameters/seed and preserves relative final compiled order. Check correct
   positions and swapped incompatible descriptors; same-descriptor positions are positional, not
   identity-remapped. Reject wrong count, null list/element/mode, dtype/shape/layout presence or
   geometry/requiresGrad mismatch, absent/short/dead/inaccessible storage and storage dtype mismatch
   where constructible. Reuse Engine's checked static geometry for unresolved layouts; do NOT
   strengthen old `run`'s layout-span-or-zero storage checks, diagnostics, admission or precedence.
3. Reject repeated nonempty values, overlapping distinct external slices and parameter aliases
   before execution/update; preserve compiled-order overlap diagnostics and zero-byte treatment.
   Keep fixed live source/path/Tensor/storage validation and native-only parameter admission.
4. Malformed fresh input, non-finite decoded gradient, post-execution/precommit live-binding failure
   and injected later parameter-write failure preserve all parameter/master bytes, slots, pending
   accumulation and counters. Reuse hooks to prove result cleanup and rollback; retry succeeds
   after repairing the test fault. Target-indexed gradient publication mapping stays unchanged.
5. Cover RESET_AND_STEP, ACCUMULATE and ACCUMULATE_AND_STEP with pending-state behavior, plus one
   stateful optimizer continuation that mixes old `run`, fresh `runValues`, `state` and `restore`.
   Preserve default/group SGD/Adam/AdamW and low-precision master/narrowing semantics through the
   shared transaction and existing regressions, not a new dtype × optimizer × mode test matrix.
6. Old `run` still accepts exact IDs in arbitrary order and rejects foreign fresh IDs, duplicates
   and missing inputs. New entry point keeps Engine-closed → session-closed → busy → arguments
   admission; deterministic hooks show an admitted fresh run finishes before close cleanup.
   Result-before-prepared-owner-before-seed cleanup, failure replay and detached lifetimes remain.
7. Isolated Training-only consumer compiles the new method with no NN/back-edge. Focused CPU
   integration proves fresh batches/targets update the same parameters through public Engine;
   no new test framework, backend route or native rebuild is required.

## Validation

Implementation worker: focused development tests, then one final affected Training suite after
executable edits stabilize; final Javadoc after the independent docs pass. Run from repository root:

```bash
./gradlew :extensions:training:test --tests '*TrainingSessionTest' --tests '*TrainingSessionFailureContractTest' --tests '*AdamOptimizerTest' --tests '*TrainingSessionValuesTest' --tests '*TrainingSessionValueAdmissionTest' --tests '*TrainingSessionValueFailureTest'
./gradlew :extensions:training:test :extensions:training:javadoc :testing:architecture-tests:test --tests '*NnTrainingDependencyContractTest'
./gradlew :testing:integration-tests:test --tests '*TrainingSessionCpuIntegrationTest.freshValuesTrainOnCpu'
git diff --check
```

Architecture `test` includes `compileTrainingApiConsumerJava`. Euclid reused successful worker
evidence. Main's accepted source-matched root gate reused the existing Metal dylib, without rebuilding:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/private/tmp/synaptik-checkpoint-leaf.oCW897/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc :testing:architecture-tests:test --rerun :testing:integration-tests:test --rerun --no-build-cache --no-configuration-cache --max-workers=2 --console=plain
```

## Documentation and review impact

Same-change scope: `ARCHITECTURE.md` Core lifecycle; owning
`docs/architecture/contracts/extensions-training.md` modules/training and Optimizer/training
lifecycle; `docs/architecture/lifecycle.md` Public training lifecycle and
`docs/architecture/training-graph.md` Optimizer as a backend-agnostic step / Continuing at the next
training input; `docs/api/training-api.md` construction/input mapping, atomicity and Engine
coordination; `docs/user-guide/training.md` runnable scalar problem, lifetimes and common errors;
ADR0039 Source/shape/state/ownership and cutover status; targeted glossary Training session and
Fresh-value binding entries. Review `docs/architecture/module-boundaries.md` Training heading and
`modules/training/src/main/java/io/github/pho001/synaptik/training/package-info.java`;
update only affected claims or record no impact.
Finalize TrainingSession class/open/inputs/run/runValues and changed private-contract Javadocs,
including every parameter/result/failure and synchronous borrow; no new term/framework is needed.
Use the general and matching documentation profiles and an executable fresh-batch/target example.
Separate clean implementation and independent code/docs contexts are mandatory. Main owns concise
status/link updates in this brief, Training master, program and roadmap after review/acceptance;
preserve completed histories and keep NN0028 TARGET/Draft. No unrelated guides or authority changes.

## Result

- `runValues` shares transaction/captures; old checks retained; no new production type/edge/policy.
- Java: TrainingSession, three Value* tests/fixture, API consumer, CPU regression; docs/Javadoc updated above.
- Commands above PASS: focused 64; Training 93; architecture 2 + consumer compile; CPU 1; skips/failures 0; Javadoc/diff clean.
- Fixture fixes: Model forbids short resolved stores; CPU forbids unresolved compile. Metadata-only test preserves old checks.
- Guide CPU-tested; links checked; package/module-boundary claims accurate.
- Euclid definitive APPROVE/no findings; main accepted. Gate above: exit 0, BUILD SUCCESSFUL 5m10s; 107 tasks, 8 executed/99 up-to-date.
- Fresh architecture 12/0 skips; integration 133/1 skip (CPU fixture lacks local tuning alternatives); both actual Metal training cases PASS.
- Final 19-project reports: 4199 cases, 0 failures/errors, 41 skips (CPU 30/Metal 10 opt-ins/integration 1); not all fresh; Training 93/Javadoc reused.
- HEAD/base unchanged; 2271 files byte-identical through gate; protected changes unchanged; no unresolved issue/follow-up.
Status: Complete

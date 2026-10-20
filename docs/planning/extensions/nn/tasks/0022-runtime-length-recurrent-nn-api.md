# Task NN0022 Runtime length recurrent NN expressions

## Status

Complete — accepted 2026-10-20 after Mencius final APPROVE, zero findings, and coordinator validation.
Launch verification: main checked completed prerequisites, unchanged named contracts, recorded
HEAD/main/origin/main base and inactive conflicts; user authorized implementation on 2026-10-20.
NN/CPU/Data frontiers: None. NN0022A/0023/0024 remain Draft; no automatic successor launch.

## Change class

Class B — supported NN API and first-scan initialization using existing cell-local publication.
Use clean implementation and independent code/docs review. Stop and reclassify before changing
locking, ownership, dependencies or execution lifecycles.

## Goal and scope

Construct one fixed RNN/GRU/LSTM scan from an NN-owned cell, runtime length Tensor and explicit
carried state. Return dense original-time-aligned `[T,B,H]` output and typed final states; reuse
public Engine across lengths with compatible descriptors. Preserve every `long[]` compact/static
API, provenance and supported static training. Data integration is successor NN0022A, not a prerequisite.

## Non-goals

No Data module/type/edge, mask, BPTT (backpropagation through time), builder/bidirectional migration,
Model.compute/train extension, runner/session, lifecycle wrapper, persistence/backend widening,
numerical-policy change, tuning or performance campaign. No zero-state/all-valid overload family;
callers supply lengths and all recurrent states explicitly.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then read:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals).
- Full [Fixed recurrent scan without graph regions](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions):
  family/surface, static Shape/runtime values, traversal/outputs, lifecycle, performance and migration.
- [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn) and
  [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  existing parameter ownership and list Model boundary, not a new graph/execution pipeline.
- [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution):
  exact positional descriptors, borrowed storage, existing prepared/result owners and closure.

Stop on missing/ambiguous/conflicting contracts; report the heading. No authority or ADR change.

## Dependencies and integration

- Depends on: NN0020/0020C (cell initialization and static compatibility), NN0021B + CPU0028
  (accepted forward baseline), Model0025E/0025F (fixed family/namespace), Compiler0006A
  (flat forward capture and backward rejection), Engine0021 (fresh positional values); all Complete.
  CP0010/CP0011/CPU0027 remain completed history, not new gates; no Data dependency.
- Conflicts with: NN0022A/0023/0024, held NN0027/CP0005C, or any writer to NN recurrent API,
  cell initialization/publication, owning contracts, or the shared documents below; serialize.
- Parallel group: None.
- Common base revision: `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a` (main/origin/main).
- Integration order: choices accepted → clean implementation → independent code/docs review → main integration.
- Integration validation: NN module test/Javadoc plus RuntimeLengthRecurrentNnIntegrationTest,
  below; reuse CPU0028/NN0021B backend/root evidence unless concrete cross-module change merits a gate.
- Shared-document integration owner: main coordinator; NN/Data plans, roadmap and shared docs integrate serially.

## Pre-implementation source facts and affected files

- NN `layers/{Rnn,Gru,Lstm}Sequence.java` owns the matching cell and compact static results.
- `{Rnn,Gru,Lstm}Cell.java` owns automatic initialization; `module/Module.java` owns atomic reserved
  binding. Reuse their mechanisms and current cell monitor; replacement/load stays caller-coordinated.
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/RecurrentScan.java`:
  six biased/bias-free methods; static ranks/extents, exact common floating dtype without promotion,
  non-gradient INT64 lengths, canonical producer slots, unresolved layouts and floating-input flag OR.
- NN `module/{Model,ModelExecutionState,ModelBuilder,Parameter}.java`: generic forward and owned list
  graph inputs do not grant scan compute authority; builder recurrence stays static/all-valid/zero-state.
- Baseline tests: NN `layers/{Rnn,Gru,Lstm}{Sequence,SequencePacking,CellInitialization}Test.java`,
  Model `RecurrentScanExpressionTest`, Compiler `RecurrentScanCompilerTest`, integration
  `EngineCpuRecurrentScanIntegrationTest`. Preserve assertions except additive Sequence API expectations.

## Accepted API and decisions — 2026-10-20

Accepted unchanged from the reviewed draft: methods on existing concrete Sequences, with these signatures:

```java
RecurrentScanResult RnnSequence.forwardScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, RecurrentDirection direction);
RecurrentScanResult GruSequence.forwardScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, RecurrentDirection direction);
LstmRecurrentScanResult LstmSequence.forwardScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, Tensor initialCell,
        RecurrentDirection direction);
```

Reuse `model.tensor.RecurrentScanResult`, `LstmRecurrentScanResult`,
`model.operation.recurrent.RecurrentDirection`, and current NN Parameter wrappers/`cell.*` state paths.
New production types/packages: None. Keep compact NN carriers unchanged. Use package-private
`initializeForScanIfNeeded` per Cell with its Sequence method's Tensor/direction arguments and private
descriptor checks reusing `AutomaticInput`; no shared abstraction, public binding API or new record.

Main accepts placement/name/order, current Model carriers, explicit states/direction and descriptor-only
first-scan automatic initialization even at T/B=0. Direct public Engine is the sole execution acceptance;
`Owned tensor Models` grants no scan compute/train support. Model.compute/train, BPTT, Data and builder
widening remain excluded. All construction/initialization rules below are accepted; no new concurrency guarantee.

## Initialization and forward construction

1. Check null arguments in signature order. Before draws, Tensor identifiers or publication,
   validate the complete scan descriptor schema against configured/prospective cell parameters:
   static `[T,B,I]`, `T/B >= 0`, `I/H > 0`, states `[B,H]`, exact configured floating dtype,
   lengths non-gradient `INT64[B]`, non-null direction, packed gate counts 1/3/4, prospective
   weight/bias Shapes and checked initialization array/count limits. Never read length values,
   inspect host storage, use promotion/broadcast as an exact-type substitute, or create a dummy
   cell forward, select, eager index, temporary Tensor or discarded scan to trigger initialization.
2. For unbound automatic cells, recheck under the existing cell monitor and reuse
   `initializeAutomatically(AutomaticInput)`/`bindReservedParameters` for the same reserved group.
   Random policies restart L64X128MixRandom from the seed per attempt; input then hidden matrix,
   then zero bias; zero/one create no generator. Ordinary pre-publication failure leaves reservations
   retryable, not rolled-back allocations/IDs; later scan failure keeps published parameters.
   A competing first forward uses the same gate/monitor.
3. Descriptor-valid scan construction binds even when T or B is zero, or future runtime lengths
   are all zero: one scan still needs parameter inputs. This differs deliberately from legacy
   unrolling's no-represented-step behavior. Strict-loaded or already-bound cells never reinitialize;
   snapshot current parameter bindings once in declaration order and delegate to the matching
   biased/bias-free Model constructor. Recheck exact current schema without changing wrappers.
4. Return the Model carrier unchanged. Accept ordinary produced input/state/length/parameter
   provenance; no copies, detach, leaf-only guards or requiresGrad clearing. NN parameters remain
   gradient-eligible, so scan output flags are true even in eval; this does not authorize BPTT.
   Both modes construct identical meaning. The module retains no input, lengths, recurrent state or
   result; the producer retains ordered inputs. Replacement is caller-coordinated.

## Acceptance criteria and example

- Test all families, both directions, bias/no-bias, supplied/eager/automatic/strict-loaded cells,
  exact input references/ports/flags, mode invariance, replacement affecting only later occurrences,
  arbitrary upstream provenance, and a constant one-producer shape independent of T/length values.
- Invalid null/type/rank/static/extents/length flags/direction and initialization count overflow
  fail before initialization effects. Cover successful first-scan atomic publication, compatible
  concurrent first scan/ordinary forward, failed initialization retry, no dummy-expression IDs,
  empty T/B and all-zero runtime lengths without weakening existing static no-op tests.
- Public Engine integration: canonical FLOAT32 with ordinary trainable NN parameter flags; prepare
  once, run two fresh exact-descriptor length patterns, preserve previous results, reject out-of-range
  lengths before scan publication, then repair/reuse. Include produced lengths, padded NaN input,
  reverse valid-prefix alignment, signed-zero initial-state preservation and explicit LSTM carry.
- Verify dense `[T,B,H]` and `[B,H]` final states, positive-zero padding, zero-length semantic
  initial-state preservation (not Java Tensor-reference identity), and fail-closed backward requests.
  CPU0028 supports canonical FLOAT32 scalar execution, not other dtypes/layouts/routes; Model-level
  expression validity must not be narrowed to that backend or described as execution coverage.

Conceptual API example: automatic H=4/FLOAT32/bias/Glorot/seed42; `x: [3,2,2]`, non-grad `lengths: INT64[2]`, `h0/c0: [2,4]`.

```java
var r = rnn.forwardScan(x, lengths, h0, RecurrentDirection.FORWARD);
var g = gru.forwardScan(x, lengths, h0, RecurrentDirection.REVERSE);
var l = lstm.forwardScan(x, lengths, h0, c0, RecurrentDirection.FORWARD);
var next = lstm.forwardScan(nextX, nextLengths, l.finalHidden(), l.finalCell(),
        RecurrentDirection.FORWARD); // explicit symbolic carry, never module state
```

Each result has dense `[3,2,4]` outputs and `[2,4]` final states. For runtime lengths `[3,1]`,
row 1 at times 1/2 is positive zero in either direction; reverse row 0 consumes 2,1,0 and stores
at original coordinates. Separately compile the selected result Tensors with existing Engine,
prepare once and use runValues with values in `compiled.inputs()` order, including weights and
states. Replace lengths with `[0,2]` using the same full descriptors, without recomposition;
row 0 then preserves h0/c0 semantically. `[4,1]` fails; repair and reuse. Separate-run continuation
requires caller-supplied materialized state values matching the next graph's full descriptors.

## Validation and documentation review

Executed with JDK 26.0.1; commands below are evidence references, not instructions to rerun.

```bash
./gradlew :extensions:nn:test --tests '*RuntimeLengthRecurrentSequenceTest' --tests '*RnnSequenceTest' --tests '*GruSequenceTest' --tests '*LstmSequenceTest' --tests '*CellInitializationTest'
./gradlew :extensions:nn:test :extensions:nn:javadoc
./gradlew :testing:integration-tests:test --tests '*RuntimeLengthRecurrentNnIntegrationTest'
```

Worker owns focused/module evidence; main owns integration; independent review reuses passing evidence.
CPU0028/NN0021B backend/root evidence reused; no new root/conformance/performance campaign.
Six Sequence/Cell Javadocs cover scan/static distinction, descriptor validation, initialization,
state/provenance, flags and failures; legacy contracts reviewed and unchanged where still accurate.
Independent code/docs review applies general, API/user and planning documentation profiles.

## Result

- Implemented explicit-length/state/direction `forwardScan`, existing carriers and atomic cell initialization; static `long[]` APIs/training preserved.
- Java files (11): NN `layers/{Rnn,Gru,Lstm}{Cell,Sequence}.java` (6), existing
  `layers/{Rnn,Gru,Lstm}SequenceTest.java` (3), new `RuntimeLengthRecurrentSequenceTest.java`
  (76 cases), and new integration `RuntimeLengthRecurrentNnIntegrationTest.java` (19 cases).
- Above focused command: 147 passed (including 76 new), zero failures/errors/skips;
  full NN test/Javadoc command: PASS, 506 tests/56 suites, zero failures/errors/skips.
- Main's integration-only command: PASS, 19 tests, zero failures/errors/skips; BUILD SUCCESSFUL in 1s, 38 tasks (2 executed/36 UP-TO-DATE).
- Post-review change only: concurrency-test cleanup releases latch in finally before executor close.
  `./gradlew :extensions:nn:test --tests '*RuntimeLengthRecurrentSequenceTest.compatibleConcurrentFirstScanAndForwardShareOneAtomicGroup'`:
  PASS, 3 affected cases, zero failures/errors/skips. Production, integration and other API tests
  remain byte-exact; prior full/focused/integration evidence reused, not claimed as freshly rerun.
- Mencius final APPROVE, zero findings; final 11-Java SHA-256: `a2dcef0b35c74932f6c7dd8c9ad9e5ad792976e4bc0548362e9cef417ea6821c`.
- Documentation: `docs/api/{public-api,training-api}.md`, `docs/user-guide/running-models.md`,
  `docs/architecture/module-boundaries.md`, `docs/glossary.md` approved/frozen; six Java Javadocs
  passed generation, no later Javadoc edits. Glossary clarifies automatic cells, static packing,
  fixed scans and valid-length Tensors; no new term. Four planning files synchronized at closure.
- Main reports 431 local Markdown links across five docs PASS, diff check PASS, protected hashes
  unchanged and index empty. Authority, backend/build, dirty notices/oracle and unrelated files preserved.
- Limits: canonical FLOAT32 CPU Engine execution only; no Data/new production type, Model.compute/train,
  builder or BPTT widening. No unresolved issue or required follow-up within NN0022. Status: Complete.

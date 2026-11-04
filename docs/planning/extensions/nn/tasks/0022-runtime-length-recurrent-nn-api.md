# Task NN0022 [Runtime](../../../../glossary.md#term-runtime) length recurrent [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) expressions

## Status

Complete — accepted 2026-10-20 after Mencius final APPROVE, zero findings, and coordinator [validation](../../../../glossary.md#term-validation).
Launch verification: main checked completed prerequisites, unchanged named contracts, recorded
HEAD/main/origin/main base and inactive conflicts; user authorized implementation on 2026-10-20.
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../../../../glossary.md#term-cpu)/Data frontiers: None. NN0022A/0023/0024 remain Draft; no automatic successor launch.

## Change class

Class B — supported [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [API](../../../../glossary.md#term-application-programming-interface-api) and first-[scan](../../../../glossary.md#term-cumulative-scan) initialization using existing cell-local [publication](../../../../glossary.md#term-publication).
Use clean implementation and independent code/docs review. Stop and reclassify before changing
locking, [ownership](../../../../glossary.md#term-ownership-meanings), dependencies or execution lifecycles.

## Goal and [scope](../../../../glossary.md#term-scope)

Construct one fixed [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) [scan](../../../../glossary.md#term-cumulative-scan) from an [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned cell, [runtime](../../../../glossary.md#term-runtime) length [Tensor](../../../../glossary.md#tensor) and explicit
carried state. Return dense original-time-aligned `[T,B,H]` output and typed final states; reuse
public [Engine](../../../../glossary.md#term-engine) across lengths with compatible [descriptors](../../../../glossary.md#term-descriptor). Preserve every `long[]` compact/static
[API](../../../../glossary.md#term-application-programming-interface-api), [provenance](../../../../glossary.md#provenance) and supported static training. Data integration is successor NN0022A, not a prerequisite.

## Non-goals

No Data [module](../../../../glossary.md#term-module)/type/edge, [mask](../../../../glossary.md#term-boolean-mask), [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) ([backpropagation through time](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)), builder/bidirectional migration,
[Model](../../../../glossary.md#term-model).compute/train extension, runner/session, [lifecycle](../../../../glossary.md#lifecycle) wrapper, [persistence](../../../../glossary.md#term-persistence)/[backend](../../../../glossary.md#backend) [widening](../../../../glossary.md#term-widening),
numerical-policy change, tuning or performance campaign. No zero-state/all-valid overload family;
callers supply lengths and all recurrent states explicitly.

## Contracts

Start with [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then read:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals).
- Full [Fixed recurrent scan without graph regions](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [graph glossary](../../../../glossary.md#graph))</span>:
  family/surface, static Shape/runtime values, traversal/outputs, lifecycle, performance and migration.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn) and
  [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>:
  existing parameter ownership and list Model boundary, not a new graph/execution pipeline.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Model glossary](../../../../glossary.md#term-model))</span>
- [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>:
  exact positional descriptors, borrowed storage, existing prepared/result owners and closure.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

Stop on missing/ambiguous/conflicting contracts; report the heading. No authority or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) change.

## Dependencies and integration

- Depends on: NN0020/0020C (cell initialization and static compatibility), NN0021B + CPU0028
  (accepted forward baseline), Model0025E/0025F (fixed family/namespace), Compiler0006A
  (flat forward capture and backward rejection), Engine0021 (fresh positional values); all Complete.
  CP0010/CP0011/CPU0027 remain completed history, not new gates; no Data dependency.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: NN0022A/0023/0024, held NN0027/CP0005C, or any writer to NN recurrent API,
  cell initialization/publication, owning contracts, or the shared documents below; serialize.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication))</span>
- Parallel group: None.
- Common base revision: `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a` (main/origin/main).
- Integration order: choices accepted → clean implementation → independent code/docs review → main integration.
- Integration validation: NN module test/Javadoc plus RuntimeLengthRecurrentNnIntegrationTest,
  below; reuse CPU0028/NN0021B backend/root evidence unless concrete cross-module change merits a gate.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordinator; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Data plans, roadmap and shared docs integrate serially.

## Pre-implementation source facts and affected files

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `layers/{Rnn,Gru,Lstm}Sequence.java` owns the matching cell and compact static results.
- `{Rnn,Gru,Lstm}Cell.java` owns automatic initialization; `module/Module.java` owns atomic reserved
  binding. Reuse their mechanisms and current cell monitor; replacement/load stays caller-coordinated.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding))</span>
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/RecurrentScan.java`:
  six biased/bias-free methods; static ranks/extents, exact common floating dtype without promotion,
  non-gradient INT64 lengths, canonical producer slots, unresolved layouts and floating-input flag OR.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [dtype glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [INT64 glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form); [producer glossary](../../../../glossary.md#term-producer); [layouts glossary](../../../../glossary.md#layout))</span>
- NN `module/{Model,ModelExecutionState,ModelBuilder,Parameter}.java`: generic forward and owned list
  graph inputs do not grant scan compute authority; builder recurrence stays static/all-valid/zero-state.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Baseline tests: NN `layers/{Rnn,Gru,Lstm}{Sequence,SequencePacking,CellInitialization}Test.java`,
  Model <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>`RecurrentScanExpressionTest`, [Compiler](../../../../glossary.md#term-compiler) `RecurrentScanCompilerTest`, integration
  `EngineCpuRecurrentScanIntegrationTest`. Preserve assertions except additive Sequence [API](../../../../glossary.md#term-application-programming-interface-api) expectations.

## Accepted [API](../../../../glossary.md#term-application-programming-interface-api) and decisions — 2026-10-20

Accepted unchanged from the reviewed draft: methods on existing concrete Sequences, with these [signatures](../../../../glossary.md#term-signature):

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
`model.operation.recurrent.RecurrentDirection`, and current [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [Parameter](../../../../glossary.md#term-parameter) wrappers/`cell.*` state paths.
New production types/packages: None. Keep compact NN carriers unchanged. Use package-private
`initializeForScanIfNeeded` per Cell with its Sequence method's [Tensor](../../../../glossary.md#tensor)/direction [arguments](../../../../glossary.md#term-argument) and private
[descriptor](../../../../glossary.md#term-descriptor) checks reusing `AutomaticInput`; no shared abstraction, public [binding](../../../../glossary.md#term-binding) API or new record.

Main accepts placement/name/order, current [Model](../../../../glossary.md#term-model) carriers, explicit states/direction and descriptor-only
first-[scan](../../../../glossary.md#term-cumulative-scan) automatic initialization even at T/B=0. Direct public [Engine](../../../../glossary.md#term-engine) is the sole execution acceptance;
`Owned tensor Models` grants no scan compute/train support. Model.compute/train, [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), Data and builder
[widening](../../../../glossary.md#term-widening) remain excluded. All construction/initialization rules below are accepted; no new [concurrency](../../../../glossary.md#term-concurrency) guarantee.

## Initialization and forward construction

1. Check null arguments in signature order. Before draws, Tensor identifiers or publication,
   validate the complete scan descriptor schema against configured/prospective cell parameters:
   static <span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [signature glossary](../../../../glossary.md#term-signature); [Tensor glossary](../../../../glossary.md#tensor); [publication glossary](../../../../glossary.md#term-publication); [scan glossary](../../../../glossary.md#term-cumulative-scan); [descriptor glossary](../../../../glossary.md#term-descriptor); [schema glossary](../../../../glossary.md#term-schema); [parameters glossary](../../../../glossary.md#term-parameter))</span>`[T,B,I]`, `T/B >= 0`, `I/H > 0`, states `[B,H]`, exact configured floating dtype,
   lengths non-gradient <span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>`INT64[B]`, non-null direction, packed gate counts 1/3/4, prospective
   weight/bias Shapes and checked initialization array/count limits. Never read length values,
   inspect host storage, use promotion/broadcast as an exact-type substitute, or create a dummy
   cell forward, select, eager index, temporary Tensor or discarded scan to trigger initialization.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [Shapes glossary](../../../../glossary.md#shape); [host storage glossary](../../../../glossary.md#host-storage); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
2. For unbound automatic cells, recheck under the existing cell monitor and reuse
   `initializeAutomatically(AutomaticInput)`/`bindReservedParameters` for the same reserved group.
   Random policies restart L64X128MixRandom from the seed per attempt; input then hidden matrix,
   then zero bias; zero/one create no generator. Ordinary pre-publication failure leaves reservations
   retryable, not rolled-back allocations/IDs; later scan failure keeps published parameters.
   A competing first forward uses the same gate/monitor.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
3. Descriptor-valid scan construction binds even when T or B is zero, or future runtime lengths
   are all zero: one scan still needs parameter inputs. This differs deliberately from legacy
   unrolling's no-represented-step behavior. Strict-loaded or already-bound cells never reinitialize;
   snapshot current parameter bindings once in declaration order and delegate to the matching
   biased/bias-free Model constructor. Recheck exact current schema without changing wrappers.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [unrolling glossary](../../../../glossary.md#term-unrolling); [snapshot glossary](../../../../glossary.md#term-snapshot); [bindings glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model))</span>
4. Return the Model carrier unchanged. Accept ordinary produced input/state/length/parameter
   provenance; no copies, detach, leaf-only guards or requiresGrad clearing. NN parameters remain
   gradient-eligible, so scan output flags are true even in eval; this does not authorize BPTT.
   Both modes construct identical meaning. The module retains no input, lengths, recurrent state or
   result; the producer retains ordered inputs. Replacement is caller-coordinated.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [provenance glossary](../../../../glossary.md#provenance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [module glossary](../../../../glossary.md#term-module); [recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [producer glossary](../../../../glossary.md#term-producer))</span>

## Acceptance criteria and example

- Test all families, both directions, bias/no-bias, supplied/eager/automatic/strict-loaded cells,
  exact input references/ports/flags, mode invariance, replacement affecting only later occurrences,
  arbitrary upstream provenance, and a constant one-producer shape independent of T/length values.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [provenance glossary](../../../../glossary.md#provenance); [constant glossary](../../../../glossary.md#term-constant); [producer glossary](../../../../glossary.md#term-producer); [shape glossary](../../../../glossary.md#shape))</span>
- Invalid null/type/rank/static/extents/length flags/direction and initialization count overflow
  fail before initialization effects. Cover successful first-scan atomic publication, compatible
  concurrent first scan/ordinary forward, failed initialization retry, no dummy-expression IDs,
  empty T/B and all-zero runtime lengths without weakening existing static no-op tests.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [overflow glossary](../../../../glossary.md#term-overflow); [scan glossary](../../../../glossary.md#term-cumulative-scan); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>
- Public Engine integration: canonical FLOAT32 with ordinary trainable NN parameter flags; prepare
  once, run two fresh exact-descriptor length patterns, preserve previous results, reject out-of-range
  lengths before scan publication, then repair/reuse. Include produced lengths, padded NaN input,
  reverse valid-prefix alignment, signed-zero initial-state preservation and explicit LSTM carry.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication); [NaN glossary](../../../../glossary.md#term-nan); [alignment glossary](../../../../glossary.md#term-alignment); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [carry glossary](../../../../glossary.md#term-carry))</span>
- Verify dense `[T,B,H]` and `[B,H]` final states, positive-zero padding, zero-length semantic
  initial-state preservation (not Java Tensor-reference identity), and fail-closed backward requests.
  CPU0028 supports canonical FLOAT32 scalar execution, not other dtypes/layouts/routes; Model-level
  expression validity must not be narrowed to that backend or described as execution coverage.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [reference identity glossary](../../../../glossary.md#term-reference-identity); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [scalar glossary](../../../../glossary.md#term-scalar); [dtypes glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend))</span>

Conceptual [API](../../../../glossary.md#term-application-programming-interface-api) example: automatic H=4/[FLOAT32](../../../../glossary.md#term-float32)/[bias](../../../../glossary.md#term-bias)/[Glorot](../../../../glossary.md#term-glorot-uniform)/seed42; `x: [3,2,2]`, non-grad `lengths: INT64[2]`, `h0/c0: [2,4]`.

```java
var r = rnn.forwardScan(x, lengths, h0, RecurrentDirection.FORWARD);
var g = gru.forwardScan(x, lengths, h0, RecurrentDirection.REVERSE);
var l = lstm.forwardScan(x, lengths, h0, c0, RecurrentDirection.FORWARD);
var next = lstm.forwardScan(nextX, nextLengths, l.finalHidden(), l.finalCell(),
        RecurrentDirection.FORWARD); // explicit symbolic carry, never module state
```

Each result has dense `[3,2,4]` outputs and `[2,4]` final states. For [runtime](../../../../glossary.md#term-runtime) lengths `[3,1]`,
row 1 at times 1/2 is [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) in either direction; reverse row 0 consumes 2,1,0 and stores
at original [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). Separately [compile](../../../../glossary.md#compile) the selected result [Tensors](../../../../glossary.md#tensor) with existing [Engine](../../../../glossary.md#term-engine),
[prepare](../../../../glossary.md#prepare) once and use runValues with values in `compiled.inputs()` order, including [weights](../../../../glossary.md#term-weight) and
states. Replace lengths with `[0,2]` using the same full [descriptors](../../../../glossary.md#term-descriptor), without recomposition;
row 0 then preserves h0/c0 semantically. `[4,1]` fails; repair and reuse. Separate-[run](../../../../glossary.md#term-run) continuation
requires caller-supplied materialized state values matching the next [graph](../../../../glossary.md#graph)'s full descriptors.

## [Validation](../../../../glossary.md#term-validation) and documentation review

Executed with [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1; commands below are evidence references, not instructions to rerun.

```bash
./gradlew :extensions:nn:test --tests '*RuntimeLengthRecurrentSequenceTest' --tests '*RnnSequenceTest' --tests '*GruSequenceTest' --tests '*LstmSequenceTest' --tests '*CellInitializationTest'
./gradlew :extensions:nn:test :extensions:nn:javadoc
./gradlew :testing:integration-tests:test --tests '*RuntimeLengthRecurrentNnIntegrationTest'
```

Worker owns focused/[module](../../../../glossary.md#term-module) evidence; main owns integration; independent review reuses passing evidence.
CPU0028/NN0021B [backend](../../../../glossary.md#backend)/root evidence reused; no new root/[conformance](../../../../glossary.md#term-conformance)/performance campaign.
Six Sequence/Cell Javadocs cover [scan](../../../../glossary.md#term-cumulative-scan)/static distinction, [descriptor](../../../../glossary.md#term-descriptor) validation, initialization,
state/[provenance](../../../../glossary.md#provenance), flags and failures; legacy contracts reviewed and unchanged where still accurate.
Independent code/docs review applies general, [API](../../../../glossary.md#term-application-programming-interface-api)/user and [planning](../../../../glossary.md#term-planning) documentation profiles.

## Result

- Implemented explicit-length/state/direction `forwardScan`, existing carriers and [atomic](../../../../glossary.md#resource-transaction-and-rollback) cell initialization; static `long[]` APIs/training preserved.
- Java files (11): [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `layers/{Rnn,Gru,Lstm}{Cell,Sequence}.java` (6), existing
  `layers/{Rnn,Gru,Lstm}SequenceTest.java` (3), new `RuntimeLengthRecurrentSequenceTest.java`
  (76 cases), and new integration `RuntimeLengthRecurrentNnIntegrationTest.java` (19 cases).
- Above focused command: 147 passed (including 76 new), zero failures/errors/skips;
  full NN test/Javadoc command: PASS, 506 tests/56 suites, zero failures/errors/skips.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Main's integration-only command: PASS, 19 tests, zero failures/errors/skips; BUILD SUCCESSFUL in 1s, 38 tasks (2 executed/36 UP-TO-DATE).
- Post-review change only: [concurrency](../../../../glossary.md#term-concurrency)-test cleanup [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) [latch](../../../../glossary.md#control-flow-graph) in finally before executor close.
  `./gradlew :extensions:nn:test --tests '*RuntimeLengthRecurrentSequenceTest.compatibleConcurrentFirstScanAndForwardShareOneAtomicGroup'`:
  PASS, 3 affected cases, zero failures/errors/skips. Production, integration and other API tests
  remain byte-exact; prior full/focused/integration evidence reused, not claimed as freshly rerun.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Mencius final APPROVE, zero findings; final 11-Java [SHA-256](../../../../glossary.md#term-sha-256): `a2dcef0b35c74932f6c7dd8c9ad9e5ad792976e4bc0548362e9cef417ea6821c`.
- Documentation: `docs/api/{public-api,training-api}.md`, `docs/user-guide/running-models.md`,
  `docs/architecture/module-boundaries.md`, `docs/glossary.md` approved/frozen; six Java Javadocs
  passed generation, no later Javadoc edits. Glossary clarifies automatic cells, static packing,
  fixed scans and valid-length Tensors; no new term. Four planning files synchronized at closure.<span data-glossary-navigation="true"> ([Tensors glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Main reports 431 local Markdown links across five docs PASS, diff check PASS, protected hashes
  unchanged and index empty. Authority, backend/build, dirty notices/oracle and unrelated files preserved.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Limits: canonical FLOAT32 CPU Engine execution only; no Data/new production type, Model.compute/train,
  builder or BPTT widening. No unresolved issue or required follow-up within NN0022. Status: Complete.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../../glossary.md#term-float32); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [widening glossary](../../../../glossary.md#term-widening))</span>

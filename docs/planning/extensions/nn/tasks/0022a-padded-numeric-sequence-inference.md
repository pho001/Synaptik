# Task NN0022A Padded numeric sequence inference

## Status

Complete. Main accepted on 2026-10-21 after independent APPROVE and the source-matched
architecture/root test/Javadoc checkpoint. No successor is authorized.

## Change class

Class C: public Data export and a cross-module input/lifecycle boundary. Separate clean
implementation and independent targeted API/lifecycle/documentation review are required.

## Goal

Run already padded numeric FLOAT32 sequences through existing RNN/GRU/LSTM scan Models,
reusing one preparation for fresh values and changing valid counts at the same static signature.
Data exports counts; the application passes ordinary Tensors to existing Model.compute; Engine
retains preparation and runtime-binding ownership.

## Scope and approved API

First slice: RIGHT padding, TIME_MAJOR values, canonical contiguous offset-zero non-view
FLOAT32 `[T,B,F]`, requiresGrad=false, with positive static T/B/F for the fluent builder.
Applications already holding numeric tensors wrap them with PaddedSequenceBatch; no new float
batcher is required. Data0005 remains a separate future capability.

Add one method on existing Data record PaddedSequenceBatch, in existing package
`io.github.pho001.synaptik.data`: `public Tensor toScanLengthsTensor()`.
It exports a fresh heap-backed, provenance-free, canonical contiguous offset-zero non-view INT64
`[B]` leaf with requiresGrad=false on every call, including B=0. Allocate through public
TensorFactory.allocate and fill public Model host storage from immutable counts in original row
order. O(B) allocation/copy is explicit; no cached mutable length Tensor or new resource owner.

The method checks paddingSide first, then axisOrder, throwing IllegalArgumentException for LEFT
or BATCH_MAJOR before Tensor creation, allocation or storage access. It never reads values storage,
changes values descriptors/flags, transposes, sorts, repacks or copies sequence contents. Existing
constructor geometry validation remains unchanged. Export does not validate numeric execution
support: the application must supply the exact Model input descriptor. Even storage-free values
can export counts; only compute requires live supported input storage.

Counts describe valid intervals, not padding sentinels or graph bindings. The exported Tensor is
a mutable ordinary Model value independent of the immutable metadata; editing one export cannot
change counts or another export. Callers retain it for the synchronous run and follow existing
Model storage obligations; no close protocol, native allocation or device transfer is added.

The new method only materializes counts. Existing two-input compute binds `[values, lengths]`;
it neither accepts Data objects nor infers side/order. RNN and LSTM use their existing scan recipes.
Omitting finalHidden returns dense `[T,B,H]`, not a compact sequence or the final valid timestep.
Expression output layout remains unresolved under Model semantics; execution uses the existing
qualified canonical CPU route, without changing public scan port order or descriptor inference.

## Approval and launch verification

On 2026-10-21 main recorded exact user approval: “ok,pustse do toho”, for this method,
failure/lifetime contract and numeric-only implementation. Root Data summary/routing, the owning
[Data contract](../../../../architecture/contracts/extensions-training.md#extensionsdata),
Data explanation and [ADR0044](../../../../design/decisions/0044-opt-in-scan-length-tensor-export.md)
now enact that narrow permission. NN must not depend on Data/Text; Data uses only Model.
External consumers compose exported public APIs, not a production bridge.

Numeric-only acceptance is approved independently of Text. ModelBuilder.input currently accepts only
floating types, has no Embedding recipe, and instantiate passes the original input dtype to every
recipe. Therefore a Data length export cannot enable INT64 Text IDs → Embedding → scan in one
fluent Model. A separately selected NN builder integer-input/Embedding capability must establish
typed dtype transitions before that full Text workflow can be promised; it is not part of NN0022A.

Main verified HEAD=origin/main `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`, completed dependencies
NN0022/0022B/0028/0029 and Data0001–0003, and inactive conflicts. Data/Text predecessors are
committed/pushed. The sole clean implementation executor used that verified packet and has closed.
Independent review/documentation and main's source-matched acceptance checkpoint are complete.
Protected unrelated dirty files, never edit:
`THIRD_PARTY_NOTICES.md` SHA256 `8efe7e3c1f83bd5b30b913d447fa3ee95f9c0ee60485c30de7c0008638e7037f`;
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`
SHA256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842`.

## Non-goals

No generic input protocol/interface/facade, new production type/package, runner/session, buffer
owner, hidden sequence copy/repack, arbitrary-shape reuse, LEFT scan, BATCH_MAJOR conversion,
automatic float batching, Text pipeline, BPTT/training, native route, tuning, performance campaign
or save-format change. Existing advanced zero-T/B scans remain supported but do not widen the
builder's positive-input contract. Existing gradient eligibility is preserved, never cleared to
bypass inference-only admission.

## Contracts

- Root [Repository layout](../../../../../ARCHITECTURE.md#repository-layout),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- Extensions/training [extensions/data](../../../../architecture/contracts/extensions-training.md#extensionsdata),
  [extensions/text](../../../../architecture/contracts/extensions-training.md#extensionstext),
  [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models).
- Recurrent scan [Static Shape and runtime-value boundary](../../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary),
  [Valid lengths, traversal, and outputs](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs),
  [Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership).

Stop on missing/ambiguous authority; plans do not override current contracts.

## Dependencies and integration

- Depends on: NN0022/0022B/0028/0029 and Data0001–0003 Complete; approval recorded above.
  Text0001 Complete is context, not a numeric prerequisite.
- Conflicts with: Data public API/ownership, NN Model input contracts and shared authority/docs;
  serialize any changes to these scopes. Protected unrelated dirty files remain out of scope.
- Parallel group: None.
- Common base revision: `6eabea0c1ebf77b9bd8382855c7131eb89b52e04`.
- Integration order: approved contract packet → Data export/tests/docs → external numerical
  integration → independent targeted Class C review → coordinator acceptance.
- Integration validation: worker affected Data and focused unchanged NN reuse tests; coordinator
  external-consumer/public Engine reuse, architecture and root test/Javadoc checkpoint once below.
- Shared-document integration owner: main coordinator; this clean documentation context exclusively
  owns shared authority/docs/plans during cutover, then hands them back to main.

No new Data/Text/CPU frontier or successor is authorized; NN0023/0024 remain Draft.

## Files and symbols

- `extensions/data/…/data/PaddedSequenceBatch.java` — only approved production method;
  `ValidSequenceLengths.java` and `LongSequenceBatcher.java` — existing count/allocation references.
- `extensions/nn/…/nn/module/ModelBuilder.java`, `Model.java` — existing scan recipes,
  buildWithLengths and compute(firstInput,secondInput); no production NN edit authorized.
- `extensions/data/src/test/java/io/github/pho001/synaptik/data/PaddedSequenceBatchTest.java`
  — update exact API assertions and export tests.
- Existing NN `RuntimeLengthScanModelTest`, `RuntimeLengthRecurrentSequenceTest` and external
  `RuntimeLengthScanModelIntegrationTest` — focused behavior/oracle references.
- `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`
  — narrow public export witness through Data's existing Model API export; no new dependency.
- New external test `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PaddedNumericSequenceInferenceIntegrationTest.java`.
  Existing Text API export exposes Data to that consumer; no new project dependency is proposed.

## Acceptance criteria

- Exact export descriptor, row ordering, empty B, fresh identity/storage and mutation isolation
  are tested. Metadata and borrowed values remain unchanged; no values storage inspection occurs.
- Architecture DataApiConsumer compiles/uses the exact new public export through existing edges.
- LEFT and BATCH_MAJOR reject deterministically before allocation/storage work. Canonical numeric
  inputs execute without hidden repack; incompatible compute descriptors reject under existing
  Model rules, without implicit dtype/layout/gradient conversion.
- External consumer uses only public Data/NN/Model/Engine APIs. For all three scan families,
  fresh `[T,B,F]` values and counts `[3,1,0]`, `[0,0,0]`, `[1,3,2]` reuse the same Model preparation;
  independent equations verify dense and final-hidden results, positive-zero padded outputs,
  zero-length initial states and original row order, including NaN sentinels outside valid prefixes.
- Demonstrate Engine.prepare once and repeated Engine.runValues on that same PreparedExecution
  with fresh Data exports. Retain existing NN preparation-identity evidence without introducing
  Data into NN tests. No graph rebuild, per-pattern specialization or hidden state carry.
- Output expression gradient eligibility remains the OR of floating roles; exported lengths
  never contribute. Default trainable parameters may make outputs gradient-eligible even though
  values/lengths are not; detached compute publication does not imply BPTT support.

## Validation

Successful worker commands (reuse without rerunning):

```bash
./gradlew :extensions:nn:test --tests '*RuntimeLengthScanModelTest' --tests '*RuntimeLengthRecurrentSequenceTest'
./gradlew :extensions:data:test :extensions:data:javadoc
./gradlew :testing:integration-tests:test --tests '*PaddedNumericSequenceInferenceIntegrationTest'
```

Successful coordinator checkpoint after ordered integration and independent review:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew :testing:architecture-tests:test test javadoc
```

Reuse successful unchanged NN evidence; no second full NN suite. The coordinator root checkpoint
is selected because the approved architecture/input boundary changes. Retain up-to-date task
evidence; configure the existing Metal library/environment before the root gate, recorded by main.
No benchmark or native rebuild is required.

## Documentation and review impact

Separate clean implementation and independent Class C code/API/lifecycle/docs review completed.
Data Javadoc/public workflow, root/scoped status, module explanation, ADR0044, Text boundary and
targeted host-sequence glossary wording align with the exact export. Existing NN Javadoc remains
accurate and unchanged; no new terminology. Historical briefs are unchanged.

## Result

Implemented export/type Javadoc, PaddedSequenceBatchTest, DataApiConsumer and new numeric integration test.
Finalized root/scoped status, module-boundaries, ADR0044, Data/Text API, targeted glossary and planning.
Independent Class C review: APPROVE, zero findings; Data/consumer and existing NN Javadoc accurate.
Worker commands above PASS with the coordinator's Java 26 JAVA_HOME: NN 16+76=92 tests;
Data 52 tests (33 batch), Javadoc generated; integration 3 families, dense/final × 3 count patterns,
6 explicit preparations, 18 same-prepared runValues and 18 Model.compute checks. Evidence reused.
Main's standalone ScanExample PASS: `[3,2,1]`, counts `[3,2]`, seed42 RNN4/final/Linear1,
finite `[2,1]` scores; published program matches the executed source.
Coordinator command above PASS, exit 0, BUILD SUCCESSFUL in 14s: 119 tasks, 7 executed,
1 FROM-CACHE, 111 UP-TO-DATE; this is not a claim that every test executed anew.
Current XML: architecture 25 tests/11 suites, integration 187/35, Data 52/5; zero failures/errors.
Architecture/Data zero skips; integration one unrelated EngineModelAutotuningIntegrationTest
`eligibleCpuAlternativesCompleteBothPublicTuningPhases` skip: "host CPU fixture has no eligible local-workload alternatives".
Both isolated Data/Text API consumers compiled, including the new Data export witness.
Main confirms all four reviewed source/test hashes and both protected hashes match the freeze;
independent APPROVE applies to those exact sources. No native rebuild or reviewer test rerun.
Local links/anchors, whitespace and brief-size checks pass; no executable edits in review/closure.
Main accepted 2026-10-21; no remaining task follow-up or automatic successor launch.
Status: Complete

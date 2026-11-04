# Task NN0022B [Runtime](../../../../glossary.md#term-runtime) length scan [Model](../../../../glossary.md#term-model) workflow

> **Editorial reading context — not a status update.** The builder describes a two-input
> Model whose positions are values then valid lengths; the application supplies fresh stored
> Tensors for those positions at compute. finalHidden selects the state after each row's
> last valid transition, unlike lastTimeStep, which selects the final rectangular time index.
> Each scan ignores its own padded suffix; ordinary bias or sigmoid layers do not acquire a
> model-wide padding mask. Private Model state is acquired once, not copied anew per batch.
> Today's [builder contract](../../../../reference/nn/model-builder.md#fluent-model-builder) and
> [runtime-length composition](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition)
> distinguish these meanings and identify later capabilities separately from the historical inference-only gate.

## Status

Complete — accepted 2026-10-20 after independent reviews and the coordinator's final root gate.
Current authorized [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../../../../glossary.md#term-cpu)/Data frontiers: None. NN0022A/0023/0024 remain Draft; no automatic launch.

## Change class

Class C: public two-input workflow/limited [Model](../../../../glossary.md#term-model) contract cutover; clean implementation and independent review required.

## Goal

Build/save/load/compute scan Models through existing [Model](../../../../glossary.md#term-model)/[Engine](../../../../glossary.md#term-engine) [owners](../../../../glossary.md#term-owner); no Data prerequisite or dynamic [shapes](../../../../glossary.md#shape).

## [Scope](../../../../glossary.md#term-scope) and approved [API](../../../../glossary.md#term-application-programming-interface-api)

Approved: five methods on existing ModelBuilder and one on existing [Model](../../../../glossary.md#term-model):

```java
ModelBuilder rnnScan(long hiddenSize);
ModelBuilder gruScan(long hiddenSize);
ModelBuilder lstmScan(long hiddenSize);
ModelBuilder finalHidden();
Model<List<Tensor>, List<Tensor>> buildWithLengths(); // ModelBuilder
HostTensorValue compute(Tensor firstInput, Tensor secondInput); // Model
```

Each scan recipe uses FORWARD biased zero-state Sequence.forwardScan with existing [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and positive hidden width.
The first scan requires positive static [TBF](../../../../glossary.md#term-tbf) `[T,B,F]`, establishing one [canonical](../../../../glossary.md#term-canonical-form) non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype) `[B]`
length [prototype](../../../../glossary.md#term-prototype); later scans keep T/B, infer only F and own independent [parameters](../../../../glossary.md#term-parameter)/zero h (and c).
Length values may include zero. Advanced Model.[graph](../../../../glossary.md#graph) retains the scan contract's zero T/B and
explicit-state [signatures](../../../../glossary.md#term-signature), both directions and [bias](../../../../glossary.md#term-bias) forms; no broader [backend](../../../../glossary.md#backend) support is promised.

`buildWithLengths()` requires at least one scan and returns the existing list-boundary Model with
declared input order `[values, validLengths]` and one chain result. `build()` retains its existing
single-[Tensor](../../../../glossary.md#tensor) type/static behavior and rejects scan recipes; the new terminal rejects no-scan
chains. Both reject incompatible terminal choices before acquisition/build side effects, [snapshot](../../../../glossary.md#term-snapshot)
recipes and build fresh private state. The list type keeps advanced cold forward truthful.

`finalHidden()` is valid only immediately after a new scan recipe. It selects that [occurrence](../../../../glossary.md#term-occurrence)'s
existing finalHidden port `[B,H]`, not a time index or new [operation](../../../../glossary.md#operation). Reject it after intervening
transforms, after static recurrence, or twice; a classifier appends ordinary layers AFTER it.
A zero-length row returns the semantic initial h (zero for this builder), not `select(length-1)`.
[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) finalCell, explicit h/c and continuation use advanced graph/list boundaries; no implicit [carry](../../../../glossary.md#term-carry) between [batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters).

Implemented usage: [FLOAT32](../../../../glossary.md#term-float32) `[4,3,2]` values and INT64 `[3]` lengths `[4,2,0]`
return [detached](../../../../glossary.md#detached-model-results) `[3,5]` scores; the head's bias may make the zero-length row nonzero.

```java
try (var model = Model.builder().input(DataType.FLOAT32, 4, 3, 2)
        .gruScan(8).lstmScan(6).finalHidden().linear(5).buildWithLengths()) {
    HostTensorValue scores = model.compute(values, validLengths);
    model.save(path);
}
try (var loaded = Model.loadGraph(path)) {
    HostTensorValue scores = loaded.compute(values, validLengths);
}
```

The compute overload removes list/result indexing: EXACTLY two declared inputs and one output
in positional order for ANY [owned Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions), not only scans; no [dtype](../../../../glossary.md#data-type--datatype),
length-role or [artifact](../../../../glossary.md#term-artifact) [inference](../../../../glossary.md#term-inference-meanings). Use Long.MAX_VALUE and one begin/end [admission](../../../../glossary.md#term-admission): [arity](../../../../glossary.md#term-arity) before
[arguments](../../../../glossary.md#term-argument), then computeOpen, not nested public compute. Capability/[lifecycle](../../../../glossary.md#lifecycle) precedence stays unchanged.
List/bounded compute and optional cold forward remain advanced; forward is never a compute prerequisite.

### Ordinary steps and [padding](../../../../glossary.md#padding)

The following rules describe composition, not an implicit [model](../../../../glossary.md#term-model)-wide [mask](../../../../glossary.md#term-boolean-mask). Only each scan skips
its own padded recurrent transitions and zero-pads its dense output; ordinary layers do not.

| Existing step | Meaning before, between or after scans |
|---|---|
| input, [seed](../../../../glossary.md#term-seed) | Existing exact [canonical](../../../../glossary.md#term-canonical-form) input and deterministic per-step seed rules; seed counts every appended recipe, including finalHidden. |
| [linear](../../../../glossary.md#term-linear-meanings), supplied linear, relu, [tanh](../../../../glossary.md#term-tanh), [sigmoid](../../../../glossary.md#term-sigmoid) | Ordinary last-[axis](../../../../glossary.md#term-axis) projection or [elementwise](../../../../glossary.md#term-elementwise) meaning at every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix). [Bias](../../../../glossary.md#term-bias) may change padded zeros; sigmoid(0)=0.5. No automatic masking, zero-preservation or skipped-work promise. |
| conv1d/2d/3d, maxPool2d | Existing channels-first [rank](../../../../glossary.md#term-rank)/geometry guards and ordinary dense meaning, not sequence-aware [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling). They may mix or transform axes interpreted elsewhere as time/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); no global original-padding independence is promised. |
| flatten | Existing axis-zero-preserving reshape; `[T,B,H]` becomes `[T,B*H]`, not `[B,T*H]`. A subsequent scan rejects rank/geometry mismatch. |
| rnn, gru, lstm | Unchanged all-valid static unroll with zero state, ignoring [runtime](../../../../glossary.md#term-runtime) lengths. No implicit conversion; static-only chains still train, scan-containing chains remain [inference](../../../../glossary.md#term-inference)-only. |
| lastTimeStep | Unchanged rank-three `select(0,T-1)`, including after a scan; it selects a padded zero for short rows only if intervening ordinary steps left it zero. It is not finalHidden. |
| New scan recipes | Produce dense `[T,B,H]`; direct stacks share lengths and skip padded transitions independently. Between scans, ordinary transformed suffix values are ignored by the NEXT scan, not erased from Model results. |

Allow ordinary steps wherever their existing [shape](../../../../glossary.md#shape) contracts hold. At each later scan check exact
T/B [alignment](../../../../glossary.md#term-alignment); do not guess, reshape lengths, or silently redefine an axis. A prefix establishes
the first scan's [TBF](../../../../glossary.md#term-tbf) interpretation; callers supply lengths for that interpretation. Selecting
finalHidden discards the time axis; later scan recipes require a valid rank-three boundary again.

## Non-goals

No static migration, [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [backend](../../../../glossary.md#backend)/Data [widening](../../../../glossary.md#term-widening), bidirectional builder, dynamic [shapes](../../../../glossary.md#shape), new [gather](../../../../glossary.md#gather),
[mask](../../../../glossary.md#term-boolean-mask) propagation, streaming state, wrapper/session/hierarchy/[registry](../../../../glossary.md#term-registry), [codec](../../../../glossary.md#term-codec) or performance campaign.
Do not backend-gate valid floating expressions. Execution [qualification](../../../../glossary.md#term-qualification) is [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [CPU](../../../../glossary.md#term-cpu);
[Engine](../../../../glossary.md#term-engine) owns unsupported [occurrence](../../../../glossary.md#term-occurrence) diagnostics.

## Contracts

Read the root [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
[Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals);
owning [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn) and
[Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>;
the full [recurrent-scan contract](../../../../architecture/contracts/recurrent-scan.md); and
[Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
For [persistence](../../../../glossary.md#term-persistence) read [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
and [graph-model v1](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [model glossary](../../../../glossary.md#term-model))</span>/
[v3](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3).

The approved cutover is enacted in the root, owning [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) heading and [module-boundaries explanation](../../../../architecture/module-boundaries.md#owned-tensor-model-composition)<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>,
with [ADR0041](../../../../design/decisions/0041-runtime-length-scan-model-inference.md).
[Engine](../../../../glossary.md#term-engine)/scan semantics, [ownership](../../../../glossary.md#term-ownership-meanings) and allowed edges remain unchanged; stop on any further gap.

## Dependencies and integration

- Depends on: NN0022 accepted dirty Complete; NN0028/0029, CPU0028/NN0021B, Engine0021,
  CP0010/CP0011 and their Model0025E/0025F/Compiler0006A source prerequisites Complete.
- Conflicts with: [NN Model](../../../../glossary.md#model-module-buffer-and-scan-distinctions)/builder/source/[lifecycle](../../../../glossary.md#lifecycle) authority and shared documentation; NN0022A/0023/0024 when touching that [scope](../../../../glossary.md#term-scope). No concurrent writer.
- Parallel group: None.
- Common base revision: HEAD=origin/main `2d1c1d6d0867a0bebda34e9177ad52ba9f13071a`, PLUS
  accepted uncommitted NN0022; coordinator-verified combined 11-Java-file SHA256
  `a2dcef0b35c74932f6c7dd8c9ad9e5ad792976e4bc0548362e9cef417ea6821c`. HEAD does NOT contain NN0022.
- Integration order: approved cutover → independent contract review → clean implementation → independent code/docs review → coordinator checkpoint;
  then NN0022A only after its separate Data0001/0002 gates. No task becomes Ready automatically.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: focused public scan Model integration plus one coordinated root test/Javadoc
  gate because this capability changes the owned Model contract boundary; reuse CPU0028 route evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [owned Model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main coordinator, sole writer of shared plans/authority/explanations.

Pre-launch verification — 2026-10-20: coordinator verified [Ready](../../../../glossary.md#term-ready)/frontier, Complete dependencies, dirty NN0022/[hash](../../../../glossary.md#term-hash),
base/protected/native hashes and sole writer; Carson contract APPROVE, zero findings, preceded clean implementation.

## Files and symbols

Existing `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/` [owners](../../../../glossary.md#term-owner): ModelBuilder
recipes/adapters and terminals; [Model](../../../../glossary.md#term-model).fromComposition/TensorListModel; OwnedModelSource
acquireComposition/inventory/forward; ModelExecutionState computeOpen/configureOpen/train [admission](../../../../glossary.md#term-admission).
Composition uses ordered lists, excludes every input from [constants](../../../../glossary.md#term-constant) and retains instantiated owned
steps through existing adapters/Sequential/ModelForward, not callbacks/recipes or per-[run](../../../../glossary.md#term-run) reconstruction.
New production types/packages: None, including nested types; no second [graph](../../../../glossary.md#graph) or execution path.

Tests: RuntimeLengthScanModelTest, ModelBuilderTest, OwnedModelExecutionTest, OwnedModelLifecycleTest;
RuntimeLengthScanModelIntegrationTest in the existing testing integration package.

## Acceptance criteria

1. Preserve the approved API, extent/state/direction and step policies; verify the recorded
   Ready/frontier/base/dependencies and independent contract review before launch.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [extent glossary](../../../../glossary.md#term-extent); [Ready glossary](../../../../glossary.md#term-ready))</span>
2. buildWithLengths rejects no-scan chains; build rejects scans without changing static behavior.
   Advanced list cold forward accepts two exact-descriptor symbolic inputs, including produced
   substitutions; no Engine or retained input storage. Two inputs, source roles, canonical state
   paths, supplied-state aliases, initialization and fresh-build isolation survive construction/load.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Engine glossary](../../../../glossary.md#term-engine); [canonical glossary](../../../../glossary.md#term-canonical-form); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
3. For all three families, direct and mixed stacks, `[T,short,0]`, all-zero and changed lengths,
   compare every output against independent equations or the accepted direct oracle. Validate
   finalHidden versus dense last-time results, nonzero explicit initial states through graph,
   biased/sigmoid inter-step suffixes, ordinary prefix/head geometry and invalid-length repair.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [graph glossary](../../../../glossary.md#graph); [sigmoid glossary](../../../../glossary.md#term-sigmoid))</span>
4. New compute(first,second) equals list compute's sole result for scan AND ordinary two-input
   Models, built/graph/loaded. Test wrong arities, nulls, closed/busy precedence, one admission,
   no freeze/Engine work on invalid arguments, and same compute before/after load without forward.
   Reuse computeOpen's cached compile/prepare/bindings/runValues path. Prove reuse, exact unused-slot descriptors,
   unchanged binding freeze/Engine borrowing, busy/close/guard semantics and detached aggregate-bounded results.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [arguments glossary](../../../../glossary.md#term-argument); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [bindings glossary](../../../../glossary.md#term-binding); [borrowing glossary](../../../../glossary.md#term-borrowing); [guard glossary](../../../../glossary.md#term-guard); [detached glossary](../../../../glossary.md#detached-model-results))</span>
5. Cache scan presence at acquisition, including graph/load, heads and final-state-only roots.
   Both configure/train overload families reject UnsupportedOperationException after lifecycle/live-source
   admission but before freeze, callbacks, Engine creation, cached-owner closure or native copy.
   Before/after compute/load, preserve state bytes, stores, caches, freeze status and requiresGrad.
   Ordinary static-unroll training and generic unsupported capability behavior remain unchanged.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>
6. Save before compute, close, loadGraph and compute ordered values through the same path; persist
   positions, not inferred roles. v3 retains scan siblings for finalHidden-only roots; rank-three
   axis-zero SELECT/CONCAT and rank-two last-axis unit-step SLICE remain narrow, not arbitrary gather/feature concat.
   Direct stacks/finalHidden/dense heads need no format change. Validate re-save/old compatibility and
   unsupported-save isolation; stop on a source-slice gap.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [SLICE glossary](../../../../glossary.md#slice); [gather glossary](../../../../glossary.md#gather); [feature glossary](../../../../glossary.md#term-feature))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker/[module](../../../../glossary.md#term-module) and coordinator commands/results are recorded below; successful executable evidence is reused.
Final [planning](../../../../glossary.md#term-planning)-only checks: local links/anchors, brief ≤200 lines/15 KB, whitespace and protected [hashes](../../../../glossary.md#term-hash); no [Gradle](../../../../glossary.md#term-gradle) rerun.

## Documentation and review impact

[Model](../../../../glossary.md#term-model)/builder and affected implementation [Javadoc](../../../../glossary.md#term-javadoc), public-api/running-models examples, training-api rejection,
root/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) contract, [module](../../../../glossary.md#term-module)-boundaries, ADR0041 and targeted glossary terms finalized; independent docs/code review passed.

## Result

Accepted 2026-10-20; all acceptance criteria reviewed. Status: Complete.

- Hilbert implemented the six methods in the four [module](../../../../glossary.md#term-module) [owners](../../../../glossary.md#term-owner) and tests above; no new production types/packages.
- Worker `./gradlew :extensions:nn:test :extensions:nn:javadoc`: PASS, 524 tests/57 suites,
  zero failures/errors/skips; focused 72 PASS plus corrected prefix fixture 1 PASS.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- Main `./gradlew :testing:integration-tests:test --tests '*RuntimeLengthScanModelIntegrationTest'`:
  11 PASS, zero failures/errors/skips; exact guide example compiled/executed on Java 26: `[3,5]`, same scores after load.
- Carson contract APPROVE and Cicero FINAL code APPROVE: zero findings, final [hashes](../../../../glossary.md#term-hash) bound.
- Einstein independent docs Complete: 102 link checks; [Javadoc](../../../../glossary.md#term-javadoc)/comment-only edits preserved executable [tokens](../../../../glossary.md#term-token) and five test files.
- Main `./gradlew test javadoc`: exit 0, BUILD SUCCESSFUL in 21s; 107 tasks, 7 executed/100 UP-TO-DATE;
  configuration cache reused. JDK 26.0.1 + existing <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`SYNAPTIK_METAL_TEST_LIBRARY`; not all tests freshly executed.
- Root XML: NN 524/57 suites and architecture 13/9 suites, zero failures/errors/skips;
  integration 183/33 suites, zero failures/errors, one unrelated environment-assumption skip:<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
  `EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases`
  (host CPU fixture has no eligible local-workload alternatives). Focused scan integration remains 11/0 skips.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Frozen nine-Java [manifest](../../../../glossary.md#term-manifest) SHA256 `563a71aa18157f3436486d0577701a12bc3cb625bfbf56b9a831afe1fe607ee4`
  unchanged after final review/root gate; protected notice/oracle/native hashes unchanged.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Inference-only, canonical FLOAT32 CPU execution; no BPTT, Data, dtype, backend or format widening.
  No unresolved issue or capability follow-up; Data remains a separate Draft. Current NN/CPU/Data frontiers: None.<span data-glossary-navigation="true"> ([Inference glossary](../../../../glossary.md#term-inference); [canonical glossary](../../../../glossary.md#term-canonical-form); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [dtype glossary](../../../../glossary.md#data-type--datatype); [backend glossary](../../../../glossary.md#backend); [widening glossary](../../../../glossary.md#term-widening))</span>

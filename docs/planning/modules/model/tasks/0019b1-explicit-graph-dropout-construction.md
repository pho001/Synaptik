# Task 0019B1: Explicit Graph Dropout Construction

## Status

Complete

## Goal

Add one backend-independent, training-only dropout expression that consumes explicit
`GraphRngState`, returns the dropped value and next state, and records a non-public mask output for
later compiler-owned backward construction. Construction must expose no generator, perform no
sampling, mutate no state, and implement no compiler, runtime, prepare, or backend behavior.

Newcomer mental model:

```java
GraphRngState start = GraphRngState.initial(0x1234L, 0L);

DropoutResult first = activations.dropout(0.1d, start);
DropoutResult replay = activations.dropout(
        0.1d, GraphRngState.initial(0x1234L, 0L));
DropoutResult second = first.output().dropout(0.1d, first.nextState());
```

`first` and `replay` request the same abstract random interval. `second` requests the following
interval by threading `first.nextState()`. Separately constructed expressions remain distinct
Tensor/producer occurrences even when their abstract values replay.

## Rationale and foundation audit

Completed tasks 0018K–0018L and 0019B make this operation representable without an architecture
change. `OperationSignature.fixed(..., 2, 3)` accepts the occurrence; one `TensorProducer` can
retain ordered inputs and three ordered descriptors; `TensorFactory.createDerivedOutputs(...)`
creates one indexed Tensor wrapper for every output; and the package-private `GraphRngState.tensor()`
and constructor seams provide typed state input and output wrapping without a public Tensor
accessor.

The factory cannot create a descriptor-only output. It necessarily allocates three Tensor
wrappers and three Tensor IDs, one for each output slot. The slot-one mask wrapper is created with
provenance and is then intentionally not retained by the public result. Its descriptor and output
position remain retained by the shared producer reachable from output slots zero and two, which is
the current information boundary a later compiler capture/autograd task needs to represent the
auxiliary value. If implementation instead needs a live mask Tensor reference, sibling-output
registry, public mask, graph-local identity, or another module, stop and report the conflict.

## Scope

- Add `DropoutKind.DROPOUT` and immutable `DropoutAttrs(double probability)` in
  `model.operation.random`.
- Add exactly this public Tensor method and result carrier in `model.tensor`:

  ```java
  public DropoutResult dropout(double probability, GraphRngState state)
  public record DropoutResult(Tensor output, GraphRngState nextState) {}
  ```

- Add one package-private, field-free `TensorDropoutExpressions` construction helper.
- Construct ordered inputs `[input, stateTensor]` and ordered outputs `[output, mask, nextState]`.
- Fix floating eligibility, descriptor propagation, validation order/messages, three-ID behavior,
  state advancement, replay/branching, special-value meaning, and training-only semantics.
- Add focused semantic, expression, API-inventory, signature-inventory, provenance, ID-allocation,
  validation, and immutability tests.
- Finalize public Javadocs, Tensor API, Compile API, glossary, capability/master/roadmap status, and
  this task's evidence through a mandatory separate clean-context documentation pass.

## Out of scope

- sampling, eager evaluation, storage reads/writes, host or device allocation, or
  `TensorRandoms` delegation
- a public mask accessor, public state Tensor accessor, sibling-output lookup, producer registry,
  state accessor, reset, split, fork, fold-in, jump, or copy API
- a portable PRNG, key schedule, counter-to-bits function, uniform conversion, algorithm enum,
  generator interface, service, registry, framework, or statistical-quality claim
- compiler capture, gradient formulas, autograd traversal, backward graph construction, saved-value
  lifetime policy, common-subexpression policy, or inference rewrite implementation
- runtime session or run state, prepare contracts, kernels, lowering, backend routes, conformance
  behavior, numerical execution algorithms, or cross-backend bitwise identity
- attention dropout, an attention API change, generic graph-random distributions, Bernoulli API,
  shuffling, permutation, normal/uniform/integral/categorical sampling, or eager random changes
- a `training` flag, inference mode, module/layer object, parameter, optimizer, or training workflow
- Gradle, dependencies, architecture documents/tests, another module, or a detailed 0019C-or-later
  task specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Tasks 0018K](0018k-operation-signature-and-construction-hardening.md),
  [0018L](0018l-shared-multi-output-tensor-provenance.md), and
  [0019B](0019b-explicit-graph-rng-state-foundation.md)
- completed Tensor/TensorFactory/provenance, typed-scalar, eager-random, and operation-family tasks
- [Tensor API](../../../../api/tensor-api.md), [Compile API](../../../../api/compile-api.md),
  [Runtime API](../../../../api/runtime-api.md), [Training API](../../../../api/training-api.md), and
  [glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-neutral Tensor expression and Operation semantics. It does not
  sample, execute, retain runtime state, choose a backend, or implement gradients.
- `Tensor` remains public mutable API state and not IR. `GraphRngState` remains an opaque wrapper
  around one storage-free Tensor occurrence, not a generator, runtime session, or Tensor metadata.
- One dropout call creates one exact `TensorProducer`; all three wrappers receive indexed
  `TensorProvenance` referencing that producer. The producer retains descriptors, not output
  Tensor references.
- The public output and private next-state Tensor wrapper keep the producer reachable. The mask
  wrapper is intentionally not publicly retained; the producer still describes slot one for later
  compiler-owned auxiliary-value construction.
- Compiler owns backward graph construction and any decision to retain/materialize the forward
  mask. Prepare/backend/runtime own sampling, state materialization, numerical implementation, and
  execution. Inference bypass is caller/compiler graph composition, not model execution mode.
- No global/default/thread-local RNG, mutable static state, service locator, registry, reflection,
  backend handle, storage, or runtime object may enter these contracts.
- If a public mask, exposed state Tensor, changed factory/provenance architecture, portable PRNG,
  another module, or architecture rule change is required, stop before implementation.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — public Tensor/state/result contracts and package-private
  multi-output construction.
- `io.github.pho001.synaptik.model.operation.random` — explicit graph-random semantics, distinct
  from eager host-data population.
- `io.github.pho001.synaptik.model.datatype` and `.shape` — output descriptor facts.

Packages added or changed:

- No new package. Extend the established random operation family and tensor expression package.

Type placement:

- `io.github.pho001.synaptik.model.operation.random.DropoutAttrs` — probability-bearing immutable
  semantic attributes beside graph RNG semantics.
- `io.github.pho001.synaptik.model.operation.random.DropoutKind` — focused one-constant dropout
  operation family.
- `io.github.pho001.synaptik.model.tensor.DropoutResult` — public value-plus-state result beside
  the Tensor and opaque state types it returns.
- `io.github.pho001.synaptik.model.tensor.TensorDropoutExpressions` — package-private construction
  helper with access to the state and factory seams.

Test placement:

- `model.operation.random.DropoutSemanticsTest` — attributes, probability boundary, exact kind,
  signature, and ideal meaning.
- `model.tensor.TensorDropoutExpressionTest` — public API, descriptors, provenance, state wrapping,
  hidden mask slot, validation, identities, and ID effects.

## Required contract

### Operation attributes, kind, and signature

Create:

```java
public record DropoutAttrs(double probability) implements OperationAttrs {}

public enum DropoutKind implements OperationKind {
    DROPOUT
}
```

`DropoutAttrs` accepts a finite binary64 drop probability numerically in `[0.0, 1.0)`. Positive
and negative zero are both accepted, retained bit-for-bit by the record, and mean zero dropout.
NaN, either infinity, negative finite values, and `1.0` or larger fail with:

```text
probability must be finite and in [0.0, 1.0): <value>
```

`DropoutKind.DROPOUT` accepts only `DropoutAttrs`, exactly two inputs, and exactly three outputs:

```java
OperationSignature.fixed(DropoutAttrs.class, 2, 3)
```

The operation is first-class because one stochastic occurrence must preserve explicit state,
auxiliary-mask, and next-state semantics. It is not a decomposition, eager sampler, generator,
kernel, or training-mode switch.

### Public API and result

Add to `Tensor`:

```java
public DropoutResult dropout(double probability, GraphRngState state)
```

The receiver is named `input` inside the helper, the scalar argument is `probability`, the state
argument is `state`, and `DropoutResult` record components are exactly `output` then `nextState`.
The public record's compact constructor null-checks `output` then `nextState`, with parameter-name
messages, and retains exact references. It is shallowly immutable and uses record value semantics;
it exposes no mask, producer, counter, or state Tensor.

### Ordered producer inputs and outputs

One call creates exactly one producer with:

```text
operation: DropoutKind.DROPOUT + DropoutAttrs(probability)
inputs:
  0 input
  1 state.tensor()
outputs:
  0 public dropped output
  1 auxiliary keep mask
  2 next RNG state Tensor
```

Output zero and output two are returned through `DropoutResult`; output two is wrapped with the
existing package-private `new GraphRngState(tensor)` seam. Output one is never exposed. All three
factory-created wrappers have provenance indices matching their slots and share the exact producer.
The returned output and wrapped next-state Tensor therefore provide two reachable provenance paths
to the same producer and its slot-one BOOL descriptor.

Do not add a public mask or retain it in `DropoutResult`. Compiler capture/autograd may later create
its own graph value for producer slot one and retain that value for backward use. This task defines
the mask as the selected saved forward value but implements neither capture nor a gradient rule.

### Construction and Tensor IDs

`TensorDropoutExpressions.apply(input, probability, state)` performs all local validation and
descriptor construction, then calls `TensorFactory.createDerivedOutputs(...)` exactly once with
the exact ordered inputs and descriptors. It reads slots zero, one, and two from the immutable
result list, intentionally discards the local mask reference after verifying construction by
position, wraps slot two, and returns slot zero plus that wrapper.

A successful call allocates exactly three fresh Tensor IDs in output order. It creates exactly
three fresh Tensor wrappers, three provenance values, one producer, one operation, one attributes
value, three descriptors, one public result record, and one `GraphRngState` wrapper. It allocates
no storage and mutates neither input nor state. No claim is made about adjacency under concurrent
construction; tests that inspect allocation use an isolated factory-ID seam and assert the three
positions in one call.

All known validation occurs before factory delegation and consumes no ID. Identifier exhaustion
uses the existing factory contract: it may consume IDs for earlier output positions, returns no
partial list/result/state wrapper, and throws `tensor identifier space exhausted`. There is no
rollback. This identity-allocation effect is not RNG-state advancement or mutation.

### Output descriptors

For input descriptor `(dataType, shape, layout, requiresGrad)`:

```text
slot 0 output:
  dataType: exact input floating data type
  shape: exact input Shape reference
  layout: unresolved
  requiresGrad: exact input requiresGrad
  label/storage: absent

slot 1 mask:
  dataType: BOOL
  shape: exact input Shape reference
  layout: unresolved
  requiresGrad: false
  label/storage: absent

slot 2 next state:
  dataType: INT64
  shape: Shape.of(2), structurally exact state shape
  layout: unresolved
  requiresGrad: false
  label/storage: absent
```

Only FLOAT64, FLOAT32, and BFLOAT16 input are eligible through `DataType.isFloating()`. The output
preserves the exact type and Shape reference and the input's `requiresGrad` flag; it deliberately
does not preserve resolved layout, label, or storage. The mask never requires gradients. The state
descriptor must satisfy every existing `GraphRngState` wrapper invariant.

### Ideal dropout and special-value meaning

For each logical input element at row-major logical position `i`, consume exactly one abstract
uniform draw `u_i` in `[0, 1)`. The BOOL keep mask is true exactly when
`u_i >= probability`. The ideal result is:

```text
mask[i] ? input[i] / (1 - probability) : +0 of the input data type
```

This is inverted dropout. A dropped value is positive zero even when the input is negative zero,
NaN, or an infinity. A kept positive/negative zero retains its sign under the positive scale; a
kept NaN remains NaN without a NaN payload/sign promise; and kept positive/negative infinity
retains its infinity sign. Finite values use the ideal real scale followed by the result type's
eventual conforming rounding. This task selects no finite-precision evaluation order, fused route,
underflow/overflow detail beyond the type's conforming arithmetic, or bitwise NaN behavior.

At either signed probability zero, every conforming draw keeps the value because draws are in
`[0,1)`. The result is numerical identity, including signed zero, NaN classification, and infinity
sign, but the operation still consumes one draw per element, creates fresh outputs, and advances
state. Do not canonicalize probability zero into a no-op at model construction.

### State advancement, replay, and dynamic shapes

Let the input's bound logical element count be `N`. The operation consumes abstract counter
positions `counter` through `counter + N - 1`, retains the unsigned 64-bit key unchanged, and
produces `(key, counter + N mod 2^64)`. The count is independent of probability, input values,
mask outcomes, type, layout, and backend vectorization.

- A fully static Shape uses the mathematical product of its static extents as `N`; when
  `Shape.knownElementCount()` is present, that value is exact. A product larger than signed
  `long` is not a model-construction rejection: later execution must advance by that product
  modulo `2^64` without requiring it to fit a signed host count.
- A dynamic or expression Shape records shape-dependent advancement. Compiler/prepare must bind
  all extents, and the conforming prepared execution advances by the resulting non-negative
  logical element count. Model construction neither binds nor stores `N` in attributes.
- Any bound Shape containing a zero extent has `N == 0`: output and mask are empty, the next state
  has the same key/counter value, and slot two is still a distinct Tensor/state occurrence.
- Scalar Shape has `N == 1`.
- Counter arithmetic is unsigned modulo `2^64`; wrap does not fail in model.

Equal input values/Shape, probability, input state value, and the same conforming prepared
implementation path and configuration must replay equal mask, output, and next-state values.
Branching the same state into two dropout occurrences intentionally reuses the same counter
interval. Sequential non-overlap requires explicit `nextState` threading. Public objects are
immutable/identity-bearing expression occurrences and are safe to share for construction, but
sharing does not serialize executions. No cross-backend, cross-route, cross-provider,
cross-version, or bitwise sample identity is promised until a portable algorithm is selected.
The key/counter advancement result itself is portable semantic state.

### Training, inference, and backward boundary

`DROPOUT` always means training dropout. There is no model-level `training` boolean and no
inference behavior inside the operation. A caller or future compiler constructs inference by
using the original input and original state directly, so inference consumes no draw, advances no
state, and creates no dropout producer. A future compiler owns the rule that backward multiplies
the upstream gradient by the retained keep mask and the same inverted scale; this task neither
adds that rule nor claims that backward compilation exists.

## Validation order and side effects

The package-private helper validates in this exact order:

1. non-null `input`, else `NullPointerException("input")`;
2. floating input type, else `IllegalArgumentException` with
   `dropout input data type must be floating: <type>`;
3. construct `DropoutAttrs`, applying the exact probability message above;
4. non-null `state`, else `NullPointerException("state")`;
5. obtain the exact package-private state Tensor;
6. construct output, mask, and fixed-state descriptors and the operation;
7. delegate once to `createDerivedOutputs(...)`; and
8. wrap slot two, then construct `DropoutResult` from slots zero and two.

Thus input eligibility precedes probability, and probability precedes state nullity. Every local
failure occurs before producer/ID allocation and has no Tensor-ID, state, storage, sampling, or
generator side effect. The already-validated opaque state needs no repeated invariant validation
before factory construction; the existing wrapper validates the generated next-state Tensor after
allocation. Unexpected wrapper failure would expose an implementation defect, not a public input
case, and does not justify rollback or mutable state.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutAttrs.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/DropoutKind.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/DropoutResult.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressions.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/GraphRngState.java`

Expected model tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/random/DropoutSemanticsTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDropoutExpressionTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`

Expected documentation/planning:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task specification
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Mandatory inspected inventories/seams, changed only if a listed acceptance criterion proves a
stale contract that cannot be covered in the focused files above:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java` and
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorProducer.java`,
  `TensorProvenance.java`, plus
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorProducerTest.java` and
  `TensorProvenanceTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/random/GraphRngStateAttrs.java`,
  `GraphRngKind.java`, and the existing `GraphRngStateSemanticsTest.java` plus
  `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/GraphRngStateTest.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/Operation.java`,
  `OperationKind.java`, `OperationAttrs.java`, and `OperationSignature.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java`,
  `TensorRandoms.java`, plus `model/datatype/DataType.java` and `model/shape/Shape.java`
- `docs/api/runtime-api.md` and `docs/api/training-api.md`, for reasoned no-change conclusions

## Maximum scope

This task may modify or create exactly the 19 expected paths above: six production, six test, and
seven documentation/planning paths. The sixth test path is the authorized stale global Tensor
public-method inventory in `TensorMatmulExpressionTest`; its only permitted change is the expected
count from 162 to 163, with every MATMUL-specific assertion preserved. Do not modify a mandatory
inspected seam merely to consume scope.

No Java path outside `modules/model`, Gradle/dependency file, architecture/focused-architecture
document, architecture test, backend-conformance/integration test, another module, or later task
specification may change. If a live auxiliary Tensor, compiler API, another package concept, or
more than 19 paths is required, stop and propose a focused follow-up or architecture decision.

## Javadoc and explanatory documentation requirements

- Document `DropoutAttrs`, `DropoutKind`, `DropoutResult`, the helper contract, and the Tensor
  method with probability, nullability, descriptors, three outputs, state advancement, ID effects,
  special values, replay boundary, training/inference meaning, and deferred execution/gradient
  ownership.
- Update `GraphRngState` Javadoc from future tense to current state threading without adding a
  public accessor or algorithm promise.
- Tensor API must include the runnable construction example above, output-slot table, p=0 and empty
  behavior, branching versus threading, and honest current-versus-planned execution boundary.
- Compile API must explain that later capture/autograd can observe producer slot one even though
  no public mask Tensor is returned; it must not claim compiler capture or gradients are current.
- Glossary must define dropout/inverted dropout, auxiliary mask, state threading, and bounded
  replay at first use without duplicating the complete API guide.
- Runtime API and Training API are expected no-change: no executable/runtime or training-extension
  API is added. Record that conclusion. Architecture docs, conformance/integration guides, and
  other modules are also expected no-change because ownership/boundaries do not change.

## Acceptance criteria

- Exact public API is `Tensor.dropout(double probability, GraphRngState state)` returning public
  `DropoutResult(Tensor output, GraphRngState nextState)` in `model.tensor`; no public mask or state
  Tensor accessor exists.
- `DropoutAttrs` accepts and retains both signed zeros and every other finite value in `[0,1)`, and
  rejects all other values with the exact selected message.
- `DropoutKind.DROPOUT` has exactly `OperationSignature.fixed(DropoutAttrs.class, 2, 3)` and no
  backend/execution knowledge.
- One producer has exact inputs `[input, stateTensor]`, exact descriptors `[output, mask,
  nextState]`, and all three factory wrappers have indices zero, one, and two with the same exact
  producer reference.
- Exactly three Tensor wrappers/IDs are allocated successfully. The mask wrapper is not exposed or
  retained in the public result, while its BOOL descriptor and slot remain producer-visible.
- Output/mask/state descriptors, exact Shape references, gradient flags, absent layout/label/
  storage, and state-wrapper invariants match this specification.
- Floating inputs succeed; INT32, INT64, and BOOL fail before probability/state/ID work according
  to the selected order. Null/input/probability/state messages and no-ID behavior are exact.
- Ideal keep rule, inverted scaling, signed zero, NaN, infinity, p=0, scalar, empty, static,
  dynamic/bound-count, modulo advancement, replay, branching, threading, identity, and
  non-mutation semantics are documented and tested at the representation level possible now.
- No sampling, storage access, TensorRandoms call, random algorithm/framework, hidden global state,
  execution, gradient rule, compiler change, runtime state, backend route, or attention dropout is
  added or claimed.
- Tensor public-method count/name inventory, global operation-signature inventory, result/kind/attrs
  shapes, factory/provenance behavior, and GraphRngState public surface remain locked by tests.
- One final `:modules:model:test` run occurs after executable Java stabilizes. The separate docs
  pass reuses that evidence and does not repeat it unless executable Java changes.
- The documentation-focused pass finalizes all affected Javadocs/docs/glossary/planning evidence,
  runs Javadoc and documentation checks, and records no-change conclusions.
- Task/master/roadmap show 0019B1 Complete with tasks 0019 through 0019B. Tasks 0019C–0019E and
  later stay Draft without detailed specifications; no model task is Ready or Review needed.

## Tests / validation

During development, run focused dropout semantic/expression tests and directly affected
signature/API/factory/provenance tests as needed. After executable Java stabilizes, run exactly one
final model suite:

```bash
./gradlew :modules:model:test
```

The first final run failed solely because the previously unlisted stale
`TensorMatmulExpressionTest` global count expected 162 methods. The explicitly authorized scope
repair adds that nineteenth path, changes only the count to 163, and permits one necessary final
model-suite rerun. Record this reason and do not perform another model-suite rerun.

The separate clean-context documentation pass then reuses that result and runs after final
Javadoc/documentation edits:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

It must also compile/run the newcomer example against current model classes; validate affected
local links and anchors, balanced fences, final newlines, trailing whitespace, exact path scope,
status/dependency coherence, no Ready or Review-needed frontier after checkpoint completion, and
absence of a 0019C-or-later task file.

This task is the RNG/dropout capability checkpoint because it closes the explicit-state plus first
state-consuming operation sequence and exercises the shared multi-output foundation in production.
After both passes, run repository-wide:

```bash
./gradlew test
```

Also run affected architecture tests only if the final diff changes a dependency or boundary; such
a change is not expected and otherwise requires stopping. Backend conformance and integration
tests remain deferred until an executable backend/runtime capability exists.

## Dependencies

- Task 0018K: operation signature and construction hardening — Complete.
- Task 0018L: shared multi-output Tensor provenance and factory seam — Complete.
- Tasks 0018N and 0018S: typed scalar/factory surface boundaries — Complete.
- Task 0019B: explicit graph RNG state foundation and package-private wrapping seam — Complete.
- Existing floating DataType, Shape, TensorDescriptor, Tensor, and producer/provenance contracts —
  Complete.

## Follow-up tasks

- 0019C — sorting, argsort, and true multi-output top-K; remains Draft without a detailed spec.
- 0019D — public linear convenience; remains Draft without a detailed spec.
- 0019E — scaled dot-product attention without dropout; remains Draft without a detailed spec.
- A later compiler task owns dropout capture, auxiliary-mask retention, and backward construction.
- A later backend/runtime capability owns sampling and execution. A portable PRNG requires a
  separate explicit semantic decision before cross-backend bitwise replay can be promised.
- Any later attention-dropout extension must consume/produce explicit state and is not part of
  initial 0019E.

## Architecture impact

Expected impact: None.

This task uses existing model-owned Tensor/Operation semantics, explicit state, and multi-output
provenance. It changes no module ownership, dependency direction, lifecycle, compiler boundary, or
runtime/prepare/backend responsibility. If implementation reveals otherwise, stop and report the
exact conflict before changing architecture documents or another module.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Read AGENTS.md, ARCHITECTURE.md, docs/developer-guide/documentation-rules.md,
docs/planning/planning-guide.md, docs/planning/modules/model/capabilities.md, the model master plan,
roadmap, completed tasks 0018K, 0018L, and 0019B, and task 0019B1 in full. Inspect every affected
source/test and mandatory inventory named by 0019B1.

Implement task 0019B1 exactly as specified, only in modules/model and the authorized paths. Stop
on an architecture, cross-module, live-mask, factory/provenance, or maximum-scope conflict. Do not
implement sampling, gradients, compiler/runtime/backend behavior, or another task. Run focused
tests while developing and exactly one final model suite after executable Java stabilizes.

Then hand the actual diff and recorded Java-test evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect the
final contracts, finalize Javadocs, Tensor/Compile API, glossary, planning/status/evidence and
no-change conclusions, run model Javadoc and documentation validation, and reuse the successful
Java evidence unless it changes executable behavior. Run the recorded checkpoint only after both
passes. Do not mark Complete until all acceptance criteria pass. Do not commit or push.
```

## Separate documentation handoff

The implementation agent must provide the documentation agent: this task file; the actual diff;
the exact final model-test command/result; affected public API and three-output/state semantics;
the no-algorithm, no-execution, compiler-owned-backward, and architecture boundaries; expected
Tensor/Compile API and glossary changes; expected Runtime/Training/architecture no-change reviews;
and all validation commands above. The docs agent must report its clean context ID and whether
executable Java changed after the reused test evidence.

## Local decisions

- The existing wrapper-per-descriptor factory seam is retained. Dropout creates all three indexed
  wrappers and IDs, discards the mask wrapper from its public result, and relies on the shared
  producer's slot-one descriptor for later compiler capture. No descriptor-only output or sibling
  registry is introduced.
- The public result remains exactly `(output, nextState)`. The auxiliary keep mask is deliberately
  compiler-facing metadata rather than public Tensor API.
- Signed-zero probability is preserved by `DropoutAttrs` record semantics. Probability zero is not
  canonicalized away because eventual execution still consumes the operation's logical draw count.

## Known limitations

- Model construction represents dropout but cannot execute it. No current compiler captures the
  auxiliary slot, no backend samples it, and no gradient rule consumes it.
- Replay of sampled values is bounded to the same conforming prepared implementation path and
  configuration because no portable PRNG/bitstream is selected.
- The factory creates a short-lived mask Tensor wrapper because current multi-output construction
  is wrapper-per-slot. Only its producer descriptor/position remains reachable publicly.

## Validation evidence

Planning evidence:

- Clean planning context inspected the authoritative/focused architecture, documentation/planning
  rules and profiles, model plans, completed foundation/random/state tasks, current source/tests,
  producer/factory/provenance seams, global signature/Tensor API inventories, APIs, and glossary.
- `TensorFactory.createDerivedOutputs(...)` was verified to validate one producer before allocating
  and then create one Tensor/provenance per descriptor. Therefore three dropout outputs require
  exactly three wrappers/IDs; a descriptor-only mask is not supported.
- `GraphRngState` was verified to provide package-private exact Tensor access and validated
  package-private wrapping, so no public accessor or cross-module change is required.

Implementation evidence:

- The implementation context ran focused tests for the two dropout test classes, broader focused
  tests for ten affected classes, and focused final-refinement tests; each run reported
  `BUILD SUCCESSFUL`.
- Its first final `./gradlew :modules:model:test` run reached 799 tests and failed only because the
  unlisted stale `TensorMatmulExpressionTest` inventory expected 162 public Tensor methods. The
  user authorized the exact 19-path repair, whose only change in that test is `162` to `163`.
- After that repair, the implementation context reran `./gradlew :modules:model:test`: `BUILD
  SUCCESSFUL` in 1s; 3 actionable tasks, 2 executed and 1 up-to-date. Executable production Java
  did not change after this successful run.

Documentation-agent evidence:

- Clean context `/root/task_0019b1_implementation/task_0019b1_docs` independently inspected the
  architecture contract and focused pages, documentation rules and General/API-Javadoc/Planning/
  Example profiles, planning guide/roadmap/capabilities/master plan, tasks 0018K/0018L/0019B/
  0019B1, the actual production and test diff, construction seams and inventories, Tensor/Compile/
  Runtime/Training APIs, and glossary.
- The pass finalized all six production Javadocs/comments plus Tensor API, Compile API, glossary,
  capability/master/roadmap status and this evidence without changing executable Java.
- `./gradlew :modules:model:javadoc` passed after the final Javadoc edit: `BUILD SUCCESSFUL` in 1s;
  2 actionable tasks, both executed. Generated pages for `Tensor`, `GraphRngState`, `DropoutAttrs`,
  `DropoutKind`, and `DropoutResult` were inspected for the new contracts and resolved links.
- The documented `DropoutConstructionExample` compiled against current model classes and ran with
  exact output `2`, `3`, `BOOL`, `true`, `false`, `false`.
- A targeted checker resolved every local Markdown file target and heading anchor in the seven
  affected documentation/planning files. Fence balance, final newlines, trailing whitespace,
  status/dependency coherence, absent 0019C-or-later specs, production imports/claims, and exact
  19-path scope were also checked. Final `git diff --check` passed.
- `javap` confirmed the public `dropout(double, GraphRngState)`, `DropoutResult.output()` and
  `nextState()`, `GraphRngState.initial(long, long)`, `DropoutKind.DROPOUT`, and
  `DropoutAttrs.probability()` surface. Automated implementation tests remain the authoritative
  inventories for the full Tensor method count and operation signatures.
- Runtime API and Training API remain unchanged because this task adds model construction only:
  there is no prepared execution, sampling, training-session, optimizer, or gradient API.
  Architecture pages, ADRs, and architecture tests remain unchanged because module ownership,
  dependency direction, and lifecycle boundaries are preserved. Backend conformance and
  integration tests remain unchanged because no backend or end-to-end execution behavior exists.
  Gradle/dependencies and other modules remain unchanged because the implementation uses only
  existing model contracts and Java dependencies.

Checkpoint evidence:

- After the implementation and documentation passes, the coordinator ran `./gradlew test`:
  `BUILD SUCCESSFUL` in 1s; 36 actionable tasks, 2 executed and 34 up-to-date; configuration cache
  reused. No file or executable Java changed after the successful documentation validation and
  before this checkpoint.
- The completion sync changed only the four already-authorized planning paths. A final lightweight
  pass confirmed valid local links/anchors in those files, balanced fences, final newlines, no
  trailing whitespace, exactly 19 overall paths, 0019B1 Complete, 0019C–0019E Draft, no model
  Ready/Review-needed row, no later specification file, clean status scope, and a passing
  `git diff --check`. Java tests, Javadoc, the example, and the repository checkpoint were not
  repeated.

## Implementation notes

- Added `DropoutAttrs`, `DropoutKind.DROPOUT`, public `DropoutResult`, the package-private
  construction helper, and `Tensor.dropout(double, GraphRngState)` inside `modules/model`.
- One successful call records exact ordered inputs `[input, stateTensor]`, creates output/mask/state
  descriptors and three indexed wrappers under one producer, returns slots zero and two, and leaves
  slot one's auxiliary BOOL descriptor producer-visible without a public mask.
- Added focused semantic/expression coverage and synchronized the global signature and Tensor
  public-surface inventories, including the authorized stale MATMUL inventory count repair.
- Finalized current API documentation, glossary terminology, Javadocs, and planning status while
  retaining the explicit no-execution/no-gradient/no-algorithm boundaries.

## Completion summary

- Completed changes: explicit-state training-dropout model construction, three-output provenance,
  public value/next-state result, validation and inventory tests, Javadocs, API/glossary coverage,
  and synchronized planning evidence.
- Files changed or created: exactly the six production, six test, and seven documentation/planning
  paths listed under Affected files.
- Tests and validation: final model suite passed in the implementation context; documentation
  validation passed in the clean documentation context; the coordinator's repository-wide
  `./gradlew test` checkpoint passed with 36 actionable tasks.
- Documentation-agent review: clean context
  `/root/task_0019b1_implementation/task_0019b1_docs`; executable Java was not changed.
- Documentation impact: Tensor and Compile API now describe current construction and the planned
  capture/autograd boundary. Runtime and Training API remain unchanged because no executable or
  training-extension API was added.
- Javadoc review: all six authorized production paths reviewed and finalized.
- Glossary impact: added dropout/inverted dropout, auxiliary mask, state threading, and bounded
  replay distinctions.
- Unresolved issues: None.
- Follow-up required: None. Tasks 0019C–0019E remain Draft without detailed specifications; the
  next frontier requires a separate planning step.

Status: Complete

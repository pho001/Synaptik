# Task 0015: Static Packed RNN Sequence

## Status

Complete

## Goal

Add one final public vanilla-RNN sequence container that statically unrolls a time-major Tensor
expression from explicit construction-time sequence lengths. At each represented time step it
gathers only rows whose sequence is still active, invokes the owned `RnnCell` once on that compact
batch, exposes the exact compact step outputs, and restores each sequence's final hidden row to
original batch order. A zero Tensor value never means padding.

This task also records the recurrent-scan decision forced by current contracts:

```text
static input Shape + caller Java long[] lengths
  -> construction-time active-row sets are known
  -> ordinary SELECT + eager integer indices + GATHER + RnnCell + STACK can express the graph

runtime Tensor lengths or mask
  -> active rows and carried-state Shape depend on runtime values
  -> WHERE would still construct cell work for padded rows
  -> a genuine recurrent Model scan/control-flow contract is required and remains deferred
```

The first container is deliberately cell-specific. `RnnCell` and `GruCell` each carry one hidden
Tensor, while `LstmCell` carries hidden and cell Tensors and returns a distinct result type. No
current consumer proves a shared sequence abstraction that preserves those signatures without
state erasure. Later Draft tasks may reuse the proven packing policy for GRU and then extend it
truthfully for LSTM final hidden and cell-state capture.

## Scope

- Add final public `io.github.pho001.synaptik.nn.layers.RnnSequence` extending
  `io.github.pho001.synaptik.nn.module.Module` directly.
- Add final public record `io.github.pho001.synaptik.nn.layers.RnnSequenceForwardResult` with
  exactly the components `List<Tensor> packedOutputs` and `Tensor finalHidden`, in that order.
- Add exactly the constructor, cell accessor, and three-argument forward method in the public API
  table below. Add no overload, builder, options object, generic recurrent interface, functional
  facade, direction enum, size getter, mask overload, or default state.
- Permanently register the exact caller-supplied `RnnCell` as the sole child under local name
  `cell`. The container declares no direct parameter or buffer and retains no sequence state.
- Accept a time-major input with exact fully static rank-three Shape
  `[time, batch, inputSize]`, an exact fully static rank-two initial hidden Shape
  `[batch, hiddenSize]`, and one snapshotted Java `long[]` length per original batch row.
- Require every length to be in `[0, time]` and the maximum length to fit Java list indexing
  (`<= Integer.MAX_VALUE`). Lengths are explicit metadata and are never inferred from input
  values, hidden values, labels, storage, gradients, or padding contents.
- Clone the caller's lengths before traversing them. Do not retain either the caller array or the
  clone after forward construction returns.
- Use stable active-batch compaction without sorting: step `t` contains original row indices `b`
  in ascending order exactly when `lengths[b] > t`. The result must therefore preserve original
  relative order at every step and need no inverse permutation.
- Statically unroll exactly `max(lengths)` cell steps. At each step select the time slice, gather
  its active original rows, gather the corresponding carried hidden rows, and call the owned cell
  once on Shapes `[activeCount, inputSize]` and `[activeCount, hiddenSize]`.
- Create ordinary eager dense-contiguous unlabeled provenance-free INT64 index leaves with
  `requiresGrad == false` for each required gather. Index leaves are Model data, not Parameters,
  Buffers, module state, public packing metadata, or retained configuration.
- At step zero, use one active-original-index leaf for both input and initial-hidden gathers. At
  every later step, use one active-original-index leaf for input and one relative-survivor-index
  leaf for the previous packed hidden. Do not special-case identity gathers; the fixed producer
  structure keeps compaction provenance uniform and testable.
- Return an immutable snapshot with exactly one packed output Tensor per constructed time step.
  `packedOutputs.get(t)` is the exact `RnnCell.forward` result shaped
  `[activeCount(t), hiddenSize]`, with rows in ascending original batch order.
- Restore final hidden state in original batch order. For row `b`, use the exact row selected from
  `initialHidden` when `lengths[b] == 0`; otherwise select the row corresponding to `b` from
  `packedOutputs.get(lengths[b] - 1)`. Stack those rows on axis zero.
- When the batch is empty or every length is zero, return the exact `initialHidden` reference as
  `finalHidden` and an empty packed-output list, creating no Tensor or cell expression.
- Complete caller-controlled descriptor, length, type, feature, and step-shape validation before
  creating the first index leaf or expression. Once construction begins, preserve the exact
  no-rollback side effects described below.
- Keep forward mode-insensitive. It accepts no `ForwardContext`, reads neither container nor cell
  mode, and does not mutate mode or module state. Existing recursive `train()`/`eval()` calls
  still propagate through the registered child normally.
- Preserve stable child identity, cell parameter replacement behavior, recursive discovery, and
  state-dictionary paths through existing `Module`, `Parameter`, and `RnnCell` contracts.
- Add focused exact-surface/result-carrier, ownership, static packing, validation-order,
  provenance, final-state restoration, replacement, mode, zero-length, and exclusion tests.
- Add complete type, record-component, constructor, member, and package Javadocs. After executable
  work and final NN testing, use a separate clean documentation-focused context to finalize the
  Javadocs, Training API explanation, glossary impact, planning evidence, generated Javadoc, and
  documented no-change conclusions.

For lengths `[5, 3, 1]`, Java calls `cell.forward` five times, once per non-empty time step. Their
active batch extents are `[3, 2, 2, 1, 1]`, so the graph represents nine logical recurrent row
applications instead of the fifteen rows in a dense padded batch. “Nine” is not nine Java method
calls: one cell call constructs a batched expression for each active time slice.

## Out of scope

- Runtime-dynamic Tensor lengths, a Tensor Boolean mask, a Java mask overload, synthesized masks,
  sentinel lengths, padding indices, or inference of padding from zero-valued data.
- Calling a cell on the full batch and applying `where`, multiplication, zeroing, or another
  post-cell mask. Such composition would still represent padded cell work and does not satisfy
  packed-sequence semantics.
- A general recurrent Model scan, loop/body graph, carried-value tuple, dynamic active set,
  runtime-dependent result Shape, early exit, condition operation, or control-flow region.
- Reinterpreting fixed associative `CUM_SUM` or `CUM_PROD` as recurrent scan. Their operation
  contracts have no cell body, carried hidden value, active-set transition, or final-state exit
  capture.
- `GruSequence`, `LstmSequence`, shared `RecurrentCell`, shared `RecurrentSequence`, generic state
  tuple, erased collection of states, or a public `PackedSequence` abstraction. GRU and LSTM are
  later cohesive tasks after this concrete contract is proven.
- Sorting by descending length, an exposed permutation, inverse-permutation metadata, unstable
  compaction, or reordering equal or unequal length rows. This task preserves original order
  throughout.
- Dense padded output, caller-selected padding values, padding materialization, unpacking to a
  rectangular time/batch result, batch-first input, direction configuration, reverse traversal,
  bidirectionality, stacked recurrent layers, residuals, dropout, attention, or state detachment.
- Retaining initial, intermediate, or final hidden Tensors, lengths, index leaves, outputs, or a
  step counter in a field, Buffer, Parameter, state dictionary, thread-local, static, runtime
  object, session, or another hidden lifecycle.
- Default or zero initial hidden construction, state initializers, reset APIs, stateful forward
  overloads, numerical hidden mutation, or in-place compaction.
- `UnaryTensorModule`, participation in `Sequential`, an adapter into `Sequential`, or changes to
  `Sequential`, `UnaryTensorModule`, `Module`, `Parameter`, `Buffer`, or the recurrent cells.
- Backpropagation through time, gradient rule work, optimizer/session orchestration, checkpoint
  transport, serialization, graph capture, compiler scheduling, Runtime/Prepare/Engine behavior,
  backend lowering, kernels, numerical execution, or end-to-end support claims.
- A Model, Training executable API, Gradle, dependency, architecture-contract, ADR,
  architecture-test, global-roadmap, CPU, or backend source change during implementation.
- Detailed task specifications or implementation for NN 0016 or later.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Completed NN task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)
- [Completed NN task 0014: LSTM cell](0014-lstm-cell.md)
- [Completed Model task 0017H: Slice expressions](../../../modules/model/tasks/0017h-slice-tensor-expressions.md)
- [Completed Model task 0018D: Gather expressions](../../../modules/model/tasks/0018d-axis-gather-tensor-expressions.md)
- [Completed Model task 0017L: Composition expressions](../../../modules/model/tasks/0017l-tensor-composition-expressions.md)
- [Completed Model task 0023E: Cumulative scan normalization](../../../modules/model/tasks/0023e-cumulative-scan-normalization-and-product.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns the cell-specific module composition, construction-time length policy, and NN result
  carrier. Model remains the sole owner of SELECT, GATHER, STACK, eager leaves, type promotion,
  Shape algebra, descriptors, layouts, provenance, Tensor identity, and cell operation semantics.
- Tensor identity, descriptors, storage association, and provenance remain immutable. The
  sequence container constructs new leaves and expressions but neither evaluates nor mutates
  them.
- Recurrent hidden state is caller-threaded Tensor data, never module-owned persistent state. The
  registered `RnnCell` owns only its trainable parameters.
- `Module` retains no universal forward method. `RnnSequence` is a direct subclass with a
  truthful input/state/length signature and must not extend `UnaryTensorModule`.
- `Sequential` remains a container only for `UnaryTensorModule`; no adapter may capture, erase,
  default, or synthesize recurrent state or lengths.
- Static Java lengths are construction metadata, not graph values. Their snapshot may decide how
  many expressions and what static Shapes to construct. A runtime Tensor mask cannot make those
  same decisions under the current storage-free expression contract.
- Current `CUM_SUM` and `CUM_PROD` accept one Tensor and fixed associative operation semantics.
  They cannot express an arbitrary cell body with carried hidden state, active-batch compaction,
  or exit-state capture and must not be reused or renamed for this purpose.
- `RnnSequenceForwardResult` is an NN composition value, not a Model multi-output producer,
  Tensor, module state, checkpoint payload, runtime result, or execution handle.
- Construction produces eager host-backed index leaves and storage-free expression metadata only.
  It does not prove graph capture, gradient implementation, compiled execution, backend support,
  numerical values, skipped kernels, or storage residency. “Skipped padded work” means no cell
  expression contains a padded logical row.
- If implementation needs a new Model method or operation, runtime Tensor mask/length input,
  dynamic Shape, scan/body contract, shared recurrent abstraction, state Buffer, dependency,
  architecture rule, or a path outside the exact task scope, stop and report the exact blocker
  instead of widening this task.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.layers`
- `io.github.pho001.synaptik.nn.module`
- `io.github.pho001.synaptik.model.tensor`
- existing Model descriptor, data-type, Shape, and layout packages needed by eager INT64 leaves

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — two public types and existing package documentation only

Type placement:

- `io.github.pho001.synaptik.nn.layers.RnnSequence` — NN owns the cell-specific module
  composition and construction-time packing policy.
- `io.github.pho001.synaptik.nn.layers.RnnSequenceForwardResult` — NN owns the typed result of
  that composition; it is not a Model producer result or runtime value.

No package is added. `nn.module`, Model packages, Training, compiler/runtime/prepare/engine, and
backend packages remain unchanged.

## Public API

`RnnSequence` declares exactly:

```java
public RnnSequence(RnnCell cell)

public RnnCell cell()

public RnnSequenceForwardResult forward(
        Tensor input,
        Tensor initialHidden,
        long[] lengths)
```

| Member | Exact contract |
|---|---|
| `RnnSequence(cell)` | Permanently registers and retains the exact available non-null cell under child name `cell`; declares no direct state. |
| `cell()` | Returns the exact stable registered child reference. |
| `forward(input, initialHidden, lengths)` | Snapshots and validates static inputs and lengths, constructs stable compact time-step expressions, and returns immutable packed outputs plus original-order final hidden state. |

The class declares no other public or protected constructor, method, field, nested type,
interface, or overload. Inherited final `Module` APIs remain available normally.

`RnnSequenceForwardResult` declares exactly:

```java
public record RnnSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor finalHidden)
```

Its canonical constructor rejects null `packedOutputs`, traverses once to reject the first null
element, captures an immutable structural snapshot retaining every exact Tensor reference, and
then rejects null `finalHidden`. A success performs no descriptor relationship validation,
expression construction, Tensor copy, mutation, evaluation, or Tensor-ID allocation. Generated
record accessors expose the immutable list and exact final-hidden reference. The record declares
no additional public method or nested type beyond its canonical constructor and ordinary
generated accessors, `equals`, `hashCode`, and `toString`.

## State schema and ownership

`RnnSequence` declares no direct Parameter or Buffer. Its sole child is the exact supplied cell:

| Kind | Local name | Value |
|---|---|---|
| Child | `cell` | exact caller-supplied `RnnCell` |

Direct `parameters()` and `buffers()` are empty. `children()` contains exactly `cell` in its sole
entry. Recursive parameter and state-dictionary paths are `cell.inputWeight`,
`cell.hiddenWeight`, and optional `cell.bias`, preserving the cell's declaration order. Input,
initial hidden, lengths, eager index leaves, selected time slices, compact hidden values, packed
outputs, and final hidden never enter module discovery or a state dictionary.

Construction rejects null `cell` before child registration. Ordinary `Module.child` validation
then rejects a previously owned cell and preserves existing ownership. A successful constructor
does not copy the cell or synchronize its pre-existing mode with the new parent's initial mode.
Later `train()` and `eval()` calls on the sequence propagate through the child under existing
Module semantics. Forward is mode-insensitive.

## Forward input contract

The caller supplies:

| Input | Required contract |
|---|---|
| `input` | non-null floating fully static rank-three Tensor shaped `[time, batch, inputSize]` |
| `initialHidden` | non-null floating fully static rank-two Tensor shaped `[batch, hiddenSize]` |
| `lengths` | non-null Java array with exactly `batch` elements, each in `[0, time]`, with maximum at most `Integer.MAX_VALUE`; snapshotted and not retained |

`time` and `batch` may be zero. `inputSize` and `hiddenSize` must equal the positive static sizes
configured by the current cell weights. Input and hidden types may differ only when the complete
cell projections and addition accept them under existing Model numeric promotion. Both must be
floating. Fully static Shapes are required because Java lengths decide concrete time traversal,
active batch extents, eager index storage, and final row restoration before runtime.

The caller owns input, initial hidden, and the original lengths array. Forward does not mutate,
retain, label, evaluate, copy, or attach either Tensor. It clones the array so mutation racing
after the snapshot cannot change the active-set plan. The method provides no synchronization with
concurrent mutation of the caller array before cloning or concurrent cell parameter replacement.

## Packing and final-state semantics

For snapshotted length `L[b]`, define:

```text
steps = max(L), or zero for an empty batch
active(t) = [b in 0..<batch, in ascending order, where L[b] > t]
activeCount(t) = size(active(t))
```

For each `t` from zero through `steps - 1`:

1. select `input` axis zero at `t`, yielding `[batch, inputSize]`;
2. create one INT64 eager leaf holding `active(t)`;
3. gather the selected input on axis zero with that leaf, yielding
   `[activeCount(t), inputSize]`;
4. for `t == 0`, gather `initialHidden` on axis zero with the same active-original leaf;
5. for `t > 0`, create one INT64 leaf whose entries are the positions in `active(t - 1)` occupied
   by the surviving original rows in `active(t)`, then gather the prior packed output on axis
   zero, yielding `[activeCount(t), hiddenSize]`;
6. call `cell.forward(compactInput, compactHidden)` exactly once; and
7. append that exact returned Tensor to the result list.

Because `active(t)` is a subsequence of `active(t - 1)`, relative survivor positions are defined
without sorting. Every padded row `L[b] <= t` is absent from both compact operands and therefore
from that cell expression. Input values equal to numeric zero remain ordinary active data when
their length says they are active.

Final hidden Shape is `[batch, hiddenSize]`, in original batch order. For each original row:

```text
L[b] == 0  -> initialHidden.select(0, b)
L[b] > 0   -> packedOutputs[L[b] - 1].select(0, position of b in active(L[b] - 1))
```

When at least one length is positive, stack those exact rank-one row selections on axis zero in
`b = 0..<batch` order. When no length is positive, return the exact `initialHidden` reference and
construct neither selections nor a stack.

Examples:

| Lengths | Active original rows by time | Final row sources |
|---|---|---|
| `[5, 3, 1]` | `[0,1,2]`, `[0,1]`, `[0,1]`, `[0]`, `[0]` | row 0 from step 4; row 1 from step 2; row 2 from step 0 |
| `[1, 3, 2]` | `[0,1,2]`, `[1,2]`, `[1]` | row 0 from step 0 position 0; row 1 from step 2 position 0; row 2 from step 1 position 1 |
| `[0, 2, 0]` | `[1]`, `[1]` | rows 0 and 2 from initial hidden; row 1 from step 1 |
| `[0, 0]` | none | exact initial-hidden Tensor |

The result is packed by time step rather than flattened into one Tensor. Step boundaries and
active extents therefore remain explicit without a second public batch-size array or a generic
packed-sequence value. It contains no padded dense output.

## Validation and side-effect order

`forward(input, initialHidden, lengths)` performs exactly:

1. reject null `input`, then null `initialHidden`, then null `lengths`;
2. clone `lengths` and use only that snapshot afterward;
3. read the current cell input-weight, hidden-weight, and optional-bias bindings once for
   descriptor preflight only, in cell declaration order;
4. revalidate the current cell parameter schema and derive exact positive `inputSize` and
   `hiddenSize` without creating a Tensor;
5. require input floating type, rank exactly three, and fully static Shape;
6. require initial-hidden floating type, rank exactly two, and fully static Shape;
7. require input axis two to equal `inputSize` and initial-hidden axis one to equal `hiddenSize`;
8. require the input and initial-hidden batch extents to be equal;
9. require snapshotted length count to equal batch, then validate each element in original order
   as non-negative and no greater than time while deriving the maximum, and require that maximum
   to be at most `Integer.MAX_VALUE` for immutable Java-list indexing;
10. prevalidate the exact step-zero and survivor compact Shapes, cell projection promotion, ADD
    broadcast, and TANH eligibility for every distinct active count required by the snapshot;
11. prevalidate every final row selection and, when needed, final STACK Shape/type contract; and
12. only after all caller-controlled checks succeed, construct eager indices and expressions in
    the fixed packing order above.

Validation must use Model-owned helpers or equivalent non-constructing descriptor algebra. It
must not call `cell.forward` merely to discover whether a Shape is valid. Every null, rank,
dynamic-Shape, type, feature, batch, length-count, length-range, promotion, broadcast, select,
gather, or stack failure is detected before the first new Tensor identity, eager storage
allocation, or expression producer.

The descriptor preflight snapshot does not make the entire sequence call atomic with concurrent
parameter replacement. Each actual `cell.forward` call observes the cell's then-current bindings
under its existing contract; external replacement during one sequence construction can therefore
make later steps use different compatible parameter Tensors. Callers must coordinate replacement
and forward construction when a consistent multi-step parameter snapshot matters.

After construction begins, successful eager index leaves, SELECT/GATHER producers, cell
expression prefixes, row selections, and host allocations are not rolled back if Tensor-ID
exhaustion, allocation failure, or another unexpected delegated failure occurs later. No partial
result carrier is returned. The container itself retains none of the created values after the
call exits except through references reachable from the returned result expressions.

## Exact expression and provenance contract

- Every time step begins with one `input.select(0, t)` whose exact input is the caller's input.
- Each active-original index is an independent eager INT64 leaf shaped `[activeCount(t)]`, dense
  contiguous, unlabeled, host-backed, gradient-ineligible, and provenance-free.
- Step-zero input and initial-hidden gathers share that step's exact active-original index leaf.
- Each later relative-survivor index is an independent eager INT64 leaf shaped
  `[activeCount(t)]` with the same metadata policy.
- Every gather uses axis zero and preserves exact source/index references in ordinary Model
  provenance.
- Every packed output is exactly the TANH result returned by the owned cell, not a wrapper,
  relabel, copy, extra identity expression, or dense padded reconstruction.
- Final row selections use axis zero and exact source positions. A nontrivial final hidden is the
  exact axis-zero STACK result over original-order row selections.
- The container creates no direct `Operation`, producer, descriptor, or Tensor wrapper outside
  public Model construction APIs.

This provenance specifies expression metadata only. It does not assert numerical results,
gradient traversal, compiler capture, backend lowering, kernel count, work scheduling, or actual
runtime avoidance of a padded kernel.

## Affected files

Implementation is limited to these nine paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequenceForwardResult.java`
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
4. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnSequenceTest.java`
5. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnSequencePackingTest.java`
6. `docs/api/training-api.md`
7. `docs/glossary.md`
8. `docs/planning/extensions/nn/tasks/0015-static-packed-rnn-sequence.md`
9. `docs/planning/extensions/nn/master-plan.md`

Do not edit `ARCHITECTURE.md`, architecture explanations, ADRs, Tensor API, Training Java source,
other tasks/master plans, global roadmap, build files, Model source/tests, recurrent-cell source,
architecture tests, conformance tests, integration tests, compiler/runtime/prepare/engine code, or
backend code. If a required change falls outside these nine paths, stop and report it.

## Maximum scope

This task may create or modify at most the nine exact paths listed above. If a tenth path, a new
package, or a change to an excluded path is needed, stop and propose a follow-up task or report an
architecture/API blocker.

## Test requirements

### Public surface and result value

- Verify `RnnSequence` is final, extends `Module` directly, and declares exactly its constructor,
  accessor, and forward method with no public field, overload, nested type, mask input, length
  Tensor, or unary signature.
- Verify the result record declares exactly `packedOutputs` then `finalHidden`, rejects null list,
  first null list element, then null final hidden, snapshots a mutable source list, exposes an
  unmodifiable list, and retains every exact Tensor reference.
- Verify generated record equality remains ordinary list/Tensor value equality and no custom
  behavior or descriptor coupling is added.

### Ownership, state, replacement, and mode

- Verify the constructor registers the exact cell under `cell`, exposes it through `cell()`, and
  leaves direct parameters/buffers empty.
- Verify null and already-owned cells fail under documented order without changing existing
  ownership.
- Verify recursive parameter, state-dictionary, and child paths/order are exactly the existing
  cell schema prefixed by `cell.`.
- Verify `train()` and `eval()` propagate to the child; forward is identical in both modes and
  does not alter mode or state.
- Verify compatible cell-parameter replacement affects later sequence calls while an already
  constructed result retains its earlier exact expression inputs.

### Static packing and restoration

- For `[5, 3, 1]`, verify five packed outputs with active Shapes
  `[3,H]`, `[2,H]`, `[2,H]`, `[1,H]`, `[1,H]`; active extents total nine rather than dense fifteen;
  and final rows select step/position `(4,0)`, `(2,1)`, `(0,2)` before original-order stacking.
- For unsorted `[1, 3, 2]`, inspect index host data and provenance to verify active-original sets
  `[0,1,2]`, `[1,2]`, `[1]`, relative survivor sets `[1,2]`, `[0]`, no sort/permutation, and
  final row sources `(0,0)`, `(2,0)`, `(1,1)` in original order.
- Verify an active input row whose stored values are all numeric zero remains present. No storage
  read or value-based padding decision may occur.
- Verify `[0,2,0]` gathers only original row one, uses initial-hidden rows zero and two for final
  state, and constructs no cell operand for either padded row.
- Verify all-zero lengths, zero time with valid all-zero lengths, and an empty batch each return an
  empty immutable list plus the exact initial-hidden reference with no new Tensor ID.
- Verify final output Shapes/types and fixed SELECT/GATHER/cell/STACK producer order without
  evaluating numerical values.

### Validation and failure effects

- Cover null order, non-floating input/hidden, wrong ranks, every dynamic input/hidden axis,
  feature mismatch, batch mismatch, length-count mismatch, first negative length, first length
  above time, maximum length above `Integer.MAX_VALUE`, and mixed-type promotion failure.
- Verify every caller-controlled failure occurs before the first Tensor ID or producer and leaves
  module state, cell state, and caller values unchanged.
- Verify the lengths array is snapshotted rather than retained or reread after planning.
- Verify failures after valid construction begins preserve already created eager leaves and
  expression prefixes without returning a result or mutating module state.

### Exclusions

- Verify there is no Tensor-length/mask overload, Java mask overload, retained length/index/output
  field, Buffer state, hidden default, dense padded output, `Sequential` compatibility, shared
  recurrent abstraction, direct Model `Operation` construction, or dependency outside Model.
- Verify `CUM_SUM` and `CUM_PROD` kinds never occur in constructed sequence provenance.

## Documentation requirements

The implementation context may draft complete public and package Javadocs while coding. After
the final executable NN test run, a separate clean documentation-focused context must read the
final source, tests, generated Javadocs, documentation rules, General/API-Javadoc/Planning/Example
profiles, Training API, glossary, NN master/task, completed cell tasks, and relevant Model
SELECT/GATHER/STACK/cumulative-scan contracts. It must finalize:

- class, record, constructor, accessor, forward, component, failure, ownership, mode,
  concurrency, side-effect, and provenance Javadocs;
- `layers/package-info.java` with static packed sequence semantics and its distinction from
  one-step cells, `Sequential`, gate-parameter packing, and runtime scan;
- `docs/api/training-api.md` with the current NN public surface, a short `[5,3,1]` example, exact
  result interpretation, and current-vs-planned execution boundary;
- `docs/glossary.md` definitions for static packed recurrent sequence, active batch,
  construction-time lengths, and recurrent scan, without implying zero-value padding; and
- final planning evidence, no-change conclusions, links, anchors, status, and completion summary.

The documentation pass reuses stable Java test evidence unless it changes executable Java
behavior. It must state reasoned no-change conclusions for `ARCHITECTURE.md`, architecture
explanations/ADR/tests, Tensor API, Training Java API and training graph, Model capabilities,
compiler/runtime/prepare/engine, backend conformance, integration tests, Gradle dependencies,
global roadmap, and other modules.

## Acceptance criteria

- The exact two-type public API exists with no extra overload or abstraction.
- The sequence owns exactly one `RnnCell` child and no direct or hidden state.
- Fully static time-major input plus snapshotted Java lengths constructs one compact cell
  expression per non-empty time step.
- Packed outputs preserve stable original row order, exclude padded logical rows, and expose exact
  step boundaries without a dense padded result.
- Final hidden rows come from each sequence's exit step, with zero-length rows from initial hidden,
  and are restored to exact original batch order.
- Zero-valued data is never interpreted as padding; only explicit lengths control activity.
- All-zero, zero-time, and empty-batch requests have the documented exact-reference no-expression
  result.
- Caller-controlled validation precedes every new Tensor ID and expression; later failures retain
  documented prefixes without module mutation.
- Result collection ownership, cell ownership, parameter replacement, mode, state paths, and
  concurrency limits match existing contracts.
- No runtime Tensor mask/length promise, recurrent scan, cumulative-scan misuse, dense masking,
  shared recurrent abstraction, GRU/LSTM sequence, execution claim, dependency change, or
  out-of-scope file is introduced.
- Focused and full NN tests, NN Javadocs, API-surface/import/dependency checks, Markdown/link gates,
  exact-scope checks, and whitespace validation pass.
- A separate clean documentation pass finalizes all affected documentation and records no-change
  conclusions before the task becomes Complete.

## Tests / validation

### Implementation context

Run once after executable source and tests stabilize:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.layers.RnnSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.RnnSequencePackingTest

./gradlew :extensions:nn:test
```

Record test/suite counts and failures, errors, and skips. Do not repeat the full NN suite after
documentation-only edits.

### Documentation context

Run after final documentation edits:

```bash
./gradlew :extensions:nn:javadoc

rg -n "RnnSequence|packed recurrent|active batch|recurrent scan|construction-time lengths" \
  extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers \
  docs/api/training-api.md \
  docs/glossary.md \
  docs/planning/extensions/nn

rg -n "extensions\.training|compiler|runtime|prepare|engine|backends" \
  extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence*.java

git diff --name-only
git diff --check
```

Also perform:

- reflection or `javap` verification of the exact two-type public surface and absence of retained
  length/output/index fields;
- import/dependency inspection proving NN production still imports only Java and Model/NN types;
- generated-Javadoc inspection for every public/protected member, record component, `@param`,
  `@return`, `@throws`, nullability, ownership, failure, and side-effect contract;
- Markdown heading/anchor/link/fence validation for changed documentation;
- exact nine-path scope verification;
- master-plan frontier verification: 0015 is Complete, while 0016 and later rows are concise Draft
  entries without task specifications;
- newline-at-EOF and whitespace checks.

Repository-wide tests are not required: this task changes one leaf extension and explanatory
documentation without changing module dependencies, shared build configuration, architecture
rules, Model behavior, or another module. Architecture, backend-conformance, and integration
tests are not required for the same reason.

## Dependencies

- NN 0011 complete: direct versus unary composition and `Sequential` exclusion are fixed.
- NN 0012 complete: exact vanilla-RNN cell signature, state, Shape, promotion, provenance, and
  replacement contracts are available.
- NN 0013 and 0014 complete: GRU and LSTM prove that a shared sequence/state API is not yet
  justified and define later cell-specific follow-up requirements.
- Model SELECT, GATHER, STACK, eager INT64 leaf, static Shape, promotion, and provenance contracts
  complete.
- No CPU, compiler, runtime, Prepare, Engine, backend, optimizer, or numerical execution task is
  a prerequisite for static expression construction.

## Follow-up tasks

- NN 0016: add cell-specific static packed `GruSequence` only after 0015 validates the one-hidden-
  state packing and restoration contract.
- NN 0017: add cell-specific static packed `LstmSequence`, carrying and restoring both hidden and
  cell states as sequences leave the active set.
- A future Model task may specify a genuine recurrent scan/control-flow primitive if a concrete
  runtime-dynamic length/mask consumer supplies body, carried-value, dynamic-Shape, compiler, and
  execution requirements. It is not an automatic part of the NN sequence tasks.
- A later consumer may justify dense unpacking, batch-first views, sorting/permutation metadata,
  bidirectionality, stacking, or a shared recurrent abstraction. None is implied here.

## Architecture impact

Expected impact: None. This is a cell-specific NN composition over completed public Model
contracts and the existing `extensions/nn -> modules/model` dependency. No module boundary,
dependency direction, lifecycle owner, operation semantics, or architecture rule changes.

If implementation requires an architecture-contract, Model operation, runtime-dynamic scan,
dependency, package-boundary, or architecture-test change, stop and report the issue.

## Implementation prompt

Use this prompt in a separate clean agentic task/thread:

```text
You are working in the Synaptik repository. Do not use any GSD skill or workflow, and do not
commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0015-static-packed-rnn-sequence.md in full, plus every directly
required contract named by that task.

Implement task 0015 exactly as specified in its nine-path maximum scope. Do not implement
out-of-scope items. If current public contracts cannot express the specified static composition,
or architecture/scope uncertainty appears, stop and report the exact blocker instead of inventing
architecture.

After executable implementation and the final NN module validation, hand the diff and recorded
test evidence to a distinct clean documentation-focused agent/thread. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize affected Javadocs,
documentation, glossary, planning evidence, and no-change conclusions in the same overall change,
and must not repeat successful Java tests unless executable behavior changes or a concrete risk is
recorded.

At the end, update the task with local decisions, known limitations, validation evidence,
implementation notes, completion summary, and final status. Do not mark it Complete before the
documentation pass and every exact-scope gate finish.
```

## Local decisions

- Use direct `RnnSequence extends Module` with one exact owned `RnnCell`; no shared recurrent
  abstraction is introduced because the completed cell signatures still differ materially.
- Represent packed output as one immutable list entry per time step. This preserves static active
  extents and exact step boundaries without a second batch-size array or dense padded Tensor.
- Keep original batch order throughout. Each later hidden gather uses positions relative to the
  prior active set, and final row positions are captured when a row exits rather than recovered by
  sorting or an inverse permutation.
- Build each INT64 gather index as one ordinary eager dense-contiguous unlabeled Model leaf. The
  sequence retains none of these leaves; ordinary expression provenance remains their only owner.
- Permit mixed floating promotion when every final row comes from recurrent output. If a
  zero-length initial-hidden row would have a different exact type from recurrent rows, reject the
  final STACK contract before constructing any Tensor.

## Known limitations

- Lengths must be construction-time Java values and all Tensor Shapes must be fully static.
- The returned outputs are compact by step and cannot be interpreted as one dense rectangular
  sequence without a later explicit unpacking capability.
- Parameter replacement and a multi-step forward construction are not an atomic snapshot. Each
  cell call observes its current compatible bindings under the existing cell contract.
- Construction expresses compact logical rows but does not claim compiler, backend, numerical,
  gradient, scheduling, or physical kernel support.
- Runtime Tensor lengths or masks remain blocked on a genuine Model recurrent scan/control-flow
  design; cumulative sum and product do not provide that contract.

## Validation evidence

- Clean implementation context `/root/nn_0015_implementation` read the repository instructions,
  architecture and planning contracts, documentation profiles, NN recurrent history and current
  task, Module/state/cell APIs, and relevant Model Tensor/Shape/layout/composition contracts before
  editing.
- The first focused development run exposed one incorrect test assertion about the existing
  GATHER axis-attribute record; production behavior was correct and the test was corrected to the
  exact `IndexAxisAttrs(0)` contract.
- The stabilized focused command
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.RnnSequenceTest --tests io.github.pho001.synaptik.nn.layers.RnnSequencePackingTest`
  passes 2 suites and 14 tests with no skips, failures, or errors.
- After executable Java and tests stabilized, the authoritative
  `./gradlew :extensions:nn:test` passes 25 suites and 168 tests with no skips, failures, or
  errors. No executable Java or test changed afterward.
- Preliminary `./gradlew :extensions:nn:javadoc` passed before final documentation review.
  `git diff --check` also passed at that point.
- `javap -public` shows only the exact `RnnSequence(RnnCell)`, `cell()`, and
  `forward(Tensor, Tensor, long[])` surface plus the canonical two-component record surface.
  `javap -private` shows the sequence retains only one `RnnCell cell` field and no lengths,
  indices, inputs, outputs, or state. Production import inspection found only Java, Model, and NN
  module types; the existing NN Gradle dependency remains unchanged.
- New-file no-index whitespace checks passed for both production and both focused test files.
  Worktree inspection found exactly seven current NN implementation-owned paths; the Training API
  and glossary are intentionally reserved for the required documentation pass. Concurrent CPU
  source/test/planning work and the global roadmap remain untouched.
- Focused tests cover the exact public surfaces, immutable result snapshot, sole-child ownership,
  recursive state paths, mode propagation and independence, compatible replacement, all-zero,
  zero-time and empty-batch no-ID results, null/type/rank/every-dynamic-axis/feature/batch/length/
  mixed-stack failures before an expression ID, stable `[5,3,1]` and unsorted `[1,3,2]` packing,
  `[0,2,0]` restoration, copied lengths, exact eager-index metadata and host data, fixed one-step
  ID/provenance order, and absence of cumulative-scan operations.
- Repository-wide, architecture, Model, backend-conformance, and integration tests were not run:
  the task changes one leaf extension, adds no dependency or Model/backend behavior, and the task
  assigns final documentation gates to the independent documentation context.
- Independent clean documentation context `/root/nn_0015_docs` read the final implementation and
  tests, repository instructions, architecture/planning/documentation contracts, recurrent and
  module-state boundaries, relevant Model SELECT/GATHER/STACK/eager-index/cumulative-scan
  contracts, Training and Compile API boundaries, Model capabilities, the Training graph, the
  central glossary, generated Javadocs, and the exact diff. It found no executable/API defect,
  architecture uncertainty, scope blocker, or reason to change executable Java or tests.
- The documentation context finalized `RnnSequence`, `RnnSequenceForwardResult`, and layers-
  package Javadocs. It clarified the active-batch definition, defensive Java-array snapshot and
  caller coordination, exact result ownership, no-zero-value inference, allocation failures, and
  runtime recurrent-scan boundary without adding execution, numerical, gradient, or backend
  claims.
- The Training API now records the current direct-cell signatures and the current cell-specific
  `RnnSequence` surface, fully static time-major Shapes, copied explicit lengths, stable active-
  batch compaction, packed step ordering, `[5,3,1]` example, zero-length/final-hidden restoration,
  and current-versus-planned execution boundary. Static GRU/LSTM sequence containers remain
  planned.
- The existing central `docs/glossary.md` now defines construction-time length, active batch,
  static packed recurrent sequence, and recurrent scan. The task's erroneous draft reference to
  absent `docs/reference/glossary.md` was corrected in the affected-file and validation lists;
  no duplicate glossary was created.
- The documentation pass reused the implementation context's focused two-suite/14-test result and
  authoritative 25-suite/168-test NN result because no executable Java or test changed afterward.
  Javadoc and prose edits do not stale that evidence, so no Java test suite was repeated.
- Final `./gradlew :extensions:nn:javadoc` passed after the Javadoc edits (`BUILD SUCCESSFUL`; 3
  actionable tasks, 2 executed and 1 up-to-date). Generated `RnnSequence.html`,
  `RnnSequenceForwardResult.html`, and `layers/package-summary.html` contain the exact public
  signatures, inputs, results, failures, ownership, active-row packing, defensive lengths,
  zero-value, concurrency, side-effect, mode, and non-execution boundaries.
- Final `javap -public` and `javap -private` confirmed final direct `Module`, exactly one public
  constructor plus `cell()` and `forward(Tensor, Tensor, long[])`, the ordinary two-component
  record surface, no protected surface or nested type, and only one retained `RnnCell cell` field
  on the sequence. The implementation focused tests provide the independent reflection checks for
  the same exact surfaces and absence of retained sequence fields.
- Production import inspection found only Java, Model, and existing NN types. The unchanged
  `extensions/nn` build retains only `implementation(project(":modules:model"))`; no Training,
  Compiler, Runtime, Prepare, Engine, concrete-backend, or new Gradle dependency appears.
- The targeted Markdown validator passed the Training API, glossary, NN master plan, and task with
  362 local links, 302 anchors, balanced fences, and final newlines. Exact-scope inspection found
  exactly the nine authorized NN 0015 paths while preserving concurrent CPU and global-roadmap
  work. Master/task status is Complete; NN 0016 and 0017 remain concise Draft rows without task
  specifications.
- `ARCHITECTURE.md`, focused architecture explanations, ADR 0007, and architecture tests require
  no change because this remains cell-specific Model-only NN composition on the existing
  dependency edge. Tensor and Compile APIs, the Training Java API and Training graph, Model
  source/master/capabilities, and related operation contracts require no change because the task
  only composes current metadata and adds no Model semantic, autograd, capture, optimizer, or
  runtime-dynamic control-flow contract.
- Compiler, Runtime, Prepare, Engine, CPU and other backends, backend conformance, integration
  tests, Gradle configuration, other recurrent cells/modules, the CPU master/task, and the global
  roadmap require no change because no executable lifecycle, dependency, backend behavior, or
  cross-module capability changed. Repository-wide and those specialized suites remain deferred
  to their owning checkpoints or CI.
- Final new-file no-index whitespace checks, final-newline checks, exact-scope/status/frontier
  checks, and whole-worktree `git diff --check` passed. Concurrent CPU changes were preserved and
  were neither reviewed nor modified by this context.

## Implementation notes

- Added `RnnSequence` and `RnnSequenceForwardResult` under the existing public `nn.layers`
  package, with no new package, dependency, Model operation, shared recurrent interface, mask
  overload, retained sequence state, or unary adapter.
- Forward clones lengths before descriptor planning, snapshots the three current cell bindings for
  preflight, validates all caller-controlled contracts without Tensor creation, and then emits the
  specified SELECT/index/GATHER/cell chain. Later steps gather survivor positions from the prior
  compact output.
- Exit positions are captured in a temporary primitive array as each original row leaves the
  active set. Final rows select either that exit output or initial hidden for length zero and are
  stacked in original order.
- Draft Javadocs cover ownership, Shapes, explicit lengths, no-zero inference, side effects,
  concurrency, mode, result identity, and non-execution boundaries. The package draft distinguishes
  sequence packing from gate-parameter packing, unary `Sequential`, and a runtime recurrent scan.
- No required change was found for architecture contracts, Model, Training Java source, Gradle,
  global roadmap, CPU/backend work, or tests outside `extensions/nn`.

## Completion summary

- Completed changes: stable static packed vanilla-RNN composition, immutable result carrier,
  focused tests, finalized Java/package documentation, Training API reference, glossary terms,
  and synchronized planning records.
- Files changed or created: the exact nine authorized paths—two production files, two focused test
  files, layers package Javadoc, Training API, central glossary, NN task, and NN master plan.
- Tests and validation: focused 14/14 and full NN 168/168 passed in the implementation context;
  final Javadoc, generated-page, public/private-surface, reflection, import/dependency, Markdown,
  exact-scope/status/frontier, newline, whitespace, and diff gates passed.
- Documentation-agent review: independent clean context `/root/nn_0015_docs` completed the final
  review and validation without changing executable Java or tests.
- Documentation impact: Training API now explains current static RNN sequence composition and its
  future runtime-scan boundary; architecture and unrelated API documentation remain unchanged for
  the reasons recorded above.
- Javadoc review: final public, record-component, constructor, accessor, forward, package,
  ownership, failure, side-effect, concurrency, and non-execution contracts passed generation and
  rendered-page inspection.
- Glossary impact: the existing central glossary defines static packed recurrent sequence, active
  batch, construction-time length, and recurrent scan without creating a duplicate file.
- Unresolved issues: None in executable implementation.
- Follow-up required: None for NN 0015. NN 0016 and NN 0017 remain optional ordered Draft
  cell-specific follow-ups.

Status: Complete

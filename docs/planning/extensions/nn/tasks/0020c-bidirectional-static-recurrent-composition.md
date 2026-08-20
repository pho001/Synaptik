# Task 0020C: Type-Safe Bidirectional Static Recurrent Composition

## Status

Complete

## Goal

Add the smallest coherent directional recurrent capability that current Model expressions can
represent truthfully: separate final public bidirectional sequence containers for vanilla RNN,
GRU, and LSTM. Each container owns one forward cell and one backward cell with independent
parameter trees and initialization seeds, traverses only each batch row's valid prefix in the
selected direction, and returns original-time compact outputs merged by an explicit final-axis
`CONCAT`.

Mental model:

```text
fully static [time, batch, inputSize] input + Java lengths
  -> forward cell visits x[0], x[1], ... x[length - 1]
  -> backward cell visits x[length - 1], ... x[1], x[0]
  -> neither direction visits the right-padded suffix
  -> backward hidden outputs are realigned to original time and batch order
  -> packedOutputs[t] = concat(forwardOutput[t], backwardOutput[t], axis=-1)
  -> directional final states remain separate and retain original batch order
```

For original time `t`, both directional halves contain exactly the rows whose length exceeds
`t`, in ascending original batch order. The backward half for one such row is the state after the
backward cell has consumed that row's valid suffix from `length - 1` down through `t`. Forward
and backward final hidden states are distinct. LSTM additionally returns the corresponding
directional final cell states.

This task deliberately narrows the former broad “bidirectional and multidirectional” row to the
only distinct traversal pair defined by the current single time axis. It does not introduce an
arbitrary direction collection, generic recurrent base, or stacked/multidimensional recurrence.

## Motivation

The three current static sequence families already prove explicit-state packing, stable original
row order, zero-length handling, automatic cell initialization, strict state loading, and
cell-specific results. Directionality adds four decisions that must not remain implicit:

- forward and backward cells must be different owned module identities with separate state paths;
- backward traversal must reverse each row's valid prefix, never the complete padded time axis;
- per-time backward outputs must be restored to original time and batch order before merging; and
- the merge and directional final-state result shape must be explicit and cell-family-specific.

Current Model `GATHER_ND`, axis `GATHER`, `SELECT`, and `CONCAT` expressions are sufficient. A
whole-axis `flip(0)` is not sufficient because it would move right padding ahead of valid data for
short rows. Dense `WHERE` masking is not sufficient because it would still construct padded cell
work. `SUM` is not selected because it discards the two directional feature subspaces and requires
an equal-Shape interpretation that the first API does not need. Final-axis `CONCAT` retains both
halves visibly and is already a current differentiable Model operation.

## Scope

- Add final public `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
  `BidirectionalLstmSequence` classes in `io.github.pho001.synaptik.nn.layers`. Each extends
  `Module` directly and has only its family-specific typed cell and result contracts.
- Add final public `BidirectionalRnnSequenceForwardResult`,
  `BidirectionalGruSequenceForwardResult`, and `BidirectionalLstmSequenceForwardResult` records.
- Give every container one constructor accepting exact caller-supplied forward and backward cells
  and one constructor creating two automatic cells from an explicit common hidden size, bias
  choice, floating type, `ParameterInitialization`, forward seed, and backward seed.
- Require the two supplied cells to be non-null, identity-distinct, and configured with the same
  hidden size and exact parameter data type. Bias presence may differ for supplied cells. Input
  width is validated for both cells against each forward call because an automatic cell may still
  be unbound and supplied cells can carry independently constructed input schemas.
- Permanently register the forward cell under exact child name `forward` and the backward cell
  under exact child name `backward`, in that order. Complete two-child validation must precede
  either parent link.
- Widen the existing already-tested package-private
  `Module.registerNamedChildren(Map<String, ? extends Module>)` primitive to `protected final` so
  a layer-package subclass can atomically install the two descriptive children. Do not add a
  second named-child primitive, public child mutation, detach/reparent operation, or generic
  recurrent owner.
- Keep forward and backward parameters independent. Automatic construction creates two distinct
  cell objects, each with its own private reservation group and exact supplied seed. Equal numeric
  seed values are allowed but are still two separately owned initialization configurations and
  create distinct parameter Tensor identities.
- Preserve each cell's existing parameter order, gate packing, formula, optional bias policy,
  automatic first-use binding, retry, strict-load, replacement, and mode-insensitive behavior.
- Add the four forward overload forms already proven by the one-directional sequences: explicit
  states plus lengths, explicit states with all-valid derived lengths, default states plus
  lengths, and default states with all-valid derived lengths.
- Keep the fully static time-major input contract `[time, batch, inputSize]` and static Java
  `long[]` length compatibility contract. Lengths remain construction metadata, not Tensor values.
- Forward traversal at original time `t` uses the current stable active set
  `L[b] > t`, axis-zero input `SELECT`, active-row index leaf, axis-zero `GATHER`, survivor gather,
  and one concrete forward-cell call.
- Backward traversal at reverse depth `d` uses the same active set `L[b] > d`. For each active row
  `b`, its compact input coordinate is `(L[b] - 1 - d, b)`. One dense INT64 coordinate leaf shaped
  `[activeCount(d), 2]` and one `input.gatherNd(coordinates)` occurrence construct the compact
  `[activeCount(d), inputSize]` reverse input without touching a padded position.
- Carry backward state between reverse depths through the same stable survivor-position policy as
  the current forward packing. Invoke the backward cell exactly once for every non-empty reverse
  depth. LSTM shares each exact active/survivor index leaf between hidden and cell gathers.
- Flatten the reverse-depth hidden outputs in increasing reverse-depth order with one axis-zero
  `CONCAT` when more than one reverse step exists; retain the exact sole output when there is one.
  For each original time `t`, build one dense INT64 alignment index leaf and one axis-zero
  `GATHER` from that flattened Tensor so backward rows return to `active(t)` order. This is one
  alignment expression per represented time step, not one expression per logical row.
- Merge each exact forward packed output first and its exact aligned backward output second with
  `Tensor.concat(-1, forwardOutput, backwardOutput)`. Each merged output has Shape
  `[activeCount(t), 2 * hiddenSize]` and exact common promoted floating type.
- Restore all directional final states to original batch order with shared dense INT64 final-row
  indices and axis-zero `CONCAT`/`GATHER` composition over initial state followed by flattened
  directional step states. RNN/GRU return forward then backward final hidden. LSTM returns forward
  final hidden/cell followed by backward final hidden/cell.
- Preserve exact initial-state references when no length is positive. Default-state overloads
  create distinct fresh zero leaves in documented direction/state order and return those exact
  references for an all-zero request. Neither cell initializes when no step is represented.
- Complete all caller-controlled constructor and forward validation before child installation,
  default-state/index allocation, automatic initialization, or expression creation as applicable.
- Add focused public-surface, ownership/state-path, reverse-prefix, merge/alignment, state,
  automatic-seed, strict-load, validation/effect-order, provenance, graph-size, mode, replacement,
  and exclusion tests for all three concrete families.
- Finalize affected Javadocs, layers package documentation, Training API, glossary, task evidence,
  and master-plan status in the required separate clean documentation-focused context.

## Out of scope

- An arbitrary direction count, direction enum/list, direction registry, multidimensional spatial
  recurrence, duplicate forward/reverse directions, stacked recurrent layers, residuals,
  recurrent dropout, attention, or projection layers. The current one-dimensional time axis has
  exactly two distinct complete traversal orders.
- A generic `RecurrentCell`, `RecurrentSequence`, `BidirectionalSequence<S>`, erased state tuple,
  type switch, reflection, callback-driven per-step cell invocation, public packing plan, common
  recurrent base/interface, or duplicated generic result abstraction.
- `SUM`, `AVERAGE`, configurable merge, no-merge/raw-direction result mode, a merge enum, caller
  callback, projection-after-concat, or an implicit default. This task selects only ordered
  forward-then-backward final-axis `CONCAT`.
- Reversing the complete time axis, using `Tensor.flip(0)` as the padded-row policy, reading a
  padded suffix value, inferring padding from numeric zero, sorting rows by length, or exposing a
  row permutation.
- Dense padded output, caller-selected padding output, unpacking, batch-first input, a Tensor
  mask, runtime Tensor lengths, Java Boolean masks, dense post-cell masking, or runtime-dependent
  loop count/active Shape.
- A recurrent Model scan/control-flow operation, body/subgraph representation, carried-value
  tuple, runtime early exit, or reinterpretation of `CUM_SUM`/`CUM_PROD` as recurrent scan.
- Changes to current one-directional cells, sequences, results, formulas, gate packing, default
  states, construction-time length semantics, or state-path schema.
- A new Model operation/helper/public method, direct `Operation` construction, new compiler rule,
  numerical evaluator, backend capability, execution route, runtime work-skipping guarantee, or
  end-to-end value claim.
- Hidden mutable recurrent state, a Buffer for current state, cached inputs/lengths/indices/results,
  state reset/detach, a runtime tape, NN backward method, Tensor gradient field, optimizer,
  Training session, checkpoint bytes, or serialization.
- Adding bidirectional recipes to `ModuleFactory`. The new direct automatic constructors already
  remove two-cell assembly while keeping two seeds visible. Extend the closed standard factory
  only after these concrete families are stable and a caller proves that another recipe removes
  real construction burden; do not widen this task's public facade speculatively.
- Changes to Training Java source, Tensor or Compile API references, Model capabilities,
  Compiler, Runtime, Prepare, Engine, backend source, Gradle/build structure, architecture
  contracts/ADRs/tests, conformance/integration tests, the global roadmap, active CPU work, other
  master plans, or later NN task files.
- Detailed task specifications for NN 0021–0024.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)
- [Completed vanilla-RNN cell](0012-vanilla-rnn-cell.md)
- [Completed GRU cell](0013-gru-cell.md)
- [Completed LSTM cell](0014-lstm-cell.md)
- [Completed static packed RNN sequence](0015-static-packed-rnn-sequence.md)
- [Completed static packed GRU sequence](0016-static-packed-gru-sequence.md)
- [Completed static packed LSTM sequence](0017-static-packed-lstm-sequence.md)
- [Completed typed functional Model topology](0018-typed-functional-model-topology.md)
- [Completed automatic Linear initialization](0019-automatic-first-forward-linear-initialization.md)
- [Completed automatic recurrent initialization and defaults](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Completed initialized Embedding](0020a-initialized-embedding.md)
- [Completed stateless standard ModuleFactory](0020b-stateless-standard-module-factory.md)
- [Completed Model Gather-ND expressions](../../../modules/model/tasks/0018f-gather-nd-tensor-expressions.md)
- [Completed Model composition expressions](../../../modules/model/tasks/0017l-tensor-composition-expressions.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns cell-specific directional composition, independent module state trees, construction-time
  length interpretation, explicit default-state conveniences, and typed results. Model remains the
  sole owner of Tensor identity, descriptors, eager leaves, Shape algebra, promotion, SELECT,
  GATHER, GATHER_ND, CONCAT, cell operations, and provenance.
- Recurrent hidden and cell values remain caller-/expression-threaded Tensors, never Parameters,
  Buffers, retained fields, optimizer state, or runtime tape.
- `Module` retains no universal forward method. Every new container is a direct final `Module`
  subclass with a truthful family-specific signature and remains outside `UnaryTensorModule` and
  `Sequential`.
- Cell identity and state must not be shared between directions. One cell shares its own exact
  parameter leaves only across time within that direction. The other direction owns another cell,
  wrappers, leaves, seed configuration, and state paths.
- The existing named-child primitive remains the sole parent-link owner. Its visibility change
  enables an actual subclass consumer but does not change exclusive ownership, cycle checks,
  validate-before-install atomicity, traversal order, or state-path grammar.
- Java lengths may specialize static expression structure. No current API may claim runtime
  length reuse or graph-topology independence from those concrete Java values.
- `GATHER_ND` coordinates express reverse valid-prefix selection as ordinary Model metadata.
  They do not evaluate index bounds, execute values, or prove backend support in this NN task.
- Merged outputs are ordinary `CONCAT` expressions. Compiler-owned existing derivative rules may
  later traverse their exact ancestry; NN neither invokes compiler internals nor claims numerical
  backpropagation or execution.
- If implementation needs a new Model semantic, dependency, runtime scan, public generic recurrent
  abstraction, hidden state, different merge, architecture change, ModuleFactory change, or path
  outside the exact list, stop and report the blocker rather than widening the task.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.nn.layers` — owns the six new public cell-family-specific
  bidirectional sequence/result types and their complete static directional composition.
- `io.github.pho001.synaptik.nn.module` — widens only the existing atomic named-child installation
  primitive from package-private to protected for the concrete two-child layer consumer.

Existing Model packages are consumed unchanged for Tensor, data type, Shape, layout, eager index,
SELECT, GATHER, GATHER_ND, CONCAT, and provenance contracts. No package or dependency is added.

Type placement:

- `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
  `BidirectionalLstmSequence` — concrete public directional composition belongs beside the current
  recurrent cells and static sequence containers in `nn.layers`.
- Their matching `ForwardResult` records — NN owns typed compact output and continuation-state
  values; they are not Model multi-output producers or Runtime results.
- `Module.registerNamedChildren` — Module remains the only type able to validate and install
  parent links atomically.
- Three same-package tests own the exact family APIs and behavior; `ModuleTreeTest` owns the
  protected atomic named-child semantics regression, and `ModelTest` updates its existing exact
  Module-surface assertion for the planned protected visibility.

## Public API

`BidirectionalRnnSequence` declares exactly:

```java
public final class BidirectionalRnnSequence extends Module {
    public BidirectionalRnnSequence(RnnCell forwardCell, RnnCell backwardCell)

    public BidirectionalRnnSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long forwardSeed,
            long backwardSeed)

    public RnnCell forwardCell()
    public RnnCell backwardCell()

    public BidirectionalRnnSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor backwardInitialHidden,
            long[] lengths)

    public BidirectionalRnnSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor backwardInitialHidden)

    public BidirectionalRnnSequenceForwardResult forward(Tensor input, long[] lengths)
    public BidirectionalRnnSequenceForwardResult forward(Tensor input)
}

public record BidirectionalRnnSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor forwardFinalHidden,
        Tensor backwardFinalHidden) {}
```

`BidirectionalGruSequence` declares exactly:

```java
public final class BidirectionalGruSequence extends Module {
    public BidirectionalGruSequence(GruCell forwardCell, GruCell backwardCell)

    public BidirectionalGruSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long forwardSeed,
            long backwardSeed)

    public GruCell forwardCell()
    public GruCell backwardCell()

    public BidirectionalGruSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor backwardInitialHidden,
            long[] lengths)

    public BidirectionalGruSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor backwardInitialHidden)

    public BidirectionalGruSequenceForwardResult forward(Tensor input, long[] lengths)
    public BidirectionalGruSequenceForwardResult forward(Tensor input)
}

public record BidirectionalGruSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor forwardFinalHidden,
        Tensor backwardFinalHidden) {}
```

`BidirectionalLstmSequence` declares exactly:

```java
public final class BidirectionalLstmSequence extends Module {
    public BidirectionalLstmSequence(LstmCell forwardCell, LstmCell backwardCell)

    public BidirectionalLstmSequence(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long forwardSeed,
            long backwardSeed)

    public LstmCell forwardCell()
    public LstmCell backwardCell()

    public BidirectionalLstmSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor forwardInitialCell,
            Tensor backwardInitialHidden,
            Tensor backwardInitialCell,
            long[] lengths)

    public BidirectionalLstmSequenceForwardResult forward(
            Tensor input,
            Tensor forwardInitialHidden,
            Tensor forwardInitialCell,
            Tensor backwardInitialHidden,
            Tensor backwardInitialCell)

    public BidirectionalLstmSequenceForwardResult forward(Tensor input, long[] lengths)
    public BidirectionalLstmSequenceForwardResult forward(Tensor input)
}

public record BidirectionalLstmSequenceForwardResult(
        List<Tensor> packedOutputs,
        Tensor forwardFinalHidden,
        Tensor forwardFinalCell,
        Tensor backwardFinalHidden,
        Tensor backwardFinalCell) {}
```

The classes declare no other public/protected constructor, method, field, nested type, interface,
merge selector, direction selector, state getter, or overload. Inherited final `Module` APIs remain
available normally.

Each result canonical constructor rejects null `packedOutputs`, traverses once to reject the first
null element while making an immutable structural snapshot, then rejects state components in
record declaration order. It retains exact Tensor references and performs no descriptor
relationship validation, copy, expression creation, identity allocation, or execution.

## Directional state schema and ownership

Every container declares no direct Parameter or Buffer and owns exactly two children:

| Order | Child name | Exact value |
|---|---|---|
| 1 | `forward` | exact supplied or newly created matching forward cell |
| 2 | `backward` | exact supplied or newly created matching backward cell |

Recursive parameter and state-dictionary order is the forward subtree followed by the backward
subtree. Exact paths are:

```text
forward.inputWeight
forward.hiddenWeight
forward.bias        when configured
backward.inputWeight
backward.hiddenWeight
backward.bias       when configured
```

The same names apply to all three families; GRU/LSTM retain their existing packed Shape and gate
meaning within each path. Inputs, lengths, initial/final states, eager coordinate/index leaves,
directional outputs, alignment values, and merged outputs never enter module discovery or state
dictionaries.

`train()` and `eval()` propagate to `forward` then `backward` in normal Module preorder. Forward
composition is mode-insensitive. Strict state export/load uses the same child order. A complete
dictionary may bind both automatic cells without running either initializer. Each reservation
group remains cell-local; generic strict load validates the complete tree before installing any
binding under the existing Module contract.

## Construction and seed contract

Supplied-cell construction performs:

1. reject null `forwardCell`, then null `backwardCell`;
2. reject repeated exact cell identity;
3. require equal configured positive hidden sizes;
4. require the exact configured parameter data types to match;
5. build the encounter-ordered `forward`, `backward` named-child snapshot; and
6. invoke the now-protected atomic named-child primitive once.

All checks complete before either parent link changes. A name, identity, cycle, or existing-owner
failure leaves both candidates unchanged. The constructor retains exact cells and creates no
Tensor, Parameter, generator, index, state, or result.

Automatic construction validates the common hidden size, bias, data type, initialization policy,
gate multiplier, hidden-weight Shape/count, and Java-array limits under the existing matching cell
constructor. It then creates the forward automatic cell with `forwardSeed`, creates the backward
automatic cell with `backwardSeed`, and atomically registers them only after both constructors
return. Cell construction creates reservations but no Tensor, Parameter wrapper, or generator.

On first represented use, the complete forward request is prevalidated from immutable schemas.
The first forward cell step initializes only the forward cell from its seed; the first reverse
step initializes only the backward cell from its seed. Each random cell starts one fresh exact
`L64X128MixRandom` stream per attempt, while zero/one policies create no generator. There is no
cross-direction initialization transaction: if forward initialization and expressions succeed
but backward initialization later fails, the forward cell remains bound and the backward cell
remains independently retryable. Completed draws, Tensor IDs, allocations, and expression prefixes
are not rolled back.

## Forward input and result contract

The most-explicit RNN/GRU call requires:

| Input | Required contract |
|---|---|
| `input` | non-null floating fully static rank-three `[time, batch, inputSize]` |
| `forwardInitialHidden` | non-null floating fully static `[batch, hiddenSize]` |
| `backwardInitialHidden` | non-null floating fully static `[batch, hiddenSize]` |
| `lengths` | non-null Java `long[]` of exactly `batch` entries in `[0, time]` |

The LSTM call additionally requires exact forward/backward initial cell Tensors with the same
`[batch, hiddenSize]` contract. Directional initial state types may differ only where each cell's
existing projection/state promotion accepts them and the final merged hidden outputs have the same
exact data type required by `CONCAT`.

For copied length `L[b]`:

```text
steps = max(L), or zero for an empty batch
active(k) = [b in ascending original order where L[b] > k]
```

The result `packedOutputs` contains exactly `steps` merged Tensors. Entry `t` has Shape
`[activeCount(t), 2 * hiddenSize]`. Its first final-axis interval is the exact forward-cell output
at original time `t`. Its second interval is the aligned backward-cell hidden output after that
row has consumed original positions `L[b] - 1` down through `t`.

RNN/GRU final hidden semantics are:

```text
forwardFinalHidden[b]  = forward state after x[L[b] - 1], or forwardInitialHidden[b] if L[b] == 0
backwardFinalHidden[b] = backward state after x[0], or backwardInitialHidden[b] if L[b] == 0
```

LSTM applies the same direction semantics to both hidden and cell state. The visible packed output
contains only merged hidden state; intermediate cell state remains an internal carried value.

When `steps == 0`, `packedOutputs` is empty and every final state is the exact corresponding
initial-state reference. No index, expression, automatic parameter, or result-state wrapper Tensor
is created. The result record itself is an ordinary Java value.

Overload behavior is:

| Supplied arguments | Derived values |
|---|---|
| input, all explicit directional states, lengths | none |
| input, all explicit directional states | one all-valid Java length array |
| input, lengths | distinct typed-zero states for both directions |
| input | one all-valid array plus distinct typed-zero states for both directions |

Default RNN/GRU state creation order is forward hidden then backward hidden. Default LSTM order is
forward hidden, forward cell, backward hidden, backward cell. Each is a fresh eager unlabeled,
provenance-free, non-gradient leaf shaped `[batch, hiddenSize]` with the common cell parameter type
and is never retained. Omitted lengths mean `L[b] = time` for every row.

## Packing, reverse traversal, alignment, and merge

### Forward traversal

For `t` in `[0, steps)`, preserve the current static packing policy:

1. create `input.select(0, t)`;
2. create one dense INT64 active-original-row leaf for `active(t)`;
3. gather the selected input on axis zero;
4. at `t == 0`, gather the forward initial state with the same active leaf; later gather the prior
   forward compact state with one relative-survivor leaf;
5. for LSTM, share the same state leaf between hidden and cell gathers;
6. invoke the concrete forward cell once; and
7. retain the exact next-hidden output and, for LSTM, exact next-cell carried state.

### Backward traversal

For reverse depth `d` in `[0, steps)`:

1. enumerate `active(d)` in ascending original row order;
2. create one dense, contiguous, unlabeled, provenance-free, non-gradient INT64 coordinate leaf
   shaped `[activeCount(d), 2]` whose row for original batch `b` is
   `[L[b] - 1 - d, b]`;
3. create `input.gatherNd(coordinates)`, yielding `[activeCount(d), inputSize]`;
4. at `d == 0`, create one active-original-row leaf and gather backward initial state; later create
   one relative-survivor leaf and gather prior reverse-depth state;
5. for LSTM, share the exact state leaf between hidden and cell gathers;
6. invoke the concrete backward cell once; and
7. retain exact reverse-depth hidden output and, for LSTM, exact next-cell carried state.

Every coordinate satisfies `0 <= L[b] - 1 - d < L[b] <= time`, so no backward input occurrence
addresses padding. For lengths `[5, 3, 1]`, reverse depth coordinates are:

```text
d=0: [(4,0), (2,1), (0,2)]
d=1: [(3,0), (1,1)]
d=2: [(2,0), (0,1)]
d=3: [(1,0)]
d=4: [(0,0)]
```

### Backward alignment and merge

Let `reverseOutputs[d]` retain the backward hidden output for reverse depth `d`, and let
`offset[d]` be the sum of active counts before `d`. Flatten those outputs along axis zero in
increasing `d` order. For original time `t` and row `b in active(t)`, its flat backward index is:

```text
d = L[b] - 1 - t
flatIndex(t, b) = offset[d] + position of b in active(d)
```

One eager INT64 leaf contains those indices in ascending `active(t)` order. One axis-zero gather
creates the aligned backward Tensor for the whole compact time step. Then:

```text
packedOutputs[t] = Tensor.concat(
        -1,
        forwardOutputs[t],
        alignedBackwardOutputs[t])
```

The implementation must not create one Tensor `SELECT` per logical recurrent row merely to align
reverse outputs. Java may fill primitive index arrays with `O(sum(L))` scalar work, but the Tensor
graph adds only one alignment index leaf, one GATHER, and one merge CONCAT per represented time.

### Final-state restoration

For a non-empty request, construct one final-row INT64 index leaf in original batch order and
reuse it across both directions and, for LSTM, across hidden and cell state. For each forward or
cell-state stream, one source CONCAT consumes the exact initial state first and every compact step
state afterward in traversal order; that variadic CONCAT is the flattening, so no intermediate
flatten expression is created. The backward hidden source CONCAT instead consumes its exact
initial state followed by the already-created flattened reverse hidden Tensor. Gather each source
with the shared final-row leaf. A zero-length row indexes its original initial-state row. A
positive row indexes the compact state at its exit step. Forward exit step is original time
`L[b] - 1`; backward exit depth is `L[b] - 1`, which is the state after consuming original time
zero.

This composition replaces per-row SELECT/STACK restoration inside the new classes only. Existing
one-directional sequence provenance remains unchanged. It uses current exact CONCAT/GATHER
semantics and preserves original row order with a bounded number of graph occurrences.

## Validation and side-effect order

The most-explicit RNN/GRU call performs:

1. reject null `input`, `forwardInitialHidden`, `backwardInitialHidden`, then `lengths`;
2. read immutable configured hidden-size/type facts for both cells and require constructor
   invariants still hold;
3. require input floating type, rank three, fully static Shape, Java collection limits, and a
   positive input feature extent accepted by both cell schemas;
4. validate forward initial hidden then backward initial hidden for floating type, rank two,
   fully static `[batch, hiddenSize]`, and direction-specific promotion;
5. validate length count and entries from the caller array in encounter order while deriving
   `steps`, with caller coordination required until the later snapshot;
6. prevalidate every distinct forward and reverse compact Shape, both cell formulas, survivor
   gathers, reverse coordinate GATHER_ND result, flatten/alignment GATHER, ordered final-axis
   CONCAT merge, final-state source CONCAT/GATHER, checked `2 * hiddenSize`, count, primitive-array,
   and Tensor-ID-independent Shape/layout fact;
7. clone lengths immediately before traversal and use only the snapshot afterward; and
8. only then create eager leaves, initialize cells when reached, and construct expressions.

LSTM null/state validation order is `input`, `forwardInitialHidden`, `forwardInitialCell`,
`backwardInitialHidden`, `backwardInitialCell`, `lengths`. State descriptor checks follow that same
direction/state order. Complete hidden/cell formula and final-merge preflight precedes effects.

Default-state and all-valid overloads first complete every validation knowable without the derived
values. They then create an all-valid host array when required, create zero states in the specified
order, and delegate to one canonical traversal implementation. They must not duplicate either
direction loop.

Every ordinary caller-controlled null, type, rank, dynamic-Shape, feature, batch, length,
promotion, cell schema, GATHER/GATHER_ND, CONCAT, count, or Java-limit failure precedes the first
new Tensor ID, eager storage, random generator, draw, parameter publication, or expression. After
effects start, allocations, draws, published parameter groups, eager leaves, Tensor IDs, and
expression prefixes are not rolled back. No partial result record is returned and the container
retains no per-call value.

Validation is not a transaction with concurrent parameter replacement or strict load. Actual
cell calls observe their then-current compatible bindings. Callers coordinate forward
construction, replacement, loading, mode changes, and discovery when one cross-direction snapshot
matters.

## Tensor identity and provenance order

After overload-specific derived values exist, successful construction order is exact:

1. all forward steps in increasing original time, each using SELECT, active/survivor leaves,
   GATHER occurrences, and the exact family cell chain;
2. all backward steps in increasing reverse depth, each using coordinate leaf, GATHER_ND,
   active/survivor leaf, state GATHER occurrence(s), and the exact family cell chain;
3. zero or one backward-hidden flatten CONCAT: no Tensor when there is one reverse output, one
   axis-zero CONCAT when there are two or more;
4. for each original time in increasing order, one alignment index leaf, one backward alignment
   GATHER, then one forward-first final-axis CONCAT;
5. one final-row index leaf;
6. forward hidden source CONCAT then GATHER;
7. for LSTM, forward cell source CONCAT then GATHER;
8. backward hidden source CONCAT then GATHER; and
9. for LSTM, backward cell source CONCAT then GATHER.

Each final-state source CONCAT has at least two inputs: the exact initial state plus either the
ordered compact step-state Tensors directly or the already-flattened backward hidden Tensor. When
the backward hidden stream has one step, that exact step Tensor is used without a one-input
flatten CONCAT. The same exact final-row index leaf is reused by every final-state gather. LSTM
shares each state carry index leaf hidden first then cell, preserving its established component
order.

Every represented step calls the same exact cell object for its direction and constructs fresh
producer/Tensor identities. All forward step projections retain exact forward parameter leaves;
all backward step projections retain exact backward parameter leaves. No leaf identity crosses
directions. Temporal state ancestry remains explicit inside each direction, and CONCAT exposes
both ancestries to current compiler exact-identity traversal without NN owning gradients.

Tests must lock this relative producer and Tensor-ID order, including one-step exact counts after
the implementation fixes its private preflight, but must not assert numerical values, runtime
kernel count, backend support, or compiler internals.

## Replacement, load, mode, and continuation behavior

- `forwardCell()` and `backwardCell()` return the exact stable children.
- Parameter replacement in one direction changes later expressions in only that direction. It
  neither replaces nor aliases the other direction's wrapper or Tensor.
- Strict load uses the complete path set and ordinary validate-before-install semantics. A
  dictionary exported from an equivalently configured initialized bidirectional container can
  bind both automatic children without random generation. Forward later validates that both loaded
  input widths accept the same input.
- Mode propagation visits the forward subtree then backward subtree. Both cells and the container
  construct the same expressions in training and evaluation modes.
- Returned directional final states are exact continuation inputs for a later caller-controlled
  call. The module never installs or retains them.
- Result list snapshots remain immutable while every Tensor reference and producer ancestry
  remains exact.

## Performance and graph-size boundary

Let `T = max(lengths)` and `S = sum(lengths)`.

- Exactly `T` forward-cell calls and `T` backward-cell calls are constructed when `T > 0`; each
  call is batched over its compact active rows. They represent `2S` logical recurrent row
  applications, not `2 * time * batch` dense rows.
- Reverse input uses one coordinate leaf and one GATHER_ND per reverse depth, not one SELECT per
  active row.
- Reverse output alignment uses at most one flatten CONCAT plus exactly `T` alignment leaves,
  `T` GATHER occurrences, and `T` merge CONCAT occurrences, not `S` Tensor row selections.
- Final restoration uses one shared final index leaf and one source CONCAT/GATHER pair per returned
  directional state, rather than one SELECT per row plus STACK.
- Java construction still performs `O(S)` primitive coordinate/index filling and the Tensor graph
  remains statically specialized to the copied lengths. There is no claim about compiled node
  optimization, backend fusion, kernel count, numerical execution cost, or runtime work skipping.
- The implementation must use primitive arrays and bounded indexed loops. It must not allocate a
  callback, boxed state tuple, stream pipeline, reflection object, map lookup, or polymorphic
  direction strategy per recurrent step or logical row.

## Affected files

Implementation and documentation are limited to these seventeen paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalRnnSequence.java`
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalRnnSequenceForwardResult.java`
4. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalGruSequence.java`
5. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalGruSequenceForwardResult.java`
6. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalLstmSequence.java`
7. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/BidirectionalLstmSequenceForwardResult.java`
8. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
9. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`
10. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java`
11. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BidirectionalRnnSequenceTest.java`
12. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BidirectionalGruSequenceTest.java`
13. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/BidirectionalLstmSequenceTest.java`
14. `docs/api/training-api.md`
15. `docs/glossary.md`
16. `docs/planning/extensions/nn/tasks/0020c-bidirectional-static-recurrent-composition.md`
17. `docs/planning/extensions/nn/master-plan.md`

`ModelTest` may change only its stale named-child protected-visibility assertion. Do not edit
`ModuleFactory`, its test/package Javadoc, current cells/sequences/results/tests,
initialization source, Training Java, Tensor/Compile APIs, Model source/tests/capabilities,
architecture files/tests/ADRs, global roadmap, Gradle/build files, Compiler, Runtime, Prepare,
Engine, backends, conformance/integration tests, other planning areas, or later task files.

## Maximum scope

This task may create or modify only the seventeen exact paths above. The six public types, three
focused family test owners, the foundational visibility/semantics test, the existing exact-surface
assertion, and targeted documentation form one cohesive capability. The seventeenth path is the
coordinator-authorized correction for the planning omission discovered by the first failed full
NN run; it changes only the stale visibility assertion. If implementation requires a shared
generic helper, another test owner, ModuleFactory recipe, current sequence refactor, new package,
or eighteenth path, stop and report the concrete need rather than expanding scope.

## Acceptance criteria

- The exact six type-safe public APIs exist with no generic recurrent base, direction/merge enum,
  callback, builder, hidden state, or extra surface.
- Supplied constructors validate both cells and atomically own exact identity-distinct children as
  `forward` then `backward`; automatic constructors create independent cell identities and seed
  configurations.
- Recursive state paths/order, discovery, state dictionaries, strict load, replacement isolation,
  and forward-then-backward mode propagation match the exact contract.
- Static lengths alone define both traversals. Reverse coordinates address only valid-prefix
  positions, including unsorted, zero, and mixed lengths, and never reverse or read padding.
- Each direction invokes its cell once per non-empty compact step, shares parameters only across
  time within that direction, and preserves fresh temporal producer ancestry.
- Backward outputs are restored to original time and batch order by flattened CONCAT plus one
  GATHER per represented original time. No per-logical-row Tensor alignment expression or dense
  padded output is introduced.
- Every visible packed output is an exact ordered forward-then-backward final-axis CONCAT with
  Shape `[activeCount(t), 2 * hiddenSize]`.
- RNN/GRU return separate forward/backward final hidden states; LSTM returns separate hidden/cell
  states grouped forward then backward. Zero-length and all-zero exact-reference semantics pass.
- Default-state and all-valid overloads create values in the specified order, retain nothing, and
  leave both automatic cells unbound for an all-zero request.
- Complete validation precedes caller-controlled effects; relative Tensor-ID/provenance order and
  documented cross-cell failure non-rollback are tested.
- The implementation contains no `flip(0)` whole-time reversal, dense mask, SUM merge, direct
  `Operation`, runtime scan, type switch, reflection, per-step callback, cross-direction shared
  cell/Parameter/Tensor identity, or non-Model dependency.
- Focused and final NN tests pass after executable stabilization. Repository-wide validation is
  deferred because the change remains inside the existing NN/Model dependency boundary.
- A separate clean documentation-focused pass finalizes every affected Javadoc and explanatory
  document, reviews unchanged Tensor/Compile APIs, Training Java, glossary impact, Model
  capabilities, architecture/tests, ModuleFactory, Gradle, conformance/integration, global
  roadmap, CPU, and other modules, and records reasoned no-change conclusions.
- Final Javadoc, generated-page, public/protected/private surface, external-use, state path,
  forbidden-mechanism/import, Markdown link/anchor/fence, exact seventeen-path, frontier/status,
  newline, no-index whitespace, and whole-worktree `git diff --check` gates pass.

## Tests / validation

Implementation context runs focused suites while developing:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.module.ModuleTreeTest \
  --tests io.github.pho001.synaptik.nn.module.ModelTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalRnnSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalGruSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalLstmSequenceTest
```

After executable Java and tests stabilize, run exactly one authoritative affected-module suite:

```bash
./gradlew :extensions:nn:test
```

The separate documentation-focused context reuses that unchanged Java evidence and runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates generated Javadoc pages, exact public/protected surface, direct external Java
use of all three families, recursive paths and strict-load wording, forbidden imports/mechanisms,
local Markdown links/anchors/headings/fences, exact scope, status/frontier counts, final newlines,
and trailing whitespace in untracked files without staging them.

Repository-wide, architecture, Model, Compiler, Training, backend-conformance, and integration
suites are deferred to CI or their owning checkpoints because this task adds no dependency,
architecture, generic Model semantic, execution behavior, or cross-module Java change. The
Module protected-surface change preserves the existing tested ownership rule inside the same NN
module and is covered by the focused/final NN suites.

## Documentation pass

After implementation and the authoritative NN suite, hand the exact diff and evidence to a
separate documentation-focused clean context. It must follow the general, API/Javadoc, planning,
and example profiles named by `docs/developer-guide/documentation-rules.md` and must:

- independently inspect the six new types, Module visibility change, all five affected tests, and
  current recurrent/Module contracts;
- finalize every new type/component/constructor/method Javadoc plus the affected `Module` and
  layers package Javadocs, including all parameters, results, nullability, state ownership,
  reverse-prefix, merge, failure, concurrency, and no-execution boundaries;
- update Training API and glossary for bidirectional packing, CONCAT order, directional final
  state, independent parameter trees/seeds, and the distinction from runtime scan and arbitrary
  multidirectional/stacked recurrence;
- retain current one-directional and ModuleFactory explanations accurately and record why
  Tensor/Compile APIs, Training Java, Model capabilities, architecture/ADR/tests, build,
  conformance/integration, global roadmap, CPU, and other modules need no edit; and
- record exact commands/results and reused Java evidence in this task before marking Complete.

The documentation pass must not rerun successful Java tests unless it changes executable Java or
records a concrete risk that requires a rerun.

## Dependencies

- NN 0012–0014 provide the exact concrete cell schemas, formulas, and typed LSTM state result.
- NN 0015–0017 provide the proven fully static time-major active-set, survivor, default shortcut,
  and original-order final-state contracts.
- NN 0018–0020 provide atomic named topology, private automatic parameter reservations, two
  independent standard cell constructors, zero-state/all-valid conveniences, strict load, and
  exact shared-parameter/fresh-producer semantics.
- NN 0020A–0020B close initialized standard layer construction and leave directionality as the
  next ordered frontier.
- Current Model SELECT, GATHER, GATHER_ND, CONCAT, eager INT64 leaves, Shape/promotion,
  provenance, and compiler gradient coverage are complete.
- The user-authorized NN interleave remains file- and dependency-isolated from active CPU/backend
  and global-roadmap work.

## Follow-up tasks

- NN 0021 remains Draft for the cross-module recurrent scan/control-flow and runtime input-binding
  prerequisite. It must preserve this task's directional ownership, reverse-prefix, merge, and
  final-state semantics without treating static Java lengths as the target runtime API.
- NN 0022 remains Draft for Data-owned runtime valid lengths after NN 0021.
- NN 0023–0024 remain Draft for optional masks and the later integration checkpoint.
- Add arbitrary multidimensional directions or stacked recurrent layers only when a concrete
  input layout and consumer define additional distinct traversal axes, merge, state, and ownership
  semantics. Do not generalize this two-direction contract merely from naming symmetry.
- Reassess ModuleFactory recipes only after the new concrete constructors are implemented and a
  caller proves that a recipe removes meaningful remaining assembly without hiding two seeds.

## Architecture impact

Expected impact: None.

The architecture already assigns stateful neural-network composition, parameters, buffers,
mode, and functional conveniences to `extensions/nn`, with Model as its sole dependency. This
task adds a concrete NN composition over existing Model semantics and widens only an existing
Module-owned child-installation primitive for a real subclass consumer. It changes no module
boundary, dependency direction, operation ownership, or runtime lifecycle. Stop if implementation
reveals otherwise.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are implementing Synaptik NN task 0020C. Do not use GSD. Do not commit, push, stage, revert,
or modify concurrent unrelated CPU/backend/global-roadmap work.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md, the NN master plan, and
docs/planning/extensions/nn/tasks/0020c-bidirectional-static-recurrent-composition.md in full.
Read the completed recurrent/automatic/factory tasks named by the specification, final
Module/Topology/ModuleFactory and RNN/GRU/LSTM Cell/Sequence/result source/tests/Javadocs, the
directly used Model Tensor operations, Training API, glossary, and documentation rules/profiles.

Implement exactly the task specification in its seventeen authorized paths. Preserve concrete
RNN/GRU/LSTM typing, independent forward/backward cells and seeds, forward/backward state paths,
valid-prefix-only reverse coordinates, original-time alignment, fixed forward-first CONCAT,
directional final states, validation/effect/Tensor-ID order, and bounded graph construction. Do
not add a generic recurrent base, merge/direction enum, ModuleFactory recipe, dense mask, runtime
scan, new Model operation, architecture/build/dependency change, or later task file. Stop and
report an architecture, API, ownership, operation-expressibility, or scope conflict rather than
inventing a broader design.

Run the focused and one final NN suite after executable Java stabilizes. Then hand the exact diff
and evidence to a separate clean documentation-focused context, which must finalize Javadocs,
Training API, glossary, task/master evidence, final Javadoc, examples, links, no-change reasoning,
scope, and status without repeating unchanged Java tests. Do not mark Complete before both passes
and every acceptance gate succeed. Return a completion summary naming completed changes, exact
paths, tests/validation, unresolved issues, required follow-up, canonical context, and exactly
`Status: Complete` or `Status: Incomplete` with a specific follow-up under the repository rules.
```

## Documentation-agent handoff

The implementation context must provide the documentation context with this task, the exact
seventeen-path diff, affected APIs/behavior, focused and final NN commands/results, cell
initialization and failure effects, producer/ID evidence, architecture constraints, and every
expected documentation/no-change topic listed above. The documentation context independently
reviews source and tests rather than accepting the handoff summary as proof.

## Local decisions

- Select only two directions. One time axis defines forward and reverse-valid-prefix orders; an
  arbitrary direction collection would duplicate those orders or invent unplanned spatial
  semantics.
- Select fixed ordered final-axis `CONCAT`. It retains both directional features and makes the
  merge visible in provenance. `SUM` would erase direction identity and configurable merge would
  add policy surface without a second current consumer.
- Own cells directly under `forward` and `backward`, not through two nested one-directional
  Sequence children. This produces concise stable paths and avoids building a padded reversed
  Tensor merely to reuse an API whose private loop cannot accept compact reverse coordinates.
- Widen the existing atomic named-child primitive to protected rather than sequentially calling
  `child(...)`. Sequential calls could strand the first supplied cell if validation of the second
  failed; a second primitive or wrapper module would duplicate ownership machinery or add a path
  segment.
- Use GATHER_ND for one compact reverse input per depth. Whole-time `flip` is wrong for mixed
  lengths, and per-row SELECT/STACK would add avoidable `O(S)` Tensor expressions.
- Flatten reverse outputs and realign them with one GATHER per original time. This retains compact
  output semantics while keeping graph occurrence growth proportional to represented steps beyond
  the unavoidable recurrent cell graph.
- Use CONCAT/GATHER final restoration in the new types and share one final index leaf across
  direction/state streams. Existing one-directional SELECT/STACK provenance remains unchanged.
- Keep cell-family-specific loops and results. A callback/generic state machine would add per-step
  indirection or erase LSTM cell state; structural similarity alone does not justify it.
- Leave ModuleFactory unchanged. The direct automatic constructors already hide two-cell assembly
  and deliberately expose both seeds; another factory recipe is not needed to complete this
  capability.

## Known limitations

- Lengths are copied Java construction metadata and all participating Shapes are fully static.
  Different length values specialize the expression graph.
- The result is a compact list by original time, not a dense padded Tensor. Consumers must retain
  the same lengths to interpret active extents.
- Only hidden outputs are merged. LSTM cell states are carried internally and only directional
  final cell states are public.
- Only equal hidden sizes and exact common cell parameter types are accepted. Supplied direction
  biases may differ, but both cell input schemas must accept the same call input.
- Construction is not atomic with concurrent parameter replacement/load, and two automatic cells
  do not form one initialization transaction.
- Static graph construction omits padded rows from cell expressions but makes no runtime kernel,
  backend support, numerical value, or work-skipping guarantee.
- Runtime lengths, reusable graph topology across different length values, arbitrary masks,
  stacked recurrence, multidimensional directionality, and ModuleFactory recipes remain future
  work with separate consumer evidence.

## Validation evidence

Planning-only work completed in clean context `/root/nn_0020c_planning`. No production Java,
tests, Javadoc, API guide, glossary, architecture, build, global-roadmap, CPU, backend, or later
task implementation was performed.

- Read the repository instructions, authoritative architecture contract, current architecture
  plan, planning guide and roadmap, NN master plan, completed NN 0012–0020B task records, actual
  Module/Topology/ModuleFactory and RNN/GRU/LSTM Cell/Sequence/result source/Javadocs/tests,
  directly used Model Tensor operations, Training API, glossary, and documentation rules/profiles.
- The operation review found no expressibility blocker. Existing `GATHER_ND` can select one
  compact valid-prefix reverse input per depth, exact-type axis `GATHER` can realign flattened
  reverse outputs, and existing axis `CONCAT` can expose ordered directional features and restore
  final states. Whole-axis flip and dense masking were rejected for mixed valid lengths.
- The API/ownership review selected three concrete family containers and three matching records,
  independent direct `forward`/`backward` children, fixed CONCAT, and one protected visibility
  widening of the existing atomic named-child primitive. It found no need for a generic recurrent
  base, arbitrary directions, ModuleFactory recipe, Model operation, runtime scan, hidden state,
  dependency, or architecture change.
- Targeted Ruby Markdown validation passed for this task and the NN master plan: every local link
  target exists, there are no fragment references requiring a foreign anchor, effective headings
  are unique, fences are balanced, terminal newlines are present, and trailing whitespace is
  absent.
- Frontier validation passed: 0020C is the sole task and master-plan row with scalar status
  `Ready`; 0021–0024 remain Draft and have no task files.
- `git status --short docs/planning/extensions/nn` reported exactly the modified NN master plan
  and this untracked task file. Pre-existing concurrent CPU/backend and global-roadmap changes
  were inspected for attribution and left untouched, unstaged, unreverted, and unformatted.
- `git diff --check` and the tracked NN-master `git diff --check` passed. The task-file
  `git diff --no-index --check /dev/null` check returned the expected difference status with no
  whitespace diagnostic.
- No Java, Javadoc, Gradle, architecture, conformance, integration, Model, Compiler, Training,
  Runtime, Prepare, Engine, backend, CPU, or repository-wide test was run because this change is
  planning-only and changes no executable or authoritative architecture behavior.

Implementation work completed in clean context `/root/nn_0020c_implementation`; executable Java
and tests are frozen for the mandatory separate documentation-focused handoff.

- `./gradlew :extensions:nn:compileJava` passed after the six production types and Module
  visibility change were implemented.
- The original four-owner focused selection passed 23 tests: ModuleTree 9, bidirectional RNN 6,
  bidirectional GRU 4, and bidirectional LSTM 4; zero skips, failures, or errors.
- The first authoritative `./gradlew :extensions:nn:test` ran 280 tests and failed only the
  pre-existing `ModelTest` assertion that the exact named-child primitive was not protected. The
  coordinator authorized `ModelTest` as the exact seventeenth path because the task already
  requires protected visibility; only that stale assertion changed.
- The corrected five-owner focused selection, including all 13 Model tests, passed 36 tests with
  zero skips, failures, or errors. The replacement authoritative `./gradlew :extensions:nn:test`
  then passed 36 suites and 280 tests with zero skips, failures, or errors.
- Preliminary `./gradlew :extensions:nn:javadoc` first identified draft tag omissions; after
  Javadoc completion it passed warning-free. The separate documentation context must rerun final
  Javadoc only after its documentation edits.
- External `javac` use of all three automatic families and exact result accessors passed. `javap
  -public` confirmed exactly the planned two constructors, two typed cell accessors, and four
  forward overloads per family plus the exact record components. `javap -protected` confirmed
  `registerNamedChildren(Map)` is protected final, with no second named-child primitive.
- Generated Javadoc pages for all six new public types are present. Production imports remain
  limited to Model, NN module/initialization, and JDK collections; forbidden production scans,
  reflection, callbacks, direction/merge registries, hidden recurrent state, direct Operation
  construction, dense masks, whole-axis flips, and non-Model dependencies are absent.
- Whole-worktree `git diff --check` passed while concurrent CPU/backend/global-roadmap work
  remained untouched. Final exact-scope, untracked no-index whitespace, newline, Markdown, and
  explanatory-document checks remain for the documentation pass after its two documentation
  paths are finalized.

Documentation work completed in independent clean context `/root/nn_0020c_docs`; no executable
Java or test changed after the frozen passing implementation evidence.

- Independent review found no executable, API, ownership, provenance, performance, architecture,
  dependency, or scope blocker. It confirmed the three concrete direct-`Module` families, exact
  public/private surfaces, independent directional children/state/seeds, atomic validate-before-
  attach child ownership, valid-prefix-only reversal and original-order realignment, fixed
  forward-first final-axis CONCAT, directional final states, zero/default overloads, and the
  specified validation/effect/Tensor-identity order.
- Finalized meaningful Javadocs for the six new public types, the protected-final named-child
  primitive, and the layers package. The contracts now cover nullability, ownership, state paths,
  effects and failure boundaries, snapshot/threading limits, result identity, construction-time
  length specialization, and the absence of runtime scan, graph reuse, backend work-skipping, or
  numerical-execution guarantees.
- Updated the Training API and glossary with the concrete bidirectional families, stable state
  paths, valid-prefix reversal and alignment, forward-first CONCAT, directional continuation
  states, the `[3, 1, 2]` example, bounded static graph-size explanation, and current-versus-future
  runtime-scan boundary. `ModuleFactory` remains explicitly one-directional.
- Final `./gradlew :extensions:nn:javadoc` passed warning-free. Generated pages for every new
  public type, the layers package, and `Module` were inspected for the finalized contracts.
- External compilation and reflection passed for all three families and exact result accessors.
  `javap -public` confirmed two constructors, two typed cell accessors, and four `forward`
  overloads per family plus exact record components; `javap -protected` confirmed the sole
  protected-final named-child primitive.
- Import and dependency checks found only Model, current NN, JDK, and the existing one-way
  `extensions:nn` to `modules:model` dependency. Forbidden scans, Tensor lengths, dense masks,
  configurable merge/directions, generic recurrent base, hidden recurrent state, direct
  Operation construction, reflection-driven production behavior, and `ModuleFactory` expansion
  remain absent.
- Markdown paths, affected anchors/headings/fences, terminal newlines, trailing whitespace,
  untracked-file no-index checks, exact seventeen-path scope, 0020C Complete frontier, later
  0021–0024 Draft/no-spec frontier, and final `git diff --check` passed. Concurrent CPU/backend and
  global-roadmap changes remained untouched.

## Implementation notes

Clean implementation context `/root/nn_0020c_implementation` added the three concrete sequence
families and results, protected atomic child installation, the authorized stale surface assertion
correction, and focused tests. Each direct concrete loop performs forward compaction, valid-prefix
reverse Gather-ND traversal, flattened reverse realignment, ordered final-axis CONCAT, and shared-
index directional state restoration without a generic recurrent abstraction. The executable diff
is frozen after the passing replacement authoritative NN suite. Architecture/ADR/tests,
Model/Tensor/Compile/Training Java APIs, existing cells/sequences, ModuleFactory, compiler/runtime/
prepare/Engine/backends, Gradle/dependencies, conformance/integration, global roadmap, CPU, and
other modules require no change because the implementation composes current Model expressions
inside the existing NN-only boundary and changes no existing sequence behavior or dependency.
The independent documentation pass confirmed that architecture/ADR/architecture tests need no
change because no module or dependency boundary changed. Model/Tensor/Compile/Training Java APIs,
existing cell/sequence semantics, `ModuleFactory`, compiler/runtime/prepare/Engine/backends,
Gradle/dependencies, conformance/integration tests, CPU work, the global roadmap, and other modules
need no change because 0020C composes existing Model expressions inside the existing NN-only
boundary. The Training API prose and glossary are the only explanatory documents affected.

## Completion summary

- Completed changes: Added type-safe bidirectional static RNN, GRU, and LSTM composition with
  independent forward/backward cell ownership, valid-prefix reversal, original-time realignment,
  forward-first CONCAT, and separate directional final states; widened the existing named-child
  primitive and corrected its stale exact-surface assertion.
- Files changed or created: Exactly seventeen task paths: six new production types,
  `layers/package-info.java`, `Module.java`, `ModuleTreeTest.java`, the authorized stale protected-
  surface assertion in `ModelTest.java`, three new family tests, Training API, glossary, this task,
  and the NN master plan.
- Tests and validation: Focused 36/36 and replacement authoritative NN 280/280 passed; final
  Javadoc, generated pages, javap, external compilation/reflection, imports/dependencies,
  Markdown, exact-scope, newline, no-index, whitespace, and diff checks passed.
- Documentation-agent review: Independent clean context `/root/nn_0020c_docs` found no blocker and
  finalized all authorized documentation and evidence.
- Documentation impact: Finalized affected public/module/package Javadocs, Training API, glossary,
  task, and master evidence; all other documentation has a reasoned no-change conclusion.
- Javadoc review: Complete; final warning-free build and generated-page inspection passed.
- Glossary impact: Added the concrete bidirectional static packed sequence term and corrected the
  surrounding current-capability/future-scan boundary.
- Unresolved issues: None.
- Required follow-up: None for task 0020C; NN 0021–0024 remain separate Draft work.

Status: Complete

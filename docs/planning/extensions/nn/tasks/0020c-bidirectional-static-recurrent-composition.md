# Task 0020C: Type-Safe Bidirectional Static Recurrent Composition

## Status

Complete

## Goal

Add the smallest coherent directional recurrent capability that current [Model](../../../../glossary.md#term-model) expressions can
represent truthfully: separate final public bidirectional sequence containers for vanilla [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn),
[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru), and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm). Each container owns one forward cell and one backward cell with independent
[parameter](../../../../glossary.md#term-parameter) trees and initialization [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), traverses only each [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) row's valid prefix in the
selected direction, and returns original-time compact outputs merged by an explicit final-[axis](../../../../glossary.md#term-axis)
`CONCAT`.

[Mental model](../../../../glossary.md#term-mental-model):

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

- forward and backward cells must be different owned [module](../../../../glossary.md#term-module) [identities](../../../../glossary.md#term-identity-meanings) with separate state paths;
- backward traversal must reverse each row's valid prefix, never the complete padded time [axis](../../../../glossary.md#term-axis);
- per-time backward outputs must be restored to original time and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) order before merging; and
- the merge and directional final-state result [shape](../../../../glossary.md#shape) must be explicit and cell-family-specific.

Current [Model](../../../../glossary.md#term-model) `GATHER_ND`, axis [`GATHER`](../../../../glossary.md#gather), `SELECT`, and `CONCAT` expressions are sufficient. A
whole-axis `flip(0)` is not sufficient because it would move right [padding](../../../../glossary.md#padding) ahead of valid data for
short rows. Dense `WHERE` masking is not sufficient because it would still construct padded cell
work. `SUM` is not selected because it discards the two directional [feature](../../../../glossary.md#term-software-feature) subspaces and requires
an equal-Shape interpretation that the first [API](../../../../glossary.md#term-application-programming-interface-api) does not need. Final-axis `CONCAT` retains both
halves visibly and is already a current [differentiable](../../../../glossary.md#derivative-and-chain-rule) Model [operation](../../../../glossary.md#operation).

## [Scope](../../../../glossary.md#term-scope)

- Add final public `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
  `BidirectionalLstmSequence` classes in `io.github.pho001.synaptik.nn.layers`. Each extends
  [`Module`](../../../../glossary.md#term-module) directly and has only its family-specific typed cell and result contracts.
- Add final public `BidirectionalRnnSequenceForwardResult`,
  `BidirectionalGruSequenceForwardResult`, and `BidirectionalLstmSequenceForwardResult` records.
- Give every container one constructor accepting exact caller-supplied forward and backward cells
  and one constructor creating two automatic cells from an explicit common hidden size, bias
  choice, floating type, <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>`ParameterInitialization`, forward [seed](../../../../glossary.md#term-seed), and backward seed.
- Require the two supplied cells to be non-null, identity-distinct, and configured with the same
  hidden size and exact parameter data type. Bias presence may differ for supplied cells. Input
  width is validated for both cells against each forward call because an automatic cell may still
  be unbound and supplied cells can carry independently constructed input schemas.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [data type glossary](../../../../glossary.md#data-type--datatype); [carry glossary](../../../../glossary.md#term-carry); [schemas glossary](../../../../glossary.md#term-schema))</span>
- Permanently register the forward cell under exact child name `forward` and the backward cell
  under exact child name `backward`, in that order. Complete two-child validation must precede
  either parent link.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Widen the existing already-tested package-private
  `Module.registerNamedChildren(Map<String, ? extends Module>)` primitive to `protected final` so
  a layer-package subclass can atomically install the two descriptive children. Do not add a
  second named-child primitive, public child mutation, detach/reparent operation, or generic
  recurrent owner.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [owner glossary](../../../../glossary.md#term-owner))</span>
- Keep forward and backward parameters independent. Automatic construction creates two distinct
  cell objects, each with its own private reservation group and exact supplied seed. Equal numeric
  seed values are allowed but are still two separately owned initialization configurations and
  create distinct parameter Tensor identities.
- Preserve each cell's existing parameter order, gate packing, formula, optional bias policy,
  automatic first-use binding, retry, strict-load, replacement, and mode-insensitive behavior.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Add the four forward overload forms already proven by the one-directional sequences: explicit
  states plus lengths, explicit states with all-valid derived lengths, default states plus
  lengths, and default states with all-valid derived lengths.
- Keep the fully static [time-major](../../../../glossary.md#term-time-major) input contract `[time, batch, inputSize]` and static Java
  `long[]` length compatibility contract. Lengths remain construction [metadata](../../../../glossary.md#term-metadata), not [Tensor](../../../../glossary.md#tensor) values.
- Forward traversal at original time `t` uses the current stable active set
  `L[b] > t`, [axis](../../../../glossary.md#term-axis)-zero input `SELECT`, active-row index [leaf](../../../../glossary.md#term-leaf), axis-zero [`GATHER`](../../../../glossary.md#gather), survivor gather,
  and one concrete forward-cell call.
- Backward traversal at reverse depth `d` uses the same active set `L[b] > d`. For each active row
  `b`, its compact input [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) is `(L[b] - 1 - d, b)`. One dense [INT64](../../../../glossary.md#data-type--datatype) coordinate leaf shaped
  `[activeCount(d), 2]` and one `input.gatherNd(coordinates)` [occurrence](../../../../glossary.md#term-occurrence) construct the compact
  `[activeCount(d), inputSize]` reverse input without touching a padded position.
- Carry backward state between reverse depths through the same stable survivor-position policy as
  the current forward packing. Invoke the backward cell exactly once for every non-empty reverse
  depth. LSTM shares each exact active/survivor index leaf between hidden and cell gathers.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Flatten the reverse-depth hidden outputs in increasing reverse-depth order with one axis-zero
  `CONCAT` when more than one reverse step exists; retain the exact sole output when there is one.
  For each original time `t`, build one dense INT64 [alignment](../../../../glossary.md#term-alignment) index leaf and one axis-zero
  `GATHER` from that flattened Tensor so backward rows return to `active(t)` order. This is one
  alignment expression per represented time step, not one expression per logical row.
- Merge each exact forward packed output first and its exact aligned backward output second with
  `Tensor.concat(-1, forwardOutput, backwardOutput)`. Each merged output has [Shape](../../../../glossary.md#shape)
  `[activeCount(t), 2 * hiddenSize]` and exact common promoted floating type.
- Restore all directional final states to original batch order with shared dense INT64 final-row
  indices and axis-zero <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`CONCAT`/`GATHER` composition over initial state followed by flattened
  directional step states. RNN/GRU return forward then backward final hidden. LSTM returns forward
  final hidden/cell followed by backward final hidden/cell.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- Preserve exact initial-state references when no length is positive. Default-state overloads
  create distinct fresh zero leaves in documented direction/state order and return those exact
  references for an all-zero request. Neither cell initializes when no step is represented.
- Complete all caller-controlled constructor and forward validation before child installation,
  default-state/index allocation, automatic initialization, or expression creation as applicable.
- Add focused public-surface, ownership/state-path, reverse-prefix, merge/alignment, state,
  automatic-seed, strict-load, validation/effect-order, provenance, graph-size, mode, replacement,
  and exclusion tests for all three concrete families.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph))</span>
- Finalize affected Javadocs, layers package documentation, Training API, glossary, task evidence,
  and master-plan status in the required separate clean documentation-focused context.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- An arbitrary direction count, direction enum/list, direction registry, multidimensional spatial
  recurrence, duplicate forward/reverse directions, stacked recurrent layers, residuals,
  recurrent dropout, attention, or projection layers. The current one-dimensional time axis has
  exactly two distinct complete traversal orders.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [spatial glossary](../../../../glossary.md#term-spatial-axes); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- A generic `RecurrentCell`, `RecurrentSequence`, `BidirectionalSequence<S>`, erased state tuple,
  type switch, reflection, callback-driven per-step cell invocation, public packing plan, common
  recurrent base/interface, or duplicated generic result abstraction.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback))</span>
- `SUM`, `AVERAGE`, configurable merge, no-merge/raw-direction result mode, a merge enum, caller
  callback, projection-after-concat, or an implicit default. This task selects only ordered
  forward-then-backward final-axis `CONCAT`.
- Reversing the complete time [axis](../../../../glossary.md#term-axis), using `Tensor.flip(0)` as the padded-row policy, reading a
  padded suffix value, inferring padding from numeric zero, sorting rows by length, or exposing a
  row permutation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [permutation glossary](../../../../glossary.md#permutation))</span>
- Dense padded output, caller-selected padding output, unpacking, batch-first input, a Tensor
  mask, runtime Tensor lengths, Java Boolean masks, dense post-cell masking, or runtime-dependent
  loop count/active Shape.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [Tensor glossary](../../../../glossary.md#tensor); [mask glossary](../../../../glossary.md#term-boolean-mask); [runtime glossary](../../../../glossary.md#term-runtime); [Boolean glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>
- A recurrent Model scan/control-flow operation, body/subgraph representation, carried-value
  tuple, runtime early exit, or reinterpretation of `CUM_SUM`/`CUM_PROD` as recurrent [scan](../../../../glossary.md#term-cumulative-scan).
- Changes to current one-directional cells, sequences, results, formulas, gate packing, default
  states, construction-time length semantics, or state-path schema.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- A new [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation)/helper/public method, direct `Operation` construction, new compiler rule,
  numerical evaluator, backend capability, execution route, runtime work-skipping guarantee, or
  end-to-end value claim.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Hidden mutable recurrent state, a Buffer for current state, cached inputs/lengths/indices/results,
  state reset/detach, a runtime tape, NN backward method, Tensor gradient field, optimizer,
  Training session, checkpoint bytes, or serialization.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Buffer glossary](../../../../glossary.md#term-buffer); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [Training session glossary](../../../../glossary.md#training-session--trainingsession); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Adding bidirectional recipes to `ModuleFactory`. The new direct automatic constructors already
  remove two-cell assembly while keeping two seeds visible. Extend the closed standard factory
  only after these concrete families are stable and a caller proves that another recipe removes
  real construction burden; do not widen this task's public facade speculatively.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [facade glossary](../../../../glossary.md#term-facade))</span>
- Changes to Training Java source, Tensor or Compile API references, Model capabilities,
  Compiler, Runtime, Prepare, Engine, backend source, Gradle/build structure, architecture
  contracts/ADRs/tests, conformance/integration tests, the global roadmap, active CPU work, other
  master plans, or later NN task files.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed task specifications for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0021–0024.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Completed vanilla-RNN cell](0012-vanilla-rnn-cell.md)<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn))</span>
- [Completed GRU cell](0013-gru-cell.md)<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- [Completed LSTM cell](0014-lstm-cell.md)<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- [Completed static packed RNN sequence](0015-static-packed-rnn-sequence.md)
- [Completed static packed GRU sequence](0016-static-packed-gru-sequence.md)
- [Completed static packed LSTM sequence](0017-static-packed-lstm-sequence.md)
- [Completed typed functional Model topology](0018-typed-functional-model-topology.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [topology glossary](../../../../glossary.md#term-topology))</span>
- [Completed automatic Linear initialization](0019-automatic-first-forward-linear-initialization.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [Completed automatic recurrent initialization and defaults](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Completed initialized Embedding](0020a-initialized-embedding.md)<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding))</span>
- [Completed stateless standard ModuleFactory](0020b-stateless-standard-module-factory.md)
- [Completed Model Gather-ND expressions](../../../modules/model/tasks/0018f-gather-nd-tensor-expressions.md)<span data-glossary-navigation="true"> ([Gather-ND glossary](../../../../glossary.md#gather-nd))</span>
- [Completed Model composition expressions](../../../modules/model/tasks/0017l-tensor-composition-expressions.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns cell-specific directional composition, independent module state trees, construction-time
  length interpretation, explicit default-state conveniences, and typed results. Model remains the
  sole owner of Tensor identity, descriptors, eager leaves, Shape algebra, promotion, SELECT,
  GATHER, GATHER_ND, CONCAT, cell operations, and provenance.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [eager glossary](../../../../glossary.md#term-eager-evaluation); [Shape glossary](../../../../glossary.md#shape); [GATHER glossary](../../../../glossary.md#gather); [provenance glossary](../../../../glossary.md#provenance))</span>
- Recurrent hidden and cell values remain caller-/expression-threaded Tensors, never Parameters,
  Buffers, retained fields, optimizer state, or runtime tape.<span data-glossary-navigation="true"> ([Parameters glossary](../../../../glossary.md#term-parameter); [Buffers glossary](../../../../glossary.md#term-buffer); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>
- [`Module`](../../../../glossary.md#term-module) retains no universal forward method. Every new container is a direct final `Module`
  subclass with a truthful family-specific signature and remains outside <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`UnaryTensorModule` and
  `Sequential`.
- Cell identity and state must not be shared between directions. One cell shares its own exact
  parameter leaves only across time within that direction. The other direction owns another cell,
  wrappers, leaves, seed configuration, and state paths.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- The existing named-child primitive remains the sole parent-link owner. Its visibility change
  enables an actual subclass consumer but does not change exclusive ownership, cycle checks,
  validate-before-install atomicity, traversal order, or state-path grammar.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- Java lengths may specialize static expression structure. No current API may claim runtime
  length reuse or graph-topology independence from those concrete Java values.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology))</span>
- `GATHER_ND` coordinates express reverse valid-prefix selection as ordinary Model metadata.
  They do not evaluate index bounds, execute values, or prove backend support in this NN task.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Merged outputs are ordinary `CONCAT` expressions. Compiler-owned existing derivative rules may
  later traverse their exact ancestry; NN neither invokes compiler internals nor claims numerical
  backpropagation or execution.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [backpropagation glossary](../../../../glossary.md#term-backpropagation))</span>
- If implementation needs a new Model semantic, dependency, runtime scan, public generic recurrent
  abstraction, hidden state, different merge, architecture change, ModuleFactory change, or path
  outside the exact list, stop and report the blocker rather than widening the task.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scan glossary](../../../../glossary.md#term-cumulative-scan); [widening glossary](../../../../glossary.md#term-widening))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.nn.layers` — owns the six new public cell-family-specific
  bidirectional sequence/result types and their complete static directional composition.
- `io.github.pho001.synaptik.nn.module` — widens only the existing atomic named-child installation
  primitive from package-private to protected for the concrete two-child layer consumer.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

Existing [Model](../../../../glossary.md#term-model) packages are consumed unchanged for [Tensor](../../../../glossary.md#tensor), [data type](../../../../glossary.md#data-type--datatype), [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [eager](../../../../glossary.md#term-eager-evaluation) index,
SELECT, [GATHER](../../../../glossary.md#gather), GATHER_ND, CONCAT, and [provenance](../../../../glossary.md#provenance) contracts. No package or [dependency](../../../../glossary.md#term-dependency) is added.

Type placement:

- `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
  `BidirectionalLstmSequence` — concrete public directional composition belongs beside the current
  recurrent cells and static sequence containers in `nn.layers`.
- Their matching `ForwardResult` records — NN owns typed compact output and continuation-state
  values; they are not Model multi-output producers or Runtime results.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `Module.registerNamedChildren` — Module remains the only type able to validate and install
  parent links atomically.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- Three same-package tests own the exact family APIs and behavior; `ModuleTreeTest` owns the
  protected atomic named-child semantics regression, and <span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>`ModelTest` updates its existing exact
  Module-surface assertion for the planned protected visibility.

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

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
merge selector, direction selector, state getter, or overload. Inherited final [`Module`](../../../../glossary.md#term-module) APIs remain
available normally.

Each result [canonical](../../../../glossary.md#term-canonical-form) constructor rejects null `packedOutputs`, traverses once to reject the first
null [element](../../../../glossary.md#term-element) while making an [immutable](../../../../glossary.md#term-immutable) structural [snapshot](../../../../glossary.md#term-snapshot), then rejects state components in
record declaration order. It retains exact [Tensor](../../../../glossary.md#tensor) references and performs no [descriptor](../../../../glossary.md#term-descriptor)
relationship [validation](../../../../glossary.md#term-validation), copy, expression creation, [identity](../../../../glossary.md#term-identity-meanings) allocation, or execution.

## Directional state [schema](../../../../glossary.md#term-schema) and [ownership](../../../../glossary.md#term-ownership-meanings)

Every container declares no direct [Parameter](../../../../glossary.md#term-parameter) or [Buffer](../../../../glossary.md#term-buffer) and owns exactly two children:

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

The same names apply to all three families; [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) retain their existing packed [Shape](../../../../glossary.md#shape) and gate
meaning within each path. Inputs, lengths, initial/final states, [eager](../../../../glossary.md#term-eager-evaluation) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/index leaves,
directional outputs, [alignment](../../../../glossary.md#term-alignment) values, and merged outputs never enter [module](../../../../glossary.md#term-module) discovery or state
dictionaries.

`train()` and `eval()` propagate to `forward` then `backward` in normal Module preorder. Forward
composition is mode-insensitive. Strict state export/load uses the same child order. A complete
dictionary may bind both automatic cells without running either initializer. Each reservation
group remains cell-local; generic strict load validates the complete tree before installing any
[binding](../../../../glossary.md#term-binding) under the existing Module contract.

## Construction and [seed](../../../../glossary.md#term-seed) contract

Supplied-cell construction performs:

1. reject null `forwardCell`, then null `backwardCell`;
2. reject repeated exact cell [identity](../../../../glossary.md#term-identity-meanings);
3. require equal configured positive hidden sizes;
4. require the exact configured [parameter](../../../../glossary.md#term-parameter) data types to match;
5. build the encounter-ordered `forward`, `backward` named-child [snapshot](../../../../glossary.md#term-snapshot); and
6. invoke the now-protected [atomic](../../../../glossary.md#resource-transaction-and-rollback) named-child primitive once.

All checks complete before either parent link changes. A name, identity, cycle, or existing-[owner](../../../../glossary.md#term-owner)
failure leaves both candidates unchanged. The constructor retains exact cells and creates no
[Tensor](../../../../glossary.md#tensor), Parameter, generator, index, state, or result.

Automatic construction validates the common hidden size, [bias](../../../../glossary.md#term-bias), [data type](../../../../glossary.md#data-type--datatype), initialization policy,
gate multiplier, hidden-[weight](../../../../glossary.md#term-weight) [Shape](../../../../glossary.md#shape)/count, and Java-array limits under the existing matching cell
constructor. It then creates the forward automatic cell with `forwardSeed`, creates the backward
automatic cell with `backwardSeed`, and atomically registers them only after both constructors
return. Cell construction creates reservations but no Tensor, Parameter wrapper, or generator.

On first represented use, the complete forward request is prevalidated from [immutable](../../../../glossary.md#term-immutable) [schemas](../../../../glossary.md#term-schema).
The first forward cell step initializes only the forward cell from its seed; the first reverse
step initializes only the backward cell from its seed. Each random cell starts one fresh exact
`L64X128MixRandom` stream per attempt, while zero/one policies create no generator. There is no
cross-direction initialization [transaction](../../../../glossary.md#term-transaction): if forward initialization and expressions succeed
but backward initialization later fails, the forward cell remains bound and the backward cell
remains independently retryable. Completed draws, Tensor IDs, allocations, and expression prefixes
are not rolled back.

## Forward input and result contract

The most-explicit [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) call requires:

| Input | Required contract |
|---|---|
| `input` | non-null floating fully static [rank](../../../../glossary.md#term-rank)-three `[time, batch, inputSize]` |
| `forwardInitialHidden` | non-null floating fully static `[batch, hiddenSize]` |
| `backwardInitialHidden` | non-null floating fully static `[batch, hiddenSize]` |
| `lengths` | non-null Java `long[]` of exactly [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) entries in `[0, time]` |

The [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) call additionally requires exact forward/backward initial cell [Tensors](../../../../glossary.md#tensor) with the same
`[batch, hiddenSize]` contract. Directional initial state types may differ only where each cell's
existing projection/state promotion accepts them and the final merged hidden outputs have the same
exact [data type](../../../../glossary.md#data-type--datatype) required by `CONCAT`.

For copied length `L[b]`:

```text
steps = max(L), or zero for an empty batch
active(k) = [b in ascending original order where L[b] > k]
```

The result `packedOutputs` contains exactly `steps` merged Tensors. Entry `t` has [Shape](../../../../glossary.md#shape)
`[activeCount(t), 2 * hiddenSize]`. Its first final-[axis](../../../../glossary.md#term-axis) interval is the exact forward-cell output
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
initial-state reference. No index, expression, automatic [parameter](../../../../glossary.md#term-parameter), or result-state wrapper Tensor
is created. The result record itself is an ordinary Java value.

Overload behavior is:

| Supplied [arguments](../../../../glossary.md#term-argument) | Derived values |
|---|---|
| input, all explicit directional states, lengths | none |
| input, all explicit directional states | one all-valid Java length array |
| input, lengths | distinct typed-zero states for both directions |
| input | one all-valid array plus distinct typed-zero states for both directions |

Default RNN/GRU state creation order is forward hidden then backward hidden. Default LSTM order is
forward hidden, forward cell, backward hidden, backward cell. Each is a fresh [eager](../../../../glossary.md#term-eager-evaluation) unlabeled,
[provenance](../../../../glossary.md#provenance)-free, non-[gradient](../../../../glossary.md#term-gradient) [leaf](../../../../glossary.md#term-leaf) shaped `[batch, hiddenSize]` with the common cell parameter type
and is never retained. Omitted lengths [mean](../../../../glossary.md#term-mean) `L[b] = time` for every row.

## Packing, reverse traversal, [alignment](../../../../glossary.md#term-alignment), and merge

### Forward traversal

For `t` in `[0, steps)`, preserve the current static packing policy:

1. create `input.select(0, t)`;
2. create one dense [INT64](../../../../glossary.md#data-type--datatype) active-original-row [leaf](../../../../glossary.md#term-leaf) for `active(t)`;
3. [gather](../../../../glossary.md#gather) the selected input on [axis](../../../../glossary.md#term-axis) zero;
4. at `t == 0`, gather the forward initial state with the same active leaf; later gather the prior
   forward compact state with one relative-survivor leaf;
5. for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), share the same state leaf between hidden and cell gathers;
6. invoke the concrete forward cell once; and
7. retain the exact next-hidden output and, for LSTM, exact next-cell carried state.

### Backward traversal

For reverse depth `d` in `[0, steps)`:

1. enumerate `active(d)` in ascending original row order;
2. create one dense, contiguous, unlabeled, provenance-free, non-gradient INT64 coordinate leaf
   shaped <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`[activeCount(d), 2]` whose row for original [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) `b` is
   `[L[b] - 1 - d, b]`;
3. create `input.gatherNd(coordinates)`, yielding `[activeCount(d), inputSize]`;
4. at `d == 0`, create one active-original-row leaf and gather backward initial state; later create
   one relative-survivor leaf and gather prior reverse-depth state;<span data-glossary-navigation="true"> ([gather glossary](../../../../glossary.md#gather))</span>
5. for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), share the exact state [leaf](../../../../glossary.md#term-leaf) between hidden and cell gathers;
6. invoke the concrete backward cell once; and
7. retain exact reverse-depth hidden output and, for LSTM, exact next-cell carried state.

Every [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) satisfies `0 <= L[b] - 1 - d < L[b] <= time`, so no backward input [occurrence](../../../../glossary.md#term-occurrence)
addresses [padding](../../../../glossary.md#padding). For lengths `[5, 3, 1]`, reverse depth coordinates are:

```text
d=0: [(4,0), (2,1), (0,2)]
d=1: [(3,0), (1,1)]
d=2: [(2,0), (0,1)]
d=3: [(1,0)]
d=4: [(0,0)]
```

### Backward [alignment](../../../../glossary.md#term-alignment) and merge

Let `reverseOutputs[d]` retain the backward hidden output for reverse depth `d`, and let
`offset[d]` be the sum of active counts before `d`. Flatten those outputs along [axis](../../../../glossary.md#term-axis) zero in
increasing `d` order. For original time `t` and row `b in active(t)`, its flat backward index is:

```text
d = L[b] - 1 - t
flatIndex(t, b) = offset[d] + position of b in active(d)
```

One [eager](../../../../glossary.md#term-eager-evaluation) [INT64](../../../../glossary.md#data-type--datatype) [leaf](../../../../glossary.md#term-leaf) contains those indices in ascending `active(t)` order. One axis-zero [gather](../../../../glossary.md#gather)
creates the aligned backward [Tensor](../../../../glossary.md#tensor) for the whole compact time step. Then:

```text
packedOutputs[t] = Tensor.concat(
        -1,
        forwardOutputs[t],
        alignedBackwardOutputs[t])
```

The implementation must not create one Tensor `SELECT` per logical recurrent row merely to align
reverse outputs. Java may fill primitive index arrays with `O(sum(L))` [scalar](../../../../glossary.md#term-scalar) work, but the Tensor
[graph](../../../../glossary.md#graph) adds only one alignment index leaf, one GATHER, and one merge CONCAT per represented time.

### Final-state restoration

For a non-empty request, construct one final-row [INT64](../../../../glossary.md#data-type--datatype) index [leaf](../../../../glossary.md#term-leaf) in original [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) order and
reuse it across both directions and, for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), across hidden and cell state. For each forward or
cell-state stream, one source CONCAT consumes the exact initial state first and every compact step
state afterward in traversal order; that [variadic](../../../../glossary.md#term-variadic) CONCAT is the flattening, so no intermediate
flatten expression is created. The backward hidden source CONCAT instead consumes its exact
initial state followed by the already-created flattened reverse hidden [Tensor](../../../../glossary.md#tensor). [Gather](../../../../glossary.md#gather) each source
with the shared final-row leaf. A zero-length row indexes its original initial-state row. A
positive row indexes the compact state at its exit step. Forward exit step is original time
`L[b] - 1`; backward exit depth is `L[b] - 1`, which is the state after consuming original time
zero.

This composition replaces per-row SELECT/STACK restoration inside the new classes only. Existing
one-directional sequence [provenance](../../../../glossary.md#provenance) remains unchanged. It uses current exact CONCAT/GATHER
semantics and preserves original row order with a bounded number of [graph](../../../../glossary.md#graph) occurrences.

## [Validation](../../../../glossary.md#term-validation) and side-effect order

The most-explicit [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) call performs:

1. reject null `input`, `forwardInitialHidden`, `backwardInitialHidden`, then `lengths`;
2. read immutable configured hidden-size/type facts for both cells and require constructor
   invariants still hold;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
3. require input floating type, rank three, fully static Shape, Java collection limits, and a
   positive input feature extent accepted by both cell schemas;<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>
4. validate forward initial hidden then backward initial hidden for floating type, rank two,
   fully static `[batch, hiddenSize]`, and direction-specific promotion;
5. validate length count and entries from the caller array in encounter order while deriving
   `steps`, with caller coordination required until the later [snapshot](../../../../glossary.md#term-snapshot);
6. prevalidate every distinct forward and reverse compact Shape, both cell formulas, survivor
   gathers, reverse coordinate GATHER_ND result, flatten/alignment GATHER, ordered final-axis
   CONCAT merge, final-state source CONCAT/GATHER, checked <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span><span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [axis glossary](../../../../glossary.md#term-axis))</span>`2 * hiddenSize`, count, primitive-array,
   and Tensor-ID-independent Shape/layout fact;<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
7. clone lengths immediately before traversal and use only the snapshot afterward; and
8. only then create [eager](../../../../glossary.md#term-eager-evaluation) leaves, initialize cells when reached, and construct expressions.

[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) null/state validation order is `input`, `forwardInitialHidden`, `forwardInitialCell`,
`backwardInitialHidden`, `backwardInitialCell`, `lengths`. State [descriptor](../../../../glossary.md#term-descriptor) checks follow that same
direction/state order. Complete hidden/cell formula and final-merge [preflight](../../../../glossary.md#term-preflight) precedes effects.

Default-state and all-valid overloads first complete every validation knowable without the derived
values. They then create an all-valid [host](../../../../glossary.md#term-host) array when required, create zero states in the specified
order, and delegate to one [canonical](../../../../glossary.md#term-canonical-form) traversal implementation. They must not duplicate either
direction loop.

Every ordinary caller-controlled null, type, [rank](../../../../glossary.md#term-rank), dynamic-[Shape](../../../../glossary.md#shape), [feature](../../../../glossary.md#term-feature), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), length,
promotion, cell [schema](../../../../glossary.md#term-schema), [GATHER](../../../../glossary.md#gather)/GATHER_ND, CONCAT, count, or Java-limit failure precedes the first
new [Tensor](../../../../glossary.md#tensor) ID, eager storage, random generator, draw, [parameter](../../../../glossary.md#term-parameter) [publication](../../../../glossary.md#term-publication), or expression. After
effects start, allocations, draws, published parameter groups, eager leaves, Tensor IDs, and
expression prefixes are not rolled back. No partial result record is returned and the container
retains no per-call value.

Validation is not a [transaction](../../../../glossary.md#term-transaction) with concurrent parameter replacement or strict load. Actual
cell calls observe their then-current compatible [bindings](../../../../glossary.md#term-binding). Callers coordinate forward
construction, replacement, loading, mode changes, and discovery when one cross-direction snapshot
matters.

## [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings) and [provenance](../../../../glossary.md#provenance) order

After overload-specific derived values exist, successful construction order is exact:

1. all forward steps in increasing original time, each using SELECT, active/survivor leaves,
   GATHER occurrences, and the exact family cell chain;
2. all backward steps in increasing reverse depth, each using coordinate leaf, GATHER_ND,
   active/survivor leaf, state GATHER occurrence(s), and the exact family cell chain;<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
3. zero or one backward-hidden flatten CONCAT: no Tensor when there is one reverse output, one
   axis-zero CONCAT when there are two or more;<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
4. for each original time in increasing order, one alignment index leaf, one backward alignment
   GATHER, then one forward-first final-axis CONCAT;<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
5. one final-row index [leaf](../../../../glossary.md#term-leaf);
6. forward hidden source CONCAT then [GATHER](../../../../glossary.md#gather);
7. for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), forward cell source CONCAT then GATHER;
8. backward hidden source CONCAT then GATHER; and
9. for LSTM, backward cell source CONCAT then GATHER.

Each final-state source CONCAT has at least two inputs: the exact initial state plus either the
ordered compact step-state Tensors directly or the already-flattened backward hidden Tensor. When
the backward hidden stream has one step, that exact step Tensor is used without a one-input
flatten CONCAT. The same exact final-row index leaf is reused by every final-state gather. LSTM
shares each state [carry](../../../../glossary.md#term-carry) index leaf hidden first then cell, preserving its established component
order.

Every represented step calls the same exact cell object for its direction and constructs fresh
[producer](../../../../glossary.md#term-producer)/Tensor identities. All forward step projections retain exact forward [parameter](../../../../glossary.md#term-parameter) leaves;
all backward step projections retain exact backward parameter leaves. No leaf identity crosses
directions. Temporal state ancestry remains explicit inside each direction, and CONCAT exposes
both ancestries to current [compiler](../../../../glossary.md#term-compiler) exact-identity traversal without [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owning [gradients](../../../../glossary.md#term-gradient).

Tests must lock this relative producer and Tensor-ID order, including one-step exact counts after
the implementation fixes its private [preflight](../../../../glossary.md#term-preflight), but must not assert numerical values, [runtime](../../../../glossary.md#term-runtime)
[kernel](../../../../glossary.md#kernel) count, [backend](../../../../glossary.md#backend) support, or compiler internals.

## Replacement, load, mode, and continuation behavior

- `forwardCell()` and `backwardCell()` return the exact stable children.
- Parameter replacement in one direction changes later expressions in only that direction. It
  neither replaces nor aliases the other direction's wrapper or Tensor.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Strict load uses the complete path set and ordinary validate-before-install semantics. A
  dictionary exported from an equivalently configured initialized bidirectional container can
  bind both automatic children without random generation. Forward later validates that both loaded
  input widths accept the same input.
- Mode propagation visits the forward subtree then backward subtree. Both cells and the container
  construct the same expressions in training and evaluation modes.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Returned directional final states are exact continuation inputs for a later caller-controlled
  call. The module never installs or retains them.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Result list snapshots remain immutable while every Tensor reference and producer ancestry
  remains exact.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [producer glossary](../../../../glossary.md#term-producer))</span>

## Performance and [graph](../../../../glossary.md#graph)-size boundary

Let `T = max(lengths)` and `S = sum(lengths)`.

- Exactly `T` forward-cell calls and `T` backward-cell calls are constructed when `T > 0`; each
  call is batched over its compact active rows. They represent `2S` logical recurrent row
  applications, not `2 * time * batch` dense rows.
- Reverse input uses one coordinate leaf and one GATHER_ND per reverse depth, not one SELECT per
  active row.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Reverse output [alignment](../../../../glossary.md#term-alignment) uses at most one flatten CONCAT plus exactly `T` alignment leaves,
  `T` [GATHER](../../../../glossary.md#gather) occurrences, and `T` merge CONCAT occurrences, not `S` [Tensor](../../../../glossary.md#tensor) row selections.
- Final restoration uses one shared final index leaf and one source CONCAT/GATHER pair per returned
  directional state, rather than one SELECT per row plus STACK.
- Java construction still performs `O(S)` primitive coordinate/index filling and the Tensor graph
  remains statically specialized to the copied lengths. There is no claim about compiled node
  optimization, backend fusion, kernel count, numerical execution cost, or runtime work skipping.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The implementation must use primitive arrays and bounded indexed loops. It must not allocate a
  callback, boxed state tuple, stream pipeline, reflection object, map lookup, or polymorphic
  direction strategy per recurrent step or logical row.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [lookup glossary](../../../../glossary.md#term-lookup))</span>

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
`ModuleFactory`, its test/package [Javadoc](../../../../glossary.md#term-javadoc), current cells/sequences/results/tests,
initialization source, Training Java, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) APIs, [Model](../../../../glossary.md#term-model) source/tests/capabilities,
architecture files/tests/ADRs, global roadmap, [Gradle](../../../../glossary.md#term-gradle)/build files, [Compiler](../../../../glossary.md#term-compiler), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare),
[Engine](../../../../glossary.md#term-engine), [backends](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance)/integration tests, other [planning](../../../../glossary.md#term-planning) areas, or later task files.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify only the seventeen exact paths above. The six public types, three
focused family test [owners](../../../../glossary.md#term-owner), the foundational visibility/semantics test, the existing exact-surface
assertion, and targeted documentation form one cohesive capability. The seventeenth path is the
coordinator-authorized correction for the [planning](../../../../glossary.md#term-planning) omission discovered by the first failed full
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [run](../../../../glossary.md#term-run); it changes only the stale visibility assertion. If implementation requires a shared
generic helper, another test [owner](../../../../glossary.md#term-owner), ModuleFactory recipe, current sequence refactor, new package,
or eighteenth path, stop and report the concrete need rather than expanding scope.

## Acceptance criteria

- The exact six type-safe public APIs exist with no generic recurrent base, direction/merge enum,
  callback, builder, hidden state, or extra surface.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback))</span>
- Supplied constructors validate both cells and atomically own exact [identity](../../../../glossary.md#term-identity-meanings)-distinct children as
  `forward` then `backward`; automatic constructors create independent cell identities and seed
  configurations.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- Recursive state paths/order, discovery, state dictionaries, strict load, replacement isolation,
  and forward-then-backward mode propagation match the exact contract.
- Static lengths alone define both traversals. Reverse coordinates address only valid-prefix
  positions, including unsorted, zero, and mixed lengths, and never reverse or read padding.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [padding glossary](../../../../glossary.md#padding))</span>
- Each direction invokes its cell once per non-empty compact step, shares parameters only across
  time within that direction, and preserves fresh temporal producer ancestry.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [producer glossary](../../../../glossary.md#term-producer))</span>
- Backward outputs are restored to original time and batch order by flattened CONCAT plus one
  GATHER per represented original time. No per-logical-row Tensor alignment expression or dense
  padded output is introduced.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [GATHER glossary](../../../../glossary.md#gather); [Tensor glossary](../../../../glossary.md#tensor); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Every visible packed output is an exact ordered forward-then-backward final-axis CONCAT with
  Shape <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape))</span>`[activeCount(t), 2 * hiddenSize]`.
- RNN/GRU return separate forward/backward final hidden states; LSTM returns separate hidden/cell
  states grouped forward then backward. Zero-length and all-zero exact-reference semantics pass.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Default-state and all-valid overloads create values in the specified order, retain nothing, and
  leave both automatic cells unbound for an all-zero request.
- Complete validation precedes caller-controlled effects; relative Tensor-ID/provenance order and
  documented cross-cell failure non-rollback are tested.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- The implementation contains no `flip(0)` whole-time reversal, dense [mask](../../../../glossary.md#term-boolean-mask), SUM merge, direct
  [`Operation`](../../../../glossary.md#operation), runtime scan, type switch, reflection, per-step callback, cross-direction shared
  cell/Parameter/Tensor identity, or non-Model dependency.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [Model glossary](../../../../glossary.md#term-model); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused and final NN tests pass after executable stabilization. Repository-wide validation is
  deferred because the change remains inside the existing NN/Model dependency boundary.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A separate clean documentation-focused pass finalizes every affected Javadoc and explanatory
  document, reviews unchanged Tensor/Compile APIs, Training Java, glossary impact, Model
  capabilities, architecture/tests, ModuleFactory, Gradle, conformance/integration, global
  roadmap, CPU, and other modules, and records reasoned no-change conclusions.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [modules glossary](../../../../glossary.md#term-module))</span>
- Final Javadoc, generated-page, public/protected/private surface, external-use, state path,
  forbidden-mechanism/import, Markdown link/anchor/fence, exact seventeen-path, frontier/status,
  newline, no-index whitespace, and whole-worktree `git diff --check` gates pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context runs focused suites while developing:

```bash
./gradlew :extensions:nn:test \
  --tests io.github.pho001.synaptik.nn.module.ModuleTreeTest \
  --tests io.github.pho001.synaptik.nn.module.ModelTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalRnnSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalGruSequenceTest \
  --tests io.github.pho001.synaptik.nn.layers.BidirectionalLstmSequenceTest
```

After executable Java and tests stabilize, [run](../../../../glossary.md#term-run) exactly one authoritative affected-[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :extensions:nn:test
```

The separate documentation-focused context reuses that unchanged Java evidence and runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates generated [Javadoc](../../../../glossary.md#term-javadoc) pages, exact public/protected surface, direct external Java
use of all three families, recursive paths and strict-load wording, forbidden imports/mechanisms,
local Markdown links/anchors/headings/fences, exact [scope](../../../../glossary.md#term-scope), status/frontier counts, final newlines,
and trailing whitespace in untracked files without staging them.

Repository-wide, architecture, [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), Training, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and integration
suites are deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) or their owning [checkpoints](../../../../glossary.md#term-checkpoint) because this task adds no [dependency](../../../../glossary.md#term-dependency),
architecture, generic Model semantic, execution behavior, or cross-module Java change. The
Module protected-surface change preserves the existing tested [ownership](../../../../glossary.md#term-ownership-meanings) rule inside the same [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
module and is covered by the focused/final NN suites.

## Documentation pass

After implementation and the authoritative [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite, hand the exact diff and evidence to a
separate documentation-focused clean context. It must follow the general, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [planning](../../../../glossary.md#term-planning),
and example profiles named by `docs/developer-guide/documentation-rules.md` and must:

- independently inspect the six new types, Module visibility change, all five affected tests, and
  current recurrent/Module contracts;
- finalize every new type/component/constructor/method Javadoc plus the affected [`Module`](../../../../glossary.md#term-module) and
  layers package Javadocs, including all parameters, results, nullability, state ownership,
  reverse-prefix, merge, failure, concurrency, and no-execution boundaries;<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [nullability glossary](../../../../glossary.md#term-nullability); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- update Training API and glossary for bidirectional packing, CONCAT order, directional final
  state, independent parameter trees/seeds, and the distinction from runtime scan and arbitrary
  multidirectional/stacked recurrence;<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- retain current one-directional and ModuleFactory explanations accurately and record why
  Tensor/Compile APIs, Training Java, Model capabilities, architecture/ADR/tests, build,
  conformance/integration, global roadmap, CPU, and other modules need no edit; and<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- record exact commands/results and reused Java evidence in this task before marking Complete.

The documentation pass must not rerun successful Java tests unless it changes executable Java or
records a concrete risk that requires a rerun.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0012–0014 provide the exact concrete cell [schemas](../../../../glossary.md#term-schema), formulas, and typed [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) state result.
- NN 0015–0017 provide the proven fully static time-major active-set, survivor, default shortcut,
  and original-order final-state contracts.<span data-glossary-navigation="true"> ([time-major glossary](../../../../glossary.md#term-time-major))</span>
- NN 0018–0020 provide atomic named topology, private automatic parameter reservations, two
  independent standard cell constructors, zero-state/all-valid conveniences, strict load, and
  exact shared-parameter/fresh-producer semantics.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [topology glossary](../../../../glossary.md#term-topology); [parameter glossary](../../../../glossary.md#term-parameter); [producer glossary](../../../../glossary.md#term-producer))</span>
- NN 0020A–0020B close initialized standard layer construction and leave directionality as the
  next ordered frontier.
- Current Model SELECT, GATHER, GATHER_ND, CONCAT, eager INT64 leaves, Shape/promotion,
  provenance, and compiler gradient coverage are complete.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [GATHER glossary](../../../../glossary.md#gather); [eager glossary](../../../../glossary.md#term-eager-evaluation); [INT64 glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- The user-authorized NN interleave remains file- and dependency-isolated from active CPU/backend
  and global-roadmap work.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>

## Follow-up tasks

- NN 0021 remains Draft for the cross-module recurrent scan/control-flow and runtime input-binding
  prerequisite. It must preserve this task's directional ownership, reverse-prefix, merge, and
  final-state semantics without treating static Java lengths as the target runtime API.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [scan glossary](../../../../glossary.md#term-cumulative-scan); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [target glossary](../../../../glossary.md#term-target); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0022 remains Draft for Data-owned [runtime](../../../../glossary.md#term-runtime) valid lengths after NN 0021.
- NN 0023–0024 remain Draft for optional [masks](../../../../glossary.md#term-boolean-mask) and the later integration [checkpoint](../../../../glossary.md#term-checkpoint).
- Add arbitrary multidimensional directions or stacked recurrent layers only when a concrete
  input layout and consumer define additional distinct traversal axes, merge, state, and ownership
  semantics. Do not generalize this two-direction contract merely from naming symmetry.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [axes glossary](../../../../glossary.md#term-axis))</span>
- Reassess ModuleFactory recipes only after the new concrete constructors are implemented and a
  caller proves that a recipe removes meaningful remaining assembly without hiding two seeds.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Architecture impact

Expected impact: None.

The architecture already assigns stateful neural-network composition, [parameters](../../../../glossary.md#term-parameter), [buffers](../../../../glossary.md#term-buffer),
mode, and functional conveniences to `extensions/nn`, with [Model](../../../../glossary.md#term-model) as its sole [dependency](../../../../glossary.md#term-dependency). This
task adds a concrete [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) composition over existing Model semantics and widens only an existing
[Module](../../../../glossary.md#term-module)-owned child-installation primitive for a real subclass consumer. It changes no module
boundary, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [operation](../../../../glossary.md#operation) [ownership](../../../../glossary.md#term-ownership-meanings), or [runtime](../../../../glossary.md#term-runtime) [lifecycle](../../../../glossary.md#lifecycle). Stop if implementation
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

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

The implementation context must provide the documentation context with this task, the exact
seventeen-path diff, affected APIs/behavior, focused and final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) commands/results, cell
initialization and failure effects, [producer](../../../../glossary.md#term-producer)/ID evidence, architecture constraints, and every
expected documentation/no-change topic listed above. The documentation context independently
reviews source and tests rather than accepting the handoff summary as proof.

## Local decisions

- Select only two directions. One time axis defines forward and reverse-valid-prefix orders; an
  arbitrary direction collection would duplicate those orders or invent unplanned spatial
  semantics.<span data-glossary-navigation="true"> ([spatial glossary](../../../../glossary.md#term-spatial-axes))</span>
- Select fixed ordered final-[axis](../../../../glossary.md#term-axis) `CONCAT`. It retains both directional features and makes the
  merge visible in provenance. <span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-software-feature); [provenance glossary](../../../../glossary.md#provenance))</span>`SUM` would erase direction identity and configurable merge would
  add policy surface without a second current consumer.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Own cells directly under `forward` and `backward`, not through two nested one-directional
  Sequence children. This produces concise stable paths and avoids building a padded reversed
  Tensor merely to reuse an API whose private loop cannot accept compact reverse coordinates.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Widen the existing [atomic](../../../../glossary.md#resource-transaction-and-rollback) named-child primitive to protected rather than sequentially calling
  `child(...)`. Sequential calls could strand the first supplied cell if validation of the second
  failed; a second primitive or wrapper module would duplicate ownership machinery or add a path
  segment.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Use GATHER_ND for one compact reverse input per depth. Whole-time `flip` is wrong for mixed
  lengths, and per-row SELECT/STACK would add avoidable `O(S)` [Tensor](../../../../glossary.md#tensor) expressions.
- Flatten reverse outputs and realign them with one GATHER per original time. This retains compact
  output semantics while keeping graph occurrence growth proportional to represented steps beyond
  the unavoidable recurrent cell graph.<span data-glossary-navigation="true"> ([GATHER glossary](../../../../glossary.md#gather); [graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Use CONCAT/GATHER final restoration in the new types and share one final index leaf across
  direction/state streams. Existing one-directional SELECT/STACK provenance remains unchanged.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Keep cell-family-specific loops and results. A callback/generic state machine would add per-step
  indirection or erase LSTM cell state; structural similarity alone does not justify it.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Leave ModuleFactory unchanged. The direct automatic constructors already hide two-cell assembly
  and deliberately expose both seeds; another factory recipe is not needed to complete this
  capability.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Known limitations

- Lengths are copied Java construction metadata and all participating Shapes are fully static.
  Different length values specialize the expression graph.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [Shapes glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph))</span>
- The result is a compact list by original time, not a dense padded Tensor. Consumers must retain
  the same lengths to interpret active extents.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Only hidden outputs are merged. LSTM cell states are carried internally and only directional
  final cell states are public.<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Only equal hidden sizes and exact common cell parameter types are accepted. Supplied direction
  biases may differ, but both cell input schemas must accept the same call input.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [biases glossary](../../../../glossary.md#term-bias); [schemas glossary](../../../../glossary.md#term-schema))</span>
- Construction is not atomic with concurrent parameter replacement/load, and two automatic cells
  do not form one initialization transaction.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Static graph construction omits padded rows from cell expressions but makes no runtime kernel,
  backend support, numerical value, or work-skipping guarantee.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [kernel glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend))</span>
- Runtime lengths, reusable graph topology across different length values, arbitrary masks,
  stacked recurrence, multidimensional directionality, and ModuleFactory recipes remain future
  work with separate consumer evidence.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only work completed in clean context `/root/nn_0020c_planning`. No production Java,
tests, [Javadoc](../../../../glossary.md#term-javadoc), [API](../../../../glossary.md#term-application-programming-interface-api) guide, glossary, architecture, build, global-roadmap, [CPU](../../../../glossary.md#term-cpu), [backend](../../../../glossary.md#backend), or later
task implementation was performed.

- Read the repository instructions, authoritative architecture contract, current architecture
  plan, planning guide and roadmap, NN master plan, completed NN 0012–0020B task records, actual
  Module/Topology/ModuleFactory and RNN/GRU/LSTM Cell/Sequence/result source/Javadocs/tests,
  directly used Model Tensor operations, Training API, glossary, and documentation rules/profiles.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Topology glossary](../../../../glossary.md#term-topology); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The [operation](../../../../glossary.md#operation) review found no expressibility blocker. Existing `GATHER_ND` can select one
  compact valid-prefix reverse input per depth, exact-type axis <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>[`GATHER`](../../../../glossary.md#gather) can realign flattened
  reverse outputs, and existing axis `CONCAT` can expose ordered directional features and restore
  final states. Whole-axis flip and dense masking were rejected for mixed valid lengths.<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-feature))</span>
- The API/ownership review selected three concrete family containers and three matching records,
  independent direct <span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`forward`/`backward` children, fixed CONCAT, and one protected visibility
  widening of the existing atomic named-child primitive. It found no need for a generic recurrent
  base, arbitrary directions, ModuleFactory recipe, Model operation, runtime scan, hidden state,
  dependency, or architecture change.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Targeted Ruby Markdown validation passed for this task and the NN master plan: every local link
  target exists, there are no fragment references requiring a foreign anchor, effective headings
  are unique, fences are balanced, terminal newlines are present, and trailing whitespace is
  absent.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Frontier validation passed: 0020C is the sole task and master-plan row with [scalar](../../../../glossary.md#term-scalar) status
  [`Ready`](../../../../glossary.md#term-ready); 0021–0024 remain Draft and have no task files.
- `git status --short docs/planning/extensions/nn` reported exactly the modified NN master plan
  and this untracked task file. Pre-existing concurrent CPU/backend and global-roadmap changes
  were inspected for attribution and left untouched, unstaged, unreverted, and unformatted.
- `git diff --check` and the tracked [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-master `git diff --check` passed. The task-file
  `git diff --no-index --check /dev/null` check returned the expected difference status with no
  whitespace diagnostic.
- No Java, Javadoc, Gradle, architecture, conformance, integration, Model, Compiler, Training,
  Runtime, Prepare, Engine, backend, CPU, or repository-wide test was run because this change is
  planning-only and changes no executable or authoritative architecture behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [run glossary](../../../../glossary.md#term-run))</span>

Implementation work completed in clean context `/root/nn_0020c_implementation`; executable Java
and tests are frozen for the mandatory separate documentation-focused [handoff](../../../../glossary.md#term-handoff).

- `./gradlew :extensions:nn:compileJava` passed after the six production types and Module
  visibility change were implemented.
- The original four-owner focused selection passed 23 tests: ModuleTree 9, bidirectional RNN 6,
  bidirectional GRU 4, and bidirectional LSTM 4; zero skips, failures, or errors.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
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
  construction, dense masks, whole-axis flips, and non-Model dependencies are absent.<span data-glossary-navigation="true"> ([recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [masks glossary](../../../../glossary.md#term-boolean-mask))</span>
- Whole-worktree `git diff --check` passed while concurrent CPU/backend/global-roadmap work
  remained untouched. Final exact-scope, untracked no-index whitespace, newline, Markdown, and
  explanatory-document checks remain for the documentation pass after its two documentation
  paths are finalized.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

Documentation work completed in independent clean context `/root/nn_0020c_docs`; no executable
Java or test changed after the frozen passing implementation evidence.

- Independent review found no executable, API, ownership, provenance, performance, architecture,
  dependency, or scope blocker. It confirmed the three concrete direct-<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>[`Module`](../../../../glossary.md#term-module) families, exact
  public/private surfaces, independent directional children/state/seeds, atomic validate-before-
  attach child ownership, valid-prefix-only reversal and original-order realignment, fixed
  forward-first final-axis CONCAT, directional final states, zero/default overloads, and the
  specified validation/effect/Tensor-identity order.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized meaningful Javadocs for the six new public types, the protected-final named-child
  primitive, and the layers package. The contracts now cover nullability, ownership, state paths,
  effects and failure boundaries, snapshot/threading limits, result identity, construction-time
  length specialization, and the absence of runtime scan, graph reuse, backend work-skipping, or
  numerical-execution guarantees.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [snapshot glossary](../../../../glossary.md#term-snapshot); [specialization glossary](../../../../glossary.md#term-specialization); [graph glossary](../../../../glossary.md#graph))</span>
- Updated the Training API and glossary with the concrete bidirectional families, stable state
  paths, valid-prefix reversal and alignment, forward-first CONCAT, directional continuation
  states, the <span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>`[3, 1, 2]` example, bounded static graph-size explanation, and current-versus-future
  runtime-scan boundary. `ModuleFactory` remains explicitly one-directional.
- Final `./gradlew :extensions:nn:javadoc` passed warning-free. Generated pages for every new
  public type, the layers package, and `Module` were inspected for the finalized contracts.
- External [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and reflection passed for all three families and exact result accessors.
  `javap -public` confirmed two constructors, two typed cell accessors, and four `forward`
  overloads per family plus exact record components; `javap -protected` confirmed the sole
  protected-final named-child primitive.
- Import and [dependency](../../../../glossary.md#term-dependency) checks found only [Model](../../../../glossary.md#term-model), current NN, [JDK](../../../../glossary.md#term-java-development-kit-jdk), and the existing one-way
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
families and results, protected [atomic](../../../../glossary.md#resource-transaction-and-rollback) child installation, the authorized stale surface assertion
correction, and focused tests. Each direct concrete loop performs forward compaction, valid-prefix
reverse [Gather-ND](../../../../glossary.md#gather-nd) traversal, flattened reverse realignment, ordered final-[axis](../../../../glossary.md#term-axis) CONCAT, and shared-
index directional state restoration without a generic recurrent abstraction. The executable diff
is frozen after the passing replacement authoritative [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite. Architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/tests,
[Model](../../../../glossary.md#term-model)/[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training Java APIs, existing cells/sequences, ModuleFactory, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/
[prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/[backends](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle)/dependencies, [conformance](../../../../glossary.md#term-conformance)/integration, global roadmap, [CPU](../../../../glossary.md#term-cpu), and
other [modules](../../../../glossary.md#term-module) require no change because the implementation composes current Model expressions
inside the existing NN-only boundary and changes no existing sequence behavior or [dependency](../../../../glossary.md#term-dependency).
The independent documentation pass confirmed that architecture/ADR/architecture tests need no
change because no module or dependency boundary changed. Model/Tensor/Compile/Training Java APIs,
existing cell/sequence semantics, `ModuleFactory`, compiler/runtime/prepare/Engine/backends,
Gradle/dependencies, conformance/integration tests, CPU work, the global roadmap, and other modules
need no change because 0020C composes existing Model expressions inside the existing NN-only
boundary. The Training [API](../../../../glossary.md#term-application-programming-interface-api) prose and glossary are the only explanatory documents affected.

## Completion summary

- Completed changes: Added type-safe bidirectional static RNN, GRU, and LSTM composition with
  independent forward/backward cell ownership, valid-prefix reversal, original-time realignment,
  forward-first CONCAT, and separate directional final states; widened the existing named-child
  primitive and corrected its stale exact-surface assertion.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Files changed or created: Exactly seventeen task paths: six new production types,
  `layers/package-info.java`, `Module.java`, `ModuleTreeTest.java`, the authorized stale protected-
  surface assertion in `ModelTest.java`, three new family tests, Training API, glossary, this task,
  and the NN master plan.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Focused 36/36 and replacement authoritative NN 280/280 passed; final
  Javadoc, generated pages, javap, external compilation/reflection, imports/dependencies,
  Markdown, exact-scope, newline, no-index, whitespace, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Independent clean context `/root/nn_0020c_docs` found no blocker and
  finalized all authorized documentation and evidence.
- Documentation impact: Finalized affected public/module/package Javadocs, Training API, glossary,
  task, and master evidence; all other documentation has a reasoned no-change conclusion.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Complete; final warning-free build and generated-page inspection passed.
- Glossary impact: Added the concrete bidirectional static packed sequence term and corrected the
  surrounding current-capability/future-scan boundary.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Unresolved issues: None.
- Required follow-up: None for task 0020C; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0021–0024 remain separate Draft work.

Status: Complete

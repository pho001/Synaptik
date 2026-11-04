# [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru) and [LSTM](../../glossary.md#term-long-short-term-memory-lstm) layer contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A [recurrent neural network](../../glossary.md#neural-network-families) (**RNN**) carries a hidden value from one sequence step to the next.
A [gated recurrent unit](../../glossary.md#neural-network-families) (**GRU**) uses learned controls to mix old and new hidden values;
[long short-term memory](../../glossary.md#neural-network-families) (**LSTM**) carries both hidden state and a separate cell state. State is
explicit here: a cell describes one transition, and a sequence composes transitions without
silently keeping the caller's final state for another call.

<a id="current-nn-recurrent-composition-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) recurrent composition contract

`RnnCell`, `GruCell`, and `LstmCell` each describe one recurrent step and extend [`Module`](../../glossary.md#term-module)
directly. [RNN](../../glossary.md#term-recurrent-neural-network-rnn) and [gated recurrent unit](../../glossary.md#term-gated-recurrent-unit-gru) ([GRU](../../glossary.md#neural-network-families)) cells accept input plus hidden state and return one
next-hidden [Tensor](../../glossary.md#tensor). The [long short-term memory](../../glossary.md#term-long-short-term-memory-lstm) ([LSTM](../../glossary.md#neural-network-families)) cell accepts input, hidden state, and cell
state and returns both next states. None retains caller-threaded [recurrent state](../../glossary.md#backpropagation-through-time-and-recurrent-state), and none fits the
one-input contract of `UnaryTensorModule` or `Sequential`.

`RnnSequence`, `GruSequence`, and `LstmSequence` are intentionally cell-specific one-directional
containers. Each permanently owns one exact matching cell. Their existing `forward` overloads
share the same static packing policy without claiming that the different cell-state [signatures](../../glossary.md#term-signature)
are interchangeable. Static calls are:

```java
RnnSequenceForwardResult rnnResult =
        rnnSequence.forward(input, initialHidden, lengths);
GruSequenceForwardResult gruResult =
        gruSequence.forward(input, initialHidden, lengths);
LstmSequenceForwardResult lstmResult =
        lstmSequence.forward(input, initialHidden, initialCell, lengths);
```

Here `input` must have a fully static [time-major](../../glossary.md#term-time-major) [Shape](../../glossary.md#shape) `[time, batch, inputSize]`,
`initialHidden` must have Shape `[batch, hiddenSize]`, and the LSTM call additionally requires
`initialCell` with that same Shape. `lengths` is a Java `long[]` with one value in `[0, time]` per
original [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) row. Each method validates directly from the caller array and, when at least one
step is represented, clones it immediately before traversal; neither array is retained. Callers
must coordinate mutation throughout [validation](../../glossary.md#term-validation) and any [snapshot](../../glossary.md#term-snapshot).

Each sequence also has an overload that omits lengths and treats every row as valid for the full
static time [extent](../../glossary.md#term-extent). Overloads that omit recurrent state create fresh [eager](../../glossary.md#term-eager-evaluation) typed-zero state with
Shape `[batch, hiddenSize]`, the cell's [parameter](../../glossary.md#term-parameter) type, no name, and `requiresGrad == false`.
RNN/GRU create one such [leaf](../../glossary.md#term-leaf); LSTM creates distinct hidden and cell leaves. The state is local to
that call and never retained. When every length is zero, no cell is invoked, an automatic cell
remains unbound, the output list is empty, and the final-state accessors return the exact explicit
or freshly derived initial references.

The copied lengths determine an **active batch** at each time step: original batch rows whose
length exceeds that step, kept in ascending original order. The sequence statically constructs
one compact batched cell expression for every non-empty step. It never examines Tensor values to
decide activity, so an all-zero input row remains active whenever its explicit length includes the
step.

For example, lengths `[5, 3, 1]` produce five packed outputs with active batch extents
`[3, 2, 2, 1, 1]`:

```text
time step          0      1      2      3      4
active rows      0,1,2   0,1    0,1     0      0
active extent      3      2      2      1      1
```

There are five Java cell calls—one per non-empty time step—and their compact Shapes represent
nine logical recurrent row applications rather than fifteen dense padded rows. For every
container, `packedOutputs().get(t)` is the exact next-hidden Tensor returned by its cell for that
compact step. `finalHidden()` restores one row per original batch entry: row 0 comes from step 4,
row 1 from step 2, and row 2 from step 0. A zero-length row instead uses its row from
`initialHidden`.

RNN and GRU [carry](../../glossary.md#term-carry) only hidden state. Their result types therefore contain `packedOutputs` and
`finalHidden`; when all lengths are zero, the list is empty and `finalHidden()` is the exact
`initialHidden` reference. LSTM carries hidden and cell state. `LstmSequence` publishes only its
compact next-hidden Tensors by step, carries each exact compact next-cell Tensor internally, and
returns both `finalHidden()` and `finalCell()` so the caller can continue recurrence. A
zero-length LSTM row uses the corresponding initial hidden and initial cell rows; an all-zero
request returns both exact initial-state references.

The matching `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
`BidirectionalLstmSequence` containers add exactly the two traversal orders of that same static
time [axis](../../glossary.md#term-axis). Each permanently owns [identity](../../glossary.md#term-identity-meanings)-distinct cells under `forward` and `backward`, so state
paths begin, for example, with `forward.inputWeight` and `backward.inputWeight`. The cells have
equal hidden width and exact parameter type but independent wrappers, parameter Tensor identities,
automatic reservations, and [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings). Supplied cells may differ in [bias](../../glossary.md#term-bias) presence. No parameter or
recurrent state is shared between directions; each cell shares only its own parameters across
time.

For example, this GRU request uses input Shape `[3, 3, inputSize]`, lengths `[3, 1, 2]`, hidden
width 64, and fresh default states:

```java
BidirectionalGruSequence encoder = new BidirectionalGruSequence(
        64,
        true,
        DataType.FLOAT32,
        ParameterInitialization.glorotUniform(),
        41L,
        42L);

BidirectionalGruSequenceForwardResult encoded =
        encoder.forward(input, new long[] {3, 1, 2});
```

Forward traversal visits original times from zero upward. Backward traversal reverses each row's
valid prefix separately: its first compact input uses [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) `(2,0)`, `(0,1)`, and `(1,2)`,
not a reversal of the complete padded time axis. Later backward hidden outputs are gathered back
to their original time and ascending active-row order. The three `packedOutputs()` Shapes are
`[3, 128]`, `[2, 128]`, and `[1, 128]`; each final axis contains the exact forward [features](../../glossary.md#term-feature) first
and aligned backward features second. `forwardFinalHidden()` is the forward state after each
row's last valid [element](../../glossary.md#term-element), while `backwardFinalHidden()` is the backward state after original time
zero. A zero-length row instead keeps its corresponding directional initial-state row.

RNN and GRU bidirectional results expose the merged compact hidden-output list plus separate
forward and backward final hidden states. The LSTM counterpart merges only hidden outputs and
returns four continuation values in forward-hidden, forward-cell, backward-hidden, backward-cell
order. Omitting explicit state creates distinct typed-zero leaves in that same direction/state
order. An all-zero request invokes neither cell, leaves automatic cells unbound, and returns the
exact explicit or derived initial references.

Every represented step calls the same Java cell and therefore reuses the same exact Parameter
leaf Tensor identities. Select, [gather](../../glossary.md#gather), gate/cell operations, and restored-state [producers](../../glossary.md#term-producer) are
fresh for each step, and later states retain temporal ancestry through earlier producers. This is
static expression [provenance](../../glossary.md#provenance): [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) can see repeated exact parameter identity [fan-out](../../glossary.md#term-fan-out),
and the existing compiler [reverse-mode](../../glossary.md#primal-adjoint-and-differentiation-seed) contract combines contributions for one identity-unique
[target](../../glossary.md#term-target). The sequence [API](../../glossary.md#term-application-programming-interface-api) itself neither defines numerical [gradients](../../glossary.md#term-gradient) nor exposes training or
execution.

This is static Tensor-expression construction, not numerical execution. For one direction,
`T = max(lengths)` produces `T` batched cell calls representing `S = sum(lengths)` compact logical
rows. A bidirectional container constructs `2T` batched calls representing `2S` logical rows,
plus bounded reverse-[alignment](../../glossary.md#term-alignment) expressions per represented time. These counts describe the
constructed provenance only. They do not prove [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), physical [kernel](../../glossary.md#kernel) skipping,
[fusion](../../glossary.md#term-fusion), kernel count, execution cost, or a public training workflow, and different Java length
values specialize different Tensor-expression [topology](../../glossary.md#term-topology).

The same three one-directional Sequences additionally expose `forwardScan`, which constructs one
fixed [scan](../../glossary.md#term-cumulative-scan) using that cell's current parameters. All [arguments](../../glossary.md#term-argument) are explicit:

```java
RecurrentScanResult r = rnnSequence.forwardScan(input, validLengths, initialHidden, direction);
RecurrentScanResult g = gruSequence.forwardScan(input, validLengths, initialHidden, direction);
LstmRecurrentScanResult l = lstmSequence.forwardScan(
        input, validLengths, initialHidden, initialCell, direction);
```

Here `validLengths` is a non-gradient `INT64[batch]` Tensor, not a Java array. Input is fully static
`[time,batch,inputSize]`; states are `[batch,hiddenSize]`. All floating inputs and cell parameters
must have the exact configured floating type, without promotion or broadcast. `direction` is
`RecurrentDirection.FORWARD` or `REVERSE`. Nulls are checked in signature order, then [descriptor](../../glossary.md#term-descriptor)
and prospective initialization-count checks precede draws, Tensor identifiers and [publication](../../glossary.md#term-publication).
Construction reads no length values or [host storage](../../glossary.md#host-storage) and creates no dummy cell expression.

The returned [Model](../../glossary.md#term-model) [carrier](../../glossary.md#term-storage-carrier) has dense `outputs()` shaped `[time,batch,hiddenSize]` and
`finalHidden()` shaped `[batch,hiddenSize]`; LSTM also has `finalCell()` of that state Shape.
Reverse traversal consumes only each row's valid prefix and stores outputs at original time
coordinates. For `[time,batch,hiddenSize] = [3,2,4]` and lengths `[3,1]`, row 1 has positive-zero
[padding](../../glossary.md#padding) at times 1 and 2 in either direction. A zero-length row preserves its initial state
values semantically, not the initial Java Tensor reference. These are not `packedOutputs()`.

A descriptor-valid first scan initializes an automatic cell even for empty time/batch or future
all-zero lengths, because the [producer](../../glossary.md#term-producer) still needs parameter inputs. It reuses the ordinary
cell monitor and [atomic](../../glossary.md#resource-transaction-and-rollback) group-publication gate. Pre-publication failure leaves reservations
retryable from the [seed](../../glossary.md#term-seed); later scan-construction failure retains published parameters and does
not roll back identifiers or allocations. Bound and strict-loaded cells never reinitialize.
Each scan reads current [bindings](../../glossary.md#term-binding) once in declaration order and rechecks their exact [schema](../../glossary.md#term-schema);
replacement affects later occurrences and remains caller-coordinated.

RNN additionally exposes `forwardGeneralScan` with the same explicit arguments/result carrier,
for configured [homogeneous](../../glossary.md#term-homogeneous) floating types, including both [OFP8](../../glossary.md#term-ofp8) types. It shares ordinary cell mathematics and the
existing initialization gate, but preflights [canonical](../../glossary.md#term-canonical-form) formals, emission and actual latent
histories before initialization/body IDs, including overflowing empty-domain [stride](../../glossary.md#term-stride) geometry.
Its dense emission is general port 1, final carry port 0; fixed ordering stays 0/1.
Both directions, bias modes and zero T/B are supported construction domains. Qualified [CPU](../../glossary.md#term-cpu)
first-order forward/backward and v5 [persistence](../../glossary.md#term-persistence) are accepted under NN0033; execution still
requires the actual general-Scan binder/read domain. No hidden state or downstream [loss](../../glossary.md#loss) [mask](../../glossary.md#term-boolean-mask) is added.
Low general bodies preserve ordinary low [operation](../../glossary.md#operation)-result [rounding](../../glossary.md#term-rounding), unlike fixed low RNN's
[FLOAT32](../../glossary.md#term-float32) working projections/aggregate/[tanh](../../glossary.md#term-tanh) and next-state [narrowing](../../glossary.md#term-narrowing). The builder keeps that
distinct fixed low program; it is not a type-support fallback. Low cold acquisition/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) and
persistence preserve logical carriers, with [FP8](../../glossary.md#term-fp8) selecting [graph](../../glossary.md#graph) v7 rather than the old v5 domain.
Actual [Metal](../../glossary.md#term-metal) training requires its complete first-order closure and admits no FP8.
The explicit FLOAT32/[FLOAT64](../../glossary.md#term-float64) [scalar objective](../../glossary.md#loss) policy remains unchanged.


One call returns the canonical ports of one producer regardless of time extent. Produced inputs,
states, lengths and parameter bindings retain their provenance without copies or detach. Both
train/eval modes preserve gradient eligibility, including [trainable parameters](../../glossary.md#trainable-parameter) in eval; this is
not permission to differentiate a fixed forwardScan. The module retains no caller state or result. Pass
LSTM `finalHidden()` and `finalCell()` explicitly to a later scan for symbolic carry; across
separate runs, supply materialized state values with the next graph's exact descriptors.

Model provides fixed RNN-tanh, reset-after GRU, and LSTM recurrent-scan expression
construction with an ordinary `INT64[batch]` `validLengths` Tensor. The advanced low-level static
`RecurrentScan.rnn`, `gru`, and `lstm` namespace has exactly six biased or bias-free methods with
explicit time-major input first. Each creates one flat multi-output producer with dense time-major
output and explicit final states. The namespace is not an NN layer or module and does not read
length values, statically unroll a cell, own parameters, or execute.
The Compiler supports the current forward family and selected first-order [AD](../../glossary.md#autograd) through its shared
cell/Scan construction; actual reverse bodies still require independent backend [admission](../../glossary.md#term-admission).
Higher-order recurrence remains unsupported. The CPU [scalar](../../glossary.md#term-scalar) fixed route qualifies homogeneous
static floating types, including all four low types, and resolved nonnegative read [layouts](../../glossary.md#layout), both directions/
bias modes, with trainable NN parameter flags. [Compile](../../glossary.md#compile) through public [Engine](../../glossary.md#term-engine), [prepare](../../glossary.md#prepare) once and runValues
with fresh full-descriptor values in `compiled.inputs()` order, including parameters and states.
Each [run](../../glossary.md#term-run) validates every length in `[0,time]` before writing scan results; invalid values may be
repaired and the preparation reused. Fixed expression validity remains distinct from execution
[qualification](../../glossary.md#term-qualification). Owned fixed-scan-containing Models compute and configure qualified first-order
training through those same [owners](../../glossary.md#term-owner); see [owned Model training](../training/sessions.md#owned-model-training-convenience)<span data-glossary-navigation="true"> ([owned Model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions))</span>.
Low fixed scans use FLOAT32 working projections and declared next-state narrowing through the
existing CPU [owner](../../glossary.md#term-owner); [optimizer](../../glossary.md#term-optimizer) masters do not change that recipe. General RNN composition keeps
its distinct ordinary operation-result rounding and working/publication recipe.

The additive scan method does not redirect any existing `forward` overload or bidirectional
container. Their Java `long[]` construction-time lengths, compact per-step outputs, provenance
and supported static training remain unchanged. No arbitrary mask with holes, stacked or
multidimensional recurrent [facade](../../glossary.md#term-facade), arbitrary direction collection, or configurable merge is added.
The current `ModuleFactory` constructs only the existing one-directional RNN, GRU, and LSTM Sequences; the
concrete bidirectional constructors already expose their two seeds and no factory recipe is
current. Existing builder `rnn/gru/lstm` steps remain statically [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) and retain supported training.
Explicit `rnnScan/gruScan/lstmScan` steps build two-input Models with shared [runtime](../../glossary.md#term-runtime) lengths,
FORWARD biased transitions and independent zero states. [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) RNN uses trainable [general Scan](../../glossary.md#general-differentiable-scan);
retained fixed families use their existing first-order AD when qualified. `finalHidden` selects the actual producer's semantic
port, not a universal index. See the [runtime-length Model workflow](../../user-guide/running-models.md#run-a-runtime-length-scan-model)
and [RNN training journey](../../user-guide/running-models.md#train-a-runtime-length-rnn).
No global [padding mask](../../glossary.md#masks-padding-and-eager-evaluation), hidden state carry or NN-to-Data [dependency](../../glossary.md#term-dependency) is introduced. Data exports
values/counts separately; bidirectional scan migration remains future. Initialized [Embedding](../../glossary.md#embedding) stays eager.


---

[Previous topic](declarative.md) · [Next topic](state.md)

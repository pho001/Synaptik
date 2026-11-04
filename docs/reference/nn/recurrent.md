# [Runtime](../../glossary.md#term-runtime)-length recurrent composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A **[valid length](../../glossary.md#term-valid-length)** says how many initial time steps belong to each sequence; the remaining
suffix is [padding](../../glossary.md#padding). **Runtime-length** means those counts arrive as [Tensor](../../glossary.md#tensor) values for execution,
rather than Java counts used to construct a different expression for each [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). A final hidden
state is the state after that row's last valid transition, not necessarily its last padded output.

<a id="nn-runtime-length-recurrent-composition"></a>
### [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [runtime](../../glossary.md#term-runtime)-length recurrent composition

`RnnSequence` and `GruSequence` add
`forwardScan(input, validLengths, initialHidden, direction)` returning [Model](../../glossary.md#term-model)'s
`RecurrentScanResult`. `LstmSequence` adds
`forwardScan(input, validLengths, initialHidden, initialCell, direction)` returning
`LstmRecurrentScanResult`. These are expression constructors using the same owned cell and
`cell.*` [parameter](../../glossary.md#term-parameter) paths, not execution methods or replacement static APIs.

Input is fully static `[T,B,I]`, lengths are non-[gradient](../../glossary.md#term-gradient) `INT64[B]`, states are `[B,H]`, and
all floating roles match the configured cell type exactly. `I/H` are positive; `T/B` may be zero.
Direction selects forward or reverse traversal of each valid prefix. The one [canonical](../../glossary.md#term-canonical-form) [producer](../../glossary.md#term-producer)
returns dense original-time-aligned `[T,B,H]` outputs, final hidden and, for [LSTM](../../glossary.md#term-long-short-term-memory-lstm), final cell.
[Padding](../../glossary.md#padding) is [positive zero](../../glossary.md#nan-infinity-and-signed-zero); zero-length rows preserve initial state values semantically. No length
values are read during construction, and no state is retained by the [module](../../glossary.md#term-module).

Complete [descriptor](../../glossary.md#term-descriptor) checks precede automatic initialization. Even empty scans bind the existing
reserved [parameter group](../../glossary.md#parameter-group--parametergroup) through the same cell-local gate; already-bound/strict-loaded cells do
not reinitialize. [Provenance](../../glossary.md#provenance) and gradient flags remain intact in train and eval. Callers coordinate
replacement; old occurrences keep old [bindings](../../glossary.md#term-binding). Existing Java `long[]` `forward` overloads,
compact result carriers, static training and builder recurrence remain unchanged.

Execute selected [scan](../../glossary.md#term-cumulative-scan) outputs separately through public [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/runValues with
fresh exact descriptors in compiled input order. Fixed [CPU](../../glossary.md#term-cpu) [scalar](../../glossary.md#term-scalar) forward supports its declared
[homogeneous](../../glossary.md#term-homogeneous) floating types, including all four low types, and resolved nonnegative read domain.
Selected first-order [AD](../../glossary.md#autograd) uses the shared [Compiler](../../glossary.md#term-compiler) closure and qualifies actual reverse bodies independently.
Owned Models can acquire fixed expressions through [`graph`](../../glossary.md#graph) or retained fixed builder recipes,
while unsupported occurrences still reject independently. No implicit state/default-length
overload or bidirectional migration is added. See the
[NN recurrent contract](recurrent-layers.md#current-nn-recurrent-composition-contract) and
[scan construction example](../../user-guide/running-models.md#build-a-fluent-model).
[BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [FP8](../../glossary.md#term-fp8) fixed implementations execute through the existing CPU [owner](../../glossary.md#term-owner),
with [FLOAT32](../../glossary.md#term-float32) working sites and declared state [narrowing](../../glossary.md#term-narrowing) under the same geometry/[derivative](../../glossary.md#term-derivative) guards.

`RnnSequence.forwardGeneralScan(input, validLengths, initialHidden, direction)` separately composes
ordinary [general Scan](../../glossary.md#general-differentiable-scan) for homogeneous configured floating types, including both FP8 types. It reuses the same cell
initialization and equation, including [bias](../../glossary.md#term-bias) order, with static T/B >= 0 and positive I/H.
Complete actual/formal/emission/latent geometry checks precede parameter [publication](../../glossary.md#term-publication) or body IDs.
Canonical general outputs are emission port 1 and final hidden [carry](../../glossary.md#term-carry) port 0; the result [carrier](../../glossary.md#term-storage-carrier)'s
component names are semantic, not universal producer indices. Fixed `forwardScan` is unchanged.
Qualified CPU reads use dense zero-offset geometry or existing proved scalar-broadcast exceptions,
not every fixed-scan read [layout](../../glossary.md#layout). No copy, conversion or fallback repairs unsupported actuals.
All four low types preserve ordinary low [rounding](../../glossary.md#term-rounding) at each projection, bias/ADD and [TANH](../../glossary.md#term-tanh) result.
That is a different represented program from fixed low [RNN](../../glossary.md#term-recurrent-neural-network-rnn)'s FLOAT32 working sites followed
by state narrowing; low builder `rnnScan` therefore deliberately keeps its fixed semantics.
Low source acquisition/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction)/save/load does not execute CPU low arithmetic. [Metal](../../glossary.md#term-metal) execution
and actual first-order closure remain [backend](../../glossary.md#backend)-qualified; the scalar Training [objective](../../glossary.md#term-objective) still
requires an explicit FLOAT32/[FLOAT64](../../glossary.md#term-float64) cast when derived from low outputs.
NN0033 qualifies first-order compute/train and v5 save/close/load/new training, with fresh lengths;
see the [complete RNN journey](../../user-guide/running-models.md#train-a-runtime-length-rnn).



---

[Previous topic](owned-model.md) · [Next topic](model-builder.md)

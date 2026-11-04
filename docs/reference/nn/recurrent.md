# Runtime-length recurrent composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A **valid length** says how many initial time steps belong to each sequence; the remaining
suffix is padding. **Runtime-length** means those counts arrive as Tensor values for execution,
rather than Java counts used to construct a different expression for each batch. A final hidden
state is the state after that row's last valid transition, not necessarily its last padded output.

<a id="nn-runtime-length-recurrent-composition"></a>
### NN runtime-length recurrent composition

`RnnSequence` and `GruSequence` add
`forwardScan(input, validLengths, initialHidden, direction)` returning Model's
`RecurrentScanResult`. `LstmSequence` adds
`forwardScan(input, validLengths, initialHidden, initialCell, direction)` returning
`LstmRecurrentScanResult`. These are expression constructors using the same owned cell and
`cell.*` parameter paths, not execution methods or replacement static APIs.

Input is fully static `[T,B,I]`, lengths are non-gradient `INT64[B]`, states are `[B,H]`, and
all floating roles match the configured cell type exactly. `I/H` are positive; `T/B` may be zero.
Direction selects forward or reverse traversal of each valid prefix. The one canonical producer
returns dense original-time-aligned `[T,B,H]` outputs, final hidden and, for LSTM, final cell.
Padding is positive zero; zero-length rows preserve initial state values semantically. No length
values are read during construction, and no state is retained by the module.

Complete descriptor checks precede automatic initialization. Even empty scans bind the existing
reserved parameter group through the same cell-local gate; already-bound/strict-loaded cells do
not reinitialize. Provenance and gradient flags remain intact in train and eval. Callers coordinate
replacement; old occurrences keep old bindings. Existing Java `long[]` `forward` overloads,
compact result carriers, static training and builder recurrence remain unchanged.

Execute selected scan outputs separately through public Engine compile/prepare/runValues with
fresh exact descriptors in compiled input order. Fixed CPU scalar forward supports its declared
homogeneous floating types, including all four low types, and resolved nonnegative read domain.
Selected first-order AD uses the shared Compiler closure and qualifies actual reverse bodies independently.
Owned Models can acquire fixed expressions through `graph` or retained fixed builder recipes,
while unsupported occurrences still reject independently. No implicit state/default-length
overload or bidirectional migration is added. See the
[NN recurrent contract](recurrent-layers.md#current-nn-recurrent-composition-contract) and
[scan construction example](../../user-guide/running-models.md#build-a-fluent-model).
BFLOAT16, FLOAT16 and both FP8 fixed implementations execute through the existing CPU owner,
with FLOAT32 working sites and declared state narrowing under the same geometry/derivative guards.

`RnnSequence.forwardGeneralScan(input, validLengths, initialHidden, direction)` separately composes
ordinary general Scan for homogeneous configured floating types, including both FP8 types. It reuses the same cell
initialization and equation, including bias order, with static T/B >= 0 and positive I/H.
Complete actual/formal/emission/latent geometry checks precede parameter publication or body IDs.
Canonical general outputs are emission port 1 and final hidden carry port 0; the result carrier's
component names are semantic, not universal producer indices. Fixed `forwardScan` is unchanged.
Qualified CPU reads use dense zero-offset geometry or existing proved scalar-broadcast exceptions,
not every fixed-scan read layout. No copy, conversion or fallback repairs unsupported actuals.
All four low types preserve ordinary low rounding at each projection, bias/ADD and TANH result.
That is a different represented program from fixed low RNN's FLOAT32 working sites followed
by state narrowing; low builder `rnnScan` therefore deliberately keeps its fixed semantics.
Low source acquisition/rebind/save/load does not execute CPU low arithmetic. Metal execution
and actual first-order closure remain backend-qualified; the scalar Training objective still
requires an explicit FLOAT32/FLOAT64 cast when derived from low outputs.
NN0033 qualifies first-order compute/train and v5 save/close/load/new training, with fresh lengths;
see the [complete RNN journey](../../user-guide/running-models.md#train-a-runtime-length-rnn).



---

[Previous topic](owned-model.md) · [Next topic](model-builder.md)

# Selective [graph](../../glossary.md#graph) format versions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

A format version identifies the rules for reading stored bytes. An **[opcode](../../glossary.md#term-opcode)** is the numeric
label for one stored [operation](../../glossary.md#operation), and a data-type ID identifies its [element](../../glossary.md#term-element) format. **[Serialization](../../glossary.md#term-serialization)**
encodes supported source descriptions and state into bytes; [deserialization](../../glossary.md#term-deserialization) rebuilds them. Format
support is separate from a [backend](../../glossary.md#backend)'s ability to execute the rebuilt computation or train it.

<a id="fp8-graph-persistence"></a>
## [FP8](../../glossary.md#term-fp8) [graph](../../glossary.md#graph) [persistence](../../glossary.md#term-persistence)

Sources containing `FLOAT8_E4M3` or `FLOAT8_E5M2` select `graph-model.v7`, with eight
[ASCII](../../glossary.md#term-ascii) [magic bytes](../../glossary.md#term-magic-bytes) `SYNGRF07` and [big-endian](../../glossary.md#text-encoding-and-file-boundaries) i32 version 7. This applies to every encoded
[descriptor](../../glossary.md#term-descriptor) and typed [scalar](../../glossary.md#term-scalar)/cast attribute, including contained [Scan](../../glossary.md#term-cumulative-scan) bodies, not just named
state. V7 appends [dtype](../../glossary.md#data-type--datatype) IDs 8=[E4M3](../../glossary.md#term-e4m3) and 9=[E5M2](../../glossary.md#term-e5m2) and stores exact one-byte raw [lanes](../../glossary.md#term-lane); it adds
no scale or quantization [metadata](../../glossary.md#term-metadata) and no new [operation](../../glossary.md#operation) inventory.

Old dtype IDs 1..7 remain unchanged. Non-FP8 sources retain byte-identical encoding and the
smallest sufficient v1-v6 selection described below; v1-v6 do not admit FP8 IDs. Readers require
matching kind, magic and version, and older readers reject v7. Copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction), [cold reconstruction](../../glossary.md#term-cold-reconstruction),
Scan mode and geometry constraints remain separate from [CPU](../../glossary.md#term-cpu) numerical capability. A graph file
stores source and logical values, not [FLOAT32](../../glossary.md#term-float32) [optimizer](../../glossary.md#term-optimizer) masters or Training continuation.

<a id="selective-graph-format-versions"></a>
## Selective [graph](../../glossary.md#graph) format versions

Within the original eight-kind [slice](../../glossary.md#slice), sources containing RELU, [TANH](../../glossary.md#term-tanh) or [SIGMOID](../../glossary.md#term-sigmoid) save as
`graph-model.v2` (`SYNGRF02`, i32 version 2).
Other sources in that slice, including zero-[node](../../glossary.md#node) graphs, retain byte-identical `graph-model.v1`
(`SYNGRF01`, i32 version 1). Version 2 appends opcodes 6, 7 and 8 respectively, each with one
preceding input ID and no attribute bytes; all other fields, payloads and bounds are unchanged.
For example, `x.mul(weight).relu()` selects v2, whereas `x.mul(weight)` still selects v1.

Load recognizes only matching kind/header/version tuples. V1 still rejects opcodes 6..8; v2
accepts only 1..8 and may contain only the original five kinds, in which case re-save selects v1.
Unknown or malformed data rejects before state import and reconstruction. An old graph reader
rejects v2 rather than reinterpreting it. ArtifactFile [framing](../../glossary.md#term-framing) and separate old [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) readers do not
change. No caller version selector or implicit [operation](../../glossary.md#operation) [decomposition](../../glossary.md#term-decomposition) is introduced.

<a id="selected-cnn-and-recurrent-persistence"></a>
### Selected [CNN](../../glossary.md#term-convolutional-neural-network-cnn) and recurrent [persistence](../../glossary.md#term-persistence)

The v3 extension adds only the exact selected **flat**
[Compiler source mappings](../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler))</span>: SUB,
RESHAPE, [axis](../../glossary.md#term-axis)-2 EXPAND_DIMS/SQUEEZE, axis-0 SELECT/[GATHER](../../glossary.md#gather)/CONCAT, [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) last-axis [SLICE](../../glossary.md#slice),
three-input biased default [CONV2D](../../glossary.md#convolution--conv2d)/[CONV3D](../../glossary.md#conv3d--ncdhw), builder MAX_POOL2D, and the fixed RNN_TANH,
GRU_RESET_AFTER and [LSTM](../../glossary.md#term-long-short-term-memory-lstm) scans. Attribute and [signature](../../glossary.md#term-signature) restrictions are deliberate; this is not
arbitrary [convolution](../../glossary.md#term-channels-first-convolution-layers) geometry, general recurrent bodies or all-family [serialization](../../glossary.md#term-serialization).

Any selected added body-free [operation](../../glossary.md#operation) uses `graph-model.v3`, even alongside RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid); old sources
keep their byte-identical smallest v1/v2 format. Readers use exact tuples, reject unknown
operations/attributes/cardinalities and retain the separate [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) readers. A v3 containing only older
kinds re-saves in the older sufficient version. The owning contract is the sole wire-field reference.

For CNNs and static recurrence, save retains ordinary expression [provenance](../../glossary.md#provenance) and reachable named
[constants](../../glossary.md#term-constant), not builder recipes or per-layer serializers. Raw unresolved intermediate [layouts](../../glossary.md#layout) stay
unresolved; [capture](../../glossary.md#term-capture-meanings)/[preflight](../../glossary.md#term-preflight) does not optimize them or force materializing copies. GATHER checks
index [descriptors](../../glossary.md#term-descriptor) only; execution retains index-value bounds checks.

For a fixed [scan](../../glossary.md#term-cumulative-scan), one constructor produces all [canonical](../../glossary.md#term-canonical-form) sibling wrappers, even if only finalCell
is a root or downstream operands reuse a port. Copy/load/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) retain that [occurrence](../../glossary.md#term-occurrence) once;
result-[carrier](../../glossary.md#term-storage-carrier) objects are not serialized. Static time/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero, and [inference](../../glossary.md#term-inference) preserves
the existing unresolved output layouts and [gradient](../../glossary.md#term-gradient) eligibility. Persistence itself grants no
execution or [BPTT](../../glossary.md#term-backpropagation-through-time-bptt). Separately, ordinary [CPU](../../glossary.md#term-cpu) [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run) supports static [homogeneous](../../glossary.md#term-homogeneous)
[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) scans with resolved nonnegative read layouts and canonical outputs
after logical layout closure, including [PARAMETER](../../glossary.md#term-parameter) [weights](../../glossary.md#term-weight) restored by this [API](../../glossary.md#term-application-programming-interface-api);
[INT64](../../glossary.md#data-type--datatype) length values are checked at execution, not during save/load/rebind.
[BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [OFP8](../../glossary.md#term-ofp8) CPU numerical execution reuse existing Scan [owners](../../glossary.md#term-owner) with FLOAT32
working values and final [narrowing](../../glossary.md#term-narrowing) under the same family/geometry and actual [AD](../../glossary.md#autograd) constraints.

For a cold scan roundtrip, declare a FLOAT32 sequence `[2,1,3]`,
non-gradient INT64 lengths `[1]` and FLOAT32 initial hidden `[1,4]` as ordered storage-free inputs;
register compatible [RNN](../../glossary.md#term-recurrent-neural-network-rnn) weights `[4,3]` and `[4,4]` as named state. Save a scan's finalHidden,
load, then rebind fresh inputs with those exact descriptors. The result is an ordinary symbolic
`[1,4]` [Tensor](../../glossary.md#tensor), not an evaluated value. Only named weights contribute [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bytes; [prototype](../../glossary.md#term-prototype)
lengths and initial-state bytes do not. Cold preflight never reads [runtime](../../glossary.md#term-runtime) lengths or checks their
value bounds `[0,2]`. Declaring lengths as frozen state instead would not provide runtime input
variation. Use NN `Model.graph`/`loadGraph` for the same list boundary; single `Model.load` still
rejects multiple boundaries. Do not call compute/train to validate this cold workflow.

<a id="approved-general-scan-persistence-implementation-pending"></a>

<a id="bounded-general-scan-persistence"></a>
### Bounded general-[scan](../../glossary.md#term-cumulative-scan) [persistence](../../glossary.md#term-persistence)

The [single owning v4 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
adds all-valid [general SCAN](../../glossary.md#general-differentiable-scan) to the outer v3 inventory and admits the pre-[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) subset of the
bounded cold body inventory (opcodes at most32), not arbitrary [PURE](../../glossary.md#term-pure) operations or only [RNN](../../glossary.md#term-recurrent-neural-network-rnn) algebra.
Existing `save(Path, Source,
ArtifactLimits)` and `load(Path, ArtifactLimits)` [signatures](../../glossary.md#term-signature) and failures remain; there is no
version selector or new public type. All-valid general-body sources without Pool3d select
`graph-model.v4`; body-free sources without Pool3d retain smallest sufficient exact v1/v2/v3.
Pool3d in either [scope](../../glossary.md#term-scope) selects v6; old v4/v5 reject opcodes34/35. Old readers reject v4.

For the three-step [FLOAT32](../../glossary.md#term-float32) `[2]` [carry](../../glossary.md#term-carry) / `[3,2]` sequence example above, save with the named `[2]`
[invariant](../../glossary.md#term-invariant) [weight](../../glossary.md#term-weight), [release](../../glossary.md#term-release) the caller-owned storage, then load and
[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) fresh exact-[descriptor](../../glossary.md#term-descriptor) inputs. The result is symbolic [provenance](../../glossary.md#provenance) with private raw weight
values, fresh independent body formals, tied body roles and distinct [canonical](../../glossary.md#term-canonical-form) outer ports.
No sequence [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), [residual](../../glossary.md#scan-residual-port-and-history-demand) history, [objective](../../glossary.md#term-objective), [optimizer state](../../glossary.md#term-optimizer-state) or prepared plan is restored.
Where the actual [occurrence](../../glossary.md#term-occurrence) qualifies, a caller can compute or configure fresh training after load.

V4 dry-sizes all inline [metadata](../../glossary.md#term-metadata) and all six limits before state [admission](../../glossary.md#term-admission)/copy, validates every
scope and residual [binding](../../glossary.md#term-binding) before [host](../../glossary.md#term-host) import/body construction, then delegates once to [Compiler](../../glossary.md#term-compiler)
and compares normalized full recapture before [publication](../../glossary.md#term-publication). Zero time does not bypass [validation](../../glossary.md#term-validation).
Malformed late records return no partial source or state change; publication retains ArtifactFile's
existing uncertain-move/cleanup behavior. Low/integral/[BOOL](../../glossary.md#data-type--datatype) cold metadata roundtrips do not qualify
execution. The [owned general-scan example](../../user-guide/running-models.md#train-an-all-valid-general-scan)
shows save/load through existing [Model](../../glossary.md#term-model) APIs. The [CP0013 packet](../../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
records focused evidence and main's final gate/capability acceptance.

<a id="length-aware-general-scan-persistence"></a>
### Length-aware general-[scan](../../glossary.md#term-cumulative-scan) [persistence](../../glossary.md#term-persistence)

[Checkpoint v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint))</span>
adds the static floating length-aware cold intersection without changing v4 records or [widening](../../glossary.md#term-widening)
its pre-[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) constructor inventory. Qualified mixed per-role floating [dtypes](../../glossary.md#data-type--datatype) and explicit casts
retain their exact [descriptors](../../glossary.md#term-descriptor). Non-[FP8](../../glossary.md#term-fp8) sources needing length mode without Pool3d select
`graph-model.v5`; Pool3d selects v6 and FP8 selects v7. An image can contain both all-valid and
length-aware occurrences. Sources needing none of these retain smallest sufficient exact v1-v4.

Save/load retain ordered ROW/SHARED roles, positional [aliases](../../glossary.md#aliasing-and-arena), unused formals, both traversal
directions and every [canonical](../../glossary.md#term-canonical-form) final/emission/[residual port](../../glossary.md#scan-residual-port-and-history-demand). Lengths is a separate outer actual,
not a body formal. External length values are not saved: supply fresh exact-descriptor control
inputs after load. Explicitly named length state follows ordinary named-state persistence.

Malformed flags, counts, [axes](../../glossary.md#term-axis), descriptors, references or late bodies reject before import or
[producer](../../glossary.md#term-producer) allocation. [Scalar](../../glossary.md#term-scalar) and checked actual/formal/output cardinalities are validated before
allocating the bounded role list; zero time/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) does not bypass [validation](../../glossary.md#term-validation). Existing limits
and [atomic](../../glossary.md#resource-transaction-and-rollback)-live-[publication](../../glossary.md#term-publication) behavior remain unchanged.

The [CP0014 Result](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
records independent old/new wire evidence and a public [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) [Model](../../glossary.md#term-model) numerical journey: fresh
lengths, repeated [SGD](../../glossary.md#term-stochastic-gradient-descent-sgd), save, close, loadGraph, fresh [inference](../../glossary.md#term-inference) and newly configured training.
The [codec](../../glossary.md#term-codec) does not migrate fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn) files or [resume](../../glossary.md#schema-serialization-and-checkpoint) [optimizer history](../../glossary.md#term-optimizer-history). The separate
[NN0033 Result](../../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
accepts the public recurrent-composition workflow; its completion does not widen the wire inventory.

<a id="complete-pool3d-geometry-in-v6"></a>
### Complete [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) geometry in v6

`graph-model.v6` uses eight [ASCII](../../glossary.md#term-ascii) [magic bytes](../../glossary.md#term-magic-bytes) `SYNGRF06` and i32 version 6. It appends
34=MAX_POOL3D and 35=AVERAGE_POOL3D, each with one input/output and exact typed [Model](../../glossary.md#term-model) attributes.
The attribute record contains twelve [big-endian](../../glossary.md#text-encoding-and-file-boundaries) i64 values: [kernel](../../glossary.md#kernel) depth/height/width, [stride](../../glossary.md#term-stride)
depth/height/width, symmetric [padding](../../glossary.md#padding) depth/height/width, [dilation](../../glossary.md#term-dilation) depth/height/width; then one
[canonical](../../glossary.md#term-canonical-form) u8 ceil flag (0=floor, 1=ceil). Invalid flags reject. Positive kernel/stride/dilation,
nonnegative padding and literal floor/ceil geometry retain Model guards, including nondefault
anisotropic geometry. [CONV2D](../../glossary.md#convolution--conv2d)/[CONV3D](../../glossary.md#conv3d--ncdhw) and MAX_POOL2D keep their previous narrow cold bounds.

An outer or contained-body Pool3d [occurrence](../../glossary.md#term-occurrence) selects v6, including otherwise-qualified all-valid
and length-aware bodies. No nested scans or other constructor [widening](../../glossary.md#term-widening) is implied. Without
Pool3d, save retains the byte-identical smallest sufficient v1–v5 format; old readers reject v6.
Existing section order, roles, [aliases](../../glossary.md#aliasing-and-arena), raw state bytes, bounds and import/[publication](../../glossary.md#term-publication) ordering
remain unchanged. No [optimizer](../../glossary.md#term-optimizer), executable or [Engine](../../glossary.md#term-engine)/session state is saved; load supports
ordinary fresh execution and newly configured training, not training continuation.




---

[Previous topic](graph.md) · [Next topic](graph-example.md)

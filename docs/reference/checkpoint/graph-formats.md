# Selective graph format versions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

A format version identifies the rules for reading stored bytes. An **opcode** is the numeric
label for one stored operation, and a data-type ID identifies its element format. **Serialization**
encodes supported source descriptions and state into bytes; deserialization rebuilds them. Format
support is separate from a backend's ability to execute the rebuilt computation or train it.

<a id="fp8-graph-persistence"></a>
## FP8 graph persistence

Sources containing `FLOAT8_E4M3` or `FLOAT8_E5M2` select `graph-model.v7`, with eight
ASCII magic bytes `SYNGRF07` and big-endian i32 version 7. This applies to every encoded
descriptor and typed scalar/cast attribute, including contained Scan bodies, not just named
state. V7 appends dtype IDs 8=E4M3 and 9=E5M2 and stores exact one-byte raw lanes; it adds
no scale or quantization metadata and no new operation inventory.

Old dtype IDs 1..7 remain unchanged. Non-FP8 sources retain byte-identical encoding and the
smallest sufficient v1-v6 selection described below; v1-v6 do not admit FP8 IDs. Readers require
matching kind, magic and version, and older readers reject v7. Copy/rebind, cold reconstruction,
Scan mode and geometry constraints remain separate from CPU numerical capability. A graph file
stores source and logical values, not FLOAT32 optimizer masters or Training continuation.

<a id="selective-graph-format-versions"></a>
## Selective graph format versions

Within the original eight-kind slice, sources containing RELU, TANH or SIGMOID save as
`graph-model.v2` (`SYNGRF02`, i32 version 2).
Other sources in that slice, including zero-node graphs, retain byte-identical `graph-model.v1`
(`SYNGRF01`, i32 version 1). Version 2 appends opcodes 6, 7 and 8 respectively, each with one
preceding input ID and no attribute bytes; all other fields, payloads and bounds are unchanged.
For example, `x.mul(weight).relu()` selects v2, whereas `x.mul(weight)` still selects v1.

Load recognizes only matching kind/header/version tuples. V1 still rejects opcodes 6..8; v2
accepts only 1..8 and may contain only the original five kinds, in which case re-save selects v1.
Unknown or malformed data rejects before state import and reconstruction. An old graph reader
rejects v2 rather than reinterpreting it. ArtifactFile framing and separate old NN readers do not
change. No caller version selector or implicit operation decomposition is introduced.

<a id="selected-cnn-and-recurrent-persistence"></a>
### Selected CNN and recurrent persistence

The v3 extension adds only the exact selected **flat**
[Compiler source mappings](../../architecture/contracts/compiler-autograd.md#compile-lifecycle): SUB,
RESHAPE, axis-2 EXPAND_DIMS/SQUEEZE, axis-0 SELECT/GATHER/CONCAT, rank-two last-axis SLICE,
three-input biased default CONV2D/CONV3D, builder MAX_POOL2D, and the fixed RNN_TANH,
GRU_RESET_AFTER and LSTM scans. Attribute and signature restrictions are deliberate; this is not
arbitrary convolution geometry, general recurrent bodies or all-family serialization.

Any selected added body-free operation uses `graph-model.v3`, even alongside RELU/TANH/SIGMOID; old sources
keep their byte-identical smallest v1/v2 format. Readers use exact tuples, reject unknown
operations/attributes/cardinalities and retain the separate NN readers. A v3 containing only older
kinds re-saves in the older sufficient version. The owning contract is the sole wire-field reference.

For CNNs and static recurrence, save retains ordinary expression provenance and reachable named
constants, not builder recipes or per-layer serializers. Raw unresolved intermediate layouts stay
unresolved; capture/preflight does not optimize them or force materializing copies. GATHER checks
index descriptors only; execution retains index-value bounds checks.

For a fixed scan, one constructor produces all canonical sibling wrappers, even if only finalCell
is a root or downstream operands reuse a port. Copy/load/rebind retain that occurrence once;
result-carrier objects are not serialized. Static time/batch may be zero, and inference preserves
the existing unresolved output layouts and gradient eligibility. Persistence itself grants no
execution or BPTT. Separately, ordinary CPU Engine compile/prepare/run supports static homogeneous
FLOAT32/FLOAT64 scans with resolved nonnegative read layouts and canonical outputs
after logical layout closure, including PARAMETER weights restored by this API;
INT64 length values are checked at execution, not during save/load/rebind.
BFLOAT16, FLOAT16 and both OFP8 CPU numerical execution reuse existing Scan owners with FLOAT32
working values and final narrowing under the same family/geometry and actual AD constraints.

For a cold scan roundtrip, declare a FLOAT32 sequence `[2,1,3]`,
non-gradient INT64 lengths `[1]` and FLOAT32 initial hidden `[1,4]` as ordered storage-free inputs;
register compatible RNN weights `[4,3]` and `[4,4]` as named state. Save a scan's finalHidden,
load, then rebind fresh inputs with those exact descriptors. The result is an ordinary symbolic
`[1,4]` Tensor, not an evaluated value. Only named weights contribute payload bytes; prototype
lengths and initial-state bytes do not. Cold preflight never reads runtime lengths or checks their
value bounds `[0,2]`. Declaring lengths as frozen state instead would not provide runtime input
variation. Use NN `Model.graph`/`loadGraph` for the same list boundary; single `Model.load` still
rejects multiple boundaries. Do not call compute/train to validate this cold workflow.

<a id="approved-general-scan-persistence-implementation-pending"></a>

<a id="bounded-general-scan-persistence"></a>
### Bounded general-scan persistence

The [single owning v4 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
adds all-valid general SCAN to the outer v3 inventory and admits the pre-Pool3d subset of the
bounded cold body inventory (opcodes at most32), not arbitrary PURE operations or only RNN algebra.
Existing `save(Path, Source,
ArtifactLimits)` and `load(Path, ArtifactLimits)` signatures and failures remain; there is no
version selector or new public type. All-valid general-body sources without Pool3d select
`graph-model.v4`; body-free sources without Pool3d retain smallest sufficient exact v1/v2/v3.
Pool3d in either scope selects v6; old v4/v5 reject opcodes34/35. Old readers reject v4.

For the three-step FLOAT32 `[2]` carry / `[3,2]` sequence example above, save with the named `[2]`
invariant weight, release the caller-owned storage, then load and
rebind fresh exact-descriptor inputs. The result is symbolic provenance with private raw weight
values, fresh independent body formals, tied body roles and distinct canonical outer ports.
No sequence batch, residual history, objective, optimizer state or prepared plan is restored.
Where the actual occurrence qualifies, a caller can compute or configure fresh training after load.

V4 dry-sizes all inline metadata and all six limits before state admission/copy, validates every
scope and residual binding before host import/body construction, then delegates once to Compiler
and compares normalized full recapture before publication. Zero time does not bypass validation.
Malformed late records return no partial source or state change; publication retains ArtifactFile's
existing uncertain-move/cleanup behavior. Low/integral/BOOL cold metadata roundtrips do not qualify
execution. The [owned general-scan example](../../user-guide/running-models.md#train-an-all-valid-general-scan)
shows save/load through existing Model APIs. The [CP0013 packet](../../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
records focused evidence and main's final gate/capability acceptance.

<a id="length-aware-general-scan-persistence"></a>
### Length-aware general-scan persistence

[Checkpoint v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
adds the static floating length-aware cold intersection without changing v4 records or widening
its pre-Pool3d constructor inventory. Qualified mixed per-role floating dtypes and explicit casts
retain their exact descriptors. Non-FP8 sources needing length mode without Pool3d select
`graph-model.v5`; Pool3d selects v6 and FP8 selects v7. An image can contain both all-valid and
length-aware occurrences. Sources needing none of these retain smallest sufficient exact v1-v4.

Save/load retain ordered ROW/SHARED roles, positional aliases, unused formals, both traversal
directions and every canonical final/emission/residual port. Lengths is a separate outer actual,
not a body formal. External length values are not saved: supply fresh exact-descriptor control
inputs after load. Explicitly named length state follows ordinary named-state persistence.

Malformed flags, counts, axes, descriptors, references or late bodies reject before import or
producer allocation. Scalar and checked actual/formal/output cardinalities are validated before
allocating the bounded role list; zero time/batch does not bypass validation. Existing limits
and atomic-live-publication behavior remain unchanged.

The [CP0014 Result](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
records independent old/new wire evidence and a public F32/F64 Model numerical journey: fresh
lengths, repeated SGD, save, close, loadGraph, fresh inference and newly configured training.
The codec does not migrate fixed RNN files or resume optimizer history. The separate
[NN0033 Result](../../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
accepts the public recurrent-composition workflow; its completion does not widen the wire inventory.

<a id="complete-pool3d-geometry-in-v6"></a>
### Complete Pool3d geometry in v6

`graph-model.v6` uses eight ASCII magic bytes `SYNGRF06` and i32 version 6. It appends
34=MAX_POOL3D and 35=AVERAGE_POOL3D, each with one input/output and exact typed Model attributes.
The attribute record contains twelve big-endian i64 values: kernel depth/height/width, stride
depth/height/width, symmetric padding depth/height/width, dilation depth/height/width; then one
canonical u8 ceil flag (0=floor, 1=ceil). Invalid flags reject. Positive kernel/stride/dilation,
nonnegative padding and literal floor/ceil geometry retain Model guards, including nondefault
anisotropic geometry. CONV2D/CONV3D and MAX_POOL2D keep their previous narrow cold bounds.

An outer or contained-body Pool3d occurrence selects v6, including otherwise-qualified all-valid
and length-aware bodies. No nested scans or other constructor widening is implied. Without
Pool3d, save retains the byte-identical smallest sufficient v1–v5 format; old readers reject v6.
Existing section order, roles, aliases, raw state bytes, bounds and import/publication ordering
remain unchanged. No optimizer, executable or Engine/session state is saved; load supports
ordinary fresh execution and newly configured training, not training continuation.




---

[Previous topic](graph.md) · [Next topic](graph-example.md)

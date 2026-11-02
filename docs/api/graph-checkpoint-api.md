# Generic graph checkpoint API

## Purpose and mental model

`GraphCheckpoint` copies or saves a bounded ordinary Tensor expression and named host values without
executing it. Copy/load return ordinary symbolic Tensors and private mutable heap state; rebind
substitutes inputs symbolically. None creates an Engine, prepared execution or new Model hierarchy.
Copy/rebind support bounded all-valid general-scan bodies and the static homogeneous
FLOAT32/FLOAT64 length-aware cold intersection. [V4 persistence](../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
retains the all-valid portion; [v5](../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
adds that static length-aware intersection. [V6](../architecture/contracts/extensions-training.md#canonical-graph-model-version-6)
adds maximum and average Pool3d in outer graphs and otherwise-qualified bodies. V1–v3 remain
body-free. Cold and wire admissions do not imply numerical execution support.
The [owning contract](../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
defines the capability, [original wire](../architecture/contracts/extensions-training.md#canonical-graph-model-version-1)
and [selective activation wire](../architecture/contracts/extensions-training.md#canonical-graph-model-version-2).
The [v3 contract](../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)
extends the same APIs to selected convolutional neural network (CNN), static recurrent and fixed
flat-scan sources. Persistence adds no execution capability, scan backpropagation through time
(BPTT), or optimizer resume. Existing default biased Conv3d sources can use
[qualified backend training](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
after load with a fresh training configuration. Pool3d similarly supports qualified CPU
FLOAT32/FLOAT64 and Metal FLOAT32 training; it does not restore optimizer history.

```text
borrowed Source -> copy -> private state + reconstructed ordinary provenance
borrowed Source -> save -> graph-model.v1/v2/v3/v4/v5/v6 -> load -> private heap state + ordinary provenance
Source + replacement inputs -> rebind -> ordinary outputs -> explicit caller Engine
```

The arrows distinguish cold acquisition/persistence/substitution from explicit execution.
Use only `:extensions:checkpoint` (physically `modules/checkpoint`); it exports Model, not Compiler,
NN or Engine. Old `nn.checkpoint.ModelCheckpoint` lives in NN and retains flexible declarative
`model.v1`/`whole-model.v1` readers, defaults and failures. Those files are not silently converted.

## Owned NN Model consumer

NN's `Model.defineTensor` and `Model.graph` use `copy` for one-time private acquisition, then
reuse that ordinary source for cached Engine execution, configured Training and save. `Model.load`
and `loadGraph` adopt `load`'s final heap leaves without another copy; constants stay outside the
Module state dictionary. Graph/load state paths must be in canonical Module traversal order.
Short Model save/load overloads use NN's six ceilings (256 MiB file/aggregate, 4 MiB manifest,
16384 entries, 64-byte keys, 64 MiB payload), not the old whole-model definition codec limits.
The generic facade remains cold and NN-free. See the
[owned workflow](../user-guide/running-models.md#owned-static-model-workflow).

NN's implemented all-valid general-scan workflow acquires graph/definition sources through this
bounded cold inventory, independently of CPU capability. Public `Model.graph` retains Source
list/null snapshots; definitions perform one prototype forward before acquisition. Qualified
static canonical homogeneous FLOAT32/FLOAT64 occurrences can compute/train through existing
Engine/Training owners. All-valid means every static time step participates, not arbitrary body
support. Fixed recurrent training rejection stays unchanged. All-valid general-body Model
save/load uses Checkpoint v4; static homogeneous length-aware bodies use v5, unless an outer or
body Pool3d occurrence selects v6. The same APIs add no NN wire, execution path or optimizer resume.

## Inputs, outputs and identity

`Source(inputs, outputs, leaves)` snapshots non-null lists in declaration order, borrowing Tensors.
Inputs are ordered unique provenance-free prototypes, possibly empty or unused. Their storage is
ignored and never saved. Outputs are nonempty and distinct. Every reachable non-input leaf must
have a `Leaf(name, role, tensor)` registration; unreachable named identities reject.

Roles are explicit: PARAMETER requires floating `requiresGrad=true`; BUFFER/CONSTANT retain their
declared gradient eligibility. Multiple names may alias one exact Tensor with the same role.
Equal values or physically overlapping distinct Tensors never establish aliases. Names are strict
UTF-8 nonblank dot segments, without normalization, duplicates or prefix collisions. Registration
order is retained independently of first source encounter and input signature order.

Checkpoint's body-free disk source admission is the closed static FORWARD **flat**
[Compiler slice](../architecture/contracts/compiler-autograd.md#compile-lifecycle): the original
eight kinds plus the selected CNN/static recurrent transforms and fixed scans described below.
Full descriptors, sharing and occurrences remain exact. All seven raw types persist where Model
permits them; outer SUB, NEG and activations require floating operands. V4 adds general SCAN with
the bounded body inventory below; v5 adds its static homogeneous FLOAT32/FLOAT64 length mode.
Body SUB also preserves Model's integral forms where admitted.
No optimization, layout closure
or implicit family extension occurs. Zero-node graphs are valid data, not a new Engine execution
promise. No backend rank-16 ceiling is imposed by persistence.

### General-scan in-memory copy and rebind

`copy` and `rebind` accept all-valid general SCAN and the bounded static homogeneous
FLOAT32/FLOAT64 length intersection through the exact
[Compiler cold constructor inventory](../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
A body is the scan's contained ordinary Tensor algebra, with distinct declared carry, step and
invariant formals. This bounded inventory is not all PURE Model operations and is not a backend
capability check: applicable low TANH bodies can be copied while Java low-type computation remains locked.
Body-only operations do not extend the flat outer-source inventory or old disk formats.

Length-mode cold support preserves ROW(0), rank-two ROW(1), SHARED sequences/invariants and
terminal SHARED accumulators. Complete declaration/interior and substitution-role preflight,
including late/dead/T=0 scans, precedes state copy or body formal/producer allocation. External
lengths prototypes remain unread; explicitly named control state follows ordinary state-copy
rules. Compatible repeated and opaque produced substitutions are supported; newly tied
incompatible outer row roles reject, including ties to retained named state.

Symbolic cold and newly Model-valid mixed/low/discrete length bodies remain unsupported.
V5 save independently rechecks the same length-mode cold eligibility through Compiler before
dry sizing, schema allocation or named storage admission; a raw association is not an admission
credential. Cold support alone does not qualify arbitrary numerical execution or training.
Pool3d adds only MAX_POOL3D and AVERAGE_POOL3D to the narrow length constructor whitelist;
static homogeneous FLOAT32/FLOAT64, ROW/SHARED, axis and terminal-SHARED restrictions remain.

Normalization changes only outer IDs. The immutable compiled-body attachment keeps its own local
IDs and ordered role bindings; equal numeric IDs in different scopes never identify the same value.
Compiler rebuilds fresh storage-free formals (including unused declarations), one body and one scan
per occurrence, preserving both traversal flags, repeated actual positions, role aliases and every
canonical final-carry, emission and residual port. A residual port describes a retained body value;
cold copying allocates no execution history. Reusing a template never merges occurrence scopes,
and reconstruction does not unroll time steps.

Complete outer/body preflight precedes state reads or reconstruction, including unsupported
unrequested emissions and zero-step bodies. Copy detaches named state once per identity and creates
storage-free input prototypes. Rebind reads no storage, retains named state and accepts produced or
repeated exact-descriptor substitutions without inspecting their upstream provenance.

For example, take a FLOAT32 scan with carry `[2]`, sequence `[3,2]` and a named invariant weight
`[2]`, whose body computes `tanh(carry + step * weight)` and emits that same result. Copy preserves
the tied next-carry/emission body value but retains distinct canonical outer ports and private
weight storage. Rebinding a fresh sequence and carry yields ordinary symbolic outputs using that
private weight, not evaluated values. The same source with time zero still validates its entire
body; changing an emission to unsupported ABS rejects before state access. Saving either valid
general-body source selects v4: v1/v2/v3 contain no body opcode or attachment field.
Use an explicitly composed Engine only where its whole occurrence qualifies;
this cold API itself grants no execution support. NN's separately implemented owned workflow is
described above uses the same cold persistence boundary.

Complete decoded metadata passes `GraphCompilationPort.validateSource(CompiledGraphModel)`
before any candidate Tensor is created, for all supported versions. This metadata-only Compiler
seam checks every stored node/body without Tensor identities, host reads/copies or tensor-storage
allocation. Normal leaf association and reconstruction checks still follow.

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

### Selected CNN and recurrent persistence

The v3 extension adds only the exact selected **flat**
[Compiler source mappings](../architecture/contracts/compiler-autograd.md#compile-lifecycle): SUB,
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
BF16/FP16 source, storage and cold reconstruction remain retained, but their CPU numerical
execution is temporarily locked; persistence does not bypass that backend boundary.

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

### Bounded general-scan persistence

The [single owning v4 contract](../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
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
execution. The [owned general-scan example](../user-guide/running-models.md#train-an-all-valid-general-scan)
shows save/load through existing Model APIs. The [CP0013 packet](../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
records focused evidence and main's final gate/capability acceptance.

### Length-aware general-scan persistence

[Checkpoint v5](../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
adds the static homogeneous FLOAT32/FLOAT64 length-aware cold intersection without changing v4
records or widening its pre-Pool3d constructor inventory. A source needing length mode without
Pool3d selects `graph-model.v5`; an image can contain both all-valid and length-aware occurrences.
Sources needing neither length mode nor Pool3d keep byte-identical smallest sufficient v1-v4.

Save/load retain ordered ROW/SHARED roles, positional aliases, unused formals, both traversal
directions and every canonical final/emission/residual port. Lengths is a separate outer actual,
not a body formal. External length values are not saved: supply fresh exact-descriptor control
inputs after load. Explicitly named length state follows ordinary named-state persistence.

Malformed flags, counts, axes, descriptors, references or late bodies reject before import or
producer allocation. Scalar and checked actual/formal/output cardinalities are validated before
allocating the bounded role list; zero time/batch does not bypass validation. Existing limits
and atomic-live-publication behavior remain unchanged.

The [CP0014 Result](../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result)
records independent old/new wire evidence and a public F32/F64 Model numerical journey: fresh
lengths, repeated SGD, save, close, loadGraph, fresh inference and newly configured training.
The codec does not migrate fixed RNN files or resume optimizer history. The separate
[NN0033 Result](../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md#result)
accepts the public recurrent-composition workflow; its completion does not widen the wire inventory.

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



## Complete cold example

This Java 26 method needs only Checkpoint and its exported Model dependency. The caller supplies
a file path in an existing trusted directory. Its FLOAT32 scalar weight is `2`, and the symbolic
output is `input * weight`. It copies state, saves before any execution, loads, then substitutes a
fresh scalar `3`. The returned Tensor represents `3 * 2`; it has not been evaluated.

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

Tensor roundtrip(Path file) throws IOException {
    var descriptor = new TensorDescriptor(DataType.FLOAT32, Shape.of(), Optional.empty(), true);
    Tensor input = TensorFactory.create(descriptor);
    Tensor weight = TensorFactory.scalar(2.0f, Optional.empty(), true);
    var source = new GraphCheckpoint.Source(List.of(input), List.of(input.mul(weight)), List.of(
            new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight)));
    var acquired = GraphCheckpoint.copy(source);
    var limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
    GraphCheckpoint.save(file, acquired, limits);
    var loaded = GraphCheckpoint.load(file, limits);
    // Match full prototype metadata, including unresolved layout and requiresGrad.
    Tensor fresh = TensorFactory.create(descriptor);
    fresh.replaceHostStorage(TensorFactory.scalar(3.0f, Optional.empty(), true)
            .hostStorage().orElseThrow());
    return GraphCheckpoint.rebind(loaded, List.of(fresh)).getFirst();
}
```

Copy creates storage-free prototypes and copies each state identity once, then rebuilds expressions
against the copies; the original weight/storage remain unchanged and are not retained. Load already
owns fresh ordinary heap state, so it needs no second ownership copy or caller arena. The stored
input is a descriptor, not the scalar `3` or any previous batch. Rebind does not inspect state
storage. Produced and repeated replacements keep their upstream provenance; repeated output
references remain possible and ordinary compilation still rejects duplicate outputs.

To evaluate, explicitly use the existing Engine API with its own dependency and lifecycle. The
integration tests cover Linear MATMUL/PERMUTE/ADD and each activation with two fresh inputs and
ordinary forward/autograd equivalence. Owned MLP training additionally compares updates against
an independent Tensor/Engine oracle, not just another restored Model.
Replacing the fresh scalar with a vector, resolved-layout scalar or no-gradient scalar rejects
because the entire input descriptor must match. No automatic recapture/recompile is promised.

## Storage, bounds and failures

Named leaves must be host-readable, live and current-thread-accessible, even for empty values.
Present leaf layouts must exactly equal zero-offset non-view contiguous geometry; absent layouts
retain their absence and use checked canonical geometry. Intermediate PERMUTE views are metadata.
Read-only heap/native sources are accepted. Logical lanes are native-order, unaligned integer-width
accesses; signed zero and NaN payload/sign bits remain exact. BOOL bytes must be 0/1. Excess
capacity is ignored and distinct overlapping Tensors are copied independently. Labels/native
owners, batches, compiled plans, device state and Training state are not persisted.

Callers exclude mutation, storage rebinding and closure for each whole operation. Copy/save order
is metadata, checked geometry and save bounds, all host admission, BOOL validation/raw copying,
complete result/image, then file publication. Save uses the unchanged force/revalidate/single
atomic-move policy, not authentication or portable power-loss durability. A failed move can have
provider-dependent target state; inspect the target. No partial source result is returned.

Save/load require explicit `ArtifactLimits`, with no defaults. All six limits and individual
payload ceilings apply before encoding or bulk copies. For schema bytes D, unique state bytes B
and identity count S: manifest M=72+58*S, aggregate D+B, file 48+M+D+B, entries 1+S; kind/key
ceiling must be at least 14. These are encoded bounds, not peak-heap promises. Copy/rebind have
no file ceiling; copy still preflights each payload against `Integer.MAX_VALUE - 8`.

Nulls fail in declaration/list order. Invalid source, unsupported geometry, BOOL or bounds raise
`IllegalArgumentException`; checked sizing overflow raises `ArithmeticException`, retaining existing
Compiler wrapping. Absent/dead/inaccessible state raises `IllegalStateException`. Rebind checks
source metadata before replacement count/full descriptors, without a value gate. Load validates
complete framing/wire/keys/descriptors/aliases/lengths/BOOL, then descriptor-only Compiler source
preflight, host import, reconstruction and recapture before publication. Stored semantic/overflow
failures become `ArtifactFormatException` with causes; I/O stays `IOException`; Tensor identifier
exhaustion and VM failures propagate.

## Related API

- [Opaque artifact file API and publication](checkpoint-artifact-api.md)
- [Retained NN model checkpoint API](model-checkpoint-api.md)
- [Generic graph checkpoint source glossary](../glossary.md#generic-graph-checkpoint-source)
- [Accepted ownership and migration decision](../design/decisions/0039-portable-graph-model-persistence.md)

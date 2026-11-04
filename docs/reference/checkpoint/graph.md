<a id="generic-graph-checkpoint-api"></a>
# Graph checkpoint sources, copy and rebind

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

A graph source records operations, their dependencies and declared inputs; named state supplies
the persistent values those operations use. **Cold reconstruction** rebuilds ordinary Tensor
expressions from that description without evaluating them. Copying detaches named state into
private storage, whereas rebinding changes which input expressions are used without reading storage.

<a id="purpose-and-mental-model"></a>
## Purpose and mental model

`GraphCheckpoint` copies or saves a bounded ordinary Tensor expression and named host values without
executing it. Copy/load return ordinary symbolic Tensors and private mutable heap state; rebind
substitutes inputs symbolically. None creates an Engine, prepared execution or new Model hierarchy.
Copy/rebind support bounded all-valid general-scan bodies and the static floating
length-aware cold intersection. [V4 persistence](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
retains the all-valid portion; [v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
adds that static length-aware intersection. [V6](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-6)
adds maximum and average Pool3d in outer graphs and otherwise-qualified bodies. V1–v3 remain
body-free. Cold and wire admissions do not imply numerical execution support.
The [owning contract](../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
defines the capability, [original wire](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1)
and [selective activation wire](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-2).
The [v3 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)
extends the same APIs to selected convolutional neural network (CNN), static recurrent and fixed
flat-scan sources. Persistence adds no execution capability, scan backpropagation through time
(BPTT), or optimizer resume. Existing default biased Conv3d sources can use
[qualified backend training](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
after load with a fresh training configuration. Pool3d similarly supports qualified CPU
FLOAT32/FLOAT64 and Metal FLOAT32 training; it does not restore optimizer history.

```text
borrowed Source -> copy -> private state + reconstructed ordinary provenance
borrowed Source -> save -> graph-model.v1..v7 -> load -> private heap state + ordinary provenance
Source + replacement inputs -> rebind -> ordinary outputs -> explicit caller Engine
```

The arrows distinguish cold acquisition/persistence/substitution from explicit execution.
Use only `:extensions:checkpoint` (physically `modules/checkpoint`); it exports Model, not Compiler,
NN or Engine. Old `nn.checkpoint.ModelCheckpoint` lives in NN and retains flexible declarative
`model.v1`/`whole-model.v1` readers, defaults and failures. Those files are not silently converted.

<a id="owned-nn-model-consumer"></a>
## Owned NN Model consumer

NN's `Model.defineTensor` and `Model.graph` use `copy` for one-time private acquisition, then
reuse that ordinary source for cached Engine execution, configured Training and save. `Model.load`
and `loadGraph` adopt `load`'s final heap leaves without another copy; constants stay outside the
Module state dictionary. Graph/load state paths must be in canonical Module traversal order.
Short Model save/load overloads use NN's six ceilings (256 MiB file/aggregate, 4 MiB manifest,
16384 entries, 64-byte keys, 64 MiB payload), not the old whole-model definition codec limits.
The generic facade remains cold and NN-free. See the
[owned workflow](../../user-guide/running-models.md#owned-static-model-workflow).

NN's implemented all-valid general-scan workflow acquires graph/definition sources through this
bounded cold inventory, independently of CPU capability. Public `Model.graph` retains Source
list/null snapshots; definitions perform one prototype forward before acquisition. Qualified
static canonical floating occurrences, including the four low types, can compute/train through
existing Engine/Training and shared Compiler AD owners under their actual family/geometry guards.
All-valid means every static time step participates, not arbitrary body support. Fixed recurrence
uses its own qualified shared derivative closure, not an inference-only blanket rejection.
Non-FP8 general-body sources select v4, length-aware sources v5 and Pool3d sources v6 as needed;
FP8 anywhere in the encoded source selects v7. These APIs add no execution path or optimizer resume.

<a id="inputs-outputs-and-identity"></a>
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
[Compiler slice](../../architecture/contracts/compiler-autograd.md#compile-lifecycle): the original
eight kinds plus the selected CNN/static recurrent transforms and fixed scans described below.
Full descriptors, sharing and occurrences remain exact. All nine raw types persist where Model
permits them, with FP8 requiring v7; outer SUB, NEG and activations require floating operands.
V4 adds general SCAN with the bounded body inventory below; v5 adds its static length mode.
Body SUB also preserves Model's integral forms where admitted.
No optimization, layout closure
or implicit family extension occurs. Zero-node graphs are valid data, not a new Engine execution
promise. No backend rank-16 ceiling is imposed by persistence.

<a id="general-scan-in-memory-copy-and-rebind"></a>
### General-scan in-memory copy and rebind

`copy` and `rebind` accept all-valid general SCAN and the bounded static floating
length intersection, including all six floating types and qualified mixed per-role dtypes, through the exact
[Compiler cold constructor inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
A body is the scan's contained ordinary Tensor algebra, with distinct declared carry, step and
invariant formals. This bounded inventory is not all PURE Model operations and is not a backend
capability check: copying a low TANH body does not qualify a backend route.
Body-only operations do not extend the flat outer-source inventory or old disk formats.

Length-mode cold support preserves ROW(0), rank-two ROW(1), SHARED sequences/invariants and
terminal SHARED accumulators. Complete declaration/interior and substitution-role preflight,
including late/dead/T=0 scans, precedes state copy or body formal/producer allocation. External
lengths prototypes remain unread; explicitly named control state follows ordinary state-copy
rules. Compatible repeated and opaque produced substitutions are supported; newly tied
incompatible outer row roles reject, including ties to retained named state.

Symbolic cold remains unsupported. Floating actuals, formals and body intermediates are checked
per role, including low types and explicit floating casts; this is not arbitrary discrete-body
admission. Save independently rechecks the same length-mode cold eligibility through Compiler before
dry sizing, schema allocation or named storage admission; a raw association is not an admission
credential. Cold support alone does not qualify arbitrary numerical execution or training.
Pool3d adds only MAX_POOL3D and AVERAGE_POOL3D to the narrow length constructor whitelist;
static geometry, ROW/SHARED, operation/rank, axis and terminal-SHARED restrictions remain.

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


---

[Next topic](graph-formats.md)

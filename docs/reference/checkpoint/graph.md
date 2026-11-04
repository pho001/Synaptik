<a id="generic-graph-checkpoint-api"></a>
# [Graph](../../glossary.md#graph) [checkpoint](../../glossary.md#term-checkpoint) sources, copy and [rebind](../../glossary.md#capture-graph-ports-and-reconstruction)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

A graph source records operations, their dependencies and declared inputs; named state supplies
the persistent values those operations use. **[Cold reconstruction](../../glossary.md#term-cold-reconstruction)** rebuilds ordinary [Tensor](../../glossary.md#tensor)
expressions from that description without evaluating them. Copying detaches named state into
private storage, whereas rebinding changes which input expressions are used without reading storage.

## The copy, rebind and load callers have different needs

The application or NN Model passes `GraphCheckpoint.Source` a description of output expressions,
ordered input prototypes and named stored state. The Source snapshots lists but borrows the
Tensors. `copy` reads admitted named-state bytes and creates private copies; `rebind` instead
builds expressions using supplied replacement references without reading their bytes. `save`
writes the supported description and named state, not the previous input batches. `load` creates
fresh ordinary heap state and reconstructed expressions, with no Engine or native Arena to close.
A **leaf** means no producing expression: only the named-state role requires admitted stored
values, while input prototypes may be storage-free. Compare [the complete cold example](graph-example.md)
with [Training recovery](../training/recovery.md) before treating a loaded graph as optimizer continuation.

<a id="purpose-and-mental-model"></a>
## Purpose and [mental model](../../glossary.md#term-mental-model)

`GraphCheckpoint` copies or saves a bounded ordinary [Tensor](../../glossary.md#tensor) expression and named [host](../../glossary.md#term-host) values without
executing it. Copy/load return ordinary symbolic Tensors and private [mutable](../../glossary.md#term-mutable) [heap](../../glossary.md#term-heap) state; [rebind](../../glossary.md#capture-graph-ports-and-reconstruction)
substitutes inputs symbolically. None creates an [Engine](../../glossary.md#term-engine), [prepared execution](../../glossary.md#prepared-execution--preparedexecution) or new [Model](../../glossary.md#term-model) hierarchy.
Copy/rebind support bounded all-valid general-[scan](../../glossary.md#term-cumulative-scan) bodies and the static floating
length-aware cold intersection. [V4 persistence](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)
retains the all-valid portion; [v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)
adds that static length-aware intersection. [V6](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-6)
adds maximum and average [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) in outer [graphs](../../glossary.md#graph) and otherwise-qualified bodies. V1–v3 remain
body-free. Cold and wire admissions do not imply numerical execution support.
The [owning contract](../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
defines the capability, [original wire](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-1)
and [selective activation wire](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-2).
The [v3 contract](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)
extends the same APIs to selected [convolutional neural network](../../glossary.md#term-convolutional-neural-network-cnn) ([CNN](../../glossary.md#neural-network-families)), static recurrent and fixed
flat-scan sources. [Persistence](../../glossary.md#term-persistence) adds no execution capability, scan [backpropagation through time](../../glossary.md#term-backpropagation-through-time-bptt)
([BPTT](../../glossary.md#backpropagation-through-time-and-recurrent-state)), or [optimizer](../../glossary.md#term-optimizer) [resume](../../glossary.md#schema-serialization-and-checkpoint). Existing default biased [Conv3d](../../glossary.md#conv3d--ncdhw) sources can use
[qualified backend training](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
after load with a fresh training configuration. Pool3d similarly supports qualified [CPU](../../glossary.md#term-cpu)
[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) and [Metal](../../glossary.md#term-metal) FLOAT32 training; it does not restore [optimizer history](../../glossary.md#term-optimizer-history).

```text
borrowed Source -> copy -> private state + reconstructed ordinary provenance
borrowed Source -> save -> graph-model.v1..v7 -> load -> private heap state + ordinary provenance
Source + replacement inputs -> rebind -> ordinary outputs -> explicit caller Engine
```

The arrows distinguish cold acquisition/persistence/substitution from explicit execution.
Use only `:extensions:checkpoint` (physically `modules/checkpoint`); it exports Model, not [Compiler](../../glossary.md#term-compiler),
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or Engine. Old `nn.checkpoint.ModelCheckpoint` lives in NN and retains flexible declarative
`model.v1`/`whole-model.v1` readers, defaults and failures. Those files are not silently converted.

<a id="owned-nn-model-consumer"></a>
## Owned [NN Model](../../glossary.md#model-module-buffer-and-scan-distinctions) consumer

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s `Model.defineTensor` and `Model.graph` use `copy` for one-time private acquisition, then
reuse that ordinary source for cached [Engine](../../glossary.md#term-engine) execution, configured Training and save. `Model.load`
and `loadGraph` adopt `load`'s final [heap](../../glossary.md#term-heap) leaves without another copy; [constants](../../glossary.md#term-constant) stay outside the
[Module](../../glossary.md#term-module) state dictionary. [Graph](../../glossary.md#graph)/load state paths must be in [canonical](../../glossary.md#term-canonical-form) Module traversal order.
Short [Model](../../glossary.md#term-model) save/load overloads use NN's six ceilings (256 MiB file/aggregate, 4 MiB [manifest](../../glossary.md#term-manifest),
16384 entries, 64-byte keys, 64 MiB [payload](../../glossary.md#artifact-envelope-and-atomic-publication)), not the old whole-model definition [codec](../../glossary.md#term-codec) limits.
The generic [facade](../../glossary.md#term-facade) remains cold and NN-free. See the
[owned workflow](../../user-guide/running-models.md#owned-static-model-workflow).

NN's implemented all-valid general-[scan](../../glossary.md#term-cumulative-scan) workflow acquires graph/definition sources through this
bounded cold inventory, independently of [CPU](../../glossary.md#term-cpu) capability. Public `Model.graph` retains Source
list/null snapshots; definitions perform one [prototype](../../glossary.md#term-prototype) forward before acquisition. Qualified
static canonical floating occurrences, including the four low types, can compute/train through
existing Engine/Training and shared [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) [owners](../../glossary.md#term-owner) under their actual family/geometry guards.
All-valid means every static time step participates, not arbitrary body support. Fixed recurrence
uses its own qualified shared [derivative](../../glossary.md#term-derivative) closure, not an [inference](../../glossary.md#term-inference)-only blanket rejection.
Non-[FP8](../../glossary.md#term-fp8) general-body sources select v4, length-aware sources v5 and [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) sources v6 as needed;
FP8 anywhere in the encoded source selects v7. These APIs add no execution path or [optimizer](../../glossary.md#term-optimizer) [resume](../../glossary.md#schema-serialization-and-checkpoint).

<a id="inputs-outputs-and-identity"></a>
## Inputs, outputs and [identity](../../glossary.md#term-identity-meanings)

`Source(inputs, outputs, leaves)` snapshots non-null lists in declaration order, [borrowing](../../glossary.md#term-borrowing) [Tensors](../../glossary.md#tensor).
Inputs are ordered unique [provenance](../../glossary.md#provenance)-free [prototypes](../../glossary.md#term-prototype), possibly empty or unused. Their storage is
ignored and never saved. Outputs are nonempty and distinct. Every reachable non-input [leaf](../../glossary.md#term-leaf) must
have a `Leaf(name, role, tensor)` registration; unreachable named identities reject.

Roles are explicit: [PARAMETER](../../glossary.md#term-parameter) requires floating `requiresGrad=true`; [BUFFER](../../glossary.md#term-buffer)/[CONSTANT](../../glossary.md#term-constant) retain their
declared [gradient](../../glossary.md#term-gradient) eligibility. Multiple names may [alias](../../glossary.md#aliasing-and-arena) one exact Tensor with the same role.
Equal values or physically overlapping distinct Tensors never establish aliases. Names are strict
[UTF-8](../../glossary.md#term-utf-8) nonblank dot segments, without [normalization](../../glossary.md#term-normalization), duplicates or prefix collisions. Registration
order is retained independently of first source encounter and input [signature](../../glossary.md#term-signature) order.

[Checkpoint](../../glossary.md#term-checkpoint)'s body-free disk source [admission](../../glossary.md#term-admission) is the closed static FORWARD **flat**
[Compiler slice](../../architecture/contracts/compiler-autograd.md#compile-lifecycle): the original
eight kinds plus the selected [CNN](../../glossary.md#term-convolutional-neural-network-cnn)/static recurrent transforms and fixed scans described below.
Full [descriptors](../../glossary.md#term-descriptor), sharing and occurrences remain exact. All nine raw types persist where [Model](../../glossary.md#term-model)
permits them, with [FP8](../../glossary.md#term-fp8) requiring v7; outer SUB, NEG and [activations](../../glossary.md#term-activation) require floating operands.
V4 adds [general SCAN](../../glossary.md#general-differentiable-scan) with the bounded body inventory below; v5 adds its static length mode.
Body SUB also preserves Model's integral forms where admitted.
No optimization, [layout](../../glossary.md#layout) closure
or implicit family extension occurs. Zero-[node](../../glossary.md#node) [graphs](../../glossary.md#graph) are valid data, not a new [Engine](../../glossary.md#term-engine) execution
promise. No [backend](../../glossary.md#backend) [rank](../../glossary.md#term-rank)-16 ceiling is imposed by [persistence](../../glossary.md#term-persistence).

<a id="general-scan-in-memory-copy-and-rebind"></a>
### General-[scan](../../glossary.md#term-cumulative-scan) in-memory copy and [rebind](../../glossary.md#capture-graph-ports-and-reconstruction)

`copy` and `rebind` accept all-valid [general SCAN](../../glossary.md#general-differentiable-scan) and the bounded static floating
length intersection, including all six floating types and qualified mixed per-role [dtypes](../../glossary.md#data-type--datatype), through the exact
[Compiler cold constructor inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
A body is the scan's contained ordinary [Tensor](../../glossary.md#tensor) algebra, with distinct declared [carry](../../glossary.md#term-carry), step and
[invariant](../../glossary.md#term-invariant) formals. This bounded inventory is not all [PURE](../../glossary.md#term-pure) [Model](../../glossary.md#term-model) operations and is not a [backend
capability](../../glossary.md#backend-capability) check: copying a low [TANH](../../glossary.md#term-tanh) body does not qualify a [backend route](../../glossary.md#backend-route).
Body-only operations do not extend the flat outer-source inventory or old disk formats.

Length-mode cold support preserves ROW(0), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1), SHARED sequences/invariants and
terminal SHARED accumulators. Complete declaration/interior and substitution-role [preflight](../../glossary.md#term-preflight),
including late/dead/T=0 scans, precedes state copy or body formal/[producer](../../glossary.md#term-producer) allocation. External
lengths [prototypes](../../glossary.md#term-prototype) remain unread; explicitly named control state follows ordinary state-copy
rules. Compatible repeated and [opaque](../../glossary.md#term-opaque) produced substitutions are supported; newly tied
incompatible outer row roles reject, including ties to retained named state.

Symbolic cold remains unsupported. Floating actuals, formals and body intermediates are checked
per role, including low types and explicit floating casts; this is not arbitrary discrete-body
[admission](../../glossary.md#term-admission). Save independently rechecks the same length-mode cold eligibility through [Compiler](../../glossary.md#term-compiler) before
[dry sizing](../../glossary.md#term-dry-sizing), [schema](../../glossary.md#term-schema) allocation or named storage admission; a raw association is not an admission
credential. Cold support alone does not qualify arbitrary numerical execution or training.
[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) adds only MAX_POOL3D and AVERAGE_POOL3D to the narrow length constructor whitelist;
static geometry, ROW/SHARED, [operation](../../glossary.md#operation)/[rank](../../glossary.md#term-rank), [axis](../../glossary.md#term-axis) and terminal-SHARED restrictions remain.

[Normalization](../../glossary.md#term-normalization) changes only outer IDs. The [immutable](../../glossary.md#term-immutable) compiled-body attachment keeps its own local
IDs and ordered role [bindings](../../glossary.md#term-binding); equal numeric IDs in different scopes never identify the same value.
Compiler rebuilds fresh storage-free formals (including unused declarations), one body and one scan
per [occurrence](../../glossary.md#term-occurrence), preserving both traversal flags, repeated actual positions, role [aliases](../../glossary.md#aliasing-and-arena) and every
[canonical](../../glossary.md#term-canonical-form) final-carry, emission and [residual port](../../glossary.md#scan-residual-port-and-history-demand). A residual port describes a retained body value;
cold copying allocates no execution history. Reusing a template never merges occurrence scopes,
and reconstruction does not unroll time steps.

Complete outer/body preflight precedes state reads or reconstruction, including unsupported
unrequested emissions and zero-step bodies. Copy detaches named state once per [identity](../../glossary.md#term-identity-meanings) and creates
storage-free input prototypes. Rebind reads no storage, retains named state and accepts produced or
repeated exact-[descriptor](../../glossary.md#term-descriptor) substitutions without inspecting their upstream [provenance](../../glossary.md#provenance).

For example, take a [FLOAT32](../../glossary.md#term-float32) scan with carry `[2]`, sequence `[3,2]` and a named invariant [weight](../../glossary.md#term-weight)
`[2]`, whose body computes `tanh(carry + step * weight)` and emits that same result. Copy preserves
the tied next-carry/emission body value but retains distinct canonical outer ports and private
weight storage. Rebinding a fresh sequence and carry yields ordinary symbolic outputs using that
private weight, not evaluated values. The same source with time zero still validates its entire
body; changing an emission to unsupported ABS rejects before state access. Saving either valid
general-body source selects v4: v1/v2/v3 contain no body [opcode](../../glossary.md#term-opcode) or attachment field.
Use an explicitly composed [Engine](../../glossary.md#term-engine) only where its whole occurrence qualifies;
this cold [API](../../glossary.md#term-application-programming-interface-api) itself grants no execution support. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s separately implemented owned workflow is
described above uses the same cold [persistence](../../glossary.md#term-persistence) boundary.

Complete decoded [metadata](../../glossary.md#term-metadata) passes `GraphCompilationPort.validateSource(CompiledGraphModel)`
before any candidate Tensor is created, for all supported versions. This metadata-only Compiler
seam checks every stored [node](../../glossary.md#node)/body without Tensor identities, [host](../../glossary.md#term-host) reads/copies or tensor-storage
allocation. Normal [leaf](../../glossary.md#term-leaf) association and reconstruction checks still follow.


---

[Next topic](graph-formats.md)

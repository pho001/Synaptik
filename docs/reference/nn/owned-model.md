# Owned [tensor](../../glossary.md#tensor) [Model](../../glossary.md#term-model) execution

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

**Owned** means this Model acquires private named state and coordinates the lifetimes needed
for its high-level execution methods. It does not mean every generic Model supports those methods.
A **cold forward** constructs output expressions without running them; compute evaluates through
[Engine](../../glossary.md#term-engine). [Graph](../../glossary.md#graph) load restores supported source and values, not a prepared plan or [optimizer history](../../glossary.md#term-optimizer-history).

## Who owns the inputs, state and results?

The application calls `Model.builder().build()`, `defineTensor(...)` or `graph(...)` to acquire
an owned Model. Supplied named parameter Tensors are **leaves** when they have no producing
expression; being a leaf alone does not mean storage is present. Acquisition requires supported
stored state and copies it into private Model storage, once per exact Tensor identity. The
application still owns the original state and its storage lifetime; later changes to it do not
update the acquired Model. This differs from ordinary [Module binding replacement](state.md#current-nn-parameter-update-contract),
which installs a reference without copying its bytes.

For `compute`, the application supplies current stored input values, not another network recipe.
The Model calls Engine using its retained source and cached preparation, borrowing each batch
only for the synchronous call. It closes the temporary result before returning independent host
bytes. Close the Model to release its preparations and private resources. A selected external
Engine is borrowed and must outlive its use; the Model closes only an Engine it opened itself.
Use [graph persistence](../checkpoint/graph.md#owned-nn-model-consumer) to recover source and
current state, or [Training recovery](../training/recovery.md) when the next optimizer step matters.

<a id="owned-tensor-model-capability"></a>
### Owned [tensor](../../glossary.md#tensor) [Model](../../glossary.md#term-model) capability

`Model<I,O>` remains generic and now implements `AutoCloseable`. Its fixed tensor methods are
independent of I/O: compute returns [`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue) or `List<HostTensorValue>`, and train returns
`TrainingStep`, never generic O. The fluent builder's nonempty-chain `build()`,
`defineTensor(signature, definition)` and `load(path[, limits])`
return `Model<Tensor,Tensor>`; builder `buildWithLengths()`, `graph(inputs, outputs, namedLeaves)` and `loadGraph(path[, limits])`
return `Model<List<Tensor>,List<Tensor>>`. Definition/[graph](../../glossary.md#graph)/load factories have explicit [borrowed](../../glossary.md#ownership-borrowing-and-lease)-[Engine](../../glossary.md#term-engine)
overloads (load forms take path, limits, Engine); built Models select it through `withEngine`.
Old generic factories and flexible v1 readers remain separate.

#### Select the Engine before execution setup

`withEngine(engine)` returns the same receiver and must precede the first valid compute/configure
setup attempt, including failed setup. Without a selection, execution lazily opens a model-owned
[CPU](../../glossary.md#term-cpu)-only standard Engine. Construction/load/save/cold forward open none. List compute additionally
accepts `maximumTotalBytes`; all publications are checked before [materialization](../../glossary.md#materialization), and [RunResult](../../glossary.md#run-result--runresult)
closes before [detached](../../glossary.md#detached-model-results) values return. Configuration takes a [target](../../glossary.md#term-target) [descriptor](../../glossary.md#term-descriptor) and [scalar objective](../../glossary.md#loss)
builder, or target [prototypes](../../glossary.md#term-prototype) and a list builder, plus an [Optimizer](../../glossary.md#term-optimizer). It succeeds once; train delegates
`RESET_AND_STEP`. Every declared input/target requires the exact static descriptor even if unused.

#### Bind the declared input positions

`compute(firstInput, secondInput)` returns the sole detached output of any [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) with exactly
two declared inputs and one output, including ordinary graphs and loaded Models. It binds positions,
not inferred [dtype](../../glossary.md#data-type--datatype) or length roles, and uses the same cached execution as list compute with
`Long.MAX_VALUE`. Capability/[lifecycle](../../glossary.md#lifecycle) [admission](../../glossary.md#term-admission) precedes [arity](../../glossary.md#term-arity), then [arguments](../../glossary.md#term-argument). Cold `forward`
is optional, never a prerequisite. Fixed and general recurrent sources use the existing configured
[TrainingSession](../../glossary.md#training-session--trainingsession) and [Compiler](../../glossary.md#term-compiler) first-order [AD](../../glossary.md#autograd) when their actual forward/reverse bodies qualify.
Mixed fixed/general sources share one configured session. Objectives still require a [scalar](../../glossary.md#term-scalar),
[gradient](../../glossary.md#term-gradient)-eligible [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) result; low-output objectives explicitly cast to FLOAT32/FLOAT64.
CPU low execution retains the existing family/geometry and actual [derivative](../../glossary.md#term-derivative) guards and setup/lifecycle ordering.

`Model.graph` and `defineTensor` also acquire bounded all-valid general-[scan](../../glossary.md#general-differentiable-scan) bodies through [Checkpoint](../../glossary.md#term-checkpoint) and
Compiler's [cold constructor inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
This does not add a ModelBuilder recipe. Cold admission preserves applicable Model dtypes, including
low [TANH](../../glossary.md#term-tanh) bodies; it is not arbitrary [PURE](../../glossary.md#term-pure)-body support. Qualified static floating occurrences,
including [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [FP8](../../glossary.md#term-fp8) types, use the same compute/first-order Training [owners](../../glossary.md#term-owner)
when their actual forward/reverse bodies and binders qualify.
All-valid means every static time step participates, with no lengths or [padding mask](../../glossary.md#masks-padding-and-eager-evaluation).
Acquire private state once, configure once, and pass fresh exact-descriptor [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters) to the same
prepared/session owners. Connected nonempty [parameter](../../glossary.md#term-parameter) admission still applies at T=0; neither
fixed nor general recurrence bypasses the selected [backend](../../glossary.md#backend)'s actual derivative [qualification](../../glossary.md#term-qualification).
Non-FP8 bounded all-valid general-body save/load uses
[v4](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4);
the static qualified per-role floating length intersection uses
[v5](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5),
with [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) selecting v6 as needed. FP8 anywhere in the encoded source selects v7 through the
same Model/GraphCheckpoint owners. Old-only sources retain their smallest sufficient exact
format and bytes. V1/v2/v3 remain body-free. Unsupported saves reject
before file effects without freezing Engine selection or disabling compute/train. Load restores
source/current raw values, not [objective](../../glossary.md#term-objective)/[optimizer history](../../glossary.md#term-optimizer-history) or execution plans.
See the
[complete general-scan workflow](../../user-guide/running-models.md#train-an-all-valid-general-scan).

Model's separate [length-aware Scan source API](../tensor/general-scan.md#length-aware-source-only-construction)
constructs bounded source [metadata](../../glossary.md#term-metadata) and structural [IR](../../glossary.md#term-intermediate-representation-ir) only. Its final [INT64](../../glossary.md#data-type--datatype) lengths actual has no
body formal. Compiler independently supports length-mode [inference](../../glossary.md#term-inference-meanings), bounded first-order reverse
construction and final role/demand queries; original-source AD requires every [carry](../../glossary.md#term-carry) and every
original sequence to be ROW(0), including unused/non-gradient positions. Generated reverse inputs
may retain explicit mixed roles. Prepare0011 transports those exact [bindings](../../glossary.md#term-binding) and demand through
the existing staged resource [handoff](../../glossary.md#term-handoff) after static boundary [validation](../../glossary.md#term-validation); it reads no length values.
Static length-aware owned Models are implemented: bounded cold acquisition and reconstruction
feed the same cached compute and configured Training owners, and accepted sources save/load
through v5 (or v6 for Pool3d, v7 for FP8). `ModelBuilder.buildWithLengths()` builds the two-input
values/lengths workflow. This does not grant symbolic owned-Model acquisition or [persistence](../../glossary.md#term-persistence),
arbitrary PURE-body copying, or a new [backend route](../../glossary.md#backend-route). The [runtime-length workflow](recurrent.md)
describes the independently qualified static composition.

#### Handle unsupported sources and close

Unsupported generic/custom/declarative tensor methods throw `UnsupportedOperationException` before
file/Engine work; their close is a no-op. Owned operations reject busy before inspecting arguments.
Source mutation invalidates the capability without recapture; close waits and attempts all [owner](../../glossary.md#term-owner)
cleanup with retained failure replay, never closing a borrowed Engine. See the complete
[owned model example](../../user-guide/running-models.md#owned-static-model-workflow) and
[owning contract](../../architecture/contracts/extensions-training.md#owned-tensor-models).

Graph-model v3 uses these same methods and no new public type. The
[exact source slice](../../architecture/contracts/compiler-autograd.md#compile-lifecycle) admits
selected narrow transforms/indexing, biased default convolutions, builder [pooling](../../glossary.md#term-pooling) and fixed
RNN_TANH/GRU_RESET_AFTER/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) scans. It is not all-family [serialization](../../glossary.md#term-serialization). Advanced `graph`/`loadGraph`
may declare sequence, non-gradient INT64 [runtime](../../glossary.md#term-runtime) lengths and initial states as ordered external
inputs for cold persistence/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction). Prototypes are descriptor-only; their bytes are neither read
nor saved. A scan remains one multi-output [producer](../../glossary.md#term-producer) with [canonical](../../glossary.md#term-canonical-form) sibling [identities](../../glossary.md#term-identity-meanings), even when
only a final-state port is a root. Cold persistence executes nothing. Fixed-scan first-order
[backpropagation through time](../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../glossary.md#backpropagation-through-time-and-recurrent-state)) is implemented by Compiler's
[RecurrentGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/RecurrentGradientRules.java),
which builds one transition recipe and uses the shared length-aware Scan derivative owner,
not time [unrolling](../../glossary.md#term-unrolling) or a separate cell derivative engine. Fixed and mixed stacks can configure
training, consume fresh batches, save, reload, and configure a fresh session when their actual
forward/reverse occurrences qualify. Persistence restores neither optimizer history nor an
execution plan. Existing default biased [Conv3d](../../glossary.md#conv3d--ncdhw) can train after reload in its
[qualified backend domain](../../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)
with a fresh training configuration. Single `load` still rejects multiple boundaries.


---

[Previous topic](../engine/lifecycle.md) · [Next topic](recurrent.md)

# Extensions and training contract

> This scoped contract is incorporated by the [authoritative architecture root](../../../ARCHITECTURE.md).
> It is normative only within the scope stated below and has no independent authority. Root global
> invariants and dependency rules apply everywhere and take precedence. Any overlap, contradiction,
> missing applicable scope, or ambiguity requires an explicit architecture update; do not choose
> between contracts silently.<span data-glossary-navigation="true"> ([normative glossary](../../glossary.md#term-normative); [scope glossary](../../glossary.md#term-scope); [dependency glossary](../../glossary.md#term-dependency))</span>

## [Scope](../../glossary.md#term-scope)

This contract owns Data [host](../../glossary.md#term-host) preparation, Text [tokenization](../../glossary.md#frozen-whitespace-tokenizer)/adaptation, [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training, generic [graph](../../glossary.md#graph)/value [Checkpoint](../../glossary.md#term-checkpoint), the optional
future Training Checkpoint adapter, and [ONNX](../../glossary.md#term-onnx) responsibilities plus the [optimizer](../../glossary.md#term-optimizer)/training [lifecycle](../../glossary.md#lifecycle).
It does not own [Model](../../glossary.md#term-model) semantics, [Compiler](../../glossary.md#term-compiler) [autograd](../../glossary.md#autograd), [concrete backend](../../glossary.md#concrete-backend) optimizer routes, or [Engine](../../glossary.md#term-engine)
[materialization](../../glossary.md#materialization).

## `extensions/data`

Data owns [host](../../glossary.md#term-host)-side numeric [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters)/sequence preparation outside [NN model](../../glossary.md#model-module-buffer-and-scan-distinctions) computation. Its
authorized foundation is [Gradle](../../glossary.md#term-gradle) `:extensions:data` at physical `extensions/data`, with Java
namespace `io.github.pho001.synaptik.data` and exactly `api(project(":modules:model"))` as its
sole project [dependency](../../glossary.md#term-dependency). [Model](../../glossary.md#term-model) is exported because [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [signatures](../../glossary.md#term-signature) expose Model [Tensor](../../glossary.md#tensor) values;
this export alone does not authorize additional Data capabilities.

The accepted foundation added no production type, marker, placeholder, speculative [API](../../glossary.md#term-application-programming-interface-api) or reserved
subpackage. The selected capability permits `ValidSequenceLengths`, `SequenceAxisOrder`,
`PaddingSide`, `PaddedSequenceBatch` and `LongSequenceBatcher` directly in
`io.github.pho001.synaptik.data`, with nested construction Builders. Package documentation
requires a real package contract. No other project dependency is permitted, including test-only
edges. Inward [modules](../../glossary.md#term-module) and existing extensions must not depend on Data except Text's explicit
API edge below. Vision requires a separate coordinated architecture decision.

`ValidSequenceLengths` is final with a private constructor and `of(long... lengths)`, `batchSize()`,
`length(int row)` and `toLongArray()`. It owns one copied non-negative `long` valid count per original
batch row, with ordered [structural equality](../../glossary.md#term-structural-equality)/[hash](../../glossary.md#term-hash) and [metadata](../../glossary.md#term-metadata)-only diagnostics; it stores no time
[extent](../../glossary.md#term-extent). `of` rejects null with `NullPointerException("lengths")`, clones once and checks that copy
in row order, rejecting negatives with `IllegalArgumentException` identifying the row. Callers
coordinate source mutation during construction. `batchSize()` returns `int`; invalid row indices
throw `IndexOutOfBoundsException`, and every array export is fresh, including an empty export.
`SequenceAxisOrder` has exactly `TIME_MAJOR` and `BATCH_MAJOR`, with no custom methods; these are
logical sequence [axes](../../glossary.md#term-axis), not Model physical [layout](../../glossary.md#layout).

Counts are position independent: `PaddingSide` has exactly `LEFT` and `RIGHT`, with no custom
methods. At extent T and count L, RIGHT declares valid positions `[0,L)` and LEFT `[T-L,T)`.
There are no holes, value-sentinel [inference](../../glossary.md#term-inference), stored [padding](../../glossary.md#padding) lengths or stored dense [mask](../../glossary.md#term-boolean-mask).

`PaddedSequenceBatch(Tensor values, ValidSequenceLengths validLengths, SequenceAxisOrder axisOrder,
PaddingSide paddingSide)` is a record retaining the exact four references. This is an explicit
source migration from the current unpublished `RightPaddedSequenceBatch`, not a compatibility family.
Its public [canonical](../../glossary.md#term-canonical-form) constructor is the single
geometry-[validation](../../glossary.md#term-validation) algorithm: null-check `values`, `validLengths`, then `axisOrder`, with
`NullPointerException` naming the [parameter](../../glossary.md#term-parameter); then null-check `paddingSide` likewise before [rank](../../glossary.md#term-rank),
numeric leading axes, batch count
and length upper bounds in row order. Rank must be at least two; `TIME_MAJOR` is `[T,B,...]` and
`BATCH_MAJOR` is `[B,T,...]`. Both leading axes must be `StaticDimension`; B must equal widened
`batchSize()` and every length must be <= T from [Shape](../../glossary.md#shape). Geometry failures throw
`IllegalArgumentException` identifying the mismatch or offending row and T. Never narrow extents
to `int` or calculate [element](../../glossary.md#term-element) products/byte spans: B above `Integer.MAX_VALUE` cannot match the
host array, while T and lengths may reach `Long.MAX_VALUE`. B=0 accepts empty lengths at any T;
T=0 requires all lengths zero. [Opaque](../../glossary.md#term-opaque) trailing Model [dimensions](../../glossary.md#dimension) may be absent, zero or dynamic;
zero [feature](../../glossary.md#term-feature) extents do not force zero lengths. No additional [dtype](../../glossary.md#data-type--datatype) (including [BOOL](../../glossary.md#data-type--datatype)), [gradient](../../glossary.md#term-gradient),
[provenance](../../glossary.md#provenance), [leaf](../../glossary.md#term-leaf), physical-layout or storage restriction is permitted.

The record also exposes `static Builder builder(Tensor values)` and one `public static final`
nested `Builder` with a private constructor. The factory borrows a non-null values reference,
fixed for that builder, rejecting null with `NullPointerException("values")`; creation reads no
Shape. Builder exposes exactly `timeMajor()`, `batchMajor()`, `validLengths(long... lengths)` and
`leftPadded()`, `rightPadded()` and `build()`. There are no default axis order, side or lengths:
all three selections are required, and explicit
empty varargs is a valid selection for B=0, distinct from missing lengths. Setters return the same
Builder; the last successful selection wins, including side/axis changes and length replacement.
`validLengths` immediately calls `ValidSequenceLengths.of` before assignment, copying and checking
negativity once; a failed setter preserves prior state. `build()` throws `IllegalStateException`
for missing selections before geometry checks, otherwise delegates to the canonical constructor
without recopying [immutable](../../glossary.md#term-immutable) lengths. Each successful build returns a fresh batch; later setters
or failures cannot change earlier batches. The reusable Builder is caller-coordinated and not
[thread-safe](../../glossary.md#term-thread-safe). No overloads, generic axis setter, staged interface, [facade](../../glossary.md#term-facade) or pipeline are permitted.
The direct record constructor and standalone lengths remain available to advanced callers.

Only immutable [descriptor](../../glossary.md#term-descriptor)/Shape metadata is inspected. Construction never calls `hostStorage()`,
[captures](../../glossary.md#term-capture-meanings) storage, checks [liveness](../../glossary.md#term-liveness), obtains a segment, [leases](../../glossary.md#term-lease) a [scope](../../glossary.md#term-scope), attaches/replaces/clears
storage, closes resources, scans/copies Tensor contents, transposes, materializes or executes.
All Model-valid layouts and original row order survive unchanged. Immutability covers lengths and
fixed references, not Tensor contents or its [mutable](../../glossary.md#term-mutable) [borrowed](../../glossary.md#ownership-borrowing-and-lease)-storage association. No readiness,
liveness or thread-safety guarantee is added. Equality uses Tensor object [identity](../../glossary.md#term-reference-identity), structural
lengths and both enum [identities](../../glossary.md#term-identity-meanings), never Tensor contents. Length creation and batch validation are [O(B)](../../glossary.md#memory-management-atomic-counters-and-complexity),
independent of T and element count; no Tensor/expression is created. For example, declared lengths
`[3,2,1]` at T=4 derive padding lengths `[1,2,3]`; zeros remain ordinary data and arbitrary values
outside the declared valid interval do not affect validation. Caller-supplied values must
already have the declared order and padding side; metadata checks do not prove contents.

`LongSequenceBatcher` is final with private construction, static `builder()`, a public static final
nested `Builder`, and `batch(List<long[]> sequences)` returning `PaddedSequenceBatch`. Builder
exposes only `paddingValue(long)`, `leftPadded()`, `rightPadded()`, `timeMajor()`, `batchMajor()` and
`build()`. Padding value, side and axis order all require explicit selection; missing choices throw
`IllegalStateException`. Setters return the same builder, last selection wins, and each build
returns a fresh immutable configuration unaffected by later builder use. Builders are not thread-safe.

Batching preserves list/element order, uses B=list size, T=maximum row length, stores the exact
row counts, and pads to T on the selected side with the configured long value. No [truncation](../../glossary.md#term-truncation-toward-zero) or
limit-policy API is introduced. Empty outer input has B=0,T=0; all-empty rows have T=0. Null list
or row throws `NullPointerException` naming `sequences` or the row. Callers coordinate list/row
mutation throughout the [synchronous](../../glossary.md#term-synchronous) call; no transactional [snapshot](../../glossary.md#term-snapshot) guarantee is made.
[Preflight](../../glossary.md#term-preflight) uses checked long arithmetic for element count B*T and byte span (B*T)*Long.BYTES,
rejecting [overflow](../../glossary.md#term-overflow) or element count above Integer.MAX_VALUE with `IllegalArgumentException`
before output allocation. There is no Integer.MAX_VALUE byte cap. Incrementally check
B*currentMaxLength while reading rows, before any B-sized lengths/reference snapshot: an enormous
virtual list with a first nonempty row must fail as soon as that row proves oversized geometry,
without scanning the remaining rows or allocating batch-sized metadata. No public bounds helper.
Allocate one destination through public `TensorFactory.allocate` and fill its writable public Model
[host storage](../../glossary.md#host-storage) directly, without a full flat staging array and import recopy. Output is a fresh
[heap](../../glossary.md#term-heap)-backed, [contiguous](../../glossary.md#term-contiguous) offset-zero [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [INT64](../../glossary.md#data-type--datatype) non-gradient leaf, `[T,B]` or `[B,T]`, retaining
no caller list/arrays. Zero-size outputs use ordinary empty heap storage. Factory failures propagate;
Data adds no native allocation, [Arena](../../glossary.md#term-arena), close protocol, storage [owner](../../glossary.md#term-owner) or execution [lifecycle](../../glossary.md#lifecycle).

Host metadata is not a [runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) input. The implemented additive export on the existing record is
exactly `public Tensor toScanLengthsTensor()`. It checks
`paddingSide == RIGHT` first, then `axisOrder == TIME_MAJOR`, throwing `IllegalArgumentException`
for LEFT or BATCH_MAJOR before Tensor creation, output allocation or storage access. It does not
convert either representation or widen [scan](../../glossary.md#term-cumulative-scan) semantics.

Each successful call allocates through public `TensorFactory.allocate` and fills writable public
Model host storage directly from immutable valid counts in original row order. The result is a
fresh non-null ordinary heap-backed, provenance-free INT64 `[B]` leaf, canonical contiguous,
offset-zero, non-[view](../../glossary.md#term-view-meanings) and `requiresGrad=false`; B=0 returns fresh ordinary empty heap storage.
O(B) allocation and copying of counts are explicit. No mutable export is cached: mutating one
result cannot change metadata or another export. Model factory failures propagate.

This opt-in method never reads values storage or checks its liveness, copies/reorders/repackages
sequence contents, or changes values descriptors, gradient flags or storage associations. All
constructor geometry and accessor contracts above remain unchanged, including unrestricted
Model-valid dtype/layout/provenance, opaque trailing dimensions and zero extents. Export therefore
also works with storage-free values; it does not certify numeric execution support or exact Model
input compatibility. The application retains ordinary Tensors for synchronous computation and
performs existing public [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Engine](../../glossary.md#term-engine) [binding](../../glossary.md#term-binding) itself. No graph input protocol, native allocation,
[device](../../glossary.md#term-device) transfer, close protocol, [resource owner](../../glossary.md#ownership-borrowing-and-lease) or execution owner is added. First integration is
canonical [FLOAT32](../../glossary.md#term-float32) RIGHT/TIME_MAJOR numeric scan inference, not Text IDs/[Embedding](../../glossary.md#embedding), LEFT or
BATCH_MAJOR scans, or [BPTT](../../glossary.md#term-backpropagation-through-time-bptt).

Existing Model storage/Tensor [ownership](../../glossary.md#term-ownership-meanings), NN static
length APIs, runtime recurrent-scan semantics, Training and Engine execution remain unchanged.
Data owns no execution, [optimizer](../../glossary.md#term-optimizer), [persistence](../../glossary.md#term-persistence), file loading, [dataset](../../glossary.md#term-dataset)/sampler/[epoch](../../glossary.md#term-epoch-meanings) lifecycle,
[tokenization](../../glossary.md#frozen-whitespace-tokenizer), image semantics, device transfer, or new Tensor/storage [lifetime](../../glossary.md#term-lifetime).

## `extensions/text`

Text owns frozen whitespace [tokenization](../../glossary.md#frozen-whitespace-tokenizer) and adaptation to Data [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), at physical
`extensions/text`, [Gradle](../../glossary.md#term-gradle) `:extensions:text`, flat namespace `io.github.pho001.synaptik.text`.
Its sole project [dependency](../../glossary.md#term-dependency) in all configurations is `api(project(":extensions:data"))`; [Model](../../glossary.md#term-model)
comes transitively through Data. Only public Data/Model APIs may be used; Text has no [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Engine](../../glossary.md#term-engine),
Training, [Checkpoint](../../glossary.md#term-checkpoint), [Compiler](../../glossary.md#term-compiler), [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), tuning or [backend](../../glossary.md#backend) edge/import. NN has no Data/Text
edge. External integration tests compose Text, unchanged [Embedding](../../glossary.md#embedding) and public Engine.

The final concrete `WhitespaceTokenizer` has private construction and static
`of(List<String> ordinaryTokens)`. It copies ordered ordinary [tokens](../../glossary.md#term-token), assigns IDs from 2 in list
order, reserves [PAD](../../glossary.md#frozen-whitespace-tokenizer)=0 and [UNK](../../glossary.md#frozen-whitespace-tokenizer)=1, and never grows. Public methods are `int vocabularySize()`
(including both reserved IDs), `long paddingId()`, `long unknownId()` and `long[] encode(String text)`.
Null list, token or text throws `NullPointerException` naming the [parameter](../../glossary.md#term-parameter)/row; empty ordinary
tokens, tokens containing a Character.isWhitespace(int) code point, and exact duplicates throw
`IllegalArgumentException`. Empty vocabulary is valid. Callers coordinate list mutation during of.
No spelling is magical: ordinary strings such as "PAD" or "UNK" get ordinary IDs if listed.
Encoding emits one ID per maximal nonempty code-point [run](../../glossary.md#term-run) separated by
`Character.isWhitespace(int)`, or UNK for an absent run. It preserves case, punctuation and exact
non-normalized [UTF-16](../../glossary.md#term-utf-16), including unpaired surrogates; it does not replace source characters.
Every encode returns a fresh array, including empty/whitespace-only input. Vocabulary state is
[immutable](../../glossary.md#term-immutable) and safely shareable. No [normalization](../../glossary.md#term-normalization), [BOS](../../glossary.md#term-bos)/[EOS](../../glossary.md#term-eos), decode, interface, vocabulary trainer,
[BPE](../../glossary.md#term-byte-pair-encoding-bpe), [artifact](../../glossary.md#term-artifact), [fingerprint](../../glossary.md#term-fingerprint), file/network I/O or [persistence](../../glossary.md#term-persistence) capability is authorized by this [slice](../../glossary.md#slice).

The final `TextSequenceBatcher` has private construction, static
`builder(WhitespaceTokenizer tokenizer)`, one public static final nested Builder exposing
`leftPadded()`, `rightPadded()`, `timeMajor()`, `batchMajor()`, `build()`, and
`batch(List<String> texts)` returning Data's `PaddedSequenceBatch` on the built batcher.
[Tokenizer](../../glossary.md#frozen-whitespace-tokenizer) is non-null and fixed; side and order have no defaults and missing choices throw
`IllegalStateException`. Setters/repeated builds have the same independent-configuration semantics
as Data's batcher. Batch rejects null list/rows with named `NullPointerException`, encodes each
String once in original order and invokes one configured `LongSequenceBatcher.batch` exactly once,
using tokenizer.paddingId(). Text never reimplements [padding](../../glossary.md#padding), [Tensor](../../glossary.md#tensor) allocation or valid counts;
empty and all-empty input inherit Data semantics. Callers coordinate list mutation during batch.
No new execution/[lifetime](../../glossary.md#term-lifetime) is introduced. PAD is an ordinary Embedding row: no row freezing,
zeroing, [gradient](../../glossary.md#term-gradient) suppression or masking is added. LEFT batch support does not change the current
[scan](../../glossary.md#term-cumulative-scan)'s right-valid-prefix contract, add runtime [host](../../glossary.md#term-host)-length [binding](../../glossary.md#term-binding) or extend [BPTT](../../glossary.md#term-backpropagation-through-time-bptt).

## `extensions/nn`

`extensions/nn` owns the stateful neural-network composition layer. It defines [model](../../glossary.md#term-model) [modules](../../glossary.md#term-module),
their [trainable parameters](../../glossary.md#trainable-parameter) and persistent [buffers](../../glossary.md#term-buffer), training/[evaluation](../../glossary.md#term-evaluation) mode, and the forward
context needed to apply that mode consistently through a module tree.

Allowed:

- `Module`
- [`Parameter`](../../glossary.md#term-parameter)
- `Buffer`
- module-owned parameter and buffer traversal
- training/evaluation mode propagation
- forward-context contracts
- neural-network layers, blocks, and functional conveniences composed from model semantics

Forbidden:

- [optimizer](../../glossary.md#term-optimizer) algorithms
- optimizer update orchestration
- [autograd](../../glossary.md#autograd) construction
- [backend](../../glossary.md#backend) storage access (public Model [host storage](../../glossary.md#host-storage) for owned Models and [checkpoint](../../glossary.md#term-checkpoint) restoration is permitted)
- backend [kernel](../../glossary.md#kernel) selection
- [concrete backend](../../glossary.md#concrete-backend) dependencies

`extensions/nn` has exactly public Model, Training, Checkpoint and [Engine](../../glossary.md#term-engine) project dependencies,
with no Data or Text edge, including tests. Model supplies [tensor](../../glossary.md#tensor)
and [operation](../../glossary.md#operation) semantics; Training supplies the generic [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters) [admission](../../glossary.md#term-admission) boundary.
`Module.trainingParameters()` returns a private live adapter retaining the root and exact ordered
paths, Parameter wrappers and Tensors. Creation rejects unbound reservations. Its [immutable](../../glossary.md#term-immutable) entries
[snapshot](../../glossary.md#term-snapshot) is not [validation](../../glossary.md#term-validation): `validateUnchanged()` re-traverses the live root and rejects any count,
order, path, wrapper or Tensor change. It deliberately does not freeze storage associations or
bytes, so an explicit native copy does not invalidate its own source. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) still owns no optimizer
algorithm or update orchestration. Owned Models delegate updates to [TrainingSession](../../glossary.md#training-session--trainingsession) and execution
to public Engine preparations; Checkpoint supplies cold [graph](../../glossary.md#graph)/value [persistence](../../glossary.md#term-persistence). The retained NN
persistence [signatures](../../glossary.md#term-signature) below remain supported. Model must not depend on NN.

### Owned [tensor](../../glossary.md#tensor) Models

The accepted symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) contract is future capability, not current owned-[Model](../../glossary.md#term-model) [admission](../../glossary.md#term-admission).
Dynamic external [signatures](../../glossary.md#term-signature) require separate [qualification](../../glossary.md#term-qualification) over public [Engine](../../glossary.md#term-engine) with one owned
[parameter](../../glossary.md#term-parameter) state and, for training, one [optimizer state](../../glossary.md#term-optimizer-state). Preserve exclusion, [Arena](../../glossary.md#term-arena) [lifetime](../../glossary.md#term-lifetime),
source guards, cached [owners](../../glossary.md#term-owner) and transactions; do not rebuild a [graph](../../glossary.md#graph) or mutate [prototypes](../../glossary.md#term-prototype) per B.
Symbolic `Model.graph`/`defineTensor` acquisition additionally needs [Compiler](../../glossary.md#term-compiler) [cold reconstruction](../../glossary.md#term-cold-reconstruction)
and [Checkpoint](../../glossary.md#term-checkpoint) copy support; symbolic save/load needs separate wire qualification. Builder
acquisition qualifies independently rather than inheriting cold or execution support. These
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/cold/wire/builder gates do not block ordinary low-level Engine+[CPU](../../glossary.md#term-cpu) forward/actual-backward
qualification. All current static saved Models/readers and fixed/general-[scan](../../glossary.md#term-cumulative-scan) policies remain.

All-valid [general-scan](recurrent-scan.md#general-differentiable-scan) owned Models use existing
`Model.graph`/`defineTensor`, `compute` and `configureTraining`/`train`, one private state set and
existing Engine/[TrainingSession](../../glossary.md#training-session--trainingsession) owners. The implemented compute/first-order training workflow
qualifies independently admitted static floating occurrences, including the CPU low-[precision](../../glossary.md#term-precision)
cutover; source validity and bounded cold reconstruction remain independent of [backend capability](../../glossary.md#backend-capability). All-valid
means every static time step participates, not support for arbitrary [PURE](../../glossary.md#term-pure) bodies. No public [API](../../glossary.md#term-application-programming-interface-api),
type, [dependency](../../glossary.md#term-dependency) or execution/[AD](../../glossary.md#autograd) layer is added. Graph and definition acquisition delegate the
exact cold body inventory to Checkpoint and Compiler, not CPU [dtype](../../glossary.md#data-type--datatype), [operation](../../glossary.md#operation) or [layout](../../glossary.md#layout) predicates.
Public graph acquisition retains Source list/null snapshots. A definition's one prototype forward
still precedes acquisition; [callback](../../glossary.md#term-callback) effects are not rolled back. Fixed-only scan detection and
its training policy below are unchanged, including mixed fixed/general sources. [Padding](../../glossary.md#padding)/row
skipping remains excluded from the qualified execution workflow. Compiler0018 consumer
qualification covers inherited static [homogeneous](../../glossary.md#term-homogeneous) length-mode graph/definition acquisition and
explicit cold forward through Checkpoint. V5 [persistence](../../glossary.md#term-persistence) and loaded generic Model numerical
proof are separately accepted under Checkpoint0014; NN0033 implements and qualifies the public
[runtime](../../glossary.md#term-runtime)-length [RNN](../../glossary.md#term-recurrent-neural-network-rnn) composition/[lifecycle](../../glossary.md#lifecycle) journey. Existing Model.graph/list compute/configureTraining/
train/save/loadGraph owners remain the only workflow; cold admission itself adds no builder API,
fluent migration, second-order waiver or implicit head/[loss](../../glossary.md#loss) [mask](../../glossary.md#term-boolean-mask).
Body persistence uses implemented [v4](#canonical-graph-model-version-4) for all-valid bodies and
[v5](#canonical-graph-model-version-5) for the static homogeneous length intersection when neither
contains [Pool3d](../../glossary.md#pool3d--ncdhw-pooling); outer or contained Pool3d selects [v6](#canonical-graph-model-version-6).

Model remains abstract and generic and implements AutoCloseable. Generic/custom/declarative
forward behavior is unchanged; unsupported tensor operations throw UnsupportedOperationException
before Engine or file work, and default close is a no-op. defineTensor returns `Model<Tensor,Tensor>`;
graph/loadGraph return `Model<List<Tensor>,List<Tensor>>`. Single load rejects multiple boundaries.
The single public final ModelBuilder and Model.builder() provide closed fluent construction in
nn.[module](../../glossary.md#term-module); concrete Model specializations and source/execution state remain private/package-private.

Builder input is a floating, positive fully static [rank](../../glossary.md#term-rank)-at-least-one [shape](../../glossary.md#shape), declared once before
steps and snapshotted into a storage-free [canonical](../../glossary.md#term-canonical-form) [contiguous](../../glossary.md#term-contiguous) offset-zero non-[view](../../glossary.md#term-view-meanings) prototype
with requiresGrad=false. A build requires input and a nonempty chain. The reusable caller-coordinated
builder snapshots recipes and creates fresh private children for every build; built Models retain
neither recipes nor caller state. Supplied [Linear](../../glossary.md#term-linear-meanings) [weights](../../glossary.md#term-weight)/[biases](../../glossary.md#term-bias) satisfy existing Linear guards,
match input dtype and are stored [provenance](../../glossary.md#provenance)-free [leaves](../../glossary.md#term-leaf). Copy each supplied [identity](../../glossary.md#term-identity-meanings) once through
GraphCheckpoint's zero-[node](../../glossary.md#node) raw leaf copier BEFORE constructing expressions, preserving [aliases](../../glossary.md#aliasing-and-arena).
Automatic recipes initialize fresh private state. No completed builder chain is copied, captured
or reconstructed through GraphCheckpoint during construction or cold forward.

The frozen Sequential is owned under `layers`; existing layers and private typed Tensor adapters
compose its prototype once in [eval mode](../../glossary.md#inference-and-evaluation). Registered wrappers determine roles and canonical paths;
ordinary provenance inventories reachable fresh [constants](../../glossary.md#term-constant), excludes every declared input from
constants, and verifies registered-state reachability.
The source retains exact final leaves/outputs and the chain, using GraphCheckpoint.Source only as
a [borrowed](../../glossary.md#ownership-borrowing-and-lease) Tensor-list [carrier](../../glossary.md#term-storage-carrier). Cold forward validates every full input [descriptor](../../glossary.md#term-descriptor), then applies
the same owned chain, accepting produced substitutions with arbitrary upstream provenance.
Compute/configure/train use fixed outputs and existing cached [bindings](../../glossary.md#term-binding), never recipes, forward
composition or reconstruction per batch. All live guards, exclusion, native admission and cleanup
below apply unchanged. Definition/graph/load acquisition and cold reconstruction remain unchanged.

Builder steps are Linear (automatic or supplied weight/optional bias), RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid),
Conv1d/2d/3d, MaxPool2d, MaxPool3d, AveragePool3d, flatten, RNN/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) and lastTimeStep.
`maxPool3d(kd,kh,kw)` and `averagePool3d(kd,kh,kw)` append ordinary Model operations
through the same private [pooling](../../glossary.md#term-pooling) adapter as MaxPool2d, not a second mathematical implementation.
Automatic layers use existing L64X128MixRandom algorithms with [Glorot](../../glossary.md#term-glorot-uniform) uniform and exact zero bias
enabled by default. Default [seed](../../glossary.md#term-seed) is zero; the last seed selected before build applies to all steps. Zero-based appended step i
(including [activations](../../glossary.md#term-activation)/transforms) uses `seed + 0x9E3779B97F4A7C15L * i` with Java long [wraparound](../../glossary.md#term-wraparound)
and a fresh transient generator; recurrent [matrix](../../glossary.md#term-matrix-meanings) order/fans remain the existing cell's.
Linear infers the last [feature](../../glossary.md#term-feature) [extent](../../glossary.md#term-extent). Convolutions use [NCW](../../glossary.md#term-ncw)/[NCHW](../../glossary.md#term-nchw)/[NCDHW](../../glossary.md#conv3d--ncdhw), [stride](../../glossary.md#term-stride)/[dilation](../../glossary.md#term-dilation)/groups one,
padding zero, bias true. Pool stride equals [kernel](../../glossary.md#kernel), padding zero, dilation one, floor geometry.
Flatten requires rank at least two and preserves [axis](../../glossary.md#term-axis) zero with a checked remaining-extent product.
For [TBF](../../glossary.md#term-tbf) (time/batch/features), flatten maps `[T,B,H]` to `[T,B*H]`; lastTimeStep first retains batch rows.
Recurrent steps statically unroll existing all-valid sequence.forward, reshape packed outputs to
`[1,B,H]` and concatenate on axis zero, returning every hidden step `[T,B,H]` even for LSTM.
Stacks retain TBF; lastTimeStep requires rank three and explicitly selects `select(0,T-1)` to `[B,H]`.
Cold composition starts zero h (and c for LSTM); cached execution reuses [immutable](../../glossary.md#term-immutable) zero constants,
never prior final state. This is not scan [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) and adds no adjoints or execution hierarchy.

The additive runtime-length builder API is exactly:

```java
ModelBuilder rnnScan(long hiddenSize);
ModelBuilder gruScan(long hiddenSize);
ModelBuilder lstmScan(long hiddenSize);
ModelBuilder finalHidden();
Model<List<Tensor>, List<Tensor>> buildWithLengths();
```

Retained fixed recipes delegate to the matching Sequence.forwardScan; NN0033's bounded [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding)
RNN exception below instead composes [general Scan](../../glossary.md#general-differentiable-scan). All use FORWARD direction, bias enabled,
positive hidden width and existing initialization/seed rules. The first scan requires
positive static TBF `[T,B,F]` and establishes one storage-free canonical non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) `[B]`
length prototype. All scans share that input; later scans require exact T/B [alignment](../../glossary.md#term-alignment) and infer
only F. Each owns independent parameters and zero initial h (and c for LSTM), never hidden [carry](../../glossary.md#term-carry).
Runtime lengths include zero. Advanced graph/list boundaries retain explicit initial/final states,
both directions, bias forms and the scan contract's zero T/B extents.

The production program authorizes this additive `RnnSequence` composition:
```java
public RecurrentScanResult forwardGeneralScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, RecurrentDirection direction);
```
It accepts any configured homogeneous floating dtype, including both [OFP8](../../glossary.md#term-ofp8) formats, static input [T,B,F], initial hidden [B,H],
non-gradient INT64[B] lengths, T/B >= 0, F/H > 0, bias on/off and both directions. Unsupported
dtype/geometry rejects before initialization or body construction. Reuse the existing descriptor-only
[atomic](../../glossary.md#resource-transaction-and-rollback) cell initialization and the same cell equation as ordinary forward, preserving its
prevalidation and bias/ADD order. Read no actual storage/length values and retain no runtime state.
Low-floating types retain ordinary cell operation-result [rounding](../../glossary.md#term-rounding) (including both projections
and the ordered bias/ADD/TANH sites), not fixed recurrent [FLOAT32](../../glossary.md#term-float32) working-site semantics.
Graph acquisition/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) preserve original carriers; [FP8](../../glossary.md#term-fp8) persistence requires v7, while old-only
sources retain their smallest sufficient old format. Numerical execution requires complete
[backend](../../glossary.md#backend) forward/actual-[adjoint](../../glossary.md#term-adjoint) qualification. Low-output training objectives explicitly cast to FLOAT32/[FLOAT64](../../glossary.md#term-float64).
[Snapshot](../../glossary.md#term-snapshot) current weights once; bind them as ordered outer invariants to canonical storage-free
carry, step and parameter formals. One ordinary next state is both next carry and emission;
Scan.apply uses rowCarryCount=1 and sequenceRowAxes=[ROW(0)] with the selected traversal.

RecurrentScanResult names semantic components, not universal [producer](../../glossary.md#term-producer) indices: fixed RNN/GRU
keeps outputs/finalHidden slots 0/1; this single-carry general RNN uses emission/carry slots 1/0.
The F32/F64 `rnnScan` recipe selects this general composition and its canonical result layouts;
low RNN stays fixed because its FLOAT32 contractions/aggregate/tanh and final state [narrowing](../../glossary.md#term-narrowing)
are observably different from the ordinary low cell program, not because general Scan rejects
the dtype. GRU/LSTM and direct fixed forwardScan also remain fixed. Final-hidden selection
must follow the actual producer's declared roles. Do not insert copies or erase view [metadata](../../glossary.md#term-metadata) to
imitate old fixed result layouts. General backend/layout qualification remains independent.
NN0033 is implemented and MAIN accepted after independent reviews, affected gates and public
F32/F64 repeated train/save/close/load/fresh-training proof. Exact canonical formal/emission/latent
geometry [preflight](../../glossary.md#term-preflight) precedes initialization and IDs. Qualified fixed and mixed fixed/general sources
now share the existing configured TrainingSession and Compiler AD. Ordinary `.rnn` is not migrated
and NN0032 compatibility is not waived.

`buildWithLengths()` requires at least one scan and declares `[values, validLengths]` in that order,
with one chain output. Existing `build()` retains `Model<Tensor,Tensor>` and rejects scan recipes;
the new terminal rejects no-scan chains. Incompatible terminals reject before acquisition/build
side effects. Both snapshot recipes and build fresh private state. `finalHidden()` selects the
immediately preceding scan [occurrence](../../glossary.md#term-occurrence)'s existing `[B,H]` port; intervening transforms, static
recurrence and repeated selection reject. It counts as an appended seed step. A zero-length row
returns semantic initial h, not a time-index selection. LSTM finalCell is not a builder result.

Ordinary steps remain legal wherever their existing shape contracts hold, without model-wide
masking. Bias/activations may change padded zeros; [convolution](../../glossary.md#term-channels-first-convolution-layers)/pooling retain channels-first dense
meaning, not sequence-aware axes. Flatten preserves axis zero (`[T,B,H]` to `[T,B*H]`). Static
recurrence ignores runtime lengths and still unrolls all-valid inputs. `lastTimeStep` remains
`select(0,T-1)`, not finalHidden. A prefix establishes the first scan's TBF interpretation; lengths
describe that boundary. Direct scan stacks share lengths; each scan ignores its own padded suffix
and zero-pads its own output, not intervening ordinary results. Never guess/redefine T/B or reshape
lengths to fit a later scan. After finalHidden removes time, another scan needs a rank-three boundary.

Cold list composition retains only instantiated owned steps through existing adapters, Sequential
[ownership](../../glossary.md#term-ownership-meanings) and ModelForward, not caller callbacks, builder recipes or per-[run](../../glossary.md#term-run) reconstruction.
It accepts exact-descriptor symbolic substitutions, including produced inputs, without Engine or
retained input storage. Compute uses the existing fixed source/cached execution path. No new
production type (including nested types), hierarchy, graph representation or execution path is
authorized; stop and replan if existing owners/adapters cannot express the composition.

Builder construction/cold forward/compute/train are independent of the closed [serialization](../../glossary.md#term-serialization)
domain below. Only save applies that admission: unsupported chains reject before file work,
without creating an Engine, freezing binding or invalidating/disabling compute or training.
The selected default [CNN](../../glossary.md#term-convolutional-neural-network-cnn) and static recurrent compositions fit the
[Compiler source slice](compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([slice glossary](../../glossary.md#slice))</span> and select
[graph-model v3](#canonical-graph-model-version-3). Static recurrence persists its existing [DAG](../../glossary.md#term-directed-acyclic-graph-dag),
including reachable INT64 row-index and floating zero-state CONSTANTs, not cell result objects or
layer recipes. Constants remain outside stateDictionary; equal-valued distinct identities remain
distinct, while shared identities retain their roles and aliases.

defineTensor seals its definition once, selects eval before one prototype forward (including
deferred initialization), then discards callbacks. Registered wrappers establish PARAMETER/[BUFFER](../../glossary.md#term-buffer)
roles, never requiresGrad alone; remaining reachable stored leaves receive deterministic collision-free
CONSTANT names. Every registered state must be reachable. Graph/load state paths must follow canonical
Module traversal order. Dotted namespaces contain no forward recipe. Constants are outside stateDictionary.
GraphCheckpoint.copy acquires each external identity once and reconstructs against private leaves;
strict state installation preserves wrappers, paths and aliases without mutating caller Tensors or stores.
Load adopts final [heap](../../glossary.md#term-heap) state without another copy. Callback side effects are not rolled back.
Implemented definition/graph acquisition uses the closed static FORWARD
[Compiler slice](compiler-autograd.md#compile-lifecycle), including selected fixed flat scans and
the bounded general-body exception. Persistence uses the same save/load APIs, with no NN wire:
[v4](#canonical-graph-model-version-4) for pre-Pool3d all-valid bodies, v5 for qualified length
mode without Pool3d, and [v6](#canonical-graph-model-version-6) for outer or contained Pool3d.
V1/v2/v3 file admission stays body-free.
The list boundary may declare sequence, non-gradient INT64 runtime lengths and initial states as
unique provenance-free inputs, disjoint from named state. Their prototype storage is never copied
or saved; copy/load recreate storage-free prototypes and cold forward accepts fresh exact full
descriptors. Explicit scan recipes and graph/load scan sources may compute through existing cached
public Engine execution; static recipes and sequence containers are not redirected. Fixed sources
may use configured first-order training when their actual backend/AD closure qualifies. Raw
persistence supports all nine dtypes independently of execution, with FP8 selecting graph v7.
The NN-[owned model](../../glossary.md#model-module-buffer-and-scan-distinctions)/whole-model codecs retain their separate conditional v1/v2 domains and flexible forward.

`HostTensorValue Model.compute(Tensor firstInput, Tensor secondInput)` is a generic convenience for
any owned Model with exactly two declared inputs and one output, including graph/loaded sources,
not a scan-only API. It binds declared positions, never inferred dtype, length role or [artifact](../../glossary.md#term-artifact)
metadata. After existing capability/lifecycle admission, check [arity](../../glossary.md#term-arity) before [arguments](../../glossary.md#term-argument) and use one
begin/end admission, `Long.MAX_VALUE` and existing computeOpen; do not nest public compute calls.
List/bounded compute and optional cold forward remain supported; forward is not a compute prerequisite.

All declared static slots, including unused ones, require exact descriptors (layout presence and
requiresGrad included). [Compile](../../glossary.md#compile)/session input order determines a cached positional binding plan;
surviving batch/[target](../../glossary.md#term-target) slots and owned leaves fill it. Unmapped [objective](../../glossary.md#term-objective) inputs reject. Compute/train
reuse public runValues: no per-batch source rebuild, rebind, prototype mutation, copy or provider [lookup](../../glossary.md#term-lookup).
Compute checks every [publication](../../glossary.md#term-publication) and aggregate byte bound before materializing any ordered occurrence,
closes [RunResult](../../glossary.md#run-result--runresult) on every path, and returns [detached](../../glossary.md#detached-model-results) [HostTensorValue](../../glossary.md#host-tensor-value--hosttensorvalue) values (default Long.MAX_VALUE).
Batch storage is borrowed for the whole [synchronous](../../glossary.md#term-synchronous) call. TrainingStep is detached too.

withEngine returns the same receiver and only selects a borrowed Engine, never closing it. Admission
checks selected-Engine closure, Model closing/closure, busy, then capability. withEngine then checks
freeze before null/closed candidate [validation](../../glossary.md#term-validation). Selection remains unchanged on failure. Otherwise,
valid compute/configure arguments and live-source checks freeze binding before Engine creation,
callback, native copy, compile or [prepare](../../glossary.md#prepare), even on failure; same-Engine rebinding then rejects.
Without explicit selection, setup lazily creates a model-owned CPU-only Engine.standard. The CPU
[owner](../../glossary.md#term-owner) may perform its existing optional native-route discovery/qualification.
Construction/load/save/cold forward never create an Engine.

Fixed recurrent presence is not an NN-level training rejection or separately cached source flag.
Builder, definition, graph and loaded sources use the same configured TrainingSession/Compiler AD
as ordinary graphs, including fixed recurrence behind ordinary heads, final-state roots and mixed
fixed/general sources. Preserve fixed versus general recipes, original storage carriers and
rounding boundaries; never redirect low fixed recipes through an ordinary-cell general Scan.
Objective validation remains [scalar](../../glossary.md#term-scalar) gradient-eligible FLOAT32/FLOAT64, before Engine/native setup.
Unsupported backend/AD requests follow the existing TrainingSession admission and native-state
transition ordering. CPU low arithmetic, higher-order recurrence and saved-[residual](../../glossary.md#scan-residual-port-and-history-demand) AD restrictions
are unchanged. Load restores raw parameters/source, not objectives, [optimizer history](../../glossary.md#term-optimizer-history) or sessions;
fresh configured training is distinct from exact continuation.

configureTraining succeeds once, keeps mode unchanged, and calls a pure objective builder once per
explicit attempt, retaining no callback. The scalar FLOAT32/FLOAT64 objective uses current Compiler
support but is not persisted. TrainingSession receives Module.trainingParameters. Close [inference](../../glossary.md#term-inference)
preparations before [TrainingParameterStorage](../../glossary.md#native-parameter-copy--trainingparameterstorage).copyToSharedNative installs parameter storage in a
model-owned shared Arena without changing Tensor identity. Copy failure preserves old associations;
successful native admission is not repeated on setup retry. Failed session setup cleans unpublished
owners and permits explicit frozen-Engine retry. Training retains unresolved-layout, empty, nonfinite,
alias/overlap rejection and RESET_AND_STEP [transaction](../../glossary.md#term-transaction) semantics; NN never converts or splits state.

Before compute/train/configure/cold forward/save, [guard](../../glossary.md#term-guard) every module mode and ordered
path/wrapper/Tensor/storage association. Mutation invalidates the capability and closes cached
owners, without recapture, adopting replacement state or resetting optimizer history. Same-store
trained bytes stay valid. Inherited Module mutators remain caller-coordinated.
Only one high-level operation is admitted; busy rejects before arguments. Close bypasses admission,
rejects new work, waits for active work, then closes session/preparation, native owners and owned
Engine, attempting all cleanup, suppressing distinct failures and replaying the retained result
on repeated/concurrent close, even when a borrowed Engine closed first.

Save uses the fixed GraphCheckpoint source and current owned leaves only, including before compute
and after training. Explicit ArtifactLimits remain. Short forms use 256 MiB file/aggregate,
4 MiB [manifest](../../glossary.md#term-manifest), 16384 entries, 64-byte keys and 64 MiB [payload](../../glossary.md#artifact-envelope-and-atomic-publication) ceilings. No [optimizer](../../glossary.md#term-optimizer)/[RNG](../../glossary.md#graph-rng-state--graphrngstate)/cursor
[resume](../../glossary.md#schema-serialization-and-checkpoint) or Model/Training [autotuning](../../glossary.md#autotuning--model-autotuning) is implied. Zero-node source can persist; Engine's empty-plan
rejection remains unchanged.

### Closed declarative models

[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns [immutable](../../glossary.md#term-immutable) semantic [model](../../glossary.md#term-model) specifications and reconstruction of a closed [Tensor](../../glossary.md#tensor)-to-Tensor
Sequential tree. The NN0026 [API](../../glossary.md#term-application-programming-interface-api) is implemented, independently reviewed and coordinator-validated.
The specification contains no live Tensor, [callback](../../glossary.md#term-callback), Java class name, [backend](../../glossary.md#backend), storage handle,
persistent version, or bytes. Existing generic Model/ModelDefinition and Sequential APIs remain
supported. Description never [captures](../../glossary.md#term-capture-meanings) arbitrary Model.define code or custom [module](../../glossary.md#term-module) subclasses.

The supported tree has nested Sequential, [Linear](../../glossary.md#term-linear-meanings), affine LayerNorm, [Embedding](../../glossary.md#embedding), Conv1d/2d/3d,
and closed RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid) steps. [Parameter](../../glossary.md#term-parameter) types are all four existing floating types:
[FLOAT64](../../glossary.md#term-float64), [FLOAT32](../../glossary.md#term-float32), [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16). There is no [homogeneous](../../glossary.md#term-homogeneous)-input-type shortcut:

| Step | Retained configuration and accepted input/output |
|---|---|
| Sequential | Ordered children; each receives its predecessor's exact result once. Empty returns the exact input. |
| Linear | Positive resolved in/out [features](../../glossary.md#term-feature), [bias](../../glossary.md#term-bias) presence, parameter type, and whether constructed with automatic input guards. Supplied-state form retains Model promotion and [contraction](../../glossary.md#contraction-dimension) checks; automatic form retains exact configured input type and positive static final feature [extent](../../glossary.md#term-extent). Output preserves leading [Dimensions](../../glossary.md#dimension) and replaces the final extent with out-features. |
| LayerNorm | Positive fully static normalized trailing [Shape](../../glossary.md#shape), parameter type and exact [ScalarValue](../../glossary.md#typed-scalar-value--scalarvalue) [epsilon](../../glossary.md#epsilon) type/bits; mandatory scale then bias. Input promotion, trailing matching and result-arithmetic epsilon checks remain Model-owned, including rejection when promotion makes retained epsilon incompatible. Output Shape is the input Shape. |
| Embedding | Positive vocabulary/embedding sizes and floating table type; no [padding](../../glossary.md#padding)/frozen row. [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) input of any [rank](../../glossary.md#term-rank); output appends embedding width and has table type. Index-value bounds remain the existing Model/backend obligation. |
| Conv1d/2d/3d | Positive resolved in/out channels and [kernel](../../glossary.md#kernel) extents, bias, exact parameter/input type and complete existing Conv1dAttrs/Conv2dAttrs/Conv3dAttrs ([stride](../../glossary.md#term-stride), symmetric padding, [dilation](../../glossary.md#term-dilation), groups). Fixed [NCW](../../glossary.md#term-ncw)/[NCHW](../../glossary.md#term-nchw)/[NCDHW](../../glossary.md#conv3d--ncdhw) [layout](../../glossary.md#layout); ranks 3/4/5, positive static matching channels and existing geometry guards. [Batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[spatial](../../glossary.md#term-spatial-axes) Dimensions may vary compatibly; output geometry remains Model-owned. |
| [Activation](../../glossary.md#term-activation) | One closed kind, delegating to the matching Model Tensor [operation](../../glossary.md#operation); floating input, same Shape/type, existing [gradient](../../glossary.md#term-gradient) and numerical semantics. |

Input/output constraints are derived by composing these rules, never saved from an example batch.
Compatible variable leading Dimensions/ranks for Linear and LayerNorm, arbitrary index Shape for
Embedding, and variable [convolution](../../glossary.md#term-channels-first-convolution-layers) batch/spatial Dimensions remain supported. Dynamic dimensions
are accepted only where the existing corresponding layer/Model contract accepts them. Description
requires all [lazy](../../glossary.md#term-lazy-evaluation) parameters already bound; it does not [run](../../glossary.md#term-run) forward to initialize them. Initializers,
[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), source labels/layouts/[provenance](../../glossary.md#provenance) and modes are not reconstruction configuration: selected
steps are mode-insensitive, reconstructed models start in [evaluation](../../glossary.md#term-evaluation) mode, and state values load
separately. BatchNorm, [Dropout](../../glossary.md#dropout--inverted-dropout), recurrence, branches, reused execution [nodes](../../glossary.md#node) and custom I/O are out.
An empty/activation-only tree is semantically eligible; reconstruction itself promises neither
generic zero-operation [Engine](../../glossary.md#term-engine) execution nor a newly admitted [backend route](../../glossary.md#backend-route), [dtype](../../glossary.md#data-type--datatype), layout or gradient.
Existing biased [Conv3d](../../glossary.md#conv3d--ncdhw) recipes may train after reload when [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) and the complete backend
closure qualify: current production proof covers [CPU](../../glossary.md#term-cpu) FLOAT32/FLOAT64 and [Metal](../../glossary.md#term-metal) FLOAT32. This
changes no definition or [graph](../../glossary.md#graph)-model inventory/version, [optimizer](../../glossary.md#term-optimizer) [persistence](../../glossary.md#term-persistence) or exact-[resume](../../glossary.md#schema-serialization-and-checkpoint) contract.

The public boundary is ModelSpecification and final DeclarativeModel extending Model<Tensor,Tensor>
in nn.module. A specification has one closed Sequence root, immutable typed step records and
explicit AliasGroup path lists. ModelSpecification.describe(Sequential) describes an ordinary
supported tree without taking [ownership](../../glossary.md#term-ownership-meanings); DeclarativeModel.describe() describes its current tree.
Both read current [bindings](../../glossary.md#term-binding) under caller-owned [quiescence](../../glossary.md#term-quiescence), not creation-time [alias](../../glossary.md#aliasing-and-arena) [metadata](../../glossary.md#term-metadata).
Each supported concrete layer exposes one typed specification() step [snapshot](../../glossary.md#term-snapshot); public Activation
in nn.layers supplies closed activation construction. No [mutable](../../glossary.md#term-mutable)-layer reflection or [registry](../../glossary.md#term-registry).
ModelSpecification.stateDescriptors() exposes immutable (path, kind, dataType, Shape, requiresGrad)
entries derived from configuration in exact Module.stateDictionary traversal order, without Tensors.
ModelSpecification.outputDescriptor(TensorDescriptor) checks composed input constraints and returns
the corresponding Model result metadata without evaluation, storage allocation or backend [admission](../../glossary.md#term-admission).

The root is represented directly by DeclarativeModel's indexed children, not by a named wrapper.
Root children are "0", "1", etc.; nested children extend the same dot-separated paths. Parameters
remain [weight](../../glossary.md#term-weight) then optional bias, or LayerNorm scale then bias. No [buffers](../../glossary.md#term-buffer) occur in this [matrix](../../glossary.md#term-matrix-meanings).
Numeric encounter order, not lexical sorting, defines traversal. A closed value tree carries no
node IDs, arbitrary connections or references: wiring/ownership cycles and dangling references
are unrepresentable through its safe public constructors. Reusing a step value means independent
occurrences, never shared module [identity](../../glossary.md#term-identity-meanings). NN validates nulls, supported kinds, positive dimensions,
parameter [schema](../../glossary.md#term-schema), epsilon, geometry and checked [element](../../glossary.md#term-element)/byte/count arithmetic before returning a
specification or skeleton. Composition compatibility is checked for a supplied input [descriptor](../../glossary.md#term-descriptor)
through existing storage-free Model metadata operations on the [cold path](../../glossary.md#term-cold-path), and on each forward;
there is no generic shape solver or proof that a tree admits every possible input. [Runtime](../../glossary.md#term-runtime)-dependent
checks remain with their current [owners](../../glossary.md#term-owner); description needs no speculative input or fixed batch.
NN bounds each root to 4096 step occurrences (including sequences) and depth 64 (root depth 1),
checks limits before recursive reconstruction/copy, and uses stack-safe [validation](../../glossary.md#term-validation). Shape and
convolution array-count constraints retain their existing [owner](../../glossary.md#term-owner) rules. These bounds protect cold
tree processing, not [backend capability](../../glossary.md#backend-capability) or runtime shape sizes. Future wire parsers separately
own unknown IDs/fields, [canonical](../../glossary.md#term-canonical-form) ordering, string/rank/length limits and aggregate/file-byte
[preflight](../../glossary.md#term-preflight) before allocating decoded objects; safe Java tree constructors do not validate raw bytes.

An alias group means repeated exact parameter Tensor identity at distinct state paths, not equal
values or overlapping physical storage. Groups have at least two distinct existing parameter paths,
equal type/Shape/gradient schema, disjoint membership, members in state order, and groups ordered
by first member. Unlisted paths are independent singletons. NN rejects invalid/noncanonical groups.
Description recomputes groups by exact current identity; independent equal-valued parameters stay
independent. Legal Parameter.replace or strict state loading may join/split groups; the next describe
must reflect that change. No physical-overlap [topology](../../glossary.md#term-topology), [view](../../glossary.md#term-view-meanings) layout or source expression is recreated.
Training's duplicate-identity/overlap rejection is unchanged; aliases are not split to evade it.

DeclarativeModel.reconstruct(ModelSpecification) returns a private-candidate, fully bound descriptor
skeleton, not a value-loaded executable model. It creates one storage-free, provenance-free,
gradient-eligible Tensor per alias group or [singleton](../../glossary.md#term-singleton), with the derived static parameter descriptor,
and installs the exact same reference at every group path through NN-owned constructors/strict load.
Linear/Embedding/LayerNorm accept supplied descriptor Tensors; convolutions bind reservations via
strict load before any forward. Linear.fromState(Tensor, Optional<Tensor>, [boolean](../../glossary.md#data-type--datatype) automaticInputGuards)
creates a fully bound layer retaining the selected supplied-state or automatic [guard](../../glossary.md#term-guard) policy, without
calling initialization. It is not a new lazy-initialization path or a change to existing constructors.
No random source, weight-sized zero/one array, throwaway initialized weights, evaluation, native
resource or new Engine path is permitted. Existing public stateDictionary/loadStateDictionary remain
the value bridge: a later recovery owner sees these fully bound alias references and can load one
fresh value Tensor per [target](../../glossary.md#term-target) identity group without changing its existing target-load contract.
NN reconstruction validates before returning; failure exposes no partial candidate and never mutates
the described source. Caller quiescence still owns consistency; NN adds no lock or [transaction](../../glossary.md#term-transaction).

### Retained [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [checkpoint](../../glossary.md#term-checkpoint) APIs

The nine old [model](../../glossary.md#term-model) [persistence](../../glossary.md#term-persistence) classes now live in `nn.checkpoint`, without compatibility
forwarders. The three public types remain ModelCheckpoint, ModelCheckpointInspection and
InputArtifactFingerprint. The following old-v1 rules belong to NN; they do not restrict the new
generic [graph](../../glossary.md#graph) format or transfer NN semantics into Checkpoint.

[FP8](../../glossary.md#term-fp8) values append `model.v2` with magic `SYNMOD02` and version 2, preserving the existing record
[layout](../../glossary.md#layout) and adding [dtype](../../glossary.md#data-type--datatype) IDs 8=FLOAT8_E4M3 and 9=FLOAT8_E5M2. Whole-model FP8 definitions use
`SYNDEF02` and the corresponding `whole-model.v2` kind. Non-FP8 sources retain byte-identical v1
encoding. Old v1 readers reject the new tuples; no [optimizer](../../glossary.md#term-optimizer), history or prepared state is added.

Its model package owns a separate version-one [metadata](../../glossary.md#term-metadata) [codec](../../glossary.md#term-codec), a [detached](../../glossary.md#detached-model-results) candidate, and a public
`ModelCheckpoint` file and model-envelope [facade](../../glossary.md#term-facade); it is not a state installer. For the existing
values-only operations, under caller-owned [quiescence](../../glossary.md#term-quiescence), [capture](../../glossary.md#term-capture-meanings) calls
`Module.stateDictionary()` exactly once; an unbound or incomplete NN state fails there. Before
any [Engine](../../glossary.md#term-engine) call or Checkpoint-owned [payload](../../glossary.md#artifact-envelope-and-atomic-publication)-sized allocation, capture validates exact NN paths,
32-byte caller-supplied [SHA-256](../../glossary.md#term-sha-256) configuration and optional input-[artifact](../../glossary.md#term-artifact) fingerprints, static
[Shape](../../glossary.md#shape), checked [canonical](../../glossary.md#term-canonical-form) byte length, entry and payload limits, aggregate bytes including every
[alias](../../glossary.md#aliasing-and-arena), [manifest](../../glossary.md#term-manifest) bytes, and final file bytes. It deduplicates exact [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-reference-identity) in first-
encounter order only for one `Engine.compute(List<Tensor>, limit)` call. Empty state makes no
Engine call. It checks returned [data type](../../glossary.md#data-type--datatype), Shape, and byte length before copying detached values
into the [opaque](../../glossary.md#term-opaque) envelope. No live Tensor, [backend](../../glossary.md#backend) storage, or Engine handle escapes the candidate.
The caller defines what its versioned configuration and optional input-artifact digests cover;
Checkpoint cannot serialize arbitrary Java model code or infer [tokenizer](../../glossary.md#frozen-whitespace-tokenizer)/configuration meaning.

The canonical model metadata payload has key [`schema`](../../glossary.md#term-schema) and encodes, in [big-endian](../../glossary.md#text-encoding-and-file-boundaries) order: eight
[ASCII](../../glossary.md#term-ascii) bytes `SYNMOD01`, signed 32-bit version `1`, 32 configuration-[digest](../../glossary.md#term-digest) bytes, one input-
digest presence byte (`0` or `1`), optional 32 input-digest bytes, signed 32-bit entry count,
then ordered entries. Each entry is a signed 32-bit positive [UTF-8](../../glossary.md#term-utf-8) path byte length, exact path
bytes, one StateKind ID, one [DataType](../../glossary.md#data-type--datatype) ID, signed 32-bit nonnegative Shape [rank](../../glossary.md#term-rank), that many signed
64-bit nonnegative extents, and signed 64-bit expected canonical payload length. Paths use NN's
nonblank dot-segment grammar with valid [Unicode](../../glossary.md#term-unicode); decoding requires strict UTF-8 and retains the
exact text without [normalization](../../glossary.md#term-normalization). Kind IDs are `1` [parameter](../../glossary.md#term-parameter) and `2` [buffer](../../glossary.md#term-buffer). DataType IDs are
`1` [FLOAT64](../../glossary.md#term-float64), `2` [FLOAT32](../../glossary.md#term-float32), `3` [BFLOAT16](../../glossary.md#term-bfloat16), `4` [INT32](../../glossary.md#data-type--datatype), `5` [INT64](../../glossary.md#data-type--datatype), `6` [BOOL](../../glossary.md#data-type--datatype), and `7` [FLOAT16](../../glossary.md#term-float16). IDs do
not derive from enum order or names. No [padding](../../glossary.md#padding) or trailing bytes are permitted. SHA-256 of these
exact metadata bytes is the model schema [fingerprint](../../glossary.md#term-fingerprint), so it binds both caller digests and every
ordered path, kind, type, Shape, and length.

The artifact kind is exactly `model.v1`; each state payload key is `state/` followed by its
zero-based, eight-digit ASCII decimal ordinal (`state/00000000` through `state/99999999`). NN
paths occur only in metadata, never in artifact keys or filesystem paths. Decoder acceptance
requires exactly one `schema` and one correctly keyed, correctly sized payload per ordered entry,
with no extra or missing entry, repeated path, unknown wire ID, negative [extent](../../glossary.md#term-extent), malformed UTF-8,
noncanonical length, or trailing metadata. A candidate is fully detached and [immutable](../../glossary.md#term-immutable). The
configured aggregate limit bounds schema plus every state's canonical output bytes, including
aliased paths; it does not bound Engine workspace, source detached values, defensive copies, or
peak [heap](../../glossary.md#term-heap). The envelope's own v1 [framing](../../glossary.md#term-framing) and checksums remain unchanged. No cross-call [cache](../../glossary.md#term-cache),
implicit initialization, or model-defining code reconstruction is provided. Public inspection
fully validates file framing, schema, payload correspondence, and canonical BOOL bytes before
returning detached immutable typed [descriptors](../../glossary.md#term-descriptor) and defensive digest copies. Load rereads and
revalidates the file, checks exact caller configuration and optional input-digest presence and
bytes, and rejects an unbound [target](../../glossary.md#term-target) by requiring a complete NN state [snapshot](../../glossary.md#term-snapshot) before
installation. It validates every path, kind, type, Shape, [gradient](../../glossary.md#term-gradient) requirement, and target
Tensor-[identity](../../glossary.md#term-identity-meanings) alias group before constructing a complete candidate dictionary. The v1 wire
stores one value per path and cannot prove source Tensor identity: each fully bound target alias
group must have identical stored bytes, then receives one fresh dense-[contiguous](../../glossary.md#term-contiguous) [host](../../glossary.md#term-host) Tensor per
group. A group containing a parameter receives a gradient-eligible Tensor; buffer-only groups do
not. Canonical big-endian bits are restored into data-type-compatible native-order heap carriers
without numeric conversion or [NaN](../../glossary.md#term-nan) normalization. NN alone performs the one strict whole-tree
installation attempt after all checkpoint [validation](../../glossary.md#term-validation) and reconstruction. Save and load require
caller-owned quiescence; a prior inspection is not trusted for later load.

The public model-envelope composition boundary implements the [scope](../../glossary.md#term-scope) approved for Checkpoint
0005A alongside the file-based facade. `ModelCheckpoint` exposes these static operations using
existing public types:

- `capture(Module, Engine, byte[] configDigest, byte[] inputDigest, ArtifactLimits)` returns
  `ArtifactEnvelope` and declares checked `ArtifactFormatException` for semantic decode failure.
- `inspect(ArtifactEnvelope, ArtifactLimits)` returns `ModelCheckpointInspection` and declares
  checked `ArtifactFormatException`.
- `canonicalStateBytes(ArtifactEnvelope, ArtifactLimits)` returns `Map<String, ByteBuffer>` and
  declares checked `ArtifactFormatException`.
- `load(ArtifactEnvelope, Module, byte[] expectedConfigDigest, byte[] expectedInputDigest,
  ArtifactLimits)` returns `void` and declares checked `ArtifactFormatException`.

These operations must reuse the current internal capture, model codec, and restoration machinery,
including `ModelArtifactCodec.decode` and its [preflight](../../glossary.md#term-preflight) of every `ArtifactLimits` [dimension](../../glossary.md#dimension) before
schema copying. Standalone envelopes have not passed `ArtifactFile.read`: validate kind, schema,
exact payload correspondence, canonical BOOL bytes, and all limits, including the equivalent v1
manifest and complete file size, on each call. An in-memory envelope has no encoded file [checksum](../../glossary.md#term-checksum)
to verify; file framing and [integrity](../../glossary.md#term-integrity) validation remain the file reader's responsibility.
Capture returns only a fully validated detached envelope. No live [Module](../../glossary.md#term-module), Tensor, Engine, or
[borrowed](../../glossary.md#ownership-borrowing-and-lease) storage [lifetime](../../glossary.md#term-lifetime) escapes. Limits retain their existing meaning: aggregate canonical
bytes include schema and every aliased path, not defensive copies, workspace, or peak heap.

Canonical bytes include every parameter and buffer under its exact NN path in model metadata
order, not lexical path order or artifact-key order. The map structure is immutable and each
call returns fresh independent read-only buffers positioned at zero, with exact payload limits
and `BIG_ENDIAN` [byte order](../../glossary.md#text-encoding-and-file-boundaries). Payload bytes are immutable; buffer position, limit, and byte-order
settings remain caller-[mutable](../../glossary.md#term-mutable) and must not affect another call or path [view](../../glossary.md#term-view-meanings). All seven stored
data types retain their raw logical bits, including [signed zero](../../glossary.md#term-signed-zero) and NaN payloads; no conversion
or numerical normalization is authorized. Inspection supplies descriptors, not a validation
credential: canonical-byte access and load independently validate the supplied envelope.

Envelope load must validate the complete model and expected digests before delegating one strict
NN installation attempt, preserving the fully bound target and target-defined alias policies.
The file [API](../../glossary.md#term-application-programming-interface-api) must delegate to the same semantic machinery while retaining rereads on every
inspect/load, existing checked `IOException` [signatures](../../glossary.md#term-signature), failure types and validation ordering,
and the existing save [publication](../../glossary.md#term-publication) policy. The internal `ModelArtifactCandidate`,
`ModelArtifactCodec`, `ModelArtifactCapture`, and `ModelStateRestoration` remain package-private.
This boundary adds no public model-image wrapper, schema, codec, [registry](../../glossary.md#term-registry), temporary-file
composition pipeline, backend access, Training validator, or shared [transaction](../../glossary.md#term-transaction).

For a trusted existing local directory and [quiescent](../../glossary.md#term-quiescent) caller-owned target, save writes a newly
created sibling temporary file, forces that file through its writing [channel](../../glossary.md#term-channel), closes, reopens, and
fully validates it against the source image, then attempts exactly one `ATOMIC_MOVE`. It may
replace an existing regular target; it rejects a symlink or directory target before work. There
is no copy/non-[atomic](../../glossary.md#resource-transaction-and-rollback) fallback or portable exclusive no-replace promise. A successful move gives
atomic live visibility of the prior complete file (or absence for a new target) or the new complete
file, not a cross-platform guarantee that the rename survives power [loss](../../glossary.md#loss). A failed move may have
provider-dependent target state: save reports failure, never deliberately deletes the target,
and callers must inspect it. Pre-move failures attempt to clean the owned temporary file. An explicit
unsupported-atomic-move result permits attempted temporary cleanup; other failed move attempts leave the
temporary path untouched because their outcome is uncertain. No filesystem [operation](../../glossary.md#operation) follows a
successful move. The envelope is not encryption or adversarial [authentication](../../glossary.md#term-authentication).

These relocated NN APIs retain their existing semantics, readers, bounds and failures. They use
Checkpoint's opaque artifact package and public Engine [materialization](../../glossary.md#materialization), never [Compiler](../../glossary.md#term-compiler) or backend
internals. Their package is `nn.checkpoint`; no old-package forwarders remain.

### Whole-[model](../../glossary.md#term-model) definition and composition

This is the accepted CP0005B contract (2026-10-14), independently approved and coordinator-verified
for implementation. CP0005B is Complete (2026-10-15): implementation, independent code and
documentation reviews, [module](../../glossary.md#term-module)/[Javadoc](../../glossary.md#term-javadoc) and coordinator gates passed. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s closed declarative
contract above owns meaning; NN's relocated `nn.checkpoint` package owns the
following old persistent representation and public composition. Generic [Checkpoint](../../glossary.md#term-checkpoint) owns only the
[opaque](../../glossary.md#term-opaque) envelope and the separate [graph](../../glossary.md#graph)-model.v1 boundary below. Training and recurrent extensions
remain Proposed. Existing values-only methods, `model.v1`, ArtifactFile v1, and ArtifactEnvelope's
defensive-copy boundary remain unchanged. The relocation adds only the root-declared NN public Checkpoint/[Engine](../../glossary.md#term-engine) edges; no Java-object
[serialization](../../glossary.md#term-serialization) or second execution pipeline is added.

#### Whole-[model](../../glossary.md#term-model) public [API](../../glossary.md#term-application-programming-interface-api) and failures

Add only these static overloads to existing `nn.checkpoint.ModelCheckpoint` (types use their
existing owning packages). File methods declare `IOException`; envelope methods declare
`ArtifactFormatException`, the checked IOException subtype. These [signatures](../../glossary.md#term-signature) are accepted:

```text
void save(Path, Module, Engine)
void save(Path, Module, Engine, ArtifactLimits)
void save(Path, Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel load(Path)
DeclarativeModel load(Path, ArtifactLimits)
DeclarativeModel load(Path, InputArtifactFingerprint, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, InputArtifactFingerprint, ArtifactLimits)
```

The [Module](../../glossary.md#term-module) source is recognized only as Sequential or final DeclarativeModel; use respectively
ModelSpecification.describe(source) or source.describe(). Reject custom roots/children and
unbound state before [Engine](../../glossary.md#term-engine) or filesystem work. A save checks its required [arguments](../../glossary.md#term-argument) and [captures](../../glossary.md#term-capture-meanings)
before calling ArtifactFile.save; no temporary component files. Entire-call caller [quiescence](../../glossary.md#term-quiescence)
spans description and public values [capture](../../glossary.md#term-capture-meanings). Each may read state: values capture's existing
stateDictionary-once rule is not a combined whole-model [snapshot](../../glossary.md#term-snapshot), lock or [transaction](../../glossary.md#term-transaction) promise.

The only new public type is final `io.github.pho001.synaptik.nn.checkpoint.InputArtifactFingerprint`:
public constructor `InputArtifactFingerprint(byte[] bytes)`, public `byte[] bytes()`, and value
`boolean equals(Object)`/`int hashCode()`. Constructor and accessor copy exactly 32 [SHA-256](../../glossary.md#term-sha-256) bytes; null rejects
with NullPointerException, wrong length with IllegalArgumentException. Equality uses all 32 bytes.
It is expected application-verified [identity](../../glossary.md#term-identity-meanings), not verification performed by [Checkpoint](../../glossary.md#term-checkpoint), a trust
credential or [authentication](../../glossary.md#term-authentication). [Fingerprint](../../glossary.md#term-fingerprint) overloads require a non-null value. Overloads without
it save absent input [digest](../../glossary.md#term-digest) and require absence on load; explicit load requires presence and exact
bytes. A mismatch, including stored presence on default load, throws ArtifactFormatException with
an explicit-expected-identity diagnostic. No generic options, [registry](../../glossary.md#term-registry) or input-[artifact](../../glossary.md#term-artifact) parsing.

Required null arguments throw NullPointerException. Unsupported/unbound sources, source [schema](../../glossary.md#term-schema)
and [preflight](../../glossary.md#term-preflight) failures retain existing IllegalArgumentException/IllegalStateException and checked
arithmetic ArithmeticException. Engine unchecked failures propagate. Malformed/unsupported stored
data, incoherent [descriptors](../../glossary.md#term-descriptor)/config/[aliases](../../glossary.md#aliasing-and-arena), input mismatch and decode bounds/[overflow](../../glossary.md#term-overflow) are
ArtifactFormatException (wrap [owner](../../glossary.md#term-owner) [validation](../../glossary.md#term-validation) failures with their cause). File access/[publication](../../glossary.md#term-publication)
failures remain IOException; fatal VM failures are not converted. [Tensor](../../glossary.md#tensor)-identifier exhaustion
remains IllegalStateException. A values-only file fails high-level load with a missing-definition
message directing callers to existing [target](../../glossary.md#term-target)-load, never inferred [topology](../../glossary.md#term-topology) or silent upconversion.

#### Whole-[model](../../glossary.md#term-model) limits and [preflight](../../glossary.md#term-preflight)

Short file overloads use `ArtifactLimits(268435456L, 4194304, 16384, 64, 67108864, 268435456L)`:
complete file/aggregate 256 MiB, [manifest](../../glossary.md#term-manifest) 4 MiB, entries 16384, [ASCII](../../glossary.md#term-ascii) kind/key 64 bytes, single
[payload](../../glossary.md#artifact-envelope-and-atomic-publication) 64 MiB; MiB is 1048576 bytes. Advanced overloads reuse existing ArtifactLimits, but every
definition is at most 1048576 bytes even with larger limits. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s 4096 occurrences/depth 64 still
apply (root counts once at depth 1). These are encoded bounds, not peak-[heap](../../glossary.md#term-heap)/workspace promises.

Dry-size the [canonical](../../glossary.md#term-canonical-form) definition with checked arithmetic before creating its byte array. Let D
be that size and n the predicted inner model entry count (stateDescriptors.size() + 1 for [schema](../../glossary.md#term-schema)).
The inner manifest is `66 + 58*(n-1)` bytes. Outer kind length is 14 instead of 8, each inner key
gains the 6-byte `model/` prefix, and the new 17-byte `definition/schema` key adds 61 bytes:
manifest delta is `67 + 6*n`, file delta `D + 67 + 6*n`, aggregate delta D, entry delta 1.
Check outer kind/definition key lengths, D against both definition and single-payload ceilings,
and checked residuals before invoking public [capture](../../glossary.md#term-capture-meanings). Pass projected limits with file minus file
delta, manifest minus manifest delta, entries minus 1, key bytes minus 6, unchanged maxPayloadBytes,
and aggregate minus D. Reject [residual](../../glossary.md#scan-residual-port-and-history-demand) file <48, manifest <8, negative counts/bytes or unusable
key ceilings safely before constructing ArtifactLimits. With separately checked outer keys/D,
these residuals are faithful equivalence to the outer [framing](../../glossary.md#term-framing), not a new public validator or a
duplicate private schema/ordinal parser. The feasible capture order is: dry-size D and check outer
definition/residual [metadata](../../glossary.md#term-metadata) bounds; encode the already bounded definition (at most 1 MiB); [hash](../../glossary.md#term-hash)
those exact bytes; pass that [digest](../../glossary.md#term-digest) and projected limits to public values capture, whose actual
component preflight precedes [Engine](../../glossary.md#term-engine) and bulk value-sized allocations/copies. The bounded definition
allocation is metadata, not bulk values; no new public validator/streaming proof layer is required.
After capture, check complete outer sizes/coherence again before prefixed payload copies; caller
[quiescence](../../glossary.md#term-quiescence) makes the described count stable. Validate final correspondence anew.

On envelope load, preflight all six outer encoded limits using entry sizes/read-only [views](../../glossary.md#term-view-meanings) and
checked `48 + manifest + aggregate` before any definition/payload copy or projected envelope
construction. Check exact outer keys/kind and definition ceiling, then apply the equivalent
projected limits and public component [validation](../../glossary.md#term-validation). File load uses ArtifactFile.read first; its
framing preflight already precedes file payload allocation. Raw nested counts/lengths are bounded
by remaining definition bytes and derived state counts before arrays/lists/strings are allocated.
In particular positive LayerNorm [rank](../../glossary.md#term-rank) requires room for `8*rank` [extent](../../glossary.md#term-extent) bytes plus its typed
[scalar](../../glossary.md#term-scalar) before allocating long[]; no [backend](../../glossary.md#backend) rank-16 restriction belongs in this format.

#### [Canonical](../../glossary.md#term-canonical-form) definition version 1

All fields are [big-endian](../../glossary.md#text-encoding-and-file-boundaries), with no [padding](../../glossary.md#padding), optional extensions or trailing bytes. `i32`/`i64`
are signed 32/64-bit fields; counts are nonnegative unless specified positive. `u8` is one byte.
Flags accept exactly 0 or 1. The [payload](../../glossary.md#artifact-envelope-and-atomic-publication) is eight [ASCII](../../glossary.md#term-ascii) [magic bytes](../../glossary.md#term-magic-bytes) `SYNDEF01`, i32 version 1,
one root [node](../../glossary.md#node) (must be Sequence), then i32 [alias](../../glossary.md#aliasing-and-arena)-group count and those groups. Nodes use prefix
order: one u8 kind ID then precisely the fields below; Sequence children follow immediately.
No IDs, edge tables, cycles, saved [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [signatures](../../glossary.md#term-signature) or redundant output [descriptors](../../glossary.md#term-descriptor) are encoded:
the closed value tree represents wiring, and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) derives input/output constraints from these fields.

| ID / kind | Fields after kind, in exact order |
|---|---|
| 1 Sequence | i32 child count; each child node in numeric encounter order. |
| 2 [Linear](../../glossary.md#term-linear-meanings) | i64 inFeatures; i64 outFeatures; u8 [bias](../../glossary.md#term-bias); u8 [parameter](../../glossary.md#term-parameter) [DataType](../../glossary.md#data-type--datatype); u8 automaticInputGuards. |
| 3 LayerNorm | u8 parameter DataType; positive i32 normalized [rank](../../glossary.md#term-rank); positive i64 extents in [axis](../../glossary.md#term-axis) order; [ScalarValue](../../glossary.md#typed-scalar-value--scalarvalue) [epsilon](../../glossary.md#epsilon). |
| 4 [Embedding](../../glossary.md#embedding) | i64 vocabularySize; i64 embeddingSize; u8 parameter DataType. |
| 5 Conv1d | i64 inChannels, outChannels, kernelWidth; u8 bias; u8 parameter DataType; i64 [stride](../../glossary.md#term-stride), padding, [dilation](../../glossary.md#term-dilation), groups. |
| 6 [Conv2d](../../glossary.md#convolution--conv2d) | i64 inChannels, outChannels, kernelHeight, kernelWidth; u8 bias; u8 parameter DataType; i64 strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth, groups. |
| 7 [Conv3d](../../glossary.md#conv3d--ncdhw) | i64 inChannels, outChannels, kernelDepth, kernelHeight, kernelWidth; u8 bias; u8 parameter DataType; i64 strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth, dilationDepth, dilationHeight, dilationWidth, groups. |
| 8 [Activation](../../glossary.md#term-activation) | u8 activation ID: 1 RELU, 2 [TANH](../../glossary.md#term-tanh), 3 [SIGMOID](../../glossary.md#term-sigmoid). |

[Feature](../../glossary.md#term-feature)/[channel](../../glossary.md#term-channel)/vocabulary/[kernel](../../glossary.md#kernel)/stride/dilation/group fields are positive [element](../../glossary.md#term-element) counts;
padding is nonnegative elements per side, not total padding. NN's existing geometry, divisibility,
epsilon, element/byte arithmetic and Java-array constraints apply through its public constructors.
DataType IDs are the unchanged [model](../../glossary.md#term-model).v1 assignments: 1 [FLOAT64](../../glossary.md#term-float64), 2 [FLOAT32](../../glossary.md#term-float32), 3 [BFLOAT16](../../glossary.md#term-bfloat16), 4 [INT32](../../glossary.md#data-type--datatype),
5 [INT64](../../glossary.md#data-type--datatype), 6 [BOOL](../../glossary.md#data-type--datatype), 7 [FLOAT16](../../glossary.md#term-float16). Parameter fields permit only 1/2/3/7. No enum ordinal or class name.
ScalarValue is u8 DataType followed by [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) at declared logical width: FLOAT64/INT64 8 bytes,
FLOAT32/INT32 4, BFLOAT16/FLOAT16 2, BOOL 1 (only 0/1). Floating decoders use raw-bit Model
factories (no conversion/[NaN](../../glossary.md#term-nan) [normalization](../../glossary.md#term-normalization)); integral fields are [two's-complement](../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks). LayerNorm's
epsilon must satisfy NN: [finite](../../glossary.md#term-finite) positive, parameter type except FLOAT16 uses FLOAT32 arithmetic.
Thus integral/BOOL/FLOAT16 epsilon encodings are invalid here even though their [scalar](../../glossary.md#term-scalar) [codec](../../glossary.md#term-codec) is
defined. Unknown [dtype](../../glossary.md#data-type--datatype)/activation/node/version values [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation), including all recurrent IDs.
Recurrent adoption needs an explicit future versioned format decision, never a [registry](../../glossary.md#term-registry).

Canonical empty root/no-alias example is 21 bytes:
`53594e4445463031 00000001 01 00000000 00000000` (magic, version, Sequence, children, groups).
It is semantic [identity](../../glossary.md#term-identity-meanings), not evidence of an executable zero-node [backend route](../../glossary.md#backend-route).

Each alias group is positive i32 member count (at least 2), then members as positive i32 [UTF-8](../../glossary.md#term-utf-8)
byte length plus exact strict UTF-8 path bytes. Paths must equal derived numeric dot-separated
state paths without a root prefix or leading-zero child index. No normalization is permitted.
Members follow NN state traversal order, groups ascend by first member; disjoint existing paths
must have identical kind/type/[Shape](../../glossary.md#shape)/[gradient](../../glossary.md#term-gradient) [schema](../../glossary.md#term-schema). Unlisted paths are [singleton](../../glossary.md#term-singleton) identities;
equal bytes never imply sharing. Bound group/member totals by derived state count and remaining
bytes before allocation. Reject duplicate/reordered/unknown paths, malformed UTF-8, noncanonical
flags, count/depth/rank/size/[overflow](../../glossary.md#term-overflow) violations and trailing bytes before model [publication](../../glossary.md#term-publication).

#### Whole-[model](../../glossary.md#term-model) coherence and [publication](../../glossary.md#term-publication)

Outer kind is exactly `whole-model.v1`, with `definition/schema` and every unchanged `model.v1`
entry prefixed `model/` (`model/schema`, `model/state/00000000`, ...); no other entries. Compute
configuration [SHA-256](../../glossary.md#term-sha-256) from exact [canonical](../../glossary.md#term-canonical-form) definition bytes, not caller input. Project the values
component using existing ArtifactEnvelope defensive copies, then use public ModelCheckpoint
[capture](../../glossary.md#term-capture-meanings)/inspect/canonicalStateBytes/load only: no private ordinal parser or component temp files.
Independently validate complete ordered value [descriptors](../../glossary.md#term-descriptor)/lengths against [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-derived [parameter](../../glossary.md#term-parameter)
descriptors (all [gradient](../../glossary.md#term-gradient)-eligible), configuration [digest](../../glossary.md#term-digest), exact expected input presence/bytes,
and byte equality for every declared [alias](../../glossary.md#aliasing-and-arena) group before reconstruction/value installation returns
anything. Do not trust an inspection result as a [validation](../../glossary.md#term-validation) credential. On capture check the same
coherence before returning an envelope. [Hashes](../../glossary.md#term-hash) bind configuration but do not prove [authenticity](../../glossary.md#term-authenticity).

Reconstruct only a private NN descriptor skeleton, then public [target](../../glossary.md#term-target)-load values; return one
independent [heap](../../glossary.md#term-heap)-backed [evaluation](../../glossary.md#term-evaluation) DeclarativeModel only after full validation and installation.
No [RNG](../../glossary.md#graph-rng-state--graphrngstate), throwaway [weights](../../glossary.md#term-weight), [autograd](../../glossary.md#autograd), [compile](../../glossary.md#compile), native [ownership](../../glossary.md#term-ownership-meanings), source mutation or weakened
publication. No [Engine](../../glossary.md#term-engine) is required for load. Semantic empty/[identity](../../glossary.md#term-identity-meanings) trees are allowed with
no Engine call for empty state, not a generic zero-[node](../../glossary.md#node) execution promise. Save reuses existing
ArtifactFile force/revalidate/one-[atomic](../../glossary.md#resource-transaction-and-rollback)-move policy unchanged. Failure exposes no partial model.

A separate optional `extensions/training-checkpoint` adapter is reserved for durable recovery.
Its selected first product is Model + TrainingState only, with a saved-state restoration
guarantee, not complete automatic exact continuation. A concrete downstream need for the current
model-envelope [API](../../glossary.md#term-application-programming-interface-api) is comparison of the complete model parameter descriptors and raw logical
bytes against TrainingState before restore/publication. External input position/cursor, RNG,
[Module](../../glossary.md#term-module) modes, effective forward contexts, scheduler, and scaler remain application-owned.
Broader exact-continuation claims still require all affecting state under the [lifecycle](../../glossary.md#lifecycle) below.
The adapter may depend on Training and [Checkpoint](../../glossary.md#term-checkpoint), but is neither included in
[Gradle](../../glossary.md#term-gradle) settings nor implemented until its later adapter gate. Training keeps [optimizer](../../glossary.md#term-optimizer)/session
state meaning, and the caller and other state [owners](../../glossary.md#term-owner) keep their own validation and restore
authority under the optimizer/training lifecycle below. Neither Training nor Checkpoint gains a
reverse [dependency](../../glossary.md#term-dependency) or a generic [persistence](../../glossary.md#term-persistence) or state-installation API from this reservation.
Model-envelope implementation does not select a training [wire format](../../glossary.md#term-wire-format) or open the Checkpoint 0006
adapter build gate; 0006 remains Draft.


<a id="extensionstraining"></a>

## `modules/training`

Accepted future dynamic external [admission](../../glossary.md#term-admission) composes the same public [Engine](../../glossary.md#term-engine) and one existing
[parameter](../../glossary.md#term-parameter)/[optimizer state](../../glossary.md#term-optimizer-state); it adds no [dependency](../../glossary.md#term-dependency) or execution [owner](../../glossary.md#term-owner). It must qualify independently
after ordinary [CPU](../../glossary.md#term-cpu) forward and actual first-order backward. Parameters/private [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) remain
fixed-shaped. Accumulation is a SUM of supplied [gradients](../../glossary.md#term-gradient), not implicit [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) weighting across
different B. Existing nonempty overlap rejection, parameter/[optimizer](../../glossary.md#term-optimizer) transactions and [finite](../../glossary.md#term-finite)
[objective](../../glossary.md#term-objective)/gradient guards stay: empty-[MEAN](../../glossary.md#term-mean) training rejects transactionally, while a qualified
empty SUM may supply zero gradients. No automatic step skip, masking, sample weighting or
dynamic support claim follows from this authority cutover; current matching remains static.

`modules/training` (unchanged [Gradle](../../glossary.md#term-gradle) project `:extensions:training` and Java packages) owns generic
live parameter admission, optimizer algorithms and training orchestration independently of [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context).

Allowed:

- `Optimizer`
- `Sgd`
- [`Adam`](../../glossary.md#adaptive-optimizer--adam-adamw)
- [`AdamW`](../../glossary.md#adaptive-optimizer--adam-adamw)
- [`ParameterGroup`](../../glossary.md#parameter-group--parametergroup)
- [`TrainingSession`](../../glossary.md#training-session--trainingsession)
- [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters) with [immutable](../../glossary.md#term-immutable) ordered `Entry(path, tensor)` membership and live-owner
  `validateUnchanged()`; Training snapshots and validates admission before [seed](../../glossary.md#term-seed)/[compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)
- `TrainingStep`
- [`TrainingParameterStorage`](../../glossary.md#native-parameter-copy--trainingparameterstorage) for an explicit pre-session copy into caller-owned shareable native [host](../../glossary.md#term-host) memory
- immutable in-memory optimizer/session state snapshots
- an explicit dependency on the public Engine [facade](../../glossary.md#term-facade) for the session [lifecycle](../../glossary.md#lifecycle)
- fresh stored external values through `TrainingSession.runValues`, reusing public [`Engine.runValues`](../../glossary.md#fresh-value-binding--enginerunvalues)
- [backend](../../glossary.md#backend)-neutral access to caller-[owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) [host storage](../../glossary.md#host-storage) within the supported optimizer domain

Forbidden:

- dependency on [Runtime](../../glossary.md#term-runtime), Prepare, tuning, or concrete backends
- use of non-public Engine implementation state
- a second [compiler](../../glossary.md#term-compiler), preparer, scheduler, runner, [publication](../../glossary.md#term-publication) [model](../../glossary.md#term-model), or backend adapter
- `MetalOptimizerBridge`
- `CudaOptimizerBridge`
- `CpuOptimizerBridge`
- backend-specific optimizer execution
- backend [kernel](../../glossary.md#kernel) selection

Training has exactly `api` project dependencies on `modules/model` and `modules/engine`, with no
NN dependency, import, NN-taking compatibility overload, or test-only back-edge. Its public
[signatures](../../glossary.md#term-signature) name `TrainingParameters`, [`Tensor`](../../glossary.md#tensor), [`DataType`](../../glossary.md#data-type--datatype), [`Shape`](../../glossary.md#shape), Engine and Engine [metadata](../../glossary.md#term-metadata).
NN consumes Training's live admission, explicit native copy and TrainingSession APIs. The public Engine edge is the explicit
Class C composition edge needed to reuse one
compile/Prepare/Runtime path; it does not transfer optimizer [ownership](../../glossary.md#term-ownership-meanings) into Engine or authorize
Training to inspect backend selection or representations. `train()` and `eval()` mode are [module](../../glossary.md#term-module)
forward-behavior concerns owned by `extensions/nn`; an optimizer neither selects nor changes that
mode.

Training owns backend-neutral optimizer mathematics over supported caller-owned host parameter
storage, not backend-specific optimizer execution. A later [fused](../../glossary.md#specialization-and-fusion) optimizer route belongs to
backend prepare and [kernels](../../glossary.md#kernel) and requires a separate architecture decision.

## `extensions/checkpoint`

`modules/checkpoint` retains [Gradle](../../glossary.md#term-gradle) [identity](../../glossary.md#term-identity-meanings) `:extensions:checkpoint` via settings mapping.
Its [artifact](../../glossary.md#term-artifact) package owns a version-one, single-regular-file envelope for [opaque](../../glossary.md#term-opaque)
bytes only; that package assigns no [model](../../glossary.md#term-model) meaning to a [payload](../../glossary.md#artifact-envelope-and-atomic-publication). The
[canonical](../../glossary.md#term-canonical-form) [big-endian](../../glossary.md#text-encoding-and-file-boundaries) file has a fixed magic/version header, [SHA-256](../../glossary.md#term-sha-256) [manifest](../../glossary.md#term-manifest) [digest](../../glossary.md#term-digest), sorted strict
[ASCII](../../glossary.md#term-ascii) names, explicit bounded lengths, and SHA-256 per-payload digests. Readers reject malformed,
noncanonical, unsupported, over-limit, truncated, or trailing data before returning a [detached](../../glossary.md#detached-model-results)
[immutable](../../glossary.md#term-immutable) image. Callers supply bounds for file and manifest bytes, entry count, name bytes,
individual payload bytes, and aggregate payload bytes. Readers bound the manifest length before
allocating its array, then check aggregate length arithmetic before allocating payloads.
It has exactly `api(Model)` and `implementation(Compiler)` project dependencies. [Compiler](../../glossary.md#term-compiler) access
uses only GraphCompilationPort's cold [capture](../../glossary.md#term-capture-meanings)/reconstruction boundary. No Compiler type appears
in public GraphCheckpoint [signatures](../../glossary.md#term-signature). No [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training, [Engine](../../glossary.md#term-engine), [Runtime](../../glossary.md#term-runtime), [Prepare](../../glossary.md#prepare), tuning or [backend](../../glossary.md#backend)
[dependency](../../glossary.md#term-dependency) is permitted, including tests. NN alone consumes both generic [Checkpoint](../../glossary.md#term-checkpoint) and Engine
for [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) composition and the relocated old [persistence](../../glossary.md#term-persistence) APIs; other inward [modules](../../glossary.md#term-module) do not depend on Checkpoint.

### Generic [graph](../../glossary.md#graph) source and values

The user-authorized [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) completion extends the existing fluent [Tensor](../../glossary.md#tensor) [Model](../../glossary.md#term-model) builder and cold
graph source boundary with ordinary maximum and average Pool3d expressions. Their mathematical
and [gradient](../../glossary.md#term-gradient) [owners](../../glossary.md#term-owner) remain Model and [Compiler](../../glossary.md#term-compiler). A new append-only graph-model.v6 wire admits these
two exact typed [operation](../../glossary.md#operation) kinds/attributes, including qualified body placement through the existing
bounded [Scan](../../glossary.md#term-cumulative-scan) protocol. Preserve every already accepted v1-v5 file and choose the smallest sufficient
format for sources without the new operations. A v6 marker must not admit unrelated constructors,
change stored roles/[aliases](../../glossary.md#aliasing-and-arena)/payloads, restore [optimizer](../../glossary.md#term-optimizer)/[runtime](../../glossary.md#term-runtime) state or bypass cold [metadata](../../glossary.md#term-metadata) checks.

Ordinary symbolic-B [Engine](../../glossary.md#term-engine) execution grants no cold acquisition or graph wire support.
Current static source/copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction)/save/load [admission](../../glossary.md#term-admission) below remains unchanged; symbolic
Compiler [cold reconstruction](../../glossary.md#term-cold-reconstruction)/[Checkpoint](../../glossary.md#term-checkpoint) copy and symbolic wire [qualification](../../glossary.md#term-qualification) are separate gates.
Old static saved Models/readers remain supported. None is a prerequisite for the ordinary
low-level Engine+[CPU](../../glossary.md#term-cpu) dynamic vertical; no new Checkpoint [dependency](../../glossary.md#term-dependency) or reconstruction [owner](../../glossary.md#term-owner) is added.

General-scan **in-memory** copy/rebind is implemented for the exact
[bounded Compiler cold inventory](compiler-autograd.md#bounded-general-scan-cold-reconstruction),
not every Model-valid [PURE](../../glossary.md#term-pure) body. [Normalization](../../glossary.md#term-normalization) retains each [immutable](../../glossary.md#term-immutable) CompiledScanBody attachment
unchanged while remapping only outer [node](../../glossary.md#node)/value IDs. Body IDs and ordered role [bindings](../../glossary.md#term-binding) remain
local to that attachment, even when their numbers coincide with outer IDs. Compiler alone
preflights and reconstructs fresh formals and one body per [occurrence](../../glossary.md#term-occurrence), including unused formals,
all [canonical](../../glossary.md#term-canonical-form) ports and latent residuals; no reconstruction table or body copier belongs in
Checkpoint. Applicable Model [dtypes](../../glossary.md#data-type--datatype) remain independent of [backend capability](../../glossary.md#backend-capability).
Compiler0018 implements and qualifies the same in-memory path for the static [homogeneous](../../glossary.md#term-homogeneous)
[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) length-mode cold intersection. Checkpoint retains ordinary copy/rebind semantics;
no extra Compiler flag or provider qualification gate is introduced. Existing [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) graph/definition
acquisition and explicit cold forward inherit this accepted admission. Cold qualification alone
is not evidence of complete numerical execution or training.

Implemented [graph-model v4](#canonical-graph-model-version-4) persists the pre-Pool3d all-valid
portion of that bounded cold inventory. [V5](#canonical-graph-model-version-5) adds its static
homogeneous length intersection under Checkpoint0014; [v6](#canonical-graph-model-version-6)
adds Pool3d in either [scope](../../glossary.md#term-scope). None changes v1/v2/v3's body-free bytes.
Unsupported bodies reject before [host](../../glossary.md#term-host) admission/copy or file effects.
Model constructors still reject malformed/missing attachments. NN's [owned workflow](#owned-tensor-models) consumes the same cold boundary for
acquisition and [persistence](../../glossary.md#term-persistence) without adding an execution or reconstruction owner.

GraphCheckpoint is a final cold [facade](../../glossary.md#term-facade) with only nested Source(inputs, outputs, leaves),
[Leaf](../../glossary.md#term-leaf)(name, role, tensor), Role { [PARAMETER](../../glossary.md#term-parameter), [BUFFER](../../glossary.md#term-buffer), [CONSTANT](../../glossary.md#term-constant) }, and copy(Source),
save(Path, Source, ArtifactLimits), load(Path, ArtifactLimits), rebind(Source, List<Tensor>).
Save/load declare IOException. Records [snapshot](../../glossary.md#term-snapshot) lists and null-check in declaration/list order;
Tensors are [borrowed](../../glossary.md#ownership-borrowing-and-lease) ordinary model objects, not another graph, executor or closeable owner.
Operations revalidate. Inputs are ordered unique [provenance](../../glossary.md#provenance)-free [prototypes](../../glossary.md#term-prototype), possibly unused or
empty. Outputs are nonempty and distinct. Named provenance-free state is disjoint from inputs;
every reachable non-input leaf is named and every named [identity](../../glossary.md#term-identity-meanings) is reachable. Names use strict
[UTF-8](../../glossary.md#term-utf-8), nonblank dot segments without normalization, duplicates or prefix collisions; registration
order is retained. Aliases [mean](../../glossary.md#term-mean) exact Tensor [identity](../../glossary.md#term-reference-identity) and identical role, not equal bytes or
physical overlap. PARAMETER requires floating gradient eligibility. BUFFER and CONSTANT preserve
requiresGrad without inferring roles.

[Capture](../../glossary.md#term-capture-meanings) uses Compiler captureSource exactly once and associates its first-encounter inputs by
exact Tensor identity, never by declared position. Existing immutable Model graph records normalize
file-local IDs and retain unused [signature](../../glossary.md#term-signature) slots; no second graph representation or Tensor
reconstruction table belongs here. Compiler retains source admission and call-local reconstruction
[ownership](../../glossary.md#term-ownership-meanings) under the exact [closed source slice](compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([slice glossary](../../glossary.md#slice))</span>, without
optimization or [layout](../../glossary.md#layout) closure. Preserve full [descriptors](../../glossary.md#term-descriptor), occurrences, sharing
and ordered roots; all nine dtypes where Model permits (no integral/[BOOL](../../glossary.md#data-type--datatype) NEG). Zero-node data
is not new Engine capability. Rebind checks metadata then replacement count/full descriptors,
retains named state and permits produced/repeated substitutions with [opaque](../../glossary.md#term-opaque) upstream provenance;
its immutable ordinary outputs may repeat, which ordinary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) still rejects.

Normalization assigns ordered ValueIds to every canonical [producer](../../glossary.md#term-producer) output, including unconsumed
sibling ports; [NodeId](../../glossary.md#nodeid) and [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot) are not interchangeable. Only Compiler reconstructs Tensors,
calling each family constructor once and mapping all returned components. GraphCheckpoint retains
no ValueId-to-Tensor table or result-[carrier](../../glossary.md#term-storage-carrier) objects. Capture/[preflight](../../glossary.md#term-preflight)/rebind inspect no storage,
including [GATHER](../../glossary.md#gather) indices and scan valid lengths. Runtime lengths may be external signature inputs;
their prototype bytes never enter the [schema](../../glossary.md#term-schema) or payloads. Copy/save read only named host values.

Copy/save read named identities only. Check static canonical geometry even for unresolved layouts;
present leaf layouts must equal zero-offset non-[view](../../glossary.md#term-view-meanings) [contiguous](../../glossary.md#term-contiguous) geometry. Intermediate views are
metadata. Snapshot each hostStorage association once; admit every state's type, capacity, [liveness](../../glossary.md#term-liveness)
and thread access, including empty state, before BOOL checks or bulk copying. Read-only [heap](../../glossary.md#term-heap)/native
segments are allowed; absent/[device](../../glossary.md#term-device)-only/dead/inaccessible/short state rejects. Caller excludes
mutation, rebinding and closure; no lock, hidden execution or ownership transfer. Read native-order
unaligned integer [lanes](../../glossary.md#term-lane) at widths 8 (FLOAT64/[INT64](../../glossary.md#data-type--datatype)), 4 (FLOAT32/[INT32](../../glossary.md#data-type--datatype)), 2 ([BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16)), 1
(BOOL). Preserve all [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) including [signed zero](../../glossary.md#term-signed-zero) and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign; BOOL must be 0/1. Ignore
excess capacity and copy distinct overlapping Tensors independently. Use matching primitive heap
carriers and MemorySegmentStorage with TensorFactory.create retaining the original descriptor,
not layout-closing allocate. Copy creates storage-free prototypes and copies each state identity
once, then delegates reconstruction. Load adopts final heap state without a second ownership copy,
original graph/state or saved input storage. Labels/native owners are not persisted.

Copy/save order: metadata, checked geometry and (save) encoded bounds, all storage admission,
BOOL/raw copy, complete result/image, then ArtifactFile.save. Source/unsupported/bounds failures are
IllegalArgumentException; sizing [overflow](../../glossary.md#term-overflow) is ArithmeticException (retain Compiler wrapping),
absent/dead/inaccessible state is IllegalStateException. Rebind has no value gate. Load performs
ArtifactFile.read, complete wire/keys/descriptors/aliases/lengths/BOOL [validation](../../glossary.md#term-validation), descriptor-only
candidate leaves and Compiler SourceCapture preflight, host import, reconstructSource, [publication](../../glossary.md#term-publication).
Stored semantic/overflow errors become ArtifactFormatException with cause; I/O, identifier
exhaustion and VM failures propagate. Never publish partial candidates or mutate source bindings.

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 1

ArtifactFile v1 kind is graph-model.v1: exactly [schema](../../glossary.md#term-schema) plus state/00000000… per unique named
[identity](../../glossary.md#term-identity-meanings). [Big-endian](../../glossary.md#text-encoding-and-file-boundaries) signed i32/i64 and unsigned u8, nonnegative counts, flags exactly 0/1,
no [padding](../../glossary.md#padding), extensions or trailing bytes. Fixed IDs never use enum ordinals or class names.

| Schema order | Exact fields |
|---|---|
| Header | Eight [ASCII](../../glossary.md#term-ascii) bytes SYNGRF01, i32 version 1. |
| Inputs | i32 I, I [descriptors](../../glossary.md#term-descriptor) in declared order, including unused slots; ValueIds 0… I−1, no [payload](../../glossary.md#artifact-envelope-and-atomic-publication). |
| State | i32 S, then descriptor, u8 role (1 [PARAMETER](../../glossary.md#term-parameter), 2 [BUFFER](../../glossary.md#term-buffer), 3 [CONSTANT](../../glossary.md#term-constant)), i64 byte length per first named identity; ValueIds I… I+S−1, payload ordinal is state index. |
| Registrations | i32 A, then positive i32 strict [UTF-8](../../glossary.md#term-utf-8) name length, exact bytes, i32 state index in registration order; all states referenced and first new indices are 0,1,…; repeats are [aliases](../../glossary.md#aliasing-and-arena). |
| [Nodes](../../glossary.md#node) | i32 N, then u8 phase=1, u8 op (1 ADD, 2 MUL, 3 NEG, 4 [MATMUL](../../glossary.md#matrix-multiplication--matmul), 5 PERMUTE), fixed-[arity](../../glossary.md#term-arity) i32 preceding input ValueIds (two for ADD/MUL/MATMUL, one otherwise); PERMUTE adds i32 2, i32 1, i32 0; output descriptor. [NodeId](../../glossary.md#nodeid) n, output [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot) I+S+n. |
| Outputs | Positive i32 O and O distinct ordered i32 ValueIds. |
| Descriptor | u8 [dtype](../../glossary.md#data-type--datatype) (1 [FLOAT64](../../glossary.md#term-float64), 2 [FLOAT32](../../glossary.md#term-float32), 3 [BFLOAT16](../../glossary.md#term-bfloat16), 4 [INT32](../../glossary.md#data-type--datatype), 5 [INT64](../../glossary.md#data-type--datatype), 6 [BOOL](../../glossary.md#data-type--datatype), 7 [FLOAT16](../../glossary.md#term-float16)), u8 requiresGrad, i32 [rank](../../glossary.md#term-rank), rank nonnegative i64 extents, u8 [layout](../../glossary.md#layout)-present; when present, i64 [element](../../glossary.md#term-element) offset, u8 [view](../../glossary.md#term-view-meanings), rank nonnegative i64 [strides](../../glossary.md#term-stride); derive kind/span via LayoutDescriptor.of. |

Reject forward/dangling/cyclic references, dead nodes, unreachable state, noncanonical aliases,
unknown IDs/phases/attrs, keys, lengths and BOOL. [Compiler](../../glossary.md#term-compiler) owns family/descriptor [validation](../../glossary.md#term-validation);
normalized recapture must match stored [topology](../../glossary.md#term-topology) in [capture](../../glossary.md#term-capture-meanings) order, retaining unused inputs.
Payloads are exact [row-major](../../glossary.md#term-row-major) bits, [shape](../../glossary.md#shape)-count × width; no TensorIds, [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters) or [runtime](../../glossary.md#term-runtime) state.
Save/load require explicit ArtifactLimits, with no new defaults or restriction of old [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) trees.
Bound schema by maxPayloadBytes (≤Integer.MAX_VALUE−8), all counts/ranks/strings by remaining bytes
before allocation, S≤100000000, S+1≤maxEntries and checked I+S+N≤Integer.MAX_VALUE. Traversal is
iterative with no [backend](../../glossary.md#backend) rank-16 bound. For schema D and unique state bytes B: [manifest](../../glossary.md#term-manifest)
M=72+58*S, aggregate=D+B, file=48+M+D+B, entries=1+S, kind/key ceiling≥14. Check all six limits
and individual payloads with checked sums before encoding or bulk copies. These bound encoded
size, not peak [heap](../../glossary.md#term-heap). Copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) have no implicit file ceiling; checked collection/geometry/[JVM](../../glossary.md#term-java-virtual-machine-jvm)
limits apply and copy preflights each unique payload≤Integer.MAX_VALUE−8. Complete save image
validation precedes unchanged ArtifactFile force/revalidate/one-[atomic](../../glossary.md#resource-transaction-and-rollback)-move [publication](../../glossary.md#term-publication) and its
cleanup/failure semantics; this is not [authentication](../../glossary.md#term-authentication) or portable power-[loss](../../glossary.md#loss) [durability](../../glossary.md#term-durability).

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 2

Within the original eight-kind [slice](../../glossary.md#slice), save selects the smallest sufficient closed version:
sources containing any RELU, [TANH](../../glossary.md#term-tanh) or [SIGMOID](../../glossary.md#term-sigmoid)
write ArtifactFile v1 kind `graph-model.v2`, eight [ASCII](../../glossary.md#term-ascii) bytes `SYNGRF02` and i32 version 2.
Other sources within that slice, including zero-[node](../../glossary.md#node) sources, write byte-identical version 1.
The [version-3 policy](#canonical-graph-model-version-3) governs sources using added kinds,
including when combined with [activations](../../glossary.md#term-activation). Version 2 retains
every version-1 field, order, [descriptor](../../glossary.md#term-descriptor)/role/[dtype](../../glossary.md#data-type--datatype) ID, [payload](../../glossary.md#artifact-envelope-and-atomic-publication) and limit above, appending only
u8 opcodes 6=RELU, 7=TANH and 8=SIGMOID. Each has one preceding input [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot) and no attribute
bytes; only `NoOperationAttrs.INSTANCE` is admitted. Encoder [dry sizing](../../glossary.md#term-dry-sizing) explicitly validates every
kind, attributes and fixed [arity](../../glossary.md#term-arity), rejecting unknown operations rather than defaulting to PERMUTE.

For these two versions, load accepts only their exact kind/magic/version tuples. Version 1 still admits
only opcodes 1..5; version 2 admits only 1..8. Version 2 may contain only the original five kinds;
re-saving that source selects version 1. Reject mismatched tuples, unknown versions/opcodes and
malformed fields before [host](../../glossary.md#term-host) import, reconstruction or [publication](../../glossary.md#term-publication). New readers retain every valid
version-1 file; old graph readers [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) on version 2. The equal-length kind strings leave
[manifest](../../glossary.md#term-manifest)/file sizing unchanged. ArtifactFile [framing](../../glossary.md#term-framing) and the separate model.v1/whole-model.v1
readers remain unchanged; no public version selector, [registry](../../glossary.md#term-registry) or enum-ordinal encoding is added.

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 3

Version 3 is one closed successor for the selected [CNN](../../glossary.md#term-convolutional-neural-network-cnn)/static recurrent source [slice](../../glossary.md#slice) and fixed
flat scans. It uses ArtifactFile v1 kind `graph-model.v3`, eight [ASCII](../../glossary.md#term-ascii) bytes `SYNGRF03`, and i32
version 3. No intermediate one-output v3 or separate [scan](../../glossary.md#term-cumulative-scan) format is defined. It retains the
[version-1](#canonical-graph-model-version-1) section order, input/state/registration/output
fields, [descriptor](../../glossary.md#term-descriptor)/role/[dtype](../../glossary.md#data-type--datatype) IDs, raw payloads and envelope [framing](../../glossary.md#term-framing). The [node](../../glossary.md#node) section alone
changes as specified below; all integers remain [big-endian](../../glossary.md#text-encoding-and-file-boundaries) with no [padding](../../glossary.md#padding) or trailing fields.

In version 3, scan wire records [mean](../../glossary.md#term-mean) only the existing fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) family. [General scan](../../glossary.md#general-differentiable-scan)
source/compiled body attachments and [residual](../../glossary.md#scan-residual-port-and-history-demand) protocols are not encodable by v1/v2/v3. Low-level
[AD](../../glossary.md#autograd) or in-memory cold-copy acceptance grants no new [opcode](../../glossary.md#term-opcode), field, version or Model.save/load
capability. The wire inventory stays closed independently of [Compiler](../../glossary.md#term-compiler) source [admission](../../glossary.md#term-admission).

| Node record order | Exact fields |
|---|---|
| Prefix | u8 phase=1, u8 opcode. Explicit opcodes 1..8 retain their version-1/2 meanings. |
| Inputs | For CONCAT and scans only, i32 input count immediately after opcode; then that many i32 input ValueIds. All other kinds use their fixed [arity](../../glossary.md#term-arity) without a count. |
| Attributes | The exact fields below, after input IDs; existing PERMUTE remains i32 2, i32 1, i32 0. Kinds 1..4 and 6..9 have no attribute bytes. |
| Results | Positive i32 Q, followed by Q ordered output descriptors in canonical [producer](../../glossary.md#term-producer)-port order. Q=1 for opcodes 1..19, Q=2 for 20/21, Q=3 for 22. No output IDs are serialized here. |

For input count I and unique state count S, [NodeId](../../glossary.md#nodeid) n has first output
`I+S+sum(Q[k], k<n)` and [output port](../../glossary.md#capture-graph-ports-and-reconstruction) j has [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot) `I+S+sum(Q[k], k<n)+j`, for `0<=j<Q[n]`.
IDs are i32; every input reference must precede its node's first output. Check cumulative
`I+S+sum(Q) <= Integer.MAX_VALUE`, not merely `I+S+N`. Inputs, state and implicit output ranges
define each value once. Repeated input references and repeated port uses are valid; duplicate
stored roots, self/forward/dangling references and duplicate value definitions are not.

The following table defines the appended wire fields. The
[Compiler source contract](compiler-autograd.md#compile-lifecycle) owns the exact narrow semantic
admission and public-constructor reconstruction; encoding independently rejects any kind,
attributes or input/output count that does not satisfy it.

| Opcode and kind | Input count | Attribute fields after input IDs |
|---|---|---|
| 9 SUB | 2 | None. |
| 10 RESHAPE | 1 | i32 [target](../../glossary.md#term-target) [rank](../../glossary.md#term-rank), then rank i64 target extents. |
| 11 EXPAND_DIMS / 12 SQUEEZE | 1 | i32 [axis](../../glossary.md#term-axis). |
| 13 SELECT | 1 | i32 axis, i64 index. |
| 14 [GATHER](../../glossary.md#gather) | 2 | i32 axis. |
| 15 SLICE | 1 | i32 entry count, then ordered entries `(i64 start, i64 length, i32 axis, i64 step)`. The selected slice has exactly one entry. |
| 16 CONCAT | Positive [variadic](../../glossary.md#term-variadic) count | i32 axis; the count precedes input IDs, not attributes. |
| 17 [CONV2D](../../glossary.md#convolution--conv2d) | 3 | Seven i64 fields: strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth, groups; admitted tuple `(1,1,0,0,1,1,1)`. |
| 18 [CONV3D](../../glossary.md#conv3d--ncdhw) | 3 | Ten i64 fields: strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth, dilationDepth, dilationHeight, dilationWidth, groups; admitted tuple `(1,1,1,0,0,0,1,1,1,1)`. |
| 19 MAX_POOL2D | 1 | Eight i64 fields: kernelHeight, kernelWidth, strideHeight, strideWidth, paddingHeight, paddingWidth, dilationHeight, dilationWidth; then u8 ceilMode=0. |
| 20 RNN_TANH | 5 or 6 | u8 direction: 1 FORWARD, 2 REVERSE. |
| 21 GRU_RESET_AFTER | 5 or 6 | u8 direction: 1 FORWARD, 2 REVERSE. |
| 22 LSTM | 6 or 7 | u8 direction: 1 FORWARD, 2 REVERSE. |

Scan input order is input, validLengths, initialHidden, then inputWeight and hiddenWeight for
RNN/GRU; LSTM inserts initialCell before the [weights](../../glossary.md#term-weight). An optional [bias](../../glossary.md#term-bias) is last. Input count alone
selects the overload; there is no bias flag, length value, loop count or body attribute. Outputs
are outputs/finalHidden and, for LSTM, finalCell. The
[fixed recurrent descriptor semantics](recurrent-scan.md#static-shape-and-runtime-value-boundary)
remain unchanged, including unresolved [layouts](../../glossary.md#layout), floating-role OR [gradient](../../glossary.md#term-gradient) eligibility and permitted
zero time/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). Valid-length bounds `[0,T]` are execution checks, not cold [preflight](../../glossary.md#term-preflight) checks.

Canonical [topology](../../glossary.md#term-topology) uses a call-local value-to-producer/port index and iterative producer-level
depth-first postorder, never Java-stack recursion or value-minus-boundary node arithmetic. Walk
ordered roots and ordered operands in [capture](../../glossary.md#term-capture-meanings) order. Visiting any port visits its producer once
and retains all sibling descriptors, including unused siblings. Reject dead producers and
unreachable state, not unused sibling ports; retain declared unused [signature](../../glossary.md#term-signature) inputs. Preserve
distinct occurrences and normalized recapture equality before returning a loaded source. For
example, a graph rooted only at LSTM finalCell retains one producer and all three output
descriptors; it neither creates three scans nor treats the other two ports as dead nodes.

For body-free sources, select the smallest sufficient version: zero-node and original five-kind sources retain
byte-identical v1; sources using only the original eight kinds and at least one [activation](../../glossary.md#term-activation) retain
byte-identical v2; any opcode 9..22 selects v3. Load accepts only exact kind/magic/version tuples:
v1 admits 1..5, v2 1..8, v3 1..22 with their own exact records. A v3 containing only older kinds
is valid and re-saves as v1/v2. Unknown versions, opcodes, directions, attributes or cardinalities
[fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation). Old readers reject v3. No published v1/v2 field changes, public version selector,
new [API](../../glossary.md#term-application-programming-interface-api)/type/[module](../../glossary.md#term-module)/[registry](../../glossary.md#term-registry), enum-ordinal wire or implicit [operation](../../glossary.md#operation) [decomposition](../../glossary.md#term-decomposition) is permitted.
Separate `model.v1`/`whole-model.v1` readers remain unchanged.

All prior raw-bit, role, [alias](../../glossary.md#aliasing-and-arena), geometry, storage and [atomic](../../glossary.md#resource-transaction-and-rollback)-[publication](../../glossary.md#term-publication) rules apply. Bound every
count, rank, string and cumulative value count by remaining bytes and checked [JVM](../../glossary.md#term-java-virtual-machine-jvm) arithmetic before
allocation. The dry Sink traverses exactly the fields used by encoding, including Q and every
descriptor, before [schema](../../glossary.md#term-schema) allocation, [host](../../glossary.md#term-host) admission or copying. For schema bytes D (including
all result descriptors), unique state bytes B and state count S, retain `M=72+58*S`,
`aggregate=D+B`, `file=48+M+D+B`, `entries=1+S` and the 14-byte kind/key ceiling. All six limits
and individual [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bounds precede state admission/copy. Complete wire and descriptor-only
source preflight, including late-invalid and dead-producer rejection, precede host import and
producer reconstruction; normalized recapture precedes publication. Failure returns no partial
source, mutates no source [binding](../../glossary.md#term-binding) and retains the existing ArtifactFile target/cleanup semantics.

This is cold semantic graph [persistence](../../glossary.md#term-persistence), not serialized CompileArtifacts, [prepared execution](../../glossary.md#prepared-execution--preparedexecution),
native/[backend](../../glossary.md#backend)/[runtime](../../glossary.md#term-runtime) state or saved input batches. It itself grants no execution or gradients,
[optimizer](../../glossary.md#term-optimizer) [resume](../../glossary.md#schema-serialization-and-checkpoint) or automatic [exact training continuation](../../glossary.md#exact-training-continuation); independent current Compiler/backend
[qualification](../../glossary.md#term-qualification) determines whether a reconstructed graph may train.

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 4

This successor persists ordinary general [`Scan`](../../glossary.md#term-cumulative-scan) source and named raw values, not a
compiled executable or [runtime](../../glossary.md#term-runtime) history. The [codec](../../glossary.md#term-codec) is implemented with focused wire, cold-source
and consumer evidence; main accepted final [module](../../glossary.md#term-module)/[Javadoc](../../glossary.md#term-javadoc), integration, architecture and configured
root gates in [CP0013](../../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md#result).
It adds no public [API](../../glossary.md#term-application-programming-interface-api), production type, [dependency](../../glossary.md#term-dependency), ArtifactLimits default or execution grant.
[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s existing `Model.save`/`load`/`loadGraph` delegate to this same cold graph boundary. Save before
compute or after training uses current owned values; load creates no [Engine](../../glossary.md#term-engine) and adopts final [heap](../../glossary.md#term-heap)
state without another [ownership](../../glossary.md#term-ownership-meanings) copy. Objectives/[optimizer state](../../glossary.md#term-optimizer-state) must be configured anew, not restored.

Outer v4 [admission](../../glossary.md#term-admission) is exactly version 3 plus all-valid [general SCAN](../../glossary.md#general-differentiable-scan). Contained body admission is
the pre-[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) subset of the
[bounded Compiler cold inventory](compiler-autograd.md#bounded-general-scan-cold-reconstruction):
flat constructors minus scans and Pool3d, with body-only arithmetic, SUM, [layout](../../glossary.md#layout) and same-[dtype](../../glossary.md#data-type--datatype) CAST
overrides/additions, opcodes at most32 and unchanged narrow indexing/[CNN](../../glossary.md#term-convolutional-neural-network-cnn)/[MATMUL](../../glossary.md#matrix-multiplication--matmul) guards. This is neither an [RNN](../../glossary.md#term-recurrent-neural-network-rnn)-only
subset nor arbitrary [PURE](../../glossary.md#term-pure) [serialization](../../glossary.md#term-serialization). Applicable low/integral/[BOOL](../../glossary.md#data-type--datatype) Model [signatures](../../glossary.md#term-signature) remain
eligible independently of [CPU](../../glossary.md#term-cpu) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) execution; cross-dtype CAST remains unsupported.
[Compiler](../../glossary.md#term-compiler) retains all family [inference](../../glossary.md#term-inference) and constructor ownership; Scan semantics remain with
[their sole owner](recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>.

ArtifactFile v1 kind is `graph-model.v4`, with eight [ASCII](../../glossary.md#term-ascii) bytes `SYNGRF04` and i32 version 4.
Retain version-3 section order, [descriptors](../../glossary.md#term-descriptor), registrations, raw state payloads and opcodes 1..22.
All-valid general SCAN without Pool3d or length mode selects v4; otherwise sources needing neither
general Scan nor Pool3d select the smallest sufficient byte-identical old version. Length mode
without Pool3d requires v5; Pool3d in either [scope](../../glossary.md#term-scope) requires v6.
Accept exact kind/magic/version tuples only: v1/v2/v3 keep their own exact records and
closed [opcode](../../glossary.md#term-opcode) sets. Old readers reject v4. A valid body-free v4 re-saves as v1/v2/v3 as appropriate.
No old field mutation, public selector, enum-ordinal encoding, class name or [registry](../../glossary.md#term-registry) is permitted.

The following records extend the v3 [node](../../glossary.md#node) format; body-local IDs never resolve in an outer scope.

| Record | Fields in wire order and interpretation |
|---|---|
| General SCAN | u8 phase=1, u8 opcode=23, i32 actual count and ordered outer input IDs; ScanAttrs fields i64 T, i32 [carry](../../glossary.md#term-carry)/sequence/[invariant](../../glossary.md#term-invariant)/emission/[residual](../../glossary.md#scan-residual-port-and-history-demand) counts, u8 reverse; inline body; positive i32 Q and all canonical outer output descriptors. Only opcode 23 has a body. |
| Inline body | i32 formal count and ordered descriptors (carry, step, invariant); i32 node count and ordinary v3-style node records; count-prefixed unique graph roots and ordered nextCarry/emissions/residual local-ID lists; i32 demand count=0. No state, nested body or cross-scope reference. |
| Implicit IDs | For J body formals, input IDs 0..J−1, NodeIds 0..N−1, output IDs J+sum(previous Q)+port. Outer IDs retain the I+S boundary. Actual-input order is not local numeric [identity](../../glossary.md#term-identity-meanings). Retain unused formals and every canonical sibling descriptor. |
| Body-only opcodes | Append 24=[tensor](../../glossary.md#tensor) DIV; 25/26/27/28=[scalar](../../glossary.md#term-scalar) ADD/SUB/MUL/DIV; 29=SUM; 30=EXPAND; 31=[CONTIGUOUS](../../glossary.md#term-contiguous); 32=same-dtype CAST. [Arity](../../glossary.md#term-arity) 2 for 24, otherwise 1; all Q=1. Forbid 24..32 outside bodies and 20..23 inside bodies. |
| Added attributes | DIV/CONTIGUOUS: none. Scalar: u8 dtype and raw width bits. SUM: u8 tag 0=none, 1=i32 [axis](../../glossary.md#term-axis)+u8 keep, 2=i32 count+ordered i32 axes+u8 keep, 3=[target](../../glossary.md#term-target) [shape](../../glossary.md#shape). EXPAND: target shape. CAST: u8 dtype. Shape: i32 [rank](../../glossary.md#term-rank) and i64 extents. |

All integers/[raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) remain [big-endian](../../glossary.md#text-encoding-and-file-boundaries), dtype IDs remain 1..7 and flags exactly 0/1. Existing
attribute fields apply to body opcodes 1..19; body 9/10/11/12 retain their fields with Compiler's
cold-body semantic overrides. Other mappings retain old guards. Unknown opcodes/tags/flags/attrs,
[derivative](../../glossary.md#term-derivative)-only attributes, nonempty demand, [padding](../../glossary.md#padding) and trailing bytes reject.

Apply the existing [producer](../../glossary.md#term-producer)-aware canonical [topology](../../glossary.md#term-topology) independently in each scope, retaining unused
formals/sibling ports but rejecting dead producers; neither flatten scopes nor merge occurrences.

Ordered role lists may repeat IDs. Unique graph roots must be exactly the first-[occurrence](../../glossary.md#term-occurrence) union
of nextCarry, emissions and residuals, in that order. Preserve sharing, tied actual positions,
distinct occurrences and distinct canonical outer wrappers even for tied body roles. Residual
[bindings](../../glossary.md#term-binding)/descriptors are required source [metadata](../../glossary.md#term-metadata), not saved history payloads. Validate their
derived order with existing Compiler rules; never trim latent ports or persist post-[compile](../../glossary.md#compile) demand.
Retain accepted cold residual edges/roots without [widening](../../glossary.md#term-widening) public derivative/residual-root [AD](../../glossary.md#autograd).

There is one inline body per SCAN, at most one per outer node, depth one. Before count-proportional
allocation, bound every count/rank/arity/root/role/residual array by remaining [schema](../../glossary.md#term-schema) bytes and
checked [JVM](../../glossary.md#term-java-virtual-machine-jvm) arithmetic, not a [backend](../../glossary.md#backend) rank-16 ceiling. For header counts c/s/i/e/r, check formal
count=c+s+i, Q=c+e+r, nextCarry=c, emissions=e and residuals=r (canonical unique inventory).
Zero carry counts are valid when the remaining Model roles and nonempty [publication](../../glossary.md#term-publication) inventory
qualify; do not invent a carry or force time [unrolling](../../glossary.md#term-unrolling) to serialize a carry-free body.
Unique roots cannot exceed local values or c+e+r. Each scope's boundary+sum(Q) and contained
aggregate formal/node/value/role-reference counts must fit signed i32 budgets. Validate static
geometry and every output descriptor, including unused siblings and zero-time bodies.

Dry Sink must count every body field before schema allocation, [host](../../glossary.md#term-host) admission or copying.
Schema D includes all inline bodies and remains within maxPayloadBytes and the existing JVM ceiling.
For unique state bytes B and count S retain M=72+58*S, aggregate=D+B, file=48+M+D+B, entries=1+S
and kind/key ceiling≥14; all six ArtifactLimits and individual [payload](../../glossary.md#artifact-envelope-and-atomic-publication) bounds apply. These bound
encoded size, not peak heap. No time-unrolling allowance or new limits/defaults are introduced.

Decode complete wire, keys, [aliases](../../glossary.md#aliasing-and-arena), BOOL, descriptors and topology before host import. Reject dead
producers, duplicate definitions/roots, forward/dangling/cyclic/cross-scope references, nested scans,
free [leaves](../../glossary.md#term-leaf), missing/extra attachments and malformed roles/residuals. Invoke Compiler's
`GraphCompilationPort.validateSource(CompiledGraphModel)` on complete decoded metadata before
creating any candidate Tensor in every supported version. This shares the complete cold [preflight](../../glossary.md#term-preflight)
without Tensor identities, host reads/copies or tensor-storage allocation; ordinary source
association checks still follow. [Checkpoint](../../glossary.md#term-checkpoint) must not gain a Tensor reconstruction table or
Tensor-constructor/reconstruction dispatcher. Wire opcode-to-[Operation](../../glossary.md#operation) metadata parsing remains codec-owned.

Invoke existing Compiler reconstruction once for the outer source. It constructs every scoped body
and node once with fresh canonical formals and Scan.apply per occurrence, including T=0; no unrolling.
Normalized full recapture equality must precede publication. Copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) retain their current
behavior. Source/semantic/bounds failures mutate no source/state/target and return no partial result;
retain existing exception translation, ArtifactFile uncertain-move/failure/cleanup and [atomic](../../glossary.md#resource-transaction-and-rollback)-live
publication semantics, not a new [durability](../../glossary.md#term-durability) guarantee.

Acceptance must include valid last records and late-invalid bodies, zero time/empty [dimensions](../../glossary.md#dimension),
multiple occurrences/template reuse, unused formals, repeated roles/actuals, all ports/aliases/raw
bits and applicable cold dtypes. Exact/one-below bounds and malformed fixtures with storage/ID probes
must establish rejection before copy/import and formal/producer construction. Consumer proof saves
before compute/after training, closes the source, cold-loads/composes, computes/trains fresh [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
and re-saves FLOAT32/FLOAT64 against independent [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) oracles. Preserve static-RNN/old-format
tests. Other dtype cold roundtrips do not qualify execution or [exact Training continuation](../../glossary.md#exact-training-continuation).

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 5

Implemented under Checkpoint0014 after Compiler0018 acceptance. Length-aware sources without
[Pool3d](../../glossary.md#pool3d--ncdhw-pooling) select ArtifactFile v1 kind `graph-model.v5`, eight [ASCII](../../glossary.md#term-ascii) [magic bytes](../../glossary.md#term-magic-bytes) `SYNGRF05`, i32 version 5.
New **outer [opcode](../../glossary.md#term-opcode) 33** extends the previous [closed inventory](../../glossary.md#term-closed-inventory) ending at 32 for the changed
length-mode header/actual roles. Never extend opcode 23's seven-field [layout](../../glossary.md#layout) in place. Preserve
v4 inline-body inventory and independent scopes, no nested scans or new constructor inventory.
The [scan owner](recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span> owns header/role meaning;
[Checkpoint](../../glossary.md#term-checkpoint) owns exact wire encoding, [dry sizing](../../glossary.md#term-dry-sizing), bounds, decoding and [publication](../../glossary.md#term-publication).

Outer opcode 33 retains phase/opcode and the count-prefixed ordered actual IDs, including the
final lengths reference exactly once. Write the old seven fields unchanged: i64 timeCount,
i32 carryCount, sequenceCount, invariantCount, emissionCount, residualCount, then u8 reverse.
Append i32 rowCarryCount, i32 sequence-[axis](../../glossary.md#term-axis) count equal to sequenceCount, then one u8 presence
flag per ordered sequence (0=SHARED, 1=ROW), with an i32 step axis only when present. Current
row geometry admits axes zero and one. Continue with the unchanged v4 inline body, Q and every
canonical output [descriptor](../../glossary.md#term-descriptor). Body formal count is checked c+s+i, never inputCount including lengths.
Require 0 <= rowCarryCount <= carryCount and checked actual count c+s+i+1.
Zero ROW count, including [carry](../../glossary.md#term-carry)-free length mode and SHARED [scalar](../../glossary.md#term-scalar) accumulation, uses these same
fields. Derive static/named [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) geometry from the final lengths descriptor, never from a first
carry/formal or from reading actual length values. Preserve zero-time and empty-batch sources.
Validate old scalar domains and checked actual/formal/output cardinalities, including parsed actual [arity](../../glossary.md#term-arity),
before reading/allocating axis roles. Bound axis count by sequenceCount and remaining bytes;
reject other flags, negative/out-of-domain axes, truncated entries and descriptor/role mismatches.

Length **values** supplied as external inputs are [runtime](../../glossary.md#term-runtime) batch payloads, not serialized snapshots.
Ordinary named state still follows the existing named-value rules. Preserve canonical ports,
positional [aliases](../../glossary.md#aliasing-and-arena), descriptors and raw named values; persist no histories, prepared state or code.

Sources needing neither length mode nor Pool3d continue selecting byte-identical v1–v4 with exact
old tuples, opcodes and header interpretation. Pool3d in either [scope](../../glossary.md#term-scope) requires v6; v4/v5 reject
opcodes34/35. New readers default old [Scan](../../glossary.md#term-cumulative-scan) records to all-valid; old readers
reject v5. Existing ArtifactLimits/defaults and [atomic](../../glossary.md#resource-transaction-and-rollback)-live-publication/failure behavior remain.
Extend complete dry-size/count/descriptor/role checks before allocation/import, including
late/dead/T=0 occurrences; [cold reconstruction](../../glossary.md#term-cold-reconstruction) must first support the same roles without [unrolling](../../glossary.md#term-unrolling).
The exact encoding and implementation have independent APPROVE under
[Checkpoint0014](../../planning/extensions/checkpoint/tasks/0014-length-aware-general-scan-v5.md#result),
with source-matched [module](../../glossary.md#term-module)/[Javadoc](../../glossary.md#term-javadoc) and public loaded [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) Model proof. V5 may contain both outer
opcodes 23 and 33; neither is legal in a contained body, whose existing inventory is unchanged.
Wire version and [CPU](../../glossary.md#term-cpu) canonical/[specialization](../../glossary.md#term-specialization) versioning are independent obligations.

V5 encoding independently rechecks static [homogeneous](../../glossary.md#term-homogeneous) length-mode cold eligibility through
[Compiler](../../glossary.md#term-compiler) `SourceCapture` before dry sizing, [schema](../../glossary.md#term-schema) allocation, storage [admission](../../glossary.md#term-admission) or [payload](../../glossary.md#artifact-envelope-and-atomic-publication) copy.
A raw association is not an admission credential; earlier public [capture](../../glossary.md#term-capture-meanings) rejection alone is
insufficient. Never serialize length mode through seven-field opcode 23. Structural Model
support does not independently deliver cold, wire or owned numerical workflow capability.

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 6

Implemented [Pool3d](../../glossary.md#pool3d--ncdhw-pooling) [persistence](../../glossary.md#term-persistence) appends ArtifactFile v1 kind `graph-model.v6`, eight [ASCII](../../glossary.md#term-ascii) [magic
bytes](../../glossary.md#term-magic-bytes) `SYNGRF06` and i32 version 6. Append only [opcode](../../glossary.md#term-opcode) 34=MAX_POOL3D and 35=AVERAGE_POOL3D;
each has one input and one output and exact `MaxPool3dAttrs` / `AveragePool3dAttrs`.
Its attribute record is twelve [big-endian](../../glossary.md#text-encoding-and-file-boundaries) i64 fields: [kernel](../../glossary.md#kernel) depth/height/width, [stride](../../glossary.md#term-stride)
depth/height/width, symmetric [padding](../../glossary.md#padding) depth/height/width, then [dilation](../../glossary.md#term-dilation) depth/height/width;
one canonical u8 ceil flag follows (0=floor, 1=ceil; all other values reject).
Positive kernel/stride/dilation, nonnegative padding and literal floor/ceil output geometry remain
Model-owned, checked through exact [Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference); this is not a default-geometry restriction.

These two opcodes are legal in the outer graph and otherwise-qualified all-valid or length-aware
inline bodies. Length bodies add only these two constructors to their whitelist and retain static
geometry, ROW/SHARED, terminal-SHARED and all existing role/[axis](../../glossary.md#term-axis) restrictions. The low-[precision](../../glossary.md#term-precision)
cutover extends Model-legal floating roles separately; no nested [Scan](../../glossary.md#term-cumulative-scan), other [CNN](../../glossary.md#term-convolutional-neural-network-cnn) constructor or [Metal](../../glossary.md#term-metal) Scan grant follows.
Version selection examines outer and contained occurrences: either Pool3d opcode requires at least
v6 even when it occurs only inside a body; [FP8](../../glossary.md#term-fp8) requires v7. Other sources retain their byte-identical
smallest sufficient old format; old exact tuples, opcodes and Scan records remain unchanged.

Retain section order, independent body scopes, canonical ports/roles, positional [aliases](../../glossary.md#aliasing-and-arena), raw named
state bytes, checked [dry sizing](../../glossary.md#term-dry-sizing) and all ArtifactLimits/import/[publication](../../glossary.md#term-publication) ordering. Complete decoded
[metadata](../../glossary.md#term-metadata) passes `GraphCompilationPort.validateSource` before any candidate [Tensor](../../glossary.md#tensor) allocation in
all versions. Reconstruct through the existing Compiler [owner](../../glossary.md#term-owner), then require normalized recapture
equality. Save no [optimizer](../../glossary.md#term-optimizer), executable, [Engine](../../glossary.md#term-engine)/session state or training history; loaded Models
use fresh caller-configured training.

### [Canonical](../../glossary.md#term-canonical-form) [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) version 7

[FP8](../../glossary.md#term-fp8) [persistence](../../glossary.md#term-persistence) appends ArtifactFile v1 kind `graph-model.v7`, eight [ASCII](../../glossary.md#term-ascii) [magic bytes](../../glossary.md#term-magic-bytes)
`SYNGRF07` and i32 version 7. It appends only [dtype](../../glossary.md#data-type--datatype) IDs 8=FLOAT8_E4M3 and 9=FLOAT8_E5M2;
IDs 1–7, [operation](../../glossary.md#operation) records, [scalar](../../glossary.md#term-scalar) representation and all old version tuples remain unchanged.
Each FP8 state [element](../../glossary.md#term-element) or scalar occupies one raw byte without [BOOL](../../glossary.md#data-type--datatype) [normalization](../../glossary.md#term-normalization) or conversion.
Version selection examines every [descriptor](../../glossary.md#term-descriptor), scalar and CAST [target](../../glossary.md#term-target) in the outer and contained
scopes, including otherwise-unpublished body intermediates. Any FP8 [occurrence](../../glossary.md#term-occurrence) selects v7;
sources using only old dtypes retain their byte-identical smallest sufficient v1–v6 domain.
Old-domain readers reject the new dtype IDs; a v7 source containing only old types re-saves
through its smallest sufficient old domain.

The existing closed source constructors, static geometry, row roles, [aliases](../../glossary.md#aliasing-and-arena), canonical ports,
limits, complete [metadata](../../glossary.md#term-metadata) [preflight](../../glossary.md#term-preflight) and [atomic publication](../../glossary.md#term-atomic-publication) ordering remain mandatory. All-valid
and length-mode cold bodies accept Model-legal floating roles and intermediates, including
explicit casts; length-mode operation/[rank](../../glossary.md#term-rank)/ROW/SHARED restrictions are not removed. This extends
the dtype namespace, not the [opcode](../../glossary.md#term-opcode) inventory, native execution or arbitrary [PURE](../../glossary.md#term-pure) persistence.
[Compiler](../../glossary.md#term-compiler) reconstructs the source; [Checkpoint](../../glossary.md#term-checkpoint) neither owns a graph executor nor serializes
compiled activity, [optimizer](../../glossary.md#term-optimizer) slots, histories, prepared state or code.

## `extensions/onnx`

`extensions/onnx` owns [ONNX](../../glossary.md#term-onnx) import/export and mapping.

It must not be part of [runtime](../../glossary.md#term-runtime) [hot path](../../glossary.md#term-hot-path).

Allowed:

- ONNX import
- ONNX export
- ONNX-to-[model](../../glossary.md#term-model) mapping
- model-to-ONNX mapping

Forbidden:

- runtime execution
- [backend](../../glossary.md#backend)-specific [lowering](../../glossary.md#lowering)
- [kernel](../../glossary.md#kernel) selection
- runtime [residency](../../glossary.md#residency)


## [Optimizer](../../glossary.md#term-optimizer)/training [lifecycle](../../glossary.md#lifecycle)

The initial public lifecycle is:

```text
open:
  ordered generic live parameter path/Tensor/storage snapshot
  -> one scalar-objective first-order Engine compile
  -> one Engine PreparedExecution preparation
  -> TrainingSession

run:
  validate the complete live source/input/storage view before execution
  -> reuse the prepared forward/backward execution
  -> validate and detach the objective and every target-indexed gradient occurrence
  -> stage gradient accumulation, parameter values, and optimizer slots
  -> validate the complete parameter view again
  -> commit all parameter bytes, optimizer slots, counters, and accumulation state
```

Rules:

- `extensions/nn` owns module-declared trainable parameters, persistent buffers, and train/eval
  forward behavior.<span data-glossary-navigation="true"> ([module glossary](../../glossary.md#term-module); [trainable parameters glossary](../../glossary.md#trainable-parameter); [buffers glossary](../../glossary.md#term-buffer))</span>
- A `TrainingState` snapshot restores only its detached optimizer/session state: exact optimizer
  configuration and ordered parameter schema, logical and master parameter bytes, optimizer slots,
  pending gradient sums, and execution, optimizer-step, and accumulation counters. It does not
  contain persistent NN buffers, Module modes, caller inputs or progress, graph random state, or
  external scheduler/scaler state. <span data-glossary-navigation="true"> ([schema glossary](../../glossary.md#term-schema); [master parameter glossary](../../glossary.md#term-master-parameter); [gradient glossary](../../glossary.md#term-gradient); [NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`Module.stateDictionary()` retains exact Tensor references and
  is not a detached value snapshot; <span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [detached value glossary](../../glossary.md#detached-model-results))</span>`TrainingState` supplies detached parameter bytes within its
  domain, while a recovery owner must separately detach NN buffer values and any parameter values
  outside that domain. Each Module has a local mode field; recursive <span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>`train()`
  and `eval()` set the whole owned tree, while `StateDictionary` omits modes.
  A supplied immutable <span data-glossary-navigation="true"> ([immutable glossary](../../glossary.md#term-immutable))</span>`ForwardContext` controls a mode-sensitive forward even when it differs
  from that module's local mode. An exact continuation therefore requires the effective
  objective-construction contexts, not merely saved Module modes, to be known and reproducible.
  Training-mode `BatchNorm.forward` installs symbolic next-buffer expressions during graph
  construction. Neither a shallow dictionary nor a successful session run establishes a
  step-aligned detached value for those expressions; that workflow is currently ineligible for
  exact continuation until a supported capture/publication boundary proves the values belong to
  the completed step.<span data-glossary-navigation="true"> ([publication glossary](../../glossary.md#term-publication))</span>
- An exact continuation claim for the currently supported session means the same admitted
  computation route resumes at the next caller-selected input with all state that can affect that
  run restored. The caller owns an explicit handoff for the next input position (including sampler
  cursor and epoch when applicable), external random-generator state, and any program-owned
  scheduler or scaler state. Those participants are not inferred from Training counters. If a
  participant cannot be captured, validated, and restored, the result is a partial restore or
  restart, not exact continuation. No built-in scheduler or scaler state exists today. A durable
  exact artifact may include caller-owned participants only where each has a supported versioned,
  bounded schema and codec plus owner validation and restore; arbitrary caller objects have no
  generic exact-resume encoding.<span data-glossary-navigation="true"> ([handoff glossary](../../glossary.md#term-handoff); [epoch glossary](../../glossary.md#term-epoch-meanings); [artifact glossary](../../glossary.md#term-artifact); [schema and codec glossary](../../glossary.md#schema-serialization-and-checkpoint); [validation glossary](../../glossary.md#term-validation); [resume glossary](../../glossary.md#schema-serialization-and-checkpoint))</span>
- [Graph](../../glossary.md#graph) random state is an explicit [Model](../../glossary.md#term-model) expression chosen when the [objective](../../glossary.md#term-objective) is built.
  `GraphRngState.initial(key, counter)` creates a storage-free fixed expression, not a mutable
  stream. The current <span data-glossary-navigation="true"> ([mutable glossary](../../glossary.md#term-mutable))</span>[`TrainingSession`](../../glossary.md#training-session--trainingsession) compiles that objective once and has no per-run graph-RNG
  input/output handoff; repeated runs do not automatically advance dropout state. A caller may
  record the key/counter and rebuild an equivalent graph before opening a new session, but must
  not claim a changing per-step dropout stream within the current fixed session. Exact
  continuation does not promise bitwise-identical results across devices, backends, versions, or
  execution routes.<span data-glossary-navigation="true"> ([RNG glossary](../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../glossary.md#dropout--inverted-dropout))</span>
- No new public Training value or [API](../../glossary.md#term-application-programming-interface-api) is required to [capture](../../glossary.md#term-capture-meanings) the existing [detached](../../glossary.md#detached-model-results)
  `TrainingState`; caller-owned progress and random state remain explicit, separately typed
  handoff data under the caller's schema. A changing graph-random stream requires a separately
  designed per-run handoff before that claim can be widened.<span data-glossary-navigation="true"> ([random stream glossary](../../glossary.md#graph-rng-state--graphrngstate))</span>
- NN state installation and Training restore currently validate only their own boundaries and
  do not form a cross-owner atomic transaction. <span data-glossary-navigation="true"> ([atomic glossary](../../glossary.md#resource-transaction-and-rollback); [transaction glossary](../../glossary.md#term-transaction))</span>`Module.loadStateDictionary(...)` replaces Tensor
  bindings and is forbidden while a Training session over that tree is open: the session captured
  fixed wrapper, Tensor, storage, and objective identities at open. For a coordinated exact
  recovery, the caller quiesces published state and prepares a distinct, isolated, unpublished
  replacement tree and session. Before mutating any published object, it identifies all required
  participants: detached NN values or a bounded model artifact, per-module modes, Training state,
  effective forward contexts, fixed-objective construction inputs, and typed caller-owned state.
  It rejects missing or unsupported participants and validates every available detached fact,
  including schemas, exposed values, and declared route requirements. A shallow<span data-glossary-navigation="true"> ([bindings glossary](../../glossary.md#term-binding); [Training session glossary](../../glossary.md#training-session--trainingsession))</span>
  `StateDictionary` is not detached value evidence. The caller stages external progress, random,
  scheduler, or scaler state so it can be handed off under exclusion without changing published
  state during candidate preparation.

  On the private tree, install NN bindings and modes first, revalidating any model artifact during
  load. Prepare supported native parameter storage if needed, then construct the objective from
  those final Tensors and open the replacement session. Before restore and publication, compare
  the candidate's canonical logical parameter bytes with the supplied Training state's logical
  bytes at every stable path. This comparison
  may be made from already comparable detached values before open, or from the new private
  session's <span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare); [canonical glossary](../../glossary.md#term-canonical-form))</span>`state()` [snapshot](../../glossary.md#term-snapshot) after open; `ModelCheckpoint.inspect` alone exposes no value bytes.
  Session-specific compatibility and payload validation may then occur through<span data-glossary-navigation="true"> ([payload glossary](../../glossary.md#artifact-envelope-and-atomic-publication))</span>
  `TrainingSession.restore`; it need not be duplicated in a pure pre-open Training validator.
  Validate the resulting candidate and caller-state handoff completely, then publish the whole
  replacement only after success under caller-owned exclusion. On ordinary failure, close and
  discard the private candidate without publishing it. The old session need not be closed before
  a distinct private candidate is prepared; the caller owns its quiescence and retirement during
  the handoff. No end-to-end rollback of an already closed old session, general transaction API,
  or simultaneous visibility to readers outside the exclusion is promised. If a participant
  cannot be staged and handed off safely, do not claim exact recovery. Training 0004 must test
  both rejection before publication and the successful next-input handoff. A later Checkpoint
  adapter may persist only supported, validated participant schemas. NN's live-parameter adapter
  edge grants no recovery coordination or optimizer ownership.<span data-glossary-navigation="true"> ([quiescence glossary](../../glossary.md#term-quiescence); [rollback glossary](../../glossary.md#term-rollback); [Checkpoint glossary](../../glossary.md#term-checkpoint); [ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- Before opening a session, a caller may explicitly ask Training to copy a generic live source's
  current parameters into a caller-owned shareable Arena. A zero-byte Arena shareability probe
  retains its existing failure precedence. Training validates the complete ordered parameter set
  and source storage before destination parameter allocation or replacing any Tensor association,
  copies exact represented bytes into fresh, aligned, exact-capacity writable native segments,
  and installs the associations in path order without replacing Parameter or Tensor identities.
  Source storage may be heap or native and read-only; duplicate Tensor identities and overlapping
  source parameter regions are rejected. Ordinary installation failure restores earlier exact
  associations. The caller owns both source and destination lifetimes, must keep the destination
  Arena alive through any session, and must give the copy exclusive source/storage use. Membership
  is revalidated before installation; successful copy preserves the same live source's membership.
  Copying
  while a session is open is forbidden. This does not change the native-only capture domain of<span data-glossary-navigation="true"> ([Arena glossary](../../glossary.md#term-arena); [heap glossary](../../glossary.md#term-heap))</span>
  `TrainingSession.open` or introduce a [backend](../../glossary.md#backend) transfer.
- A `TrainingSession` borrows exactly one caller-owned [Engine](../../glossary.md#term-engine) and one caller-owned live
  [`TrainingParameters`](../../glossary.md#live-parameter-source--trainingparameters) source. It
  owns exactly one Engine [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution), one optimizer-state set, and its counters. It opens
  that owner with `Engine.prepare` and executes exact-ID inputs with `Engine.run`, or fresh stored
  values with [`Engine.runValues`](../../glossary.md#fresh-value-binding--enginerunvalues). Both share one private optimizer transaction. Close releases
  the prepared owner before the private cotangent-seed arena, preserving failed-open rollback and
  update transactions.<span data-glossary-navigation="true"> ([releases glossary](../../glossary.md#commit-release-distribution-log-and-seed-meanings); [cotangent glossary](../../glossary.md#cotangent); [seed glossary](../../glossary.md#term-seed))</span>
- Session construction snapshots nonempty, unique, nonblank paths and exact Tensors in caller
  encounter order, never sorted; null sources, lists, entries, paths or Tensors reject. The source
  validates live membership at capture and every complete-view validation site, including before
  run/state/restore and precommit for accumulation, update and restore. NN adapters additionally
  retain wrappers. Training owns descriptor, exact storage identity/schema/liveness/accessibility/
  capacity and finiteness checks. Every Tensor identity and non-overlapping storage occurrence
  of the captured parameters remains fixed until close; callers grant exclusive source/parameter
  storage use and must not replace membership, bindings or storage associations while the session
  is open.<span data-glossary-navigation="true"> ([view glossary](../../glossary.md#term-view-meanings); [precommit glossary](../../glossary.md#term-precommit); [descriptor glossary](../../glossary.md#term-descriptor); [identity glossary](../../glossary.md#term-identity-meanings); [liveness glossary](../../glossary.md#term-liveness); [occurrence glossary](../../glossary.md#term-occurrence))</span>
- The public session supports a [scalar](../../glossary.md#term-scalar) [`FLOAT32`](../../glossary.md#term-float32) or [`FLOAT64`](../../glossary.md#term-float64) objective and connected, non-empty
  floating parameters, including BFLOAT16, FLOAT16, FLOAT8_E4M3 and FLOAT8_E5M2, with fully static, dense-contiguous,
  offset-zero, non-view descriptors and exact-capacity writable native host storage accessible to
  every thread that operates on the session. Every parameter has a strictly positive element count
  and finite initial values; rejection occurs during capture before cotangent-seed allocation,
  compilation, or preparation. Every low parameter type retains a private FLOAT32 master and
  publishes one correctly narrowed logical value only after a successful update. Parameter storage is updated
  in place so compiled Tensor identities remain stable. JVM-heap, read-only,
  confined-to-another-thread, device-only, empty, integral, BOOL, and non-finite parameter storage
  are outside the update domain and fail before compile or mutation.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../glossary.md#term-float16); [contiguous glossary](../../glossary.md#term-contiguous); [host storage glossary](../../glossary.md#host-storage); [element glossary](../../glossary.md#term-element); [finite glossary](../../glossary.md#term-finite); [compilation glossary](../../glossary.md#compiler-engine-and-execution-stage-names); [JVM glossary](../../glossary.md#term-java-virtual-machine-jvm); [device glossary](../../glossary.md#term-device); [BOOL glossary](../../glossary.md#data-type--datatype))</span>
- Compiler input occurrence order and publication target indices are the only mapping authorities.
  Parameters and the private seed bind internally by captured Tensor identity; callers fill every
  external bindable position according to the selected admission mode below. Incompatible
  descriptors or absent, dead, inaccessible, insufficient, nonempty overlapping or parameter-aliasing
  storage fail before optimizer mutation.<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [target glossary](../../glossary.md#term-target); [admission glossary](../../glossary.md#term-admission); [aliasing glossary](../../glossary.md#term-aliasing))</span>
- [`run`](../../glossary.md#term-run) requires exact external TensorIds in arbitrary list order, retaining its existing checks
  and diagnostics. `runValues` requires exactly the external positions returned by `inputs()` in
  relative final compiled order, excluding parameters and the private seed, including targets and
  every other external bindable leaf. Fresh IDs are accepted only with exact full static
  descriptors (dtype, shape, layout presence/geometry and requiresGrad). Values are stored values,
  not expressions to execute. Training inserts fixed parameters/seed using its captured binding
  plan and delegates checked static storage geometry to public Engine positional admission; it
  does not recapture, recompile, prepare, copy or rebind prototypes. Old <span data-glossary-navigation="true"> ([leaf glossary](../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([dtype glossary](../../glossary.md#data-type--datatype); [shape glossary](../../glossary.md#shape); [layout glossary](../../glossary.md#layout); [rebind glossary](../../glossary.md#capture-graph-ports-and-reconstruction); [prototypes glossary](../../glossary.md#term-prototype))</span>`run` retains its
  layout-span-or-zero capacity checks. Caller external storage remains borrowed and exclusively
  usable through the synchronous call and internal result closure, never retained for a later run.
  Both paths reject nonempty external/external, external/parameter and seed overlap in compiled
  order, including the same nonempty value in two positions. Engine's repeated-value permission
  does not relax Training's rule; zero-byte storage remains exempt from overlap checks.<span data-glossary-navigation="true"> ([borrowed glossary](../../glossary.md#ownership-borrowing-and-lease); [synchronous glossary](../../glossary.md#term-synchronous))</span>
- `Sgd`, [`Adam`](../../glossary.md#adaptive-optimizer--adam-adamw), and [`AdamW`](../../glossary.md#adaptive-optimizer--adam-adamw) are immutable built-in optimizer configurations. The exact default
  and ordered parameter groups resolve one algorithm per path at open. All parameters update on
  every successful global step through one atomic commit, with no backend or per-element
  algorithm lookup. <span data-glossary-navigation="true"> ([commit glossary](../../glossary.md#term-commit); [lookup glossary](../../glossary.md#term-lookup))</span>`Sgd` owns immutable learning-rate, momentum, dampening,
  weight-decay, and Nesterov configuration; its session owns one zero-initialized momentum slot
  per parameter. Coefficients are finite; momentum is in <span data-glossary-navigation="true"> ([learning-rate glossary](../../glossary.md#gradient-and-learning-rate); [momentum glossary](../../glossary.md#term-momentum); [weight glossary](../../glossary.md#term-weight); [Nesterov glossary](../../glossary.md#term-nesterov))</span>`[0, 1)`, [dampening](../../glossary.md#term-dampening) in `[0, 1]`, and
  weight decay is non-negative. Nesterov requires positive momentum and exactly zero dampening.
  Coupled weight decay first produces <span data-glossary-navigation="true"> ([weight decay glossary](../../glossary.md#term-weight-decay))</span>`adjusted = gradient + weightDecay * parameter`. The first
  momentum step stores `adjusted` without dampening. Later steps store
  `momentum * previous + (1 - dampening) * adjusted`. Ordinary momentum selects that new slot;
  Nesterov selects `adjusted + momentum * updatedMomentum`. The [parameter](../../glossary.md#term-parameter) candidate is
  `parameter - learningRate * selectedGradient`. FLOAT32 and all low-floating parameter sessions
  validate the same invariants after coefficient narrowing, including rejecting positive
  Nesterov momentum that narrows to zero, and use FLOAT32 master, momentum, accumulation, and
  candidate arithmetic. A successful low update narrows the validated master candidate once to
  the declared logical type. FLOAT64 uses the configured binary64 values.<span data-glossary-navigation="true"> ([narrowing glossary](../../glossary.md#term-narrowing); [binary64 glossary](../../glossary.md#floating-point-precision-and-rounding))</span>
- `Adam` and `AdamW` require finite positive learning rate and epsilon, finite beta1 and beta2
  in <span data-glossary-navigation="true"> ([learning rate glossary](../../glossary.md#term-learning-rate); [epsilon glossary](../../glossary.md#epsilon))</span>`[0, 1)`, and finite non-negative weight decay. FLOAT32 and all low-floating assignments
  narrow coefficients once to binary32 at open and reject a rate or epsilon that becomes zero,
  a beta that becomes one, or any out-of-range or non-finite coefficient. Both have zero-initialized
  first and second moments in session state. At one-based global step <span data-glossary-navigation="true"> ([binary32 glossary](../../glossary.md#floating-point-precision-and-rounding))</span>`t`, they calculate
  `m = beta1 * oldM + (1 - beta1) * g` and `v = beta2 * oldV + (1 - beta2) * g * g`,
  divide them by `1 - beta1^t` and `1 - beta2^t`, and subtract
  `learningRate * correctedM / (sqrt(correctedV) + epsilon)`. Adam first adds
  `weightDecay * masterParameter` to `g` (coupled decay). AdamW leaves `g` unchanged and first
  multiplies the master parameter by `1 - learningRate * weightDecay` (decoupled decay).
  Accumulate-and-step uses the pending gradient sum, not its mean. Bias corrections are validated
  outside element loops; all moments and arithmetic intermediates must be finite and second
  moments non-negative.<span data-glossary-navigation="true"> ([mean glossary](../../glossary.md#term-mean); [Bias glossary](../../glossary.md#term-bias))</span>
- A reset-and-step run uses only that run's gradients and clears older accumulation on success.
  An accumulate run atomically adds the run's gradients without updating parameters or optimizer
  step count. An accumulate-and-step run applies the pending sum plus that run's gradients and
  clears it on success. <span data-glossary-navigation="true"> ([optimizer step glossary](../../glossary.md#optimizer-and-stochastic-gradient-descent))</span>`zeroGrad` clears only pending accumulation.
- Optimizer step numbers start at one and advance exactly once after each successful parameter
  update. Execution numbers advance after every successful forward/backward run, including an
  accumulate-only run. Failed runs advance neither.
- Run/materialization, gradient decoding, candidate calculation, and complete precommit
  validation precede all mutation. Current parameters, decoded gradients, accumulated gradients,
  optimizer slots, every arithmetic intermediate, and parameter candidates must remain finite in
  the parameter precision. NaN, either infinity, or represented overflow is an ordinary rejected
  failure; either signed zero is finite and supported. A non-finite detached objective alone is
  not optimizer state and does not invalidate a run whose required gradients and optimizer
  arithmetic are finite. Restore rejects a non-finite parameter, optimizer slot, or accumulation
  payload. Supported native parameter stores make ordered path-order primitive writes
  non-throwing after validation. Any ordinary validation, execution, materialization, non-finite,
  overflow, or optimizer failure leaves parameter bytes, optimizer slots, counters, and pending
  gradients unchanged. Caller violation of exclusive storage use and fatal virtual-machine
  failure are outside that atomicity guarantee.<span data-glossary-navigation="true"> ([materialization glossary](../../glossary.md#materialization); [precision glossary](../../glossary.md#term-precision); [NaN glossary](../../glossary.md#term-nan); [infinity glossary](../../glossary.md#term-infinity); [overflow glossary](../../glossary.md#term-overflow); [signed zero glossary](../../glossary.md#term-signed-zero); [optimizer state glossary](../../glossary.md#term-optimizer-state); [optimizer slot glossary](../../glossary.md#term-optimizer-slot); [atomicity glossary](../../glossary.md#term-atomicity))</span>
- Every operation first attempts lifecycle admission without inspecting its arguments. Admission
  observes Engine closure first, then session closing/closure, then an already-admitted busy
  operation; only the admitted operation validates its arguments inside a <span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation); [arguments glossary](../../glossary.md#term-argument))</span>`finally`-guarded
  operation lifetime. Close atomically rejects later admission, waits for the admitted operation
  without holding an Engine or Runtime lock, closes all owned resources once, and makes
  repeated/concurrent close replay the retained cleanup result. An operation admitted before
  session close begins may finish normally before cleanup.<span data-glossary-navigation="true"> ([lifetime glossary](../../glossary.md#term-lifetime); [Runtime glossary](../../glossary.md#term-runtime))</span>
- `TrainingStep` and immutable in-memory session-state snapshots own detached canonical bytes and
  remain readable after session or Engine close. Logical parameter payloads use their declared
  width; master, first-slot, optional second-moment, and accumulated-gradient payloads use
  FLOAT32 width for BFLOAT16,
  FLOAT16, and FLOAT32 parameters and FLOAT64 width for FLOAT64 parameters. A restore validates the
  complete optimizer kind/configuration, path/order/schema, counters, payload finiteness, exact
  slot presence and widths, and non-negative Adam/AdamW second moments before installing any state.
  SGD has no second-moment payload. The existing momentum payload remains the first-slot payload
  for every algorithm, with a neutral first-moment alias. The default optimizer accessor returns<span data-glossary-navigation="true"> ([SGD glossary](../../glossary.md#term-stochastic-gradient-descent-sgd); [alias glossary](../../glossary.md#aliasing-and-arena))</span>
  `Optimizer`, including Adam and AdamW defaults, rather than only `Sgd`.
- CPU, Metal, and supported mixed Engine preparations use this same lifecycle. Training never
  branches on backend identity. This lifecycle does not independently prove or widen low-precision
  Metal graph capability. Unsupported compile, prepare, execution, publication, or materialization
  fails before optimizer mutation.<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [Metal glossary](../../glossary.md#term-metal); [backend identity glossary](../../glossary.md#backend-identity--backendid))</span>
- `FORWARD_ONLY` performs no [autograd](../../glossary.md#autograd). `FORWARD_AND_BACKWARD` and the initial `TRAINING_STEP` use
  the same combined pre-capture forward/backward construction; the initial `TRAINING_STEP` adds no
  optimizer-update graph work.

A later architecture version may [compile](../../glossary.md#compile) optimizer updates into the graph. That change requires a
separate explicit decision and does not weaken the current concrete-backend [dependency](../../glossary.md#term-dependency) prohibition.

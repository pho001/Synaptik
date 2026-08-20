# NN Extension Master Plan

## Goal

Define the stateful neural-network composition layer: module ownership, trainable parameters,
persistent buffers, train/eval behavior, forward context, layers, blocks, and functional
conveniences built from `modules/model` Tensor semantics.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [ADR 0007: Neural-network module and training boundary](../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Training graph](../../../architecture/training-graph.md)

## Scope

- `Module`, `Parameter`, and `Buffer` contracts
- module-tree parameter and buffer discovery
- train/eval mode propagation and forward context
- typed functional model definition with immutable named topology
- deferred input-dependent parameter binding after its lifecycle is specified exactly
- explicit parameter-initialization conveniences over caller-owned sources or exactly documented
  per-layer seeded standard sources, never a global mutable generator
- neural-network layers, blocks, and functional conveniences that compose model operations

## Out of scope

- generic Tensor or operation semantics
- compiler autograd construction
- optimizer algorithms, update orchestration, and training sessions
- backend storage, lowering, kernel selection, and backend-specific optimizer execution

## Module invariants

- NN owns module-declared `Parameter` and `Buffer` values.
- A discovered `Parameter` is the downstream capability for installing one schema-compatible
  trainable Tensor binding; NN retains declaration, schema, and validation ownership.
- NN owns train/eval forward behavior and forward context.
- NN composes `modules/model`; it does not make model depend on nn.
- NN must not depend on `extensions/training`, compiler, runtime, prepare, engine, or concrete backends.
- Optimizers and training-step orchestration belong to `extensions/training` downstream of nn.

## Allowed dependencies

- modules/model

## Forbidden dependencies

- extensions/training
- modules/compiler
- modules/runtime
- modules/prepare
- modules/engine
- backends/cpu
- backends/metal
- backends/cuda

## Package structure

```text
io.github.pho001.synaptik.nn/
  module/       public module ownership, named state, typed model topology, standard construction recipes, and forward-context contracts
  initialization/ explicit eager parameter-initialization conveniences and policies
  layers/       public stateful neural-network layers
  functional/   stateless NN-oriented conveniences over model operations
```

The implemented foundation remains deliberately small, and task 0005 adds only the first
concrete type under `layers/`:

- `Module` owns direct named state declarations and its local train/eval mode; it has no universal
  forward signature.
- `Parameter` and `Buffer` retain a named current `Tensor` binding without becoming `Tensor`
  subtypes.
- `ForwardMode` and immutable `ForwardContext` provide the mode snapshot supplied to a concrete
  layer's own typed forward method.

Task 0004 adds only the public stateless `ParameterInitializers` namespace and package
documentation under `initialization/`. It creates eager gradient-enabled unlabeled Tensor leaves;
it does not change `Parameter`, retain a random source, or introduce an initializer object model.
Task 0005 adds final public `Linear` under `layers/` without a generic layer abstraction. Task
0006 adds final public `LayerNorm` beside it and reuses the existing affine Model expression.
Completed task 0007 adds final public `Embedding` beside those layers with one caller-supplied
table parameter and exact delegation to the existing Model embedding convenience. It deliberately
adds no layer-owned initializer or padding-row policy because the current contracts provide
neither a selected embedding initialization policy nor a way to preserve a padding row without
hidden mutation or new gradient/update behavior. Completed task 0008 adds final public
`BatchNorm` as the first mode-sensitive layer, with exact rank-one affine parameters and running-
statistic buffers plus context-selected Model inference or training composition. Training installs
the pure producer's next mean followed by next variance through stable buffer wrappers; it adds no
execution, transaction, or checkpoint contract. `functional/` remains future work. Task 0010
places the in-memory state-dictionary values beside the module ownership contracts under
`module/`; it deliberately leaves bytes, files, codecs, versions, and other persistent checkpoint
transport for a future concrete consumer.

Completed task 0011 selects one narrow public `UnaryTensorModule extends Module` base because the real
`Sequential` consumer must know both ownership and `Tensor forward(Tensor)` at compile time.
`Sequential` is implemented under `module/` as immutable structural composition with numeric child
names. Only mode-insensitive `Linear`, `LayerNorm`, and `Embedding` participate; context-sensitive
`BatchNorm` and explicit-state/result `Dropout` remain direct `Module` subclasses.

The recurrent milestone adds composition without hidden runtime state. Task 0012 starts with one
vanilla tanh `RnnCell`, and task 0013 follows with one packed reset-after `GruCell`; each receives
and returns the hidden Tensor explicitly on every call. Both remain direct `Module` subclasses:
their two-Tensor signatures are not `UnaryTensorModule` contracts and cannot be placed in
`Sequential`. Task 0014 adds one packed `LstmCell` whose caller supplies input, hidden, and cell
Tensors and receives exact next-hidden and next-cell references in a cell-specific result. A
separate sequence task must then choose static unrolling through concrete cell signatures or
justify a genuinely general recurrent Model scan; existing cumulative sum/product operations are
not recurrent scan primitives. Completed task 0015 selects the capability current contracts can
represent honestly: one cell-specific vanilla `RnnSequence` statically unrolled from fully static
time-major input metadata and a snapshotted Java `long[]` of sequence lengths. Stable original-
order active batches omit padded logical rows, and final hidden rows are restored to original
batch order. Runtime Tensor lengths or masks remain deferred because they require a genuine
data-dependent recurrent scan/control-flow contract rather than dense post-cell masking.
Completed tasks 0016 and 0017 apply that same static packing policy through the concrete GRU and
LSTM signatures. GRU restores one final hidden state; LSTM exposes compact hidden outputs while
carrying cell state internally and restores both final hidden and final cell state.

The next model-composition milestone begins with task 0018. It adds typed `Model<I, O>`,
`Model.define`, and a definition-scoped `Topology.addModule` collector while preserving current
eager layers unchanged. Complete topology validation precedes ownership installation so a failed
definition does not strand modules under an unreachable partial model. The topology seals after
the callback and produces descriptive state paths such as `hidden.weight`; Model owns no backward,
compile, training, or execution method.

Detailed task 0019 now separates input-dimension inference from that structural foundation. It
keeps one final public `Linear`, every public `Parameter` fully bound, and future names privately
reserved in `Module`. One normal Linear constructor omits `inFeatures`; its first forward performs
complete layer-local initialization before returning the ordinary usable Tensor expression from
that same call. There is no public lazy type, factory, bind/build/initialize operation, or status
query. Parameter discovery and state export fail closed while an owned reservation is unbound;
strict load may instead use a complete candidate dictionary as the binding source. Initialization
infers only the positive static final input extent, uses an explicit closed weight policy plus a
deterministic random factory/seed, publishes a layer's state together, and serializes concurrent
first calls. Arbitrary functional forward code still prevents a whole-model first-forward
transaction. Batch/time leading extents remain variable; hidden/output widths, embedding size,
class count, vocabulary size, and recurrent hidden size remain architecture or schema choices.

Detailed task 0020 applies that proven private reservation lifecycle uniformly to the three
existing final recurrent cells and their cell-specific sequence containers. In the same atomic
change it deliberately replaces the recent Linear-only policy enum with one closed immutable
`ParameterInitialization` value covering Glorot/Kaiming normal/uniform, configured normal/uniform,
and zero/one through the existing stateless initializers. The old type is removed without alias;
Linear and repository-owned consumers migrate while completed task 0019 stays historical evidence.
An automatic cell constructor omits only `inputSize`; hidden size, bias, floating parameter type,
general policy, and seed remain explicit. Random high-level recurrent policies use the exact
deterministic JDK `L64X128MixRandom` algorithm and no caller-supplied factory; zero/one use no RNG,
while existing eager constructors retain caller-owned `RandomGenerator` control. Matching sequence
constructors create one standard cell, zero-state overloads derive `[batch, hiddenSize]` from
static input and cell schema, and overloads without lengths mean every row is valid for the
complete static time extent. Current caller-cell and explicit-state/length contracts remain
advanced paths.

One cell instance owns one parameter set across the complete static unroll. Each represented time
step still invokes ordinary Tensor operations anew, producing fresh Tensor identities/producers
whose provenance retains the same exact parameter leaves and prior carried state. That is the
forward ancestry consumed by current compiler exact-identity reverse traversal and contribution
accumulation; NN adds no backward, tape, compiler dependency, runtime scan, or execution claim.
Initialized Embedding, a stateless construction factory, and directional recurrence are the
separate concise NN 0020A–0020C follow-ups before the cross-module runtime-scan program. Detailed
[task 0020A](tasks/0020a-initialized-embedding.md) is Complete: it added one eager constructor to
the existing final `Embedding`, applied the common initialization policy to the complete explicit
table Shape, and deliberately gave every row ordinary trainable semantics. Detailed
[task 0020B](tasks/0020b-stateless-standard-module-factory.md) is Complete. It adds one
instance-field-free final `ModuleFactory` with five concrete standard recipes, explicit per-layer
type/policy/seed, exact `L64X128MixRandom` selection for automatic Linear, no ownership or
registry behavior, and unchanged advanced direct constructors. Detailed
[task 0020C](tasks/0020c-bidirectional-static-recurrent-composition.md) is Complete. It added
separate concrete RNN, GRU, and LSTM bidirectional static
sequences; direct `forward`/`backward` child ownership; valid-prefix-only reverse `GATHER_ND`;
original-time alignment; fixed forward-first final-axis `CONCAT`; and type-specific directional
final states. Arbitrary multidirectional/stacked recurrence, a generic recurrent base,
configurable merge, and another `ModuleFactory` recipe remain outside it.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-module-parameter-buffer-and-forward-context-foundation.md) | Module, parameter, buffer, and forward-context foundation | Complete | Completed `modules/model` Tensor contracts; ADR 0007 | Created the NN Gradle module and its minimal model-only state/mode API, activated the required one-way training build edge, and enforced it with architecture tests; it adds no layers or execution behavior. |
| [0002](tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md) | Module-tree ownership and recursive mode propagation | Complete | 0001 | Added exclusively owned child modules, deterministic collision-free dot-path parameter/buffer snapshots, and atomic recursive `train()`/`eval()` propagation without checkpoint or optimizer behavior. |
| [0003](tasks/0003-validated-parameter-and-buffer-binding-replacement.md) | Validated parameter and buffer binding replacement | Complete | 0002 | Added Module-owned replacement of one direct current Tensor binding with exact wrapper identity, structural snapshot, validation, and explicit no-concurrency-guarantee semantics; it adds no optimizer or checkpoint behavior. |
| [0004](tasks/0004-explicit-eager-parameter-initializers.md) | Explicit eager parameter initializers | Complete | 0001–0003; completed Model eager constant/random contracts | Added one stateless eight-method parameter-initializer surface: floating zero/one and explicit-source normal/uniform leaves plus fixed rank-two `[outFeatures, inFeatures]` Glorot and Kaiming-ReLU policies, with no label, hidden RNG, or Parameter policy. |
| [0004A](tasks/0004a-parameter-update-and-traversal-hardening.md) | Parameter update and traversal hardening | Complete | 0001–0004; post-0004 code review | Closed the three review findings with a public schema-validated `Parameter.replace`, iterative identity-defended deep-tree traversal, and complete fan-initializer Java-array-limit contracts before adding a layer. |
| [0005](tasks/0005-linear-layer.md) | Linear layer | Complete | 0001, 0004, 0004A; completed model `Tensor.linear` | Added the first stateful layer with caller-supplied or explicit-source Glorot-uniform `weight`, optional caller-supplied or zero `bias`, stable parameter handles, and exact delegation to visible Model linear composition without execution behavior. |
| [0006](tasks/0006-layer-normalization-layer.md) | Layer normalization layer | Complete | 0001–0005; completed Model 0021 | Added mandatory exact-Shape `scale` and `bias` parameters, caller-supplied or ones/zeros initialized state, stored typed epsilon, and mode-insensitive exact delegation to affine `Tensor.layerNorm`. |
| [0007](tasks/0007-embedding-layer.md) | Embedding layer | Complete | 0006; completed Model 0019A1 | Added one positive fully static rank-two `weight` parameter supplied by the caller and mode-insensitive delegation exactly to axis-zero `Tensor.embedding`, with no layer-owned initialization or padding-row contract. |
| [0008](tasks/0008-batch-normalization-layer.md) | Batch normalization layer | Complete | 0007; completed Model 0021B–0021C | Added final affine `BatchNorm` with exact rank-one state, explicit typed scalars and channel axis, context-selected inference/training composition, and training-only installation of the two pure next-statistic expressions into stable buffers. |
| [0009](tasks/0009-dropout-layer.md) | Dropout layer | Complete | 0008; completed Model 0019B–0019B1 | Added a parameterless/bufferless mode-sensitive layer with caller-threaded `GraphRngState`, a minimal NN result carrier, exact Model training delegation, and identity-preserving evaluation bypass without hidden random state. |
| [0010](tasks/0010-state-dictionary-and-checkpoint-contract.md) | State dictionary and checkpoint contract | Complete | 0009; stable module-tree traversal and replacement contracts | Added an immutable ordered in-memory dictionary with path/kind/Tensor entries and strict validate-before-install Module load; every byte/file/codec/version format and optimizer state remains deferred. |
| [0011](tasks/0011-unary-tensor-module-composition-and-sequential.md) | Unary Tensor module composition and Sequential | Complete | 0010; completed Linear, LayerNorm, and Embedding unary Tensor APIs | Added a narrow abstract Module subtype and immutable numeric-child Sequential with type-safe left-to-right Tensor composition; BatchNorm context and Dropout state/result signatures remain outside it. |
| [0012](tasks/0012-vanilla-rnn-cell.md) | Vanilla tanh RNN cell | Complete | 0011; completed Model linear, ADD, and TANH expressions | Added one direct Module with explicit caller-threaded hidden state, two recurrent projections, optional shared bias, fixed tanh activation, and no sequence traversal or hidden state. |
| [0013](tasks/0013-gru-cell.md) | GRU cell | Complete | 0012 | Added one direct Module with explicit caller-threaded hidden state, packed reset/update/candidate projections, fixed reset-after gating and interpolation, one optional packed input-side bias, and no sequence traversal or hidden state. |
| [0014](tasks/0014-lstm-cell.md) | LSTM cell | Complete | 0013 | Added one direct Module with explicit caller-threaded hidden/cell state, input/forget/candidate/output packed projections, fixed equations, one optional input-side packed bias with an all-zero initialized policy, and a cell-specific next-hidden/next-cell result. |
| [0015](tasks/0015-static-packed-rnn-sequence.md) | Static packed RNN sequence | Complete | 0012–0014; completed Model SELECT, GATHER, STACK, and eager INT64 leaves | Added one cell-specific vanilla-RNN container with construction-time Java lengths, stable original-order active-batch compaction, compact per-step outputs, and final-hidden restoration; runtime Tensor masks/lengths require a future genuine recurrent scan. |
| [0016](tasks/0016-static-packed-gru-sequence.md) | Static packed GRU sequence | Complete | 0015 | Reused the proven one-hidden-state static packing and restoration policy for `GruCell` without a shared recurrent abstraction. |
| [0017](tasks/0017-static-packed-lstm-sequence.md) | Static packed LSTM sequence | Complete | 0014–0015 | Added cell-specific static packing with compact hidden outputs, internal cell-state carrying, and restoration of both final states. |
| [0018](tasks/0018-typed-functional-model-topology.md) | Typed functional Model topology | Complete | 0010–0017; stable Module ownership | Added typed `Model<I,O>`, functional definition/forward contracts, sealed `Topology.addModule`, atomic descriptive child ownership, and stable state paths without lazy state or training behavior. |
| [0019](tasks/0019-automatic-first-forward-linear-initialization.md) | Automatic first-forward Linear initialization | Complete | 0018; exact initialization/state-dictionary decision | Kept one public Linear type, reserved future state privately, initialized its complete parameter set before constructing the first returned forward expression, failed closed on incomplete discovery/export, and allowed strict dictionary initialization while inferring only `inFeatures`. |
| [0020](tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md) | Automatic recurrent initialization and sequence defaults | Complete | 0019; current recurrent cell/sequence and Model provenance contracts | Replaced the recent Linear-only enum with one general closed `ParameterInitialization`; added automatic input-width binding to all three existing final cells with explicit hidden size/type/policy/seed and standard `L64X128MixRandom` only for random policies; added standard-cell sequence constructors plus zero-state/all-valid overloads while preserving caller cells, explicit states, static lengths, and shared-parameter fresh-node unroll provenance. |
| [0020A](tasks/0020a-initialized-embedding.md) | Initialized Embedding | Complete | 0020; current Embedding and tokenizer/schema boundaries | Added one eager initialized constructor to the existing final `Embedding` with explicit positive vocabulary size, embedding size, floating type, common `ParameterInitialization`, and seed. Random policies use the documented standard PRNG; zero/one use no RNG. Every table row is an ordinary trainable row: padding identity and valid lengths remain future Text/Data schema, with no padding index or frozen-row contract. |
| [0020B](tasks/0020b-stateless-standard-module-factory.md) | Stateless standard ModuleFactory | Complete | 0020–0020A | Added one final instance-field-free standard recipe namespace: `embedding`/`linear` return `Embedding`/`Linear`, while `rnn`/`gru`/`lstm` return the concrete matching Sequence with Cell assembly hidden. Every call creates a fresh module and takes explicit per-layer type/`ParameterInitialization`/seed; standard Linear selects exact `L64X128MixRandom`. `Topology.addModule` remains the sole owner, and advanced direct constructors retain caller-controlled random sources/factories. |
| [0020C](tasks/0020c-bidirectional-static-recurrent-composition.md) | Type-safe bidirectional static recurrent composition | Complete | 0020–0020B; stable static sequence/result and Model gather/composition contracts | Added separate concrete RNN/GRU/LSTM containers with independent `forward`/`backward` cells, parameter trees, and seeds; reversed only each valid prefix with `GATHER_ND`; realigned backward outputs to original time/batch order; merged exact forward then backward hidden features by fixed final-axis `CONCAT`; and returned type-specific directional final hidden plus LSTM cell states. One cell shares weights only across time within its own direction. |
| 0021 | Runtime recurrent scan/control-flow prerequisite program | Draft | 0020C; explicit cross-module architecture decision and Model/Compiler/Prepare/Runtime/Engine/backend support | Establish a fixed recurrent body/node plus runtime input-binding and execution contracts before NN exposes a new valid-length recurrent API; specific length values must not specialize Model topology or compiled graph structure, and the design must preserve the selected directional/state contracts. |
| 0022 | Valid-length recurrent API and Data integration | Draft | 0021; Data 0001–0002 architecture and valid-length contracts | Consume Data-owned runtime valid lengths through the proven scan, derive zero states as selected, and deliberately migrate or retain the current static `long[]` compatibility contracts without presenting a host adapter as the target API. |
| 0023 | Arbitrary dense validity-mask semantics | Draft | 0022; concrete attention/loss/recurrent consumer | Reassess an explicit Boolean mask only for validity patterns with holes; keep it derived or separately supplied for that consumer, never a second stored representation of ordinary right padding, and never a claim of skipped recurrent work. |
| 0024 | Typed model/recurrent/data integration checkpoint | Draft | 0020–0022; 0023 only if selected; Checkpoint model-state and Training publication readiness | Validate model state paths, automatic-initialization/checkpoint compatibility, variable-batch behavior, recurrent continuation, autograd/training handoff, documentation, and integration without moving persistent checkpoint I/O into NN. |

## Milestones

- Module, parameter, buffer, and forward-mode contracts
- Explicit eager initialization and first `Linear` layer
- Initial stateful layers and functional conveniences
- Explicit-state stochastic and state-transition layers
- Deterministic state-dictionary/checkpoint contract
- Narrow unary composition only when justified by `Sequential`
- Explicit-state recurrent cells, beginning with vanilla tanh RNN
- Recurrent sequence composition only after the cell signatures prove its type boundary
- Typed functional Model definition with sealed descriptive topology
- Honest deferred binding for input-dependent parameter dimensions
- Automatic recurrent binding with explicit per-layer policy/seed and ergonomic explicit-state defaults
- Type-safe directional recurrence before the runtime recurrent-scan program
- Data-owned canonical valid lengths after a genuine runtime-scan/input-binding prerequisite
- Training-extension integration over the stable parameter contract

## Current status

NN 0001 is Complete as an explicitly authorized parallel exception while CPU 0006D remains the
global active frontier. The user authorized this exception because the bounded foundation depends
only on completed `modules/model` Tensor contracts, does not require CPU, Engine, compiler,
runtime, or prepare behavior, and its planned files do not overlap the active CPU task. The
exception is implementation-order only; it does not change the global roadmap, architecture, or
dependency direction. NN work resumes in this master-plan order after 0001.

NN 0004 is intentionally placed immediately before the first `Linear` layer. It is not a
`Parameter` feature: a parameter retains one supplied Tensor; initializer conveniences create a
Tensor that a caller or layer then supplies to that parameter. They must reuse the existing eager
model construction boundary and caller-owned random source rather than add a default generator,
hidden seed, mutable module RNG, or graph-execution requirement.

NN 0002 is Complete under the same authorized parallel exception. It remains model-only and
touched only the NN module plus its own task record and master-plan row; it did not alter the
active CPU work, the global roadmap, Gradle dependencies, or architecture rules. It establishes
exclusive permanent child ownership, collision-free dot-separated recursive paths, immutable
discovery snapshots, and identity-preflighted recursive mode propagation.

NN 0003 is Complete under that same authorized parallel exception. Its bounded replacement API stays
inside the existing NN module and existing model-only dependency edge; it changes no Gradle,
global-roadmap, architecture, CPU, or Engine files. The task deliberately makes replacement a
direct-state operation of the declaring `Module`, rather than a public `Parameter`/`Buffer`
setter, and records the absent checkpoint and concurrency guarantees before a downstream training
extension can consume the stable binding contract.

Detailed NN 0004 is Complete under the same authorized parallel exception. It added only the
planned NN `initialization/` package, focused NN tests, the glossary, and synchronized NN planning
records. Its clean documentation pass finalized the eager ownership, exact fan formula, random-
source lifecycle, `Parameter`, and graph-RNG boundaries and passed generated-Javadoc, Markdown,
scope, and whitespace gates while reusing the stable focused test evidence. Those seven task-owned
files do not overlap the dirty CPU 0007 task, CPU master plan, or global roadmap. Existing Model
constant/random APIs provide the complete eager leaf, caller-owned RNG, floating conversion, and
failure semantics, so NN 0004 adds no Model, build, dependency, architecture, compiler, runtime,
Engine, or backend work.

Detailed NN 0004A is Complete as a bounded review-remediation insertion before NN 0005. The
post-0004 review proved that task 0003's protected direct Module replacement cannot be consumed by
a generic downstream optimizer, that the recursive discovery and mode implementation relies on
the Java call stack, and that the four fan-method Javadocs omit their delegated positive
Java-array-limit failure. Task 0004A closes those findings inside the existing NN and Model-only
boundary: `Parameter` becomes the public schema-validated replacement capability, Module traversal
becomes iterative and identity-defended, and initializer documentation/tests lock the existing
failure side effects. It changes no dependency, architecture, Gradle, CPU, or global-roadmap file.
The isolated implementation pass passed the final focused NN suite with 7 suites and 33 tests.
The independent documentation pass finalized Javadoc, training API, glossary, and planning
evidence and passed generated-Javadoc, Markdown, public-surface, scope, import, and whitespace
checks without repeating the stable executable suite.

Detailed NN 0005 is Complete under the same bounded parallel exception. It adds final public
`Linear`, its `layers` package contract, focused state/initialization/forward tests, the glossary,
and synchronized NN planning in exactly seven paths. Supplied state keeps exact parameter Tensor
identity and strict positive static schemas; initialized state uses explicit-source Glorot uniform
and deterministic zero bias. Forward is mode-insensitive and delegates exactly to the current
Model `Tensor.linear` composition using the bindings observed by that call. The implementation
pass passed the focused 2-suite/11-test selection and final 9-suite/44-test NN suite. The
independent documentation pass finalized glossary and planning wording, reviewed the drafted
Javadocs unchanged, and passed generated-Javadoc, Markdown, public-surface, dependency/import,
scope, and whitespace checks without repeating stable executable tests. Before this planning
step there was no later detailed NN task. Detailed
[NN 0006](tasks/0006-layer-normalization-layer.md) is Complete. Its implementation pass added the
exact planned layer and focused tests inside the model-only boundary, then the independent clean
documentation pass finalized Javadoc, glossary, and planning evidence. The implementation pass
passed focused 2-suite/9-test and full NN 11-suite/53-test validation; the documentation pass
reused that stable Java evidence and passed final generated-Javadoc, public-surface, Markdown,
dependency/import, scope, and whitespace checks. CPU, Engine, runtime, prepare, and numerical
execution were not prerequisites. Detailed
[NN 0007](tasks/0007-embedding-layer.md) is Complete as a bounded model-only layer task over
completed Model 0019A1 and the stable NN parameter contract. The clean implementation context
added the planned layer and focused test, then passed the final 8-test Embedding selection and
12-suite/61-test NN module suite. The independent clean documentation context finalized Javadocs,
the glossary, and planning evidence and passed generated-Javadoc, public-surface, Markdown,
dependency/import, exact-scope, and whitespace checks without repeating stable executable tests.
Detailed [NN 0008](tasks/0008-batch-normalization-layer.md) is Complete. Its isolated
implementation context added the exact planned BatchNorm production and test surface, passed the
final focused 2-suite/13-test selection and authoritative 14-suite/74-test NN module run with no
failures, errors, or skips, and changed no executable Java or test afterward. The independent
clean documentation context `/root/nn_0008_docs` finalized the glossary and planning evidence,
reviewed the complete package/type Javadocs without requiring a source edit, and passed final
generated-Javadoc, exact-surface, Markdown, dependency/import, seven-path scope, status, and
whitespace gates without repeating the stable executable tests. Detailed
[NN 0009](tasks/0009-dropout-layer.md) is Complete. Its isolated implementation context added the
exact planned Dropout production and test surface, passed the final focused one-suite/9-test
selection and authoritative 15-suite/83-test NN module run with no failures, errors, or skips,
and changed no executable Java or test afterward. The independent clean documentation context
`/root/nn_0009_docs` finalized the glossary and planning evidence, reviewed the complete
package/type Javadocs without requiring a source edit, and passed final generated-Javadoc,
independent reflection and `javap`, Markdown, dependency/import, seven-path scope, status, and
whitespace gates without repeating the stable executable tests. The completed
parameterless/bufferless `Dropout` module receives explicit graph RNG state on every call:
training delegates to one existing Model occurrence, while evaluation returns a fresh NN result
containing the exact input and incoming-state references and creates no Tensor or producer. Tasks
0010–0011 remained concise Draft rows with no detailed task files at that completion point, and no
NN task was Ready.

Detailed [NN 0010](tasks/0010-state-dictionary-and-checkpoint-contract.md) is Complete. It adds the
planned Module-owned immutable in-memory dictionary of ordered qualified path/kind/exact-Tensor
entries and one strict validate-before-install load. Parameters retain their existing permanent
schema; buffers compare the target's current data type and Shape while deliberately ignoring
gradient eligibility so valid BatchNorm next-statistic expressions remain loadable. It adds no
checkpoint bytes, files, codecs, versions, optimizer/session state, graph RNG state, execution
behavior, dependency, or architecture change. The implementation context's stabilized focused
suite passed 15 tests, and its sole authoritative NN module run passed 16 suites/98 tests with no
failures, errors, or skips. Independent documentation context `/root/nn_0010_docs` reused that
unchanged executable evidence, finalized the public/package Javadocs, Training API, glossary, and
planning records, and passed final Javadoc, generated-page, `javap`, reflection, Markdown,
dependency/import, exact ten-path scope, status, newline, whitespace, and diff gates. Task 0011
then became the sole detailed Ready NN frontier and is completed below.

Detailed [NN 0011](tasks/0011-unary-tensor-module-composition-and-sequential.md) is Complete as
the next ordered capability. Its real `Sequential` consumer proves the need for a narrow abstract
`UnaryTensorModule extends Module`: a public interface cannot guarantee Module ownership without a
cast or leaky bridge, and an adapter would split state identity from invocation. The selected
container accepts one immutable List snapshot, allows empty exact-reference identity, registers
children atomically under numeric names, and chains exact Tensor references left to right. Only
`Linear`, `LayerNorm`, and `Embedding` move to the base. `BatchNorm` retains explicit
`ForwardContext`; `Dropout` retains explicit context, graph RNG state, and its result carrier. The
task adds no generic Module forward method, shape-pipeline prevalidation, Model behavior,
dependency, architecture, compiler, runtime, backend, or execution work.

Its clean implementation context completed the executable and test diff. The focused Sequential
suite passed 12 tests, the affected existing layer selection passed 20 tests, and the sole
authoritative final NN run passed 17 suites/110 tests with no failures, errors, or skips. Independent
documentation context `/root/nn_0011_docs` reused that unchanged executable evidence, finalized
public/package Javadocs, Training API, glossary, and planning evidence, and passed final Javadoc,
generated-page, exact-surface, external-compilation, Markdown, dependency/import, exact 16-path,
status, newline, whitespace, and diff gates.

Detailed [NN 0012](tasks/0012-vanilla-rnn-cell.md) is Complete. It delivers the smallest truthful
vanilla recurrent capability as final `RnnCell extends Module` with explicit
`forward(input, hidden)` state threading and one Tensor result that is both the cell output and the
next hidden state. The current Model linear, ordinary ADD broadcasting, and TANH expressions can
represent the complete cell without a new operation or scan API.

Its clean implementation context `/root/nn_0012_implementation` completed the executable source,
focused tests, and draft public/package Javadocs without widening the seven-path task scope. The
focused two-suite selection passed 15 tests, and the sole authoritative NN module run passed 19
suites/125 tests with no failures, errors, or skips. Independent documentation context
`/root/nn_0012_docs` reused that unchanged executable evidence, finalized the type Javadoc,
package documentation, glossary, and planning records, and passed generated-Javadoc, exact-
surface/private-state, dependency/import, Markdown, exact seven-path scope, status, newline,
whitespace, and diff gates without repeating stable Java tests.

Detailed [NN 0013](tasks/0013-gru-cell.md) is Complete. Its isolated
implementation context has added final `GruCell extends Module` with the same explicit
`forward(input, hidden)` boundary and sole next-hidden Tensor result as `RnnCell`. Two gate-major
packed matrices and one optional packed
input-side bias use reset, update, candidate order. Current Model linear, SLICE, ADD, SUB, MUL,
SIGMOID, and TANH expressions can represent the fixed reset-after formula without a new Model
operation, split carrier, recurrent scan, hidden state, or result record. The focused two-suite
selection passed 14 tests, and the sole authoritative NN module run passed 21 suites/139 tests
with no failures, errors, or skips. Independent documentation context `/root/nn_0013_docs` reused
that unchanged executable evidence, finalized the type/package Javadocs, glossary, and planning
records, and passed final generated-Javadoc, surface/private-state, dependency/import, Markdown,
exact seven-path scope, status, newline, whitespace, and diff gates without repeating Java tests.
Detailed [NN 0014](tasks/0014-lstm-cell.md) is Complete. Its isolated implementation context added
the planned executable cell, result carrier, focused tests, and draft public/package Javadocs
without widening the Model-only boundary. It fixes
input/forget/candidate/output packing for two matrices and one optional input-side packed bias,
all-zero initialized bias, explicit `forward(input, hidden, cell)`, and one
`LstmCellForwardResult(nextHidden, nextCell)`. Current Model linear, SLICE, ADD, MUL, SIGMOID,
TANH, promotion, and Shape-broadcast contracts can express its complete preflight and formula
without a new operation, initializer, recurrent scan, or dependency. The stabilized focused
two-suite selection passed 15 tests, and the sole authoritative NN module run passed 23 suites
and 154 tests with no skips, failures, or errors. Independent clean documentation context
`/root/nn_0014_docs` reused that unchanged executable evidence, finalized the type/package
Javadocs, glossary, and planning records, and passed final generated-Javadoc, surface/private-
state, dependency/import, Markdown, exact eight-path scope, status, newline, whitespace, and diff
gates without repeating Java tests. Detailed
[NN 0015](tasks/0015-static-packed-rnn-sequence.md) is Complete. Its clean implementation context
added the exact cell-specific container and result carrier, focused packing and contract tests,
and draft public/package Javadocs. The focused two-suite selection passed 14 tests, and the
authoritative NN module run passed 25 suites and 168 tests with no skips, failures, or errors.
Independent clean documentation context `/root/nn_0015_docs` reused that unchanged executable
evidence, finalized the public/package Javadocs, Training API, central glossary, and planning
records, and passed final generated-Javadoc, public/private-surface, dependency/import, Markdown,
exact nine-path scope, status, newline, whitespace, and diff gates without repeating Java tests.
Current Model
SELECT, GATHER, STACK, eager INT64 leaf, static Shape, and vanilla-cell contracts can represent a
construction-time packed sequence from fully static time-major input and a snapshotted Java
`long[]` of lengths. The task fixes stable original-order active batches, compact step outputs,
zero-length handling, and final-hidden restoration without sorting or dense padded cell work. It
also records the honest boundary: runtime Tensor lengths or masks cannot decide loop count or
active batch Shape in the current expression model, and applying a dense `WHERE` after the cell
would not skip padded work. Such inputs require a future genuine recurrent scan/control-flow
contract.

The user explicitly authorized NN 0016 and NN 0017 to be implemented simultaneously. This is a
bounded exception to the normal single-frontier and ascending-order rules: both tasks depend on
completed NN 0015 and their respective completed cells, while neither depends on the other's
production result. Their implementation paths are disjoint. The GRU agent owns only
`GruSequence`, `GruSequenceForwardResult`, its two focused tests, and task-0016 evidence; the LSTM
agent owns only `LstmSequence`, `LstmSequenceForwardResult`, its two focused tests, and task-0017
evidence. Neither implementation agent may edit the shared layers package Javadoc, Training API,
glossary, NN master plan, global roadmap, or the other task. After both executable diffs stabilize,
one coordinated final NN module run validates their union, and one later clean documentation-
focused context independently finalizes both public Javadocs, shared documentation, both task
records, and this master plan. Both tasks are now Complete: the focused GRU selection passed 14
tests, the focused LSTM selection passed 15 tests, and the one coordinated final NN module run
passed 29 suites and 197 tests with no skips, failures, or errors. Joint clean documentation
context `/root/nn_0016_0017_docs` found no executable or architecture defect, finalized the four
public type Javadocs and shared sequence documentation, and synchronized both task records and
this plan. The parallel exception changed implementation order only and did not alter the global
roadmap, dependencies, or architecture boundaries.

The user first authorized planning of the complete typed Model, deferred-dimension,
tokenizer/batching, and recurrent-valid-length program while CPU remains the active global project
area, then explicitly authorized implementation of bounded
[NN 0018](tasks/0018-typed-functional-model-topology.md). Task 0018 is now Complete. Its isolated
implementation context delivered the model-only executable surface and authoritative NN tests;
independent clean documentation context `/root/nn_0018_docs` found no executable, API, or
architecture defect, finalized Javadocs, Training API, glossary, and planning evidence, and passed
the generated-Javadoc, public-surface, external-use, Markdown, scope, import/mechanism, and
whitespace gates without repeating stable Java tests. This interleave changes no dependency or
architecture boundary. Detailed
[NN 0019](tasks/0019-automatic-first-forward-linear-initialization.md) is Complete. Its isolated
implementation context delivered the private reservation lifecycle, unified automatic `Linear`,
strict-load initialization, and focused tests; the authoritative NN suite passed 31 suites and
226 tests. Independent clean documentation context `/root/nn_0019_docs` found no executable,
public-API, architecture, dependency, or scope defect, finalized the six affected production and
package Javadocs plus the Training API, glossary, and planning evidence, and reused the stable
Java evidence because it changed no executable source or test. Final Javadoc/rendered-page,
public/protected-surface, external-use, forbidden-API/import, Markdown, exact fifteen-path,
frontier/status, newline, whitespace, and diff gates passed. Its bounded model-only scope changes
neither the interleave, dependency direction, CPU files, nor the global roadmap. Detailed
[NN 0020](tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md) is Complete. Its
clean implementation context `/root/nn_0020_implementation` delivered the common eight-policy
value and two dispatch routes, migrated Linear, added atomic automatic binding/strict-load support
to every recurrent cell, and added standard-cell/default-state/all-valid sequence overloads with
shared-parameter fresh-producer provenance. Focused selections passed, followed by authoritative
`./gradlew :extensions:nn:clean :extensions:nn:test :extensions:nn:javadoc`: 249 tests, zero skips,
failures, or errors, and warning-free Javadoc. Independent clean documentation context
`/root/nn_0020_docs` found no executable, public-API, architecture, dependency, or scope defect;
finalized the affected Javadocs, package pages, Training API, glossary, task, and master evidence;
reused unchanged Java-test evidence; and passed final NN Javadoc/rendered-page, public/private
surface, external-use, legacy-absence, import/dependency, Markdown, exact thirty-one-path,
frontier/status, newline, no-index, whitespace, and diff gates. Detailed
[NN 0020A](tasks/0020a-initialized-embedding.md) is Complete. Its separate clean implementation
pass added the selected explicit eager table initialization and ordinary trainable semantics for
every row, stabilized focused tests, and passed the authoritative NN suite. Independent clean
documentation context `/root/nn_0020a_docs` found no executable, API, architecture, dependency, or
scope defect; finalized the four affected production/package Javadocs, Training API, glossary,
task, and master evidence; reused the unchanged Java-test evidence; and passed final NN Javadoc/
rendered-page, public/private surface, external-use, forbidden lifecycle/padding/import,
dependency, Markdown, exact nine-path, status/frontier, newline, no-index, whitespace, and diff
gates. Detailed
[NN 0020B](tasks/0020b-stateless-standard-module-factory.md) is Complete. Independent clean
documentation context `/root/nn_0020b_docs` found no executable, API, architecture, dependency,
or scope defect; finalized the ModuleFactory and package Javadocs, Training API, glossary, task,
and master evidence; reused the frozen focused 9/9 and authoritative NN 265/265 test evidence;
and passed final NN Javadoc/rendered-page, surface, external-use, import/dependency,
forbidden-mechanism, Markdown, exact seven-path, status/frontier, newline, no-index, whitespace,
and diff gates. Detailed
[NN 0020C](tasks/0020c-bidirectional-static-recurrent-composition.md) is Complete. It narrowed the
earlier broad directional row to the two distinct traversal orders of one
static time axis and fixed concrete family APIs, independent child/state/seed ownership,
valid-prefix reverse coordinates, original-time alignment, ordered `CONCAT`, directional final
states, validation/effect/Tensor-ID order, and bounded graph-construction costs. NN 0021–0024
remain concise Draft rows without task files.
The first full NN run exposed one stale `ModelTest` assertion that the named-child primitive was
not protected. The coordinator authorized that exact existing surface assertion as task 0020C's
seventeenth path because protected visibility is already required for the direct layer subclass;
no unrelated Model behavior or test changes are authorized.
Clean implementation context `/root/nn_0020c_implementation` froze the executable diff after the
corrected focused selection passed 36/36 and the replacement authoritative NN suite passed
280/280. Independent clean documentation context `/root/nn_0020c_docs` found no executable, API,
ownership, provenance, performance, architecture, dependency, or scope blocker; finalized all
affected Javadocs, package documentation, Training API, glossary, task, and master evidence;
reused the frozen Java-test evidence; and passed final warning-free NN Javadoc/generated-page,
public/protected surface, reflection, external-use, import/dependency, forbidden-mechanism,
Markdown, exact seventeen-path, status/frontier, newline, no-index, whitespace, and diff gates.
NN 0021–0024 remain Draft rows without task specifications, and no NN task is Ready or In progress.
Concurrent CPU/backend planning and global-roadmap changes remain outside this NN planning scope.

The proposed Data, Text, Vision, and Checkpoint master plans are also Draft: their modules do not
exist in the architecture or build, so their first implementation must be a coordinated
architecture, module-boundary, ADR, settings/build, and architecture-test decision rather than a
silent Gradle addition.

The final padding decision uses one valid sequence length per batch row as the sole canonical
metadata for ordinary right padding. It stores neither padding lengths nor a Boolean mask and is
never inferred from Tensor values. Proposed Data initially owns an immutable validated host value;
materializing or binding it as a rank-one non-gradient `INT64` Tensor is deferred until a genuine
runtime input lifecycle exists. A dense Boolean validity mask may be derived on demand only for a
concrete consumer such as attention or loss, while arbitrary masks with holes remain distinct.

The dependency order is now strict. NN 0020 first stabilizes one general parameter-initialization
value, automatic recurrent cells, and static sequence ergonomics. Completed NN 0020A applies that
value eagerly to one explicit complete Embedding table and selects no padding index or special
row: every row is ordinary trainable state. NN 0020B can then provide only
stateless standard construction recipes; and completed NN 0020C fixed independent directional
ownership, reverse-valid-prefix traversal, forward-first final-axis `CONCAT`, and type-specific
final-state contracts. NN 0021 then
coordinates the genuine recurrent scan/control-flow and runtime input-binding prerequisite across
its owning Model, Compiler,
Prepare, Runtime, Engine, and backend layers. Specific valid-length values influence runtime
recurrence behavior and may permit inactive rows/steps to be skipped, but must not change Model
topology or compiled graph structure. Only then may NN 0022 add the new Data-owned valid-length
recurrent API. Current static `long[]` sequence containers remain truthful compatibility/legacy
contracts until that deliberate migration; no new host-static adapter is the target API. Dense
masking alone still constructs full recurrent cell work and cannot satisfy the skipping goal.

The ordered follow-up sequence is deliberate. `Embedding` is another mode-insensitive
parameter-only wrapper. `BatchNorm` follows it as the first layer that must coordinate parameters,
buffers, mode, and pure running-statistic outputs. `Dropout` follows that state-transition design
so its explicit graph-RNG state and evaluation bypass are decided without hidden mutation. State
dictionary/checkpoint work then covers the established parameter/buffer tree. A shared unary
forward contract is implemented only in the minimal shape proven by the actual `Sequential`
consumer.

## Open questions

- Decide whether a concrete future consumer justifies configurable gain, activation, fan mode,
  convolution fan geometry, or another closed algorithm preset. NN 0020 deliberately keeps
  `ParameterInitialization` closed and its fan-based application on the existing positive
  rank-two `[fanOut, fanIn]` Shape contract rather than inventing a public convolution `Fan` value.
- Select a persistent checkpoint codec, schema-version, materialization, and storage boundary only
  when a concrete consumer exists; completed NN 0010 fixes only in-memory state and strict atomic
  validation/load.
- Specify a genuine recurrent Model scan and runtime input binding before exposing a new
  Data-owned valid-length recurrent overload. Its concrete consumer must define the body/subgraph
  boundary, carried-value tuple, dynamic active-set Shape, result metadata, compiler lowering, and
  execution requirements without specializing the graph to one batch's host lengths. Static NN
  0015 does not imply that future architecture.
- Reassess a shared recurrent sequence abstraction only when a concrete consumer proves a useful
  type-safe contract. The three current containers share static packing policy, but LSTM still
  carries and restores cell state that a one-hidden-state abstraction must not erase.
- After NN 0019, reassess whether a concrete consumer justifies model-wide descriptor tracing or a
  public state-schema inspection value. The first automatic capability intentionally initializes
  only the Linear reached by forward traversal or strict dictionary load and exposes no public
  lifecycle/status API.
- If a concrete future consumer requires a padding row that remains invariant through gradients,
  replacement, and optimizer updates, first define that update contract in its owning layer.
  Completed NN 0020A deliberately exposes no padding index, row rewrite, or frozen-row promise;
  vocabulary size and embedding width remain explicit schema and architecture inputs and are
  never inferred from the maximum token ID in one batch.
- Data/Text/Vision module names, packages, dependencies, and architecture ownership require an
  explicit coordinated architecture decision before their first implementation task. Checkpoint
  and its optional Training adapter require their own explicit downstream architecture decision.
- Persistent model/training checkpoint bytes and file I/O belong to the proposed
  [Checkpoint extension](../checkpoint/master-plan.md), downstream of NN and the Engine
  publication/materialization boundary. NN continues to own only in-memory state paths and strict
  Tensor-reference load.

## Decisions made

- `extensions/nn` owns `Module`, `Parameter`, `Buffer`, train/eval behavior, and forward context.
- `extensions/training` depends on nn and owns optimizer algorithms and training orchestration.
- Generic Tensor and operation semantics remain in `modules/model`.
- NN 0001 may run in parallel with CPU 0006D under the recorded isolated-file and model-only
  dependency exception above.
- NN 0002 uses one shared local namespace for direct parameters, buffers, and child modules;
  child attachment is exclusive and permanent, and depth-first declaration order produces
  dot-separated state paths. Local names reserve `.` for the path separator, so recursive state
  paths cannot collide. Its detailed task specifies the resulting APIs and failure behavior.
- NN 0003 replaces the value of exactly one direct binding through protected named `Module`
  methods. The `Parameter` or `Buffer` wrapper and its local name retain identity; a successful
  replacement changes only its current Tensor reference. There is no descriptor-freezing policy,
  version counter, recursive/path replacement, batch atomicity, or thread-safety guarantee in
  this first mutation contract.
- Initializers create eager model leaf Tensors through existing model APIs and receive any random
  source explicitly from the caller. They do not use `GraphRngState`, which represents deferred
  graph-random expressions rather than eager host-data initialization.
- NN 0004 exposes one field-free `ParameterInitializers` class. All eight methods require an
  explicit floating data type and fully static Shape, create fresh unlabeled leaves with
  `requiresGrad == true`, and retain no RNG. Fan-based methods accept only positive rank-two
  `[outFeatures, inFeatures]` weights: Glorot uses unit-gain fan average, and Kaiming uses fixed
  fan-in/ReLU gain. There are no Xavier aliases or ambiguous configurable Kaiming names.
- NN 0004A supersedes only task 0003's assumption that all binding replacement remains protected
  behind the declaring Module. The existing final `Parameter` exposes one public
  `void replace(Tensor)` capability so a generic downstream consumer can update recursively
  discovered parameters. Declaration requires a floating Tensor with `requiresGrad == true`;
  replacement preserves exact declaration-time data type and structural Shape while allowing a
  different Tensor identity, layout, storage, provenance, and label. Buffer replacement remains
  protected and direct through Module. No optimizer algorithm, batch transaction, checkpoint,
  version, or thread-safety contract is introduced.
- NN 0004A preserves recursive discovery and mode preorder while replacing recursive calls with
  explicit-stack depth-first traversal. Parameter and buffer discovery gain the same repeated-
  identity defense as mode propagation, and `train()`/`eval()` retain complete preflight before
  assignment. Qualified prefix text is built only when state is emitted so empty deep chains do
  not retain one String per intermediate level.
- NN 0005 fixes the first `Linear` surface to two caller-Tensor constructors plus one initialized
  constructor with explicit feature counts, bias presence, floating data type, and caller-owned
  random source. Initialized weight delegates to fixed Glorot uniform and optional bias delegates
  to deterministic zeros. The layer exposes stable weight and optional bias `Parameter` handles
  and has one mode-insensitive `forward(Tensor)` that delegates to the existing Model linear
  convenience; it adds no generic layer interface or execution behavior.
- NN 0006 uses one affine-only stateful `LayerNorm`: both `scale` and `bias` are mandatory because
  the current Model affine signature is all-or-none. Caller-supplied construction infers the exact
  normalized Shape from scale; initialized construction receives a positive fully static Shape,
  floating data type, and exact typed epsilon and uses ones then zeros. The layer exposes stable
  parameter handles but no normalized-Shape or epsilon getter because no current consumer needs a
  second configuration-introspection surface. Forward delegates to the current affine Model
  overload and is mode-insensitive.
- NN 0007 exposes only `Embedding(Tensor weight)`, `weight()`, and `forward(Tensor indices)`. The
  supplied table must be floating, gradient-eligible, fully static, rank two, and positive on both
  `[vocabularySize, embeddingSize]` axes. Forward reads the current table once and delegates
  exactly to `weight.embedding(indices)`; it adds no operation, index interpretation, execution,
  or mode branch.
- NN 0007 adds no initialized constructor. Existing generic eager normal/uniform initializers let
  callers choose an explicit table policy, while the fan-based entries are explicitly Linear-
  oriented and no current consumer selects one embedding-specific distribution. It also adds no
  padding index or padding-row invariant: ordinary `Tensor.embedding` has no such semantic, and
  preserving a zero row across parameter replacement or future optimizer updates would require
  a new update/gradient contract or hidden mutation.
- NN 0008 selects final affine `BatchNorm` with positive static rank-one state named `scale`,
  `bias`, `runningMean`, and `runningVariance`; a non-negative stored logical channel axis; exact
  state-typed momentum and epsilon; and one explicit `forward(Tensor, ForwardContext)` method.
  Evaluation delegates to Model inference without state replacement. Training delegates once to
  Model training, then installs its next mean followed by next variance through the existing
  protected direct-buffer operations before returning the normalized output. This is one layer-
  owned sequential transition, not generic atomic multi-binding, execution-side mutation, or a
  training-session/checkpoint contract.
- NN 0009 adds final public `Dropout` with one validated immutable primitive `double` drop
  probability and one `forward(Tensor, GraphRngState, ForwardContext)` method. The layer declares
  no parameter, buffer, child, seed, generator, counter, or hidden RNG state. The supplied context
  alone selects behavior: training delegates once to Model `Tensor.dropout`, while evaluation
  creates no Model operation and returns the exact input and incoming state references.
- NN 0009 introduces NN-owned `DropoutForwardResult` because Model `DropoutResult` truthfully
  describes outputs from one training producer and therefore cannot also describe evaluation
  bypass. The NN record has only `output` and `nextState`; training wraps the exact two references
  from the Model result, and evaluation retains the exact caller references. It exposes no mask,
  mode, probability, producer, or mutable state.
- NN 0010 uses final public `StateKind`, `StateEntry`, and `StateDictionary` values in `nn.module`.
  A Module exports exact Tensor references in combined parameter-then-buffer depth-first order and
  strictly loads a complete path-keyed candidate only after all kind, data-type, Shape, and
  parameter-gradient checks pass. Candidate list order is retained but need not match target
  order; duplicate paths fail at dictionary construction.
- NN 0010 keeps Buffer's existing absence of a declaration schema. Strict load compares a
  buffer's data type and Shape with the target's current binding and does not compare
  `requiresGrad`, because BatchNorm may validly install gradient-eligible next-statistic
  expressions. It exposes no public buffer setter or unchecked batch primitive.
- NN 0010 defines atomicity as complete ordinary validation before sequential non-throwing
  installation under caller coordination. It adds no lock, rollback log, linearizable concurrent
  snapshot, optimizer/session/RNG state, evaluation/materialization, serialization, file I/O,
  codec, version, migration, or persistent checkpoint format.
- NN 0011 selects public abstract `UnaryTensorModule extends Module` rather than an interface or
  adapter so a heterogeneous sequence retains both ownership and unary forwarding at compile time
  without casts, reflection, or a bridge accessor. Final `Sequential` belongs in `nn.module`,
  accepts only `List<? extends UnaryTensorModule>`, permits empty exact-reference identity,
  snapshots structure, and exposes children only through inherited Module discovery.
- NN 0011 uses one package-private Module primitive to preflight every indexed child before
  installing any parent link, avoiding an unreachable partially owned prefix on constructor
  failure without widening Module's public/protected API. Numeric names define stable nested state
  paths. `Linear`, `LayerNorm`, and `Embedding` participate; `BatchNorm` and `Dropout` remain
  excluded because their explicit context/state/result contracts are essential.
- NN 0012 selects final `RnnCell extends Module`, not `UnaryTensorModule`. Its complete forward
  signature consumes explicit input and hidden Tensors and returns exactly one next-hidden Tensor;
  vanilla RNN output and next hidden state are the same value, so a two-component result would
  duplicate one reference without a consumer need. The cell is explicitly excluded from
  `Sequential`; a later recurrent-style container owns sequence composition.
- NN 0012 fixes one shared optional bias, fixed tanh activation, and caller-supplied or explicit-
  source initialization for `inputWeight [hiddenSize, inputSize]`, `hiddenWeight
  [hiddenSize, hiddenSize]`, and optional `bias [hiddenSize]`. Forward composes only existing
  Model linear, ADD, and TANH expressions, with every leading axis treated as ordinary
  right-broadcastable batch metadata rather than a time axis.
- NN 0013 selects final `GruCell extends Module` with one Tensor result and no hidden state. Its
  `inputWeight [3 * hiddenSize, inputSize]`, `hiddenWeight [3 * hiddenSize, hiddenSize]`, and
  optional input-side `bias [3 * hiddenSize]` are packed in reset, update, candidate order. The
  candidate applies reset after the recurrent projection, and the update gate uses
  `candidate + update * (hidden - candidate)`, so update one retains hidden and update zero selects
  the candidate. Independent final-axis slices make gate provenance explicit without a new split
  API or six per-gate matrices.
- NN 0014 selects final `LstmCell extends Module` with explicit input, hidden, and cell Tensors and
  an NN-owned `LstmCellForwardResult(nextHidden, nextCell)`. Its `inputWeight
  [4 * hiddenSize, inputSize]`, `hiddenWeight [4 * hiddenSize, hiddenSize]`, and optional
  input-side `bias [4 * hiddenSize]` are packed in input, forget, candidate, output order. The
  fixed equations are `i = sigmoid(x_i + h_i)`, `f = sigmoid(x_f + h_f)`,
  `g = tanh(x_g + h_g)`, `o = sigmoid(x_o + h_o)`, `nextCell = f * cell + i * g`, and
  `nextHidden = o * tanh(nextCell)`. Independent final-axis slices expose exact provenance.
- NN 0014 initializes both complete packed matrices with Glorot uniform from one caller-owned
  source and initializes the optional complete packed bias to typed zero. It does not apply a
  special forget-bias offset: current initializers cannot produce one direct packed leaf with only
  that interval set to one without hidden host mutation or derived slice/concat parameter state.
  Its packing, input-side-only bias, and zero-bias default are a Synaptik checkpoint schema, not a
  framework-compatibility promise.
- NN 0015 selects final `RnnSequence extends Module` with one exact owned `RnnCell` child, fully
  static time-major input `[time, batch, inputSize]`, explicit initial hidden
  `[batch, hiddenSize]`, and a snapshotted construction-time Java `long[]` of lengths. It creates
  no parameter, buffer, retained state, default hidden value, mask, or unary `Sequential` adapter.
- NN 0015 packing means each unrolled step gathers only original rows whose explicit length
  exceeds that step. A zero Tensor value is ordinary data and never padding. Active rows remain in
  ascending original batch order; the task does not sort. Compact per-step cell results are
  exposed in an immutable list, and each final hidden row is selected from its exit step—or from
  initial hidden for a zero length—then stacked in original order.
- NN 0015 invokes `RnnCell.forward` once per non-empty time step, not once per active row. For
  lengths `[5,3,1]`, five batched calls have active extents `[3,2,2,1,1]` and represent nine
  logical recurrent rows instead of fifteen dense padded rows.
- Runtime Tensor lengths or masks are excluded from NN 0015. Current static Tensor expression
  construction cannot make loop count or active batch Shape depend on runtime values, while dense
  `WHERE` masking would still construct padded cell work. A future dynamic form requires a genuine
  recurrent Model scan/control-flow contract; fixed associative `CUM_SUM` and `CUM_PROD` are not
  that primitive.
- NN 0015 is cell-specific because completed signatures do not prove a type-safe shared contract:
  RNN and GRU carry one hidden Tensor, while LSTM carries and returns hidden plus cell state. The
  explicitly authorized parallel NN 0016 and NN 0017 tasks reuse the proven packing policy
  independently through concrete cell APIs. A shared abstraction remains deferred until all
  three concrete results can be compared without erasing LSTM cell state.
- NN 0016 exposes compact GRU next-hidden outputs plus restored final hidden state through a
  cell-specific result whose structure matches the proven RNN sequence result but whose type does
  not claim cell interchangeability.
- NN 0017 exposes only each step's compact next-hidden Tensor as sequence output while carrying
  compact next-cell internally for recurrence. Its result additionally restores both
  `finalHidden` and `finalCell`; zero-length rows use the corresponding initial-state row, and an
  all-zero request returns both exact initial-state references.
- Parallel NN 0016/0017 implementation is safe only under the exact ownership partition recorded
  in Current status. One joint documentation pass owns every shared documentation path and final
  status synchronization after one authoritative NN validation over the combined executable diff.
- NN 0018 names structural registration `Topology.addModule`, not `layer` or `addLayer`, because
  nested Models and `Sequential` are Modules and the operation establishes ownership and state
  paths rather than asserting a numerical layer category.
- `Model<I, O>` generics describe only the Java forward boundary. Tensor-only callers normally
  infer them with `var`; callers may use their own records for structured inputs and outputs.
  Backward remains compiler/training work.
- NN 0018 collects and validates the complete definition before child attachment, then seals the
  topology. It does not call user forward code during definition and does not add lazy state.
- NN 0019 keeps `Parameter` always bound. `Module` privately reserves future parameter names and
  validators; no public Parameter wrapper exists until one complete layer-local publication or a
  complete strict state load succeeds. Parameter discovery and state export reject unbound
  reservations rather than returning history-dependent partial state.
- NN 0019 adds one ordinary constructor to the existing final `Linear`; it accepts `outFeatures`,
  bias presence, exact floating type, one of the four current fan policies, a non-stochastic
  `RandomGeneratorFactory`, and a seed, but no `inFeatures`. It exposes no `Lazy*` type/factory,
  public bind/build/initialize operation, or initialization-status query. A fresh generator is
  created only during each first-forward attempt and never retained; only the positive static
  final input extent becomes `inFeatures`.
- Automatic initialization is synchronized and atomic only for one layer. It completes and
  verifies parameter publication before the same first call constructs and returns its ordinary
  `Tensor.linear` expression. A failed attempt is retryable and publishes no wrapper, but Tensor
  IDs are not rolled back. Arbitrary `Model.define` Java bodies make whole-model first-forward
  preflight and rollback impossible, so a prior layer may remain initialized when later body work
  fails and an unvisited registered layer may remain uninitialized.
- Strict state load includes reserved paths in its complete target preflight and may bind an
  uninitialized automatic Linear from candidate weight/bias Tensors without running an
  initializer. It retains whole-tree ordinary validate-before-install behavior and makes
  equivalent eager/automatically initialized dictionaries path/kind/type/Shape compatible.
- NN 0020 intentionally replaces the recent public `LinearWeightInitialization` enum with one
  closed immutable `ParameterInitialization` value and migrates current Linear callers atomically,
  without a deprecated alias or recurrent-specific duplicate. Its eight named factories cover
  the current Glorot/Kaiming entries, configured normal/uniform, and RNG-free zero/one; seed,
  `DataType`, Shape/fan schema, parameter order, bias, and module state remain layer-owned facts.
- Generic policy application dispatches through the existing `ParameterInitializers` algorithms.
  Normal/uniform/zero/one preserve their fully static Shape contract; fan presets retain the
  complete positive rank-two `[fanOut, fanIn]` boundary. No callback, registry, public
  convolution-fan abstraction, RNG ownership, or hidden mutable configuration is introduced.
- NN 0020 extends that same reservation lifecycle to the existing final `RnnCell`, `GruCell`, and
  `LstmCell`. One selected general policy applies independently to input then hidden complete
  matrix Shapes; random draws share one generator in that exact order, and optional packed bias
  remains entirely layer-owned zero.
- NN 0020 high-level recurrent constructors accept explicit per-cell type/policy/seed and use the
  exact deterministic JDK `L64X128MixRandom` algorithm internally only for random policies.
  Zero/one create no RNG; cells retain no factory or generator. Existing explicit-`inputSize`
  constructors preserve transient caller-owned `RandomGenerator` control for advanced algorithms
  and lifecycle needs.
- NN 0020 sequence conveniences derive zero hidden/cell states from static batch extent plus the
  cell's explicit hidden-size/type schema. An overload without lengths means all rows have the
  complete static time extent; the derived host array and zero states are per-call values and are
  never retained. The most-explicit caller-cell/state/length APIs remain available.
- One cell instance owns and reuses one parameter set across time, but every represented forward
  call creates fresh Tensor identities/producers. Repeated exact parameter-leaf references and
  carried-state ancestry form the static-unroll fan-out consumed by existing compiler
  exact-identity autograd; NN does not own backward construction.
- An all-zero static length vector invokes no cell. Explicit-state calls retain the existing exact
  state shortcut; default-state calls return their fresh zero state(s). An automatic cell remains
  unbound until a later represented step or strict load.
- NN 0020A, 0020B, and 0020C separate eager initialized Embedding, a stateless construction recipe
  facade, and type-safe directionality respectively. NN 0020A gives every Embedding row ordinary
  trainable semantics and adds no padding index or frozen-row contract. Embedding and factory
  recipes reuse `ParameterInitialization`; `Topology.addModule` remains the sole ownership
  operation, and no shared recurrent base is selected without a type-safe consumer.
- NN 0020B selects one final instance-field-free `ModuleFactory` in `nn.module`. `standard()` returns one
  immutable singleton; `embedding`, `linear`, `rnn`, `gru`, and `lstm` return fresh exact concrete
  types and accept explicit per-layer schema/policy/seed. Recurrent recipes return the matching
  Sequence with its owned Cell, standard Linear uses exact `L64X128MixRandom`, and the facade owns
  no module, topology, configuration, generator, seed sequence, registry, or extension point.
  Existing direct constructors remain the advanced caller-controlled paths.
- NN 0020C narrows directional recurrence to separate concrete RNN, GRU, and LSTM containers for
  the two distinct traversal orders of one static time axis. Each owns identity-distinct direct
  children named `forward` and `backward`; their parameter subtrees, automatic reservations, and
  seeds remain independent, while each cell shares its own parameters only across time.
- NN 0020C reverses only each sample's valid prefix with one compact `GATHER_ND` input per reverse
  depth, flattens reverse-depth outputs, and uses one axis `GATHER` per original time to restore
  original batch/time order. It fixes forward-first final-axis `CONCAT` as the only merge and
  returns separate directional hidden states plus separate directional LSTM cell states.
- NN 0020C adds no generic recurrent base, arbitrary direction collection, `SUM`/configurable
  merge, runtime scan, Tensor lengths, dense mask, hidden state, or `ModuleFactory` recipe. Its
  static graph represents `2 * sum(lengths)` logical compact row applications through
  `2 * max(lengths)` batched cell calls and avoids per-logical-row alignment expressions.
- Input-dependent batch, time, and incoming feature extents are inferred by later binding or batch
  metadata. Hidden/output widths, embedding size, class count, vocabulary size, and recurrent
  hidden width remain explicit architecture or schema decisions.
- The model does not tokenize or pad raw data. Proposed Text owns tokenization and special tokens;
  proposed Data owns sequence layout, padding/truncation policy, Tensor batching, and canonical
  valid lengths. NN must not depend on Text.
- Valid lengths, derived dense masks, and arbitrary masks with holes are distinct contracts. The
  batch stores only valid lengths for ordinary right padding. A runtime recurrent scan may consume
  them without graph specialization only after the full input-binding/execution lifecycle exists.
- `StateDictionary` remains the exact in-memory NN binding boundary. It is not a byte payload,
  tokenizer artifact, optimizer snapshot, or durable checkpoint; a downstream checkpoint adapter
  must materialize through Engine and strict-load only after complete artifact validation.

## Risks

- Coupling layer composition to optimizer APIs would reverse the architecture dependency.
- Letting train/eval mode become backend residency or per-run execution state would blur the lifecycle boundary.
- Giving NN an unspecified, global, seed-managing, or retained mutable random source would obscure
  reproducibility, ownership, and concurrent-use policy. A high-level task may instead fix an
  exact deterministic standard algorithm and require an explicit per-layer seed, while advanced
  constructors retain transient caller-owned sources.
- A reusable initialization policy could accidentally absorb layer schema, Shape/fan derivation,
  bias/order rules, seed/RNG ownership, or mutable callbacks. NN 0020 keeps the value closed and
  algorithm-only, delegates to existing stateless initializers, and leaves those facts with each
  concrete layer.
- A public parameter replacement that omits declaration schema could silently change a layer's
  expected type or Shape; NN 0004A freezes only those logical facts and keeps execution/storage
  facts replaceable.
- Recursive module-tree algorithms must not use Java call-stack depth as a hidden model-size
  limit or accept repeated identities during discovery.
- Batch-normalization or dropout wrappers could accidentally turn pure Model outputs or explicit
  graph RNG state into hidden module mutation; their frontier tasks must make state transitions
  explicit and preserve caller coordination.
- A checkpoint format selected before a concrete persistence consumer would freeze transport
  details into module ownership, while expanding the selected unary contract into a generic
  forward facade would overconstrain context-sensitive and explicitly stateful modules.
- Treating a recurrent cell as unary composition, retaining its hidden state in a Buffer, or
  calling a leading input axis "time" would hide caller-controlled state and confuse batched
  expression construction with sequence recurrence.
- Leaving GRU gate order, reset placement, bias association, or update interpolation implicit
  would make identical parameter Shapes describe different functions; task 0013 fixes each choice
  and requires exact expression-provenance tests.
- Leaving LSTM gate order, bias/default initialization, activation placement, result-component
  order, or cell-update association implicit would make identical parameter Shapes and state
  calls describe different checkpoint schemas or functions; task 0014 fixes and tests each.
- Inferring padding from zero-valued data, invoking cells for padded steps, losing final state
  during active-set compaction, or failing to restore original batch order would silently change
  sequence semantics. The word "packed" also risks confusion with gate-parameter packing or the
  unrelated fixed-associative `CUM_SUM`/`CUM_PROD` family.
- Promising runtime-dynamic masks through dense selection would preserve padded cell expressions
  while pretending to skip them. Conversely, inventing a scan in NN would bypass Model ownership
  of generic operation, Shape, provenance, compiler, and execution semantics.
- Materializing host valid lengths as an eager Tensor before runtime input binding exists could
  freeze batch-specific values into expression construction and misrepresent graph reuse.
- An automatically initialized layer could expose incomplete state paths, make checkpoint behavior
  history-dependent, consume random draws nondeterministically, or leave a partially published
  layer after failure. NN 0019 must fail complete discovery/export while reserved, use explicit
  deterministic source configuration, and publish each layer's state together.
- Retaining caller-owned randomness until an unknown first bind could create hidden lifetime and
  concurrency obligations. The deferred initializer/source contract must make ownership visible.
- Reusing one derived Tensor result across recurrent time steps would collapse distinct operation
  occurrences, while constructing one cell per step would duplicate parameter ownership. Static
  unroll must reuse one cell/parameter set and invoke fresh Tensor operations per represented step.
- A directional recurrent facade could accidentally share parameters/seeds between directions,
  reverse through padded suffixes, erase LSTM cell state, or hide an implicit CONCAT/SUM merge.
  NN 0020C fixes each fact in concrete type-specific contracts and must preserve those selections
  during implementation.
- Treating vocabulary size, output classes, hidden width, or embedding width as input-derived
  would confuse architectural choices with batch facts and may build an invalid model from one
  unrepresentative batch.
- Letting NN own tokenization or raw sequence padding would couple a general Tensor model to text
  and data preparation. Adding Data/Text without the required architecture update would silently
  change module boundaries.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and
follow [the planning guide](../../planning-guide.md).

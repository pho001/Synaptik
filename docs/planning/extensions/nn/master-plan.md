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
- explicit parameter-initialization conveniences over caller-owned sources
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
  module/       public module ownership, named state, and forward-context contracts
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

The next milestone adds recurrent composition without hidden runtime state. Task 0012 starts with
one vanilla tanh `RnnCell` whose hidden Tensor is supplied and returned explicitly on every call.
It remains a direct `Module`: its two-Tensor signature is not a `UnaryTensorModule` and cannot be
placed in `Sequential`. Later GRU and LSTM cells preserve the same explicit-state boundary. A
separate sequence task must then choose static unrolling through concrete cell signatures or
justify a genuinely general recurrent Model scan; existing cumulative sum/product operations are
not recurrent scan primitives.

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
| 0013 | GRU cell | Draft | 0012 | Add one explicitly state-threaded gated recurrent unit cell only after the vanilla cell fixes recurrent ownership, validation, and replacement precedents. |
| 0014 | LSTM cell | Draft | 0013 | Add one explicitly state-threaded long short-term memory cell with truthful hidden/cell-state inputs and outputs; do not hide either state. |
| 0015 | Recurrent sequence composition and scan decision | Draft | 0012–0014 | Decide static unrolling versus a genuinely general recurrent Model scan, then select type-safe cell-specific sequence types or a shared sequence contract only if real signatures prove it. |

## Milestones

- Module, parameter, buffer, and forward-mode contracts
- Explicit eager initialization and first `Linear` layer
- Initial stateful layers and functional conveniences
- Explicit-state stochastic and state-transition layers
- Deterministic state-dictionary/checkpoint contract
- Narrow unary composition only when justified by `Sequential`
- Explicit-state recurrent cells, beginning with vanilla tanh RNN
- Recurrent sequence composition only after the cell signatures prove its type boundary
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
represent the complete cell without a new operation or scan API. Tasks 0013–0015 remain concise
Draft rows and intentionally have no detailed task specifications.

Its clean implementation context `/root/nn_0012_implementation` completed the executable source,
focused tests, and draft public/package Javadocs without widening the seven-path task scope. The
focused two-suite selection passed 15 tests, and the sole authoritative NN module run passed 19
suites/125 tests with no failures, errors, or skips. Independent documentation context
`/root/nn_0012_docs` reused that unchanged executable evidence, finalized the type Javadoc,
package documentation, glossary, and planning records, and passed generated-Javadoc, exact-
surface/private-state, dependency/import, Markdown, exact seven-path scope, status, newline,
whitespace, and diff gates without repeating stable Java tests.

The ordered follow-up sequence is deliberate. `Embedding` is another mode-insensitive
parameter-only wrapper. `BatchNorm` follows it as the first layer that must coordinate parameters,
buffers, mode, and pure running-statistic outputs. `Dropout` follows that state-transition design
so its explicit graph-RNG state and evaluation bypass are decided without hidden mutation. State
dictionary/checkpoint work then covers the established parameter/buffer tree. A shared unary
forward contract is implemented only in the minimal shape proven by the actual `Sequential`
consumer.

## Open questions

- Decide whether a concrete future consumer justifies configurable gain, activation, fan mode,
  convolution fan geometry, or an initializer object abstraction. NN 0004 deliberately fixes only
  unit-gain Glorot and fan-in/ReLU Kaiming for positive rank-two Linear weights.
- Select a persistent checkpoint codec, schema-version, materialization, and storage boundary only
  when a concrete consumer exists; completed NN 0010 fixes only in-memory state and strict atomic
  validation/load.
- At task 0015, decide whether recurrent sequence composition should statically unroll concrete
  cell calls or first requires a genuine general recurrent Model scan. If a container is selected,
  choose cell-specific `RnnSequence`/`GruSequence`/`LstmSequence` contracts unless real signatures
  prove one shared `RecurrentSequence`; do not reserve a broader abstraction prematurely.

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
- NN 0013 and 0014 must preserve explicit hidden, and hidden-plus-cell, state respectively. NN
  0015 must not reinterpret `CUM_SUM` or `CUM_PROD` as recurrent scan and must not invent a broad
  recurrent abstraction before the concrete cell signatures establish a type-safe contract.

## Risks

- Coupling layer composition to optimizer APIs would reverse the architecture dependency.
- Letting train/eval mode become backend residency or per-run execution state would blur the lifecycle boundary.
- Giving NN a default or retained random source would obscure reproducibility, ownership, and
  concurrent-use policy; initializer tasks must keep those responsibilities with the caller.
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

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and
follow [the planning guide](../../planning-guide.md).

# Training API

## Purpose and implementation status

This reference records the implemented neural-network parameter boundary consumed by future
training and the remaining planned training concepts without inventing optimizer APIs.
`extensions/nn` now provides module-owned `Parameter` and `Buffer` declarations, recursive
parameter discovery, one public schema-validated parameter replacement capability, and strict
in-memory module-tree state export/load, including private parameter reservations that fail closed
until a concrete layer publishes a complete real parameter group. It also provides the narrow
`UnaryTensorModule` subtype
and immutable numeric-child `Sequential` composition for modules whose complete forward signature
is exactly `Tensor forward(Tensor)`, three explicit-state recurrent cells, and matching
cell-specific statically packed sequence containers. The typed `Model<I,O>` root and its sealed
functional topology now provide descriptive composition above those modules without adding a
training or execution facade.
`extensions/training`, public gradient publication, optimizer behavior, and prepared execution
are not implemented. The compiler now provides a public immutable functional-gradient request
value and a bounded package-private one/two-stage reverse-mode integration path.

Training will own backend-independent optimizer algorithms and session concepts. The compiler will own global automatic differentiation (autograd), while concrete backends will own any backend-specific lowering or fused optimizer route.

The accepted compiler design builds one or two reverse-mode stages with existing public Tensor
operations before one combined forward/backward capture. Model task 0025 and Compiler tasks 0004
and 0006 are Complete.
Public Tensors gain no gradient/backward lifecycle state. The current internal `TRAINING_STEP`
mode uses the same `FunctionalGradientRequest` contract as `FORWARD_AND_BACKWARD`; it adds no
optimizer update. Runtime gradient delivery, optimizer behavior, preparation, and execution
remain planned.

Package-private `GraphCompiler` currently returns mode-neutral `GraphCompilation`. A
`TRAINING_STEP` result may carry the same combined forward/backward graph as
`FORWARD_AND_BACKWARD`; a `FORWARD_ONLY` result has no BACKWARD nodes and empty gradient results.
This internal graph-stage result is not the later `CompileArtifacts` aggregate or a training
session/result type.

The current compiler request supports one or two bounded stages, exact forward or first-stage-
gradient output references, aligned explicit seeds or scalar default seeds, ordered identity-
unique targets, and ERROR/ZERO disconnected behavior. The compiler preflights each complete
selected slice before creating formulas, captures forward and all derivative roots once, and
retains per-node derivative order beside unchanged graph phase. These facts do not expose a
public training workflow, deliver a gradient at runtime, choose a parameter update, select a
backend, prepare a schedule, or execute training.

## Current NN typed Model composition contract

`Model<I,O>` is an NN `Module` whose generic parameters describe only its caller-visible Java
forward boundary. Tensor-only callers normally let Java infer `Model<Tensor,Tensor>` through
`var`; structured callers may use their own records for several inputs or outputs. NN introduces
no tuple, tokenizer, batch, or text-specific type, and the caller does not define backward code.

`Model.define(definition)` invokes one definition callback with a short-lived `Topology`. During
that callback, `topology.addModule(name, module)` records the exact module candidate under a
descriptive local name and returns that exact concrete type for local use. Collection changes no
parent link. The topology is sealed after every success or failure path, and a captured reference
cannot add a late child. Only after the callback returns a non-null forward body does the Model
preflight the complete ordered snapshot and permanently attach every child. A callback, null
result, name, repeated-identity, cycle, or ownership failure publishes no partial model and leaves
every previously unowned candidate unattached.

For example, the shared stateless `ModuleFactory.standard()` facade constructs two fresh
input-width-inferring `Linear` layers. `Topology.addModule` then gives those exact modules stable
descriptive names and permanent Model ownership:

```java
ModuleFactory modules = ModuleFactory.standard();

var model = Model.define(topology -> {
    Linear hidden = topology.addModule(
            "hidden",
            modules.linear(
                    64,
                    true,
                    DataType.FLOAT32,
                    ParameterInitialization.glorotUniform(),
                    41L));
    Linear output = topology.addModule(
            "output",
            modules.linear(
                    10,
                    true,
                    DataType.FLOAT32,
                    ParameterInitialization.glorotUniform(),
                    42L));

    return (Tensor input) -> output.forward(hidden.forward(input).relu());
});
```

The factory call is construction, not ownership or registration. Every recipe call returns a
fresh existing concrete type and keeps the architectural width, bias, data type, initialization
policy, and seed explicit. The factory retains none of those choices and owns no random source or
module. Its Linear recipe selects the exact deterministic JDK `L64X128MixRandom` factory; direct
constructors remain available when a caller needs a different deterministic factory, supplied
state, supplied cells, or explicit recurrent state and lengths. Constructor validation, effects,
state paths, and forward behavior remain owned by each concrete layer or sequence.

Suppose the input has Shape `[batch, 32]`. The first call initializes `hidden.weight` as `[64, 32]`
before constructing the hidden linear expression. That expression has final extent 64, so the
same Java traversal initializes `output.weight` as `[10, 64]` before constructing the returned
output expression. The hidden width 64 and output width 10 are architecture choices; only each
incoming final feature extent is inferred. Later inputs may change compatible leading batch or
time Dimensions, but the exact data type and final feature extent must continue to match each
initialized layer.

Calling `model.forward(input)` passes the exact non-null input to the retained body once and
returns its exact non-null result. If the body throws or returns null, already constructed prefix
expressions or module-local initialization effects remain and no rollback occurs. State discovery
and state dictionaries use paths such as `hidden.weight` and `output.bias`; `train()` and
`eval()` propagate through the same owned tree.

This **Model topology** is the permanent NN module-ownership tree and its state-path namespace. It
is not the Tensor producer graph, a modules/model `CompiledGraphModel`, or a compiled/runtime
schedule. Forward constructs ordinary Java and Tensor expressions only. It does not capture,
differentiate, compile, prepare, execute, update parameters, persist checkpoints, tokenize input,
or pad a batch. `Model.define` itself performs no parameter initialization. The existing final
`Linear` has one constructor that automatically initializes input-dependent state while that
layer is first reached by forward traversal; this is not a general Model build, bind, initialize,
or status lifecycle. Models inherit Module's mutable state/mode lifecycle and are not thread-safe;
callers coordinate forward construction with replacement, loading, and mode changes when one
consistent view matters.

## Current NN automatic parameter initialization contract

`ParameterInitialization` is the common closed initialization-policy value used by eager
initialized `Embedding` and automatic `Linear`, RNN, GRU, and LSTM layers. Its exact eight
factories are `glorotNormal()`,
`glorotUniform()`, `kaimingReluNormal()`, `kaimingReluUniform()`, `normal(mean,
standardDeviation)`, `uniform(lowerBoundInclusive, upperBoundExclusive)`, `zeros()`, and `ones()`.
Configured arguments must be finite; a normal standard deviation is non-negative and uniform
bounds are strictly increasing. The value owns only that algorithm selection and arguments. It
does not own a Shape, data type, Tensor, Parameter, random generator, seed, parameter order, gate
order, fan value, or bias policy. Fan presets derive fan-in and fan-out independently from each
complete rank-two Shape when applied.

`ParameterInitializers.initialize(shape, type, policy)` accepts exactly zero/one and creates no
random generator. Its overload with a `RandomGenerator` accepts exactly the other six policies
and forwards that exact caller-owned source to the corresponding eager initializer. There is no
public kind, callback, registry, stateful initializer, or default random source.

`Linear` remains one final type. Its supplied-state constructors and explicit
`inFeatures`/`outFeatures` constructor remain immediately initialized. The automatic constructor
accepts only architectural `outFeatures`, bias presence, exact floating data type, one
`ParameterInitialization`, a deterministic `RandomGeneratorFactory`, and a seed. Construction
creates no generator, Tensor, Tensor identifier, or `Parameter` and exposes no
`LazyLinear`, `Linear.lazy`, public `bind`, `build`, `initialize`, or initialization-status API.

The first compatible `forward(input)` on that layer has two phases:

```text
validate input and parameter Shapes/counts
  -> for a sampling policy, create one generator from the retained factory and seed
  -> create weight, then optional exact-zero bias; zero/one never invoke the factory
  -> validate and publish the complete direct Parameter group
  -> construct and return this call's ordinary Tensor.linear expression
```

The first phase creates eager host-backed parameter leaves; it does not numerically run the layer.
The second phase creates the same visible PERMUTE/MATMUL/optional-ADD expression used by the eager
constructors. Parameter Tensor identifiers therefore precede the identifiers of the first returned
linear expression. Later compatible calls create no generator or parameter Tensor and construct
only their ordinary expressions.

The automatic path infers one positive static final input extent as `inFeatures`. Weight Shape and
therefore fan values remain layer-owned. Bias is always exact typed zero regardless of weight
policy. The path does not infer output width, class count, vocabulary size, embedding width,
recurrent hidden width, bias, data type, policy, random algorithm, or seed. Prevalidation failures
happen before sampling or Tensor creation. A later initializer or allocation failure can consume
random draws, memory, or opaque Tensor identifiers, but it publishes no wrapper; a retry starts
with a fresh generator from the same factory and seed. Zero/one attempts create no generator and
never invoke the factory. Once a complete group is published, an expression-construction failure
does not undo it.

Initialization is synchronized only for concurrent first calls on the same `Linear`. Its direct
reservation completion uses a release/acquire publication gate so an accessor or discovery race
observes either an unbound failure or the complete direct group. Parameter replacement, strict
load, tree traversal, mode changes, and arbitrary functional Model bodies retain their external-
coordination requirements. A Model body is not one transaction: if an earlier layer initializes
and later body work fails, the earlier layer remains initialized. A registered but unvisited
automatic layer remains uninitialized and makes complete recursive parameter discovery and state
export fail closed.

The automatic recurrent-cell constructors similarly retain explicit hidden width, bias choice,
floating parameter type, policy, and seed while reserving `inputWeight`, `hiddenWeight`, and
optional `bias`. The first compatible represented cell call infers only a positive static input
width. Sampling policies create one fresh standard `L64X128MixRandom` stream per attempt and draw
the complete packed input matrix before the complete packed hidden matrix; each Shape derives its
own fan values. Zero/one policies create no generator. Optional bias is layer-owned typed zero and
never draws. RNN uses unpacked matrices, GRU owns reset/update/candidate packing, and LSTM owns
input/forget/candidate/output packing. Strict state loading may bind the complete reserved group
without initialization. Failure before publication binds nothing and is retryable from the seed;
successful publication precedes cell-expression construction and is cell-locally synchronized.

`Embedding` instead has no input-dependent parameter dimension, so its initialized constructor is
eager. It accepts explicit positive vocabulary size and embedding size, exact floating type, one
common policy, and a seed, then initializes the complete `[vocabularySize, embeddingSize]` table
before declaring its sole permanent `weight` parameter. The six random policies each use one
fresh seeded standard `L64X128MixRandom` source and the four-argument dispatcher once. Zero and
one use the three-argument dispatcher once and never create an RNG. Fan policies use the whole
rank-two Shape with vocabulary size as fan-out and embedding size as fan-in. Successful
construction consumes the initializer's one Tensor identifier; a later sampling, allocation, or
identifier failure does not roll back completed effects but publishes no parameter wrapper or
layer.

Neither size is inferred from token IDs or a batch. Embedding width remains an architecture
choice, while a future Text vocabulary supplies vocabulary identity and size explicitly. Every
table row, including row zero, is ordinary gradient-eligible trainable state. `Embedding` exposes
no padding index, special or frozen row, automatic row rewrite, optimizer/gradient masking, or
runtime skipping. Future Text input preparation owns padding-token identity, and future Data input
preparation owns canonical valid lengths; neither proposed boundary is current tokenizer/Data API
or changes the NN parameter schema.

## Current NN unary composition contract

`Module` remains the general owner of named state, children, and train/eval mode and has no
universal forward method. `UnaryTensorModule` is the narrower public subtype for a module whose
complete forward contract accepts one non-null Tensor and returns one non-null Tensor. Current
`Linear`, `LayerNorm`, and `Embedding` layers participate. `BatchNorm` does not, because it
requires an explicit `ForwardContext` and may transition running-statistic buffers. `Dropout`
does not, because it requires both an explicit context and caller-threaded graph random-number-
generator state and returns a result carrying output plus next state.

`Sequential` accepts one `List<? extends UnaryTensorModule>`, snapshots it, and permanently owns
the exact children under decimal names `0`, `1`, and so on. It passes the exact input reference to
child `0`, passes each exact child result to the next child once, and returns the exact final
result. Empty composition returns the exact input. A later failure preserves already constructed
prefix expressions and suppresses the remaining suffix; the container performs no pipeline-wide
validation, rollback, caching, flattening, fusion, compilation, or execution.

For example, a sequence containing a `Linear` child followed by a `LayerNorm` child exposes state
paths such as `0.weight`, `0.bias`, `1.scale`, and `1.bias` and composes declaratively as:

```text
input reference
  -> child 0 Linear.forward(input)
  -> child 1 LayerNorm.forward(exact child-0 result)
  -> exact child-1 result
```

Inherited recursive mode propagation and state-dictionary export/load follow those numeric child
paths. The sequence does not create a training session, choose an optimizer, publish gradients,
or imply that any Tensor expression has been executed.

## Current NN recurrent composition contract

`RnnCell`, `GruCell`, and `LstmCell` each describe one recurrent step and extend `Module`
directly. RNN and gated recurrent unit (GRU) cells accept input plus hidden state and return one
next-hidden Tensor. The long short-term memory (LSTM) cell accepts input, hidden state, and cell
state and returns both next states. None retains caller-threaded recurrent state, and none fits the
one-input contract of `UnaryTensorModule` or `Sequential`.

`RnnSequence`, `GruSequence`, and `LstmSequence` are intentionally cell-specific one-directional
containers. Each permanently owns one exact matching cell and shares the same static packing
policy without claiming that the different cell-state signatures are interchangeable. Their
calls are:

```java
RnnSequenceForwardResult rnnResult =
        rnnSequence.forward(input, initialHidden, lengths);
GruSequenceForwardResult gruResult =
        gruSequence.forward(input, initialHidden, lengths);
LstmSequenceForwardResult lstmResult =
        lstmSequence.forward(input, initialHidden, initialCell, lengths);
```

Here `input` must have a fully static time-major Shape `[time, batch, inputSize]`,
`initialHidden` must have Shape `[batch, hiddenSize]`, and the LSTM call additionally requires
`initialCell` with that same Shape. `lengths` is a Java `long[]` with one value in `[0, time]` per
original batch row. Each method validates directly from the caller array and, when at least one
step is represented, clones it immediately before traversal; neither array is retained. Callers
must coordinate mutation throughout validation and any snapshot.

Each sequence also has an overload that omits lengths and treats every row as valid for the full
static time extent. Overloads that omit recurrent state create fresh eager typed-zero state with
Shape `[batch, hiddenSize]`, the cell's parameter type, no name, and `requiresGrad == false`.
RNN/GRU create one such leaf; LSTM creates distinct hidden and cell leaves. The state is local to
that call and never retained. When every length is zero, no cell is invoked, an automatic cell
remains unbound, the output list is empty, and the final-state accessors return the exact explicit
or freshly derived initial references.

The copied lengths determine an **active batch** at each time step: original batch rows whose
length exceeds that step, kept in ascending original order. The sequence statically constructs
one compact batched cell expression for every non-empty step. It never examines Tensor values to
decide activity, so an all-zero input row remains active whenever its explicit length includes the
step.

For example, lengths `[5, 3, 1]` produce five packed outputs with active batch extents
`[3, 2, 2, 1, 1]`:

```text
time step          0      1      2      3      4
active rows      0,1,2   0,1    0,1     0      0
active extent      3      2      2      1      1
```

There are five Java cell calls—one per non-empty time step—and their compact Shapes represent
nine logical recurrent row applications rather than fifteen dense padded rows. For every
container, `packedOutputs().get(t)` is the exact next-hidden Tensor returned by its cell for that
compact step. `finalHidden()` restores one row per original batch entry: row 0 comes from step 4,
row 1 from step 2, and row 2 from step 0. A zero-length row instead uses its row from
`initialHidden`.

RNN and GRU carry only hidden state. Their result types therefore contain `packedOutputs` and
`finalHidden`; when all lengths are zero, the list is empty and `finalHidden()` is the exact
`initialHidden` reference. LSTM carries hidden and cell state. `LstmSequence` publishes only its
compact next-hidden Tensors by step, carries each exact compact next-cell Tensor internally, and
returns both `finalHidden()` and `finalCell()` so the caller can continue recurrence. A
zero-length LSTM row uses the corresponding initial hidden and initial cell rows; an all-zero
request returns both exact initial-state references.

The matching `BidirectionalRnnSequence`, `BidirectionalGruSequence`, and
`BidirectionalLstmSequence` containers add exactly the two traversal orders of that same static
time axis. Each permanently owns identity-distinct cells under `forward` and `backward`, so state
paths begin, for example, with `forward.inputWeight` and `backward.inputWeight`. The cells have
equal hidden width and exact parameter type but independent wrappers, parameter Tensor identities,
automatic reservations, and seeds. Supplied cells may differ in bias presence. No parameter or
recurrent state is shared between directions; each cell shares only its own parameters across
time.

For example, this GRU request uses input Shape `[3, 3, inputSize]`, lengths `[3, 1, 2]`, hidden
width 64, and fresh default states:

```java
BidirectionalGruSequence encoder = new BidirectionalGruSequence(
        64,
        true,
        DataType.FLOAT32,
        ParameterInitialization.glorotUniform(),
        41L,
        42L);

BidirectionalGruSequenceForwardResult encoded =
        encoder.forward(input, new long[] {3, 1, 2});
```

Forward traversal visits original times from zero upward. Backward traversal reverses each row's
valid prefix separately: its first compact input uses coordinates `(2,0)`, `(0,1)`, and `(1,2)`,
not a reversal of the complete padded time axis. Later backward hidden outputs are gathered back
to their original time and ascending active-row order. The three `packedOutputs()` Shapes are
`[3, 128]`, `[2, 128]`, and `[1, 128]`; each final axis contains the exact forward features first
and aligned backward features second. `forwardFinalHidden()` is the forward state after each
row's last valid element, while `backwardFinalHidden()` is the backward state after original time
zero. A zero-length row instead keeps its corresponding directional initial-state row.

RNN and GRU bidirectional results expose the merged compact hidden-output list plus separate
forward and backward final hidden states. The LSTM counterpart merges only hidden outputs and
returns four continuation values in forward-hidden, forward-cell, backward-hidden, backward-cell
order. Omitting explicit state creates distinct typed-zero leaves in that same direction/state
order. An all-zero request invokes neither cell, leaves automatic cells unbound, and returns the
exact explicit or derived initial references.

Every represented step calls the same Java cell and therefore reuses the same exact Parameter
leaf Tensor identities. Select, gather, gate/cell operations, and restored-state producers are
fresh for each step, and later states retain temporal ancestry through earlier producers. This is
static expression provenance: compiler capture can see repeated exact parameter identity fan-out,
and the existing compiler reverse-mode contract combines contributions for one identity-unique
target. The sequence API itself neither defines numerical gradients nor exposes training or
execution.

This is static Tensor-expression construction, not numerical execution. For one direction,
`T = max(lengths)` produces `T` batched cell calls representing `S = sum(lengths)` compact logical
rows. A bidirectional container constructs `2T` batched calls representing `2S` logical rows,
plus bounded reverse-alignment expressions per represented time. These counts describe the
constructed provenance only. They do not prove backend lowering, physical kernel skipping,
fusion, kernel count, execution cost, or a public training workflow, and different Java length
values specialize different Tensor-expression topology.

Current Model now provides fixed RNN-tanh, reset-after GRU, and LSTM recurrent-scan expression
construction with an ordinary `INT64[batch]` `validLengths` Tensor. Those six Tensor receiver
overloads create one flat multi-output producer with dense time-major output and explicit final
states. They do not read length values, statically unroll a cell, own parameters, or execute.
Current Compiler inference rejects the family, current autograd rejects it before derivative
Tensor construction, and no Engine or backend route exists; runtime length binding and BPTT are
therefore not current training capabilities.

The Model scan does not silently replace these NN containers. Current `RnnSequence`,
`GruSequence`, `LstmSequence`, and their bidirectional counterparts still accept Java `long[]`
construction-time lengths, specialize expression topology, and return compact per-step outputs.
They expose no Tensor-length overload, arbitrary mask with holes, stacked or multidimensional
recurrent facade, arbitrary direction collection, or configurable merge. The current
`ModuleFactory` constructs only the existing one-directional RNN, GRU, and LSTM Sequences; the
concrete bidirectional constructors already expose their two seeds and no factory recipe is
current. Later NN work must choose any runtime-length convenience or migration only after the
complete Model-to-execution path exists. The initialized `Embedding` remains eager.

## Current NN parameter update contract

A normal module declaration accepts a floating Tensor with `requiresGrad == true`. A concrete
input-dependent module may instead reserve private names and validators, but no incomplete or
nullable `Parameter` wrapper is exposed. Its reservation occupies the direct namespace and state
order immediately. Until its complete group is published by layer-local forward initialization or
strict state load, direct/recursive parameter discovery and state export fail with the first
unbound qualified path.

A final
`Parameter` wrapper privately retains the declaration-time exact data type and structural Shape.
A generic downstream consumer can obtain that wrapper from `Module.parametersRecursively()` and
call `Parameter.replace(value)`. Replacement requires a non-null Tensor with the same exact data
type, a structurally equal Shape, and `requiresGrad == true`; a failed validation leaves the
previous exact binding current.

A successful replacement preserves the wrapper, local name, and recursive discovery path while
installing the exact supplied Tensor reference. Tensor identity, layout, host storage,
provenance, and label are deliberately outside compatibility and may differ. A Tensor obtained
before replacement, and expressions already constructed from it, remain unchanged.

This is one mutable binding operation, not an optimizer step. It is not thread-safe and provides
no version, update sequencing, or cross-binding consistency guarantee. Callers must coordinate it
with forward construction and other work that requires a consistent binding view. `Buffer` has no
corresponding public wrapper update operation; module subclasses retain only the protected
direct-buffer replacement contract for their own state transitions.

## Current NN state-dictionary contract

`Module.stateDictionary()` exports one complete module tree as an immutable, encounter-ordered
`StateDictionary`. Every `StateEntry` contains a relative dot-separated path, a `StateKind` of
`PARAMETER` or `BUFFER`, and the exact current Tensor reference. Export visits each module's direct
parameters, then direct buffers, then child subtrees in child-registration order. It creates a
shallow value snapshot: later wrapper replacement does not change an earlier dictionary, but no
Tensor is copied, evaluated, materialized, detached, or transferred to a new owner.

`Module.loadStateDictionary(dictionary)` is strict and path-keyed, so candidate entry order need
not match export order. The candidate must contain exactly the complete target path set. Before
installing anything, load checks missing paths, unexpected paths, then each target's kind, exact
data type, and structural Shape; a parameter also requires `requiresGrad == true`. A buffer uses
the target's current data type and Shape as its schema and deliberately ignores gradient
eligibility because buffer role, not a Tensor flag, excludes it from optimizer discovery. After
all checks pass, the exact candidate Tensor references are installed through existing wrappers or
newly prepared reserved-parameter wrappers in target traversal order.

Reserved parameter paths participate in that complete target set. Strict load validates their
retained layer-specific schema, prepares real wrappers for the entire candidate tree, and publishes
each complete reserved group without invoking its initializer, creating a generator, copying a
Tensor, or consuming random draws. An equivalent initialized automatic and eager `Linear` use the
same `weight`/optional `bias` paths, kinds, exact types, and Shapes, so their state dictionaries are
compatible when their inferred and explicit feature widths match. The same rule covers automatic
and explicit RNN/GRU/LSTM cells through `inputWeight`, `hiddenWeight`, and optional `bias`; packed
gate order and exact Shapes remain cell-owned schema.

Supplied-table and initialized `Embedding` instances use the same single `weight` path, Parameter
kind, exact floating type, positive rank-two Shape, and gradient-eligibility schema. Their state
dictionaries are therefore compatible when those exact table facts match. Loading or replacement
installs the complete exact candidate table; no row receives padding-specific treatment.

Ordinary validation failure changes no binding. This validate-before-install guarantee assumes
caller-coordinated access: module state export and load are not thread-safe, linearizable,
synchronized, or simultaneously visible to racing readers. Tensors and expressions retained
before a successful load remain unchanged; later wrapper reads and forward construction observe
the new bindings.

The dictionary is an in-memory module-state boundary, not a persistent checkpoint. It contains no
bytes, files, codec, format version, migration rules, evaluated Tensor values, optimizer state,
`TrainingSession` state, graph random-number-generator state, compiler artifact, prepared
execution, runtime state, or backend state. A future training workflow may coordinate module state
with separately owned optimizer or session state, and a future persistence adapter may encode
materialized values, but neither API exists now.

For example, a root module with parameter `weight`, buffer `step`, and child `encoder` containing
parameter `scale` and buffer `runningMean` exports this order:

```text
weight                    PARAMETER
step                      BUFFER
encoder.scale             PARAMETER
encoder.runningMean       BUFFER
```

Loading those same four paths in another list order succeeds when every binding is compatible;
omitting `encoder.runningMean` fails before `weight` or any other target changes. This example
demonstrates ordering, path identity, and atomic validation only—it does not serialize or execute
Tensor values.

## Planned concepts

- `extensions/nn` currently owns `Parameter` and `Buffer` declarations plus strict in-memory
  module-tree state export/load. Training will consume discovered `Parameter` wrappers through
  their bounded replacement capability, while `ParameterGroup` will describe optimizer-group
  settings.
- `Optimizer` implementations such as SGD, Adam, and AdamW will define mathematical updates without importing CPU, Metal, or CUDA modules.
- `TrainingSession` and `TrainingStep` will coordinate forward/backward execution, gradient publication, and optimizer updates through shared lifecycle contracts.

No optimizer signatures, default hyperparameters, update sequencing, gradient-to-parameter
mapping, persistent checkpoint format, or optimizer exception types are stable yet. They will be
defined by focused extension tasks after model, compiler, runtime, and publication contracts
exist. The current NN-owned replacement and state-dictionary rules do not imply those future
training or persistence contracts.

## Planned public initial flow

```text
compile forward + backward graph
  -> prepare owned partitions
  -> run forward/backward
  -> publish gradients
  -> optimizer.step()
```

The initial optimizer step is backend-agnostic. A later architecture version may compile optimizer updates into the graph, but that direction requires the architecture update described by the contract.

## Boundary example

Adam computes moment estimates and a parameter update from gradients. That mathematical algorithm belongs to training. Choosing a fused Metal kernel for the update belongs to Metal prepare. A `MetalOptimizerBridge` inside training would reverse the required dependency direction and is not a supported design.

## Related contracts

- [Training graph](../architecture/training-graph.md)
- [Autograd guide](../user-guide/autograd.md)
- [Training guide](../user-guide/training.md)
- [Training master plan](../planning/extensions/training/master-plan.md)

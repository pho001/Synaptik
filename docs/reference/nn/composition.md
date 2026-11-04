# Typed [Model](../../glossary.md#term-model) composition and [parameter](../../glossary.md#term-parameter) initialization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A neural-network (**[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)**) [module](../../glossary.md#term-module) owns named state and child modules. Its **[topology](../../glossary.md#term-topology)** is that
[ownership](../../glossary.md#term-ownership-meanings) tree, whereas the [Tensor](../../glossary.md#tensor) expression [graph](../../glossary.md#graph) records the computation built by a forward
call. Initialization supplies starting parameter values; **forward** then describes operations
using those values without numerically evaluating them.

```mermaid
flowchart TD
    accTitle: Module ownership and expression dependencies are different structures
    accDescr: A model owns hidden and output layers. A forward call uses their current weights to build a computation from input through two projections. Shared state is owned by modules but referenced by expression dependencies.
    M["Model: owns children"] --> H["hidden layer: owns weight"]
    M --> O["output layer: owns weight"]
    X["input Tensor"] --> P["hidden projection expression"] --> Q["output projection expression"]
    H -. "current weight binding" .-> P
    O -. "current weight binding" .-> Q
```

Solid arrows at the top identify module ownership; the input-to-output arrows identify computation
flow. Dotted arrows connect the two structures through current state [bindings](../../glossary.md#term-binding). This is a conceptual
relationship, not a claim that a forward call executes either projection.

<a id="current-nn-typed-model-composition-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) typed [Model](../../glossary.md#term-model) composition contract

`Model<I,O>` is an NN [`Module`](../../glossary.md#term-module) whose generic [parameters](../../glossary.md#term-parameter) describe only its caller-visible Java
forward boundary. [Tensor](../../glossary.md#tensor)-only callers normally let Java infer `Model<Tensor,Tensor>` through
`var`; structured callers may use their own records for several inputs or outputs. NN introduces
no tuple, [tokenizer](../../glossary.md#frozen-whitespace-tokenizer), [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters), or text-specific type, and the caller does not define backward code.

`Model.define(definition)` invokes one definition [callback](../../glossary.md#term-callback) with a short-lived [`Topology`](../../glossary.md#term-topology). During
that callback, `topology.addModule(name, module)` records the exact module candidate under a
descriptive local name and returns that exact concrete type for local use. Collection changes no
parent link. The topology is [sealed](../../glossary.md#term-sealed) after every success or failure path, and a captured reference
cannot add a late child. Only after the callback returns a non-null forward body does the Model
[preflight](../../glossary.md#term-preflight) the complete ordered [snapshot](../../glossary.md#term-snapshot) and permanently attach every child. A callback, null
result, name, repeated-[identity](../../glossary.md#term-identity-meanings), cycle, or [ownership](../../glossary.md#term-ownership-meanings) failure publishes no partial model and leaves
every previously unowned candidate unattached.

For example, the shared stateless `ModuleFactory.standard()` [facade](../../glossary.md#term-facade) constructs two fresh
input-width-inferring [`Linear`](../../glossary.md#term-linear-meanings) layers. `Topology.addModule` then gives those exact modules stable
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
fresh existing concrete type and keeps the architectural width, [bias](../../glossary.md#term-bias), [data type](../../glossary.md#data-type--datatype), initialization
policy, and [seed](../../glossary.md#term-seed) explicit. The factory retains none of those choices and owns no random source or
module. Its Linear recipe selects the exact deterministic [JDK](../../glossary.md#term-java-development-kit-jdk) `L64X128MixRandom` factory; direct
constructors remain available when a caller needs a different deterministic factory, supplied
state, supplied cells, or explicit [recurrent state](../../glossary.md#backpropagation-through-time-and-recurrent-state) and lengths. Constructor [validation](../../glossary.md#term-validation), effects,
state paths, and forward behavior remain owned by each concrete layer or sequence.

Suppose the input has [Shape](../../glossary.md#shape) `[batch, 32]`. The first call initializes `hidden.weight` as `[64, 32]`
before constructing the hidden linear expression. That expression has final [extent](../../glossary.md#term-extent) 64, so the
same Java traversal initializes `output.weight` as `[10, 64]` before constructing the returned
output expression. The hidden width 64 and output width 10 are architecture choices; only each
incoming final [feature](../../glossary.md#term-feature) extent is inferred. Later inputs may change compatible leading batch or
time [Dimensions](../../glossary.md#dimension), but the exact data type and final feature extent must continue to match each
initialized layer.

Calling `model.forward(input)` passes the exact non-null input to the retained body once and
returns its exact non-null result. If the body throws or returns null, already constructed prefix
expressions or module-local initialization effects remain and no [rollback](../../glossary.md#term-rollback) occurs. State discovery
and state dictionaries use paths such as `hidden.weight` and `output.bias`; `train()` and
`eval()` propagate through the same owned tree.

This **Model topology** is the permanent NN module-ownership tree and its state-path namespace. It
is not the Tensor [producer](../../glossary.md#term-producer) [graph](../../glossary.md#graph), a modules/model [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel), or a compiled/[runtime](../../glossary.md#term-runtime)
schedule. Forward constructs ordinary Java and Tensor expressions only. It does not [capture](../../glossary.md#term-capture-meanings),
differentiate, [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), execute, update parameters, persist [checkpoints](../../glossary.md#term-checkpoint), tokenize input,
or pad a batch. `Model.define` itself performs no parameter initialization. The existing final
`Linear` has one constructor that automatically initializes input-dependent state while that
layer is first reached by forward traversal; this is not a general Model build, bind, initialize,
or status [lifecycle](../../glossary.md#lifecycle). Generic Models inherit Module's [mutable](../../glossary.md#term-mutable) state/mode lifecycle and are not [thread-safe](../../glossary.md#term-thread-safe);
callers coordinate forward construction with replacement, loading, and mode changes when one
consistent [view](../../glossary.md#term-view-meanings) matters. The separate owned capability described above acquires a static source
after one eval [prototype](../../glossary.md#term-prototype) forward and serializes high-level operations; it does not synchronize
inherited Module mutators or change Model.define semantics.

<a id="current-nn-automatic-parameter-initialization-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) automatic [parameter](../../glossary.md#term-parameter) initialization contract

`ParameterInitialization` is the common closed initialization-policy value used by [eager](../../glossary.md#term-eager-evaluation)
initialized [`Embedding`](../../glossary.md#embedding) and automatic [`Linear`](../../glossary.md#term-linear-meanings), [RNN](../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../glossary.md#term-gated-recurrent-unit-gru), and [LSTM](../../glossary.md#term-long-short-term-memory-lstm) layers. Its exact eight
factories are `glorotNormal()`,
`glorotUniform()`, `kaimingReluNormal()`, `kaimingReluUniform()`, `normal(mean,
standardDeviation)`, `uniform(lowerBoundInclusive, upperBoundExclusive)`, `zeros()`, and `ones()`.
Configured [arguments](../../glossary.md#term-argument) must be [finite](../../glossary.md#term-finite); a normal [standard deviation](../../glossary.md#term-standard-deviation) is non-negative and uniform
bounds are strictly increasing. The value owns only that algorithm selection and arguments. It
does not own a [Shape](../../glossary.md#shape), [data type](../../glossary.md#data-type--datatype), [Tensor](../../glossary.md#tensor), Parameter, random generator, [seed](../../glossary.md#term-seed), parameter order, gate
order, fan value, or [bias](../../glossary.md#term-bias) policy. Fan presets derive [fan-in](../../glossary.md#term-fan-in) and [fan-out](../../glossary.md#term-fan-out) independently from each
complete [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape when applied.

`ParameterInitializers.initialize(shape, type, policy)` accepts exactly zero/one and creates no
random generator. Its overload with a `RandomGenerator` accepts exactly the other six policies
and forwards that exact caller-owned source to the corresponding eager initializer. There is no
public kind, [callback](../../glossary.md#term-callback), [registry](../../glossary.md#term-registry), stateful initializer, or default random source.

`Linear` remains one final type. Its supplied-state constructors and explicit
`inFeatures`/`outFeatures` constructor remain immediately initialized. The automatic constructor
accepts only architectural `outFeatures`, bias presence, exact floating data type, one
`ParameterInitialization`, a deterministic `RandomGeneratorFactory`, and a seed. Construction
creates no generator, Tensor, Tensor identifier, or `Parameter` and exposes no
`LazyLinear`, `Linear.lazy`, public `bind`, `build`, `initialize`, or initialization-status [API](../../glossary.md#term-application-programming-interface-api).

The first compatible `forward(input)` on that layer has two phases:

```text
validate input and parameter Shapes/counts
  -> for a sampling policy, create one generator from the retained factory and seed
  -> create weight, then optional exact-zero bias; zero/one never invoke the factory
  -> validate and publish the complete direct Parameter group
  -> construct and return this call's ordinary Tensor.linear expression
```

The first phase creates eager [host](../../glossary.md#term-host)-backed parameter [leaves](../../glossary.md#term-leaf); it does not numerically [run](../../glossary.md#term-run) the layer.
The second phase creates the same visible PERMUTE/[MATMUL](../../glossary.md#matrix-multiplication--matmul)/optional-ADD expression used by the eager
constructors. Parameter Tensor identifiers therefore precede the identifiers of the first returned
linear expression. Later compatible calls create no generator or parameter Tensor and construct
only their ordinary expressions.

The automatic path infers one positive static final input [extent](../../glossary.md#term-extent) as `inFeatures`. [Weight](../../glossary.md#term-weight) Shape and
therefore fan values remain layer-owned. Bias is always exact typed zero regardless of weight
policy. The path does not infer output width, class count, vocabulary size, embedding width,
recurrent hidden width, bias, data type, policy, random algorithm, or seed. Prevalidation failures
happen before [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) or Tensor creation. A later initializer or allocation failure can consume
random draws, memory, or [opaque](../../glossary.md#term-opaque) Tensor identifiers, but it publishes no wrapper; a retry starts
with a fresh generator from the same factory and seed. Zero/one attempts create no generator and
never invoke the factory. Once a complete group is published, an expression-construction failure
does not undo it.

Initialization is synchronized only for concurrent first calls on the same `Linear`. Its direct
reservation completion uses a [release](../../glossary.md#term-release)/acquire [publication](../../glossary.md#term-publication) gate so an accessor or discovery [race](../../glossary.md#term-race)
observes either an unbound failure or the complete direct group. Parameter replacement, strict
load, tree traversal, mode changes, and arbitrary functional [Model](../../glossary.md#term-model) bodies retain their external-
coordination requirements. A Model body is not one [transaction](../../glossary.md#term-transaction): if an earlier layer initializes
and later body work fails, the earlier layer remains initialized. A registered but unvisited
automatic layer remains uninitialized and makes complete recursive parameter discovery and state
export [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation).

The automatic recurrent-cell constructors similarly retain explicit hidden width, bias choice,
floating parameter type, policy, and seed while reserving `inputWeight`, `hiddenWeight`, and
optional `bias`. The first compatible represented cell call, [descriptor](../../glossary.md#term-descriptor)-valid `forwardScan`, or
RNN `forwardGeneralScan` infers only a positive static input width. A [scan](../../glossary.md#term-cumulative-scan) binds even when time
or [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) is zero; static [unrolling](../../glossary.md#term-unrolling) with no represented step still leaves the cell unbound.
Sampling policies create one fresh standard `L64X128MixRandom` stream per attempt and draw
the complete packed input [matrix](../../glossary.md#term-matrix-meanings) before the complete packed hidden matrix; each Shape derives its
own fan values. Zero/one policies create no generator. Optional bias is layer-owned typed zero and
never draws. RNN uses unpacked matrices, GRU owns reset/update/candidate packing, and LSTM owns
input/forget/candidate/output packing. Strict state loading may bind the complete reserved group
without initialization. Failure before publication binds nothing and is retryable from the seed;
successful publication precedes cell-expression construction and is cell-locally synchronized.

`Embedding` instead has no input-dependent parameter [dimension](../../glossary.md#dimension), so its initialized constructor is
eager. It accepts explicit positive vocabulary size and embedding size, exact floating type, one
common policy, and a seed, then initializes the complete `[vocabularySize, embeddingSize]` table
before declaring its sole permanent `weight` parameter. The six random policies each use one
fresh seeded standard `L64X128MixRandom` source and the four-argument dispatcher once. Zero and
one use the three-argument dispatcher once and never create an [RNG](../../glossary.md#graph-rng-state--graphrngstate). Fan policies use the whole
rank-two Shape with vocabulary size as fan-out and embedding size as fan-in. Successful
construction consumes the initializer's one Tensor identifier; a later sampling, allocation, or
identifier failure does not roll back completed effects but publishes no parameter wrapper or
layer.

Neither size is inferred from [token](../../glossary.md#term-token) IDs or a batch. Embedding width remains an architecture
choice, while a future Text vocabulary supplies vocabulary [identity](../../glossary.md#term-identity-meanings) and size explicitly. Every
table row, including row zero, is ordinary [gradient](../../glossary.md#term-gradient)-eligible trainable state. `Embedding` exposes
no [padding](../../glossary.md#padding) index, special or frozen row, automatic row rewrite, [optimizer](../../glossary.md#term-optimizer)/gradient masking, or
[runtime](../../glossary.md#term-runtime) skipping. Future Text input preparation owns padding-token identity, and future Data input
preparation owns [canonical](../../glossary.md#term-canonical-form) valid lengths; neither proposed boundary is current [tokenizer](../../glossary.md#frozen-whitespace-tokenizer)/Data API
or changes the NN parameter [schema](../../glossary.md#term-schema).


---

[Previous topic](../training/recovery.md) · [Next topic](convolution.md)

# Task 0019: Automatic First-Forward [Linear](../../../../glossary.md#term-linear-meanings) Initialization

> **Editorial reading context — not a status update.** The first compatible input descriptor
> tells this Linear its input width; its contents are not sampled to infer architecture.
> The layer creates stored weights and optional bias, publishes the complete local group,
> then constructs the ordinary projection expression. A later layer's failure does not undo
> that initialized prefix. Synchronizing first publication is not synchronizing the whole Model.
> Today's [automatic initialization contract](../../../../reference/nn/composition.md#current-nn-automatic-parameter-initialization-contract)
> explains that two-phase boundary and the successor policy name; [linear composition](../../../../reference/tensor/matmul-linear.md#linear-projection-convenience)
> explains the returned expression. Historical identifiers below remain unchanged.

## Status

Complete

## Goal

Let the existing final public [`Linear`](../../../../glossary.md#term-linear-meanings) infer its input-[feature](../../../../glossary.md#term-feature) width and initialize its complete
[parameter](../../../../glossary.md#term-parameter) set automatically inside the first successful `forward(input)` call. The user declares
only the architectural output width and initialization configuration; there is no separate [lazy](../../../../glossary.md#term-lazy-evaluation)
type, factory, bind/build/initialize call, or public initialization-state [API](../../../../glossary.md#term-application-programming-interface-api).

```java
var model = Model.define(topology -> {
    Linear hidden = topology.addModule(
            "hidden",
            new Linear(
                    64,
                    true,
                    DataType.FLOAT32,
                    LinearWeightInitialization.GLOROT_UNIFORM,
                    randomFactory,
                    41L));
    Linear output = topology.addModule(
            "output",
            new Linear(
                    10,
                    true,
                    DataType.FLOAT32,
                    LinearWeightInitialization.GLOROT_UNIFORM,
                    randomFactory,
                    42L));

    return (Tensor input) -> output.forward(hidden.forward(input).relu());
});
```

For input [Shape](../../../../glossary.md#shape) `[batch, 32]`, the first call initializes `hidden.weight [64, 32]`, constructs the
hidden expression, initializes `output.weight [10, 64]` from that expression [descriptor](../../../../glossary.md#term-descriptor), and
returns the usable output expression from the same call. Later calls may change leading [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) or
time [Dimensions](../../../../glossary.md#dimension), but their exact [data type](../../../../glossary.md#data-type--datatype) and final feature [extent](../../../../glossary.md#term-extent) must remain compatible.

[Mental model](../../../../glossary.md#term-mental-model) for one automatically initialized `Linear`:

```text
construction
  -> validate immutable configuration
  -> reserve future weight/bias names privately
  -> create no Parameter, Tensor, generator, or Tensor ID

first forward(input)
  phase 1: validate input/configuration/counts
           -> create one seeded generator
           -> create unpublished weight then optional zero bias
           -> validate and atomically publish the complete layer state
           -> verify initialization completed
  phase 2: read the published bindings
           -> construct and return the ordinary Tensor.linear expression

later forward(input)
  -> validate input compatibility
  -> create no generator or parameter Tensor
  -> construct and return another ordinary Tensor.linear expression
```

Every public `Parameter` remains a real, non-null, fully bound wrapper with one permanent [schema](../../../../glossary.md#term-schema).
Before the first successful forward or strict state load, future parameter paths exist only as
private [Module](../../../../glossary.md#term-module)-owned declaration [metadata](../../../../glossary.md#term-metadata). Complete parameter discovery and state export fail
clearly instead of returning a partial tree.

## [Scope](../../../../glossary.md#term-scope)

- Keep exactly one final public [`Linear`](../../../../glossary.md#term-linear-meanings). Preserve the existing caller-supplied Tensor constructors,
  explicit `inFeatures`/`outFeatures` eager constructor, accessors, forward composition, and
  replacement behavior.
- Add one ordinary public constructor on that same type:

  ```java
  public Linear(
          long outFeatures,
          boolean bias,
          DataType dataType,
          LinearWeightInitialization weightInitialization,
          RandomGeneratorFactory<? extends RandomGenerator> randomGeneratorFactory,
          long seed)
  ```

  It accepts no `inFeatures`. Construction validates and retains [immutable](../../../../glossary.md#term-immutable) configuration, reserves
  [`weight`](../../../../glossary.md#term-weight) then optional [`bias`](../../../../glossary.md#term-bias), and creates no random generator, [Tensor](../../../../glossary.md#tensor), Tensor ID, or [Parameter](../../../../glossary.md#term-parameter).
- Add public enum `LinearWeightInitialization` with exactly the four existing positive rank-two
  Linear policies: <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`GLOROT_NORMAL`, `GLOROT_UNIFORM`, `KAIMING_RELU_NORMAL`, and
  `KAIMING_RELU_UNIFORM`. `Linear` dispatches each value directly to the matching existing
  `ParameterInitializers` method. Do not add a generic initializer abstraction, custom callback,
  registry, alias, gain/fan configuration, or default policy.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Do not add `LazyLinear`, `Linear.lazy`, `LazyModule`, a public `bind`, `build`, `initialize`,
  `isBound`, [descriptor](../../../../glossary.md#term-descriptor)-tracing [API](../../../../glossary.md#term-application-programming-interface-api), or [Model](../../../../glossary.md#term-model)-wide initialization [lifecycle](../../../../glossary.md#lifecycle).
- Add private reservation records/slots inside [`Module`](../../../../glossary.md#term-module). A reserved parameter name immediately
  occupies the existing shared direct namespace and its declaration position, but exposes no
  `Parameter` wrapper until complete [publication](../../../../glossary.md#term-publication).
- Add exactly these protected final subclass primitives to `Module`:

  ```java
  protected final void reserveParameter(String name, Consumer<Tensor> validator)
  protected final void bindReservedParameters(List<Tensor> values)
  protected final boolean parameterReservationsBound()
  protected final Parameter boundParameter(String name)
  ```

  They are lifecycle implementation support for concrete NN modules, not a public user-facing
  binding API. Do not add another production abstraction.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [binding glossary](../../../../glossary.md#term-binding))</span>
- `reserveParameter` applies the existing local-name validation first, rejects a null validator,
  records one future parameter in encounter order, and retains the exact deterministic validator.
  A validator returns normally or throws and must have no externally visible side effect; this
  task supplies only private Linear validators.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- `bindReservedParameters` accepts exactly one non-null Tensor per outstanding direct reservation
  in reservation order. It runs all reservation validators, constructs all real `Parameter`
  wrappers as unpublished locals, and publishes every direct reservation together only after all
  ordinary validation and wrapper construction succeeds. It never binds a subset.
- `parameterReservationsBound()` reports whether the receiving module has no outstanding direct
  parameter reservation. A module that never reserved a parameter returns `true`.
  `boundParameter(name)` returns the exact wrapper only after every direct reservation is bound;
  it rejects null, missing, wrong-kind, or still-reserved names.
- Preserve one declaration-ordered internal parameter namespace for eager and reserved entries.
  Eager `parameter(...)` still creates and publishes a real wrapper immediately. Parameters,
  reservations, buffers, and children keep the existing collision rules.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- Make `parameters()` fail with `IllegalStateException` naming the first unbound direct reservation.
  Make `parametersRecursively()` preflight the whole identity-defended tree and fail with the first
  qualified unbound path rather than return a history-dependent partial snapshot.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [identity glossary](../../../../glossary.md#term-identity-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
  `stateDictionary()` has the same complete-tree failure rule. Buffer-only discovery, children,
  mode propagation, and topology ownership remain available.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Validate the new Linear constructor in this order: positive `outFeatures`; non-null `dataType`;
  non-null `weightInitialization`; non-null `randomGeneratorFactory`; floating [data type](../../../../glossary.md#data-type--datatype); then
  `randomGeneratorFactory.isStochastic() == false`. Retain the exact factory and seed, but never a
  created generator. Select no default algorithm, seed, data type, bias, or policy.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- On every automatic-path `forward(input)`, reject null first, require the configured exact data
  type, rank at least one, a static final Dimension, and a positive final extent. Leading rank and
  Dimensions do not become parameter schema. Before the first factory creation, preflight exact
  weight Shape <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [schema glossary](../../../../glossary.md#term-schema))</span>`[outFeatures, inFeatures]`, optional bias [Shape](../../../../glossary.md#shape) `[outFeatures]`, checked counts,
  Model Java-array limits, and every other configuration/input failure knowable without sampling.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- During the first compatible forward, create exactly one fresh generator with
  `randomGeneratorFactory.create(seed)`. Initialize weight first in row-major order through the
  selected existing initializer; initialize optional bias second through exact typed zeros, which
  consumes no random draw. Retain neither generator nor caller input.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- Publish neither wrapper until both local parameter Tensors and their validators succeed. A
  failed attempt leaves all reservations unbound and is retryable. A new attempt creates a fresh
  generator from the same factory and seed. Completed draws, local allocations, and consumed
  opaque Tensor IDs are not rolled back or reused.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- The first forward has two ordered internal phases. Complete and verify layer-local initialization
  before calling either `Tensor.linear` overload. The same call then reads each binding once and
  returns the actual ordinary Model expression. If expression construction fails after state
  publication, the parameters stay initialized and a later compatible call does not initialize
  them again.
- Later automatic-path forwards create no generator and no parameter Tensor. They require the
  configured exact data type and the already bound final feature extent. Compatible leading
  Dimensions may vary, including a different rank. Existing eager/supplied Linear forward keeps
  its current Model promotion/rank/contraction behavior unchanged.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- `weight()` fails clearly before automatic initialization. For a configured bias, `bias()` also
  fails rather than return an empty Optional; for a no-bias automatic Linear it always returns
  `Optional.empty()`. After initialization both accessors return the exact stable wrappers used by
  discovery, replacement, state export, and strict load.
- Serialize only the one-time initialization critical section on the exact Linear instance. Use
  a release/acquire publication gate owned by Module so parameter access and discovery after
  successful initialization observe all wrappers. Compatible concurrent first forwards initialize
  once; one incompatible contender may win and fix the schema while the other then fails against
  it. Do not synchronize later expression construction or claim general Module thread safety.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
- Extend strict `Module.loadStateDictionary(...)` so reserved parameter paths participate in the
  complete target schema. For an uninitialized automatic Linear, candidate <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`weight` must be
  gradient-eligible, have the configured exact floating type, and have positive fully static
  Shape <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`[outFeatures, inFeatures]`; optional bias must match exact type and `[outFeatures]`.
  The weight's final extent establishes `inFeatures`. No initializer or [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) is used.
- Preserve strict whole-tree validate-before-install semantics. Missing/unexpected paths, kind,
  type, Shape, gradient eligibility, and every reservation validator are checked for the complete
  tree; all new wrappers are constructed as unpublished locals before one existing binding is
  replaced or one reservation is published. Ordinary failure changes no binding or reservation.
  JVM-fatal resource failure keeps the existing documented limitation.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- An equivalent [eager](../../../../glossary.md#term-eager-evaluation) or initialized automatic Linear uses the same `weight`/optional `bias` paths,
  kinds, data types, and Shapes. State dictionaries load in either direction when schemas match,
  and loading eager state into an uninitialized automatic model retains the exact candidate Tensor
  references without a forward initialization pass.
- In a functional Model, automatic initialization follows the Java forward traversal actually
  executed. If an earlier Linear initializes and a later layer or caller body fails, the earlier
  state remains initialized. There is no Model-wide trace, validation pass, transaction, rollback,
  or public <span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [transaction glossary](../../../../glossary.md#term-transaction); [rollback glossary](../../../../glossary.md#term-rollback))</span>`Model.initialize` operation. A registered module not traversed by a successful forward
  may remain uninitialized; complete recursive discovery/export then fails until every reservation
  reachable in the owned tree is bound by forward traversal or strict state load.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Add focused tests for exact API surface, reservation ordering/collisions/discovery, constructor
  and first-forward validation order, deterministic initialization, Tensor-ID ordering and failure
  effects, accessors, variable leading Dimensions, expression provenance, retry, concurrency,
  replacement, strict load, functional Model composition, unvisited modules, and partial Model
  failure effects.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Finalize every affected public/protected/package Javadoc, the Training API, glossary, and planning
  evidence through the mandatory separate clean documentation context.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any separate `Lazy*` public type, factory, interface, mixin, marker, wrapper, or naming family.
- A public/protected initialization-state query or per-layer/[model](../../../../glossary.md#term-model) `bind`, `build`, `initialize`,
  dry-run, example-input, input-specification, or descriptor-tracing method.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Automatic initialization for Embedding, LayerNorm, BatchNorm, RNN, GRU, LSTM, sequence
  containers, or another existing module; those require their own concrete policy and consumer.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [module glossary](../../../../glossary.md#term-module))</span>
- Inferring `outFeatures`, hidden width, class count, vocabulary size, embedding width, recurrent
  hidden size, bias presence, data type, initializer policy, random algorithm, or seed.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [data type glossary](../../../../glossary.md#data-type--datatype); [seed glossary](../../../../glossary.md#term-seed))</span>
- Retaining a [mutable](../../../../glossary.md#term-mutable) `RandomGenerator`, choosing a default generator, deriving per-layer seeds,
  serializing generator objects, custom initialization callbacks, or graph RNG state.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Model-wide first-forward preflight, rollback, initialization status, branch coverage, tracing,
  parameter-schema export before initialization, or optimizer construction convenience.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [rollback glossary](../../../../glossary.md#term-rollback); [parameter glossary](../../../../glossary.md#term-parameter); [schema glossary](../../../../glossary.md#term-schema); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Returning an empty or partial parameter snapshot/state dictionary for an uninitialized owned
  tree, placeholder/uninitialized Tensor values, nullable Parameter state, or incomplete wrappers.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Relaxed, partial, remapped, converting, or best-effort state load; checkpoint bytes, codecs,
  files, versions, materialization, checksums, manifests, optimizer state, or Training resume.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [materialization glossary](../../../../glossary.md#materialization); [optimizer state glossary](../../../../glossary.md#term-optimizer-state); [resume glossary](../../../../glossary.md#schema-serialization-and-checkpoint))</span>
- New Tensor/operation semantics, numerical evaluation, compiler capture, autograd, backend
  support, lowering, execution, device storage, or a claim that <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [evaluation glossary](../../../../glossary.md#term-evaluation); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [autograd glossary](../../../../glossary.md#autograd); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [device glossary](../../../../glossary.md#term-device))</span>`forward` calculates values.
- Data/Text/Vision batching, [tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer), valid-length, [padding](../../../../glossary.md#padding)/[mask](../../../../glossary.md#term-boolean-mask), recurrent [scan](../../../../glossary.md#term-cumulative-scan), or image work.
- Gradle, dependency, architecture-contract, ADR, architecture-test, global-roadmap, CPU, other
  module, conformance, integration, or unrelated refactoring changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Task 0001: Module, Parameter, Buffer, and forward context](0001-module-parameter-buffer-and-forward-context-foundation.md)<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Buffer glossary](../../../../glossary.md#term-buffer))</span>
- [Task 0002: Module-tree ownership](0002-module-tree-ownership-and-recursive-mode-propagation.md)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Task 0003: Binding replacement](0003-validated-parameter-and-buffer-binding-replacement.md)<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding))</span>
- [Task 0004: Eager parameter initializers](0004-explicit-eager-parameter-initializers.md)<span data-glossary-navigation="true"> ([Eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- [Task 0004A: Parameter/traversal hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Task 0005: Linear layer](0005-linear-layer.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [Task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary))</span>
- [Task 0018: Typed functional Model topology](0018-typed-functional-model-topology.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [topology glossary](../../../../glossary.md#term-topology))</span>
- [Checkpoint master plan](../../checkpoint/master-plan.md)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Architecture constraints

- `extensions/nn` owns module state, layer composition, parameters, buffers, and forward behavior
  and may depend only on <span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>`modules/model`. This task adds no downstream or execution [dependency](../../../../glossary.md#term-dependency).
- [`Parameter`](../../../../glossary.md#term-parameter) remains a final wrapper around one non-null real Tensor with declaration-time exact
  data type and structural Shape. Automatic initialization must not weaken that invariant.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Tensor identity, descriptor, and expression provenance remain immutable. Initializer calls create
  eager host-backed parameter leaves; <span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable); [eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host))</span>`forward` then constructs ordinary Tensor expressions and
  neither evaluates nor executes them.
- The Model/topology ownership tree stays immutable after definition. Automatic initialization
  changes only layer-owned parameter declarations along the actually invoked forward path.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [topology glossary](../../../../glossary.md#term-topology); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Compiler owns autograd and graph compilation; Training owns optimizer algorithms/orchestration;
  Engine/Runtime/backends own execution/materialization. NN must not import or reproduce them.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [optimizer glossary](../../../../glossary.md#term-optimizer); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [backends glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization))</span>
- The current strict in-memory state dictionary remains the only task-owned load boundary. Durable
  checkpoint transport remains downstream and unimplemented.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Planning is non-authoritative. If preserving the Parameter/state-dictionary invariants requires
  a dependency, architecture rule, model-wide transaction, or another public lifecycle surface,
  stop and report the conflict instead of forcing this task to Complete.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [transaction glossary](../../../../glossary.md#term-transaction); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

The design fits the current architecture: private declaration [metadata](../../../../glossary.md#term-metadata) and layer-local [publication](../../../../glossary.md#term-publication)
remain [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned state composition, and no dependency or [module](../../../../glossary.md#term-module)-boundary change is required.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns declaration slots, discovery, state load, and
  parameter publication.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication))</span>
- `io.github.pho001.synaptik.nn.layers` — owns the single public Linear layer and its automatic
  first-forward behavior.
- `io.github.pho001.synaptik.nn.initialization` — owns the closed [Linear](../../../../glossary.md#term-linear-meanings) initialization policy.
- `io.github.pho001.synaptik.model.tensor`, `.shape`, and `.datatype` — supply unchanged Tensor
  descriptors, Shape/Dimension values, data types, eager parameter leaves, and linear expressions.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>

No package is added or renamed.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — sole owner of private reserved-parameter metadata,
  namespace/order integration, complete discovery gating, group publication, and strict-load
  integration.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `io.github.pho001.synaptik.nn.layers.Linear` — the one existing final layer type; owns automatic
  validation, one-time initialization synchronization, retained factory/seed configuration, and
  ordinary forward composition.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [synchronization glossary](../../../../glossary.md#term-synchronization); [seed glossary](../../../../glossary.md#term-seed))</span>
- `io.github.pho001.synaptik.nn.initialization.LinearWeightInitialization` — closed selection among
  the four existing fan policies justified only by this Linear consumer.
- `io.github.pho001.synaptik.nn.module.ModuleDeferredParameterTest` — same-package reservation,
  namespace, discovery, publication, retry, and visibility tests.
- `io.github.pho001.synaptik.nn.module.StateDictionaryTest` — extends the owning strict-load suite
  with reserved targets and whole-tree validate-before-install behavior.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `io.github.pho001.synaptik.nn.layers.LinearTest` — owns the exact unified public surface,
  accessors, compatibility validation, provenance, and replacement behavior.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.nn.layers.LinearInitializationTest` — owns deterministic first-forward
  initialization, failure/Tensor-ID ordering, retry, and concurrent first-call evidence.
- `io.github.pho001.synaptik.nn.module.ModelTest` — owns functional two-layer inference, unvisited
  reservations, and partial-forward state effects.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>

## Public and protected [API](../../../../glossary.md#term-application-programming-interface-api)

The exact new public surface is one constructor on the existing type plus one closed enum:

```java
public final class Linear extends UnaryTensorModule {
    // Existing three constructors and weight(), bias(), forward(Tensor) remain.

    public Linear(
            long outFeatures,
            boolean bias,
            DataType dataType,
            LinearWeightInitialization weightInitialization,
            RandomGeneratorFactory<? extends RandomGenerator> randomGeneratorFactory,
            long seed)
}

public enum LinearWeightInitialization {
    GLOROT_NORMAL,
    GLOROT_UNIFORM,
    KAIMING_RELU_NORMAL,
    KAIMING_RELU_UNIFORM
}
```

The enum exposes only ordinary enum-generated API and those four [constants](../../../../glossary.md#term-constant). [Dispatch](../../../../glossary.md#term-dispatch) remains
inside [`Linear`](../../../../glossary.md#term-linear-meanings); it adds no public initialization method. There is no new public method on
`Linear`, [`Module`](../../../../glossary.md#term-module), [`Model`](../../../../glossary.md#term-model), or another type.

`Module` adds exactly these protected final methods:

```java
protected final void reserveParameter(String name, Consumer<Tensor> validator)
protected final void bindReservedParameters(List<Tensor> values)
protected final boolean parameterReservationsBound()
protected final Parameter boundParameter(String name)
```

Their implementation uses only [JDK](../../../../glossary.md#term-java-development-kit-jdk) collection/function types and private Module-owned declaration
slots. `bindReservedParameters` binds all currently outstanding direct reservations or none; it is
not recursive. Linear owns [synchronization](../../../../glossary.md#term-synchronization) around the check/create/bind sequence. Module strict
load separately prepares the complete tree and installs it under its existing caller-coordinated
contract.

## Initialization, failure, and [concurrency](../../../../glossary.md#term-concurrency) contract

Construction has no [parameter](../../../../glossary.md#term-parameter) [side effect](../../../../glossary.md#term-side-effect). The automatic-path first forward performs these steps
in exact order:

1. Reject null input.
2. Read the immutable input descriptor and validate exact configured type, positive rank, static
   positive final extent, weight/bias Shapes, checked element counts, and Java-array limits.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent); [weight glossary](../../../../glossary.md#term-weight); [Shapes glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element))</span>
3. Enter the Linear's one-time initialization critical section only if reservations appear
   unbound, repeat the compatibility check after acquiring it, and let an already successful
   contender win.
4. Create a fresh generator from the retained exact factory and [seed](../../../../glossary.md#term-seed).
5. Call the selected existing weight initializer once. It consumes one matching row-major random
   call per weight element and, on success, creates the weight Tensor and its ID.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
6. If configured, create the exact-zero [bias](../../../../glossary.md#term-bias) [Tensor](../../../../glossary.md#tensor) and its next ID without consuming [RNG](../../../../glossary.md#graph-rng-state--graphrngstate).
7. [Run](../../../../glossary.md#term-run) every retained reservation validator and construct all `Parameter` wrappers as locals.
8. Publish weight and optional bias together, then publish one release completion gate. Verify the
   reservations report bound before leaving the initialization phase.
9. Read the exact stable [bindings](../../../../glossary.md#term-binding) once and call the matching existing `Tensor.linear` overload.
   Its PERMUTE/MATMUL/optional ADD Tensor IDs follow parameter publication.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>

Steps 1–2 fail before generator creation, random draws, Tensor allocation, or Tensor-ID allocation.
A generator/fan-initializer failure publishes no state. A failure after a local Tensor exists can
consume draws, memory, and [opaque](../../../../glossary.md#term-opaque) IDs, but the local values remain undiscoverable and a retry uses
a fresh generator with the same seed. A failure in step 9 keeps successfully published state.

Each [Linear](../../../../glossary.md#term-linear-meanings) serializes only steps 3–8. The [Module](../../../../glossary.md#term-module) declaration completion gate is volatile (or an
equivalent explicit Java Memory [Model](../../../../glossary.md#term-model) [release](../../../../glossary.md#term-release)/acquire mechanism): [publication](../../../../glossary.md#term-publication) writes occur before
the release, and accessors/discovery read the acquire before reading bound slots. A discovery [race](../../../../glossary.md#term-race)
with layer-local initialization must either reject the unbound reservation or observe the complete
published set, never a partial direct set. This narrow rule does not make Module tree traversal,
replacement, mode changes, state load, or arbitrary forward bodies generally [thread-safe](../../../../glossary.md#term-thread-safe) or
linearizable; callers still [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) those operations.

## State discovery and strict-load contract

[Parameter](../../../../glossary.md#term-parameter) declaration slots are stable from [module](../../../../glossary.md#term-module) construction. A slot contains either one real
bound `Parameter` or one private reservation validator plus no public wrapper. Direct and recursive
parameter discovery [preflight](../../../../glossary.md#term-preflight) completion before producing their [immutable](../../../../glossary.md#term-immutable) snapshots. Recursive
failure reports the first qualified path in normal depth-first order. State export performs the
same preflight before reading any [binding](../../../../glossary.md#term-binding) into a returned dictionary.

Strict load is the one alternative initialization source:

1. Collect the complete [target](../../../../glossary.md#term-target) tree including reserved paths and their validators.
2. Validate missing and unexpected paths in the existing order.
3. Validate every bound target with the current rules and every reservation candidate with kind,
   exact configured type, Shape/policy constraints, and <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`requiresGrad == true`.
4. Construct every new `Parameter` wrapper as an unpublished local and prepare every existing
   replacement.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
5. Only after complete validation/preparation, install exact candidate references in target order
   and release each newly completed module declaration set.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [release glossary](../../../../glossary.md#term-release))</span>

No initializer, generator, draw, [Tensor](../../../../glossary.md#tensor) creation, [evaluation](../../../../glossary.md#term-evaluation), or copy occurs. After a successful
load, the first forward only validates the [descriptor](../../../../glossary.md#term-descriptor) against the loaded input width and builds the
expression. Ordinary load failure changes neither an existing binding nor a reservation. Load
remains externally coordinated and non-linearizable exactly as documented by current Module.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/package-info.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Linear.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/LinearWeightInitialization.java` (new)
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleDeferredParameterTest.java` (new)
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/StateDictionaryTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearInitializationTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/extensions/nn/master-plan.md`
- this task specification

Reviewed unchanged unless implementation proves a current statement inaccurate:

- [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `StateDictionary`, and `StateEntry` source — public value invariants and
  signatures remain accurate; only Module owns reservation-aware declaration/traversal/load.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Module glossary](../../../../glossary.md#term-module))</span>
- [`Model`](../../../../glossary.md#term-model), [`Topology`](../../../../glossary.md#term-topology), `Sequential`, recurrent modules, and their production contracts — ordinary
  functional traversal invokes Linear naturally; no container or public lifecycle changes.<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor API, Compile API, Model capabilities/master plan, architecture contract/explanations,
  ADR 0007, architecture tests, Checkpoint master plan, Training production/master plan,
  Data/Text/Vision plans, Gradle, conformance/integration, compiler/runtime/prepare/Engine/backends,
  CPU work, global roadmap, and legacy code — no owning rule, dependency, build, or execution
  behavior changes. Checkpoint terminology may continue to call the internal state deferred/lazy;
  it does not imply a public <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>`Lazy*` [API](../../../../glossary.md#term-application-programming-interface-api).

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the fifteen exact paths listed above: six production, five
test, and four documentation/[planning](../../../../glossary.md#term-planning) paths. If another production type, public/protected member,
test [owner](../../../../glossary.md#term-owner), document, [module](../../../../glossary.md#term-module), build file, architecture path, or sixteenth path is required, stop
and propose the smallest focused follow-up.

## Acceptance criteria

- [`Linear`](../../../../glossary.md#term-linear-meanings) remains one final public type. It declares exactly four public constructors (the three
  existing forms plus the specified automatic-input-width form) and the existing [`weight`](../../../../glossary.md#term-weight), [`bias`](../../../../glossary.md#term-bias),
  and `forward` methods. No `Lazy*`, factory, bind/build/initialize/status API, public field, or
  generic layer abstraction exists.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `LinearWeightInitialization` contains exactly the four selected constants, and each dispatches
  once to the corresponding current <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`ParameterInitializers` operation. There is no default,
  mutable state, alias, callback, registry, or serialization promise.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [mutable glossary](../../../../glossary.md#term-mutable); [alias glossary](../../../../glossary.md#aliasing-and-arena); [callback glossary](../../../../glossary.md#term-callback); [registry glossary](../../../../glossary.md#term-registry); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Existing supplied/eager Linear construction, promotion behavior, state order/identity,
  replacement snapshots, mode-insensitivity, and forward provenance remain compatible.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- The new constructor creates no generator, Tensor, Tensor ID, Parameter, expression, or hidden
  default. It reserves stable <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Parameter glossary](../../../../glossary.md#term-parameter))</span>`weight` then optional `bias` paths and validates configuration in
  the specified order.
- Before automatic initialization, `weight`, configured `bias`, direct/recursive parameter
  discovery, and state export fail clearly. No-bias `bias()` remains empty. Buffer/child/mode APIs
  remain usable. No incomplete or nullable Parameter wrapper can be observed.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer))</span>
- The first compatible forward performs complete layer initialization before expression
  construction and returns the usable exact Tensor.linear result from that same call. Tests prove
  parameter IDs precede PERMUTE/MATMUL/ADD IDs and exact provenance retains the published bindings.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [Binding](../../../../glossary.md#term-binding) infers only one positive static final `inFeatures`. `[7,32]`, `[3,5,32]`, and dynamic
  leading Dimensions ending in static 32 remain compatible; scalar input, dynamic/zero final
  extent, wrong exact type, or another final extent fails without changing published state.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent))</span>
- First-forward prevalidation failures occur before generator/Tensor/ID effects. Successful
  initialization uses the exact configured factory/seed and policy, weight-before-zero-bias order,
  exact deterministic values/draw semantics, and no retained created generator. Later compatible
  forwards create only their ordinary expression Tensors.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- A failure after local parameter creation but before complete publication leaves all reservations
  unbound and is retryable; consumed draws and Tensor IDs remain consumed. Expression-construction
  failure after publication leaves stable initialized state.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Concurrent compatible first forwards initialize one complete parameter set exactly once and
  each returns its own expression. An incompatible contender either establishes its schema first
  or fails against the winner. Same-layer access/discovery observes unbound-or-complete under the
  documented release/acquire gate; no broader thread-safety claim is made.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [release glossary](../../../../glossary.md#term-release))</span>
- Strict load includes reserved paths, initializes them from exact candidates without RNG, retains
  whole-tree ordinary validate-before-install behavior, and makes equivalent eager/automatic
  dictionaries compatible in both directions after Shapes are known.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [Shapes glossary](../../../../glossary.md#shape))</span>
- A two-Layer functional [Model](../../../../glossary.md#term-model) proves automatic hidden/output input-width [inference](../../../../glossary.md#term-inference) and paths
  `hidden.weight`, optional `hidden.bias`, `output.weight`, and optional `output.bias`. An unused
  registered automatic Linear leaves complete discovery/export unavailable. A later-body failure
  proves any earlier successfully initialized layer stays initialized without Model-wide rollback.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Public/protected/package Javadocs explain the single-type API, automatic first-forward phases,
  exact inferred axis/type, configuration ownership, validation/RNG/Tensor-ID effects, accessors,
  discovery/load behavior, concurrency/JMM boundary, retry, functional-Model partial effects, and
  expression-only/no-execution boundary with complete tags.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [validation glossary](../../../../glossary.md#term-validation); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Training API and glossary replace the old all-eager status with the exact unified Linear
  contract and explain that first forward creates parameter leaves before returning an ordinary
  expression. They must not claim a general Model initialization lifecycle, execution, durable
  checkpoint, lazy recurrent layer, or automatic vocabulary/schema inference.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>
- The implementation records reasoned no-change conclusions for Parameter/Buffer/state values,
  Model/Topology/Sequential/recurrent contracts, Tensor/Compile APIs, architecture/ADR/tests,
  Checkpoint/Training/Data/Text/Vision plans, Gradle, execution layers, backends, CPU, roadmap,
  conformance, and integration.<span data-glossary-navigation="true"> ([Topology glossary](../../../../glossary.md#term-topology); [Compile glossary](../../../../glossary.md#compile); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- A separate documentation-focused clean context independently finalizes all affected Javadocs,
  explanatory documentation, glossary impact, links, examples, and final task evidence before the
  task becomes Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass develops with focused tests and, after executable Java stabilizes, runs:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModuleDeferredParameterTest --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest --tests io.github.pho001.synaptik.nn.layers.LinearTest --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest --tests io.github.pho001.synaptik.nn.module.ModelTest
./gradlew :extensions:nn:test
git diff --check
```

The separate documentation-focused pass reuses the authoritative [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test evidence unless it
changes executable Java behavior. After final [Javadoc](../../../../glossary.md#term-javadoc) edits it runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also inspects generated [Module](../../../../glossary.md#term-module)/[Linear](../../../../glossary.md#term-linear-meanings)/enum/package pages, checks the exact public/protected
surface and absence of every forbidden [Lazy](../../../../glossary.md#term-lazy-evaluation)/[lifecycle](../../../../glossary.md#lifecycle) [API](../../../../glossary.md#term-application-programming-interface-api), compiles one external Java
`Model.define` example using the new constructor, scans production imports, validates local
Markdown links and anchors, balances fences, checks terminology, terminal newlines and trailing
whitespace, confirms exactly fifteen paths, confirms task/master Complete status and no [Ready](../../../../glossary.md#term-ready) NN
task, and confirms that NN 0020+ have no detailed task specifications.

Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), integration, numerical, [compiler](../../../../glossary.md#term-compiler), [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare),
[Engine](../../../../glossary.md#term-engine), [backend](../../../../glossary.md#backend), and [CPU](../../../../glossary.md#term-cpu) suites remain deferred to the selected NN integration [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).
This task changes one existing [model](../../../../glossary.md#term-model)-only module and no build edge, [dependency](../../../../glossary.md#term-dependency) rule, architecture
boundary, shared configuration, runtime input, or numerical execution path.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0018 are Complete.
- Current Parameter replacement, iterative Module traversal, strict state dictionary, eager
  initializers, eager/supplied Linear, unary composition, and typed functional Model contracts are
  stable.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Module glossary](../../../../glossary.md#term-module); [eager glossary](../../../../glossary.md#term-eager-evaluation); [Linear glossary](../../../../glossary.md#term-linear-meanings); [unary glossary](../../../../glossary.md#term-unary))</span>
- [Model](../../../../glossary.md#term-model) `TensorDescriptor`, Shape/Dimension/DataType, eager random Tensor creation, opaque Tensor
  IDs, and <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [DataType glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [opaque glossary](../../../../glossary.md#term-opaque))</span>`Tensor.linear` expression contracts are Complete.
- `RandomGeneratorFactory.create(seed)` supplies the explicit deterministic deferred source; this
  task rejects stochastic factories and retains no created generator.
- Checkpoint remains a Draft downstream consumer. Its future strict load may consume this in-memory
  reservation contract, but no persistent codec or Engine materialization is required here.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [codec glossary](../../../../glossary.md#term-codec); [Engine glossary](../../../../glossary.md#term-engine); [materialization glossary](../../../../glossary.md#materialization))</span>
- The user-authorized NN interleave remains isolated from the active CPU frontier; this task
  changes no CPU, roadmap, build, dependency, or architecture path.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Follow-up tasks

- NN 0020: reassess the proven internal lifecycle for recurrent input weights and inferred zero
  recurrent states; add automatic initialization only where a concrete layer policy is complete.
  Embedding still requires explicit vocabulary size and embedding width or supplied state.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle); [weights glossary](../../../../glossary.md#term-weight); [Embedding glossary](../../../../glossary.md#embedding))</span>
- NN 0021–0022: define genuine runtime recurrent scan/input binding before consuming Data-owned
  runtime valid lengths.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [binding glossary](../../../../glossary.md#term-binding))</span>
- Checkpoint 0003–0004: materialize and persist state downstream, then use strict reservation-aware
  load only after complete artifact validation.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [artifact glossary](../../../../glossary.md#term-artifact); [validation glossary](../../../../glossary.md#term-validation))</span>
- A future concrete consumer may justify model-wide descriptor tracing, seed derivation, schema
  inspection, or a different initialization policy. None is part of this first unified Linear.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [seed glossary](../../../../glossary.md#term-seed); [schema glossary](../../../../glossary.md#term-schema); [Linear glossary](../../../../glossary.md#term-linear-meanings))</span>

## Architecture impact

Expected impact: None.

The current architecture already assigns [module](../../../../glossary.md#term-module) state, neural-network layers, and forward
composition to `extensions/nn`. Private future declarations and automatic layer-local [publication](../../../../glossary.md#term-publication)
remain within that boundary and preserve the [model](../../../../glossary.md#term-model)-only [dependency](../../../../glossary.md#term-dependency). Stop if implementation needs
another dependency, Model-wide [lifecycle](../../../../glossary.md#lifecycle), [runtime](../../../../glossary.md#term-runtime) execution, or architecture rule.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not use GSD. Do not commit or push unless the user
explicitly requests it after the complete change.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/roadmap.md, docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0019-automatic-first-forward-linear-initialization.md in full.
Read every directly referenced completed NN task, final Module/Parameter/StateDictionary/Model/
Topology/Linear/initializer source and tests, the relevant Model Tensor/Shape/DataType/random/
linear contracts, Training API, glossary, Checkpoint plan, and documentation rules/profiles named
by the task.

Implement task 0019 exactly within its fifteen authorized paths. Preserve one public Linear type,
the always-bound Parameter invariant, eager/supplied compatibility, Model-only dependency, active
CPU work, roadmap, and every out-of-scope boundary. Stop and report architecture, API, lifecycle,
atomicity, JMM, or scope uncertainty instead of inventing another design. Run the focused
selection and one final NN suite after executable Java stabilizes.

Then hand the final diff and exact Java evidence to a separate documentation-focused clean
agent/thread. That pass follows docs/developer-guide/documentation-rules.md, independently
finalizes affected Javadocs, package docs, Training API, glossary and planning evidence, runs final
Javadoc/documentation validation, and updates this task completion record. Do not mark Complete
before that pass succeeds.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the documentation-focused agent this task, the final diff, exact focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evidence,
whether executable Java changed afterward, and the selected constructor/policy surface. Identify
the automatic first-forward phase order, reservation/access/discovery/load behavior, deterministic
factory/[seed](../../../../glossary.md#term-seed) effects, layer-local [synchronization](../../../../glossary.md#term-synchronization) and [release](../../../../glossary.md#term-release)/acquire [publication](../../../../glossary.md#term-publication), functional
[Model](../../../../glossary.md#term-model) partial effects, and the exact fifteen authorized paths.

The agent independently reads the repository/architecture/documentation contracts, General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), and Example profiles, final implementation and tests, relevant completed NN
tasks, [Tensor](../../../../glossary.md#tensor)/Training APIs, glossary, and [Checkpoint](../../../../glossary.md#term-checkpoint) plan. It finalizes the six affected production
and package Javadocs plus Training API, glossary, master/task evidence, and records exact no-change
conclusions for every reviewed unchanged boundary. It reuses successful Java evidence unless it
changes executable behavior or records a concrete reason to rerun it.

## Local decisions

- Use one ordinary constructor on final [`Linear`](../../../../glossary.md#term-linear-meanings). The user should describe architecture, not
  choose eager versus lazy public types or call a duplicate initialization pass.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>
- Keep [`Parameter`](../../../../glossary.md#term-parameter) always bound. A future declaration is private Module metadata until one complete
  layer-local forward initialization or strict whole-tree load publishes real wrappers.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Use `RandomGeneratorFactory` plus seed because an unknown first-forward time cannot safely retain
  a mutable caller-owned generator. A fresh non-stochastic seeded generator makes retries
  deterministic without a hidden global/default source.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Use one closed Linear-specific policy enum because the concrete consumer needs to select among
  four already implemented fan policies. Do not generalize it into an initializer object model.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Fail complete parameter discovery and state export before all owned reservations bind. Empty or
  partial results would make optimizer/checkpoint schema silently depend on forward history.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema))</span>
- Let strict state load bind reservations because its complete candidate supplies every missing
  Shape and can be validated before installation; it consumes no initializer or RNG.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Make initialization layer-local and traversal-driven. Arbitrary Java in `Model.define` prevents
  whole-body preflight or rollback without a much broader tracing lifecycle.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [rollback glossary](../../../../glossary.md#term-rollback); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Preserve opaque Tensor-ID non-rollback and distinguish parameter-leaf creation from actual
  Tensor.linear expression construction. Neither phase numerically executes a model.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [Tensor glossary](../../../../glossary.md#tensor); [leaf glossary](../../../../glossary.md#term-leaf))</span>

## Known limitations

- Only the existing [`Linear`](../../../../glossary.md#term-linear-meanings) gains automatic input-width initialization. Every other module keeps
  its current construction contract until a concrete follow-up chooses its own policy.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- The final input feature Dimension must be positive and static on first forward or strict-load
  candidate Shape. Leading Dimensions may be dynamic and may change later.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [Dimension glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape))</span>
- Parameter/state discovery of a Model containing an unvisited automatic layer fails. There is no
  public initialization status, Model-wide branch coverage, or schema export before binding.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding))</span>
- A functional Model forward can leave an earlier layer initialized when later Java/module work
  fails. The task provides layer-local publication, not a global transaction.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Only concurrent first-forward initialization on the same Linear has an internal synchronization
  guarantee. Replacement, strict load, tree traversal, mode changes, and arbitrary forward bodies
  retain their caller-coordinated contracts.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Initialization is limited to the four current fan policies and zero bias. There is no custom
  initializer, automatic seed splitting, default source, or mutable source retention.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [seed glossary](../../../../glossary.md#term-seed); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- This capability creates eager parameter leaves and declarative Model expressions. It proves no
  numeric result, compilation, gradient, backend route, physical execution, or durable checkpoint.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [gradient glossary](../../../../glossary.md#term-gradient); [backend route glossary](../../../../glossary.md#backend-route); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only replacement completed in clean task context `/root/nn_0019_replanning`.

- A targeted Ruby Markdown check passed for the NN master plan and this task: every local file
  target and heading fragment resolves, headings are unique, fences are balanced, terminal
  newlines are present, and no trailing whitespace was found.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Exact frontier checks passed: this is the only `0019-*.md` task, the rejected
  `0019-deferred-parameter-binding-and-lazy-linear.md` path is absent, no NN 0020–0024 detailed
  specification exists, and exactly one NN task/master row is [`Ready`](../../../../glossary.md#term-ready).
- Exact planning [scope](../../../../glossary.md#term-scope) passed: `git status --short` contains only the modified NN master plan and
  this new untracked task file. `git diff --name-only` plus `git ls-files --others
  --exclude-standard` reports those same two paths.
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0019-automatic-first-forward-linear-initialization.md`
  returned the expected difference status `1` with no whitespace diagnostic for the new file.
- `git diff --check` passed with no output for the tracked master-plan change.
- No Java, Javadoc, Gradle, architecture, conformance, integration, backend, or repository-wide
  test was run because this planning-only change modifies no executable or architecture behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [backend glossary](../../../../glossary.md#backend); [run glossary](../../../../glossary.md#term-run))</span>

Implementation context `/root/nn_0019_auto_forward_impl` completed the executable capability.

- The focused command selecting `ModuleDeferredParameterTest`, `StateDictionaryTest`,
  `LinearTest`, `LinearInitializationTest`, and `ModelTest` passed 5 suites and 55 tests with zero
  failures, errors, or skips.
- The one authoritative final `./gradlew :extensions:nn:test` passed 31 suites and 226 tests with
  zero failures, errors, or skips. No executable Java or test changed afterward.
- Preliminary implementation Javadoc, public-surface, external-use, forbidden-mechanism/import,
  exact thirteen-path implementation scope, newline, whitespace, and `git diff --check` checks
  passed before the documentation handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>

Independent clean documentation context `/root/nn_0019_docs` reviewed the final implementation
against the architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, planning/documentation profiles, completed [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [ownership](../../../../glossary.md#term-ownership-meanings),
initializer, [Linear](../../../../glossary.md#term-linear-meanings), state-dictionary, [unary](../../../../glossary.md#term-unary)-composition and typed-[Model](../../../../glossary.md#term-model) tasks, final source/tests,
generated Javadocs, Training [API](../../../../glossary.md#term-application-programming-interface-api), glossary, and [Checkpoint](../../../../glossary.md#term-checkpoint) plan. It found no executable,
public-API, architecture, [dependency](../../../../glossary.md#term-dependency), or task-scope defect and changed no executable Java or test.

- The pass finalized the six affected production/package Javadocs, `docs/api/training-api.md`,
  `docs/glossary.md`, the NN master plan, and this task. It explains that parameter leaves are
  validated and published before the same first call constructs its ordinary <span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span>`Tensor.linear`
  expression; neither phase numerically executes the model or introduces a public lazy/build/
  initialize lifecycle.<span data-glossary-navigation="true"> ([lazy glossary](../../../../glossary.md#term-lazy-evaluation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The final documentation covers exact inferred versus architectural Dimensions, constructor and
  forward validation, generator/factory/seed ownership, random-draw/allocation/Tensor-ID ordering
  and non-rollback, retry, layer-local synchronization and release/acquire publication, access and
  complete discovery/export gating, strict-load initialization without RNG, eager/supplied
  compatibility, variable leading Dimensions, and functional-Model partial effects.<span data-glossary-navigation="true"> ([Dimensions glossary](../../../../glossary.md#dimension); [seed glossary](../../../../glossary.md#term-seed); [Tensor glossary](../../../../glossary.md#tensor); [rollback glossary](../../../../glossary.md#term-rollback); [synchronization glossary](../../../../glossary.md#term-synchronization); [release glossary](../../../../glossary.md#term-release); [publication glossary](../../../../glossary.md#term-publication); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- The pass reused the final 5-suite/55-test and 31-suite/226-test Java evidence because no
  executable Java or test changed after it; it did not repeat either suite.
- `./gradlew :extensions:nn:javadoc` passed after final [Javadoc](../../../../glossary.md#term-javadoc) edits. Rendered `Linear`, [`Module`](../../../../glossary.md#term-module),
  `LinearWeightInitialization`, and all three affected package pages were inspected for the final
  contracts.
- Final `javap` inspection showed exactly four public `Linear` constructors and the existing
  [`weight`](../../../../glossary.md#term-weight), [`bias`](../../../../glossary.md#term-bias), and `forward` methods; the enum has exactly four constants; Module's public
  surface is unchanged; and its exact four new reservation primitives are protected final. A
  standalone external-package <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`Model.define` example using two automatic Linear layers compiled.
- Production import/dependency and forbidden-public-Lazy/lifecycle scans passed. Markdown local
  links and anchors, balanced fences, exact fifteen-path scope, task/master Complete status,
  NN 0020–0024 Draft rows without specifications or Ready status, terminal newlines, trailing
  whitespace, and final `git diff --check` passed.
- [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `StateDictionary`, and `StateEntry` remain unchanged because their public
  bound-value invariants and signatures are still exact; private reservation/schema integration is
  Module-owned. <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [schema glossary](../../../../glossary.md#term-schema))</span>`Model`, [`Topology`](../../../../glossary.md#term-topology), `Sequential`, and recurrent production contracts remain
  unchanged because automatic initialization follows ordinary Java traversal without adding a
  container lifecycle. Tensor/Compile APIs, architecture/ADR/tests, Checkpoint/Training/Data/Text/
  Vision plans, Gradle, compiler/runtime/prepare/Engine/backends, CPU, global roadmap,
  conformance, and integration remain unchanged because no semantic, dependency, build,
  persistence, or execution boundary changed.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## Implementation notes

- Isolated implementation context `/root/nn_0019_auto_forward_impl` replaced the direct Parameter
  map with private declaration slots while preserving encounter order and the shared namespace.
  A volatile completion gate publishes all wrappers in one direct reservation group and makes
  access/discovery observe unbound-or-complete state.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter))</span>
- [`Linear`](../../../../glossary.md#term-linear-meanings) remains one final type. Its new constructor retains the exact deterministic factory,
  seed, type, output width, bias choice, and closed policy while reserving state without creating
  a generator, Tensor, identifier, or Parameter. First forward validates and initializes inside
  one layer-local critical section, then constructs the ordinary expression outside it.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [bias glossary](../../../../glossary.md#term-bias); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Strict load collects reserved paths as target schema, validates and constructs every new wrapper
  before installing any target, then publishes completed reservation groups. Loading uses exact
  candidate Tensor references and no initializer or random generator.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [schema glossary](../../../../glossary.md#term-schema))</span>
- Existing supplied-Tensor and explicit-dimension Linear construction, state names, wrapper
  replacement, promotion, and expression provenance remain covered by the complete NN suite.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [provenance glossary](../../../../glossary.md#provenance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Completion summary

- Completed changes: Added the private Module reservation/publication lifecycle, one unified
  automatic-first-forward <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>[`Linear`](../../../../glossary.md#term-linear-meanings) constructor, the four-value Linear policy enum, strict-load
  initialization, eager compatibility, and focused regression coverage.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [regression glossary](../../../../glossary.md#term-regression))</span>
- Files changed or created: Exactly the planned six production/package paths, five test paths,
  Training API, glossary, NN master plan, and this task: fifteen paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: Reused the implementation context's successful focused 5-suite/55-test and
  authoritative NN 31-suite/226-test results; final NN Javadoc/rendered pages, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`javap`, external
  compilation, forbidden API/imports, Markdown, exact scope/status/frontier, newline, whitespace,
  and diff validation passed.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Independent clean context `/root/nn_0019_docs` found no executable,
  API, architecture, dependency, or scope defect and finalized all authorized documentation.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation impact: Training API and package/type Javadocs now describe automatic layer-local
  parameter publication before expression construction, failures, concurrency, strict load, and
  Model partial effects without claiming numerical execution or a general public lifecycle.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [concurrency glossary](../../../../glossary.md#term-concurrency); [Model glossary](../../../../glossary.md#term-model))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Final generation and rendered inspection passed for `Linear`, [`Module`](../../../../glossary.md#term-module), the
  policy enum, and their three packages.
- Glossary impact: Updated the module/parameter/state-dictionary, Linear, and functional-Model
  entries to define reservations, automatic first-forward initialization, and its boundaries.
- Unresolved issues: None.
- Follow-up required: None. [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0020 remains the next Draft row and has no detailed specification.

Status: Complete

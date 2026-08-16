# Task 0019: Automatic First-Forward Linear Initialization

## Status

Complete

## Goal

Let the existing final public `Linear` infer its input-feature width and initialize its complete
parameter set automatically inside the first successful `forward(input)` call. The user declares
only the architectural output width and initialization configuration; there is no separate lazy
type, factory, bind/build/initialize call, or public initialization-state API.

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

For input Shape `[batch, 32]`, the first call initializes `hidden.weight [64, 32]`, constructs the
hidden expression, initializes `output.weight [10, 64]` from that expression descriptor, and
returns the usable output expression from the same call. Later calls may change leading batch or
time Dimensions, but their exact data type and final feature extent must remain compatible.

Mental model for one automatically initialized `Linear`:

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

Every public `Parameter` remains a real, non-null, fully bound wrapper with one permanent schema.
Before the first successful forward or strict state load, future parameter paths exist only as
private Module-owned declaration metadata. Complete parameter discovery and state export fail
clearly instead of returning a partial tree.

## Scope

- Keep exactly one final public `Linear`. Preserve the existing caller-supplied Tensor constructors,
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

  It accepts no `inFeatures`. Construction validates and retains immutable configuration, reserves
  `weight` then optional `bias`, and creates no random generator, Tensor, Tensor ID, or Parameter.
- Add public enum `LinearWeightInitialization` with exactly the four existing positive rank-two
  Linear policies: `GLOROT_NORMAL`, `GLOROT_UNIFORM`, `KAIMING_RELU_NORMAL`, and
  `KAIMING_RELU_UNIFORM`. `Linear` dispatches each value directly to the matching existing
  `ParameterInitializers` method. Do not add a generic initializer abstraction, custom callback,
  registry, alias, gain/fan configuration, or default policy.
- Do not add `LazyLinear`, `Linear.lazy`, `LazyModule`, a public `bind`, `build`, `initialize`,
  `isBound`, descriptor-tracing API, or Model-wide initialization lifecycle.
- Add private reservation records/slots inside `Module`. A reserved parameter name immediately
  occupies the existing shared direct namespace and its declaration position, but exposes no
  `Parameter` wrapper until complete publication.
- Add exactly these protected final subclass primitives to `Module`:

  ```java
  protected final void reserveParameter(String name, Consumer<Tensor> validator)
  protected final void bindReservedParameters(List<Tensor> values)
  protected final boolean parameterReservationsBound()
  protected final Parameter boundParameter(String name)
  ```

  They are lifecycle implementation support for concrete NN modules, not a public user-facing
  binding API. Do not add another production abstraction.
- `reserveParameter` applies the existing local-name validation first, rejects a null validator,
  records one future parameter in encounter order, and retains the exact deterministic validator.
  A validator returns normally or throws and must have no externally visible side effect; this
  task supplies only private Linear validators.
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
  reservations, buffers, and children keep the existing collision rules.
- Make `parameters()` fail with `IllegalStateException` naming the first unbound direct reservation.
  Make `parametersRecursively()` preflight the whole identity-defended tree and fail with the first
  qualified unbound path rather than return a history-dependent partial snapshot.
  `stateDictionary()` has the same complete-tree failure rule. Buffer-only discovery, children,
  mode propagation, and topology ownership remain available.
- Validate the new Linear constructor in this order: positive `outFeatures`; non-null `dataType`;
  non-null `weightInitialization`; non-null `randomGeneratorFactory`; floating data type; then
  `randomGeneratorFactory.isStochastic() == false`. Retain the exact factory and seed, but never a
  created generator. Select no default algorithm, seed, data type, bias, or policy.
- On every automatic-path `forward(input)`, reject null first, require the configured exact data
  type, rank at least one, a static final Dimension, and a positive final extent. Leading rank and
  Dimensions do not become parameter schema. Before the first factory creation, preflight exact
  weight Shape `[outFeatures, inFeatures]`, optional bias Shape `[outFeatures]`, checked counts,
  Model Java-array limits, and every other configuration/input failure knowable without sampling.
- During the first compatible forward, create exactly one fresh generator with
  `randomGeneratorFactory.create(seed)`. Initialize weight first in row-major order through the
  selected existing initializer; initialize optional bias second through exact typed zeros, which
  consumes no random draw. Retain neither generator nor caller input.
- Publish neither wrapper until both local parameter Tensors and their validators succeed. A
  failed attempt leaves all reservations unbound and is retryable. A new attempt creates a fresh
  generator from the same factory and seed. Completed draws, local allocations, and consumed
  opaque Tensor IDs are not rolled back or reused.
- The first forward has two ordered internal phases. Complete and verify layer-local initialization
  before calling either `Tensor.linear` overload. The same call then reads each binding once and
  returns the actual ordinary Model expression. If expression construction fails after state
  publication, the parameters stay initialized and a later compatible call does not initialize
  them again.
- Later automatic-path forwards create no generator and no parameter Tensor. They require the
  configured exact data type and the already bound final feature extent. Compatible leading
  Dimensions may vary, including a different rank. Existing eager/supplied Linear forward keeps
  its current Model promotion/rank/contraction behavior unchanged.
- `weight()` fails clearly before automatic initialization. For a configured bias, `bias()` also
  fails rather than return an empty Optional; for a no-bias automatic Linear it always returns
  `Optional.empty()`. After initialization both accessors return the exact stable wrappers used by
  discovery, replacement, state export, and strict load.
- Serialize only the one-time initialization critical section on the exact Linear instance. Use
  a release/acquire publication gate owned by Module so parameter access and discovery after
  successful initialization observe all wrappers. Compatible concurrent first forwards initialize
  once; one incompatible contender may win and fix the schema while the other then fails against
  it. Do not synchronize later expression construction or claim general Module thread safety.
- Extend strict `Module.loadStateDictionary(...)` so reserved parameter paths participate in the
  complete target schema. For an uninitialized automatic Linear, candidate `weight` must be
  gradient-eligible, have the configured exact floating type, and have positive fully static
  Shape `[outFeatures, inFeatures]`; optional bias must match exact type and `[outFeatures]`.
  The weight's final extent establishes `inFeatures`. No initializer or RNG is used.
- Preserve strict whole-tree validate-before-install semantics. Missing/unexpected paths, kind,
  type, Shape, gradient eligibility, and every reservation validator are checked for the complete
  tree; all new wrappers are constructed as unpublished locals before one existing binding is
  replaced or one reservation is published. Ordinary failure changes no binding or reservation.
  JVM-fatal resource failure keeps the existing documented limitation.
- An equivalent eager or initialized automatic Linear uses the same `weight`/optional `bias` paths,
  kinds, data types, and Shapes. State dictionaries load in either direction when schemas match,
  and loading eager state into an uninitialized automatic model retains the exact candidate Tensor
  references without a forward initialization pass.
- In a functional Model, automatic initialization follows the Java forward traversal actually
  executed. If an earlier Linear initializes and a later layer or caller body fails, the earlier
  state remains initialized. There is no Model-wide trace, validation pass, transaction, rollback,
  or public `Model.initialize` operation. A registered module not traversed by a successful forward
  may remain uninitialized; complete recursive discovery/export then fails until every reservation
  reachable in the owned tree is bound by forward traversal or strict state load.
- Add focused tests for exact API surface, reservation ordering/collisions/discovery, constructor
  and first-forward validation order, deterministic initialization, Tensor-ID ordering and failure
  effects, accessors, variable leading Dimensions, expression provenance, retry, concurrency,
  replacement, strict load, functional Model composition, unvisited modules, and partial Model
  failure effects.
- Finalize every affected public/protected/package Javadoc, the Training API, glossary, and planning
  evidence through the mandatory separate clean documentation context.

## Out of scope

- Any separate `Lazy*` public type, factory, interface, mixin, marker, wrapper, or naming family.
- A public/protected initialization-state query or per-layer/model `bind`, `build`, `initialize`,
  dry-run, example-input, input-specification, or descriptor-tracing method.
- Automatic initialization for Embedding, LayerNorm, BatchNorm, RNN, GRU, LSTM, sequence
  containers, or another existing module; those require their own concrete policy and consumer.
- Inferring `outFeatures`, hidden width, class count, vocabulary size, embedding width, recurrent
  hidden size, bias presence, data type, initializer policy, random algorithm, or seed.
- Retaining a mutable `RandomGenerator`, choosing a default generator, deriving per-layer seeds,
  serializing generator objects, custom initialization callbacks, or graph RNG state.
- Model-wide first-forward preflight, rollback, initialization status, branch coverage, tracing,
  parameter-schema export before initialization, or optimizer construction convenience.
- Returning an empty or partial parameter snapshot/state dictionary for an uninitialized owned
  tree, placeholder/uninitialized Tensor values, nullable Parameter state, or incomplete wrappers.
- Relaxed, partial, remapped, converting, or best-effort state load; checkpoint bytes, codecs,
  files, versions, materialization, checksums, manifests, optimizer state, or Training resume.
- New Tensor/operation semantics, numerical evaluation, compiler capture, autograd, backend
  support, lowering, execution, device storage, or a claim that `forward` calculates values.
- Data/Text/Vision batching, tokenizer, valid-length, padding/mask, recurrent scan, or image work.
- Gradle, dependency, architecture-contract, ADR, architecture-test, global-roadmap, CPU, other
  module, conformance, integration, or unrelated refactoring changes.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Training API](../../../../api/training-api.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [NN master plan](../master-plan.md)
- [Task 0001: Module, Parameter, Buffer, and forward context](0001-module-parameter-buffer-and-forward-context-foundation.md)
- [Task 0002: Module-tree ownership](0002-module-tree-ownership-and-recursive-mode-propagation.md)
- [Task 0003: Binding replacement](0003-validated-parameter-and-buffer-binding-replacement.md)
- [Task 0004: Eager parameter initializers](0004-explicit-eager-parameter-initializers.md)
- [Task 0004A: Parameter/traversal hardening](0004a-parameter-update-and-traversal-hardening.md)
- [Task 0005: Linear layer](0005-linear-layer.md)
- [Task 0010: State dictionary](0010-state-dictionary-and-checkpoint-contract.md)
- [Task 0011: Unary composition](0011-unary-tensor-module-composition-and-sequential.md)
- [Task 0018: Typed functional Model topology](0018-typed-functional-model-topology.md)
- [Checkpoint master plan](../../checkpoint/master-plan.md)

## Architecture constraints

- `extensions/nn` owns module state, layer composition, parameters, buffers, and forward behavior
  and may depend only on `modules/model`. This task adds no downstream or execution dependency.
- `Parameter` remains a final wrapper around one non-null real Tensor with declaration-time exact
  data type and structural Shape. Automatic initialization must not weaken that invariant.
- Tensor identity, descriptor, and expression provenance remain immutable. Initializer calls create
  eager host-backed parameter leaves; `forward` then constructs ordinary Tensor expressions and
  neither evaluates nor executes them.
- The Model/topology ownership tree stays immutable after definition. Automatic initialization
  changes only layer-owned parameter declarations along the actually invoked forward path.
- Compiler owns autograd and graph compilation; Training owns optimizer algorithms/orchestration;
  Engine/Runtime/backends own execution/materialization. NN must not import or reproduce them.
- The current strict in-memory state dictionary remains the only task-owned load boundary. Durable
  checkpoint transport remains downstream and unimplemented.
- Planning is non-authoritative. If preserving the Parameter/state-dictionary invariants requires
  a dependency, architecture rule, model-wide transaction, or another public lifecycle surface,
  stop and report the conflict instead of forcing this task to Complete.

The design fits the current architecture: private declaration metadata and layer-local publication
remain NN-owned state composition, and no dependency or module-boundary change is required.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns declaration slots, discovery, state load, and
  parameter publication.
- `io.github.pho001.synaptik.nn.layers` — owns the single public Linear layer and its automatic
  first-forward behavior.
- `io.github.pho001.synaptik.nn.initialization` — owns the closed Linear initialization policy.
- `io.github.pho001.synaptik.model.tensor`, `.shape`, and `.datatype` — supply unchanged Tensor
  descriptors, Shape/Dimension values, data types, eager parameter leaves, and linear expressions.

No package is added or renamed.

Type placement:

- `io.github.pho001.synaptik.nn.module.Module` — sole owner of private reserved-parameter metadata,
  namespace/order integration, complete discovery gating, group publication, and strict-load
  integration.
- `io.github.pho001.synaptik.nn.layers.Linear` — the one existing final layer type; owns automatic
  validation, one-time initialization synchronization, retained factory/seed configuration, and
  ordinary forward composition.
- `io.github.pho001.synaptik.nn.initialization.LinearWeightInitialization` — closed selection among
  the four existing fan policies justified only by this Linear consumer.
- `io.github.pho001.synaptik.nn.module.ModuleDeferredParameterTest` — same-package reservation,
  namespace, discovery, publication, retry, and visibility tests.
- `io.github.pho001.synaptik.nn.module.StateDictionaryTest` — extends the owning strict-load suite
  with reserved targets and whole-tree validate-before-install behavior.
- `io.github.pho001.synaptik.nn.layers.LinearTest` — owns the exact unified public surface,
  accessors, compatibility validation, provenance, and replacement behavior.
- `io.github.pho001.synaptik.nn.layers.LinearInitializationTest` — owns deterministic first-forward
  initialization, failure/Tensor-ID ordering, retry, and concurrent first-call evidence.
- `io.github.pho001.synaptik.nn.module.ModelTest` — owns functional two-layer inference, unvisited
  reservations, and partial-forward state effects.

## Public and protected API

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

The enum exposes only ordinary enum-generated API and those four constants. Dispatch remains
inside `Linear`; it adds no public initialization method. There is no new public method on
`Linear`, `Module`, `Model`, or another type.

`Module` adds exactly these protected final methods:

```java
protected final void reserveParameter(String name, Consumer<Tensor> validator)
protected final void bindReservedParameters(List<Tensor> values)
protected final boolean parameterReservationsBound()
protected final Parameter boundParameter(String name)
```

Their implementation uses only JDK collection/function types and private Module-owned declaration
slots. `bindReservedParameters` binds all currently outstanding direct reservations or none; it is
not recursive. Linear owns synchronization around the check/create/bind sequence. Module strict
load separately prepares the complete tree and installs it under its existing caller-coordinated
contract.

## Initialization, failure, and concurrency contract

Construction has no parameter side effect. The automatic-path first forward performs these steps
in exact order:

1. Reject null input.
2. Read the immutable input descriptor and validate exact configured type, positive rank, static
   positive final extent, weight/bias Shapes, checked element counts, and Java-array limits.
3. Enter the Linear's one-time initialization critical section only if reservations appear
   unbound, repeat the compatibility check after acquiring it, and let an already successful
   contender win.
4. Create a fresh generator from the retained exact factory and seed.
5. Call the selected existing weight initializer once. It consumes one matching row-major random
   call per weight element and, on success, creates the weight Tensor and its ID.
6. If configured, create the exact-zero bias Tensor and its next ID without consuming RNG.
7. Run every retained reservation validator and construct all `Parameter` wrappers as locals.
8. Publish weight and optional bias together, then publish one release completion gate. Verify the
   reservations report bound before leaving the initialization phase.
9. Read the exact stable bindings once and call the matching existing `Tensor.linear` overload.
   Its PERMUTE/MATMUL/optional ADD Tensor IDs follow parameter publication.

Steps 1–2 fail before generator creation, random draws, Tensor allocation, or Tensor-ID allocation.
A generator/fan-initializer failure publishes no state. A failure after a local Tensor exists can
consume draws, memory, and opaque IDs, but the local values remain undiscoverable and a retry uses
a fresh generator with the same seed. A failure in step 9 keeps successfully published state.

Each Linear serializes only steps 3–8. The Module declaration completion gate is volatile (or an
equivalent explicit Java Memory Model release/acquire mechanism): publication writes occur before
the release, and accessors/discovery read the acquire before reading bound slots. A discovery race
with layer-local initialization must either reject the unbound reservation or observe the complete
published set, never a partial direct set. This narrow rule does not make Module tree traversal,
replacement, mode changes, state load, or arbitrary forward bodies generally thread-safe or
linearizable; callers still coordinate those operations.

## State discovery and strict-load contract

Parameter declaration slots are stable from module construction. A slot contains either one real
bound `Parameter` or one private reservation validator plus no public wrapper. Direct and recursive
parameter discovery preflight completion before producing their immutable snapshots. Recursive
failure reports the first qualified path in normal depth-first order. State export performs the
same preflight before reading any binding into a returned dictionary.

Strict load is the one alternative initialization source:

1. Collect the complete target tree including reserved paths and their validators.
2. Validate missing and unexpected paths in the existing order.
3. Validate every bound target with the current rules and every reservation candidate with kind,
   exact configured type, Shape/policy constraints, and `requiresGrad == true`.
4. Construct every new `Parameter` wrapper as an unpublished local and prepare every existing
   replacement.
5. Only after complete validation/preparation, install exact candidate references in target order
   and release each newly completed module declaration set.

No initializer, generator, draw, Tensor creation, evaluation, or copy occurs. After a successful
load, the first forward only validates the descriptor against the loaded input width and builds the
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

Expected documentation and planning files:

- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/extensions/nn/master-plan.md`
- this task specification

Reviewed unchanged unless implementation proves a current statement inaccurate:

- `Parameter`, `Buffer`, `StateDictionary`, and `StateEntry` source — public value invariants and
  signatures remain accurate; only Module owns reservation-aware declaration/traversal/load.
- `Model`, `Topology`, `Sequential`, recurrent modules, and their production contracts — ordinary
  functional traversal invokes Linear naturally; no container or public lifecycle changes.
- Tensor API, Compile API, Model capabilities/master plan, architecture contract/explanations,
  ADR 0007, architecture tests, Checkpoint master plan, Training production/master plan,
  Data/Text/Vision plans, Gradle, conformance/integration, compiler/runtime/prepare/Engine/backends,
  CPU work, global roadmap, and legacy code — no owning rule, dependency, build, or execution
  behavior changes. Checkpoint terminology may continue to call the internal state deferred/lazy;
  it does not imply a public `Lazy*` API.

## Maximum scope

This task may create or modify at most the fifteen exact paths listed above: six production, five
test, and four documentation/planning paths. If another production type, public/protected member,
test owner, document, module, build file, architecture path, or sixteenth path is required, stop
and propose the smallest focused follow-up.

## Acceptance criteria

- `Linear` remains one final public type. It declares exactly four public constructors (the three
  existing forms plus the specified automatic-input-width form) and the existing `weight`, `bias`,
  and `forward` methods. No `Lazy*`, factory, bind/build/initialize/status API, public field, or
  generic layer abstraction exists.
- `LinearWeightInitialization` contains exactly the four selected constants, and each dispatches
  once to the corresponding current `ParameterInitializers` operation. There is no default,
  mutable state, alias, callback, registry, or serialization promise.
- Existing supplied/eager Linear construction, promotion behavior, state order/identity,
  replacement snapshots, mode-insensitivity, and forward provenance remain compatible.
- The new constructor creates no generator, Tensor, Tensor ID, Parameter, expression, or hidden
  default. It reserves stable `weight` then optional `bias` paths and validates configuration in
  the specified order.
- Before automatic initialization, `weight`, configured `bias`, direct/recursive parameter
  discovery, and state export fail clearly. No-bias `bias()` remains empty. Buffer/child/mode APIs
  remain usable. No incomplete or nullable Parameter wrapper can be observed.
- The first compatible forward performs complete layer initialization before expression
  construction and returns the usable exact Tensor.linear result from that same call. Tests prove
  parameter IDs precede PERMUTE/MATMUL/ADD IDs and exact provenance retains the published bindings.
- Binding infers only one positive static final `inFeatures`. `[7,32]`, `[3,5,32]`, and dynamic
  leading Dimensions ending in static 32 remain compatible; scalar input, dynamic/zero final
  extent, wrong exact type, or another final extent fails without changing published state.
- First-forward prevalidation failures occur before generator/Tensor/ID effects. Successful
  initialization uses the exact configured factory/seed and policy, weight-before-zero-bias order,
  exact deterministic values/draw semantics, and no retained created generator. Later compatible
  forwards create only their ordinary expression Tensors.
- A failure after local parameter creation but before complete publication leaves all reservations
  unbound and is retryable; consumed draws and Tensor IDs remain consumed. Expression-construction
  failure after publication leaves stable initialized state.
- Concurrent compatible first forwards initialize one complete parameter set exactly once and
  each returns its own expression. An incompatible contender either establishes its schema first
  or fails against the winner. Same-layer access/discovery observes unbound-or-complete under the
  documented release/acquire gate; no broader thread-safety claim is made.
- Strict load includes reserved paths, initializes them from exact candidates without RNG, retains
  whole-tree ordinary validate-before-install behavior, and makes equivalent eager/automatic
  dictionaries compatible in both directions after Shapes are known.
- A two-Layer functional Model proves automatic hidden/output input-width inference and paths
  `hidden.weight`, optional `hidden.bias`, `output.weight`, and optional `output.bias`. An unused
  registered automatic Linear leaves complete discovery/export unavailable. A later-body failure
  proves any earlier successfully initialized layer stays initialized without Model-wide rollback.
- Public/protected/package Javadocs explain the single-type API, automatic first-forward phases,
  exact inferred axis/type, configuration ownership, validation/RNG/Tensor-ID effects, accessors,
  discovery/load behavior, concurrency/JMM boundary, retry, functional-Model partial effects, and
  expression-only/no-execution boundary with complete tags.
- Training API and glossary replace the old all-eager status with the exact unified Linear
  contract and explain that first forward creates parameter leaves before returning an ordinary
  expression. They must not claim a general Model initialization lifecycle, execution, durable
  checkpoint, lazy recurrent layer, or automatic vocabulary/schema inference.
- The implementation records reasoned no-change conclusions for Parameter/Buffer/state values,
  Model/Topology/Sequential/recurrent contracts, Tensor/Compile APIs, architecture/ADR/tests,
  Checkpoint/Training/Data/Text/Vision plans, Gradle, execution layers, backends, CPU, roadmap,
  conformance, and integration.
- A separate documentation-focused clean context independently finalizes all affected Javadocs,
  explanatory documentation, glossary impact, links, examples, and final task evidence before the
  task becomes Complete.

## Tests / validation

Implementation pass develops with focused tests and, after executable Java stabilizes, runs:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModuleDeferredParameterTest --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest --tests io.github.pho001.synaptik.nn.layers.LinearTest --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest --tests io.github.pho001.synaptik.nn.module.ModelTest
./gradlew :extensions:nn:test
git diff --check
```

The separate documentation-focused pass reuses the authoritative NN test evidence unless it
changes executable Java behavior. After final Javadoc edits it runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also inspects generated Module/Linear/enum/package pages, checks the exact public/protected
surface and absence of every forbidden Lazy/lifecycle API, compiles one external Java
`Model.define` example using the new constructor, scans production imports, validates local
Markdown links and anchors, balances fences, checks terminology, terminal newlines and trailing
whitespace, confirms exactly fifteen paths, confirms task/master Complete status and no Ready NN
task, and confirms that NN 0020+ have no detailed task specifications.

Repository-wide, architecture, conformance, integration, numerical, compiler, runtime, prepare,
Engine, backend, and CPU suites remain deferred to the selected NN integration checkpoint or CI.
This task changes one existing model-only module and no build edge, dependency rule, architecture
boundary, shared configuration, runtime input, or numerical execution path.

## Dependencies

- NN 0001–0018 are Complete.
- Current Parameter replacement, iterative Module traversal, strict state dictionary, eager
  initializers, eager/supplied Linear, unary composition, and typed functional Model contracts are
  stable.
- Model `TensorDescriptor`, Shape/Dimension/DataType, eager random Tensor creation, opaque Tensor
  IDs, and `Tensor.linear` expression contracts are Complete.
- `RandomGeneratorFactory.create(seed)` supplies the explicit deterministic deferred source; this
  task rejects stochastic factories and retains no created generator.
- Checkpoint remains a Draft downstream consumer. Its future strict load may consume this in-memory
  reservation contract, but no persistent codec or Engine materialization is required here.
- The user-authorized NN interleave remains isolated from the active CPU frontier; this task
  changes no CPU, roadmap, build, dependency, or architecture path.

## Follow-up tasks

- NN 0020: reassess the proven internal lifecycle for recurrent input weights and inferred zero
  recurrent states; add automatic initialization only where a concrete layer policy is complete.
  Embedding still requires explicit vocabulary size and embedding width or supplied state.
- NN 0021–0022: define genuine runtime recurrent scan/input binding before consuming Data-owned
  runtime valid lengths.
- Checkpoint 0003–0004: materialize and persist state downstream, then use strict reservation-aware
  load only after complete artifact validation.
- A future concrete consumer may justify model-wide descriptor tracing, seed derivation, schema
  inspection, or a different initialization policy. None is part of this first unified Linear.

## Architecture impact

Expected impact: None.

The current architecture already assigns module state, neural-network layers, and forward
composition to `extensions/nn`. Private future declarations and automatic layer-local publication
remain within that boundary and preserve the model-only dependency. Stop if implementation needs
another dependency, Model-wide lifecycle, runtime execution, or architecture rule.

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

## Documentation-agent handoff

Give the documentation-focused agent this task, the final diff, exact focused/final NN evidence,
whether executable Java changed afterward, and the selected constructor/policy surface. Identify
the automatic first-forward phase order, reservation/access/discovery/load behavior, deterministic
factory/seed effects, layer-local synchronization and release/acquire publication, functional
Model partial effects, and the exact fifteen authorized paths.

The agent independently reads the repository/architecture/documentation contracts, General,
API/Javadoc, Planning, and Example profiles, final implementation and tests, relevant completed NN
tasks, Tensor/Training APIs, glossary, and Checkpoint plan. It finalizes the six affected production
and package Javadocs plus Training API, glossary, master/task evidence, and records exact no-change
conclusions for every reviewed unchanged boundary. It reuses successful Java evidence unless it
changes executable behavior or records a concrete reason to rerun it.

## Local decisions

- Use one ordinary constructor on final `Linear`. The user should describe architecture, not
  choose eager versus lazy public types or call a duplicate initialization pass.
- Keep `Parameter` always bound. A future declaration is private Module metadata until one complete
  layer-local forward initialization or strict whole-tree load publishes real wrappers.
- Use `RandomGeneratorFactory` plus seed because an unknown first-forward time cannot safely retain
  a mutable caller-owned generator. A fresh non-stochastic seeded generator makes retries
  deterministic without a hidden global/default source.
- Use one closed Linear-specific policy enum because the concrete consumer needs to select among
  four already implemented fan policies. Do not generalize it into an initializer object model.
- Fail complete parameter discovery and state export before all owned reservations bind. Empty or
  partial results would make optimizer/checkpoint schema silently depend on forward history.
- Let strict state load bind reservations because its complete candidate supplies every missing
  Shape and can be validated before installation; it consumes no initializer or RNG.
- Make initialization layer-local and traversal-driven. Arbitrary Java in `Model.define` prevents
  whole-body preflight or rollback without a much broader tracing lifecycle.
- Preserve opaque Tensor-ID non-rollback and distinguish parameter-leaf creation from actual
  Tensor.linear expression construction. Neither phase numerically executes a model.

## Known limitations

- Only the existing `Linear` gains automatic input-width initialization. Every other module keeps
  its current construction contract until a concrete follow-up chooses its own policy.
- The final input feature Dimension must be positive and static on first forward or strict-load
  candidate Shape. Leading Dimensions may be dynamic and may change later.
- Parameter/state discovery of a Model containing an unvisited automatic layer fails. There is no
  public initialization status, Model-wide branch coverage, or schema export before binding.
- A functional Model forward can leave an earlier layer initialized when later Java/module work
  fails. The task provides layer-local publication, not a global transaction.
- Only concurrent first-forward initialization on the same Linear has an internal synchronization
  guarantee. Replacement, strict load, tree traversal, mode changes, and arbitrary forward bodies
  retain their caller-coordinated contracts.
- Initialization is limited to the four current fan policies and zero bias. There is no custom
  initializer, automatic seed splitting, default source, or mutable source retention.
- This capability creates eager parameter leaves and declarative Model expressions. It proves no
  numeric result, compilation, gradient, backend route, physical execution, or durable checkpoint.

## Validation evidence

Planning-only replacement completed in clean task context `/root/nn_0019_replanning`.

- A targeted Ruby Markdown check passed for the NN master plan and this task: every local file
  target and heading fragment resolves, headings are unique, fences are balanced, terminal
  newlines are present, and no trailing whitespace was found.
- Exact frontier checks passed: this is the only `0019-*.md` task, the rejected
  `0019-deferred-parameter-binding-and-lazy-linear.md` path is absent, no NN 0020–0024 detailed
  specification exists, and exactly one NN task/master row is `Ready`.
- Exact planning scope passed: `git status --short` contains only the modified NN master plan and
  this new untracked task file. `git diff --name-only` plus `git ls-files --others
  --exclude-standard` reports those same two paths.
- `git diff --no-index --check /dev/null
  docs/planning/extensions/nn/tasks/0019-automatic-first-forward-linear-initialization.md`
  returned the expected difference status `1` with no whitespace diagnostic for the new file.
- `git diff --check` passed with no output for the tracked master-plan change.
- No Java, Javadoc, Gradle, architecture, conformance, integration, backend, or repository-wide
  test was run because this planning-only change modifies no executable or architecture behavior.

Implementation context `/root/nn_0019_auto_forward_impl` completed the executable capability.

- The focused command selecting `ModuleDeferredParameterTest`, `StateDictionaryTest`,
  `LinearTest`, `LinearInitializationTest`, and `ModelTest` passed 5 suites and 55 tests with zero
  failures, errors, or skips.
- The one authoritative final `./gradlew :extensions:nn:test` passed 31 suites and 226 tests with
  zero failures, errors, or skips. No executable Java or test changed afterward.
- Preliminary implementation Javadoc, public-surface, external-use, forbidden-mechanism/import,
  exact thirteen-path implementation scope, newline, whitespace, and `git diff --check` checks
  passed before the documentation handoff.

Independent clean documentation context `/root/nn_0019_docs` reviewed the final implementation
against the architecture, ADR 0007, planning/documentation profiles, completed NN ownership,
initializer, Linear, state-dictionary, unary-composition and typed-Model tasks, final source/tests,
generated Javadocs, Training API, glossary, and Checkpoint plan. It found no executable,
public-API, architecture, dependency, or task-scope defect and changed no executable Java or test.

- The pass finalized the six affected production/package Javadocs, `docs/api/training-api.md`,
  `docs/glossary.md`, the NN master plan, and this task. It explains that parameter leaves are
  validated and published before the same first call constructs its ordinary `Tensor.linear`
  expression; neither phase numerically executes the model or introduces a public lazy/build/
  initialize lifecycle.
- The final documentation covers exact inferred versus architectural Dimensions, constructor and
  forward validation, generator/factory/seed ownership, random-draw/allocation/Tensor-ID ordering
  and non-rollback, retry, layer-local synchronization and release/acquire publication, access and
  complete discovery/export gating, strict-load initialization without RNG, eager/supplied
  compatibility, variable leading Dimensions, and functional-Model partial effects.
- The pass reused the final 5-suite/55-test and 31-suite/226-test Java evidence because no
  executable Java or test changed after it; it did not repeat either suite.
- `./gradlew :extensions:nn:javadoc` passed after final Javadoc edits. Rendered `Linear`, `Module`,
  `LinearWeightInitialization`, and all three affected package pages were inspected for the final
  contracts.
- Final `javap` inspection showed exactly four public `Linear` constructors and the existing
  `weight`, `bias`, and `forward` methods; the enum has exactly four constants; Module's public
  surface is unchanged; and its exact four new reservation primitives are protected final. A
  standalone external-package `Model.define` example using two automatic Linear layers compiled.
- Production import/dependency and forbidden-public-Lazy/lifecycle scans passed. Markdown local
  links and anchors, balanced fences, exact fifteen-path scope, task/master Complete status,
  NN 0020–0024 Draft rows without specifications or Ready status, terminal newlines, trailing
  whitespace, and final `git diff --check` passed.
- `Parameter`, `Buffer`, `StateDictionary`, and `StateEntry` remain unchanged because their public
  bound-value invariants and signatures are still exact; private reservation/schema integration is
  Module-owned. `Model`, `Topology`, `Sequential`, and recurrent production contracts remain
  unchanged because automatic initialization follows ordinary Java traversal without adding a
  container lifecycle. Tensor/Compile APIs, architecture/ADR/tests, Checkpoint/Training/Data/Text/
  Vision plans, Gradle, compiler/runtime/prepare/Engine/backends, CPU, global roadmap,
  conformance, and integration remain unchanged because no semantic, dependency, build,
  persistence, or execution boundary changed.

## Implementation notes

- Isolated implementation context `/root/nn_0019_auto_forward_impl` replaced the direct Parameter
  map with private declaration slots while preserving encounter order and the shared namespace.
  A volatile completion gate publishes all wrappers in one direct reservation group and makes
  access/discovery observe unbound-or-complete state.
- `Linear` remains one final type. Its new constructor retains the exact deterministic factory,
  seed, type, output width, bias choice, and closed policy while reserving state without creating
  a generator, Tensor, identifier, or Parameter. First forward validates and initializes inside
  one layer-local critical section, then constructs the ordinary expression outside it.
- Strict load collects reserved paths as target schema, validates and constructs every new wrapper
  before installing any target, then publishes completed reservation groups. Loading uses exact
  candidate Tensor references and no initializer or random generator.
- Existing supplied-Tensor and explicit-dimension Linear construction, state names, wrapper
  replacement, promotion, and expression provenance remain covered by the complete NN suite.

## Completion summary

- Completed changes: Added the private Module reservation/publication lifecycle, one unified
  automatic-first-forward `Linear` constructor, the four-value Linear policy enum, strict-load
  initialization, eager compatibility, and focused regression coverage.
- Files changed or created: Exactly the planned six production/package paths, five test paths,
  Training API, glossary, NN master plan, and this task: fifteen paths.
- Tests and validation: Reused the implementation context's successful focused 5-suite/55-test and
  authoritative NN 31-suite/226-test results; final NN Javadoc/rendered pages, `javap`, external
  compilation, forbidden API/imports, Markdown, exact scope/status/frontier, newline, whitespace,
  and diff validation passed.
- Documentation-agent review: Independent clean context `/root/nn_0019_docs` found no executable,
  API, architecture, dependency, or scope defect and finalized all authorized documentation.
- Documentation impact: Training API and package/type Javadocs now describe automatic layer-local
  parameter publication before expression construction, failures, concurrency, strict load, and
  Model partial effects without claiming numerical execution or a general public lifecycle.
- Javadoc review: Final generation and rendered inspection passed for `Linear`, `Module`, the
  policy enum, and their three packages.
- Glossary impact: Updated the module/parameter/state-dictionary, Linear, and functional-Model
  entries to define reservations, automatic first-forward initialization, and its boundaries.
- Unresolved issues: None.
- Follow-up required: None. NN 0020 remains the next Draft row and has no detailed specification.

Status: Complete

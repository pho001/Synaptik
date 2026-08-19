# Task 0020B: Stateless Standard Module Factory

## Status

Complete

## Goal

Add one small descriptive construction facade for the five current standard initialized module
families used most often in typed functional models:

```java
ModuleFactory modules = ModuleFactory.standard();

var model = Model.define(topology -> {
    Embedding embedding = topology.addModule(
            "embedding",
            modules.embedding(
                    vocabularySize,
                    embeddingSize,
                    DataType.FLOAT32,
                    ParameterInitialization.glorotUniform(),
                    41L));
    LstmSequence encoder = topology.addModule(
            "encoder",
            modules.lstm(
                    128,
                    true,
                    DataType.FLOAT32,
                    ParameterInitialization.glorotUniform(),
                    42L));

    return (Tensor tokenIds) -> encoder.forward(embedding.forward(tokenIds));
});
```

`ModuleFactory.standard()` is an immutable recipe namespace with no instance fields. Each recipe
returns one fresh concrete module and delegates its complete construction, validation, state,
initialization, and failure semantics to the existing public constructor for that concrete family.
The factory does not register or own the result: `Topology.addModule` remains the sole functional-
Model ownership operation.

The abstraction is justified by a current concrete use: ordinary callers should be able to name
the desired module family without manually constructing recurrent cells or supplying the
`RandomGeneratorFactory` required by the advanced automatic `Linear` constructor. It is not an
extensible provider boundary, service locator, dependency-injection container, or generic Module
lifecycle.

## Motivation

The current API already has complete, validated constructors for initialized `Embedding`,
automatic `Linear`, and automatic-cell `RnnSequence`, `GruSequence`, and `LstmSequence`. Direct
construction remains important for advanced state, cell, random-source, and random-factory
control, but it makes the common model-definition example mix topology naming with construction
mechanics:

- `Embedding` callers spell the eager table constructor directly;
- automatic `Linear` callers must select a deterministic `RandomGeneratorFactory` even when they
  want the same documented standard algorithm as recurrent layers; and
- recurrent callers must know whether to construct a cell or the matching sequence container.

One closed concrete factory makes those five standard choices descriptive while preserving every
per-layer architectural choice. It owns no behavior that belongs to a layer and adds no dispatch
on runtime type, string, registry entry, or reflection.

## Scope

- Add final public `ModuleFactory` to the existing `io.github.pho001.synaptik.nn.module` package.
- Give it no public or protected constructor and no instance field.
- Add `ModuleFactory.standard()` returning the one immutable standard instance. Repeated calls
  return the same exact factory reference; the identity carries no caller or model state.
- Add exactly five instance recipes: `embedding`, `linear`, `rnn`, `gru`, and `lstm`.
- Require every recipe to accept the complete explicit per-layer data type,
  `ParameterInitialization`, and seed. `linear`, `rnn`, `gru`, and `lstm` also require the
  architectural output/hidden width and bias choice; `embedding` requires vocabulary and
  embedding sizes.
- Return the concrete current type from every recipe:
  - `embedding` returns `Embedding`;
  - `linear` returns `Linear`;
  - `rnn` returns `RnnSequence`;
  - `gru` returns `GruSequence`; and
  - `lstm` returns `LstmSequence`.
- Every recipe invocation creates a fresh module. No module, cell, Parameter, Tensor, state
  dictionary, topology, name, result, random generator, or seed is cached or retained by the
  factory.
- `embedding` delegates once to the existing five-argument eager `Embedding` constructor.
- `linear` delegates once to the existing automatic `Linear` constructor using the exact JDK
  deterministic factory named `L64X128MixRandom` and the caller's exact seed. The returned
  `Linear` retains that immutable factory/seed configuration under its existing contract; the
  `ModuleFactory` does not retain or create a generator. Zero/one policies still create no
  generator when the layer later binds.
- `rnn`, `gru`, and `lstm` each delegate once to the matching five-argument Sequence constructor.
  The returned Sequence owns the one fresh matching automatic Cell under current child name
  `cell`; no caller has to assemble that standard pair manually.
- Preserve all direct constructors as the advanced API for supplied state, supplied cells,
  explicit states/lengths, caller-owned `RandomGenerator`, or caller-selected deterministic
  `RandomGeneratorFactory` control.
- Add focused tests for exact public shape, statelessness, fresh result identity, concrete return
  types, exact delegated initialization/state behavior, recurrent cell ownership, validation and
  effect delegation, and use together with `Topology.addModule`.
- Finalize the new type Javadoc, module-package documentation, Training API, glossary, and
  planning evidence in the required separate clean documentation-focused context.

## Out of scope

- Changing, removing, deprecating, or wrapping any current constructor, accessor, forward method,
  state path, initializer, cell, sequence, result, or topology contract.
- Calling `Topology.addModule`, accepting a topology or name, registering or owning a result,
  building a `Model`, or adding/removing/replacing children after definition.
- A `ModuleFactory` interface, implementation/provider hierarchy, builder, configurable factory,
  registry, lookup by string/class, service locator, dependency-injection container, reflection,
  annotation scanning, or `ServiceLoader` integration.
- A global/default data type, policy, bias, width, seed, random source, mutable generator, seed
  sequence, seed splitting, per-model configuration, retained callback, or hidden mutable state.
- A generic `create(Class<?>)`, generic layer descriptor, `ModuleSpec`, serialized recipe, map of
  options, custom plugin, or public factory extension point.
- Factory recipes for `LayerNorm`, `BatchNorm`, `Dropout`, `Sequential`, cells by themselves,
  supplied-state variants, or another family. Their current constructors either already express
  the complete direct contract or need explicit context/state not simplified by this capability.
- Returning a generic `Module`, `UnaryTensorModule`, or recurrent base instead of the exact
  concrete result type.
- Inferring output width, hidden width, vocabulary size, embedding width, bias, data type, policy,
  or seed from input data or another recipe.
- Bidirectional/multidirectional recurrence, stacking, merge policy, reverse traversal, runtime
  valid lengths, masks, recurrent scan/control flow, or changes to current static packing.
- New Tensor/operation semantics, numerical execution, gradients, compiler behavior, optimizer or
  Training implementation, checkpoint transport, Runtime, Prepare, Engine, backend, build,
  dependency, architecture, conformance, or integration work.
- Production or documentation changes outside the exact affected paths, including the global
  roadmap and concurrent CPU/backend work.
- A detailed task file for NN 0020C or any later NN row.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)
- [Task 0018: Typed functional Model topology](0018-typed-functional-model-topology.md)
- [Task 0019: Automatic first-forward Linear initialization](0019-automatic-first-forward-linear-initialization.md)
- [Task 0020: Automatic recurrent initialization and sequence defaults](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Task 0020A: Initialized Embedding](0020a-initialized-embedding.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `extensions/nn` owns module construction and stateful neural-network composition and continues
  to depend only on `modules/model`.
- The factory is a convenience over already selected NN constructors. It must not move layer
  validation, initialization, state ownership, forward composition, or failure semantics away
  from each concrete layer.
- `Topology.addModule` remains the only operation that gives a functional Model permanent named
  ownership. A factory result is initially unowned except for the matching Cell already owned by
  each recurrent Sequence.
- `Module` remains the general state/tree/mode owner without a generic construction or lifecycle
  protocol. The factory must not require a new method on `Module`.
- Model owns Tensor metadata and expressions; Compiler owns automatic differentiation; Training
  owns optimizer algorithms/orchestration; execution layers own execution. The factory imports or
  reproduces none of those downstream concerns.
- Standard Linear and recurrent random policies use the already documented exact deterministic
  JDK algorithm and explicit per-layer seed. No source or generator becomes factory state.
- The convenience is cold construction work outside runtime hot paths and performs no reflection,
  map lookup, string dispatch, synchronization, or allocation beyond the concrete construction it
  requests. `standard()` itself may return a precreated singleton.
- If implementation needs another module, dependency, package, public extension point, layer
  behavior change, or ownership/lifecycle rule, stop and report the conflict.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.nn.module` — already owns Module structure, typed Model topology, and
  composition contracts; it gains the descriptive standard construction entry point used beside
  `Topology.addModule`.

No package is added or renamed. Type placement:

- `io.github.pho001.synaptik.nn.module.ModuleFactory` — final public instance-field-free standard recipe
  namespace returning exact concrete NN module types.
- `io.github.pho001.synaptik.nn.module.ModuleFactoryTest` — same-package API, singleton,
  statelessness, delegation, ownership, and topology-composition tests.

The existing layer and initialization packages are consumed through their public constructors and
need no executable change.

## Public API

```java
public final class ModuleFactory {
    public static ModuleFactory standard()

    public Embedding embedding(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed)

    public Linear linear(
            long outFeatures,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed)

    public RnnSequence rnn(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed)

    public GruSequence gru(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed)

    public LstmSequence lstm(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed)
}
```

There is no public/protected constructor or field, instance field, overload, interface, nested
type, accessor, configuration method, generic create method, topology method, or lifecycle method.

`rnn`, `gru`, and `lstm` deliberately name the familiar module family while their concrete return
types make the selected sequence container explicit at compile time. The factory does not return
cells because the standard ordinary recipe includes sequence ownership and defaults; direct cell
constructors remain available when one-step recurrence or custom assembly is required.

## Construction, ownership, and state semantics

`ModuleFactory.standard()` returns the same exact immutable instance-field-free object on every
call. It creates no module and selects no per-layer value.

Every recipe invocation is independent:

1. receive the complete caller-selected schema/policy/seed arguments;
2. invoke exactly one current public concrete constructor;
3. propagate its exact ordinary exception and effects without wrapping, fallback, retry, or
   rollback; and
4. return the exact freshly constructed concrete module.

The result is never memoized. Equal recipe arguments can produce equal represented initialized
values where the existing seeded constructor promises that, but the modules, wrappers, Tensor
identities/storage, sequences, and cells remain distinct according to their current constructors.

Ownership remains explicit:

```text
ModuleFactory.standard().embedding(...)
  -> fresh unowned Embedding
  -> caller may pass it to Topology.addModule("embedding", ...)

ModuleFactory.standard().lstm(...)
  -> fresh LstmSequence
     -> already owns its one fresh LstmCell as child "cell"
  -> caller may pass the Sequence to Topology.addModule("encoder", ...)
```

The resulting recurrent state paths remain, for example,
`encoder.cell.inputWeight`, `encoder.cell.hiddenWeight`, and optional `encoder.cell.bias` after
binding. The factory introduces no path segment and never knows the caller's topology name.

## Validation and effect order

The facade adds no independent argument normalization, defaults, validation exception, or recovery
policy. Each recipe preserves the selected constructor's documented validation and effect order:

- `embedding` is eager. It validates the complete explicit table schema, then may create the
  standard generator and parameter Tensor, and returns only a fully initialized layer.
- `linear` looks up the exact named deterministic factory needed to call the current advanced
  automatic constructor, but creates no random generator, Tensor, Tensor identifier, or Parameter.
  The returned layer performs its existing complete validation, binding, retry, strict-load, and
  first-forward semantics. Any invalid constructor argument is rejected by `Linear`; no module is
  returned.
- `rnn`, `gru`, and `lstm` construct one matching unbound automatic Cell and attach it to the new
  Sequence using the current validate-before-install child contract. They create no generator,
  Tensor, Tensor identifier, Parameter, default state, or length array during successful
  construction. Their later forward effects remain unchanged.

The exact JDK `L64X128MixRandom` factory lookup for `linear` is fixed implementation-neutral
standard-recipe selection, not mutable factory configuration or generator creation. It occurs
only inside the `linear` call and the returned `Linear`, not `ModuleFactory`, retains the exact
deterministic factory and seed. A missing required JDK algorithm is propagated; there is no
fallback. The implementation must not duplicate concrete constructor validation or catch and
translate its failures.

## Affected files

Expected exact paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModuleFactory.java` (new)
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/package-info.java`
3. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleFactoryTest.java` (new)
4. `docs/api/training-api.md`
5. `docs/glossary.md`
6. `docs/planning/extensions/nn/master-plan.md`
7. `docs/planning/extensions/nn/tasks/0020b-stateless-standard-module-factory.md` (new)

## Maximum scope

This task may create or modify at most the exact seven paths above. No existing layer,
initializer, Module, Topology, Model, state-dictionary, build, architecture, roadmap, backend, or
other task file may change. If correct implementation requires an eighth path or any constructor
behavior change, stop and propose a separate task.

## Acceptance criteria

- The exact public `ModuleFactory` API exists with no public/protected constructor or field,
  instance field, additional method, overload, nested type, provider, registry, configuration,
  topology, or lifecycle surface.
- `standard()` returns the same exact instance-field-free immutable object and has no module,
  state, or RNG effect.
- Every recipe requires explicit type, policy, and seed plus its exact architectural size and bias
  inputs; no global/default per-layer choice exists.
- Each invocation returns one fresh exact concrete type and performs one direct constructor
  delegation with no fallback, wrapping, cache, or memoization.
- `embedding` preserves the completed eager table policy, random/no-RNG routing, one complete
  `weight`, ordinary rows, state behavior, and validation/effect order.
- `linear` returns the current automatic `Linear`, selects exact `L64X128MixRandom` through its
  existing deterministic-factory boundary, and preserves deferred binding, zero/one no-generator,
  retry, strict-load, access/discovery, and forward behavior.
- `rnn`, `gru`, and `lstm` return matching Sequence types that each own one fresh matching
  automatic Cell under `cell`; current default-state/all-valid/explicit-state/static-length and
  state-path semantics remain unchanged.
- A factory-created module can be passed directly to `Topology.addModule`, which remains the sole
  owner and produces no factory-derived path segment.
- Direct constructors remain present and unchanged for advanced caller-owned random sources,
  factories, cells, supplied state, states, and lengths.
- Public/package Javadocs document concrete return types, explicit choices, standard PRNG
  selection, eager versus automatic effects, ownership, fresh identity, threading, and exclusions
  with complete parameter/return/failure tags.
- Training API and glossary explain the standard factory as descriptive construction beside
  `Topology.addModule`, not as Model topology, module ownership, registration, execution, or a
  global policy.
- Exact focused/final NN validation, warning-free Javadoc, public/private surface, external-use,
  import/dependency, forbidden-mechanism, Markdown, exact-scope, status/frontier, newline, and
  whitespace gates pass.
- A distinct clean documentation-focused context independently finalizes affected Javadocs,
  explanatory documentation, glossary impact, and final planning evidence before completion.

## Tests / validation

Implementation context:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.ModuleFactoryTest
./gradlew :extensions:nn:test
git diff --check
```

The focused suite must cover:

- exact public/private shape and absence of fields other than the one private static standard
  singleton;
- repeated exact `standard()` identity and lack of public construction/configuration;
- all five concrete return types and fresh identity from repeated equal calls;
- representative random and constant Embedding values/state against direct construction;
- automatic Linear binding/state against direct construction with the exact standard deterministic
  factory;
- one matching owned automatic Cell and exact `cell.*` paths for RNN, GRU, and LSTM after binding;
- exact constructor validation/failure propagation and no returned partial module;
- use inside `Model.define(topology -> ...)` with `Topology.addModule`; and
- unchanged availability of representative advanced direct constructors.

The separate documentation-focused context reuses stable Java-test evidence unless it changes
executable Java behavior. After final Javadoc edits it runs:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also inspects the rendered `ModuleFactory` and module-package pages, uses `javap`/reflection to
confirm the exact surface and absence of instance fields, compiles one external functional-Model example,
scans production imports and forbidden registry/reflection/service/configuration mechanisms,
validates local Markdown targets/anchors/headings/fences, checks exact seven-path scope, terminal
newlines and trailing whitespace, and confirms task/master frontier consistency.

Repository-wide, architecture, conformance, integration, compiler, runtime, prepare, Engine,
backend, numerical, and CPU validation remain deferred to the NN integration checkpoint or CI.
This task changes one existing model-only module, adds no dependency/build/architecture boundary,
and delegates every executable semantic to current tested constructors.

## Documentation pass

After executable Java and final NN test evidence stabilize, hand the exact seven-path diff and
test results to a separate clean documentation-focused agent/thread. That pass reads the
documentation rules and General, API/Javadoc, Planning, and Example profiles; this task; final
source/tests/generated pages; completed NN 0018–0020A contracts; Training API; glossary; and the
direct constructors used by every recipe.

It must finalize:

- `ModuleFactory` type, `standard()`, and all five recipe Javadocs;
- the `nn.module` package description;
- the Training API construction example and ownership distinction;
- the glossary entry and cross-links; and
- final task/master status and validation evidence.

It must record reasoned no-change conclusions for existing layer/initializer/Module/Topology/
Model/state Javadocs, Tensor and Compile APIs, Training Java, architecture/current plan/ADR/tests,
build/dependencies, compiler/runtime/prepare/Engine/backends, conformance/integration, CPU, global
roadmap, and later NN rows. It must not rerun stable Java tests unless executable behavior changes
or a concrete defect risk requires it.

## Dependencies

- NN 0001–0020A are Complete.
- `Topology.addModule` and immutable typed functional Model ownership are stable.
- The common closed `ParameterInitialization` and standard high-level `L64X128MixRandom` recurrent
  contract are stable.
- Initialized eager `Embedding`, automatic `Linear`, and standard automatic-cell Sequence
  constructors are complete and documented.
- Existing direct constructors provide the advanced random-source, factory, supplied-state, cell,
  state, and length escape hatches preserved by this task.
- The user-authorized NN interleave remains isolated from active CPU/backend/global-roadmap work;
  this task owns none of those paths.

## Follow-up tasks

- NN 0020C remains the next Draft row for type-safe directional recurrent composition after this
  factory capability is Complete. It has no detailed task specification yet.
- NN 0021–0024 remain Draft for the cross-module runtime scan, valid-length integration, optional
  arbitrary masks, and the integration checkpoint.
- Add another standard recipe only when a concrete module family has a complete stable
  constructor and the recipe removes real caller-side assembly without hiding required context or
  state. Do not turn this closed facade into a provider system.

## Architecture impact

Expected impact: None.

The existing architecture assigns stateful module composition and neural-network conveniences to
`extensions/nn`. This factory is a cold descriptive wrapper over current NN constructors and
retains the sole Model dependency. Stop if implementation requires a new ownership rule,
dependency, module, package, runtime service, or architecture change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are implementing Synaptik NN task 0020B. Do not use GSD. Do not commit, push, stage, revert,
or modify concurrent unrelated work unless the user explicitly requests it after completion.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0020b-stateless-standard-module-factory.md in full. Read
completed NN 0018–0020A, final Model/Topology/Module, ParameterInitialization, Embedding, Linear,
all recurrent Cell/Sequence constructors and directly relevant tests/Javadocs, Training API,
glossary, and the documentation rules/profiles named by the task.

Implement task 0020B exactly within its seven authorized paths. Add only the final instance-field-
free standard ModuleFactory and focused tests; delegate each recipe to the existing concrete
constructor, keep every type/policy/seed explicit, keep Topology.addModule as sole ownership, and
preserve all direct advanced constructors. Do not add a provider/interface/registry, global
configuration, hidden RNG/seed state, generic lifecycle, reflection/service lookup, layer behavior,
runtime work, architecture/build changes, or a later task file. Stop and report any architecture,
API, ownership, validation/effect-order, package, or exact-scope uncertainty instead of inventing
a broader design.

Run the focused and one final NN suite after executable Java stabilizes. Then hand the final diff
and exact evidence to a distinct clean documentation-focused agent/thread. That pass must
independently finalize the affected Javadocs, module package, Training API, glossary, master/task
evidence, final Javadoc, examples, links, no-change reasoning, exact scope, and status. Do not mark
Complete before both passes and all gates succeed.
```

## Local decisions

- Use one final concrete class rather than an interface. There is one selected standard behavior,
  no alternate implementation, boundary, test seam, or plugin consumer justifying polymorphism.
- Place it in `nn.module` because it is the descriptive construction entry used alongside Model
  topology and returns Modules; the existing `layers` package continues to own every concrete
  layer behavior.
- Make `standard()` a singleton accessor because the instance is immutable and has no fields.
  Repeated allocation would provide no distinct state or semantics.
- Keep exactly five recipes proven by current constructors. Do not add a broad facade over every
  Module merely for API symmetry.
- Return concrete types so callers retain exact forward/result/state contracts and no generic
  Module forward or recurrent base is invented.
- Let `rnn`/`gru`/`lstm` mean the standard Sequence recipe. The concrete return type makes that
  choice visible, while direct Cell construction remains the explicit one-step/custom path.
- Use exact `L64X128MixRandom` for standard automatic Linear so the common facade matches the
  already documented high-level recurrent algorithm. Keep the existing direct automatic Linear
  constructor for callers needing another deterministic factory.
- Delegate validation and effects instead of duplicating layer behavior. The factory is a
  convenience namespace, not an alternate implementation of initialization or ownership.

## Known limitations

- The factory has no recipe for mode-sensitive, explicit-context, supplied-state, or custom-cell
  variants. Callers use direct constructors for those contracts.
- `embedding` eagerly allocates the complete table when called; `linear` and recurrent recipes
  return automatic modules whose state binds later under their current rules.
- `linear` uses the exact standard deterministic algorithm. A caller requiring another factory
  uses the direct `Linear` constructor.
- The factory neither registers modules nor assigns topology names. Every returned top-level
  module remains unowned until caller code installs it.
- Static recurrent sequences still use Java lengths, static Shapes, and current default states.
  This task adds no directionality, runtime scan, or runtime valid-length input.
- The convenience constructs Tensor leaves/expressions only through existing types; it proves no
  numerical result, gradient execution, backend support, or performance property.

## Validation evidence

Planning-only work completed in clean context `/root/nn_next_planning`. No Java, test, Javadoc,
build, architecture, API-guide, glossary, global-roadmap, CPU, or backend implementation was
performed.

- Read the repository instructions, authoritative architecture contract, current architecture
  index, planning guide, current roadmap, NN master plan, completed NN 0018–0020A task records,
  and the directly relevant final Model/Topology/Module, initialization, Embedding, Linear,
  recurrent Cell/Sequence source, Javadocs, and tests. The current master-plan order verifies
  0020B as the first unfinished NN row after committed 0020A.
- The constructor review found no architecture, dependency, ownership, package, or lifecycle
  blocker. Existing constructors support one narrow final factory without changing layer
  behavior: eager `embedding`; automatic `linear` with the exact standard deterministic factory;
  and automatic-cell `rnn`/`gru`/`lstm` returning concrete Sequences. The specification keeps
  every type/policy/seed explicit, preserves advanced direct constructors, and leaves
  `Topology.addModule` as the sole top-level ownership operation.
- The design review rejected an interface/provider hierarchy, configurable/global policy,
  registry, service locator, generic create method, generic lifecycle, retained RNG/seed manager,
  generic recurrent base, and recipes without a concrete current simplification. One final
  instance-field-free class and five exact concrete methods are sufficient under the repository's
  abstraction and performance discipline.
- Targeted Markdown validation passed for this task and the NN master plan: all local link targets
  and heading fragments resolve, effective heading anchors are unique, fences are balanced,
  terminal newlines are present, and no trailing whitespace was found.
- Frontier checks passed: this task is the sole NN task with scalar status `Ready`; the NN master
  has exactly one `Ready` row linked to it; 0020C and 0021–0024 remain Draft; and no later detailed
  task file exists.
- Exact NN planning scope passed. The only NN paths are the modified master plan and this new task
  file. Pre-existing concurrent CPU/backend source, tests, planning, and global-roadmap changes
  were inspected for attribution and left untouched, unstaged, unreverted, and unformatted.
- `git diff --check` passed. The new-file no-index whitespace check returned only the expected
  difference status and no diagnostic; both planning files passed terminal-newline and trailing-
  whitespace checks.
- No Java, Javadoc, Gradle, architecture, conformance, integration, compiler, runtime, prepare,
  Engine, backend, CPU, or repository-wide test was run because this change is planning-only and
  modifies no executable or authoritative architecture behavior.

Implementation work completed in clean context `/root/nn_0020b_implementation`. Executable Java
and focused tests were frozen before the mandatory distinct clean documentation-focused pass.

- `./gradlew :extensions:nn:test --tests
  io.github.pho001.synaptik.nn.module.ModuleFactoryTest` passed: 9 tests, zero skips, failures, or
  errors. The suite covers exact public/private shape, the one private static singleton and zero
  instance fields, singleton identity, construction-time Tensor-ID effects, fresh concrete
  results, exact eager Embedding and standard-Linear initialization parity, recurrent automatic
  Cell state and `cell.*` paths, delegated failures, functional Model ownership, and representative
  advanced direct constructors.
- After executable and test freeze, `./gradlew :extensions:nn:test` passed once: 265 tests across
  the NN module, zero skips, failures, or errors. No executable Java changed afterward.
- Preliminary `./gradlew :extensions:nn:javadoc` passed without warnings. The implementation pass
  inspected the rendered `ModuleFactory` and `nn.module` package pages. This evidence is preliminary;
  the documentation context must finalize Javadocs and rerun final Javadoc afterward.
- `javap -p` confirmed one final public class, one private static final `STANDARD` field, one private
  constructor, and exactly `standard`, `embedding`, `linear`, `rnn`, `gru`, and `lstm` with their
  planned concrete return and parameter types. `javap -c` confirmed that `standard()` only reads the
  singleton, every recipe invokes exactly one matching public constructor, and `linear` performs the
  one exact `RandomGeneratorFactory.of("L64X128MixRandom")` lookup before its constructor call.
- An external Java source compiled against NN and Model production classes while using
  `ModuleFactory.standard()`, factory-created Embedding/LSTM modules, `Topology.addModule`, and
  `Model.define`; no package-private construction or raw/generic fallback was needed.
- Production import and forbidden-mechanism scans found only Model, current NN, and JDK random
  imports in `ModuleFactory`; no Training, Compiler, Runtime, Prepare, Engine, backend, registry,
  provider, service lookup, reflection, generic create, map, synchronization, atomic state, default
  type/seed, retained generator, or seed-sequence mechanism was added. The existing NN Gradle
  dependency remains only `modules/model`.
- Scope inspection found the five implementation-owned authorized paths only: new
  `ModuleFactory.java`, updated `nn.module/package-info.java`, new `ModuleFactoryTest.java`, and the
  in-progress NN master/task planning files. Training API and glossary are intentionally reserved
  for the mandatory documentation pass. Concurrent CPU/backend and global-roadmap changes remain
  untouched, unstaged, unreverted, and unformatted.
- `git diff --check` passed after the implementation edits. Repository-wide, architecture,
  conformance, integration, compiler, runtime, prepare, Engine, backend, numerical, and CPU tests
  remain deferred as specified because the task changes no dependency or architecture boundary.

Independent clean documentation context `/root/nn_0020b_docs` reviewed the final implementation,
tests, generated Javadocs, constructor contracts, state/ownership boundaries, Training API,
glossary, and planning frontier. It found no executable, public-API, architecture, dependency, or
scope defect and changed no executable Java or test.

- The pass finalized `ModuleFactory` and `nn.module` package Javadocs, the Training API's
  `Model.define`/`Topology.addModule` example and ownership explanation, the glossary's standard
  module-factory definition, and synchronized task/master evidence. The documentation now states
  fresh exact concrete identity, singleton statelessness, explicit per-call schema/policy/seed,
  exact standard Linear PRNG selection, eager versus automatic effects, recurrent Cell ownership,
  caller-coordinated concrete-module threading, advanced direct constructors, and the absence of
  global configuration, registry/provider/plugin/service-locator/reflection/generic lifecycle.
- Reused the implementation context's focused 9/9 and authoritative NN 265/265 test results because
  executable Java and tests remained frozen. No Java test suite was repeated.
- Final `./gradlew :extensions:nn:javadoc` passed without warnings after the Javadoc edits. The
  rendered `ModuleFactory` and `nn.module` package pages were inspected for the final type/member,
  parameter, return, failure, identity, ownership, effect, threading, and exclusion contracts.
- Final `javap -p` and `javap -c -p` confirmed the one final public class, private static final
  singleton, private constructor, zero instance fields, exact six public methods, exact concrete
  return types, singleton-only `standard()`, one matching constructor call per recipe, and the one
  exact `RandomGeneratorFactory.of("L64X128MixRandom")` lookup for Linear. A fresh external Java
  `Model.define` example using factory-created Embedding/LSTM modules and `Topology.addModule`
  compiled, and its reflection/singleton assertions passed.
- `jdeps` and production-import scans found only current NN, Model, and `java.base` dependencies;
  `extensions/nn/build.gradle.kts` still declares only `modules/model`. Forbidden-mechanism scans
  found no Training, Compiler, Runtime, Prepare, Engine, backend, registry, provider, plugin,
  service lookup, reflection, generic create, map dispatch, synchronization, atomic/thread-local
  state, default type/seed, retained generator, or seed-sequence mechanism.
- The targeted Ruby documentation validator passed for Training API, glossary, NN master, and this
  task: local paths and heading fragments resolve, effective heading anchors are unique, fences
  are balanced, terminal newlines are present, and no trailing whitespace exists. The exact final
  task scope is the seven authorized paths and contains no other NN path. Task/master checks show
  0020B Complete, 0020C and 0021–0024 Draft, no Ready NN row or task, and no later task file.
- New-file no-index whitespace checks and final `git diff --check` passed. Concurrent CPU/backend
  source, tests, planning, and global-roadmap changes remained attributed to their other work and
  were not touched, staged, reverted, reformatted, or incorporated.
- Architecture/current-plan/ADR/architecture-test changes are unnecessary because the facade stays
  within NN's existing Model-only construction/composition boundary and changes no dependency or
  ownership rule. Existing layer/cell/sequence, `ParameterInitialization`, Module, Topology, Model,
  state-dictionary, Tensor API, Compile API, and Training Java contracts remain accurate because
  the facade delegates to them without changing validation, effects, state, forward behavior, or
  executable semantics. Training API prose and glossary did change because their former
  no-ModuleFactory statements became stale.
- Compiler, Runtime, Prepare, Engine, concrete backends, Gradle/build structure, conformance,
  integration, architecture fixtures, CPU work, other modules, and the global roadmap require no
  task change or validation because no operation, gradient, execution, build, dependency, backend,
  or cross-module contract changed. Repository-wide testing remains deferred to the recorded NN
  checkpoint or CI for that same bounded-scope reason.

## Implementation notes

- Added one final `ModuleFactory` in `nn.module` with a private constructor, one private static
  singleton, no instance field, and the exact five concrete recipes. No helper, interface, nested
  type, overload, registry, configuration surface, or generic lifecycle was needed.
- Each method is a single direct construction path. Embedding uses its eager five-argument
  constructor; Linear supplies the exact named standard deterministic factory to its existing
  automatic constructor; and each recurrent recipe calls the matching five-argument Sequence
  constructor so that the Sequence continues to own its fresh Cell under `cell`.
- Existing layer constructors, validation, state ownership, RNG creation, Tensor-ID order,
  retry/publication behavior, forward semantics, exceptions, and direct advanced APIs were not
  modified. `Topology.addModule` remains the sole top-level ownership operation available to a
  caller.
- The separate documentation-focused context finalized the type/member and package Javadocs plus
  Training API, glossary, master/task evidence, links, rendered pages, and final status under the
  General, API/Javadoc, Planning, and Example profiles.

## Completion summary

- Completed changes: Implemented the stateless singleton standard ModuleFactory and its five exact
  direct-constructor recipes, plus focused surface, identity, delegation/effect, state, failure,
  ownership, and compatibility tests.
- Files changed or created: Exactly `ModuleFactory.java`, `nn.module/package-info.java`,
  `ModuleFactoryTest.java`, Training API, glossary, NN master plan, and this task: seven paths.
- Tests and validation: Reused focused 9/9 and authoritative NN 265/265 results; final NN Javadoc,
  rendered pages, bytecode/reflection surface, external use, dependencies/imports,
  forbidden mechanisms, Markdown, exact scope/status/frontier, newline, no-index, whitespace, and
  diff gates passed.
- Documentation-agent review: Independent clean context `/root/nn_0020b_docs` found no executable,
  API, architecture, dependency, or scope defect and finalized every authorized documentation path.
- Documentation impact: Training API and package/type Javadocs now distinguish stateless
  construction from `Topology.addModule` ownership and preserve concrete constructor boundaries.
- Javadoc review: Final warning-free generation and rendered inspection passed for
  `ModuleFactory` and the `nn.module` package.
- Glossary impact: Added the standard module-factory term and removed stale future-only claims.
- Unresolved issues: None.
- Follow-up required: None. NN 0020C remains a Draft row without a detailed task specification.

Status: Complete

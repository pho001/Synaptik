# Task 0020A: Initialized Embedding

## Status

Complete

## Goal

Add one eager initialized constructor to the existing final `Embedding` while preserving its
caller-supplied constructor, single `weight` state leaf, unary forward contract, mode behavior,
replacement behavior, and state-dictionary schema:

```java
public Embedding(
        long vocabularySize,
        long embeddingSize,
        DataType dataType,
        ParameterInitialization weightInitialization,
        long seed)
```

`vocabularySize` and `embeddingSize` are positive explicit architecture or schema decisions. The
constructor creates the complete `[vocabularySize, embeddingSize]` table during construction; it
does not observe token IDs, reserve state, or wait for a first forward call. All eight existing
`ParameterInitialization` policies apply through the current `ParameterInitializers` dispatch.
The six random policies use one fresh standard `L64X128MixRandom` source created from `seed`, while
zero and one neither create nor consume a random source.

Every row in the first initialized-Embedding contract is an ordinary trainable row. There is no
padding index, padding-row rewrite, frozen-row promise, or update exception. Future Text schema
owns padding-token identity, proposed Data owns canonical valid lengths for ordinary right
padding, and neither concern changes this table's parameter semantics.

## Mental model

```text
explicit vocabularySize + embeddingSize + type + policy + seed
  -> validate the complete table schema and Java-array feasibility
  -> random policy: create one standard L64X128MixRandom source
     constant policy: create no source
  -> dispatch once to ParameterInitializers for Shape [vocabularySize, embeddingSize]
  -> bind the returned eager Tensor as the permanent Parameter named "weight"
  -> later forward calls remain currentWeight.embedding(indices)
```

Construction initializes storage, not the numerical result of a lookup. The returned parameter
Tensor is an eager leaf; `forward` continues to construct an ordinary declarative Gather
expression without reading index values or executing it.

## Scope

- Keep the existing final `Embedding extends UnaryTensorModule` as the only Embedding type.
- Retain `Embedding(Tensor weight)` unchanged as the advanced caller-supplied-state path.
- Add exactly the five-argument eager constructor shown in the goal.
- Require positive explicit `vocabularySize` and `embeddingSize`; never infer either value from an
  index Tensor, maximum token ID, vocabulary object, batch, or first call.
- Construct exact `Shape.of(vocabularySize, embeddingSize)` and initialize it eagerly.
- Accept exactly FLOAT64, FLOAT32, or BFLOAT16 parameter type through the existing floating-type
  contract.
- Apply the exact supplied `ParameterInitialization` to the complete rank-two Shape by invoking
  the current exhaustive dispatcher once.
- For Glorot and Kaiming policies, retain the generic current rank-two interpretation: Shape axis
  zero is `fanOut` and axis one is `fanIn`. For this complete Embedding Shape that means
  `fanOut = vocabularySize` and `fanIn = embeddingSize`; add no Embedding-specific fan value,
  mode, gain, or distribution.
- Use `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` exactly once for each
  random-policy construction attempt after deterministic preflight, even if a later eager effect
  fails. Retain neither factory nor generator.
- Route zero and one through `ParameterInitializers.initialize(shape, dataType, policy)` and do
  not look up, create, retain, or consume any random generator.
- Bind the exact returned Tensor as the sole direct permanent `Parameter` named `weight`.
- Preserve current `weight()`, `forward(Tensor)`, replacement, train/eval, recursive discovery,
  snapshot, strict-load, and state-dictionary behavior.
- Update the directly affected Javadocs, package documentation, Training API explanation, and
  glossary in a separate clean documentation-focused pass before completion.

## Out of scope

- `LazyEmbedding`, `Embedding.lazy`, reservation, first-forward binding, `build`, `bind`, `init`,
  `initialize`, status, retry, or public schema-inspection APIs.
- Inferring vocabulary size from observed identifiers or accepting a vocabulary/tokenizer object.
- A `paddingIndex`, special padding row, automatic row zeroing, frozen row, sparse gradient,
  maximum-norm, frequency scaling, or optimizer/update hook.
- Tokenization, vocabulary construction/freeze/fingerprint, special-token assignment, batching,
  truncation, valid lengths, masks, runtime input binding, or a Text/Data module implementation.
- `ModuleFactory` or another construction facade; that remains Draft NN 0020B.
- Bidirectional/multidirectional recurrence, runtime recurrent scan/control flow, or valid-length
  recurrence; those remain later NN tasks.
- Any new operation, numerical lookup, gradient formula, optimizer, compiler, Prepare, Runtime,
  Engine, backend, execution, conformance, integration, build, module-boundary, dependency, ADR,
  or architecture change.
- Changing the existing caller-supplied constructor or general Module/Parameter/StateDictionary
  lifecycle.
- Copying legacy code or importing a new dependency.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)
- [NN 0004 explicit eager parameter initializers](0004-explicit-eager-parameter-initializers.md)
- [NN 0007 Embedding layer](0007-embedding-layer.md)
- [NN 0010 state dictionary and checkpoint contract](0010-state-dictionary-and-checkpoint-contract.md)
- [NN 0019 automatic first-forward Linear initialization](0019-automatic-first-forward-linear-initialization.md)
- [NN 0020 automatic recurrent initialization and sequence defaults](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Glossary](../../../../glossary.md)
- [Proposed Text master plan](../../text/master-plan.md)
- [Proposed Data master plan](../../data/master-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation profile](../../../../developer-guide/documentation/general-style.md)
- [API/Javadoc profile](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning profile](../../../../developer-guide/documentation/planning-style.md)
- [Example profile](../../../../developer-guide/documentation/example-format.md)

`ARCHITECTURE.md` remains authoritative if an explanatory or planning document disagrees.

## Architecture constraints

- NN may own layer construction, parameter ownership, and typed forward composition while
  depending only on Model. This task stays inside that existing boundary.
- Model continues to own Tensor metadata, eager leaf creation, Tensor identity, expression
  provenance, and the ordinary embedding/Gather convenience.
- Training remains downstream of NN. This task documents state available to training but adds no
  NN dependency on Training and no optimizer behavior.
- Text and Data remain proposed future boundaries, not dependencies. NN receives numeric schema
  values and never tokenizes or pads raw data.
- The constructor is eager because every state Shape fact is already explicit. Reusing the
  reservation/automatic lifecycle would add incomplete state without an input-dependent fact.
- Parameter state remains one exact dense Tensor leaf. There is no hidden row mutation or second
  representation of a padding invariant.
- Existing mode/state-tree behavior and the `UnaryTensorModule` composition boundary are
  unchanged.

## Package impact

- `io.github.pho001.synaptik.nn.layers`: one constructor on existing `Embedding`, plus Javadoc and
  package documentation.
- `io.github.pho001.synaptik.nn.initialization`: no executable API change; generalize explanatory
  Javadoc that currently describes fan Shapes as Linear-only so it truthfully covers the complete
  positive rank-two Shape consumed by Embedding.
- Tests remain in the existing `EmbeddingTest`; no new production type or test class is needed.
- No package outside `extensions/nn` changes executable source.

## Public API

After this task, the complete declared public Embedding surface remains one final class with two
constructors and two methods:

```java
public final class Embedding extends UnaryTensorModule {
    public Embedding(Tensor weight);

    public Embedding(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed);

    public Parameter weight();

    @Override
    public Tensor forward(Tensor indices);
}
```

Do not add a getter for sizes, type, policy, seed, random source, or padding configuration. The
current weight descriptor and Parameter schema remain the truthful state surface.

## Construction lifecycle and state contract

The initialized constructor completes all work synchronously in its calling thread:

1. validate caller-controlled scalar, reference, type, Shape, element-count, and Java-array-limit
   facts in the exact order specified below;
2. create a random source only if the selected policy requires one;
3. invoke the matching `ParameterInitializers.initialize` overload exactly once with the complete
   Shape and exact type/policy;
4. after the initializer returns, declare the exact returned Tensor under direct name `weight`;
5. return a fully initialized layer whose discovery and state export work immediately.

Successful construction has exactly one permanent state leaf at path `weight`, kind Parameter,
Shape `[vocabularySize, embeddingSize]`, exact requested floating type, and
`requiresGrad == true`. The Tensor is the dispatcher's fresh dense-contiguous, host-backed,
provenance-free, unlabeled leaf. Success consumes exactly one Tensor identifier through that
initializer. Parameter declaration creates no second Tensor, copy, label, producer, or random
draw.

There is no reservation or incomplete discovery state. Construction failure returns no layer and
publishes no Parameter wrapper. The random source is transient local construction state and is
not retained even after success.

## Validation and effect order

The constructor must perform deterministic validation in this order before random-source lookup,
random draw, Model carrier/destination allocation, Tensor identifier creation, or Parameter
declaration:

1. reject `vocabularySize <= 0` with `IllegalArgumentException`;
2. reject `embeddingSize <= 0` with `IllegalArgumentException`;
3. require non-null `dataType` with `NullPointerException`;
4. require non-null `weightInitialization` with `NullPointerException`;
5. reject a non-floating `dataType` with `IllegalArgumentException`;
6. create exact `Shape.of(vocabularySize, embeddingSize)`, obtain its checked known element count,
   propagate `ArithmeticException` for count overflow, and reject a count above
   `Integer.MAX_VALUE` with `IllegalArgumentException`.

Policy factory argument validation has already occurred when the immutable
`ParameterInitialization` was created. After constructor preflight:

- a random policy looks up the exact standard factory name, creates one fresh source from the
  exact `seed`, and passes it once to the four-argument dispatcher;
- zero or one invokes the three-argument dispatcher once and does not look up or create a factory
  or source; the accepted `seed` has no observable effect; and
- the dispatcher's documented allocation, draw, conversion, identifier, exception, and
  non-rollback behavior remains authoritative after effects begin.

The delegated post-preflight effect order is exact:

| Policy route | Effects after dispatcher validation |
|---|---|
| six random policies | Allocate the typed source carrier, invoke the matching generator method once per row-major element while converting into that carrier, allocate and copy into the independent destination carrier, then obtain the one Tensor identifier. |
| zero | Allocate the directly zero-initialized destination carrier, then obtain the one Tensor identifier; there is no source carrier, fill loop, copy, factory lookup, or draw. |
| one | Allocate and fill the typed-one source carrier, allocate and copy into the independent destination carrier, then obtain the one Tensor identifier; there is no random factory, source, or draw. |

If a random-source, allocation, or identifier failure occurs after effects begin, already consumed
draws or transient allocations are not rolled back, no Parameter is declared, and no layer is
returned. A source exception occurs before destination allocation and identifier acquisition;
random-policy identifier exhaustion occurs after every draw and both carrier allocations. Zero
identifier exhaustion occurs after destination allocation, while one exhaustion occurs after
source and destination allocation. Do not add recovery, retry, synchronization, or transaction
semantics around these eager effects.

## Fan-policy contract

Glorot and Kaiming policies receive the exact complete
`Shape.of(vocabularySize, embeddingSize)`. They use the existing generic positive rank-two rule:

- axis zero, `vocabularySize`, is `fanOut`;
- axis one, `embeddingSize`, is `fanIn`;
- Glorot derives its current scale/bound from both values; and
- Kaiming-ReLU derives its current scale/bound from `fanIn`.

This is deliberate reuse of the common policy, not a claim that Embedding requires a distinct
fan abstraction. Do not transpose the Shape, substitute equal fans, use only embedding width for
Glorot, or add an Embedding-specific default. Configured normal/uniform and constants retain their
ordinary whole-table semantics.

## Padding policy

The selected first contract is explicit absence of special padding behavior:

- every row is initialized by the selected whole-table policy;
- every row has the same ordinary gradient eligibility and replacement semantics;
- neither row zero nor any caller-selected row is overwritten after initialization;
- `forward` remains ordinary Gather and carries no padding identity; and
- state export/load retains the complete table exactly.

A future Text vocabulary may assign a padding token ID, and proposed Data may carry valid lengths,
but those schema facts do not freeze a parameter row. A future request for an invariant padding
row is a new capability trigger: it must define compatible gradient, replacement, optimizer
update, load, and checkpoint semantics before NN exposes such an API. Merely zeroing a row at
construction would be misleading because current updates can change it.

## Affected files

Implementation is limited to these nine paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Embedding.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`
4. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`
5. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`
6. `docs/api/training-api.md`
7. `docs/glossary.md`
8. `docs/planning/extensions/nn/master-plan.md`
9. `docs/planning/extensions/nn/tasks/0020a-initialized-embedding.md` (new)

Do not edit `ParameterInitialization`, other layer/module/state-dictionary source or tests, Tensor
or Compile API, architecture files, ADRs/tests, global roadmap, Text/Data plans, other master/task
files, build files, Training Java source, compiler/runtime/prepare/engine, backends, conformance,
or integration tests. If correct implementation requires another path, stop and report the exact
blocker before expanding scope.

## Maximum scope

The task may create or modify at most the exact nine paths above. Production behavior is one
constructor in one existing layer. The two initialization documentation paths are included only
because their current Linear-only fan wording becomes inaccurate when the existing generic
dispatcher is deliberately applied to Embedding. The Training API and glossary are the two
current explanatory sources whose statements that Embedding has no initialized constructor must
change in the required documentation pass.

## Test requirements

### Public surface and compatibility

- Reflectively verify `Embedding` remains final, extends `UnaryTensorModule`, and declares exactly
  the existing supplied constructor plus the new five-argument initialized constructor.
- Verify its only declared public methods remain `weight()` and `forward(Tensor)` and that no lazy,
  status, padding, schema, initializer, seed, source, factory, or alternate Embedding type appears.
- Re-run every current supplied-table validation, forward provenance, replacement, mode,
  parameter discovery, state snapshot, and strict-load assertion unchanged.

### Initialization policy matrix

- For each of all eight policies and each of FLOAT64, FLOAT32, and BFLOAT16, construct a small
  positive table and assert exact Shape, type, gradient eligibility, dense host storage,
  provenance absence, label absence, state path, and one successful Tensor-identifier advance.
- Compare represented table values exactly with a direct call to the matching current dispatcher:
  use a separately created standard `L64X128MixRandom` with the same seed for each random policy,
  and the no-generator overload for zero/one.
- Test representative unequal `[vocabularySize, embeddingSize]` Shapes for every fan policy so a
  transpose or equal-fan shortcut fails. Assert current formulas treat vocabulary size as fanOut
  and embedding size as fanIn without introducing a new fan API.
- Two layers with equal shape/type/policy/seed have equal represented values but distinct Tensor
  identities, Parameter wrappers, and storage. Different seeds change representative random
  values; zero/one values are seed-independent.
- Verify all rows, including row zero, contain the selected policy's ordinary values and share the
  same ordinary trainable table semantics. Do not invent numerical optimizer or gradient tests.

### Validation and effects

- Cover non-positive vocabulary size before every later failure, then non-positive embedding size,
  null type, null policy, non-floating type, checked element-count overflow, and count above
  `Integer.MAX_VALUE` in the specified order.
- For every deterministic rejection, assert no Tensor identifier is consumed and no layer/state is
  returned. Where existing test seams expose Model allocation/source effects, assert none began.
- Verify random policies use the exact `L64X128MixRandom` algorithm and exactly one source stream,
  while zero/one produce exact values for multiple seeds without any random route.
- Cover identifier exhaustion for at least one random and one constant policy and preserve the
  dispatcher's documented effect order while publishing no Parameter/layer.
- Preserve initializer exceptions and allocation failures without wrapping them in a new layer
  exception or claiming rollback.

### Forward, replacement, and state schema

- Immediately after successful construction, assert `weight()` is stable and recursive/direct
  parameter discovery and `stateDictionary()` contain exactly `weight`.
- Assert compatible replacement and strict load install the exact candidate Tensor reference and
  affect only later calls; incompatible kind/type/Shape/gradient state remains rejected by current
  contracts.
- Assert `forward` reads current weight once and produces the same fresh ordinary axis-zero Gather
  expression with exact indices and weight provenance as the supplied constructor.
- Assert train/eval remain mode-insensitive and do not change initialized state.
- Assert no padding state, buffer, row rewrite, token-ID inspection, or extra operation appears.

### Exclusions

- Search source, compiled public surface, tests, Javadocs, and current explanatory docs for
  accidental `LazyEmbedding`, `paddingIndex`, padding-row mutation, ModuleFactory, tokenizer/Data,
  runtime scan, compiler/backend, or execution implementation.
- Confirm no new module dependency, build path, architecture rule, operation kind, conformance
  fixture, integration fixture, or unrelated refactor is present.

## Documentation requirements

After executable implementation and focused tests, a distinct clean documentation-focused agent
must independently review and finalize the four affected production/package Javadocs, Training
API, glossary, this task, and the NN master plan. Apply the General and API/Javadoc profiles to
production documentation, the General and Planning profiles to planning files, and the Example
profile to code examples.

Documentation must explain:

- the exact new eager constructor and why explicit complete Shape means no lazy lifecycle;
- all eight policies, exact standard random algorithm, and zero/one no-RNG behavior;
- validation and effect order, including one successful Tensor ID and non-rollback boundaries;
- the complete `[vocabularySize, embeddingSize]` fan interpretation;
- one permanent `weight` state leaf and unchanged supplied/forward/replacement/mode/load behavior;
- explicit absence of padding-row semantics and the future contract required before adding one;
- current versus proposed Text/Data boundaries without presenting proposed modules as implemented;
  and
- declarative Tensor-expression construction rather than numerical execution, gradient, compiler,
  runtime, or backend behavior.

Review `ParameterInitialization` Javadoc, Module/Parameter/StateDictionary Javadocs, Tensor API,
Compile API, Training Java API, architecture documents/tests, build files, other modules, and
conformance/integration documentation. Record reasoned no-change conclusions unless a concrete
inaccuracy requires stopping for scope expansion. Do not repeat a successful Java suite merely to
reproduce implementation evidence when the documentation pass changes no executable Java.

## Acceptance criteria

- The existing final `Embedding` and `Embedding(Tensor)` remain source-compatible.
- Exactly the specified eager constructor is added, with no companion lifecycle or factory API.
- Both explicit sizes are positive and never inferred from token IDs or first-forward input.
- Every deterministic validation completes in the specified order before random/allocation/ID
  effects.
- All eight existing policies delegate once through `ParameterInitializers` for exact complete
  Shape `[vocabularySize, embeddingSize]`.
- Six random policies use one fresh exact standard `L64X128MixRandom` source seeded by `seed`;
  zero/one neither create nor consume an RNG.
- Fan-based policies use the current complete rank-two rule with vocabulary as fanOut and
  embedding width as fanIn; no new fan abstraction exists.
- Success eagerly creates exactly one fresh gradient-enabled floating Tensor leaf and one permanent
  Parameter named `weight`; discovery and state export are immediately complete.
- Existing forward, replacement, mode, and state-dictionary behavior remains unchanged.
- Every row is ordinary trainable state. There is no padding index, row zeroing, frozen row, or
  token/data ownership leak.
- No compiler/runtime/backend/build/architecture/dependency or unrelated behavior changes.
- Focused and full NN validation, warning-free Javadoc, documentation checks, public-surface
  checks, exact-path checks, and whitespace gates pass.
- A clean documentation-focused pass has finalized affected documentation and recorded all
  no-change reviews.

## Tests / validation

Implementation context, run focused tests while iterating:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest
```

Then run the affected module and its Javadocs once as the authoritative Java validation:

```bash
./gradlew :extensions:nn:clean :extensions:nn:test :extensions:nn:javadoc
```

The implementation and documentation passes must also validate:

- reflected and external-use public constructor/method surface;
- exact initialization values, standard algorithm, state schema, identifier/effect order, and
  forward provenance;
- forbidden source/import/API and stale-current-claim searches;
- local Markdown links and anchors, unique headings, balanced fences, rendered Javadoc pages,
  and documented examples;
- exact nine-path scope, terminal newlines, trailing whitespace, no-index whitespace for new
  files, and `git diff --check`; and
- master/task frontier consistency: 0020A is Complete after the mandatory documentation pass,
  0020B–0020C and 0021–0024 remain Draft, no NN task is Ready, and no later task specifications
  exist.

Repository-wide validation is deferred because this task changes one NN public constructor and no
dependency, architecture boundary, shared build configuration, or multiple-module executable
behavior. CI or the next recorded capability checkpoint owns the wider gate.

## Dependencies

- Completed NN 0004 and 0004A eager initializer/effect contracts.
- Completed NN 0007 current Embedding state/forward contract.
- Completed NN 0010 state-dictionary replacement/load contract.
- Completed NN 0019 automatic-versus-eager lifecycle distinction.
- Completed NN 0020 common `ParameterInitialization`, dispatcher, and standard PRNG selection.
- Current Model `Shape`, `DataType`, eager Tensor factories/random factories, Tensor identity, and
  ordinary embedding/Gather expression contracts.
- Proposed Text vocabulary/padding-token and Data valid-length boundaries only as reviewed future
  ownership; this task has no dependency on an unimplemented module.

## Follow-up tasks

- NN 0020B may use the completed initialized Embedding constructor in stateless standard
  construction recipes; it remains Draft and has no task file yet.
- NN 0020C directionality and NN 0021–0024 scan/valid-length/integration work remain Draft and
  separate.
- A padding-row API is not scheduled. Reopen it only when a concrete consumer supplies an owning
  contract that preserves the invariant across initialization, gradient construction, Parameter
  replacement, optimizer updates, strict load, and checkpoint restore.
- Proposed Text/Data tasks eventually own vocabulary size/fingerprint, special-token identity,
  batching/padding policy, and valid lengths after their required architecture/module decisions.

## Architecture impact

None. The change adds one eager constructor inside the existing NN layer boundary and delegates to
existing Model-backed initializer and embedding contracts. It adds no module, dependency,
operation, runtime lifecycle, backend responsibility, architecture rule, ADR, or architecture-test
change.

## Implementation prompt

Work in a fresh implementation context. Read `AGENTS.md`, `ARCHITECTURE.md`, the current
architecture plan, planning guide, roadmap, NN master plan, this task, completed NN 0004/0007/0010/
0019/0020 tasks, final `Embedding`, `ParameterInitialization`, `ParameterInitializers`, related
Linear/recurrent constructors, Module/Parameter/StateDictionary contracts, Model Shape/DataType/
Tensor factory/random/embedding APIs, existing Embedding tests, Training API, glossary, and
proposed Text/Data boundary sections. Architecture wins; stop before editing if a real conflict
appears.

Implement only the exact nine affected paths. Keep one final Embedding type and the supplied
constructor. Add exactly the eager five-argument constructor, the specified validation/effect
order, exact `L64X128MixRandom` random path, no-RNG constant path, one whole-table dispatcher call,
one permanent `weight`, generic rank-two fan interpretation, and no padding behavior. Preserve all
current forward/replacement/mode/state behavior. Do not implement lazy state, ModuleFactory,
tokenization/Data, valid lengths, directionality, scan, execution, gradient/optimizer behavior, or
architecture/build changes.

Use focused tests while iterating, then run the authoritative NN clean/test/Javadoc command once.
Do not commit, push, stage, revert, or modify concurrent unrelated work. End with the repository-
required completion summary and exact evidence. Do not mark Complete until implementation,
documentation, and all required gates pass.

## Documentation-agent handoff

After implementation and authoritative Java validation, create a distinct clean documentation
context with only this task's exact contract, affected paths, final diff, Java evidence, required
documentation profiles, and no-change review list. That context must independently inspect final
source/tests/Javadocs and finalize documentation without broadening behavior. It should rerun
focused Java only if it changes executable Java or discovers a concrete reason; otherwise reuse
the implementation evidence and run Javadoc/Markdown/surface/path/whitespace gates.

## Local decisions

- Eager construction is selected because the complete parameter Shape is explicit.
- Vocabulary and embedding sizes are both positive explicit schema/architecture values.
- The common eight-policy value and current dispatcher are reused without a new default.
- High-level random policies use exact standard `L64X128MixRandom`; constants use no RNG.
- Fan policies consume the complete `[vocabularySize, embeddingSize]` Shape under the current
  axis-zero fanOut/axis-one fanIn rule.
- Every table row is ordinary trainable state. Padding identity is not parameter-update semantics.
- Existing supplied state is retained as the advanced constructor; no compatibility alias or
  separate type is needed.
- One `weight` leaf remains the complete state and checkpoint schema.

## Known limitations

- No padding row is protected from gradient or optimizer updates.
- No vocabulary/tokenizer artifact, padding identity, or valid-length value is accepted.
- No default initializer, type, vocabulary size, embedding width, or seed is supplied.
- Construction is eager and may allocate a large host table immediately.
- Eager initializer effects are not transactional after random draws/allocation begin.
- The layer constructs expressions only; numerical lookup, gradient execution, compilation,
  lowering, and backend execution remain separately owned.
- Persistent checkpoint transport and optimizer state remain outside NN's in-memory dictionary.

## Validation evidence

The isolated planning context `/root/nn_0020a_planning` completed the mandatory architecture,
planning, documentation-profile, completed-task, final-source/Javadoc/test, Tensor/Training, and
proposed Text/Data boundary review without finding an architecture conflict. It changed only the
NN master plan and this new task file; concurrent CPU/backend planning and global-roadmap paths
remained untouched.

Targeted planning validation passed for both NN planning paths: all local Markdown links and
anchors resolve, headings are unique, fences are balanced, terminal newlines are present, and no
trailing whitespace exists. Frontier checks found exactly one Ready NN master row and one Ready
task status, linked 0020A from the master, retained 0020B–0020C and 0021–0024 as Draft, and found
no later task files. Final no-index whitespace, exact-path, shared-worktree diff, and
`git diff --check` gates passed. The no-index check returned only the expected new-file diff
status and no whitespace diagnostics. Worktree inspection attributed exactly
`docs/planning/extensions/nn/master-plan.md` and this new task to the NN planning pass, while
separately identifying the pre-existing concurrent CPU master/task and global-roadmap paths.

The isolated clean implementation context `/root/nn_0020a_implementation` added exactly the eager
five-argument constructor, retained the caller-supplied constructor and one permanent `weight`
leaf, and preserved forward, mode, replacement, and strict-load behavior. Deterministic preflight
now completes before effects; the six random policies use one fresh standard
`L64X128MixRandom`, while zero and one use the generator-free dispatcher. All policies consume the
complete `[vocabularySize, embeddingSize]` Shape and every row remains ordinary trainable state.

The focused Embedding suite passed 15 tests with no failures, errors, or skips. The sole
authoritative `./gradlew :extensions:nn:test` run passed 256 tests with no failures, errors, or
skips. Preliminary `./gradlew :extensions:nn:javadoc` passed warning-free. `javap` and reflection
confirmed exactly two public constructors, two public declared methods, one private `weight`
field, and no nested type or lifecycle surface. External source using both constructors and
`forward` compiled against the built NN and Model classes. Dependency inspection retained only
the existing NN and Model classes plus `java.base`; no build or module dependency changed.

Independent clean documentation context `/root/nn_0020a_docs` reviewed the final implementation,
tests, state contracts, Tensor construction/embedding APIs, Training and Compile boundaries, and
proposed Text/Data ownership without finding an executable, public-API, architecture, dependency,
or scope defect. It finalized the four affected production/package Javadocs, Training API,
glossary, this task, and master evidence. The exact eager validation/effect order, one successful
Tensor identifier, whole-table fan interpretation, permanent state, unchanged forward/replacement/
mode/load behavior, and deliberate absence of padding-specific rows are now synchronized.

No architecture contract, explanatory architecture document, ADR, or architecture test changed
because the implementation stays inside the existing NN-to-Model dependency and adds no boundary
rule. No Tensor, Compile, or Training Java API changed: Embedding still delegates to the existing
ordinary Model Gather expression; the compiler receives no new operation or rule; and Training
still has only its placeholder Java type while the updated Training API document explains the NN
state it may later consume. `ParameterInitialization`, `Module`, `Parameter`, and state-dictionary
contracts remain accurate because the constructor reuses the closed policy and declares the same
single complete `weight` schema. Compiler, Runtime, Prepare, Engine, backends, build/dependencies,
conformance tests, integration tests, other modules, and the global roadmap require no change
because the task adds no behavior in those owners. Concurrent CPU/backend and global-roadmap work
remained untouched.

Final warning-free `./gradlew :extensions:nn:javadoc` passed, and the rendered `Embedding`,
`ParameterInitializers`, layers-package, and initialization-package pages were inspected. `javap`
and a fresh external reflection/use program confirmed exactly two public constructors, two public
declared methods, one private `weight` field, no nested type or lifecycle surface, and successful
use of both constructors and `forward`. `jdeps` retained only NN, Model, and `java.base` classes.
Forbidden lifecycle/padding surface and import checks passed. Local Markdown links and anchors,
unique headings, balanced fences, exact nine-path NN scope, 0020A Complete/0020B–0020C and
0021–0024 Draft/no Ready/no future task files, terminal newlines, trailing whitespace, new-file
no-index whitespace, and `git diff --check` all passed. The documentation pass changed no
executable Java or test and therefore reused the focused 15-test and authoritative 256-test NN
evidence above rather than repeating it. Repository-wide validation remains deferred for the
unchanged scope reason recorded in this task.

## Implementation notes

- Added the exact eager constructor with positive size, null/type, checked count, and Java-array
  preflight in the required order.
- Routed random policies through one fresh seeded standard generator and the four-argument
  dispatcher; routed zero/one through the three-argument dispatcher without generator creation.
- Registered the exact initialized Tensor only after initializer success and retained one
  immediately discoverable/exportable `weight` parameter.
- Expanded `EmbeddingTest` across all policies and floating types, full-Shape fan semantics,
  deterministic values and seeds, state identity/storage, validation and identifier ordering,
  constant row-zero behavior, forward provenance, replacement, mode, and strict dictionary load.
- Generalized initializer fan Javadocs from Linear-only wording to the existing generic positive
  rank-two `[fanOut, fanIn]` contract.

## Completion summary

The eager initialized `Embedding` constructor, focused coverage, and affected Javadocs and
explanatory documentation are complete. The mandatory independent documentation review found no
executable, public-API, architecture, dependency, or scope defect and preserved the existing
supplied-state, forward, mode, replacement, and state-dictionary contracts. All required Java,
Javadoc, public-surface, dependency, Markdown, exact-scope, status, newline, and whitespace
validation is recorded above. No unresolved issue or follow-up is required for task 0020A; NN
0020B–0020C and 0021–0024 remain separate Draft work.

Status: Complete

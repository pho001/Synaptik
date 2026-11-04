# Task 0007: [Embedding](../../../../glossary.md#embedding) Layer

> **Editorial reading context — not a status update.** An embedding maps integer labels to
> rows of a floating table. The application supplies table state and index Tensors; the layer
> retains its weight binding and describes an axis-zero Gather, appending the table's feature
> width to the index Shape. Construction does not read index values or perform lookup.
> A token ID called padding still selects an ordinary trainable row unless a separate contract
> says otherwise. See the current [embedding convenience](../../../../reference/tensor/gather.md#embedding-convenience)
> for producer/Shape meaning and [Text token IDs](../../../../reference/text/tokenization.md#frozen-token-ids)
> for the separate vocabulary boundary; Text's naming does not change NN gradient semantics.

## Status

Complete

## Goal

Add one final public [`Embedding`](../../../../glossary.md#embedding) [module](../../../../glossary.md#term-module) that owns exactly one positive fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
floating table [`Parameter`](../../../../glossary.md#term-parameter) and constructs indexed [lookup](../../../../glossary.md#term-lookup) [metadata](../../../../glossary.md#term-metadata) only through the existing
`Tensor.embedding` convenience. The layer retains caller-supplied state, exposes its stable
parameter handle, and remains mode-insensitive without introducing an embedding [operation](../../../../glossary.md#operation),
initialization policy, [padding](../../../../glossary.md#padding) behavior, or execution claim.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.Embedding` extending the existing
  `io.github.pho001.synaptik.nn.module.Module`.
- Add exactly this public constructor and these methods:

  ```java
  public Embedding(Tensor weight)

  public Parameter weight()

  public Tensor forward(Tensor indices)
  ```

- Declare exactly one [parameter](../../../../glossary.md#term-parameter) under local name [`weight`](../../../../glossary.md#term-weight). It is the lookup table with exact
  Shape orientation <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [Shape glossary](../../../../glossary.md#shape))</span>`[vocabularySize, embeddingSize]`; axis zero contains rows selected by an
  index value, and axis one is appended to every result as the embedding Dimension.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Dimension glossary](../../../../glossary.md#dimension))</span>
- Require the supplied weight to be non-null, floating, gradient-eligible, rank two, fully static,
  and strictly positive on both axes. Retain its exact Tensor reference through the existing<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [Tensor glossary](../../../../glossary.md#tensor))</span>
  `Parameter` declaration. Do not require a label, host storage, provenance absence, dense or
  resolved layout, or a particular Tensor identity.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Validate supplied construction in exact order: null weight, floating data type, gradient
  eligibility, rank two, fully static Shape, positive vocabulary extent at axis zero, then
  positive embedding extent at axis one. Complete validation before declaring the parameter.
  Constructor validation creates no Tensor, expression, storage, random draw, or Tensor identity
  and never mutates the supplied value.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation))</span>
- `weight()` returns the exact stable wrapper declared by the layer. Add no setter or layer-local
  replacement facade; the existing public schema-compatible <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [schema glossary](../../../../glossary.md#term-schema))</span>`Parameter.replace(Tensor)` remains
  the only public update capability.
- `forward(indices)` first rejects null with message `indices`, reads the current weight binding
  exactly once, and returns exactly <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`currentWeight.embedding(indices)`. Do not reproduce rank,
  index-type, Shape, operation, descriptor, producer, provenance, bounds, or identity logic in NN.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Forward inherits [Model](../../../../glossary.md#term-model)'s current contract: indices must use exact [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) and may
  have any rank, including scalar; result Shape is the complete indices Shape followed by the
  exact current weight axis-one Dimension; result type and gradient eligibility come only from
  the current weight; and the sole occurrence is ordinary axis-zero <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>[`GATHER`](../../../../glossary.md#gather) with ordered
  `[weight, indices]` [provenance](../../../../glossary.md#provenance).
- Forward construction reads no index value. Negative or out-of-range values remain invalid for
  eventual ordinary Gather execution; the layer adds no wrapping, clamping, default row, or
  padding interpretation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- [`Embedding`](../../../../glossary.md#embedding) is mode-insensitive. Its forward method accepts no `ForwardContext`, does not inspect
  `mode()`, and constructs the same Model expression contract in training and [evaluation](../../../../glossary.md#term-evaluation) modes.
- A compatible `Parameter.replace` affects the next forward call. Earlier Tensor references and
  already constructed expressions retain the prior exact table. The replacement schema preserves
  exact declaration-time data type and structural Shape plus gradient eligibility; replacement
  remains individual and not thread-safe.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>
- Add focused tests for the exact public surface, supplied-state validation and side effects,
  one-parameter ownership, direct Model delegation and provenance, scalar and shaped indices,
  mode irrelevance, inherited Model failures, and replacement snapshots.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add complete type, constructor, method, and package Javadocs. After executable work and focused
  validation, use a separate documentation-focused clean context to finalize those Javadocs,
  glossary impact, planning evidence, no-change conclusions, and documentation validation in the
  same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- An initialized constructor, default parameter Tensor, default data type, default or retained
  random source, layer-owned distribution, caller-selectable initializer object, initializer
  registry, or reuse of Linear-specific Glorot/Kaiming fan policy as an embedding default.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [data type glossary](../../../../glossary.md#data-type--datatype); [distribution glossary](../../../../glossary.md#term-distribution); [registry glossary](../../../../glossary.md#term-registry); [Linear glossary](../../../../glossary.md#term-linear-meanings); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming))</span>
- A padding index, guaranteed-zero padding row, gradient mask, post-update projection, sparse
  gradient, maximum-norm option, frequency scaling, negative-index wrapping, clamping, or default
  row. The current Model convenience represents none of these, and this task must not simulate
  them through eager storage mutation or an NN-owned update hook.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient); [mask glossary](../../../../glossary.md#term-boolean-mask); [norm glossary](../../../../glossary.md#term-norm); [Model glossary](../../../../glossary.md#term-model); [eager glossary](../../../../glossary.md#term-eager-evaluation); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Vocabulary or embedding-size accessors, parameter-Shape introspection, lazy or dynamic state,
  zero-sized table axes, multiple tables, sharding, quantization, integral/BOOL parameters, frozen
  parameters, mixed parameter types, or a configurable table axis.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [axes glossary](../../../../glossary.md#term-axis); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- A generic `Module.forward`, layer interface, abstract layer base, [unary](../../../../glossary.md#term-unary) [module](../../../../glossary.md#term-module) contract,
  `Sequential`, block, or stateless functional [facade](../../../../glossary.md#term-facade).
- Any `EmbeddingAttrs`, [`EMBEDDING`](../../../../glossary.md#embedding) operation kind, semantic signature, Tensor overload, Gather
  change, bounds implementation, eager lookup, host-storage read, result storage, numerical
  output assertion, or input mutation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [signature glossary](../../../../glossary.md#term-signature); [Gather glossary](../../../../glossary.md#gather); [lookup glossary](../../../../glossary.md#term-lookup); [host glossary](../../../../glossary.md#term-host))</span>
- Gradient rules, repeated-index scatter-add construction, optimizer behavior, parameter groups,
  checkpoints, state dictionaries, serialization, compiler capture or optimization, backend
  lowering, runtime/prepare/Engine behavior, execution, or end-to-end support.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoints glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Gradle, dependency, architecture-contract, ADR, explanatory-architecture, architecture-test,
  Model, Training production, CPU, global-roadmap, or unrelated refactoring changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Module boundaries](../../../../architecture/module-boundaries.md).
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Completed NN task 0006](0006-layer-normalization-layer.md) and its foundation dependencies.
- [Completed Model task 0019A1: Embedding convenience](../../../modules/model/tasks/0019a1-embedding-convenience.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Embedding glossary](../../../../glossary.md#embedding))</span>.
- [Tensor API embedding convenience](../../../../api/tensor-api.md#embedding-convenience)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>.

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- NN owns the stable table parameter and typed forward composition. Model remains the sole owner
  of generic axis-zero Gather meaning, index representation, Shape/result metadata, operation
  attributes, provenance, and construction-time validation. Training remains downstream and owns
  optimizer algorithms and update orchestration.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [axis glossary](../../../../glossary.md#term-axis); [Gather glossary](../../../../glossary.md#gather); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) remain [immutable](../../../../glossary.md#term-immutable). [`Embedding`](../../../../glossary.md#embedding) may retain one mutable
  NN parameter binding but must not mutate a Tensor, its storage, or its gradient lifecycle.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [binding glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Construction and forward create only module state around a supplied Tensor or storage-free Model
  expression metadata. They must not imply compiler capture, gradient availability, bounds
  enforcement, backend support, numerical evaluation, storage residency, or execution.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [evaluation glossary](../../../../glossary.md#term-evaluation); [residency glossary](../../../../glossary.md#residency))</span>
- Train/eval mode remains NN composition metadata. This mode-insensitive layer must not add an
  unused context argument or execution-state dependency.<span data-glossary-navigation="true"> ([eval mode glossary](../../../../glossary.md#inference-and-evaluation); [argument glossary](../../../../glossary.md#term-argument); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Preserve unrelated worktree changes exactly, including the dirty CPU master/task and global
  roadmap. This task does not authorize edits outside its exact six paths.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing [`Module`](../../../../glossary.md#term-module) and [`Parameter`](../../../../glossary.md#term-parameter) ownership,
  discovery, and replacement contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `io.github.pho001.synaptik.model.tensor` — existing `Tensor.embedding` construction and current
  Tensor descriptor/provenance contracts.<span data-glossary-navigation="true"> ([Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing immutable Shape, static
  Dimension, and floating/index data-type contracts.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension))</span>
- `io.github.pho001.synaptik.model.operation.index` — test inspection of the existing ordinary
  [`GATHER`](../../../../glossary.md#gather) and `IndexAxisAttrs(0)` [occurrence](../../../../glossary.md#term-occurrence) only; production must not construct them directly.

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — add one concrete stateful [lookup](../../../../glossary.md#term-lookup) layer beside [`Linear`](../../../../glossary.md#term-linear-meanings)
  and `LayerNorm`; no package or layer hierarchy is introduced.

Type placement:

- `io.github.pho001.synaptik.nn.layers.Embedding` — public stateful table layer; it belongs with
  concrete layers rather than generic module ownership, stateless initialization policy, or Model
  operation semantics.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- `io.github.pho001.synaptik.nn.layers.EmbeddingTest` — exact public surface, supplied state,
  validation, forward/provenance, mode, inherited failures, and replacement coverage.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Embedding.java`.

Expected test file:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`.

Documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend the existing NN-module and Embedding entries with the current
  stateful owner while preserving the existing Model/Gather and execution boundaries.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [Embedding glossary](../../../../glossary.md#embedding); [owner glossary](../../../../glossary.md#term-owner); [Model glossary](../../../../glossary.md#term-model); [Gather glossary](../../../../glossary.md#gather))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, module/dependency/training-graph explanations, and architecture
  tests — the task realizes the existing NN layer responsibility without changing a dependency,
  owner, or lifecycle boundary.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [graph glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor and Compile APIs, Model capability/master plans, and Model task 0019A1 — the layer
  delegates to the unchanged current Model convenience and adds no Model or compiler capability.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Training API — recursive discovery and individual parameter replacement remain unchanged; no
  optimizer, gradient publication, padding-update, or training workflow is added.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [padding glossary](../../../../glossary.md#padding))</span>
- Public API status, user training documentation, conformance/integration suites, Gradle,
  compiler/runtime/prepare/Engine, backends, other modules, CPU planning, and the global roadmap —
  none owns this bounded Model-composition wrapper.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- two production Java files;
- one [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test file; and
- the three documentation/[planning](../../../../glossary.md#term-planning) files listed above.

That is an exact six-path maximum. If another type, test [owner](../../../../glossary.md#term-owner), [API](../../../../glossary.md#term-application-programming-interface-api) document, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency),
[architecture test](../../../../glossary.md#term-architecture-test), [Model](../../../../glossary.md#term-model) helper, initializer API, [padding](../../../../glossary.md#padding)/update mechanism, or executable
integration path is required, stop and propose a focused follow-up rather than expanding this
task.

## Acceptance criteria

- Final public [`Embedding`](../../../../glossary.md#embedding) extends [`Module`](../../../../glossary.md#term-module) and declares exactly the one constructor and two
  methods listed in Scope, with no additional public or protected member.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Construction retains the exact caller-supplied Tensor and validates null, floating type,
  gradient eligibility, rank, fully static Shape, positive vocabulary extent, and positive
  embedding extent in the specified order before parameter declaration and without Tensor-ID,
  allocation, storage, value, or expression side effects.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent))</span>
- Direct and recursive discovery expose exactly one [parameter](../../../../glossary.md#term-parameter) named [`weight`](../../../../glossary.md#term-weight); `weight()` returns
  that exact stable wrapper, and the wrapper's value is the exact supplied table.
- `forward` null-checks indices, reads the current weight once, and delegates exactly to that
  Tensor's `embedding(indices)` method. It does not construct an operation, Shape, descriptor,
  producer, provenance, bound check, or intermediate Tensor itself.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Tests cover both exact index types, scalar and multi-axis indices, all three floating table
  types, result Shape/type/eligibility/layout metadata, one ordinary <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout); [metadata glossary](../../../../glossary.md#term-metadata))</span>[`GATHER`](../../../../glossary.md#gather) with
  `IndexAxisAttrs(0)`, exact `[weight, indices]` inputs, output index zero, and fresh result
  identity without evaluating values.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tests cover null indices and inherited Model index-type and identifier-exhaustion failures
  without duplicating the exhaustive Model 0019A1 matrix or changing its messages/order.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Training and evaluation mode produce the same composition contract; no context overload,
  generic Module forward method, hidden state, or mode-dependent operation appears.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- A compatible replacement appears only in later forward provenance; an earlier expression keeps
  the old exact table, the wrapper remains stable, and incompatible type/Shape/eligibility
  replacement stays rejected by the existing `Parameter` [schema](../../../../glossary.md#term-schema).
- No initialized constructor, initializer call, padding behavior, new semantic operation,
  gradient/update rule, optimizer, checkpoint, serialization, compiler, runtime, backend,
  execution, build, dependency, architecture, CPU, roadmap, or unrelated change enters the diff.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Public and package Javadocs cover table orientation, state schema and name, exact retained
  references, absence of a layer-owned initializer/padding policy, Model delegation, result
  metadata, mode irrelevance, replacement snapshots, nullability, validation and side effects,
  inherited failures, thread safety, and no-execution boundaries with complete tags.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability); [validation glossary](../../../../glossary.md#term-validation))</span>
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, generated Javadoc, local links/anchors, exact scope,
  reasoned no-change conclusions, and whitespace before the task becomes Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs the focused test while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite:

```bash
./gradlew :extensions:nn:test
```

The focused test owns the exact declared [API](../../../../glossary.md#term-application-programming-interface-api), state [schema](../../../../glossary.md#term-schema)/name/[identity](../../../../glossary.md#term-identity-meanings), validation order and
side effects, direct [Model](../../../../glossary.md#term-model) [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) delegation, mode irrelevance, inherited failures,
and replacement snapshots. Existing Model 0019A1 tests remain the authoritative exhaustive
[embedding](../../../../glossary.md#embedding) semantic/validation [matrix](../../../../glossary.md#term-matrix-meanings); do not repeat them or claim numerical execution.

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates changed Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `Embedding` and package pages, exact six-path [scope](../../../../glossary.md#term-scope), package placement,
public surface, forbidden imports, 0001–0007 Complete, 0008–0011 Draft, no [Ready](../../../../glossary.md#term-ready) NN task, and
absence of detailed NN 0008 or later task files. It preserves unrelated [CPU](../../../../glossary.md#term-cpu)/global-roadmap
changes and may reuse final NN test evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one existing model-only [module](../../../../glossary.md#term-module), no build edge, [dependency](../../../../glossary.md#term-dependency) rule, architecture
boundary, shared [Gradle](../../../../glossary.md#term-gradle) contract, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution path.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0006 are Complete.
- [Model](../../../../glossary.md#term-model) task 0019A1 and the exact `Tensor.embedding(Tensor)` convenience are Complete.
- Existing [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), Shape, data-type, producer, and provenance contracts are
  Complete and provide every state and forward fact used here.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and the model-only NN [dependency direction](../../../../glossary.md#java-platform-and-build-tools) remain unchanged.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force; the
  six task paths do not overlap the active dirty CPU planning work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Follow-up tasks

- NN 0008: BatchNorm remains Draft only; it owns the first explicit parameter/buffer and
  train/eval state-transition layer.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0009: [Dropout](../../../../glossary.md#dropout--inverted-dropout) remains Draft only; it owns explicit [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) threading and evaluation
  bypass without hidden RNG.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- NN 0010: state dictionary and [checkpoint](../../../../glossary.md#term-checkpoint) semantics remain Draft only.
- NN 0011: a narrow [unary](../../../../glossary.md#term-unary) forward contract and `Sequential` remain Draft only and require their
  concrete consumer.
- A future embedding-specific initializer or padding/update policy requires a concrete consumer
  and an independently specified contract. It must not retrofit hidden storage mutation or infer
  padding semantics from ordinary Gather.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [padding glossary](../../../../glossary.md#padding); [Gather glossary](../../../../glossary.md#gather))</span>
- `extensions/training` later consumes the stable discovered [`Parameter`](../../../../glossary.md#term-parameter); this task adds no
  optimizer or session behavior. Numerical end-to-end validation waits for the required compiled
  execution and backend coverage.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers composed from
[Model](../../../../glossary.md#term-model) semantics. If implementation requires a new [dependency](../../../../glossary.md#term-dependency), architecture rule, Model change,
initializer policy, [padding](../../../../glossary.md#padding)/update mechanism, generic layer abstraction, or executable behavior,
stop and report the conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in the Synaptik repository without commit or push. Do not use any GSD skill or workflow.
Read AGENTS.md, ARCHITECTURE.md, the current architecture index, planning and documentation rules,
the NN master plan and completed tasks 0001–0006, Model task 0019A1, current Module/Parameter/
existing-layer contracts, Tensor.embedding and its focused tests, and
docs/planning/extensions/nn/tasks/0007-embedding-layer.md in full.

Implement task 0007 exactly inside its six authorized paths. Preserve all unrelated worktree
changes exactly, especially CPU planning and the global roadmap. Stop and report any architecture
uncertainty, scope overflow, or need for another public type, dependency, initializer, padding
mechanism, or file.

Run the focused test and one final NN suite after executable Java stabilizes. Then hand the actual
diff, exact Java evidence, and task contract to a separate documentation-focused clean context in
the same overall change. That pass finalizes package/type Javadocs, glossary and planning
evidence, generated Javadoc, Markdown, scope, no-change conclusions, and whitespace without
repeating successful Java tests unless executable behavior changes. Mark Complete only after
every criterion passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evidence and whether Java changed afterward, the exact public surface, table
[validation](../../../../glossary.md#term-validation) order, [parameter](../../../../glossary.md#term-parameter) and replacement behavior, forward [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) and train/eval
evidence, [padding](../../../../glossary.md#padding)/initializer exclusions, and the exact six authorized paths.

That agent independently reads the repository instructions, [architecture contract](../../../../glossary.md#architecture-contract) and index,
documentation rules plus General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning) profiles, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, this task, final source
and tests, generated Javadoc, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Training APIs, glossary, NN master/task history, [Model](../../../../glossary.md#term-model)
0019A1, `Tensor.embedding`, [Gather](../../../../glossary.md#gather)/index contracts, [`Module`](../../../../glossary.md#term-module), `Parameter`, and current layers. It
finalizes only package/type Javadoc, glossary, this task, and the NN master plan. It records
reasoned no-change conclusions for architecture/ADRs/tests, Tensor/Compile/Training/public APIs,
Model plans/contracts, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), execution layers, [backends](../../../../glossary.md#backend), other modules,
[CPU](../../../../glossary.md#term-cpu) work, and the global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The public surface is deliberately only supplied-state construction, `weight()`, and
  `forward(indices)`. A caller can construct a table explicitly with Model factories or the
  existing generic NN normal/uniform initializers, so no current consumer justifies baking one
  distribution, default source, or data type into this layer.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [distribution glossary](../../../../glossary.md#term-distribution); [data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Existing [Glorot](../../../../glossary.md#term-glorot-uniform) and [Kaiming](../../../../glossary.md#term-kaiming) methods are explicitly defined for [Linear](../../../../glossary.md#term-linear-meanings) [weights](../../../../glossary.md#term-weight) in
  `[outFeatures, inFeatures]` orientation. Treating vocabulary size and embedding size as those
  fans would silently invent an embedding initialization policy, so this task does not reuse them.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding))</span>
- The table [parameter](../../../../glossary.md#term-parameter) is named `weight`, matching the receiver terminology of
  `Tensor.embedding` and the stable layer parameter convention. Its exact [Shape](../../../../glossary.md#shape) is
  `[vocabularySize, embeddingSize]` and both [axes](../../../../glossary.md#term-axis) are positive and fully static.
- Forward is mode-insensitive and delegates directly to [Model](../../../../glossary.md#term-model) without an unused
  `ForwardContext`. It performs only an indices null check before reading the current table once.
- Padding-row behavior is deferred. The current Model operation is ordinary Gather and has no
  padding attribute. Initial zeroing alone would not preserve a row across public parameter
  replacement or future optimizer updates, while enforcing it would require hidden mutation,
  gradient/update policy, or a new semantic contract outside this task.<span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding); [operation glossary](../../../../glossary.md#operation); [Gather glossary](../../../../glossary.md#gather); [optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient))</span>

## Known limitations

- Only caller-supplied positive fully static rank-two floating state is supported; there is no
  initialized, lazy, dynamic, zero-axis, sharded, quantized, or sparse table form.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [axis glossary](../../../../glossary.md#term-axis))</span>
- There is no padding index or invariant zero row. Every eventual index must satisfy the ordinary
  Gather bound <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [invariant glossary](../../../../glossary.md#term-invariant); [Gather glossary](../../../../glossary.md#gather))</span>`0 <= index < vocabularySize`.
- Parameter replacement is individual and not thread-safe. Callers must coordinate replacement
  with forward construction when one stable table snapshot matters.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Forward constructs expression metadata and proves no compiler capture, gradient rule, repeated-
  index accumulation, bounds enforcement, backend coverage, numerical value, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- There is no checkpoint, serialization, device/dtype conversion, optimizer, or training-session
  convenience.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [device glossary](../../../../glossary.md#term-device); [dtype glossary](../../../../glossary.md#data-type--datatype); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/nn_0007_planning` read the repository instructions, architecture and NN/
  training boundaries, planning/documentation rules, NN master and tasks 0001–0006, Model master
  and task 0019A1, current NN/Model source and tests, Tensor/Compile/Training APIs, glossary,
  dependency enforcement, and Java 26 Gradle configuration before fixing this API and scope.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Planning inspection confirmed that [Model](../../../../glossary.md#term-model) exposes no `EmbeddingAttrs` or [embedding](../../../../glossary.md#embedding) [operation](../../../../glossary.md#operation):
  `Tensor.embedding` validates a rank-two floating receiver and exact INT32/INT64 indices, then
  delegates to one ordinary <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>[`GATHER`](../../../../glossary.md#gather) with `IndexAxisAttrs(0)`. It offers no padding option and
  reads no values. Existing parameter replacement preserves only exact type, structural Shape,
  and gradient eligibility.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [parameter glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Targeted validation passed for both planning paths: every repository-local Markdown target and
  the Tensor API <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>`#embedding-convenience` anchor resolved; fenced blocks were balanced; both files
  had terminal newlines and no trailing whitespace. Scope inspection found exactly the NN master
  plan and this new task file, one Ready master row paired with this Ready task, four concise Draft
  rows for 0008–0011, and no 0008–0011 task files. Whole-worktree <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>`git diff --check` and the
  untracked-file `git diff --no-index --check` passed; unrelated CPU and global-roadmap changes
  remained outside the NN planning diff.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Clean implementation context `/root/nn_0007_implementation` read the required architecture,
  planning, documentation-profile, completed NN 0001–0006, Model 0019A1, final Tensor/Gather,
  module/parameter/layer, API/glossary, test, dependency, and Java 26 Gradle contracts before
  editing. It found no architecture, final-Model-API, package-placement, dependency, or six-path
  scope conflict.
- The final focused command after executable Java stabilized,
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest`, passed
  with one suite and 8 tests, zero failures, errors, or skips. Coverage includes exact final public
  surface, sole state identity/path, every constructor validation stage and no-ID effects, all
  three floating table types, both exact index types, scalar and multi-axis indices, exact
  ordinary-Gather producer/provenance and result metadata, freshness, mode irrelevance, inherited
  index/null/identifier failures, and replacement snapshots.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [axis glossary](../../../../glossary.md#term-axis); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- The sole authoritative final `./gradlew :extensions:nn:test` passed after source and tests
  stabilized (`BUILD SUCCESSFUL`, five actionable tasks: one executed and four up-to-date). XML
  reports contain 12 suites and 61 tests with zero failures, errors, or skips. No executable Java
  or test changed afterward in this implementation context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed after the final executable edit (`BUILD
  SUCCESSFUL`, three actionable tasks: one executed and two up-to-date). Generated
  `Embedding.html` and `layers/package-summary.html` exist and render the planned public members,
  orientation, delegation, replacement, padding exclusion, and no-execution boundaries. This is
  implementation-draft evidence only; the separate documentation context still owns final
  Javadoc editing, generation, and generated-page inspection.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `javap -public` and an independent Java 26 reflection program confirmed public final
  `Embedding extends Module` with exactly `Embedding(Tensor)`, `weight()`, and
  `forward(Tensor)`, and no other declared public or protected member. Manual source inspection
  confirmed null-check-first forward order, one current-binding read, and exact direct<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
  `return currentWeight.embedding(indices)` delegation.
- Production-import and [dependency](../../../../glossary.md#term-dependency) scans found only Model, existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module), and [JDK](../../../../glossary.md#term-java-development-kit-jdk) imports;
  `extensions/nn` retains its sole Model project dependency. The implementation adds no
  initializer call, operation/attributes type, padding/update mechanism, generic layer contract,
  execution dependency, or other module edge.
- Preliminary documentation-impact review found no task-owned change necessary for
  `ARCHITECTURE.md`, ADR 0007, focused architecture documents/tests, Tensor/Compile/Training APIs,
  Model capabilities/master/task 0019A1, conformance/integration, Gradle, compiler/runtime/
  prepare/Engine, backends, other modules, CPU planning, or the global roadmap. The layer realizes
  the existing NN owner and delegates to unchanged Model metadata without adding a semantic kind,
  gradient/update, bounds, backend, or execution contract. These preliminary conclusions were
  handed to the mandatory documentation context for independent verification and glossary review.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner))</span>
- Independent clean documentation context `/root/nn_0007_docs` read the repository instructions,
  architecture contract and focused architecture explanations, planning/documentation rules and
  profiles, ADR 0007, NN master/task history, Model master/capabilities/task 0019A1, final
  Tensor/Gather/Shape/DataType and NN module/parameter/layer sources and tests, API references,
  glossary, dependency rules, and Java 26 Gradle configuration. It found no executable Java or
  test defect and changed only task-owned Javadoc and prose; executable source and tests remained
  unchanged, so it reused the stable focused and final NN test evidence above.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- Final `./gradlew :extensions:nn:javadoc` passed (`BUILD SUCCESSFUL`, three actionable tasks: two
  executed and one up-to-date). Inspection of generated `Embedding.html` and
  `layers/package-summary.html` confirmed the exact public members and rendered ownership,
  validation/order, result/delegation, replacement-snapshot, padding/initializer exclusion,
  mode-insensitivity, and declarative no-execution contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Final `javap -public` and an independent Java 26 reflection check confirmed final public
  `Embedding extends Module`, exactly `Embedding(Tensor)`, `weight()`, and `forward(Tensor)`, and
  no other declared public or protected API. Production import and Gradle inspection confirmed
  only Model, existing NN module, and JDK imports and the unchanged sole Model project dependency.
- Targeted Markdown validation resolved all local paths and anchors and found balanced fences in
  the glossary, NN master plan, and task. Scope/status inspection found exactly the six task paths
  among preserved unrelated dirty CPU/global-roadmap work, NN 0001–0007 Complete, 0008–0011
  Draft, no Ready NN task, and no detailed 0008–0011 task files. Final newline, trailing-
  whitespace, tracked/untracked diff, and whole-worktree `git diff --check` gates passed.
- Final no-change review confirmed no update is warranted for `ARCHITECTURE.md`, the current
  architecture explanations, ADR 0007, architecture tests, Tensor/Compile/Training APIs, Model
  capabilities/master/task 0019A1, conformance/integration tests, Gradle, compiler/runtime/
  prepare/Engine, backends, other modules, training graph, CPU planning, the global roadmap, or
  later NN tasks. The change uses the established NN state owner and unchanged Model convenience;
  it adds no boundary, kind, numerical, gradient/update, compiler, backend, or execution contract.<span data-glossary-navigation="true"> ([training graph glossary](../../../../glossary.md#training-graph))</span>

## Implementation notes

- Added final public `io.github.pho001.synaptik.nn.layers.Embedding` with exactly one supplied-state
  constructor, stable [`weight`](../../../../glossary.md#term-weight) parameter accessor, and mode-insensitive typed forward method.
  Construction completes the specified null, floating, gradient, rank, static-Shape, vocabulary,
  and embedding-extent validation before declaring the sole parameter.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [embedding glossary](../../../../glossary.md#embedding); [extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation))</span>
- Forward rejects null indices before reading the current binding once and returning that exact
  table's existing <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`embedding(indices)` expression. Focused tests lock the ordinary
  [`GATHER`](../../../../glossary.md#gather)/`IndexAxisAttrs(0)` [occurrence](../../../../glossary.md#term-occurrence), exact `[weight, indices]` inputs, output index zero,
  result metadata, fresh identity, inherited failures, and old/new replacement snapshots without
  reading or asserting numerical values.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finalized the required type/method and layers-package Javadocs plus the existing Embedding and
  NN-module glossary entries. The glossary includes a small <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module))</span>`[10, 4]` table and `[2, 3]` indices
  metadata example and preserves the Model/Gather versus NN state-ownership boundary. No
  initialized constructor, padding policy, hidden mutation, buffer, RNG, optimizer, checkpoint,
  compiler/backend/runtime behavior, build change, or unrelated refactor was added.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [padding glossary](../../../../glossary.md#padding); [buffer glossary](../../../../glossary.md#term-buffer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Completion summary

- Completed changes: Implemented and documented the exact supplied-state Embedding layer,
  constructor schema/order, sole stable parameter, direct Model delegation, mode-insensitivity,
  compatible-replacement snapshots, and current glossary definition/example.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [schema glossary](../../../../glossary.md#term-schema); [parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model))</span>
- Files changed or created: `Embedding.java`, `layers/package-info.java`, `EmbeddingTest.java`,
  `docs/glossary.md`, the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan, and this task record: exactly six task paths.
- Tests and validation: Reused the stable final focused one-suite/8-test and authoritative full NN
  12-suite/61-test results, both with zero failures, errors, or skips. Final NN Javadoc,
  generated-page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, reflection, manual forwarding, import/dependency,
  Markdown, exact-scope/status, final-newline, trailing-whitespace, and diff checks passed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: Complete in independent context `/root/nn_0007_docs`; it found no
  executable defect and made no executable Java or test change.
- Documentation impact: Finalized affected public/package Javadocs and both relevant glossary
  entries. The reasoned no-change set is recorded in Validation evidence.
- Unresolved issues: None for task 0007.
- Required follow-up: None. NN 0008–0011 remain Draft and have no detailed task specifications.

Status: Complete

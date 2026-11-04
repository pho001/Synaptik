# Task 0006: [Layer Normalization](../../../../glossary.md#layer-normalization) Layer

> **Editorial reading context — not a status update.** LayerNorm owns scale/bias bindings
> and an exact typed epsilon; the application supplies the input. Forward delegates a
> normalization expression to Model rather than reading its values. Scale and bias span the
> selected trailing slice and are reused over leading axes. The current
> [layer-normalization explanation](../../../../reference/tensor/normalization.md#layer-normalization-expressions)
> teaches the formula and separates it from backend execution.
>
> **Mathematical reading example.** For a finite real slice $(x_1,x_2)=(1,3)$, its population
> mean is $\mu=(x_1+x_2)/2=2$ and variance is
> $v=((x_1-\mu)^2+(x_2-\mu)^2)/2=1$.
> With real $\epsilon>0$, scale one and bias zero, the normalized pair is
> $(-1,1)/\sqrt{1+\epsilon}$. This is a semantic illustration, not a claim about rounded
> execution, special values or the historical tests.

## Status

Complete

## Goal

Add one final public `LayerNorm` [module](../../../../glossary.md#term-module) that owns mandatory exact-[Shape](../../../../glossary.md#shape) `scale` and [`bias`](../../../../glossary.md#term-bias)
[parameters](../../../../glossary.md#term-parameter) plus one exact typed [epsilon](../../../../glossary.md#epsilon), and constructs its output only through the existing
affine `Tensor.layerNorm` expression. Callers can supply both parameter [Tensors](../../../../glossary.md#tensor) or request exact
ones/zeros initialization for one positive fully static [normalized Shape](../../../../glossary.md#normalized-shape) and floating [data type](../../../../glossary.md#data-type--datatype).

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.LayerNorm` extending the existing
  `io.github.pho001.synaptik.nn.module.Module`.
- Add exactly these public constructors and methods:

  ```java
  public LayerNorm(Tensor scale, Tensor bias, ScalarValue epsilon)

  public LayerNorm(
          Shape normalizedShape,
          DataType dataType,
          ScalarValue epsilon)

  public Parameter scale()

  public Parameter bias()

  public Tensor forward(Tensor input)
  ```

- Both parameters are mandatory. The Model affine layer-normalization form has exact ordered
  inputs <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [normalization glossary](../../../../glossary.md#term-normalization))</span>`[input, scale, bias]` and accepts neither scale-only nor bias-only state, so null never
  means absence and this task adds no no-affine module variant.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Declare exact local names `scale` then [`bias`](../../../../glossary.md#term-bias). Direct and recursive discovery must retain that
  order and return the same stable wrappers as `scale()` and `bias()`.
- For caller-supplied state, infer the [normalized Shape](../../../../glossary.md#normalized-shape) as the exact [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape) reference of
  `scale`. Require scale to be floating, gradient-eligible, positive-rank, fully static, and
  positive on every axis. Require bias to be floating, gradient-eligible, have the exact same data
  type, and have a structurally equal Shape. Retain both exact Tensor references; do not require
  labels, storage, provenance absence, dense layout, or particular Tensor identities.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [data type glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout))</span>
- Validate supplied construction in this exact order: null scale, null bias, null epsilon; scale
  floating type, scale gradient eligibility, positive scale rank, fully static scale Shape, each
  scale extent positive in increasing-axis order; bias floating type, bias gradient eligibility,
  exact scale/bias data-type equality, and structural scale/bias Shape equality; Model intrinsic
  normalized-Shape/epsilon validity; then exact epsilon/parameter data-type equality. Complete all
  validation before declaring either <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [validation glossary](../../../../glossary.md#term-validation))</span>[`Parameter`](../../../../glossary.md#term-parameter).
- Reuse `new AffineLayerNormAttrs(normalizedShape, epsilon)` only as the Model-owned intrinsic
  check for positive normalized rank and finite strictly-positive floating epsilon. Do not copy
  its numeric validation, select an epsilon default, convert epsilon, or create an operation at
  construction time. After that intrinsic check, require <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [operation glossary](../../../../glossary.md#operation))</span>`epsilon.dataType()` to equal the exact
  shared parameter data type.
- The initialized constructor null-checks `normalizedShape`, `dataType`, then [`epsilon`](../../../../glossary.md#epsilon); requires
  positive rank, fully static Shape, positive extents in increasing-axis order, and floating data
  type; applies the same Model intrinsic epsilon check; then requires epsilon's exact data type to
  equal `dataType`. All these caller-controlled checks happen before Tensor creation or identifier
  allocation.
- Initialize scale first with exactly
  `ParameterInitializers.ones(normalizedShape, dataType)`, then bias with exactly
  `ParameterInitializers.zeros(normalizedShape, dataType)`. There is no random source, draw,
  default data type, default epsilon, configurable affine policy, or initializer object.
- Store the exact normalized Shape and epsilon references selected by construction for subsequent
  forward calls. Do not expose `normalizedShape()` or `epsilon()` in this task: the caller already
  supplies them, forward needs them internally, and no current state-dictionary, serialization,
  generic layer, or introspection consumer justifies a second public configuration surface.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>
- `scale()` and `bias()` return the exact stable declared `Parameter` wrappers. Add no setter or
  layer-specific replacement facade; existing <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>`Parameter.replace` remains the only public
  schema-compatible binding replacement capability.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [binding glossary](../../../../glossary.md#term-binding))</span>
- `forward(input)` first rejects null input with message `input`, then reads current scale and bias
  bindings exactly once in declaration order and delegates exactly to
  `input.layerNorm(normalizedShape, currentScale, currentBias, epsilon)`. It must not reproduce
  normalization, promotion, Shape, descriptor, operation, producer, or provenance logic.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [producer glossary](../../../../glossary.md#term-producer))</span>
- Forward inherits the Model contract for trailing-Shape compatibility, ordered floating
  promotion, exact result-typed epsilon, population variance, accumulation and special-value
  semantics, result metadata, validation order, allocation, and exact affine provenance. A higher-
  precision input may therefore fail the inherited exact-epsilon-type check; the layer inserts no
  cast and defines no alternate mixed-precision policy.<span data-glossary-navigation="true"> ([population variance glossary](../../../../glossary.md#population-variance); [metadata glossary](../../../../glossary.md#term-metadata); [precision glossary](../../../../glossary.md#term-precision))</span>
- `LayerNorm` is mode-insensitive. Its forward method accepts no `ForwardContext`, does not inspect
  `mode()`, and produces the same composition contract in training and [evaluation](../../../../glossary.md#term-evaluation) modes.
- A compatible replacement affects the next forward call. Earlier Tensor references and already
  constructed expressions retain the prior exact scale and bias inputs. Replacement remains
  individual and not thread-safe; this task adds no multi-parameter snapshot or transaction.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Add focused tests for exact public surface, supplied and initialized state schema/order/identity,
  typed epsilon and validation order, exact initializer delegation and values, forward metadata
  and provenance, mode irrelevance, inherited Model failures, and replacement snapshots.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add complete public and package Javadocs, then use a separate clean-context documentation pass
  to finalize Javadoc, glossary impact, planning evidence, no-change conclusions, and
  documentation validation in the same overall change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- A no-affine module, optional scale or bias, scale-only or bias-only form, nullable state,
  broadcast affine state, frozen affine state, dynamic/lazy parameter Shape, or zero-sized
  normalized parameter axis.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [bias glossary](../../../../glossary.md#term-bias); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [parameter glossary](../../../../glossary.md#term-parameter); [axis glossary](../../../../glossary.md#term-axis))</span>
- A default [epsilon](../../../../glossary.md#epsilon), raw `double` epsilon convenience, epsilon conversion, epsilon Tensor,
  configurable correction or accumulation, caller-selectable initializer, random initialization,
  retained/default RNG, or <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate).
- A normalized-[Shape](../../../../glossary.md#shape) getter, epsilon getter, generic `Module.forward`, layer interface, unary
  module contract, <span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>`Sequential`, [activation](../../../../glossary.md#term-activation) [facade](../../../../glossary.md#term-facade), or broad layer abstraction.
- Buffer state, running statistics, BatchNorm, RMSNorm, GroupNorm, Dropout, Embedding, checkpoints,
  state dictionaries, serialization, optimizer groups, or training sessions.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Embedding glossary](../../../../glossary.md#embedding); [checkpoints glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Any new normalization operation, attributes, Tensor overload, numerical formula, Model helper,
  gradient rule, compiler capture/fusion, backend lowering, execution, or numerical result test.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- Gradle/dependency changes, architecture/ADR/explanatory-architecture changes, architecture-test
  changes, Model/Training production changes, CPU work, global-roadmap changes, or unrelated
  refactors.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Completed NN task 0005](0005-linear-layer.md) and its foundation dependencies.
- [Model task 0021: Layer normalization](../../../modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Layer normalization glossary](../../../../glossary.md#layer-normalization))</span>.
- [Tensor API layer-normalization expressions](../../../../api/tensor-api.md#layer-normalization-expressions)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [normalization glossary](../../../../glossary.md#term-normalization))</span>.

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- NN owns affine parameter declarations and typed forward composition. Model remains the sole
  owner of generic layer-normalization operation meaning, epsilon semantics, promotion, Shape,
  result metadata, and provenance. Training remains downstream and owns optimizer algorithms and
  update orchestration.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [normalization glossary](../../../../glossary.md#term-normalization); [operation glossary](../../../../glossary.md#operation); [epsilon glossary](../../../../glossary.md#epsilon); [Shape glossary](../../../../glossary.md#shape); [metadata glossary](../../../../glossary.md#term-metadata); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) remain [immutable](../../../../glossary.md#term-immutable). `LayerNorm` may retain mutable NN
  parameter bindings but must not mutate a Tensor or add gradient lifecycle state.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [bindings glossary](../../../../glossary.md#term-binding); [gradient glossary](../../../../glossary.md#term-gradient); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Construction and forward build only eager leaves or storage-free Model expression metadata.
  They must not imply compiler capture, backend support, numerical evaluation, storage residency,
  or execution.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [capture glossary](../../../../glossary.md#term-capture-meanings); [backend glossary](../../../../glossary.md#backend); [evaluation glossary](../../../../glossary.md#term-evaluation); [residency glossary](../../../../glossary.md#residency))</span>
- Train/eval mode remains NN forward-composition metadata. This mode-insensitive layer must not
  add an unused context argument or execution-state dependency.<span data-glossary-navigation="true"> ([eval mode glossary](../../../../glossary.md#inference-and-evaluation); [argument glossary](../../../../glossary.md#term-argument); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Preserve unrelated worktree changes exactly, including any concurrent [CPU](../../../../glossary.md#term-cpu) work and
  `docs/architecture/training-graph.md`; this task does not authorize edits outside its exact
  seven paths.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing [`Module`](../../../../glossary.md#term-module) and [`Parameter`](../../../../glossary.md#term-parameter) ownership,
  discovery, and replacement contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `io.github.pho001.synaptik.nn.initialization` — existing exact `ones` and `zeros` creation.
- `io.github.pho001.synaptik.model.tensor` — existing affine `Tensor.layerNorm` expression.
- `io.github.pho001.synaptik.model.operation.normalization` — existing
  `AffineLayerNormAttrs` intrinsic [Shape](../../../../glossary.md#shape)/[epsilon](../../../../glossary.md#epsilon) [validation](../../../../glossary.md#term-validation).
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing immutable Shape, floating
  data type, and exact typed <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [data type glossary](../../../../glossary.md#data-type--datatype))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) contracts.

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — add one concrete stateful [normalization](../../../../glossary.md#term-normalization) layer beside
  [`Linear`](../../../../glossary.md#term-linear-meanings); no package or layer hierarchy is introduced.

Type placement:

- `io.github.pho001.synaptik.nn.layers.LayerNorm` — public stateful affine normalization layer;
  it belongs with concrete layers rather than generic module ownership, initialization policy, or
  Model operation semantics.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- `io.github.pho001.synaptik.nn.layers.LayerNormTest` — exact public surface, caller-supplied state,
  forward, mode, replacement, and Model-delegation coverage.
- `io.github.pho001.synaptik.nn.layers.LayerNormInitializationTest` — exact ones/zeros initialization,
  typed epsilon, validation/identifier effects, and all current floating types.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LayerNorm.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LayerNormTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LayerNormInitializationTest.java`.

Documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend the existing NN and layer-normalization entries with the current
  stateful layer while keeping Model semantics and execution status distinct.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [normalization glossary](../../../../glossary.md#term-normalization); [Model glossary](../../../../glossary.md#term-model))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, dependency rules, module boundaries, training graph, and
  architecture tests — this task realizes their existing NN layer responsibility without changing
  a boundary, mode owner, or dependency.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [training graph glossary](../../../../glossary.md#training-graph); [owner glossary](../../../../glossary.md#term-owner))</span>
- Tensor API and Model task 0021 — the existing affine layer-normalization contract remains the
  sole semantic implementation and <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`LayerNorm.forward` delegates exactly to it.
- Training API — discovery and individual parameter replacement remain unchanged; no optimizer,
  buffer transition, or training workflow is added.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Model capabilities/plans, public API status, user training docs, Gradle, conformance/integration
  suites, compiler/runtime/prepare/Engine, backends, other modules, CPU planning, and the global
  roadmap — none owns this bounded Model-composition wrapper.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- two production Java files;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- the three documentation/[planning](../../../../glossary.md#term-planning) files listed above.

That is an exact seven-path maximum and four Java source/test paths, within the normal 3–12-file
task guardrail. If another type, [API](../../../../glossary.md#term-application-programming-interface-api) document, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [architecture test](../../../../glossary.md#term-architecture-test), [Model](../../../../glossary.md#term-model) helper,
or executable integration path is required, stop and propose a focused follow-up rather than
expanding this task.

## Acceptance criteria

- Final public `LayerNorm` extends [`Module`](../../../../glossary.md#term-module) and declares exactly the two constructors and three
  methods listed in Scope, with no additional public/protected member.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Supplied and initialized construction validate every argument in the specified order before
  declaration and before task-owned Tensor allocation respectively. Every documented caller-
  controlled failure leaves Tensor identifier state unchanged.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Direct and recursive state discovery expose exactly `scale` then [`bias`](../../../../glossary.md#term-bias); accessors return those
  exact wrappers, and supplied construction retains exact input Tensor references.
- Initialized construction uses the exact supplied normalized Shape and data type, delegates once
  to <span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`ones` then once to `zeros`, produces gradient-eligible scale/bias leaves for BFLOAT16,
  FLOAT32, and FLOAT64, and proves exact one/zero values and scale-before-bias ID order without RNG.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- The normalized [parameter](../../../../glossary.md#term-parameter) [Shape](../../../../glossary.md#shape) is positive, fully static, and permanently enforced by existing
  `Parameter.replace`; supplied state has one exact floating type and structural Shape.
- Epsilon is retained by exact reference, has the exact parameter data type, and uses the current
  Model intrinsic finite-positive typed validation. No default, conversion, duplicated numeric
  predicate, or untyped epsilon is introduced.<span data-glossary-navigation="true"> ([Epsilon glossary](../../../../glossary.md#epsilon); [Model glossary](../../../../glossary.md#term-model); [finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation))</span>
- `forward` null-checks input, reads both current bindings once, and delegates to the exact affine
  Model overload. Tests inspect result Shape/type/gradient metadata, <span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata))</span>`AffineLayerNormAttrs`
  identity, one producer/output index zero, and exact ordered <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [producer glossary](../../../../glossary.md#term-producer))</span>`[input, scale, bias]` inputs without
  evaluating values.
- Tests cover inherited Model trailing-Shape, non-floating input, promotion/epsilon mismatch, and
  allocation failures without duplicating the exhaustive Model 0021 matrix or changing failure
  semantics.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Training and evaluation mode produce the same composition contract; no context overload,
  generic Module forward method, hidden state, or mode-dependent operation appears.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [operation glossary](../../../../glossary.md#operation))</span>
- Compatible replacements appear only in later forward provenance, earlier expressions keep old
  bindings, and incompatible state replacement remains rejected by the existing schema.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [schema glossary](../../../../glossary.md#term-schema))</span>
- No buffer, RNG, new semantic operation, optimizer, checkpoint, serialization, compiler,
  runtime, backend, execution, build, dependency, architecture, CPU, roadmap, or unrelated change
  enters the diff.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Public and package Javadocs cover state schema/order, affine all-or-none policy, initialization,
  exact typed epsilon, stored configuration, Model delegation, mode irrelevance, replacement
  snapshots, nullability, validation and side effects, inherited failures, thread safety, and
  no-execution boundaries with complete tags.<span data-glossary-navigation="true"> ([nullability glossary](../../../../glossary.md#term-nullability))</span>
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, generated Javadoc, local links/anchors, exact scope,
  no-change conclusions, and whitespace before the task becomes Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LayerNormTest --tests io.github.pho001.synaptik.nn.layers.LayerNormInitializationTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite:

```bash
./gradlew :extensions:nn:test
```

The focused tests own the exact declared [API](../../../../glossary.md#term-application-programming-interface-api), state [schema](../../../../glossary.md#term-schema)/order/[identity](../../../../glossary.md#term-identity-meanings), initialized leaves and
identifier effects, intrinsic and exact-type [epsilon](../../../../glossary.md#epsilon) validation, [Model](../../../../glossary.md#term-model) [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance)
delegation, mode irrelevance, and replacement snapshots. Existing Model 0021 tests remain the
authoritative exhaustive semantic/validation [matrix](../../../../glossary.md#term-matrix-meanings); do not duplicate them or claim numerical
execution.

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates changed Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `LayerNorm` and package pages, exact seven-path [scope](../../../../glossary.md#term-scope), package placement,
public surface, forbidden imports, 0001–0005 Complete, 0006 [Ready](../../../../glossary.md#term-ready) before implementation and
Complete only after all evidence, exactly one Ready NN task, and absence of detailed NN 0007 or
later task files. It preserves unrelated [CPU](../../../../glossary.md#term-cpu) and training-[graph](../../../../glossary.md#graph) changes and may reuse final NN
test evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one existing model-only [module](../../../../glossary.md#term-module), no build edge, [dependency](../../../../glossary.md#term-dependency) rule, architecture
boundary, shared [Gradle](../../../../glossary.md#term-gradle) contract, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution path.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0005 are Complete.
- [Model](../../../../glossary.md#term-model) task 0021 and the exact affine `Tensor.layerNorm` overload are Complete.
- Existing `ParameterInitializers.ones` and `zeros`, [`Parameter`](../../../../glossary.md#term-parameter) schema replacement, and module
  discovery/mode contracts are Complete.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [module glossary](../../../../glossary.md#term-module))</span>
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and the model-only NN [dependency direction](../../../../glossary.md#java-platform-and-build-tools) remain unchanged.
- The user-authorized NN parallel exception remains in force; CPU, Engine, and numerical layer-
  normalization execution are not prerequisites for expression composition.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [normalization glossary](../../../../glossary.md#term-normalization))</span>

## Follow-up tasks

- NN 0007: Embedding layer — Draft only; parameter-table initialization and padding behavior will
  be decided at that frontier.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [parameter glossary](../../../../glossary.md#term-parameter); [padding glossary](../../../../glossary.md#padding))</span>
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0008: BatchNorm — Draft only; first explicit [`Buffer`](../../../../glossary.md#term-buffer) plus train/eval state-transition layer.
- NN 0009: [Dropout](../../../../glossary.md#dropout--inverted-dropout) — Draft only; explicit [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) threading and evaluation bypass, with no
  hidden RNG.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- NN 0010: state dictionary and checkpoint contract — Draft only; deterministic paths, schema,
  atomic validation/load, and serialization boundary.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [schema glossary](../../../../glossary.md#term-schema); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- NN 0011: [unary](../../../../glossary.md#term-unary) [Tensor](../../../../glossary.md#tensor) [module](../../../../glossary.md#term-module) composition and `Sequential` — Draft only; a narrow shared contract
  is allowed only when the real container requires it.
- `extensions/training` later consumes discovered parameters; this task adds no optimizer or
  session behavior. Numerical end-to-end validation follows only when the compiled execution path
  and backend coverage are available.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers composed from
[Model](../../../../glossary.md#term-model) semantics. If implementation requires a new [dependency](../../../../glossary.md#term-dependency), architecture rule, Model change,
generic layer abstraction, [buffer](../../../../glossary.md#term-buffer) [lifecycle](../../../../glossary.md#lifecycle), or executable behavior, stop and report the conflict
rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in the Synaptik repository without commit or push. Do not use any GSD skill or workflow.
Read AGENTS.md, ARCHITECTURE.md, the current architecture index, planning and documentation rules,
the NN master plan and completed NN tasks 0001–0005, Model task 0021, current Module/Parameter/
initializer contracts, Tensor.layerNorm and its focused tests, and
docs/planning/extensions/nn/tasks/0006-layer-normalization-layer.md in full.

Implement task 0006 exactly inside its seven authorized paths. Preserve all unrelated worktree
changes exactly, especially CPU work and docs/architecture/training-graph.md. Stop and report any
architecture uncertainty, scope overflow, or need for another public type, dependency, or file.

Run focused validation and one final NN suite after executable Java stabilizes. Then hand the
actual diff, exact Java evidence, and task contract to a separate documentation-focused clean
context in the same overall change. That pass finalizes package/type Javadocs, glossary and
planning evidence, generated Javadoc, Markdown, scope, no-change conclusions, and whitespace
without repeating successful Java tests unless executable behavior changes. Mark Complete only
after every criterion passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evidence and whether Java changed afterward, the selected constructor/[API](../../../../glossary.md#term-application-programming-interface-api) surface,
state/[epsilon](../../../../glossary.md#epsilon) [validation](../../../../glossary.md#term-validation) order, initializer/identifier effects, forward [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance),
replacement and train/eval evidence, and the exact seven authorized paths.

That agent independently reads the repository instructions, [architecture contract](../../../../glossary.md#architecture-contract) and index,
documentation rules plus General/API-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning) profiles, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, this task, final source
and tests, generated Javadoc, [Tensor](../../../../glossary.md#tensor) and Training APIs, glossary, NN master/task history, [Model](../../../../glossary.md#term-model)
0021, `Tensor.layerNorm`, `AffineLayerNormAttrs`, [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), and
`ParameterInitializers`. It finalizes only package/type Javadoc, glossary, this task, and NN master
plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/Training/public
APIs, Model plans/contracts, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), execution layers, [backends](../../../../glossary.md#backend), other
modules, [CPU](../../../../glossary.md#term-cpu) work, [training graph](../../../../glossary.md#training-graph), and global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The stateful layer is affine-only with mandatory `scale` and [`bias`](../../../../glossary.md#term-bias). This matches the exact
  all-or-none Model affine signature and avoids nullable state, hidden constants, or a partial
  two-input semantic form.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [constants glossary](../../../../glossary.md#term-constant))</span>
- Supplied construction infers and retains the exact scale [Shape](../../../../glossary.md#shape) as `normalizedShape`; initialized
  construction receives the Shape explicitly. Both require positive fully static parameter
  schema even though the generic Model expression can represent empty or unresolved normalized
  axes.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [schema glossary](../../../../glossary.md#term-schema); [axes glossary](../../../../glossary.md#term-axis))</span>
- [Epsilon](../../../../glossary.md#epsilon) is an exact [immutable](../../../../glossary.md#term-immutable) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue). Construction reuses `AffineLayerNormAttrs` for the
  current intrinsic finite-positive validation and separately requires its type to equal the
  parameter type. Forward leaves exact promoted-result compatibility to Model and inserts no cast.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [validation glossary](../../../../glossary.md#term-validation))</span>
- Initialization is exact ones for scale then zeros for bias. Layer normalization needs no RNG,
  and a configurable affine initializer has no current consumer.<span data-glossary-navigation="true"> ([Layer normalization glossary](../../../../glossary.md#layer-normalization); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Only stable parameter handles are public. Normalized Shape and epsilon remain exact internal
  configuration because no current introspection, serialization, or generic layer consumer needs
  public getters; a later concrete consumer must justify widening the surface.<span data-glossary-navigation="true"> ([Normalized Shape glossary](../../../../glossary.md#normalized-shape); [serialization glossary](../../../../glossary.md#term-serialization); [widening glossary](../../../../glossary.md#term-widening))</span>
- Forward is mode-insensitive and delegates directly to [Model](../../../../glossary.md#term-model) without an unused `ForwardContext`.

## Known limitations

- Only mandatory affine state with a positive fully static normalized parameter Shape is
  supported; there is no no-affine, optional-state, lazy, dynamic, or zero-size layer form.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape); [lazy glossary](../../../../glossary.md#term-lazy-evaluation))</span>
- Mixed floating input is accepted only when the existing Model promotion yields the stored
  epsilon's exact type. The layer performs no implicit cast or epsilon conversion.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [epsilon glossary](../../../../glossary.md#epsilon))</span>
- Parameter replacement is individual and not thread-safe. Callers must synchronize replacement
  with forward construction when one scale/bias snapshot matters.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [bias glossary](../../../../glossary.md#term-bias); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Forward constructs expression metadata and proves no compiler capture, gradients, fusion,
  backend coverage, numerical value, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend))</span>
- There is no state dictionary, checkpoint, serialization, device/dtype conversion, buffer, RNG,
  optimizer, or training-session convenience.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [device glossary](../../../../glossary.md#term-device); [dtype glossary](../../../../glossary.md#data-type--datatype); [buffer glossary](../../../../glossary.md#term-buffer); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `/root/nn_next_steps_plan` read the repository instructions, architecture and
  NN/training boundaries, planning/documentation rules, NN master and completed tasks, current NN
  source/tests, Model 0021 plus adjacent Embedding/BatchNorm/Dropout contracts, Tensor/Training
  APIs, and glossary before fixing this exact API and scope.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Planning inspection confirmed that affine [Model](../../../../glossary.md#term-model) LayerNorm requires exact ordered
  `[input, scale, bias]`, exact scale/bias Shapes equal to a positive-rank normalized Shape, and a
  finite positive <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [Shapes glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank); [normalized Shape glossary](../../../../glossary.md#normalized-shape); [finite glossary](../../../../glossary.md#term-finite))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) whose type equals the promoted result. Existing initializers can
  create both mandatory parameter leaves without RNG or another dependency.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Planning Markdown, link/anchor, status, later-task-absence, newline, whitespace, exact two-path
  planning scope, and `git diff --check` results are recorded by the planning agent before [handoff](../../../../glossary.md#term-handoff).
- Clean implementation context `/root/nn_0006_implementation` added the exact planned final
  `LayerNorm`, updated the existing `layers` package contract, and added the two planned focused
  test classes after reading the required repository, architecture, planning, NN, Model
  normalization, API, glossary, source, and test contracts. It found no architecture or scope
  conflict.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- The first focused command failed at test compilation because an all-null constructor call was
  ambiguous and one test attempted to use Tensor's package-private constructor from the NN
  package. Both were test-only issues: the calls now select constructor intent explicitly and the
  identifier-exhaustion check reuses a valid Tensor created before exhaustion. The next focused
  run reached execution and found that <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [run glossary](../../../../glossary.md#term-run))</span>`Long.MAX_VALUE` is one final claimable Tensor ID; the test
  now also preserves and sets the Model's maximum-ID-claimed state to exercise permanent
  exhaustion accurately.
- A preliminary successful focused run and preliminary full NN run both passed with the same
  eventual 2-suite/9-test and 11-suite/53-test counts. They were superseded when implementation
  review strengthened two validation-order assertions from generic role text to the exact
  floating-type failure class; production Java did not change. The focused and full commands were
  then rerun so the evidence below covers the final tests.
- Final focused
  `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LayerNormTest --tests io.github.pho001.synaptik.nn.layers.LayerNormInitializationTest`
  passed with `BUILD SUCCESSFUL`: 2 suites and 9 tests, with zero failures, errors, or skips.
- After source and tests stabilized, the sole authoritative final
  `./gradlew :extensions:nn:test` passed with `BUILD SUCCESSFUL` (5 actionable tasks: 1 executed,
  4 up-to-date). XML reports contain 11 suites and 53 tests, with zero failures, errors, or skips.
  No executable Java or tests changed after that final run in this implementation context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed on the implementation draft (`BUILD
  SUCCESSFUL`, 3 actionable tasks: 1 executed, 2 up-to-date). This preliminary result was
  superseded by the authoritative clean-context documentation result below.
- `javap -public` showed exactly the planned two constructors and three declared methods.
  Production imports remain confined to Model, the existing NN initialization/module packages,
  and `java.util.Objects`. `git diff --check` and new-file final-newline/trailing-whitespace checks
  passed. The clean documentation context independently repeated or completed the public-surface,
  generated-page, Markdown, glossary, exact seven-path, no-change, synchronized-status, and final
  whitespace checks below.
- Clean documentation context `/root/nn_0006_docs` independently read the repository and
  architecture contracts, focused architecture explanations and ADR, planning and documentation
  rules plus General/API-Javadoc/Planning/Example profiles, NN master and tasks 0001–0006, Model
  master and task 0021, final layer/package/test sources, Module/Parameter/initializer/Linear
  contracts, Tensor/Shape/DataType/ScalarValue/affine-attributes normalization API, Tensor and
  Training references, glossary, Java 26 Gradle configuration, generated Javadoc, and the actual
  shared-worktree diff. It changed no executable Java or tests.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Linear glossary](../../../../glossary.md#term-linear-meanings); [DataType glossary](../../../../glossary.md#data-type--datatype))</span>
- That context finalized the type, constructor, method, and package Javadocs for exact retained
  references, complete validation and side-effect order, initialized one-then-zero state,
  return/provenance semantics, direct Model delegation, mode-insensitivity, replacement snapshots,
  nullability/failures, and the declarative non-execution boundary. It extended the glossary's NN
  and layer-normalization entries with the current stateful owner and a <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [nullability glossary](../../../../glossary.md#term-nullability); [owner glossary](../../../../glossary.md#term-owner))</span>`[2, 3]` input / `[3]`
  normalized-Shape composition example that makes no numerical-result or execution claim.
- Authoritative final `./gradlew :extensions:nn:javadoc` passed with `BUILD SUCCESSFUL` (3
  actionable tasks: 2 executed, 1 up-to-date). Manual inspection of generated `LayerNorm.html` and
  `layers/package-summary.html` confirmed the finalized descriptions, links, tags, exceptions, and
  exact public members render as intended.
- `javap -public` and an independent reflection program both confirmed public final `LayerNorm`
  extends [`Module`](../../../../glossary.md#term-module) and declares exactly the two planned constructors plus `scale()`, `bias()`, and
  `forward(Tensor)`, with no other declared public or protected member. Manual source inspection
  confirmed one scale read then one bias read and exact delegation to
  `input.layerNorm(normalizedShape, currentScale, currentBias, epsilon)`.
- Production-import and [Gradle](../../../../glossary.md#term-gradle) inspection confirmed `LayerNorm` uses only Model, existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), and
  `java.util.Objects`, while `extensions/nn` still has exactly its existing Model dependency. A
  targeted local checker passed for changed Markdown targets, anchors, and balanced fences. Exact
  task-owned seven-path, later-task absence, 0001–0006 status, 0007–0011 Draft rows, final-newline,
  trailing-whitespace, and <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` checks also passed after final [synchronization](../../../../glossary.md#term-synchronization).
- No-change review concluded that `ARCHITECTURE.md`, the current architecture explanations, ADR
  0007, dependency and training-graph documents, architecture tests, Tensor API and Model task
  0021, Training API, Model capabilities/master plan, public API status, conformance/integration
  suites, compiler/runtime/prepare/Engine, backends, other modules, Gradle files, CPU planning, and
  the global roadmap remain accurate and outside scope: the layer only owns NN parameters and
  composes an existing Model expression, without changing any dependency, semantic operation,
  training workflow, backend behavior, or executable path. Concurrent CPU and global-roadmap
  worktree changes were inspected only for scope separation and preserved unchanged.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation))</span>

## Implementation notes

- Added final public `LayerNorm` with the exact two constructors and three methods, mandatory
  `scale` then [`bias`](../../../../glossary.md#term-bias) declarations, retained exact normalized Shape and typed epsilon, and direct
  mode-insensitive affine Model delegation.<span data-glossary-navigation="true"> ([normalized Shape glossary](../../../../glossary.md#normalized-shape); [epsilon glossary](../../../../glossary.md#epsilon); [Model glossary](../../../../glossary.md#term-model))</span>
- Supplied construction completes the specified null, state-schema, intrinsic-epsilon, and exact-
  type validation before either declaration. Initialized construction validates all
  caller-controlled inputs before allocating exact one scale then zero bias leaves.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [validation glossary](../../../../glossary.md#term-validation))</span>
- Focused tests lock public surface, state identity/order, validation and identifier effects,
  typed eager values, exact producer/provenance, Model failure inheritance, mode irrelevance,
  individual replacement snapshots, and identifier exhaustion without numerical execution.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [eager glossary](../../../../glossary.md#term-eager-evaluation); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- No Model, Training, build, dependency, architecture, CPU, global-roadmap, execution-layer, or
  other out-of-scope path changed. The independent documentation pass finalized the package/type
  Javadocs, glossary, planning evidence, and no-change conclusions.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: Implemented, documented, and validated the exact final affine `LayerNorm`
  module with supplied or deterministic one/zero state, exact typed epsilon, stable parameters,
  mode-insensitive direct Model delegation, and replacement snapshots.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [epsilon glossary](../../../../glossary.md#epsilon); [parameters glossary](../../../../glossary.md#term-parameter); [Model glossary](../../../../glossary.md#term-model))</span>
- Files changed or created: Exactly the planned two production Java paths, two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test paths,
  `docs/glossary.md`, NN master plan, and this task record.
- Tests and validation: Reused final implementation evidence of focused 2-suite/9-test and full NN
  11-suite/53-test success because the documentation pass changed no executable Java or tests.
  Final NN Javadoc, generated-page inspection, <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`javap`, reflection, manual source/import/dependency
  review, Markdown links/anchors/fences, exact scope/status/later-task checks, final newlines,
  trailing whitespace, and <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` all passed.
- Documentation-agent review: Clean context `/root/nn_0006_docs` completed the required independent
  API-Javadoc, package, glossary, planning, generated-output, and no-change review.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Final public/package Javadocs and the glossary now distinguish the
  stateful NN owner from existing Model mathematics and future compiler/backend/runtime execution.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: Every declared constructor and method has meaningful contract text and complete
  parameter, result, ownership, side-effect, failure, and boundary documentation; final generated
  output was inspected.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: The existing NN and layer-[normalization](../../../../glossary.md#term-normalization) entries now describe current
  `LayerNorm` state, initialization, forward composition, replacement timing, and a conceptual
  trailing-Shape example without a numerical-execution claim.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Unresolved issues: None.
- Required follow-up: None for task 0006; NN 0007 remains Draft and intentionally has no detailed
  task specification.

Status: Complete

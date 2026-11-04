# Task 0005: [Linear](../../../../glossary.md#term-linear-meanings) Layer

> **Editorial reading context — not a status update.** The application calls Linear.forward
> with an input Tensor; the layer reads its current weight/bias bindings and delegates to
> Tensor.linear to describe a projection. Supplied-state construction retains exact references;
> initialized construction creates values. Neither means forward numerically multiplies them.
> Weight stores output features by input features, so the expression transposes that orientation
> for matrix multiplication. Today's [linear producer chain](../../../../reference/tensor/matmul-linear.md#linear-projection-convenience)
> explains that composition, and [parameter bindings](../../../../reference/nn/state.md#current-nn-parameter-update-contract)
> explain why replacing a weight does not rewrite an earlier projection.

## Status

Complete

## Goal

Add the first stateful neural-network layer as one final public [`Linear`](../../../../glossary.md#term-linear-meanings) [module](../../../../glossary.md#term-module). The layer owns a
[rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[outFeatures, inFeatures]` [weight](../../../../glossary.md#term-weight) [`Parameter`](../../../../glossary.md#term-parameter), optionally owns one exact [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
`[outFeatures]` [bias](../../../../glossary.md#term-bias) `Parameter`, and constructs its output only through the existing
`Tensor.linear` convenience. Callers can either supply the exact parameter [Tensors](../../../../glossary.md#tensor) or request the
fixed initial [Glorot-uniform](../../../../glossary.md#term-glorot-uniform)-weight and zero-bias policy with an explicit floating [data type](../../../../glossary.md#data-type--datatype) and
caller-owned random source.

## [Scope](../../../../glossary.md#term-scope)

- Add final public `io.github.pho001.synaptik.nn.layers.Linear` extending the existing
  `io.github.pho001.synaptik.nn.module.Module`.
- Add exactly these public constructors and methods:

  ```java
  public Linear(Tensor weight)

  public Linear(Tensor weight, Tensor bias)

  public Linear(
          long inFeatures,
          long outFeatures,
          boolean bias,
          DataType dataType,
          RandomGenerator randomGenerator)

  public Parameter weight()

  public Optional<Parameter> bias()

  public Tensor forward(Tensor input)
  ```

- The one-Tensor constructor creates a no-bias layer. The two-Tensor constructor requires a
  non-null bias and creates a biased layer; null never means absence. The initialized constructor
  uses its explicit [`bias`](../../../../glossary.md#term-bias) flag and has no overload that silently chooses whether bias exists.
- Every layer declares the [weight](../../../../glossary.md#term-weight) under exact local name `weight`. A biased layer then declares
  bias under exact local name `bias`, so direct and recursive discovery retain the established
  weight-before-bias order and stable paths.
- Require every supplied weight to be floating, gradient-eligible, fully static, rank two, and
  positive on both axes. Interpret its exact Shape as <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [axes glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape))</span>`[outFeatures, inFeatures]`. Retain its exact
  Tensor reference through the [`Parameter`](../../../../glossary.md#term-parameter) declaration; do not require a label, host storage,
  provenance absence, dense layout, or a particular Tensor identity.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Require every supplied bias to be floating, gradient-eligible, fully static, rank one, have the
  exact same data type as the weight, and have its sole Dimension structurally equal to weight
  axis zero. Retain its exact Tensor reference. General broadcasting, a scalar, a singleton for
  non-singleton out-features, cross-type promotion, null-as-absence, and a dynamic bias are not
  accepted as layer state.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Dimension glossary](../../../../glossary.md#dimension); [broadcasting glossary](../../../../glossary.md#broadcasting); [scalar glossary](../../../../glossary.md#term-scalar); [singleton glossary](../../../../glossary.md#term-singleton); [features glossary](../../../../glossary.md#term-feature))</span>
- Validate a biased supplied construction in exact order: null weight, null bias, weight floating
  type, weight gradient eligibility, weight rank, fully-static weight Shape, positive
  out-features, positive in-features, bias floating type, bias gradient eligibility, bias rank,
  fully-static bias Shape, exact weight/bias data-type equality, and exact bias/out-features
  Dimension equality. The no-bias constructor uses the same weight subsequence. All validation
  completes before declaring either <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`Parameter`; it allocates no Tensor, consumes no Tensor ID,
  evaluates no expression, and changes neither supplied Tensor.
- The initialized constructor validates in exact order: non-null `dataType`, non-null
  `randomGenerator`, positive `inFeatures`, positive `outFeatures`, then floating data type. It
  builds exact Shapes `Shape.of(outFeatures, inFeatures)` and, when requested,
  `Shape.of(outFeatures)`, and lets the existing initializers complete their documented
  Java-array-limit, checked-arithmetic, allocation, identifier, and random-source validation.
- Use exactly `ParameterInitializers.glorotUniform(weightShape, dataType, randomGenerator)` for
  initialized weight. When `bias == true`, use exactly
  `ParameterInitializers.zeros(biasShape, dataType)` after the weight initializer succeeds. Do not
  duplicate either formula or sampling loop. The fixed weight bounds are therefore<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
  `[-sqrt(6 / (inFeatures + outFeatures)), +sqrt(6 / (inFeatures + outFeatures)))`, and bias is
  deterministic exact typed zero and consumes no random draw.
- The initialized path creates the weight Tensor before an optional bias Tensor and declares the
  matching parameters in that order. All caller-controlled validation failures happen before a
  random draw or Tensor identifier allocation. A source failure preserves the initializer's
  completed-draw/non-rollback contract and creates no Tensor. Resource, identifier-exhaustion, or
  allocation failure after a successful weight initialization does not roll back weight draws or
  its consumed identifier; no partially constructed <span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>[`Linear`](../../../../glossary.md#term-linear-meanings) instance is returned.
- `weight()` returns the exact stable declared wrapper. `bias()` returns a non-null empty
  `Optional` for a no-bias layer or an `Optional` containing the exact stable bias wrapper. Add no
  setter or replacement facade; the returned <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>`Parameter` keeps [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004A's public [schema](../../../../glossary.md#term-schema)-compatible
  `replace` capability for inspection and downstream training.
- `forward(input)` first rejects a null input with message `input`, reads each current parameter
  binding once, and delegates exactly to <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`input.linear(weight.value())` or
  `input.linear(weight.value(), bias.value())`. It must not reproduce transpose, MATMUL, ADD,
  Shape, promotion, producer, or provenance logic.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [producer glossary](../../../../glossary.md#term-producer))</span>
- Forward inherits the Model contract for input rank, floating promotion, final-axis contraction,
  Shape derivation, validation, allocation order, gradient eligibility, and visible
  PERMUTE-to-MATMUL-to-optional-ADD provenance. It constructs model metadata only and does not
  evaluate values.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- `Linear` forward behavior is identical in `TRAINING` and [`EVALUATION`](../../../../glossary.md#term-evaluation) modes. `train()` and
  `eval()` retain their inherited [metadata](../../../../glossary.md#term-metadata) behavior, but `forward` accepts no `ForwardContext`
  and does not branch on mode.
- A successful `Parameter.replace` affects the next forward call. A [Tensor](../../../../glossary.md#tensor) returned by an earlier
  `value()` call and every expression already constructed from it remain unchanged. The
  declaration schema prevents replacement with another data type, Shape, or non-gradient Tensor.
- Add focused tests for exact surface and state ownership, supplied-Tensor validation, initialized
  Tensor policy and source ownership, forward metadata/provenance, mode irrelevance, and
  replacement snapshots.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Add complete [Javadoc](../../../../glossary.md#term-javadoc) and `layers` package documentation, then use a separate clean-context
  documentation pass to finalize Javadoc, glossary impact, planning evidence, and documentation
  validation in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- A generic `Module.forward`, layer interface, abstract layer base, sequential container, block,
  activation, flattening, lazy/dynamic feature inference, or dynamic parameter Shape.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [feature glossary](../../../../glossary.md#term-feature); [inference glossary](../../../../glossary.md#term-inference-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape))</span>
- A caller-selectable initializer, initializer object, Xavier alias, configurable gain or fan
  policy, random bias, default data type, default seed, hidden or retained RNG, or module-owned
  random state.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [bias glossary](../../../../glossary.md#term-bias); [data type glossary](../../../../glossary.md#data-type--datatype); [seed glossary](../../../../glossary.md#term-seed); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [module glossary](../../../../glossary.md#term-module))</span>
- Integral or BOOL layer parameters, frozen parameters, mixed-type weight and bias, broadcast bias,
  a public state setter, buffer, checkpoint, state dictionary, serialization, device placement,
  or parameter-group behavior.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [weight glossary](../../../../glossary.md#term-weight); [buffer glossary](../../../../glossary.md#term-buffer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [device glossary](../../../../glossary.md#term-device))</span>
- Optimizer algorithms, gradient publication, training sessions, autograd rules, compiler capture
  or fusion, graph compilation, prepare/runtime/Engine behavior, backend lowering, numerical
  execution, or end-to-end output-value assertions.<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [lowering glossary](../../../../glossary.md#lowering))</span>
- A new LINEAR operation kind, attrs, producer, result carrier, Tensor overload, MATMUL/ADD/
  PERMUTE change, or duplicate linear numerical/Shape/provenance implementation.<span data-glossary-navigation="true"> ([LINEAR glossary](../../../../glossary.md#term-linear-meanings); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [Tensor glossary](../../../../glossary.md#tensor); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [provenance glossary](../../../../glossary.md#provenance))</span>
- Gradle or dependency changes, architecture-contract/ADR/explanatory-architecture changes,
  architecture-test changes, Model changes, Training production changes, CPU work, or global
  roadmap changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Tensor glossary](../../../../glossary.md#tensor); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md) through
  [task 0004A](0004a-parameter-update-and-traversal-hardening.md).
- [Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Linear glossary](../../../../glossary.md#term-linear-meanings))</span>.
- [Tensor API linear-projection convenience](../../../../api/tensor-api.md#linear-projection-convenience)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>.

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- NN owns layer state, typed forward composition, and parameter declarations. Model remains the
  sole owner of generic <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner))</span>`Tensor.linear` validation and primitive expression semantics. Training
  remains downstream and owns optimizer algorithms and orchestration.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), and [provenance](../../../../glossary.md#provenance) remain [immutable](../../../../glossary.md#term-immutable). [`Linear`](../../../../glossary.md#term-linear-meanings) may retain and discover
  mutable NN <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>[`Parameter`](../../../../glossary.md#term-parameter) [bindings](../../../../glossary.md#term-binding) but must not mutate a Tensor or add [gradient](../../../../glossary.md#term-gradient) [lifecycle](../../../../glossary.md#lifecycle) state.
- The layer constructs a visible existing primitive Tensor-expression chain and must not imply
  compiler capture, fusion, backend support, storage residency, or execution.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend); [residency glossary](../../../../glossary.md#residency))</span>
- Train/eval mode remains NN composition metadata. A mode-insensitive layer does not invent a
  meaningless <span data-glossary-navigation="true"> ([eval mode glossary](../../../../glossary.md#inference-and-evaluation); [metadata glossary](../../../../glossary.md#term-metadata))</span>`ForwardContext` [argument](../../../../glossary.md#term-argument) or execution-state [dependency](../../../../glossary.md#term-dependency).
- The authorized NN parallel exception is implementation-order only. Preserve all dirty CPU
  files, CPU planning, the global roadmap, and unrelated glossary hunks.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing [`Module`](../../../../glossary.md#term-module) and [`Parameter`](../../../../glossary.md#term-parameter) ownership and
  replacement contracts.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `io.github.pho001.synaptik.nn.initialization` — existing exact Glorot-uniform and zero Tensor
  creation policies.<span data-glossary-navigation="true"> ([Glorot-uniform glossary](../../../../glossary.md#term-glorot-uniform); [Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.model.tensor` — existing `Tensor.linear` [model](../../../../glossary.md#term-model) composition.
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing Shape and floating type
  contracts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — public stateful layer types whose typed forward methods
  compose existing Model Tensor operations.

Type placement:

- `io.github.pho001.synaptik.nn.layers.Linear` — first public stateful layer; it belongs with
  concrete NN layers rather than generic module ownership or stateless initialization policy.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `io.github.pho001.synaptik.nn.layers.LinearTest` — focused public-surface, supplied-state,
  forward-provenance, mode, and replacement contract tests.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- `io.github.pho001.synaptik.nn.layers.LinearInitializationTest` — focused initialized-construction,
  formula delegation, source ownership, metadata, validation, and side-effect tests.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [validation glossary](../../../../glossary.md#term-validation))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Linear.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearInitializationTest.java`.

Documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend the existing [`Linear projection`](../../../../glossary.md#linear-projection) and NN module entries to
  distinguish and describe the now-current stateful layer without changing Model semantics.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless the implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, dependency rules, module-boundary/training-graph explanations, and
  architecture tests — the task realizes the already-authorized layer responsibility without a
  dependency or lifecycle change.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [graph glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) and [Model](../../../../glossary.md#term-model) task 0019D — `Tensor.linear` semantics and public surface remain unchanged;
  `Linear.forward` delegates to them.
- Training API — the already-current generic parameter discovery/replacement contract remains
  exact; no optimizer or training workflow is added.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- public API status and user training guide — no runnable training or execution lifecycle becomes
  available, and complete layer usage is documented in package/type Javadoc plus the glossary.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Model capabilities/plans, Gradle, conformance/integration suites, compiler/runtime/prepare/
  Engine, backends, and the global roadmap — none owns this model-only NN layer task.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- two production Java files;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- the three documentation/[planning](../../../../glossary.md#term-planning) files listed above.

That is an exact seven-path maximum. If another production/test type, [API](../../../../glossary.md#term-application-programming-interface-api) document, [module](../../../../glossary.md#term-module),
[dependency](../../../../glossary.md#term-dependency), [architecture test](../../../../glossary.md#term-architecture-test), [Model](../../../../glossary.md#term-model) helper, or executable integration path is required, stop and
propose a focused follow-up rather than expanding this task.

## Acceptance criteria

- Final public [`Linear`](../../../../glossary.md#term-linear-meanings) extends [`Module`](../../../../glossary.md#term-module) and exposes exactly the three constructors and three
  methods listed in Scope, with no additional public/protected member declared by the class.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Supplied state is validated in the specified order before either parameter declaration. Weight
  and bias use exact stable names, wrapper identity, declaration order, input Tensor references,
  type, Shape, and gradient eligibility.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Weight glossary](../../../../glossary.md#term-weight); [identity glossary](../../../../glossary.md#term-identity-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- No-[bias](../../../../glossary.md#term-bias) and biased state discovery returns exactly `[weight]` and `[weight, bias]` respectively;
  `bias()` is empty or contains the same wrapper present in discovery.
- Initialized construction requires explicit positive features, explicit floating type, explicit
  caller-owned RNG, and explicit bias selection. Weight delegates once to Glorot uniform with
  Shape <span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-feature); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [Glorot glossary](../../../../glossary.md#term-glorot-uniform))</span>`[outFeatures, inFeatures]`; optional bias delegates once to typed zeros with [Shape](../../../../glossary.md#shape)
  `[outFeatures]` and consumes no source draw.
- Scripted-source tests prove exact Glorot bounds, row-major draw count, exact caller source use,
  all three floating result types as practical, zero bias values, weight-before-bias ID order, and
  no draw/ID allocation for caller-controlled prevalidation failures. Source/resource failure
  tests assert only the documented non-rollback boundary.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- `forward` has exactly one [Tensor](../../../../glossary.md#tensor) input and delegates to the matching `Tensor.linear` overload
  using current bindings. Tests inspect Shape/type/gradient metadata and exact primitive producers,
  ordered inputs, output indices, and weight/bias identity without executing values.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Null input and inherited Model rank/type/contraction failures are covered without duplicating
  the exhaustive Model 0019D test matrix or asserting a different message/order.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [rank glossary](../../../../glossary.md#term-rank); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Training and evaluation mode produce the same layer composition contract; no context overload or
  generic Module forward method appears.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Replacement tests prove that compatible weight and bias replacements appear in subsequent
  forward provenance, while previously constructed expressions retain the earlier exact bindings;
  incompatible replacement remains rejected by the existing declaration schema.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [schema glossary](../../../../glossary.md#term-schema))</span>
- No new operation/numerical algorithm, optimizer, checkpoint, serialization, compiler, runtime,
  backend, execution, build, dependency, architecture, CPU, global-roadmap, or unrelated refactor
  enters the diff.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Public and package Javadocs cover purpose, parameter orientation, optionality, initialization
  policy, ownership, exact result delegation, train/eval irrelevance, replacement snapshots,
  nullability, validation order, random-source ownership, side effects, failures, and lifecycle
  boundaries. Every input/result/failure has the required tags.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [nullability glossary](../../../../glossary.md#term-nullability); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, no-change conclusions, generated Javadoc, Markdown, exact
  scope, and whitespace before the task becomes Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LinearTest --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) suite:

```bash
./gradlew :extensions:nn:test
```

The focused layer tests own the exact declared [API](../../../../glossary.md#term-application-programming-interface-api) surface, state [schema](../../../../glossary.md#term-schema)/order/[identity](../../../../glossary.md#term-identity-meanings),
initializer delegation and source effects, forward [provenance](../../../../glossary.md#provenance), mode irrelevance, and replacement
snapshots. Existing [module](../../../../glossary.md#term-module), initializer, and [Model](../../../../glossary.md#term-model) tests remain unchanged and protect their
underlying contracts. Do not repeat the exhaustive Model [linear](../../../../glossary.md#term-linear-meanings) suite or make numerical execution
claims.

The separate documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `Linear` and package Javadocs, exact seven-path [scope](../../../../glossary.md#term-scope), package placement,
public constructor/method surface, 0001–0004A Complete, 0005 [Ready](../../../../glossary.md#term-ready) before implementation and
Complete only after all evidence, and no later detailed NN task. It must preserve unrelated dirty
[CPU](../../../../glossary.md#term-cpu) and global-roadmap files and may reuse the final NN test evidence when executable Java has not
changed.

Repository-wide and architecture-test validation are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one existing model-only module, no build edge, [dependency](../../../../glossary.md#term-dependency) rule, architecture
boundary, shared [Gradle](../../../../glossary.md#term-gradle) contract, [backend](../../../../glossary.md#backend) behavior, or end-to-end execution path.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0004A are Complete.
- [Model](../../../../glossary.md#term-model) task 0019D and its exact `Tensor.linear` overloads are Complete.
- Existing `ParameterInitializers.glorotUniform` and `zeros` contracts are Complete.
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 and the model-only NN [dependency direction](../../../../glossary.md#java-platform-and-build-tools) remain unchanged.
- The user-authorized parallel exception recorded in the NN master plan remains in force, with
  dirty CPU and global-roadmap paths outside this task.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Follow-up tasks

- Stateless activation/module conveniences and normalization wrappers remain future NN rows to be
  planned only at their frontier.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [module glossary](../../../../glossary.md#term-module); [normalization glossary](../../../../glossary.md#term-normalization); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Batch normalization and dropout must wait for their explicit buffer/mode and graph-RNG layer
  contracts; they are not inferred from <span data-glossary-navigation="true"> ([Batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [buffer glossary](../../../../glossary.md#term-buffer); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>[`Linear`](../../../../glossary.md#term-linear-meanings).
- [Checkpoint](../../../../glossary.md#term-checkpoint)/state-dictionary semantics remain a separate state-tree task.
- `extensions/training` later consumes the stable discovered [`Parameter`](../../../../glossary.md#term-parameter) wrappers; this task adds
  no optimizer or session contract.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Numerical end-to-end layer validation follows only when the required compiled execution path and
  backend operation coverage are available.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers composed from
[Model](../../../../glossary.md#term-model) semantics. If implementation requires a new [dependency](../../../../glossary.md#term-dependency), execution behavior, architecture
rule, generic layer abstraction, Model change, or broader [lifecycle](../../../../glossary.md#lifecycle) decision, stop and report the
conflict rather than editing architecture within this task.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Work in the Synaptik repository without commit or push. Read AGENTS.md, ARCHITECTURE.md, the
current architecture index, planning and documentation rules, the NN master plan, completed NN
tasks 0001–0004A, Model task 0019D, current Module/Parameter/initializer contracts, and
docs/planning/extensions/nn/tasks/0005-linear-layer.md in full.

Implement task 0005 exactly inside its seven authorized paths. Preserve dirty CPU work, the
global roadmap, Model code, dependency boundaries, and all out-of-scope behavior. Stop and report
any architecture uncertainty, scope overflow, or need for another type or document.

Run focused validation and one final NN suite after executable Java stabilizes. Then hand the
actual diff, exact Java evidence, and task contract to a separate documentation-focused clean
context in the same overall change. That pass finalizes package/type Javadocs, glossary and
planning evidence, no-change conclusions, generated Javadoc, Markdown, scope, and whitespace
without repeating successful Java tests unless executable behavior changes. Mark Complete only
after every criterion passes.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evidence and whether Java changed afterward, the selected constructor/[API](../../../../glossary.md#term-application-programming-interface-api) surface,
state [validation](../../../../glossary.md#term-validation)/order, initializer delegation and source effects, forward [producer](../../../../glossary.md#term-producer) chains,
replacement evidence, train/eval result, and the seven authorized paths.

That agent independently reads the repository instructions, [architecture contract](../../../../glossary.md#architecture-contract) and index,
documentation rules plus General/API-[Javadoc](../../../../glossary.md#term-javadoc)/[Planning](../../../../glossary.md#term-planning) profiles, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, this task, final source
and tests, generated Javadoc, [Tensor](../../../../glossary.md#tensor) and Training APIs, glossary, NN master plan and completed
foundation/initializer tasks, [Model](../../../../glossary.md#term-model) 0019D, `Tensor.linear`, [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), and
`ParameterInitializers`. It finalizes only package/type Javadoc, the glossary, this task, and the NN
master plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/
Training/public APIs, Model plans/contracts, [conformance](../../../../glossary.md#term-conformance)/integration, [Gradle](../../../../glossary.md#term-gradle), execution layers,
[backends](../../../../glossary.md#backend), other modules, [CPU](../../../../glossary.md#term-cpu) work, and the global roadmap.

The documentation pass reuses successful Java evidence unless it changes executable Java or
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The public surface uses constructor arity for caller-supplied bias because null is never a state
  value or absence sentinel. The initialized constructor uses one explicit boolean because two
  otherwise-identical initialization overloads would add surface without adding type safety; no
  overload silently defaults bias on or off.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [bias glossary](../../../../glossary.md#term-bias); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- The fixed first-layer policy is Glorot unit-gain uniform weight plus deterministic exact-zero
  bias. It reuses the two existing initializer entries directly and gives the random source no
  role in bias initialization.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [weight glossary](../../../../glossary.md#term-weight))</span>
- Caller-supplied parameters are fully static and positive. This first stateful layer models fixed
  learnable feature axes; lazy and dynamically shaped parameter ownership require a separate
  contract rather than weakening constructor and replacement schema.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [axes glossary](../../../../glossary.md#term-axis); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- Weight and bias must have the same exact floating type. Allowing final ADD promotion would make
  one layer own two parameter precisions and obscure its declared data type; callers can construct
  explicit casts before supplying state if a later task deliberately permits that policy.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype))</span>
- Stable `weight()` and optional `bias()` accessors expose the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned [`Parameter`](../../../../glossary.md#term-parameter)
  capability. No layer-local setter duplicates `Parameter.replace`, and discovery remains the
  generic downstream path.
- `forward` is mode-insensitive and delegates to [Model](../../../../glossary.md#term-model) rather than accepting an unused
  `ForwardContext`. Mode-sensitive future layers will define their own typed context-bearing
  signatures.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>

## Known limitations

- Only fixed positive fully static feature dimensions are supported; there is no lazy or dynamic
  parameter materialization.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [dimensions glossary](../../../../glossary.md#dimension); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [parameter glossary](../../../../glossary.md#term-parameter); [materialization glossary](../../../../glossary.md#materialization))</span>
- Initialization policy is fixed and not caller-configurable beyond feature counts, bias presence,
  floating type, and random source.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- Parameter replacement is individual and not thread-safe. Callers must synchronize replacement
  with forward construction when a consistent multi-parameter snapshot matters.<span data-glossary-navigation="true"> ([thread-safe glossary](../../../../glossary.md#term-thread-safe); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- `forward` builds expression metadata and does not prove compiler capture, gradients, fusion,
  backend support, numerical values, or execution.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion); [backend glossary](../../../../glossary.md#backend))</span>
- There is no [checkpoint](../../../../glossary.md#term-checkpoint), [serialization](../../../../glossary.md#term-serialization), [device](../../../../glossary.md#term-device), [dtype](../../../../glossary.md#data-type--datatype)-conversion, or training-session convenience.

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `/root/nn_0005_implementation` read the required repository,
  architecture, planning, documentation-profile, NN foundation/initializer, Model linear, API,
  glossary, source, and test contracts before editing. It found no architecture or scope conflict.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LinearTest --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest`
  passed twice during implementation: the first stabilized production and the initial focused
  suite; the second passed after the API-surface and supplied-failure side-effect assertions were
  completed. The final focused reports contain 2 suites and 11 tests with zero failures, errors,
  or skips.
- The single final `./gradlew :extensions:nn:test` passed on 2026-08-13 (`BUILD SUCCESSFUL`, five
  actionable tasks: one executed and four up-to-date). XML reports contain 9 suites and 44 tests
  with zero failures, errors, or skips. No executable Java or tests changed afterward in this
  implementation context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed on the implementation draft
  (`BUILD SUCCESSFUL`, three actionable tasks: one executed and two up-to-date). The mandatory
  documentation context must still independently inspect and finalize the Javadocs, then run the
  authoritative final Javadoc command after its edits.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Focused coverage proves the exact three-constructor/three-method public surface, supplied state
  schema/name/order/reference retention, constructor validation order and no Tensor-ID effects,
  exact Glorot bounds and caller-source draw counts for all three floating types, deterministic
  zero bias and weight-before-bias IDs, source/array-limit non-rollback boundaries, exact primitive
  forward producers and ordered inputs, train/eval equivalence, replacement snapshots, and
  inherited Model failures without numerical execution claims.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Tensor glossary](../../../../glossary.md#tensor); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Implementation whitespace checks passed with no output for the tracked diff and every current
  untracked task-owned text file. `javap -public` showed exactly the planned three constructors and
  three declared methods, and the production import scan found no forbidden extension or
  execution/backend edge. Final Javadoc, Markdown, generated-page, exact seven-path scope,
  glossary, planning, no-change, and final whitespace evidence remain assigned to the mandatory
  clean documentation pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Clean documentation context `/root/nn_0005_docs` independently read the repository and
  architecture contracts, documentation rules and General/API-Javadoc/Planning/Example profiles,
  NN master/task history, ADR and dependency boundary, final source/tests, Model 0019D and<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency))</span>
  `Tensor.linear`, Tensor/Training APIs, glossary, and the complete task diff. It found no
  behavioral, architecture, dependency, or scope defect.
- The documentation pass reused the implementation context's final focused 2-suite/11-test and NN
  9-suite/44-test evidence because no executable Java or tests changed afterward. It did not
  repeat either successful suite.
- `./gradlew :extensions:nn:javadoc` passed after final documentation edits (`BUILD SUCCESSFUL`,
  three actionable tasks up-to-date). Inspection of generated `Linear.html` and
  `layers/package-summary.html` confirmed all three constructors, the three declared methods,
  `[outFeatures, inFeatures]` orientation, initialization/ownership, mode, replacement, failure,
  and no-execution boundaries. The drafted source and package Javadocs were already exact and
  required no correction.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `javap -public -classpath extensions/nn/build/classes/java/main:modules/model/build/classes/java/main
  io.github.pho001.synaptik.nn.layers.Linear` showed exactly the three planned constructors and
  [`weight`](../../../../glossary.md#term-weight), [`bias`](../../../../glossary.md#term-bias), and `forward`, with no extra declared public member. Production import and
  forbidden-package scans found only Model, NN initializer/module, and Java standard-library
  dependencies.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- `ruby /tmp/nn_0005_markdown_check.rb /Users/phujka/IdeaProjects/Synaptik docs/glossary.md
  docs/planning/extensions/nn/master-plan.md
  docs/planning/extensions/nn/tasks/0005-linear-layer.md` passed all three changed Markdown files,
  325 repository-local links, 292 heading anchors, balanced fences, final newlines, and trailing
  whitespace checks.
- Final scope inspection found exactly the authorized seven paths: two production Java files, two
  NN tests, glossary, NN master plan, and this task. All NN tasks 0001–0005 and their completion
  summaries are Complete, the master row is synchronized, and no later detailed NN task exists.
  A seven-file newline/trailing-whitespace check passed, and final `git diff --check` passed with no
  output.
- Glossary wording now identifies the current stateful [`Linear`](../../../../glossary.md#term-linear-meanings) separately from the Model linear
  projection and records its parameter orientation, supplied/fixed-initialized state, explicit
  caller RNG, exact forward delegation, mode-insensitivity, and replacement snapshot boundary.<span data-glossary-navigation="true"> ([linear projection glossary](../../../../glossary.md#linear-projection); [parameter glossary](../../../../glossary.md#term-parameter); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Architecture, ADR 0007, dependency rules and architecture tests require no change because the
  layer realizes their existing NN ownership and `model -> nn -> training` direction. Tensor API,
  Model 0019D, and Model code remain exact because forward only delegates to their unchanged
  linear convenience. Training API remains exact because the task adds no optimizer or update
  orchestration and uses the existing public <span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>`Parameter.replace` contract. Public API status,
  user training documentation, Model capabilities/plans, conformance/integration suites, Gradle,
  compiler/runtime/prepare/Engine, backends, other modules, CPU work, and the global roadmap remain
  unchanged because this task adds only model-expression composition inside the existing NN
  module and makes no execution or repository-wide status claim.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Implementation notes

- Added final public `io.github.pho001.synaptik.nn.layers.Linear` and package documentation without
  adding another production type, interface, operation, dependency, or execution path.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Caller-supplied constructors complete their exact weight/bias preflight before declaring either
  parameter and retain the exact Tensor references under <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor))</span>[`weight`](../../../../glossary.md#term-weight) then optional [`bias`](../../../../glossary.md#term-bias).
- Initialized construction validates caller inputs before Shape/Tensor work, delegates weight once
  to <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`ParameterInitializers.glorotUniform`, delegates requested bias once to
  `ParameterInitializers.zeros`, and retains no random source.
- `forward` rejects null input first, reads current bindings once, and delegates directly to the
  matching <span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>`Tensor.linear` overload. Tests inspect the existing visible PERMUTE/MATMUL/optional-ADD
  chain, mode irrelevance, and old-versus-new parameter snapshots.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- The implementation changed only the planned two production files, two tests, task, and NN master
  plan. The documentation-focused pass then added the planned glossary update and finalized the
  task/master evidence; Model, Training, architecture, build, CPU, global-roadmap, and execution-
  layer files remain unchanged.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Completion summary

- Completed changes: Implemented and focused-tested the planned [`Linear`](../../../../glossary.md#term-linear-meanings) layer API, supplied and
  initialized state policies, exact Model forward delegation, mode-insensitivity, and compatible
  replacement snapshots.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>
- Files changed or created: Exactly the planned two production Java files, two NN test files,
  glossary, NN master plan, and this task record.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and validation: Focused 2-suite/11-test and final NN 9-suite/44-test runs passed with no
  failures, errors, or skips; final NN Javadoc, generated-page inspection, Markdown, exact public
  surface, forbidden-import/dependency, seven-path scope, newline/whitespace, and<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` [validation](../../../../glossary.md#term-validation) passed.
- Documentation-agent review: Completed independently in clean context `/root/nn_0005_docs`
  using the General, API/Javadoc, Planning, and Example profiles; stable executable evidence was
  reused because no executable Java changed after the implementation run.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [run glossary](../../../../glossary.md#term-run))</span>
- Documentation impact: Finalized the glossary and NN planning evidence. Architecture, ADR 0007,
  dependency rules/tests, Tensor/Training/public APIs, Model contracts/plans, conformance and
  integration suites, Gradle, execution layers, backends, other modules, CPU work, and the global
  roadmap remain accurate and unchanged because <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Tensor glossary](../../../../glossary.md#tensor); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [backends glossary](../../../../glossary.md#backend); [modules glossary](../../../../glossary.md#term-module); [CPU glossary](../../../../glossary.md#term-cpu))</span>`Linear` realizes the existing model-only NN
  layer boundary by exact `Tensor.linear` delegation.
- [Javadoc](../../../../glossary.md#term-javadoc) review: Independently verified the final `Linear` and package Javadocs against source,
  tests, Model linear semantics, parameter replacement, mode, ownership, failures, and generated
  pages; no correction was needed.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: Extended the existing NN module and Linear projection entries with the current
  stateful <span data-glossary-navigation="true"> ([Linear projection glossary](../../../../glossary.md#linear-projection))</span>`Linear` ownership, initialization, forward, replacement-snapshot, and execution
  boundaries without redefining Model semantics.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete

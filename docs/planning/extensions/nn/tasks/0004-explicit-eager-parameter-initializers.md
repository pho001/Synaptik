# Task 0004: Explicit [Eager](../../../../glossary.md#term-eager-evaluation) [Parameter](../../../../glossary.md#term-parameter) Initializers

## Status

Complete

## Goal

Add one small, stateless [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-owned public initializer namespace that creates [eager](../../../../glossary.md#term-eager-evaluation) floating
[parameter](../../../../glossary.md#term-parameter) [Tensor](../../../../glossary.md#tensor) [leaves](../../../../glossary.md#term-leaf) through the existing [Model](../../../../glossary.md#term-model) construction APIs. It must cover deterministic
zero/one values, caller-parameterized normal/uniform values, and fixed [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Glorot](../../../../glossary.md#term-glorot-uniform) and
[Kaiming](../../../../glossary.md#term-kaiming)-[ReLU](../../../../glossary.md#term-relu) [weight](../../../../glossary.md#term-weight) policies without moving initialization policy into `Parameter` or retaining a
random source.

## [Scope](../../../../glossary.md#term-scope)

- Add the public final, field-free utility class
  `io.github.pho001.synaptik.nn.initialization.ParameterInitializers` with a private zero-argument
  constructor and exactly these eight public static methods:<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>

  ```java
  public static Tensor zeros(Shape shape, DataType dataType)
  public static Tensor ones(Shape shape, DataType dataType)
  public static Tensor normal(
          Shape shape,
          DataType dataType,
          double mean,
          double standardDeviation,
          RandomGenerator randomGenerator)
  public static Tensor uniform(
          Shape shape,
          DataType dataType,
          double lowerBoundInclusive,
          double upperBoundExclusive,
          RandomGenerator randomGenerator)
  public static Tensor glorotNormal(
          Shape weightShape,
          DataType dataType,
          RandomGenerator randomGenerator)
  public static Tensor glorotUniform(
          Shape weightShape,
          DataType dataType,
          RandomGenerator randomGenerator)
  public static Tensor kaimingReluNormal(
          Shape weightShape,
          DataType dataType,
          RandomGenerator randomGenerator)
  public static Tensor kaimingReluUniform(
          Shape weightShape,
          DataType dataType,
          RandomGenerator randomGenerator)
  ```

- Make every entry accept only [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16), require a fully static Shape,
  and return a fresh dense provenance-free Tensor with <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>`requiresGrad == true` and no Tensor label.
  The module/parameter name remains the state name; the initializer must not synthesize or accept a
  duplicate Tensor label in this first convenience surface.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Implement `zeros` and `ones` by one direct call to `TensorFactory.zeros` or
  `TensorFactory.ones`, respectively, with `Optional.empty()` and `requiresGrad == true`. Do not
  add a fill loop, source carrier, constant cache, or post-construction mutation.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [constant glossary](../../../../glossary.md#term-constant); [cache glossary](../../../../glossary.md#term-cache))</span>
- Implement `normal` and `uniform` by one direct call to `TensorRandoms.randomNormal` or
  `TensorRandoms.randomUniform`, respectively, with the exact supplied Shape, data type,
  distribution arguments, and `RandomGenerator`, plus `Optional.empty()` and
  `requiresGrad == true`. Retain all Model sampling, conversion, validation, source-advancement,
  allocation, identifier, and failure semantics unchanged.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation))</span>
- Interpret every fan-based `weightShape` as one fully static [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [Linear](../../../../glossary.md#term-linear-meanings) [weight](../../../../glossary.md#term-weight) in exact
  `[outFeatures, inFeatures]` orientation. Both extents must be strictly positive. Define
  `fanOut = weightShape.dimension(0)` and `fanIn = weightShape.dimension(1)`; neither element count,
  Tensor storage, nor a transposed view is used to infer them.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Use fixed [binary64](../../../../glossary.md#floating-point-precision-and-rounding) formulas and delegate [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) exactly once:

  | Method | Delegated [distribution](../../../../glossary.md#term-distribution) |
  |---|---|
  | `glorotNormal` | normal with [mean](../../../../glossary.md#term-mean) `0.0d` and [standard deviation](../../../../glossary.md#term-standard-deviation) `sqrt(2.0d / (fanIn + fanOut))` |
  | `glorotUniform` | uniform over `[-sqrt(6.0d / (fanIn + fanOut)), +sqrt(6.0d / (fanIn + fanOut)))` |
  | `kaimingReluNormal` | normal with mean `0.0d` and standard deviation `sqrt(2.0d / fanIn)` |
  | `kaimingReluUniform` | uniform over `[-sqrt(6.0d / fanIn), +sqrt(6.0d / fanIn))` |

  Convert each positive `long` fan to binary64 before addition or division so `fanIn + fanOut`
  cannot overflow as integer arithmetic. Use <span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>`Math.sqrt` once for the selected scale. The current
  Model APIs then apply their exact binary64 sample transformation and FLOAT64/FLOAT32/BFLOAT16
  conversion semantics.<span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Validate fan-based calls before any source draw, Tensor allocation, or Tensor identifier
  allocation. Null checks occur in parameter order: `weightShape`, `dataType`, then
  `randomGenerator`. Then require a floating [data type](../../../../glossary.md#data-type--datatype), a fully static rank-two [Shape](../../../../glossary.md#shape), positive
  `outFeatures`, and positive `inFeatures`, in that order. After this local preflight, delegate
  once to the matching Model random method; Model remains authoritative for checked element-count,
  Java-array-size, dense-layout, generator-failure, and allocation/identifier effects.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [layout glossary](../../../../glossary.md#layout))</span>
- Add package Javadoc that distinguishes eager initialization from module binding ownership and
  deferred graph RNG state.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [eager glossary](../../../../glossary.md#term-eager-evaluation); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Add focused surface, metadata, value/formula, random-source ownership, validation, and
  side-effect tests. Use scripted <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`RandomGenerator` implementations to prove the exact delegated
  Gaussian or bounded-uniform method, arguments, and call count without introducing a production
  seed or generator choice.<span data-glossary-navigation="true"> ([Gaussian glossary](../../../../glossary.md#term-gaussian-distribution); [seed glossary](../../../../glossary.md#term-seed))</span>
- After implementation and focused testing, use a separate documentation-focused agent/thread to
  finalize all new public Javadoc, package Javadoc, glossary impact, this task record, and the NN
  master-plan status.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any initializer method on [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), or [`Module`](../../../../glossary.md#term-module); a `Parameter` continues to retain
  one caller-supplied Tensor without knowing how it was created.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- Default data types, default seeds, hidden/global/thread-local generators, retained generators,
  generator factories, generator splitting, synchronization, resetting, closing, or serialization.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [synchronization glossary](../../../../glossary.md#term-synchronization); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), a graph-random operation, deferred random sampling, dropout, or module-owned RNG
  state. <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`GraphRngState` represents expression-[graph](../../../../glossary.md#graph) state and is not used by [eager](../../../../glossary.md#term-eager-evaluation) initialization.
- Configurable gain, activation, fan mode, truncation, orthogonal/identity initialization,
  sparse initialization, bias-specific random policy, convolution fan calculation, or Shapes
  other than rank-two for fan-based methods.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [identity glossary](../../../../glossary.md#term-identity-meanings); [bias glossary](../../../../glossary.md#term-bias); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Shapes glossary](../../../../glossary.md#shape); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Integral or [BOOL](../../../../glossary.md#data-type--datatype) parameter initializers, caller-selected `requiresGrad`, Tensor labels,
  initializer objects/interfaces, registries, annotations, mutable policies, or service lookup.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- A [`Linear`](../../../../glossary.md#term-linear-meanings) layer, module construction changes, buffer initialization policy, optimizer,
  checkpoint, state dictionary, serialization, or parameter update orchestration.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Tensor-operation semantics, numerical execution, compiler, runtime, prepare, Engine, concrete
  backend, Gradle/dependency, architecture-test, architecture-contract, explanatory-architecture,
  or global-roadmap changes.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: NN ownership, Model composition,
  immutable Tensor identity/provenance, and extension dependency direction.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Model glossary](../../../../glossary.md#term-model); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>: current eager constants, normal/uniform random
  creation, Linear orientation, and eager-versus-graph RNG distinction.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [constants glossary](../../../../glossary.md#term-constant); [Linear glossary](../../../../glossary.md#term-linear-meanings); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [NN master plan](../master-plan.md).
- [Tasks 0001](0001-module-parameter-buffer-and-forward-context-foundation.md),
  [0002](0002-module-tree-ownership-and-recursive-mode-propagation.md), and
  [0003](0003-validated-parameter-and-buffer-binding-replacement.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and its
  [Planning profile](../../../../developer-guide/documentation/planning-style.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. The initializer surface composes
  current Model construction contracts and must not copy them into NN or introduce any other
  module dependency.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [`Parameter`](../../../../glossary.md#term-parameter) remains an NN-owned named binding, not a Tensor subtype or initialization-policy
  owner. Initializer calls return ordinary eager Model Tensor leaves which a caller may then
  supply to <span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Tensor glossary](../../../../glossary.md#tensor); [owner glossary](../../../../glossary.md#term-owner))</span>`Module.parameter(...)`.
- Tensor identity, descriptor, provenance, host-storage association, and gradient eligibility
  remain Model state. NN creates no alternative Tensor, storage, scalar-conversion, or random
  representation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Caller-owned `RandomGenerator` is transient explicit input. NN neither retains it nor assumes a
  seed, algorithm, thread-safety, serialization, or cross-provider/Java-version sequence.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- [Eager](../../../../glossary.md#term-eager-evaluation) [host](../../../../glossary.md#term-host)-data initialization is separate from [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) and random Tensor expressions.
  This task consumes samples immediately through <span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`TensorRandoms` and constructs no [graph](../../../../glossary.md#graph) [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) edge.
- The authorized NN parallel exception is implementation-order only. This task stays in the NN
  module and NN planning/glossary files and must not touch the dirty CPU task/master-plan/global-
  roadmap work, architecture, build graph, or dependency enforcement.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — existing [`Tensor`](../../../../glossary.md#tensor), `TensorFactory`, and
  `TensorRandoms` public contracts only.
- `io.github.pho001.synaptik.model.shape` — existing [immutable](../../../../glossary.md#term-immutable) [`Shape`](../../../../glossary.md#shape) and static-dimension
  inspection.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- `io.github.pho001.synaptik.model.datatype` — existing [`DataType`](../../../../glossary.md#data-type--datatype) [floating classification](../../../../glossary.md#floating-classification).
- `io.github.pho001.synaptik.nn.module` — unchanged downstream consumer boundary; no initializer
  type is placed there.

Packages added or changed:

- `io.github.pho001.synaptik.nn.initialization` — public stateless eager parameter Tensor
  creation. It owns NN initialization policy but no parameter binding, RNG lifecycle, or execution.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [parameter glossary](../../../../glossary.md#term-parameter); [binding glossary](../../../../glossary.md#term-binding); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

Type placement:

- `io.github.pho001.synaptik.nn.initialization.ParameterInitializers` — the sole public initializer
  namespace; its name makes the fixed gradient-enabled, unlabeled parameter-leaf purpose explicit.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- `io.github.pho001.synaptik.nn.initialization.ParameterInitializersTest` — black-box public
  surface and generic zero/one/normal/uniform contract tests in the production package.
- `io.github.pho001.synaptik.nn.initialization.LinearWeightInitializersTest` — [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)
  `[outFeatures, inFeatures]`, exact scale/formula, conversion, and fan-[validation](../../../../glossary.md#term-validation) tests.

No type is added to the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) root or [`module`](../../../../glossary.md#term-module) package. The tests mirror the public production
package; they need no package-private production seam.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializersTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/LinearWeightInitializersTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/glossary.md` — extend the existing NN and eager-random entries with the current
  initializer boundary and its distinction from <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>[`Parameter`](../../../../glossary.md#term-parameter) and [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate).
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

`docs/api/tensor-api.md`, architecture documents, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, and architecture tests are explicit
review/no-change candidates: the task consumes but does not modify [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) APIs, [dependency
direction](../../../../glossary.md#java-platform-and-build-tools), or architecture rules.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- two production Java files;
- two [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- the three documentation/[planning](../../../../glossary.md#term-planning) files listed above.

If implementation needs another public type, an initializer interface/object hierarchy, a new
[Model](../../../../glossary.md#term-model) [API](../../../../glossary.md#term-application-programming-interface-api), a [`Parameter`](../../../../glossary.md#term-parameter)/[`Module`](../../../../glossary.md#term-module) change, a new module [dependency](../../../../glossary.md#term-dependency), [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/general fan policy,
or more files, stop and propose a separate follow-up task.

## Acceptance criteria

- The exact eight-method static surface exists only on final field-free
  `ParameterInitializers`; it has one private zero-argument constructor, no public/protected
  constructor, no fields, no nested type, and no overload, alias, default, or stateful object API.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [alias glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Every successful call returns a fresh provenance-free, unlabeled, dense-contiguous floating
  Tensor that retains the exact supplied Shape and data type and has <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [contiguous glossary](../../../../glossary.md#term-contiguous); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`requiresGrad == true`.
  Repeated calls do not share Tensor identity, storage, or backing arrays.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `zeros` and `ones` delegate to the matching `TensorFactory` method and produce exact typed
  zero/one values for FLOAT64, FLOAT32, and BFLOAT16 without sampling or an NN-owned fill path.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- `normal` and `uniform` forward the exact caller-owned generator and exact distribution
  arguments to the one matching <span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>`TensorRandoms` call. Their output values, call count, conversion,
  validation, and late-failure effects match Model semantics; the NN class retains no source or
  result.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Model glossary](../../../../glossary.md#term-model))</span>
- Each fan method accepts exactly a fully static rank-two positive-extent weight Shape and
  interprets it as <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [weight glossary](../../../../glossary.md#term-weight))</span>`[fanOut, fanIn]`. Dynamic, scalar, rank-one, rank-three, zero-fan, non-floating,
  and null arguments fail in the documented order before any source draw or Tensor ID allocation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank))</span>
- Scripted-source tests prove each fan method's exact binary64 mean, standard deviation, lower
  bound, or upper bound from the formulas in Scope and prove one Model source call per logical
  row-major element. FLOAT32 and BFLOAT16 conversion coverage prevents an accidental FLOAT64-only
  implementation.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [Scope glossary](../../../../glossary.md#term-scope); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element))</span>
- Glorot exposes only fixed unit-gain fan-average normal/uniform policies. Kaiming exposes only
  fixed ReLU-gain, fan-in normal/uniform policies. There is no ambiguous <span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming); [ReLU glossary](../../../../glossary.md#term-relu); [fan-in glossary](../../../../glossary.md#term-fan-in))</span>`kaimingNormal` name,
  configurable nonlinearity/gain/mode, alias named Xavier, or convolution fan inference.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [inference glossary](../../../../glossary.md#term-inference))</span>
- No [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), [`Module`](../../../../glossary.md#term-module), Model, build, dependency, CPU, Engine, compiler, runtime,
  prepare, backend, architecture, architecture-test, or global-roadmap file changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- All new and affected public/package APIs have meaningful complete Javadoc for purpose,
  ownership, distribution, orientation, type/Shape restrictions, labels, gradient intent,
  conversion, failures, source advancement, and side effects.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- A separate documentation-focused agent pass finalizes the Javadoc, package documentation,
  glossary impact, links, terminology, planning evidence, and generated Javadoc in this same
  overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs one final affected-[module](../../../../glossary.md#term-module) command after executable code stabilizes:

```bash
./gradlew :extensions:nn:test
```

The focused suite must cover the exact public surface, typed [constant](../../../../glossary.md#term-constant) values, exact random
delegation/formulas, [Shape](../../../../glossary.md#shape)/data-type/result [metadata](../../../../glossary.md#term-metadata), [provenance](../../../../glossary.md#provenance)/storage independence, null and
fan-validation order, no-draw early failures, generator exceptions, and source call counts.

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass also validates local Markdown links and anchors in the changed [planning](../../../../glossary.md#term-planning)
and glossary files, inspects the generated initializer/package pages, confirms exact package/type
placement, verifies the seven-file maximum and no unrelated [CPU](../../../../glossary.md#term-cpu)/global-roadmap paths, and records
the reasoned no-change conclusions for [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), architecture documents, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, and
architecture tests. It reuses the successful implementation test unless it changes executable
Java behavior, in which case it reruns that focused module command and records why.

Repository-wide and architecture-test validation are deferred to the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one module's [leaf](../../../../glossary.md#term-leaf) API without changing [Gradle](../../../../glossary.md#term-gradle), module boundaries, or
[dependency](../../../../glossary.md#term-dependency) rules.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001, 0002, and 0003 are Complete.
- [Model](../../../../glossary.md#term-model) `TensorFactory.zeros/ones`, `TensorRandoms.randomNormal/randomUniform`, [`Shape`](../../../../glossary.md#shape),
  [`DataType`](../../../../glossary.md#data-type--datatype), [host](../../../../glossary.md#term-host)-storage, [descriptor](../../../../glossary.md#term-descriptor), and [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) contracts are Complete and accepted.
- Model `Tensor.linear` fixes the future [Linear](../../../../glossary.md#term-linear-meanings) [weight](../../../../glossary.md#term-weight) orientation as
  `[outFeatures, inFeatures]`.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force; this
  task's NN-only Java/docs files do not overlap the active dirty CPU 0007 planning work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0005 (required): add [`Linear`](../../../../glossary.md#term-linear-meanings) over the stable module state and these initializers. It may use
  fixed zero bias and one selected weight policy, while preserving caller-supplied Tensor
  construction as an explicit path.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [bias glossary](../../../../glossary.md#term-bias); [weight glossary](../../../../glossary.md#term-weight); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Any configurable gain, activation, fan-in/fan-out mode, convolution fan geometry, bias-random
  policy, orthogonal/sparse policy, or initializer object abstraction is optional future work and
  requires its own concrete consumer and task.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [fan-in glossary](../../../../glossary.md#term-fan-in); [fan-out glossary](../../../../glossary.md#term-fan-out); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Checkpoint/state-dictionary and training optimizer work remain separately owned and must not
  infer persistence or update semantics from initializer calls.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [optimizer glossary](../../../../glossary.md#term-optimizer); [persistence glossary](../../../../glossary.md#term-persistence))</span>

## Architecture impact

Expected impact: None.

This task implements the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) allowance for conveniences composed from [Model](../../../../glossary.md#term-model) semantics and
keeps the current `modules/model -> extensions/nn -> extensions/training` direction. If
implementation needs a new [dependency](../../../../glossary.md#term-dependency), changes [Parameter](../../../../glossary.md#term-parameter) [ownership](../../../../glossary.md#term-ownership-meanings), uses [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), or
requires Model/architecture changes, stop and report the conflict rather than expanding this task.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, and
docs/planning/extensions/nn/tasks/0004-explicit-eager-parameter-initializers.md in full.
Implement that task exactly as specified. Do not implement out-of-scope work, touch dirty CPU or
global-roadmap files, commit, or push. Stop and report any architecture or scope conflict.

After executable implementation and focused module validation, hand the complete diff and exact
test evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow the documentation rules and independently finalize affected Javadoc, package documentation,
glossary impact, planning evidence, and documentation validation in the same overall change. It
must not repeat successful Java tests unless executable behavior changes or it records a concrete
risk.

Update this task with implementation notes, local decisions, validation evidence including the
documentation-agent pass, and completion summary. Do not mark it Complete before that pass
finishes.
```

## Local decisions

- The four generic entries remain direct one-call delegations. In particular, deterministic
  constants rely on <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>`requiresGrad == true` plus the existing Model descriptor validation to reject
  non-floating types; NN adds no duplicate fill or descriptor path.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Fan validation is one shared private preflight with exact null order followed by floating type,
  fully static shape, rank two, positive <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [shape glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank))</span>`outFeatures`, and positive `inFeatures`. Each public fan
  method then converts the selected positive `long` extents to `double`, calls `Math.sqrt` once,
  and delegates once through the corresponding generic entry.
- Focused tests use scripted sources whose unrelated primitive methods fail. Normal output values
  prove the selected binary64 scale and <span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`nextGaussian()` route; bounded sources separately retain
  every origin/bound pair to prove the exact `nextDouble(origin, bound)` route.

## Known limitations

- Fan-based policies accept only positive fully static rank-two Linear weights. Callers needing a
  zero-feature Tensor may use the deterministic generic initializer or supply a Tensor directly,
  but no fan formula is defined for a zero denominator.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Linear glossary](../../../../glossary.md#term-linear-meanings); [weights glossary](../../../../glossary.md#term-weight); [feature glossary](../../../../glossary.md#term-feature); [Tensor glossary](../../../../glossary.md#tensor))</span>
- All results are unlabeled gradient-enabled parameter leaves. Frozen/non-trainable Tensor leaves,
  labeled Tensor diagnostics, and buffer-specific creation continue to use Model APIs directly.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [Model glossary](../../../../glossary.md#term-model))</span>
- Glorot and Kaiming policies are fixed rather than configurable; convolution and other layout-
  specific fan calculations are not inferred.<span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout))</span>
- [Reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) is bounded by the existing caller-owned `RandomGenerator` contract and has no
  cross-algorithm, provider, Java-version, serialization, or concurrent-use guarantee.<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `/root/nn_0004_implementation` ran an initial
  `./gradlew :extensions:nn:test` after adding the executable surface and tests; it passed on
  2026-08-13 with 5 actionable tasks (3 executed, 2 up-to-date). The result contained 25 NN tests
  across 6 suites, including 9 new initializer tests, with zero failures, errors, or skips.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- The same implementation context ran the single final `./gradlew :extensions:nn:test` after
  executable code stabilized and the last test-only cleanup; it passed on 2026-08-13 with 5
  actionable tasks (2 executed, 3 up-to-date). The final result again contained 25 NN tests across
  6 suites, including 9 initializer tests, with zero failures, errors, or skips.
- `git diff --check` plus `git diff --no-index --check /dev/null <new-file>` for each of the five
  untracked task-owned files passed with no output after the stable test result. The mandatory
  clean documentation pass remains.
- Repository-wide and architecture-test validation remain deferred exactly as specified because
  no build edge, dependency rule, architecture boundary, or shared module changed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Clean documentation context `/root/nn_0004_docs` independently reviewed the complete NN 0004
  diff against `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, planning guide,
  documentation rules, General/API-and-Javadoc/Planning/Example profiles, NN master plan and tasks
  0001–0004, ADR 0007, dependency rules, glossary, Tensor API, current Model construction/type/
  Shape contracts, and all final NN source and tests. It found no executable behavioral defect and
  changed only Javadoc, package documentation, glossary, and NN planning records.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The documentation pass finalized all eight initializer method contracts and package/type
  documentation for exact Shape/type retention, floating and fully static restrictions,
  `requiresGrad == true`, absent labels and [provenance](../../../../glossary.md#provenance), `[outFeatures, inFeatures]` orientation,
  all four binary64 Glorot/Kaiming-ReLU formulas, source calls and conversion, caller-owned<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming); [ReLU glossary](../../../../glossary.md#term-relu))</span>
  `RandomGenerator` [lifecycle](../../../../glossary.md#lifecycle), failure side effects, no default or retained source, no
  [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), and no [`Parameter`](../../../../glossary.md#term-parameter) initialization policy.
- `./gradlew :extensions:nn:javadoc` — passed on 2026-08-13 after final Javadoc edits (3 actionable
  tasks: 2 executed, 1 up-to-date). Generated `ParameterInitializers` and initialization-package
  pages were inspected; they expose exactly the eight planned methods and render the formulas,
  restrictions, ownership, lifecycle, result metadata, failure, and boundary text.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Targeted local Markdown validation passed for the NN master plan, task, and glossary: all
  relative file links resolved, both new `GraphRngState` links matched the existing glossary
  heading anchor, and every fenced-code-block count was balanced.
- Exact scope inspection passed: the task owns only its two production files, two test files,
  glossary, NN master plan, and this task record. Package/type placement matches the plan. The
  dirty CPU 0007 source/tests/task/master-plan and global-roadmap changes remain unrelated and
  untouched by this pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Final `git diff --check` plus `git diff --no-index --check /dev/null <new-file>` for each of the
  five untracked task-owned files passed with no output. The successful implementation
  `./gradlew :extensions:nn:test` evidence was reused because the documentation pass changed no
  executable Java behavior.
- `docs/api/tensor-api.md` and the [Model](../../../../glossary.md#term-model) `TensorFactory`, `TensorRandoms`, [`DataType`](../../../../glossary.md#data-type--datatype), and [`Shape`](../../../../glossary.md#shape)
  contracts were reviewed with no change: NN delegates their existing eager construction,
  sampling, conversion, validation, allocation, and identifier semantics without modifying the
  public Model surface. <span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>`ARCHITECTURE.md`, focused architecture documents, ADR 0007, training-
  graph documentation, dependency rules, and architecture tests also require no change because
  task 0004 adds no module edge, ownership change, Tensor lifecycle state, graph-random behavior,
  training behavior, or execution-layer dependency.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>

## Implementation notes

- Added the field-free final `ParameterInitializers` namespace with exactly the planned eight
  public static methods and no overload, alias, default source, retained state, or nested type.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Added direct eager Model delegation for typed zero/one and caller-source normal/uniform leaves,
  plus exact rank-two <span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [Model glossary](../../../../glossary.md#term-model); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[outFeatures, inFeatures]` [Glorot](../../../../glossary.md#term-glorot-uniform) and fixed [ReLU](../../../../glossary.md#term-relu)/[fan-in](../../../../glossary.md#term-fan-in) [Kaiming](../../../../glossary.md#term-kaiming) formulas.
- Added package documentation and complete public Javadocs for eager ownership, binding and graph-
  RNG boundaries, types, shapes, distributions, conversions, failures, source advancement,
  allocation, and identifier effects; the independent documentation-focused pass finalized them.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [shapes glossary](../../../../glossary.md#shape); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Added the two planned focused test classes covering exact API shape, leaf metadata and storage
  independence, typed values and conversion, exact source methods/arguments/call counts, formulas,
  validation order, no-draw/no-ID early failure, and generator failure effects.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [metadata glossary](../../../../glossary.md#term-metadata); [arguments glossary](../../../../glossary.md#term-argument); [validation glossary](../../../../glossary.md#term-validation))</span>

## Completion summary

- Completed changes: Implemented the planned eager floating parameter initializer surface and
  focused executable tests without changing module-owned bindings or Model/execution behavior.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [module glossary](../../../../glossary.md#term-module); [bindings glossary](../../../../glossary.md#term-binding); [Model glossary](../../../../glossary.md#term-model))</span>
- Files changed or created: two planned production files, two planned NN test files, the glossary,
  NN master plan, and this task record.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Tests and [validation](../../../../glossary.md#term-validation): Initial and final `./gradlew :extensions:nn:test` runs passed with 25 tests
  and no failures, errors, or skips; final NN Javadoc, generated-page inspection, Markdown links/
  anchors/fences, exact seven-file scope, package placement, and whitespace checks passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/nn_0004_docs` independently finalized the API/Javadoc,
  package documentation, glossary, and planning evidence under the General, API-and-Javadoc,
  Planning, and Example profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Updated the existing NN and eager-random glossary entries. Tensor API,
  architecture documents, ADR 0007, training graph, dependency rules, and architecture tests need
  no change because no Model API, ownership, dependency, graph-random, training, or execution
  contract changed.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [training graph glossary](../../../../glossary.md#training-graph); [dependency glossary](../../../../glossary.md#term-dependency); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [graph glossary](../../../../glossary.md#graph))</span>
- Javadoc review: All eight methods and the initialization package now document exact result
  metadata, formulas, orientation, restrictions, source lifecycle and draws, failures, and the<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  [`Parameter`](../../../../glossary.md#term-parameter)/[`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate) boundaries; final [Javadoc](../../../../glossary.md#term-javadoc) generation passed.
- Glossary impact: Updated the two existing relevant entries; no separate reusable term or new
  heading was necessary.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete

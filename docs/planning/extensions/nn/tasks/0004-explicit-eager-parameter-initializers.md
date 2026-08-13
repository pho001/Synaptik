# Task 0004: Explicit Eager Parameter Initializers

## Status

Complete

## Goal

Add one small, stateless NN-owned public initializer namespace that creates eager floating
parameter Tensor leaves through the existing Model construction APIs. It must cover deterministic
zero/one values, caller-parameterized normal/uniform values, and fixed rank-two Glorot and
Kaiming-ReLU weight policies without moving initialization policy into `Parameter` or retaining a
random source.

## Scope

- Add the public final, field-free utility class
  `io.github.pho001.synaptik.nn.initialization.ParameterInitializers` with a private zero-argument
  constructor and exactly these eight public static methods:

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

- Make every entry accept only `FLOAT64`, `FLOAT32`, or `BFLOAT16`, require a fully static Shape,
  and return a fresh dense provenance-free Tensor with `requiresGrad == true` and no Tensor label.
  The module/parameter name remains the state name; the initializer must not synthesize or accept a
  duplicate Tensor label in this first convenience surface.
- Implement `zeros` and `ones` by one direct call to `TensorFactory.zeros` or
  `TensorFactory.ones`, respectively, with `Optional.empty()` and `requiresGrad == true`. Do not
  add a fill loop, source carrier, constant cache, or post-construction mutation.
- Implement `normal` and `uniform` by one direct call to `TensorRandoms.randomNormal` or
  `TensorRandoms.randomUniform`, respectively, with the exact supplied Shape, data type,
  distribution arguments, and `RandomGenerator`, plus `Optional.empty()` and
  `requiresGrad == true`. Retain all Model sampling, conversion, validation, source-advancement,
  allocation, identifier, and failure semantics unchanged.
- Interpret every fan-based `weightShape` as one fully static rank-two Linear weight in exact
  `[outFeatures, inFeatures]` orientation. Both extents must be strictly positive. Define
  `fanOut = weightShape.dimension(0)` and `fanIn = weightShape.dimension(1)`; neither element count,
  Tensor storage, nor a transposed view is used to infer them.
- Use fixed binary64 formulas and delegate sampling exactly once:

  | Method | Delegated distribution |
  |---|---|
  | `glorotNormal` | normal with mean `0.0d` and standard deviation `sqrt(2.0d / (fanIn + fanOut))` |
  | `glorotUniform` | uniform over `[-sqrt(6.0d / (fanIn + fanOut)), +sqrt(6.0d / (fanIn + fanOut)))` |
  | `kaimingReluNormal` | normal with mean `0.0d` and standard deviation `sqrt(2.0d / fanIn)` |
  | `kaimingReluUniform` | uniform over `[-sqrt(6.0d / fanIn), +sqrt(6.0d / fanIn))` |

  Convert each positive `long` fan to binary64 before addition or division so `fanIn + fanOut`
  cannot overflow as integer arithmetic. Use `Math.sqrt` once for the selected scale. The current
  Model APIs then apply their exact binary64 sample transformation and FLOAT64/FLOAT32/BFLOAT16
  conversion semantics.
- Validate fan-based calls before any source draw, Tensor allocation, or Tensor identifier
  allocation. Null checks occur in parameter order: `weightShape`, `dataType`, then
  `randomGenerator`. Then require a floating data type, a fully static rank-two Shape, positive
  `outFeatures`, and positive `inFeatures`, in that order. After this local preflight, delegate
  once to the matching Model random method; Model remains authoritative for checked element-count,
  Java-array-size, dense-layout, generator-failure, and allocation/identifier effects.
- Add package Javadoc that distinguishes eager initialization from module binding ownership and
  deferred graph RNG state.
- Add focused surface, metadata, value/formula, random-source ownership, validation, and
  side-effect tests. Use scripted `RandomGenerator` implementations to prove the exact delegated
  Gaussian or bounded-uniform method, arguments, and call count without introducing a production
  seed or generator choice.
- After implementation and focused testing, use a separate documentation-focused agent/thread to
  finalize all new public Javadoc, package Javadoc, glossary impact, this task record, and the NN
  master-plan status.

## Out of scope

- Any initializer method on `Parameter`, `Buffer`, or `Module`; a `Parameter` continues to retain
  one caller-supplied Tensor without knowing how it was created.
- Default data types, default seeds, hidden/global/thread-local generators, retained generators,
  generator factories, generator splitting, synchronization, resetting, closing, or serialization.
- `GraphRngState`, a graph-random operation, deferred random sampling, dropout, or module-owned RNG
  state. `GraphRngState` represents expression-graph state and is not used by eager initialization.
- Configurable gain, activation, fan mode, truncation, orthogonal/identity initialization,
  sparse initialization, bias-specific random policy, convolution fan calculation, or Shapes
  other than rank-two for fan-based methods.
- Integral or BOOL parameter initializers, caller-selected `requiresGrad`, Tensor labels,
  initializer objects/interfaces, registries, annotations, mutable policies, or service lookup.
- A `Linear` layer, module construction changes, buffer initialization policy, optimizer,
  checkpoint, state dictionary, serialization, or parameter update orchestration.
- Tensor-operation semantics, numerical execution, compiler, runtime, prepare, Engine, concrete
  backend, Gradle/dependency, architecture-test, architecture-contract, explanatory-architecture,
  or global-roadmap changes.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN ownership, Model composition,
  immutable Tensor identity/provenance, and extension dependency direction.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [Tensor API](../../../../api/tensor-api.md): current eager constants, normal/uniform random
  creation, Linear orientation, and eager-versus-graph RNG distinction.
- [NN master plan](../master-plan.md).
- [Tasks 0001](0001-module-parameter-buffer-and-forward-context-foundation.md),
  [0002](0002-module-tree-ownership-and-recursive-mode-propagation.md), and
  [0003](0003-validated-parameter-and-buffer-binding-replacement.md).
- [Planning Guide](../../../planning-guide.md).
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and its
  [Planning profile](../../../../developer-guide/documentation/planning-style.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. The initializer surface composes
  current Model construction contracts and must not copy them into NN or introduce any other
  module dependency.
- `Parameter` remains an NN-owned named binding, not a Tensor subtype or initialization-policy
  owner. Initializer calls return ordinary eager Model Tensor leaves which a caller may then
  supply to `Module.parameter(...)`.
- Tensor identity, descriptor, provenance, host-storage association, and gradient eligibility
  remain Model state. NN creates no alternative Tensor, storage, scalar-conversion, or random
  representation.
- Caller-owned `RandomGenerator` is transient explicit input. NN neither retains it nor assumes a
  seed, algorithm, thread-safety, serialization, or cross-provider/Java-version sequence.
- Eager host-data initialization is separate from `GraphRngState` and random Tensor expressions.
  This task consumes samples immediately through `TensorRandoms` and constructs no graph RNG edge.
- The authorized NN parallel exception is implementation-order only. This task stays in the NN
  module and NN planning/glossary files and must not touch the dirty CPU task/master-plan/global-
  roadmap work, architecture, build graph, or dependency enforcement.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — existing `Tensor`, `TensorFactory`, and
  `TensorRandoms` public contracts only.
- `io.github.pho001.synaptik.model.shape` — existing immutable `Shape` and static-dimension
  inspection.
- `io.github.pho001.synaptik.model.datatype` — existing `DataType` floating classification.
- `io.github.pho001.synaptik.nn.module` — unchanged downstream consumer boundary; no initializer
  type is placed there.

Packages added or changed:

- `io.github.pho001.synaptik.nn.initialization` — public stateless eager parameter Tensor
  creation. It owns NN initialization policy but no parameter binding, RNG lifecycle, or execution.

Type placement:

- `io.github.pho001.synaptik.nn.initialization.ParameterInitializers` — the sole public initializer
  namespace; its name makes the fixed gradient-enabled, unlabeled parameter-leaf purpose explicit.
- `io.github.pho001.synaptik.nn.initialization.ParameterInitializersTest` — black-box public
  surface and generic zero/one/normal/uniform contract tests in the production package.
- `io.github.pho001.synaptik.nn.initialization.LinearWeightInitializersTest` — rank-two
  `[outFeatures, inFeatures]`, exact scale/formula, conversion, and fan-validation tests.

No type is added to the NN root or `module` package. The tests mirror the public production
package; they need no package-private production seam.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializersTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/LinearWeightInitializersTest.java`.

Expected documentation and planning files:

- `docs/glossary.md` — extend the existing NN and eager-random entries with the current
  initializer boundary and its distinction from `Parameter` and `GraphRngState`.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

`docs/api/tensor-api.md`, architecture documents, ADR 0007, and architecture tests are explicit
review/no-change candidates: the task consumes but does not modify Model Tensor APIs, dependency
direction, or architecture rules.

## Maximum scope

This task may create or modify at most:

- two production Java files;
- two NN test files; and
- the three documentation/planning files listed above.

If implementation needs another public type, an initializer interface/object hierarchy, a new
Model API, a `Parameter`/`Module` change, a new module dependency, convolution/general fan policy,
or more files, stop and propose a separate follow-up task.

## Acceptance criteria

- The exact eight-method static surface exists only on final field-free
  `ParameterInitializers`; it has one private zero-argument constructor, no public/protected
  constructor, no fields, no nested type, and no overload, alias, default, or stateful object API.
- Every successful call returns a fresh provenance-free, unlabeled, dense-contiguous floating
  Tensor that retains the exact supplied Shape and data type and has `requiresGrad == true`.
  Repeated calls do not share Tensor identity, storage, or backing arrays.
- `zeros` and `ones` delegate to the matching `TensorFactory` method and produce exact typed
  zero/one values for FLOAT64, FLOAT32, and BFLOAT16 without sampling or an NN-owned fill path.
- `normal` and `uniform` forward the exact caller-owned generator and exact distribution
  arguments to the one matching `TensorRandoms` call. Their output values, call count, conversion,
  validation, and late-failure effects match Model semantics; the NN class retains no source or
  result.
- Each fan method accepts exactly a fully static rank-two positive-extent weight Shape and
  interprets it as `[fanOut, fanIn]`. Dynamic, scalar, rank-one, rank-three, zero-fan, non-floating,
  and null arguments fail in the documented order before any source draw or Tensor ID allocation.
- Scripted-source tests prove each fan method's exact binary64 mean, standard deviation, lower
  bound, or upper bound from the formulas in Scope and prove one Model source call per logical
  row-major element. FLOAT32 and BFLOAT16 conversion coverage prevents an accidental FLOAT64-only
  implementation.
- Glorot exposes only fixed unit-gain fan-average normal/uniform policies. Kaiming exposes only
  fixed ReLU-gain, fan-in normal/uniform policies. There is no ambiguous `kaimingNormal` name,
  configurable nonlinearity/gain/mode, alias named Xavier, or convolution fan inference.
- No `Parameter`, `Buffer`, `Module`, Model, build, dependency, CPU, Engine, compiler, runtime,
  prepare, backend, architecture, architecture-test, or global-roadmap file changes.
- All new and affected public/package APIs have meaningful complete Javadoc for purpose,
  ownership, distribution, orientation, type/Shape restrictions, labels, gradient intent,
  conversion, failures, source advancement, and side effects.
- A separate documentation-focused agent pass finalizes the Javadoc, package documentation,
  glossary impact, links, terminology, planning evidence, and generated Javadoc in this same
  overall change.

## Tests / validation

Implementation pass runs one final affected-module command after executable code stabilizes:

```bash
./gradlew :extensions:nn:test
```

The focused suite must cover the exact public surface, typed constant values, exact random
delegation/formulas, Shape/data-type/result metadata, provenance/storage independence, null and
fan-validation order, no-draw early failures, generator exceptions, and source call counts.

Documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass also validates local Markdown links and anchors in the changed planning
and glossary files, inspects the generated initializer/package pages, confirms exact package/type
placement, verifies the seven-file maximum and no unrelated CPU/global-roadmap paths, and records
the reasoned no-change conclusions for Tensor API, architecture documents, ADR 0007, and
architecture tests. It reuses the successful implementation test unless it changes executable
Java behavior, in which case it reruns that focused module command and records why.

Repository-wide and architecture-test validation are deferred to the NN capability checkpoint or
CI because this task changes one module's leaf API without changing Gradle, module boundaries, or
dependency rules.

## Dependencies

- NN 0001, 0002, and 0003 are Complete.
- Model `TensorFactory.zeros/ones`, `TensorRandoms.randomNormal/randomUniform`, `Shape`,
  `DataType`, host-storage, descriptor, and eager leaf contracts are Complete and accepted.
- Model `Tensor.linear` fixes the future Linear weight orientation as
  `[outFeatures, inFeatures]`.
- The user-authorized NN parallel exception recorded in the NN master plan remains in force; this
  task's NN-only Java/docs files do not overlap the active dirty CPU 0007 planning work.

## Follow-up tasks

- NN 0005 (required): add `Linear` over the stable module state and these initializers. It may use
  fixed zero bias and one selected weight policy, while preserving caller-supplied Tensor
  construction as an explicit path.
- Any configurable gain, activation, fan-in/fan-out mode, convolution fan geometry, bias-random
  policy, orthogonal/sparse policy, or initializer object abstraction is optional future work and
  requires its own concrete consumer and task.
- Checkpoint/state-dictionary and training optimizer work remain separately owned and must not
  infer persistence or update semantics from initializer calls.

## Architecture impact

Expected impact: None.

This task implements the existing NN allowance for conveniences composed from Model semantics and
keeps the current `modules/model -> extensions/nn -> extensions/training` direction. If
implementation needs a new dependency, changes Parameter ownership, uses graph RNG state, or
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
  constants rely on `requiresGrad == true` plus the existing Model descriptor validation to reject
  non-floating types; NN adds no duplicate fill or descriptor path.
- Fan validation is one shared private preflight with exact null order followed by floating type,
  fully static shape, rank two, positive `outFeatures`, and positive `inFeatures`. Each public fan
  method then converts the selected positive `long` extents to `double`, calls `Math.sqrt` once,
  and delegates once through the corresponding generic entry.
- Focused tests use scripted sources whose unrelated primitive methods fail. Normal output values
  prove the selected binary64 scale and `nextGaussian()` route; bounded sources separately retain
  every origin/bound pair to prove the exact `nextDouble(origin, bound)` route.

## Known limitations

- Fan-based policies accept only positive fully static rank-two Linear weights. Callers needing a
  zero-feature Tensor may use the deterministic generic initializer or supply a Tensor directly,
  but no fan formula is defined for a zero denominator.
- All results are unlabeled gradient-enabled parameter leaves. Frozen/non-trainable Tensor leaves,
  labeled Tensor diagnostics, and buffer-specific creation continue to use Model APIs directly.
- Glorot and Kaiming policies are fixed rather than configurable; convolution and other layout-
  specific fan calculations are not inferred.
- Reproducibility is bounded by the existing caller-owned `RandomGenerator` contract and has no
  cross-algorithm, provider, Java-version, serialization, or concurrent-use guarantee.

## Validation evidence

- Implementation context `/root/nn_0004_implementation` ran an initial
  `./gradlew :extensions:nn:test` after adding the executable surface and tests; it passed on
  2026-08-13 with 5 actionable tasks (3 executed, 2 up-to-date). The result contained 25 NN tests
  across 6 suites, including 9 new initializer tests, with zero failures, errors, or skips.
- The same implementation context ran the single final `./gradlew :extensions:nn:test` after
  executable code stabilized and the last test-only cleanup; it passed on 2026-08-13 with 5
  actionable tasks (2 executed, 3 up-to-date). The final result again contained 25 NN tests across
  6 suites, including 9 initializer tests, with zero failures, errors, or skips.
- `git diff --check` plus `git diff --no-index --check /dev/null <new-file>` for each of the five
  untracked task-owned files passed with no output after the stable test result. The mandatory
  clean documentation pass remains.
- Repository-wide and architecture-test validation remain deferred exactly as specified because
  no build edge, dependency rule, architecture boundary, or shared module changed.
- Clean documentation context `/root/nn_0004_docs` independently reviewed the complete NN 0004
  diff against `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, planning guide,
  documentation rules, General/API-and-Javadoc/Planning/Example profiles, NN master plan and tasks
  0001–0004, ADR 0007, dependency rules, glossary, Tensor API, current Model construction/type/
  Shape contracts, and all final NN source and tests. It found no executable behavioral defect and
  changed only Javadoc, package documentation, glossary, and NN planning records.
- The documentation pass finalized all eight initializer method contracts and package/type
  documentation for exact Shape/type retention, floating and fully static restrictions,
  `requiresGrad == true`, absent labels and provenance, `[outFeatures, inFeatures]` orientation,
  all four binary64 Glorot/Kaiming-ReLU formulas, source calls and conversion, caller-owned
  `RandomGenerator` lifecycle, failure side effects, no default or retained source, no
  `GraphRngState`, and no `Parameter` initialization policy.
- `./gradlew :extensions:nn:javadoc` — passed on 2026-08-13 after final Javadoc edits (3 actionable
  tasks: 2 executed, 1 up-to-date). Generated `ParameterInitializers` and initialization-package
  pages were inspected; they expose exactly the eight planned methods and render the formulas,
  restrictions, ownership, lifecycle, result metadata, failure, and boundary text.
- Targeted local Markdown validation passed for the NN master plan, task, and glossary: all
  relative file links resolved, both new `GraphRngState` links matched the existing glossary
  heading anchor, and every fenced-code-block count was balanced.
- Exact scope inspection passed: the task owns only its two production files, two test files,
  glossary, NN master plan, and this task record. Package/type placement matches the plan. The
  dirty CPU 0007 source/tests/task/master-plan and global-roadmap changes remain unrelated and
  untouched by this pass.
- Final `git diff --check` plus `git diff --no-index --check /dev/null <new-file>` for each of the
  five untracked task-owned files passed with no output. The successful implementation
  `./gradlew :extensions:nn:test` evidence was reused because the documentation pass changed no
  executable Java behavior.
- `docs/api/tensor-api.md` and the Model `TensorFactory`, `TensorRandoms`, `DataType`, and `Shape`
  contracts were reviewed with no change: NN delegates their existing eager construction,
  sampling, conversion, validation, allocation, and identifier semantics without modifying the
  public Model surface. `ARCHITECTURE.md`, focused architecture documents, ADR 0007, training-
  graph documentation, dependency rules, and architecture tests also require no change because
  task 0004 adds no module edge, ownership change, Tensor lifecycle state, graph-random behavior,
  training behavior, or execution-layer dependency.

## Implementation notes

- Added the field-free final `ParameterInitializers` namespace with exactly the planned eight
  public static methods and no overload, alias, default source, retained state, or nested type.
- Added direct eager Model delegation for typed zero/one and caller-source normal/uniform leaves,
  plus exact rank-two `[outFeatures, inFeatures]` Glorot and fixed ReLU/fan-in Kaiming formulas.
- Added package documentation and complete public Javadocs for eager ownership, binding and graph-
  RNG boundaries, types, shapes, distributions, conversions, failures, source advancement,
  allocation, and identifier effects; the independent documentation-focused pass finalized them.
- Added the two planned focused test classes covering exact API shape, leaf metadata and storage
  independence, typed values and conversion, exact source methods/arguments/call counts, formulas,
  validation order, no-draw/no-ID early failure, and generator failure effects.

## Completion summary

- Completed changes: Implemented the planned eager floating parameter initializer surface and
  focused executable tests without changing module-owned bindings or Model/execution behavior.
- Files changed or created: two planned production files, two planned NN test files, the glossary,
  NN master plan, and this task record.
- Tests and validation: Initial and final `./gradlew :extensions:nn:test` runs passed with 25 tests
  and no failures, errors, or skips; final NN Javadoc, generated-page inspection, Markdown links/
  anchors/fences, exact seven-file scope, package placement, and whitespace checks passed.
- Documentation-agent review: `/root/nn_0004_docs` independently finalized the API/Javadoc,
  package documentation, glossary, and planning evidence under the General, API-and-Javadoc,
  Planning, and Example profiles.
- Documentation impact: Updated the existing NN and eager-random glossary entries. Tensor API,
  architecture documents, ADR 0007, training graph, dependency rules, and architecture tests need
  no change because no Model API, ownership, dependency, graph-random, training, or execution
  contract changed.
- Javadoc review: All eight methods and the initialization package now document exact result
  metadata, formulas, orientation, restrictions, source lifecycle and draws, failures, and the
  `Parameter`/`GraphRngState` boundaries; final Javadoc generation passed.
- Glossary impact: Updated the two existing relevant entries; no separate reusable term or new
  heading was necessary.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete

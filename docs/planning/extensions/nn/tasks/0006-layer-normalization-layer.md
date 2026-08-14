# Task 0006: Layer Normalization Layer

## Status

Complete

## Goal

Add one final public `LayerNorm` module that owns mandatory exact-Shape `scale` and `bias`
parameters plus one exact typed epsilon, and constructs its output only through the existing
affine `Tensor.layerNorm` expression. Callers can supply both parameter Tensors or request exact
ones/zeros initialization for one positive fully static normalized Shape and floating data type.

## Scope

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
  inputs `[input, scale, bias]` and accepts neither scale-only nor bias-only state, so null never
  means absence and this task adds no no-affine module variant.
- Declare exact local names `scale` then `bias`. Direct and recursive discovery must retain that
  order and return the same stable wrappers as `scale()` and `bias()`.
- For caller-supplied state, infer the normalized Shape as the exact immutable Shape reference of
  `scale`. Require scale to be floating, gradient-eligible, positive-rank, fully static, and
  positive on every axis. Require bias to be floating, gradient-eligible, have the exact same data
  type, and have a structurally equal Shape. Retain both exact Tensor references; do not require
  labels, storage, provenance absence, dense layout, or particular Tensor identities.
- Validate supplied construction in this exact order: null scale, null bias, null epsilon; scale
  floating type, scale gradient eligibility, positive scale rank, fully static scale Shape, each
  scale extent positive in increasing-axis order; bias floating type, bias gradient eligibility,
  exact scale/bias data-type equality, and structural scale/bias Shape equality; Model intrinsic
  normalized-Shape/epsilon validity; then exact epsilon/parameter data-type equality. Complete all
  validation before declaring either `Parameter`.
- Reuse `new AffineLayerNormAttrs(normalizedShape, epsilon)` only as the Model-owned intrinsic
  check for positive normalized rank and finite strictly-positive floating epsilon. Do not copy
  its numeric validation, select an epsilon default, convert epsilon, or create an operation at
  construction time. After that intrinsic check, require `epsilon.dataType()` to equal the exact
  shared parameter data type.
- The initialized constructor null-checks `normalizedShape`, `dataType`, then `epsilon`; requires
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
  generic layer, or introspection consumer justifies a second public configuration surface.
- `scale()` and `bias()` return the exact stable declared `Parameter` wrappers. Add no setter or
  layer-specific replacement facade; existing `Parameter.replace` remains the only public
  schema-compatible binding replacement capability.
- `forward(input)` first rejects null input with message `input`, then reads current scale and bias
  bindings exactly once in declaration order and delegates exactly to
  `input.layerNorm(normalizedShape, currentScale, currentBias, epsilon)`. It must not reproduce
  normalization, promotion, Shape, descriptor, operation, producer, or provenance logic.
- Forward inherits the Model contract for trailing-Shape compatibility, ordered floating
  promotion, exact result-typed epsilon, population variance, accumulation and special-value
  semantics, result metadata, validation order, allocation, and exact affine provenance. A higher-
  precision input may therefore fail the inherited exact-epsilon-type check; the layer inserts no
  cast and defines no alternate mixed-precision policy.
- `LayerNorm` is mode-insensitive. Its forward method accepts no `ForwardContext`, does not inspect
  `mode()`, and produces the same composition contract in training and evaluation modes.
- A compatible replacement affects the next forward call. Earlier Tensor references and already
  constructed expressions retain the prior exact scale and bias inputs. Replacement remains
  individual and not thread-safe; this task adds no multi-parameter snapshot or transaction.
- Add focused tests for exact public surface, supplied and initialized state schema/order/identity,
  typed epsilon and validation order, exact initializer delegation and values, forward metadata
  and provenance, mode irrelevance, inherited Model failures, and replacement snapshots.
- Add complete public and package Javadocs, then use a separate clean-context documentation pass
  to finalize Javadoc, glossary impact, planning evidence, no-change conclusions, and
  documentation validation in the same overall change.

## Out of scope

- A no-affine module, optional scale or bias, scale-only or bias-only form, nullable state,
  broadcast affine state, frozen affine state, dynamic/lazy parameter Shape, or zero-sized
  normalized parameter axis.
- A default epsilon, raw `double` epsilon convenience, epsilon conversion, epsilon Tensor,
  configurable correction or accumulation, caller-selectable initializer, random initialization,
  retained/default RNG, or `GraphRngState`.
- A normalized-Shape getter, epsilon getter, generic `Module.forward`, layer interface, unary
  module contract, `Sequential`, activation facade, or broad layer abstraction.
- Buffer state, running statistics, BatchNorm, RMSNorm, GroupNorm, Dropout, Embedding, checkpoints,
  state dictionaries, serialization, optimizer groups, or training sessions.
- Any new normalization operation, attributes, Tensor overload, numerical formula, Model helper,
  gradient rule, compiler capture/fusion, backend lowering, execution, or numerical result test.
- Gradle/dependency changes, architecture/ADR/explanatory-architecture changes, architecture-test
  changes, Model/Training production changes, CPU work, global-roadmap changes, or unrelated
  refactors.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md).
- [Completed NN task 0005](0005-linear-layer.md) and its foundation dependencies.
- [Model task 0021: Layer normalization](../../../modules/model/tasks/0021-layer-normalization-semantics-and-tensor-expressions.md).
- [Tensor API layer-normalization expressions](../../../../api/tensor-api.md#layer-normalization-expressions).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.
- NN owns affine parameter declarations and typed forward composition. Model remains the sole
  owner of generic layer-normalization operation meaning, epsilon semantics, promotion, Shape,
  result metadata, and provenance. Training remains downstream and owns optimizer algorithms and
  update orchestration.
- Tensor identity, descriptor, and provenance remain immutable. `LayerNorm` may retain mutable NN
  parameter bindings but must not mutate a Tensor or add gradient lifecycle state.
- Construction and forward build only eager leaves or storage-free Model expression metadata.
  They must not imply compiler capture, backend support, numerical evaluation, storage residency,
  or execution.
- Train/eval mode remains NN forward-composition metadata. This mode-insensitive layer must not
  add an unused context argument or execution-state dependency.
- Preserve unrelated worktree changes exactly, including any concurrent CPU work and
  `docs/architecture/training-graph.md`; this task does not authorize edits outside its exact
  seven paths.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing `Module` and `Parameter` ownership,
  discovery, and replacement contracts.
- `io.github.pho001.synaptik.nn.initialization` — existing exact `ones` and `zeros` creation.
- `io.github.pho001.synaptik.model.tensor` — existing affine `Tensor.layerNorm` expression.
- `io.github.pho001.synaptik.model.operation.normalization` — existing
  `AffineLayerNormAttrs` intrinsic Shape/epsilon validation.
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing immutable Shape, floating
  data type, and exact typed `ScalarValue` contracts.

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — add one concrete stateful normalization layer beside
  `Linear`; no package or layer hierarchy is introduced.

Type placement:

- `io.github.pho001.synaptik.nn.layers.LayerNorm` — public stateful affine normalization layer;
  it belongs with concrete layers rather than generic module ownership, initialization policy, or
  Model operation semantics.
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

Documentation and planning files:

- `docs/glossary.md` — extend the existing NN and layer-normalization entries with the current
  stateful layer while keeping Model semantics and execution status distinct.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, dependency rules, module boundaries, training graph, and
  architecture tests — this task realizes their existing NN layer responsibility without changing
  a boundary, mode owner, or dependency.
- Tensor API and Model task 0021 — the existing affine layer-normalization contract remains the
  sole semantic implementation and `LayerNorm.forward` delegates exactly to it.
- Training API — discovery and individual parameter replacement remain unchanged; no optimizer,
  buffer transition, or training workflow is added.
- Model capabilities/plans, public API status, user training docs, Gradle, conformance/integration
  suites, compiler/runtime/prepare/Engine, backends, other modules, CPU planning, and the global
  roadmap — none owns this bounded Model-composition wrapper.

## Maximum scope

This task may create or modify at most:

- two production Java files;
- two NN test files; and
- the three documentation/planning files listed above.

That is an exact seven-path maximum and four Java source/test paths, within the normal 3–12-file
task guardrail. If another type, API document, module, dependency, architecture test, Model helper,
or executable integration path is required, stop and propose a focused follow-up rather than
expanding this task.

## Acceptance criteria

- Final public `LayerNorm` extends `Module` and declares exactly the two constructors and three
  methods listed in Scope, with no additional public/protected member.
- Supplied and initialized construction validate every argument in the specified order before
  declaration and before task-owned Tensor allocation respectively. Every documented caller-
  controlled failure leaves Tensor identifier state unchanged.
- Direct and recursive state discovery expose exactly `scale` then `bias`; accessors return those
  exact wrappers, and supplied construction retains exact input Tensor references.
- Initialized construction uses the exact supplied normalized Shape and data type, delegates once
  to `ones` then once to `zeros`, produces gradient-eligible scale/bias leaves for BFLOAT16,
  FLOAT32, and FLOAT64, and proves exact one/zero values and scale-before-bias ID order without RNG.
- The normalized parameter Shape is positive, fully static, and permanently enforced by existing
  `Parameter.replace`; supplied state has one exact floating type and structural Shape.
- Epsilon is retained by exact reference, has the exact parameter data type, and uses the current
  Model intrinsic finite-positive typed validation. No default, conversion, duplicated numeric
  predicate, or untyped epsilon is introduced.
- `forward` null-checks input, reads both current bindings once, and delegates to the exact affine
  Model overload. Tests inspect result Shape/type/gradient metadata, `AffineLayerNormAttrs`
  identity, one producer/output index zero, and exact ordered `[input, scale, bias]` inputs without
  evaluating values.
- Tests cover inherited Model trailing-Shape, non-floating input, promotion/epsilon mismatch, and
  allocation failures without duplicating the exhaustive Model 0021 matrix or changing failure
  semantics.
- Training and evaluation mode produce the same composition contract; no context overload,
  generic Module forward method, hidden state, or mode-dependent operation appears.
- Compatible replacements appear only in later forward provenance, earlier expressions keep old
  bindings, and incompatible state replacement remains rejected by the existing schema.
- No buffer, RNG, new semantic operation, optimizer, checkpoint, serialization, compiler,
  runtime, backend, execution, build, dependency, architecture, CPU, roadmap, or unrelated change
  enters the diff.
- Public and package Javadocs cover state schema/order, affine all-or-none policy, initialization,
  exact typed epsilon, stored configuration, Model delegation, mode irrelevance, replacement
  snapshots, nullability, validation and side effects, inherited failures, thread safety, and
  no-execution boundaries with complete tags.
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, generated Javadoc, local links/anchors, exact scope,
  no-change conclusions, and whitespace before the task becomes Complete.

## Tests / validation

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LayerNormTest --tests io.github.pho001.synaptik.nn.layers.LayerNormInitializationTest
```

After executable Java stabilizes, run exactly one final NN suite:

```bash
./gradlew :extensions:nn:test
```

The focused tests own the exact declared API, state schema/order/identity, initialized leaves and
identifier effects, intrinsic and exact-type epsilon validation, Model producer/provenance
delegation, mode irrelevance, and replacement snapshots. Existing Model 0021 tests remain the
authoritative exhaustive semantic/validation matrix; do not duplicate them or claim numerical
execution.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates changed Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `LayerNorm` and package pages, exact seven-path scope, package placement,
public surface, forbidden imports, 0001–0005 Complete, 0006 Ready before implementation and
Complete only after all evidence, exactly one Ready NN task, and absence of detailed NN 0007 or
later task files. It preserves unrelated CPU and training-graph changes and may reuse final NN
test evidence when executable Java has not changed.

Repository-wide and architecture-test validation are deferred to the NN capability checkpoint or
CI. This task changes one existing model-only module, no build edge, dependency rule, architecture
boundary, shared Gradle contract, backend behavior, or end-to-end execution path.

## Dependencies

- NN 0001–0005 are Complete.
- Model task 0021 and the exact affine `Tensor.layerNorm` overload are Complete.
- Existing `ParameterInitializers.ones` and `zeros`, `Parameter` schema replacement, and module
  discovery/mode contracts are Complete.
- Accepted ADR 0007 and the model-only NN dependency direction remain unchanged.
- The user-authorized NN parallel exception remains in force; CPU, Engine, and numerical layer-
  normalization execution are not prerequisites for expression composition.

## Follow-up tasks

- NN 0007: Embedding layer — Draft only; parameter-table initialization and padding behavior will
  be decided at that frontier.
- NN 0008: BatchNorm — Draft only; first explicit `Buffer` plus train/eval state-transition layer.
- NN 0009: Dropout — Draft only; explicit `GraphRngState` threading and evaluation bypass, with no
  hidden RNG.
- NN 0010: state dictionary and checkpoint contract — Draft only; deterministic paths, schema,
  atomic validation/load, and serialization boundary.
- NN 0011: unary Tensor module composition and `Sequential` — Draft only; a narrow shared contract
  is allowed only when the real container requires it.
- `extensions/training` later consumes discovered parameters; this task adds no optimizer or
  session behavior. Numerical end-to-end validation follows only when the compiled execution path
  and backend coverage are available.

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful NN layers composed from
Model semantics. If implementation requires a new dependency, architecture rule, Model change,
generic layer abstraction, buffer lifecycle, or executable behavior, stop and report the conflict
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

## Documentation-agent handoff

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final NN evidence and whether Java changed afterward, the selected constructor/API surface,
state/epsilon validation order, initializer/identifier effects, forward producer/provenance,
replacement and train/eval evidence, and the exact seven authorized paths.

That agent independently reads the repository instructions, architecture contract and index,
documentation rules plus General/API-Javadoc/Planning profiles, ADR 0007, this task, final source
and tests, generated Javadoc, Tensor and Training APIs, glossary, NN master/task history, Model
0021, `Tensor.layerNorm`, `AffineLayerNormAttrs`, `Module`, `Parameter`, and
`ParameterInitializers`. It finalizes only package/type Javadoc, glossary, this task, and NN master
plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/Training/public
APIs, Model plans/contracts, conformance/integration, Gradle, execution layers, backends, other
modules, CPU work, training graph, and global roadmap.

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The stateful layer is affine-only with mandatory `scale` and `bias`. This matches the exact
  all-or-none Model affine signature and avoids nullable state, hidden constants, or a partial
  two-input semantic form.
- Supplied construction infers and retains the exact scale Shape as `normalizedShape`; initialized
  construction receives the Shape explicitly. Both require positive fully static parameter
  schema even though the generic Model expression can represent empty or unresolved normalized
  axes.
- Epsilon is an exact immutable `ScalarValue`. Construction reuses `AffineLayerNormAttrs` for the
  current intrinsic finite-positive validation and separately requires its type to equal the
  parameter type. Forward leaves exact promoted-result compatibility to Model and inserts no cast.
- Initialization is exact ones for scale then zeros for bias. Layer normalization needs no RNG,
  and a configurable affine initializer has no current consumer.
- Only stable parameter handles are public. Normalized Shape and epsilon remain exact internal
  configuration because no current introspection, serialization, or generic layer consumer needs
  public getters; a later concrete consumer must justify widening the surface.
- Forward is mode-insensitive and delegates directly to Model without an unused `ForwardContext`.

## Known limitations

- Only mandatory affine state with a positive fully static normalized parameter Shape is
  supported; there is no no-affine, optional-state, lazy, dynamic, or zero-size layer form.
- Mixed floating input is accepted only when the existing Model promotion yields the stored
  epsilon's exact type. The layer performs no implicit cast or epsilon conversion.
- Parameter replacement is individual and not thread-safe. Callers must synchronize replacement
  with forward construction when one scale/bias snapshot matters.
- Forward constructs expression metadata and proves no compiler capture, gradients, fusion,
  backend coverage, numerical value, or execution.
- There is no state dictionary, checkpoint, serialization, device/dtype conversion, buffer, RNG,
  optimizer, or training-session convenience.

## Validation evidence

- Planning context `/root/nn_next_steps_plan` read the repository instructions, architecture and
  NN/training boundaries, planning/documentation rules, NN master and completed tasks, current NN
  source/tests, Model 0021 plus adjacent Embedding/BatchNorm/Dropout contracts, Tensor/Training
  APIs, and glossary before fixing this exact API and scope.
- Planning inspection confirmed that affine Model LayerNorm requires exact ordered
  `[input, scale, bias]`, exact scale/bias Shapes equal to a positive-rank normalized Shape, and a
  finite positive `ScalarValue` whose type equals the promoted result. Existing initializers can
  create both mandatory parameter leaves without RNG or another dependency.
- Planning Markdown, link/anchor, status, later-task-absence, newline, whitespace, exact two-path
  planning scope, and `git diff --check` results are recorded by the planning agent before handoff.
- Clean implementation context `/root/nn_0006_implementation` added the exact planned final
  `LayerNorm`, updated the existing `layers` package contract, and added the two planned focused
  test classes after reading the required repository, architecture, planning, NN, Model
  normalization, API, glossary, source, and test contracts. It found no architecture or scope
  conflict.
- The first focused command failed at test compilation because an all-null constructor call was
  ambiguous and one test attempted to use Tensor's package-private constructor from the NN
  package. Both were test-only issues: the calls now select constructor intent explicitly and the
  identifier-exhaustion check reuses a valid Tensor created before exhaustion. The next focused
  run reached execution and found that `Long.MAX_VALUE` is one final claimable Tensor ID; the test
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
  shared-worktree diff. It changed no executable Java or tests.
- That context finalized the type, constructor, method, and package Javadocs for exact retained
  references, complete validation and side-effect order, initialized one-then-zero state,
  return/provenance semantics, direct Model delegation, mode-insensitivity, replacement snapshots,
  nullability/failures, and the declarative non-execution boundary. It extended the glossary's NN
  and layer-normalization entries with the current stateful owner and a `[2, 3]` input / `[3]`
  normalized-Shape composition example that makes no numerical-result or execution claim.
- Authoritative final `./gradlew :extensions:nn:javadoc` passed with `BUILD SUCCESSFUL` (3
  actionable tasks: 2 executed, 1 up-to-date). Manual inspection of generated `LayerNorm.html` and
  `layers/package-summary.html` confirmed the finalized descriptions, links, tags, exceptions, and
  exact public members render as intended.
- `javap -public` and an independent reflection program both confirmed public final `LayerNorm`
  extends `Module` and declares exactly the two planned constructors plus `scale()`, `bias()`, and
  `forward(Tensor)`, with no other declared public or protected member. Manual source inspection
  confirmed one scale read then one bias read and exact delegation to
  `input.layerNorm(normalizedShape, currentScale, currentBias, epsilon)`.
- Production-import and Gradle inspection confirmed `LayerNorm` uses only Model, existing NN, and
  `java.util.Objects`, while `extensions/nn` still has exactly its existing Model dependency. A
  targeted local checker passed for changed Markdown targets, anchors, and balanced fences. Exact
  task-owned seven-path, later-task absence, 0001–0006 status, 0007–0011 Draft rows, final-newline,
  trailing-whitespace, and `git diff --check` checks also passed after final synchronization.
- No-change review concluded that `ARCHITECTURE.md`, the current architecture explanations, ADR
  0007, dependency and training-graph documents, architecture tests, Tensor API and Model task
  0021, Training API, Model capabilities/master plan, public API status, conformance/integration
  suites, compiler/runtime/prepare/Engine, backends, other modules, Gradle files, CPU planning, and
  the global roadmap remain accurate and outside scope: the layer only owns NN parameters and
  composes an existing Model expression, without changing any dependency, semantic operation,
  training workflow, backend behavior, or executable path. Concurrent CPU and global-roadmap
  worktree changes were inspected only for scope separation and preserved unchanged.

## Implementation notes

- Added final public `LayerNorm` with the exact two constructors and three methods, mandatory
  `scale` then `bias` declarations, retained exact normalized Shape and typed epsilon, and direct
  mode-insensitive affine Model delegation.
- Supplied construction completes the specified null, state-schema, intrinsic-epsilon, and exact-
  type validation before either declaration. Initialized construction validates all
  caller-controlled inputs before allocating exact one scale then zero bias leaves.
- Focused tests lock public surface, state identity/order, validation and identifier effects,
  typed eager values, exact producer/provenance, Model failure inheritance, mode irrelevance,
  individual replacement snapshots, and identifier exhaustion without numerical execution.
- No Model, Training, build, dependency, architecture, CPU, global-roadmap, execution-layer, or
  other out-of-scope path changed. The independent documentation pass finalized the package/type
  Javadocs, glossary, planning evidence, and no-change conclusions.

## Completion summary

- Completed changes: Implemented, documented, and validated the exact final affine `LayerNorm`
  module with supplied or deterministic one/zero state, exact typed epsilon, stable parameters,
  mode-insensitive direct Model delegation, and replacement snapshots.
- Files changed or created: Exactly the planned two production Java paths, two NN test paths,
  `docs/glossary.md`, NN master plan, and this task record.
- Tests and validation: Reused final implementation evidence of focused 2-suite/9-test and full NN
  11-suite/53-test success because the documentation pass changed no executable Java or tests.
  Final NN Javadoc, generated-page inspection, `javap`, reflection, manual source/import/dependency
  review, Markdown links/anchors/fences, exact scope/status/later-task checks, final newlines,
  trailing whitespace, and `git diff --check` all passed.
- Documentation-agent review: Clean context `/root/nn_0006_docs` completed the required independent
  API-Javadoc, package, glossary, planning, generated-output, and no-change review.
- Documentation impact: Final public/package Javadocs and the glossary now distinguish the
  stateful NN owner from existing Model mathematics and future compiler/backend/runtime execution.
- Javadoc review: Every declared constructor and method has meaningful contract text and complete
  parameter, result, ownership, side-effect, failure, and boundary documentation; final generated
  output was inspected.
- Glossary impact: The existing NN and layer-normalization entries now describe current
  `LayerNorm` state, initialization, forward composition, replacement timing, and a conceptual
  trailing-Shape example without a numerical-execution claim.
- Unresolved issues: None.
- Required follow-up: None for task 0006; NN 0007 remains Draft and intentionally has no detailed
  task specification.

Status: Complete

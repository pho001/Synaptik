# Task 0005: Linear Layer

## Status

Complete

## Goal

Add the first stateful neural-network layer as one final public `Linear` module. The layer owns a
rank-two `[outFeatures, inFeatures]` weight `Parameter`, optionally owns one exact rank-one
`[outFeatures]` bias `Parameter`, and constructs its output only through the existing
`Tensor.linear` convenience. Callers can either supply the exact parameter Tensors or request the
fixed initial Glorot-uniform-weight and zero-bias policy with an explicit floating data type and
caller-owned random source.

## Scope

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
  uses its explicit `bias` flag and has no overload that silently chooses whether bias exists.
- Every layer declares the weight under exact local name `weight`. A biased layer then declares
  bias under exact local name `bias`, so direct and recursive discovery retain the established
  weight-before-bias order and stable paths.
- Require every supplied weight to be floating, gradient-eligible, fully static, rank two, and
  positive on both axes. Interpret its exact Shape as `[outFeatures, inFeatures]`. Retain its exact
  Tensor reference through the `Parameter` declaration; do not require a label, host storage,
  provenance absence, dense layout, or a particular Tensor identity.
- Require every supplied bias to be floating, gradient-eligible, fully static, rank one, have the
  exact same data type as the weight, and have its sole Dimension structurally equal to weight
  axis zero. Retain its exact Tensor reference. General broadcasting, a scalar, a singleton for
  non-singleton out-features, cross-type promotion, null-as-absence, and a dynamic bias are not
  accepted as layer state.
- Validate a biased supplied construction in exact order: null weight, null bias, weight floating
  type, weight gradient eligibility, weight rank, fully-static weight Shape, positive
  out-features, positive in-features, bias floating type, bias gradient eligibility, bias rank,
  fully-static bias Shape, exact weight/bias data-type equality, and exact bias/out-features
  Dimension equality. The no-bias constructor uses the same weight subsequence. All validation
  completes before declaring either `Parameter`; it allocates no Tensor, consumes no Tensor ID,
  evaluates no expression, and changes neither supplied Tensor.
- The initialized constructor validates in exact order: non-null `dataType`, non-null
  `randomGenerator`, positive `inFeatures`, positive `outFeatures`, then floating data type. It
  builds exact Shapes `Shape.of(outFeatures, inFeatures)` and, when requested,
  `Shape.of(outFeatures)`, and lets the existing initializers complete their documented
  Java-array-limit, checked-arithmetic, allocation, identifier, and random-source validation.
- Use exactly `ParameterInitializers.glorotUniform(weightShape, dataType, randomGenerator)` for
  initialized weight. When `bias == true`, use exactly
  `ParameterInitializers.zeros(biasShape, dataType)` after the weight initializer succeeds. Do not
  duplicate either formula or sampling loop. The fixed weight bounds are therefore
  `[-sqrt(6 / (inFeatures + outFeatures)), +sqrt(6 / (inFeatures + outFeatures)))`, and bias is
  deterministic exact typed zero and consumes no random draw.
- The initialized path creates the weight Tensor before an optional bias Tensor and declares the
  matching parameters in that order. All caller-controlled validation failures happen before a
  random draw or Tensor identifier allocation. A source failure preserves the initializer's
  completed-draw/non-rollback contract and creates no Tensor. Resource, identifier-exhaustion, or
  allocation failure after a successful weight initialization does not roll back weight draws or
  its consumed identifier; no partially constructed `Linear` instance is returned.
- `weight()` returns the exact stable declared wrapper. `bias()` returns a non-null empty
  `Optional` for a no-bias layer or an `Optional` containing the exact stable bias wrapper. Add no
  setter or replacement facade; the returned `Parameter` keeps NN 0004A's public schema-compatible
  `replace` capability for inspection and downstream training.
- `forward(input)` first rejects a null input with message `input`, reads each current parameter
  binding once, and delegates exactly to `input.linear(weight.value())` or
  `input.linear(weight.value(), bias.value())`. It must not reproduce transpose, MATMUL, ADD,
  Shape, promotion, producer, or provenance logic.
- Forward inherits the Model contract for input rank, floating promotion, final-axis contraction,
  Shape derivation, validation, allocation order, gradient eligibility, and visible
  PERMUTE-to-MATMUL-to-optional-ADD provenance. It constructs model metadata only and does not
  evaluate values.
- `Linear` forward behavior is identical in `TRAINING` and `EVALUATION` modes. `train()` and
  `eval()` retain their inherited metadata behavior, but `forward` accepts no `ForwardContext`
  and does not branch on mode.
- A successful `Parameter.replace` affects the next forward call. A Tensor returned by an earlier
  `value()` call and every expression already constructed from it remain unchanged. The
  declaration schema prevents replacement with another data type, Shape, or non-gradient Tensor.
- Add focused tests for exact surface and state ownership, supplied-Tensor validation, initialized
  Tensor policy and source ownership, forward metadata/provenance, mode irrelevance, and
  replacement snapshots.
- Add complete Javadoc and `layers` package documentation, then use a separate clean-context
  documentation pass to finalize Javadoc, glossary impact, planning evidence, and documentation
  validation in the same overall change.

## Out of scope

- A generic `Module.forward`, layer interface, abstract layer base, sequential container, block,
  activation, flattening, lazy/dynamic feature inference, or dynamic parameter Shape.
- A caller-selectable initializer, initializer object, Xavier alias, configurable gain or fan
  policy, random bias, default data type, default seed, hidden or retained RNG, or module-owned
  random state.
- Integral or BOOL layer parameters, frozen parameters, mixed-type weight and bias, broadcast bias,
  a public state setter, buffer, checkpoint, state dictionary, serialization, device placement,
  or parameter-group behavior.
- Optimizer algorithms, gradient publication, training sessions, autograd rules, compiler capture
  or fusion, graph compilation, prepare/runtime/Engine behavior, backend lowering, numerical
  execution, or end-to-end output-value assertions.
- A new LINEAR operation kind, attrs, producer, result carrier, Tensor overload, MATMUL/ADD/
  PERMUTE change, or duplicate linear numerical/Shape/provenance implementation.
- Gradle or dependency changes, architecture-contract/ADR/explanatory-architecture changes,
  architecture-test changes, Model changes, Training production changes, CPU work, or global
  roadmap changes.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN ownership, Tensor invariants,
  dependency direction, optimizer/training lifecycle, and testing requirements.
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [NN master plan](../master-plan.md).
- [Planning guide](../../../planning-guide.md).
- [Task 0001](0001-module-parameter-buffer-and-forward-context-foundation.md) through
  [task 0004A](0004a-parameter-update-and-traversal-hardening.md).
- [Model task 0019D: Linear convenience](../../../modules/model/tasks/0019d-linear-convenience.md).
- [Tensor API linear-projection convenience](../../../../api/tensor-api.md#linear-projection-convenience).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, engine, or concrete backend code.
- NN owns layer state, typed forward composition, and parameter declarations. Model remains the
  sole owner of generic `Tensor.linear` validation and primitive expression semantics. Training
  remains downstream and owns optimizer algorithms and orchestration.
- Tensor identity, descriptor, and provenance remain immutable. `Linear` may retain and discover
  mutable NN `Parameter` bindings but must not mutate a Tensor or add gradient lifecycle state.
- The layer constructs a visible existing primitive Tensor-expression chain and must not imply
  compiler capture, fusion, backend support, storage residency, or execution.
- Train/eval mode remains NN composition metadata. A mode-insensitive layer does not invent a
  meaningless `ForwardContext` argument or execution-state dependency.
- The authorized NN parallel exception is implementation-order only. Preserve all dirty CPU
  files, CPU planning, the global roadmap, and unrelated glossary hunks.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — existing `Module` and `Parameter` ownership and
  replacement contracts.
- `io.github.pho001.synaptik.nn.initialization` — existing exact Glorot-uniform and zero Tensor
  creation policies.
- `io.github.pho001.synaptik.model.tensor` — existing `Tensor.linear` model composition.
- `io.github.pho001.synaptik.model.shape` and `.datatype` — existing Shape and floating type
  contracts.

Packages added or changed:

- `io.github.pho001.synaptik.nn.layers` — public stateful layer types whose typed forward methods
  compose existing Model Tensor operations.

Type placement:

- `io.github.pho001.synaptik.nn.layers.Linear` — first public stateful layer; it belongs with
  concrete NN layers rather than generic module ownership or stateless initialization policy.
- `io.github.pho001.synaptik.nn.layers.LinearTest` — focused public-surface, supplied-state,
  forward-provenance, mode, and replacement contract tests.
- `io.github.pho001.synaptik.nn.layers.LinearInitializationTest` — focused initialized-construction,
  formula delegation, source ownership, metadata, validation, and side-effect tests.

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Linear.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearInitializationTest.java`.

Documentation and planning files:

- `docs/glossary.md` — extend the existing `Linear projection` and NN module entries to
  distinguish and describe the now-current stateful layer without changing Model semantics.
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Reviewed unchanged unless the implementation makes a current statement inaccurate:

- `ARCHITECTURE.md`, ADR 0007, dependency rules, module-boundary/training-graph explanations, and
  architecture tests — the task realizes the already-authorized layer responsibility without a
  dependency or lifecycle change.
- Tensor API and Model task 0019D — `Tensor.linear` semantics and public surface remain unchanged;
  `Linear.forward` delegates to them.
- Training API — the already-current generic parameter discovery/replacement contract remains
  exact; no optimizer or training workflow is added.
- public API status and user training guide — no runnable training or execution lifecycle becomes
  available, and complete layer usage is documented in package/type Javadoc plus the glossary.
- Model capabilities/plans, Gradle, conformance/integration suites, compiler/runtime/prepare/
  Engine, backends, and the global roadmap — none owns this model-only NN layer task.

## Maximum scope

This task may create or modify at most:

- two production Java files;
- two NN test files; and
- the three documentation/planning files listed above.

That is an exact seven-path maximum. If another production/test type, API document, module,
dependency, architecture test, Model helper, or executable integration path is required, stop and
propose a focused follow-up rather than expanding this task.

## Acceptance criteria

- Final public `Linear` extends `Module` and exposes exactly the three constructors and three
  methods listed in Scope, with no additional public/protected member declared by the class.
- Supplied state is validated in the specified order before either parameter declaration. Weight
  and bias use exact stable names, wrapper identity, declaration order, input Tensor references,
  type, Shape, and gradient eligibility.
- No-bias and biased state discovery returns exactly `[weight]` and `[weight, bias]` respectively;
  `bias()` is empty or contains the same wrapper present in discovery.
- Initialized construction requires explicit positive features, explicit floating type, explicit
  caller-owned RNG, and explicit bias selection. Weight delegates once to Glorot uniform with
  Shape `[outFeatures, inFeatures]`; optional bias delegates once to typed zeros with Shape
  `[outFeatures]` and consumes no source draw.
- Scripted-source tests prove exact Glorot bounds, row-major draw count, exact caller source use,
  all three floating result types as practical, zero bias values, weight-before-bias ID order, and
  no draw/ID allocation for caller-controlled prevalidation failures. Source/resource failure
  tests assert only the documented non-rollback boundary.
- `forward` has exactly one Tensor input and delegates to the matching `Tensor.linear` overload
  using current bindings. Tests inspect Shape/type/gradient metadata and exact primitive producers,
  ordered inputs, output indices, and weight/bias identity without executing values.
- Null input and inherited Model rank/type/contraction failures are covered without duplicating
  the exhaustive Model 0019D test matrix or asserting a different message/order.
- Training and evaluation mode produce the same layer composition contract; no context overload or
  generic Module forward method appears.
- Replacement tests prove that compatible weight and bias replacements appear in subsequent
  forward provenance, while previously constructed expressions retain the earlier exact bindings;
  incompatible replacement remains rejected by the existing declaration schema.
- No new operation/numerical algorithm, optimizer, checkpoint, serialization, compiler, runtime,
  backend, execution, build, dependency, architecture, CPU, global-roadmap, or unrelated refactor
  enters the diff.
- Public and package Javadocs cover purpose, parameter orientation, optionality, initialization
  policy, ownership, exact result delegation, train/eval irrelevance, replacement snapshots,
  nullability, validation order, random-source ownership, side effects, failures, and lifecycle
  boundaries. Every input/result/failure has the required tags.
- A separate clean-context documentation pass independently finalizes Javadoc, package docs,
  glossary wording, planning evidence, no-change conclusions, generated Javadoc, Markdown, exact
  scope, and whitespace before the task becomes Complete.

## Tests / validation

Implementation pass runs focused tests while developing:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.LinearTest --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest
```

After executable Java stabilizes, run exactly one final NN suite:

```bash
./gradlew :extensions:nn:test
```

The focused layer tests own the exact declared API surface, state schema/order/identity,
initializer delegation and source effects, forward provenance, mode irrelevance, and replacement
snapshots. Existing module, initializer, and Model tests remain unchanged and protect their
underlying contracts. Do not repeat the exhaustive Model linear suite or make numerical execution
claims.

The separate documentation pass runs after final Javadoc edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, generated `Linear` and package Javadocs, exact seven-path scope, package placement,
public constructor/method surface, 0001–0004A Complete, 0005 Ready before implementation and
Complete only after all evidence, and no later detailed NN task. It must preserve unrelated dirty
CPU and global-roadmap files and may reuse the final NN test evidence when executable Java has not
changed.

Repository-wide and architecture-test validation are deferred to the NN capability checkpoint or
CI. This task changes one existing model-only module, no build edge, dependency rule, architecture
boundary, shared Gradle contract, backend behavior, or end-to-end execution path.

## Dependencies

- NN 0001–0004A are Complete.
- Model task 0019D and its exact `Tensor.linear` overloads are Complete.
- Existing `ParameterInitializers.glorotUniform` and `zeros` contracts are Complete.
- Accepted ADR 0007 and the model-only NN dependency direction remain unchanged.
- The user-authorized parallel exception recorded in the NN master plan remains in force, with
  dirty CPU and global-roadmap paths outside this task.

## Follow-up tasks

- Stateless activation/module conveniences and normalization wrappers remain future NN rows to be
  planned only at their frontier.
- Batch normalization and dropout must wait for their explicit buffer/mode and graph-RNG layer
  contracts; they are not inferred from `Linear`.
- Checkpoint/state-dictionary semantics remain a separate state-tree task.
- `extensions/training` later consumes the stable discovered `Parameter` wrappers; this task adds
  no optimizer or session contract.
- Numerical end-to-end layer validation follows only when the required compiled execution path and
  backend operation coverage are available.

## Architecture impact

Expected impact: None.

This task implements the existing architecture allowance for stateful NN layers composed from
Model semantics. If implementation requires a new dependency, execution behavior, architecture
rule, generic layer abstraction, Model change, or broader lifecycle decision, stop and report the
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

## Documentation-agent handoff

Give the separate documentation-focused agent this task, the complete implementation diff, exact
focused/final NN evidence and whether Java changed afterward, the selected constructor/API surface,
state validation/order, initializer delegation and source effects, forward producer chains,
replacement evidence, train/eval result, and the seven authorized paths.

That agent independently reads the repository instructions, architecture contract and index,
documentation rules plus General/API-Javadoc/Planning profiles, ADR 0007, this task, final source
and tests, generated Javadoc, Tensor and Training APIs, glossary, NN master plan and completed
foundation/initializer tasks, Model 0019D, `Tensor.linear`, `Module`, `Parameter`, and
`ParameterInitializers`. It finalizes only package/type Javadoc, the glossary, this task, and the NN
master plan. It records reasoned no-change conclusions for architecture/ADRs/tests, Tensor/
Training/public APIs, Model plans/contracts, conformance/integration, Gradle, execution layers,
backends, other modules, CPU work, and the global roadmap.

The documentation pass reuses successful Java evidence unless it changes executable Java or
records a concrete reason to rerun it. It records its clean-context identifier, files/topics
reviewed, commands/results, glossary impact, limitations, and unresolved issues before completion.

## Local decisions

- The public surface uses constructor arity for caller-supplied bias because null is never a state
  value or absence sentinel. The initialized constructor uses one explicit boolean because two
  otherwise-identical initialization overloads would add surface without adding type safety; no
  overload silently defaults bias on or off.
- The fixed first-layer policy is Glorot unit-gain uniform weight plus deterministic exact-zero
  bias. It reuses the two existing initializer entries directly and gives the random source no
  role in bias initialization.
- Caller-supplied parameters are fully static and positive. This first stateful layer models fixed
  learnable feature axes; lazy and dynamically shaped parameter ownership require a separate
  contract rather than weakening constructor and replacement schema.
- Weight and bias must have the same exact floating type. Allowing final ADD promotion would make
  one layer own two parameter precisions and obscure its declared data type; callers can construct
  explicit casts before supplying state if a later task deliberately permits that policy.
- Stable `weight()` and optional `bias()` accessors expose the existing NN-owned `Parameter`
  capability. No layer-local setter duplicates `Parameter.replace`, and discovery remains the
  generic downstream path.
- `forward` is mode-insensitive and delegates to Model rather than accepting an unused
  `ForwardContext`. Mode-sensitive future layers will define their own typed context-bearing
  signatures.

## Known limitations

- Only fixed positive fully static feature dimensions are supported; there is no lazy or dynamic
  parameter materialization.
- Initialization policy is fixed and not caller-configurable beyond feature counts, bias presence,
  floating type, and random source.
- Parameter replacement is individual and not thread-safe. Callers must synchronize replacement
  with forward construction when a consistent multi-parameter snapshot matters.
- `forward` builds expression metadata and does not prove compiler capture, gradients, fusion,
  backend support, numerical values, or execution.
- There is no checkpoint, serialization, device, dtype-conversion, or training-session convenience.

## Validation evidence

- Clean implementation context `/root/nn_0005_implementation` read the required repository,
  architecture, planning, documentation-profile, NN foundation/initializer, Model linear, API,
  glossary, source, and test contracts before editing. It found no architecture or scope conflict.
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
  authoritative final Javadoc command after its edits.
- Focused coverage proves the exact three-constructor/three-method public surface, supplied state
  schema/name/order/reference retention, constructor validation order and no Tensor-ID effects,
  exact Glorot bounds and caller-source draw counts for all three floating types, deterministic
  zero bias and weight-before-bias IDs, source/array-limit non-rollback boundaries, exact primitive
  forward producers and ordered inputs, train/eval equivalence, replacement snapshots, and
  inherited Model failures without numerical execution claims.
- Implementation whitespace checks passed with no output for the tracked diff and every current
  untracked task-owned text file. `javap -public` showed exactly the planned three constructors and
  three declared methods, and the production import scan found no forbidden extension or
  execution/backend edge. Final Javadoc, Markdown, generated-page, exact seven-path scope,
  glossary, planning, no-change, and final whitespace evidence remain assigned to the mandatory
  clean documentation pass.
- Clean documentation context `/root/nn_0005_docs` independently read the repository and
  architecture contracts, documentation rules and General/API-Javadoc/Planning/Example profiles,
  NN master/task history, ADR and dependency boundary, final source/tests, Model 0019D and
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
  required no correction.
- `javap -public -classpath extensions/nn/build/classes/java/main:modules/model/build/classes/java/main
  io.github.pho001.synaptik.nn.layers.Linear` showed exactly the three planned constructors and
  `weight`, `bias`, and `forward`, with no extra declared public member. Production import and
  forbidden-package scans found only Model, NN initializer/module, and Java standard-library
  dependencies.
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
- Glossary wording now identifies the current stateful `Linear` separately from the Model linear
  projection and records its parameter orientation, supplied/fixed-initialized state, explicit
  caller RNG, exact forward delegation, mode-insensitivity, and replacement snapshot boundary.
- Architecture, ADR 0007, dependency rules and architecture tests require no change because the
  layer realizes their existing NN ownership and `model -> nn -> training` direction. Tensor API,
  Model 0019D, and Model code remain exact because forward only delegates to their unchanged
  linear convenience. Training API remains exact because the task adds no optimizer or update
  orchestration and uses the existing public `Parameter.replace` contract. Public API status,
  user training documentation, Model capabilities/plans, conformance/integration suites, Gradle,
  compiler/runtime/prepare/Engine, backends, other modules, CPU work, and the global roadmap remain
  unchanged because this task adds only model-expression composition inside the existing NN
  module and makes no execution or repository-wide status claim.

## Implementation notes

- Added final public `io.github.pho001.synaptik.nn.layers.Linear` and package documentation without
  adding another production type, interface, operation, dependency, or execution path.
- Caller-supplied constructors complete their exact weight/bias preflight before declaring either
  parameter and retain the exact Tensor references under `weight` then optional `bias`.
- Initialized construction validates caller inputs before Shape/Tensor work, delegates weight once
  to `ParameterInitializers.glorotUniform`, delegates requested bias once to
  `ParameterInitializers.zeros`, and retains no random source.
- `forward` rejects null input first, reads current bindings once, and delegates directly to the
  matching `Tensor.linear` overload. Tests inspect the existing visible PERMUTE/MATMUL/optional-ADD
  chain, mode irrelevance, and old-versus-new parameter snapshots.
- The implementation changed only the planned two production files, two tests, task, and NN master
  plan. The documentation-focused pass then added the planned glossary update and finalized the
  task/master evidence; Model, Training, architecture, build, CPU, global-roadmap, and execution-
  layer files remain unchanged.

## Completion summary

- Completed changes: Implemented and focused-tested the planned `Linear` layer API, supplied and
  initialized state policies, exact Model forward delegation, mode-insensitivity, and compatible
  replacement snapshots.
- Files changed or created: Exactly the planned two production Java files, two NN test files,
  glossary, NN master plan, and this task record.
- Tests and validation: Focused 2-suite/11-test and final NN 9-suite/44-test runs passed with no
  failures, errors, or skips; final NN Javadoc, generated-page inspection, Markdown, exact public
  surface, forbidden-import/dependency, seven-path scope, newline/whitespace, and
  `git diff --check` validation passed.
- Documentation-agent review: Completed independently in clean context `/root/nn_0005_docs`
  using the General, API/Javadoc, Planning, and Example profiles; stable executable evidence was
  reused because no executable Java changed after the implementation run.
- Documentation impact: Finalized the glossary and NN planning evidence. Architecture, ADR 0007,
  dependency rules/tests, Tensor/Training/public APIs, Model contracts/plans, conformance and
  integration suites, Gradle, execution layers, backends, other modules, CPU work, and the global
  roadmap remain accurate and unchanged because `Linear` realizes the existing model-only NN
  layer boundary by exact `Tensor.linear` delegation.
- Javadoc review: Independently verified the final `Linear` and package Javadocs against source,
  tests, Model linear semantics, parameter replacement, mode, ownership, failures, and generated
  pages; no correction was needed.
- Glossary impact: Extended the existing NN module and Linear projection entries with the current
  stateful `Linear` ownership, initialization, forward, replacement-snapshot, and execution
  boundaries without redefining Model semantics.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete

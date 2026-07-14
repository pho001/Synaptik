# Task 0002: Compile Modes and Graph Optimization Configuration

## Status

Complete

## Goal

Add the two remaining leaf values needed to describe graph scope and optional backend-neutral
optimization before a later task aggregates compile configuration.

Mental model:

```text
CompileMode.FORWARD_ONLY
CompileMode.FORWARD_AND_BACKWARD
CompileMode.TRAINING_STEP

GraphOptimizationConfig.disabled()
GraphOptimizationConfig.standard()
```

These values record requested compile semantics only. They neither capture nor transform a graph,
construct autograd, invoke an optimizer, choose backend ownership, or expose compiler pass
implementation details.

## Scope

- Add public enum `CompileMode` in `io.github.pho001.synaptik.config.compile`.
  - It contains exactly `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` in that order.
  - `FORWARD_ONLY` requests forward graph construction and requested forward publications only.
  - `FORWARD_AND_BACKWARD` requests later compiler autograd expansion plus combined forward and
    backward compile-time graph work.
  - `TRAINING_STEP` records the architecture's training-step direction. It does not itself add an
    optimizer, optimizer-update graph, training session, schedule, or execution behavior.
  - It adds no project field, method, constructor, nested type, alias, phase, publication setting,
    gradient target, optimizer setting, or metadata.
- Add public record `GraphOptimizationConfig` in the same package.
  - It contains exactly one boolean component: `optionalOptimizationsEnabled`.
  - It accepts and retains either primitive value without validation or normalization.
  - It exposes an explicitly documented accessor.
  - It preserves ordinary record equality, hashing, and diagnostic `toString()` behavior.
- Add exactly two public static factories to `GraphOptimizationConfig`:
  - `GraphOptimizationConfig disabled()` returns a new value with
    `optionalOptimizationsEnabled == false`.
  - `GraphOptimizationConfig standard()` returns a new value with
    `optionalOptimizationsEnabled == true`.
- Define the boolean narrowly:
  - `false` requests that later compiler work skip optional semantics-preserving graph
    optimizations.
  - `true` permits the compiler's standard optional semantics-preserving optimization pipeline.
  - Neither value controls graph capture, topological ordering, shape/data-type inference,
    validation, mandatory canonical representation, autograd required by the compile mode,
    publication binding, backend-neutral planning, preparation, or execution.
  - `standard()` promises no public pass list, pass order, internal graph shape, or implementation
    strategy. Compiler tasks own those choices while preserving observable semantics.
- Add no optimization level enum, pass-specific booleans, custom pass list, class reference,
  registry, plugin, budget, timeout, seed, numerical relaxation, approximate-math permission,
  backend fusion switch, or debug/tracing option.
- Update compile-package Javadoc to cover the three current standalone compile configuration
  values and their current-versus-planned boundaries.
- Add one focused test for `CompileMode` exact enum/API shape, order, identity, and absence of
  fields, methods, nesting, aliases, or behavior.
- Add one focused test for `GraphOptimizationConfig` exact record/API shape, direct construction,
  factories, primitive retention, freshness, value semantics, and absence of pass/lifecycle
  surfaces.
- Finalize Javadocs and update the focused training-graph status explanation, Public API, Compile
  API, compile workflow guide, autograd guide, glossary, config master plan, trace interleave
  status, and roadmap in the same overall change.

## Out of scope

- `CompileConfig`, aggregation, defaults for the later aggregate, builder, loader, environment or
  system-property configuration, serialization, persistence, or compatibility schema
- changing `BackendIntent`, `BackendRequirement`, backend-contract source/build/tests, hard-target
  evaluation, backend preference, partition scoring, profiles, calibration, or tuning
- compiler graph capture, traversal, indexing, inference, validation, canonicalization,
  optimization passes, autograd implementation, publication binding, planning orchestration, or
  diagnostics
- defining a public list or stable identity for dead-code elimination, common-subexpression
  elimination, constant folding, algebraic simplification, post-autograd cleanup, or another pass
- aggressive, unsafe, experimental, size, speed, memory, debug, deterministic, backend-specific,
  or numerically relaxed optimization modes
- approximate math, changed floating-point association, tolerance, reproducibility, deterministic
  execution, or backend numerical policy
- `GraphPhase` changes, backward-only mode, evaluation mode, inference alias, optimizer mode,
  gradient publication policy, optimizer choice, training session, or train/eval module state
- implementing the future compiled optimizer graph suggested by `TRAINING_STEP`; any architecture
  evolution needed for optimizer-update graph work remains explicit later work
- prepare, run, publication, runtime, engine, backend lowering, fusion, kernel selection,
  executable units, storage, physical memory, or execution behavior
- trace DTOs, attributes, payloads, emission, or serialization
- dependencies, Gradle, Java version, architecture contract, architecture tests, another module,
  concrete backends, conformance, integration, or detailed later task specifications
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)
- [Training combined forward/backward graph ADR](../../../../design/decisions/0005-training-combined-forward-backward-graph.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Config task 0001](0001-backend-intent-foundation.md)
- [Trace master plan](../../trace/master-plan.md)
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Compile workflow guide](../../../../user-guide/compiling-graphs.md)
- [Autograd guide](../../../../user-guide/autograd.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/config` owns immutable declarative compile configuration; compiler owns interpretation
  and graph transformation.
- Compile mode describes requested graph scope. It does not execute autograd, publish gradients,
  choose a runtime schedule, or transfer optimizer ownership into config.
- Optimization configuration may permit or suppress optional semantics-preserving compiler work,
  but must not expose compiler pass implementation or backend prepare choices.
- Required inference, validation, mandatory representation normalization, and other correctness
  work cannot be disabled through this value.
- Backend-specific fusion, specialization, lowering, and kernel selection remain backend prepare
  concerns regardless of optimization configuration.
- The task adds no dependency and changes no existing dependency direction.
- Stop if implementation needs an additional optimization-policy type, compiler API, training
  type, another module, dependency edit, architecture decision, or pass-specific public surface.

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.config.compile` — immutable public graph-compilation configuration
  leaves; task 0002 adds graph scope and optional-optimization permission

Types added:

- `io.github.pho001.synaptik.config.compile.CompileMode` — exact requested forward/backward/training-
  step compile-time graph scope
- `io.github.pho001.synaptik.config.compile.GraphOptimizationConfig` — exact boolean permission for
  optional semantics-preserving compiler optimization

Existing `BackendIntent` remains unchanged. Package Javadoc is updated to describe the three
current standalone values. No subpackage, helper, service, registry, builder, compiler type, or
training type is added.

## Affected files

Production — exactly three paths:

- add `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/CompileMode.java`
- add
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/GraphOptimizationConfig.java`
- Javadoc-only update to
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/package-info.java`

Tests — exactly two paths:

- add `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/CompileModeTest.java`
- add
  `modules/config/src/test/java/io/github/pho001/synaptik/config/compile/GraphOptimizationConfigTest.java`

Documentation — exactly six paths:

- current-status wording only in `docs/architecture/training-graph.md`; no architecture rule
  changes
- `docs/api/public-api.md`
- `docs/api/compile-api.md`
- `docs/user-guide/compiling-graphs.md`
- `docs/user-guide/autograd.md`
- `docs/glossary.md`

Planning — exactly four paths:

- add and finalize this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: `BackendIntent` source/tests/Javadocs, config build and architecture
dependency test, all backend-contract source/tests/build, model `GraphPhase`, compiler placeholder,
training/NN placeholders, `AGENTS.md`, `ARCHITECTURE.md`, lifecycle/module/dependency/partition-
scoring explanations, ADR 0005, architecture tests, root build/settings, other modules, concrete
backends, conformance/integration tests, and all later config tasks.

## Maximum scope

At most the exact fifteen paths above. Stop if implementation requires another production type,
test, document, dependency, package, module, architecture change, compiler/training behavior, or
detailed follow-up specification. The user's standing instruction permits a necessary Javadoc-
only path-count increase, but any expansion must be recorded and must not alter a completed
declaration or behavior.

## Acceptance criteria

- `CompileMode` is a public enum containing exactly `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and
  `TRAINING_STEP` in that order, with no project field, method, constructor argument, nested type,
  alias, metadata, or implemented interface.
- Mode Javadocs state requested graph scope without claiming that compiler, autograd, optimizer,
  schedule, publication, preparation, or execution behavior is current.
- `GraphOptimizationConfig` is a public record containing exactly one boolean
  `optionalOptimizationsEnabled` component, a public canonical constructor, explicit documented
  accessor, exactly `disabled()` and `standard()` as added public factories, ordinary record
  object methods, and no other project field, method, nested type, interface, builder, or factory.
- Direct construction preserves either primitive value. `disabled()` returns fresh equal false
  values and `standard()` returns fresh equal true values.
- Disabled configuration suppresses only optional semantics-preserving compiler optimizations;
  it does not claim to suppress inference, validation, mandatory canonical representation,
  mode-required autograd, publication binding, planning, preparation, or execution.
- Standard configuration permits a later compiler-owned standard pipeline without promising exact
  passes, order, internal graph shape, aggressive rewrites, changed numerical semantics, backend
  fusion, or execution behavior.
- No individual compiler-pass flag, pass identifier, class reference, plugin/registry, level,
  profile, budget, numerical-relaxation, backend, tracing, or lifecycle surface is added.
- Existing `BackendIntent` declaration, bytecode/API shape, behavior, tests, Javadocs, dependency,
  and hard-requirement boundary remain unchanged.
- Focused tests fail on enum order/API drift, record component/API drift, factory behavior drift,
  and forbidden added surfaces.
- Package Javadoc, training-graph status, Public API, Compile API, compile workflow, autograd guide,
  and glossary distinguish current construction values from planned compiler interpretation,
  `CompileConfig`, autograd, optimization passes, training execution, and lifecycle APIs.
- Config master plan marks task 0001 Complete and task 0002 as the sole Ready task before
  implementation; tasks 0003–0008 remain ordered Draft rows without detailed specifications.
  Trace master plan and roadmap record the current config interleave without resuming trace work.
- A separate clean-context documentation-focused pass finalizes both new type Javadocs, package
  Javadoc, and the ten documentation/planning paths after Java validation.
- Exactly one final config module suite, config Javadoc, exact fifteen-path scope, Markdown, final
  newlines, trailing whitespace, and `git diff --check` pass. Repository-wide validation remains
  deferred to the config capability checkpoint or CI because this task changes no dependency or
  cross-module executable contract.

## Tests / validation

Run focused tests while developing. After executable Java stabilizes, run exactly one final config
module suite:

```bash
./gradlew :modules:config:test
```

Record suite and test counts from XML reports. Then hand the actual diff and exact Java evidence
to a separate clean-context documentation-focused agent in the same overall change. That pass
independently inspects final source/tests, current `BackendIntent`, the architecture-defined mode
semantics, generated Javadoc, and directly affected documentation; applies General,
API/Javadoc, User guide, Planning, and Example profiles as relevant; finalizes both new type
Javadocs, package Javadoc, and ten documentation/planning paths; records reasoned no-change
conclusions; and runs:

```bash
./gradlew :modules:config:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses successful Java evidence unless executable Java changes or it
records a concrete reason to rerun. Inspect generated package, enum, and record pages; rely on
focused automated API-shape tests; confirm exact fifteen-path scope; confirm task 0002 Complete
with no Ready row after implementation; and confirm no later config or trace task specification
exists.

Do not run repository-wide tests in this task. The config master plan defers them to its next
recorded capability checkpoint or CI.

## Dependencies

- Complete config task 0001 and its stable `BackendIntent` package/dependency foundation.
- The architecture-defined forward-only, forward-and-backward, and training-step meanings.
- Completed model graph phase and provenance foundations as read-only context only.

## Follow-up tasks

- Backend-neutral partition-scoring configuration and ranking preference remain Draft task 0003.
- Immutable platform, backend, and tuning profiles remain Draft task 0004.
- `CompileConfig` aggregation of the current standalone leaves remains Draft task 0005.
- Compiler interpretation, autograd, graph optimization passes, publication binding, planning,
  trace translation, prepare, runtime, engine, training, and concrete backend behavior remain in
  their owning later project areas.

## Architecture impact

Expected impact: None.

The task implements values already named and described by the architecture contract. The focused
training-graph edit corrects implementation-status wording only. It changes no module ownership,
dependency direction, lifecycle stage, autograd decision, optimizer boundary, or backend rule.
Stop before editing architecture if implementation reveals a conflicting meaning.

## Implementation prompt

Use this prompt in one separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused lifecycle/training-graph/module/dependency architecture
documents and training ADR, documentation/planning rules and profiles, roadmap, config and trace
master plans, completed config task 0001, config task 0002, current config source/tests/Javadocs,
Public/Compile APIs, compile/autograd guides, glossary, model GraphPhase as read-only context,
compiler placeholder, and Java 26 root configuration.

Implement docs/planning/modules/config/tasks/0002-compile-modes-and-graph-optimization-
configuration.md exactly inside its fifteen authorized paths. Add only the exact three-value
CompileMode and one-boolean GraphOptimizationConfig with disabled/standard factories, focused
tests, package-Javadoc update, current-status documentation, and permitted planning changes.
Preserve BackendIntent and every dependency/boundary. Add no compiler-pass API, optimization
level, unsafe numerical mode, aggregate config, preference/scoring/profile, compiler/autograd/
training behavior, dependency, architecture change, or later task. Stop on architecture,
compile-mode semantics, optimization-boundary, package, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final :modules:config:test after executable
Java stabilizes. Then hand the actual diff and Java evidence to a separate clean-context
documentation agent in the same overall change. That pass finalizes Javadocs, training-graph
status, Public/Compile APIs, compile/autograd guides, glossary, task/master/trace/roadmap status,
config Javadoc, and documentation/scope checks while reusing Java evidence unless executable
behavior changes. Mark config task 0002 Complete only after both passes succeed. Leave config
0003–0008 and trace 0003–0008 Draft without detailed specifications. Do not run a root test.
```

## Local decisions

- Use the exact three compile modes already established by architecture. Add no alias or helper
  method; later compiler code can interpret the enum explicitly.
- Represent optional optimization permission with one boolean rather than public compiler pass
  switches or a speculative hierarchy.
- Name the component `optionalOptimizationsEnabled` to make clear that correctness-required work
  is never disabled.
- Provide only `disabled()` and `standard()` conveniences. The later aggregate owns its default;
  this task does not silently choose one.
- Let `standard()` denote a compiler-owned, semantics-preserving pipeline whose internal passes
  and ordering may evolve. Public configuration promises requested policy, not graph structure.
- Add no `AGGRESSIVE` level because acceptable numerical reassociation, approximation,
  determinism, size/speed tradeoffs, and compatibility are not yet defined.
- Keep `TRAINING_STEP` as declarative architecture vocabulary. It neither implements nor promises
  the future optimizer-update graph direction.
- Update only the stale implementation-status sentence in the focused training-graph explanation;
  do not change its architecture meaning.

## Known limitations

- The standalone values are not yet accepted by `CompileConfig` or any compiler entry point.
- No current compiler interprets the selected mode or optimization permission.
- Standard optimization does not expose, freeze, or identify a pass set, pass order, internal
  graph structure, numerical strategy, or diagnostic representation.
- Disabled optimization does not define every mandatory compiler transformation; later compiler
  contracts must distinguish correctness work from optional optimization consistently.
- `TRAINING_STEP` records future direction but does not make compiled optimizer updates current.
- No serialization or external compatibility guarantee is established.

## Validation evidence

- The implementation context `/root/implement_config_0002` ran focused tests while developing.
  Its first run exposed and corrected only an invalid test expectation that Java enums must not
  inherit `Serializable`; the corrected focused run passed seven tests.
- After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:config:test`. It passed with `BUILD SUCCESSFUL` and five actionable tasks:
  two executed and three up-to-date. XML reports record 12 tests across three suites with zero
  failures, errors, or skips: `BackendIntentTest` 5, `CompileModeTest` 2, and
  `GraphOptimizationConfigTest` 5.
- The separate clean-context documentation pass
  `/root/implement_config_0002/config_0002_docs` applied the General, API/Javadoc, Architecture,
  User guide, Planning, and Example profiles. It independently reviewed the final source, focused
  tests, generated Javadoc, `BackendIntent`, architecture and training boundaries, API and user
  guides, glossary, planning status, model `GraphPhase`, compiler placeholder, dependency/build
  configuration, and the actual diff. It changed no executable Java behavior and therefore reused
  the recorded Java evidence without rerunning the suite.
- `./gradlew :modules:config:javadoc` passed with `BUILD SUCCESSFUL`. The generated package,
  `CompileMode`, and `GraphOptimizationConfig` pages were inspected for rendered descriptions,
  enum constants, record component, constructor, factories, accessor, current-versus-planned
  boundaries, and absence of Javadoc errors.
- `python3 /tmp/validate_synaptik_markdown.py` passed repository Markdown links, anchors, fences,
  final newlines, and trailing-whitespace validation.
- `git diff --check` passed. The combined scope command listed exactly the task's fifteen paths,
  and `git status --short` showed only those authorized modifications and additions.
- Manual status checks confirmed task 0002 is Complete, no master-plan row is Ready, config tasks
  0003–0008 and trace tasks 0003–0008 remain Draft, and no later config or trace task specification
  exists.
- The no-change audit confirmed `BackendIntent` source, tests, Javadocs, behavior, and public
  backend-contract dependency remain accurate and untouched. Because the new values contain only
  declarative requests and add no dependency or lifecycle behavior, no change was needed to the
  config build, backend-contract contracts, model `GraphPhase`, compiler and training/NN
  placeholders, `AGENTS.md`, `ARCHITECTURE.md`, lifecycle/module/dependency/partition explanations,
  ADR 0005, architecture tests, root build/settings, other modules, concrete backends, or
  conformance/integration tests.
- Repository-wide tests were not run, as required: this task changes neither dependencies nor a
  cross-module executable contract, and the config capability checkpoint or continuous
  integration owns that tier.

## Implementation notes

- Added the exact method-free three-value `CompileMode` enum and one-component
  `GraphOptimizationConfig` record with only `disabled()` and `standard()` factories.
- Added focused API-shape and value-behavior tests without changing `BackendIntent`, dependencies,
  build configuration, model `GraphPhase`, compiler behavior, or another module.
- Finalized both new type Javadocs, compile-package Javadoc, current training-graph status, Public
  and Compile API references, compile and autograd guides, glossary terminology, and synchronized
  config/trace/roadmap status.
- Kept every later config and trace row Draft without creating a detailed follow-up specification.

## Completion summary

- Completed changes: added immutable requested graph-scope and optional-optimization configuration
  leaves, focused tests, complete Javadocs, current-status documentation, glossary entries, and
  synchronized planning status.
- Files changed or created: exactly three production, two test, six documentation, and four
  planning paths listed under Affected files.
- Tests and validation: reused the final 12-test config module result; config Javadoc, repository
  Markdown, generated-page inspection, exact scope, status, and whitespace checks passed.
- Documentation-agent review: `/root/implement_config_0002/config_0002_docs` completed the required
  independent pass without executable changes or duplicate Java-test execution.
- Documentation impact: current config leaves are distinguished from planned aggregation,
  compiler interpretation, autograd, optimization passes, preparation, execution, and training.
- Javadoc review: finalized and generated the two new type pages and compile package page.
- Glossary impact: added compile-mode and graph-optimization-configuration definitions and their
  boundary from `GraphPhase` and compiler behavior.
- Unresolved issues: None.
- Follow-up required: None. Tasks 0003–0008 remain ordered Draft work.

Status: Complete

# Task 0002: [Compile](../../../../glossary.md#compile) Modes and [Graph Optimization Configuration](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig)

> **Editorial reading key — not part of the historical record.**
>
> CompileMode requests which graph is built; GraphOptimizationConfig permits optional exact
> transformations. Neither value invokes Compiler, computes derivatives or runs an optimizer
> update. Disabling optional optimization cannot disable mandatory verification and canonical
> representation. TRAINING_STEP is graph-scope vocabulary, not a training-session resource. The
> original statements that consumers were not yet implemented remain milestone history; current
> consumption is documented separately.
>
> Current reading: [hard intent, compile scope and optional optimization](../../../../reference/compiler/configuration.md#current-compile-configuration-inputs); [ordinary and advanced Engine compile boundaries](../../../../reference/compiler/entry-points.md#current-ordinary-and-advanced-engine-compile-boundaries).

## Status

Complete

## Goal

Add the two remaining [leaf](../../../../glossary.md#term-leaf) values needed to describe [graph](../../../../glossary.md#graph) [scope](../../../../glossary.md#term-scope) and optional [backend](../../../../glossary.md#backend)-neutral
optimization before a later task aggregates [compile](../../../../glossary.md#compile) configuration.

[Mental model](../../../../glossary.md#term-mental-model):

```text
CompileMode.FORWARD_ONLY
CompileMode.FORWARD_AND_BACKWARD
CompileMode.TRAINING_STEP

GraphOptimizationConfig.disabled()
GraphOptimizationConfig.standard()
```

These values record requested compile semantics only. They neither [capture](../../../../glossary.md#term-capture-meanings) nor transform a graph,
construct [autograd](../../../../glossary.md#autograd), invoke an [optimizer](../../../../glossary.md#term-optimizer), choose [backend ownership](../../../../glossary.md#backend-ownership), or expose [compiler](../../../../glossary.md#term-compiler) pass
implementation details.

## [Scope](../../../../glossary.md#term-scope)

- Add public enum [`CompileMode`](../../../../glossary.md#compile-mode--compilemode) in `io.github.pho001.synaptik.config.compile`.
  - It contains exactly `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and `TRAINING_STEP` in that order.
  - `FORWARD_ONLY` requests [forward graph](../../../../glossary.md#forward-graph) construction and requested forward publications only.
  - `FORWARD_AND_BACKWARD` requests later compiler autograd expansion plus combined forward and
    backward compile-time graph work.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph))</span>
  - `TRAINING_STEP` records the architecture's training-step direction. It does not itself add an
    optimizer, optimizer-update graph, training session, schedule, or execution behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [training session glossary](../../../../glossary.md#training-session--trainingsession))</span>
  - It adds no project field, method, constructor, nested type, alias, phase, publication setting,
    gradient target, optimizer setting, or metadata.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [publication glossary](../../../../glossary.md#term-publication); [gradient glossary](../../../../glossary.md#term-gradient); [target glossary](../../../../glossary.md#term-target); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add public record [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) in the same package.
  - It contains exactly one [boolean](../../../../glossary.md#data-type--datatype) component: `optionalOptimizationsEnabled`.
  - It accepts and retains either primitive value without [validation](../../../../glossary.md#term-validation) or [normalization](../../../../glossary.md#term-normalization).
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
  - `true` permits the [compiler](../../../../glossary.md#term-compiler)'s standard optional semantics-preserving optimization pipeline.
  - Neither value controls graph capture, topological ordering, shape/data-type inference,
    validation, mandatory canonical representation, autograd required by the compile mode,
    publication binding, backend-neutral planning, preparation, or execution.<span data-glossary-navigation="true"> ([graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [topological glossary](../../../../glossary.md#term-topological-order); [shape glossary](../../../../glossary.md#shape); [inference glossary](../../../../glossary.md#term-inference); [canonical glossary](../../../../glossary.md#term-canonical-form); [compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [binding glossary](../../../../glossary.md#term-binding); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
  - `standard()` promises no public pass list, pass order, internal graph shape, or implementation
    strategy. Compiler tasks own those choices while preserving observable semantics.
- Add no optimization level enum, pass-specific booleans, custom pass list, class reference,
  registry, plugin, budget, timeout, seed, numerical relaxation, approximate-math permission,
  backend fusion switch, or debug/tracing option.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [plugin glossary](../../../../glossary.md#term-plugin); [seed glossary](../../../../glossary.md#term-seed); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Update compile-package Javadoc to cover the three current standalone compile configuration
  values and their current-versus-planned boundaries.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Add one focused test for `CompileMode` exact enum/API shape, order, identity, and absence of
  fields, methods, nesting, aliases, or behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add one focused test for `GraphOptimizationConfig` exact record/API shape, direct construction,
  factories, primitive retention, freshness, value semantics, and absence of pass/lifecycle
  surfaces.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Finalize Javadocs and update the focused training-graph status explanation, Public API, Compile
  API, compile workflow guide, autograd guide, glossary, config master plan, trace interleave
  status, and roadmap in the same overall change.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `CompileConfig`, aggregation, defaults for the later aggregate, builder, loader, environment or
  system-property configuration, serialization, persistence, or compatibility schema<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [schema glossary](../../../../glossary.md#term-schema))</span>
- changing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), [`BackendRequirement`](../../../../glossary.md#backend-requirement--backendrequirement), backend-contract source/build/tests, hard-target
  evaluation, backend preference, partition scoring, profiles, calibration, or tuning<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [target glossary](../../../../glossary.md#term-target); [evaluation glossary](../../../../glossary.md#term-evaluation); [partition glossary](../../../../glossary.md#partition))</span>
- compiler graph capture, traversal, indexing, inference, validation, canonicalization,
  optimization passes, autograd implementation, publication binding, planning orchestration, or
  diagnostics<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [autograd glossary](../../../../glossary.md#autograd); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [planning glossary](../../../../glossary.md#term-planning))</span>
- defining a public list or stable identity for dead-code elimination, common-subexpression
  elimination, constant folding, algebraic simplification, post-autograd cleanup, or another pass<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [dead-code elimination glossary](../../../../glossary.md#term-dead-code-elimination-dce); [common-subexpression elimination glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- aggressive, unsafe, experimental, size, speed, memory, debug, deterministic, backend-specific,
  or numerically relaxed optimization modes
- approximate math, changed floating-point association, tolerance, reproducibility, deterministic
  execution, or backend numerical policy<span data-glossary-navigation="true"> ([floating-point glossary](../../../../glossary.md#term-floating-point); [tolerance glossary](../../../../glossary.md#term-tolerance); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- `GraphPhase` changes, backward-only mode, evaluation mode, inference alias, optimizer mode,
  gradient publication policy, optimizer choice, training session, or train/eval module state<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [gradient glossary](../../../../glossary.md#term-gradient); [training session glossary](../../../../glossary.md#training-session--trainingsession); [module glossary](../../../../glossary.md#term-module))</span>
- implementing the future compiled [optimizer](../../../../glossary.md#term-optimizer) [graph](../../../../glossary.md#graph) suggested by `TRAINING_STEP`; any architecture
  evolution needed for optimizer-update graph work remains explicit later work
- prepare, run, publication, runtime, engine, backend lowering, fusion, kernel selection,
  executable units, storage, physical memory, or execution behavior<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [kernel glossary](../../../../glossary.md#kernel))</span>
- [trace](../../../../glossary.md#trace) DTOs, attributes, payloads, emission, or [serialization](../../../../glossary.md#term-serialization)
- dependencies, Gradle, Java version, architecture contract, architecture tests, another module,
  concrete backends, conformance, integration, or detailed later task specifications<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- unrelated refactoring or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [Training combined forward/backward graph ADR](../../../../design/decisions/0005-training-combined-forward-backward-graph.md)<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Documentation profile index](../../../../developer-guide/documentation/README.md)
- [General style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [User guide style](../../../../developer-guide/documentation/user-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Example format](../../../../developer-guide/documentation/example-format.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Config master plan](../master-plan.md)
- [Config task 0001](0001-backend-intent-foundation.md)
- [Trace master plan](../../trace/master-plan.md)<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>
- [Public API](../../../../api/public-api.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Compile workflow guide](../../../../user-guide/compiling-graphs.md)
- [Autograd guide](../../../../user-guide/autograd.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../../../glossary.md#autograd))</span>
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/config` owns immutable declarative compile configuration; compiler owns interpretation
  and graph transformation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph))</span>
- Compile mode describes requested graph scope. It does not execute autograd, publish gradients,
  choose a runtime schedule, or transfer optimizer ownership into config.<span data-glossary-navigation="true"> ([Compile mode glossary](../../../../glossary.md#compile-mode--compilemode); [scope glossary](../../../../glossary.md#term-scope); [autograd glossary](../../../../glossary.md#autograd); [gradients glossary](../../../../glossary.md#term-gradient); [runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Optimization configuration may permit or suppress optional semantics-preserving compiler work,
  but must not expose compiler pass implementation or backend prepare choices.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>
- Required inference, validation, mandatory representation normalization, and other correctness
  work cannot be disabled through this value.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Backend-specific fusion, specialization, lowering, and kernel selection remain backend prepare
  concerns regardless of optimization configuration.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [specialization glossary](../../../../glossary.md#term-specialization); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel))</span>
- The task adds no [dependency](../../../../glossary.md#term-dependency) and changes no existing [dependency direction](../../../../glossary.md#java-platform-and-build-tools).
- Stop if implementation needs an additional optimization-policy type, compiler API, training
  type, another module, dependency edit, architecture decision, or pass-specific public surface.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing package extended:

- `io.github.pho001.synaptik.config.compile` — immutable public graph-compilation configuration
  leaves; task 0002 adds graph scope and optional-optimization permission<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scope glossary](../../../../glossary.md#term-scope))</span>

Types added:

- `io.github.pho001.synaptik.config.compile.CompileMode` — exact requested forward/backward/training-
  step compile-time graph scope<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- `io.github.pho001.synaptik.config.compile.GraphOptimizationConfig` — exact boolean permission for
  optional semantics-preserving compiler optimization<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>

Existing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) remains unchanged. Package [Javadoc](../../../../glossary.md#term-javadoc) is updated to describe the three
current standalone values. No subpackage, helper, service, [registry](../../../../glossary.md#term-registry), builder, [compiler](../../../../glossary.md#term-compiler) type, or
training type is added.

## Affected files

Production — exactly three paths:

- add `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/CompileMode.java`
- add
  `modules/config/src/main/java/io/github/pho001/synaptik/config/compile/GraphOptimizationConfig.java`
- [Javadoc](../../../../glossary.md#term-javadoc)-only update to
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

[Planning](../../../../glossary.md#term-planning) — exactly four paths:

- add and finalize this task
- `docs/planning/modules/config/master-plan.md`
- `docs/planning/modules/trace/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification: [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) source/tests/Javadocs, config build and architecture
[dependency](../../../../glossary.md#term-dependency) test, all [backend](../../../../glossary.md#backend)-contract source/tests/build, [model](../../../../glossary.md#term-model) `GraphPhase`, [compiler](../../../../glossary.md#term-compiler) placeholder,
training/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) placeholders, `AGENTS.md`, `ARCHITECTURE.md`, [lifecycle](../../../../glossary.md#lifecycle)/[module](../../../../glossary.md#term-module)/dependency/[partition](../../../../glossary.md#partition)-
scoring explanations, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0005, architecture tests, root build/settings, other modules, concrete
backends, [conformance](../../../../glossary.md#term-conformance)/integration tests, and all later config tasks.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact fifteen paths above. Stop if implementation requires another production type,
test, document, [dependency](../../../../glossary.md#term-dependency), package, [module](../../../../glossary.md#term-module), architecture change, [compiler](../../../../glossary.md#term-compiler)/training behavior, or
detailed follow-up specification. The user's standing instruction permits a necessary [Javadoc](../../../../glossary.md#term-javadoc)-
only path-count increase, but any expansion must be recorded and must not alter a completed
declaration or behavior.

## Acceptance criteria

- [`CompileMode`](../../../../glossary.md#compile-mode--compilemode) is a public enum containing exactly `FORWARD_ONLY`, `FORWARD_AND_BACKWARD`, and
  `TRAINING_STEP` in that order, with no project field, method, constructor argument, nested type,
  alias, metadata, or implemented interface.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [alias glossary](../../../../glossary.md#aliasing-and-arena); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Mode Javadocs state requested graph scope without claiming that compiler, autograd, optimizer,
  schedule, publication, preparation, or execution behavior is current.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [scope glossary](../../../../glossary.md#term-scope); [compiler glossary](../../../../glossary.md#term-compiler); [optimizer glossary](../../../../glossary.md#term-optimizer); [publication glossary](../../../../glossary.md#term-publication))</span>
- [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) is a public record containing exactly one [boolean](../../../../glossary.md#data-type--datatype)
  `optionalOptimizationsEnabled` component, a public canonical constructor, explicit documented
  accessor, exactly <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`disabled()` and `standard()` as added public factories, ordinary record
  object methods, and no other project field, method, nested type, interface, builder, or factory.
- Direct construction preserves either primitive value. `disabled()` returns fresh equal false
  values and `standard()` returns fresh equal true values.
- Disabled configuration suppresses only optional semantics-preserving compiler optimizations;
  it does not claim to suppress inference, validation, mandatory canonical representation,
  mode-required autograd, publication binding, planning, preparation, or execution.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding); [planning glossary](../../../../glossary.md#term-planning))</span>
- Standard configuration permits a later compiler-owned standard pipeline without promising exact
  passes, order, internal graph shape, aggressive rewrites, changed numerical semantics, backend
  fusion, or execution behavior.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [backend glossary](../../../../glossary.md#backend); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- No individual compiler-pass flag, pass identifier, class reference, plugin/registry, level,
  profile, budget, numerical-relaxation, backend, tracing, or lifecycle surface is added.<span data-glossary-navigation="true"> ([plugin glossary](../../../../glossary.md#term-plugin); [registry glossary](../../../../glossary.md#term-registry))</span>
- Existing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) declaration, bytecode/API shape, behavior, tests, Javadocs, dependency,
  and hard-requirement boundary remain unchanged.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests fail on enum order/API drift, record component/API drift, factory behavior drift,
  and forbidden added surfaces.
- Package Javadoc, training-graph status, Public API, Compile API, compile workflow, autograd guide,
  and glossary distinguish current construction values from planned compiler interpretation,<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile))</span>
  `CompileConfig`, [autograd](../../../../glossary.md#autograd), optimization passes, training execution, and [lifecycle](../../../../glossary.md#lifecycle) APIs.
- Config master plan marks task 0001 Complete and task 0002 as the sole Ready task before
  implementation; tasks 0003–0008 remain ordered Draft rows without detailed specifications.
  Trace master plan and roadmap record the current config interleave without resuming trace work.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Trace glossary](../../../../glossary.md#trace))</span>
- A separate clean-context documentation-focused pass finalizes both new type Javadocs, package
  Javadoc, and the ten documentation/planning paths after Java validation.
- Exactly one final config module suite, config Javadoc, exact fifteen-path scope, Markdown, final
  newlines, trailing whitespace, and <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`git diff --check` pass. Repository-wide validation remains
  deferred to the config capability checkpoint or CI because this task changes no dependency or
  cross-module executable contract.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) focused tests while developing. After executable Java stabilizes, run exactly one final config
[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:config:test
```

Record suite and test counts from XML reports. Then hand the actual diff and exact Java evidence
to a separate clean-context documentation-focused agent in the same overall change. That pass
independently inspects final source/tests, current [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), the architecture-defined mode
semantics, generated [Javadoc](../../../../glossary.md#term-javadoc), and directly affected documentation; applies General,
[API](../../../../glossary.md#term-application-programming-interface-api)/Javadoc, User guide, [Planning](../../../../glossary.md#term-planning), and Example profiles as relevant; finalizes both new type
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
focused automated API-[shape](../../../../glossary.md#shape) tests; confirm exact fifteen-path [scope](../../../../glossary.md#term-scope); confirm task 0002 Complete
with no [Ready](../../../../glossary.md#term-ready) row after implementation; and confirm no later config or [trace](../../../../glossary.md#trace) task specification
exists.

Do not run repository-wide tests in this task. The config master plan defers them to its next
recorded capability [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci).

## Dependencies

- Complete config task 0001 and its stable [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent) package/[dependency](../../../../glossary.md#term-dependency) foundation.
- The architecture-defined forward-only, forward-and-backward, and training-step meanings.
- Completed [model](../../../../glossary.md#term-model) [graph phase](../../../../glossary.md#graph-phase) and [provenance](../../../../glossary.md#provenance) foundations as read-only context only.

## Follow-up tasks

- [Backend](../../../../glossary.md#backend)-neutral [partition](../../../../glossary.md#partition)-scoring configuration and [ranking](../../../../glossary.md#term-ranking) preference remain Draft task 0003.
- [Immutable](../../../../glossary.md#term-immutable) platform, backend, and tuning profiles remain Draft task 0004.
- `CompileConfig` aggregation of the current standalone leaves remains Draft task 0005.
- Compiler interpretation, autograd, graph optimization passes, publication binding, planning,
  trace translation, prepare, runtime, engine, training, and concrete backend behavior remain in
  their owning later project areas.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [graph optimization glossary](../../../../glossary.md#forward-graph-optimization); [publication glossary](../../../../glossary.md#term-publication); [binding glossary](../../../../glossary.md#term-binding); [planning glossary](../../../../glossary.md#term-planning); [trace glossary](../../../../glossary.md#trace); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>

## Architecture impact

Expected impact: None.

The task implements values already named and described by the [architecture contract](../../../../glossary.md#architecture-contract). The focused
training-[graph](../../../../glossary.md#graph) edit corrects implementation-status wording only. It changes no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings),
[dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) stage, [autograd](../../../../glossary.md#autograd) decision, [optimizer](../../../../glossary.md#term-optimizer) boundary, or [backend](../../../../glossary.md#backend) rule.
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
  method; later compiler code can interpret the enum explicitly.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [alias glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Represent optional optimization permission with one boolean rather than public compiler pass
  switches or a speculative hierarchy.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- Name the component `optionalOptimizationsEnabled` to make clear that correctness-required work
  is never disabled.
- Provide only `disabled()` and `standard()` conveniences. The later aggregate owns its default;
  this task does not silently choose one.
- Let `standard()` denote a compiler-owned, semantics-preserving pipeline whose internal passes
  and ordering may evolve. Public configuration promises requested policy, not graph structure.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Add no `AGGRESSIVE` level because acceptable numerical reassociation, approximation,
  determinism, size/speed tradeoffs, and compatibility are not yet defined.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Keep `TRAINING_STEP` as declarative architecture vocabulary. It neither implements nor promises
  the future optimizer-update graph direction.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Update only the stale implementation-status sentence in the focused training-graph explanation;
  do not change its architecture meaning.

## Known limitations

- The standalone values are not yet accepted by `CompileConfig` or any [compiler](../../../../glossary.md#term-compiler) entry point.
- No current compiler interprets the selected mode or optimization permission.
- Standard optimization does not expose, freeze, or identify a pass set, pass order, internal
  graph structure, numerical strategy, or diagnostic representation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- Disabled optimization does not define every mandatory compiler transformation; later compiler
  contracts must distinguish correctness work from optional optimization consistently.
- `TRAINING_STEP` records future direction but does not make compiled [optimizer](../../../../glossary.md#term-optimizer) updates current.
- No [serialization](../../../../glossary.md#term-serialization) or external compatibility guarantee is established.

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context `/root/implement_config_0002` ran focused tests while developing.
  Its first run exposed and corrected only an invalid test expectation that Java enums must not
  inherit `Serializable`; the corrected focused [run](../../../../glossary.md#term-run) passed seven tests.
- After executable Java stabilized, that context ran exactly one final
  `./gradlew :modules:config:test`. It passed with `BUILD SUCCESSFUL` and five actionable tasks:
  two executed and three up-to-date. XML reports record 12 tests across three suites with zero
  failures, errors, or skips: `BackendIntentTest` 5, `CompileModeTest` 2, and
  `GraphOptimizationConfigTest` 5.
- The separate clean-context documentation pass
  `/root/implement_config_0002/config_0002_docs` applied the General, API/Javadoc, Architecture,
  User guide, Planning, and Example profiles. It independently reviewed the final source, focused
  tests, generated Javadoc, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>[`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), architecture and training boundaries, API and user
  guides, glossary, planning status, model <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`GraphPhase`, compiler placeholder, dependency/build
  configuration, and the actual diff. It changed no executable Java behavior and therefore reused
  the recorded Java evidence without rerunning the suite.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- `./gradlew :modules:config:javadoc` passed with `BUILD SUCCESSFUL`. The generated package,
  [`CompileMode`](../../../../glossary.md#compile-mode--compilemode), and [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) pages were inspected for rendered descriptions,
  enum constants, record component, constructor, factories, accessor, current-versus-planned
  boundaries, and absence of Javadoc errors.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>
- `python3 /tmp/validate_synaptik_markdown.py` passed repository Markdown links, anchors, fences,
  final newlines, and trailing-whitespace validation.
- `git diff --check` passed. The combined scope command listed exactly the task's fifteen paths,
  and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git status --short` showed only those authorized modifications and additions.
- Manual status checks confirmed task 0002 is Complete, no master-plan row is Ready, config tasks
  0003–0008 and trace tasks 0003–0008 remain Draft, and no later config or trace task specification
  exists.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [trace glossary](../../../../glossary.md#trace))</span>
- The no-change audit confirmed `BackendIntent` source, tests, Javadocs, behavior, and public
  backend-contract dependency remain accurate and untouched. Because the new values contain only
  declarative requests and add no dependency or lifecycle behavior, no change was needed to the
  config build, backend-contract contracts, model <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`GraphPhase`, compiler and training/NN
  placeholders, <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`AGENTS.md`, `ARCHITECTURE.md`, lifecycle/module/dependency/partition explanations,
  ADR 0005, architecture tests, root build/settings, other modules, concrete backends, or
  conformance/integration tests.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [partition glossary](../../../../glossary.md#partition); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Repository-wide tests were not run, as required: this task changes neither dependencies nor a
  cross-module executable contract, and the config capability checkpoint or continuous
  integration owns that tier.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [continuous integration glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Implementation notes

- Added the exact method-free three-value [`CompileMode`](../../../../glossary.md#compile-mode--compilemode) enum and one-component
  [`GraphOptimizationConfig`](../../../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) record with only `disabled()` and `standard()` factories.
- Added focused [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) and value-behavior tests without changing [`BackendIntent`](../../../../glossary.md#backend-intent--backendintent), dependencies,
  build configuration, model <span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>`GraphPhase`, [compiler](../../../../glossary.md#term-compiler) behavior, or another [module](../../../../glossary.md#term-module).
- Finalized both new type Javadocs, compile-package Javadoc, current training-graph status, Public
  and Compile API references, compile and autograd guides, glossary terminology, and synchronized
  config/trace/roadmap status.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [Javadoc glossary](../../../../glossary.md#term-javadoc); [graph glossary](../../../../glossary.md#graph); [autograd glossary](../../../../glossary.md#autograd))</span>
- Kept every later config and [trace](../../../../glossary.md#trace) row Draft without creating a detailed follow-up specification.

## Completion summary

- Completed changes: added immutable requested graph-scope and optional-optimization configuration
  leaves, focused tests, complete Javadocs, current-status documentation, glossary entries, and
  synchronized planning status.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: exactly three production, two test, six documentation, and four
  planning paths listed under Affected files.
- Tests and validation: reused the final 12-test config module result; config Javadoc, repository
  Markdown, generated-page inspection, exact scope, status, and whitespace checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module))</span>
- Documentation-agent review: `/root/implement_config_0002/config_0002_docs` completed the required
  independent pass without executable changes or duplicate Java-test execution.
- Documentation impact: current config leaves are distinguished from planned aggregation,
  compiler interpretation, autograd, optimization passes, preparation, execution, and training.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: finalized and generated the two new type pages and [compile](../../../../glossary.md#compile) package page.
- Glossary impact: added compile-mode and graph-optimization-configuration definitions and their
  boundary from `GraphPhase` and [compiler](../../../../glossary.md#term-compiler) behavior.
- Unresolved issues: None.
- Follow-up required: None. Tasks 0003–0008 remain ordered Draft work.

Status: Complete

# Task 0014: [Runtime](../../../../glossary.md#term-runtime) Architecture Enforcement

> **Editorial historical reading key.**
>
> Read this as a historical enforcement checkpoint: the architecture-test suite consumes repository
> build declarations and Java source text through JDK file APIs, not live Runtime objects or backend
> values. Its exact dependency list and exhaustive hot/non-hot manifest are review locks; they do not
> become semantic architecture authority or a permanent inventory of today's sources. The synthetic
> negative cases exercise the checker without transferring resources or executing a model.
>
> The current [prepared
> runner](../../../../reference/runtime/orchestration.md#current-prepared-runner) explains the
> protected execution boundary: Runtime traverses already-prepared work using bound references rather
> than inspecting Model graph operations. Current [cold invocation
> binding](../../../../reference/runtime/executables.md#current-prepared-executable-and-bound-invocation)
> assigns compatibility checks and typed invocation construction to backend hooks; invocations own
> neither state nor representations. Historical milestone closure and Engine absence remain dated
> assertions, not claims about current coverage or execution availability.

## Status

Complete

## Goal

Resolve the final [Runtime](../../../../glossary.md#term-runtime) contract-closure finding, `ARCHITECTURE-ENFORCEMENT-001`, with one
focused, durable architecture-test suite. The suite must make two existing authoritative rules
falsifiable:

- `modules/runtime` has exactly its currently approved inward Gradle project dependencies and no
  dependency on Engine or a concrete backend; and<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Runtime hot-path production code does not use [Model](../../../../glossary.md#term-model) [`Operation`](../../../../glossary.md#operation) or `CompiledNode` [graph](../../../../glossary.md#graph) state.

This task adds enforcement for existing rules. It does not change Runtime behavior, [dependency
direction](../../../../glossary.md#java-platform-and-build-tools), the hot-path definition, or the [architecture contract](../../../../glossary.md#architecture-contract).

[Mental model](../../../../glossary.md#term-mental-model):

```text
Runtime build declaration                     Runtime production-source inventory
  -> exact three approved project edges         -> exhaustive hot/non-hot classification
  -> no extra module/backend edge                -> named hot subset
                                                    -> no Operation or CompiledNode
                         \                         /
                          durable architecture test
                                     |
                         close ARCHITECTURE-ENFORCEMENT-001
```

## [Scope](../../../../glossary.md#term-scope)

- Add one focused [JUnit](../../../../glossary.md#term-junit) suite,
  `io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest`, under
  the existing `testing/architecture-tests` test package.
- Read `modules/runtime/build.gradle.kts` as repository text, collect every line that declares a
  Gradle project dependency, and require the exact ordered list:<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>

  ```text
  implementation(project(":modules:config"))
  implementation(project(":modules:backend-contract"))
  implementation(project(":modules:trace"))
  ```

- Treat exact equality as the dependency boundary. It must reject an added public or internal
  Model, Compiler, Planning, Prepare, Engine, concrete-backend, or other project edge; a removed
  approved edge; a changed visibility; a duplicate; and an order drift that makes the declaration
  differ from the audited current contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare))</span>
- Keep the architecture-test project dependency-free beyond the root-provided JUnit test
  configuration. Inspect repository build/source text through JDK file APIs, matching the current
  architecture-test style. Do not add a <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`testImplementation` dependency on Runtime or another
  project merely to inspect it.
- Discover every `*.java` file below `modules/runtime/src/main/java`, normalize each path relative
  to that source root with `/` separators, and require an exact exhaustive source [manifest](../../../../glossary.md#term-manifest).
- [Partition](../../../../glossary.md#partition) that manifest into two explicit, [immutable](../../../../glossary.md#term-immutable) test-owned sets:
  - the hot-path set below; and
  - every other current Runtime production source, including package documentation, as the
    reviewed non-hot set.
- Require the two sets to be disjoint and their union to equal the discovered production-source
  inventory exactly. A new, removed, renamed, or silently unclassified Runtime source file must
  fail the test until the architecture owner deliberately reviews and classifies it.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Classify the current direct execution/state path as exactly these production sources:

  ```text
  io/github/pho001/synaptik/runtime/execution/BoundBufferTransfer.java
  io/github/pho001/synaptik/runtime/execution/BoundInvocation.java
  io/github/pho001/synaptik/runtime/run/BoundPublication.java
  io/github/pho001/synaptik/runtime/run/PreparedExecutionRunner.java
  io/github/pho001/synaptik/runtime/run/RunState.java
  ```

  `RunStateCreation` and immutable prepared recipes remain explicitly classified outside this
  direct traversal/action subset. The task does not reinterpret their existing Runtime contracts
  or remove their focused module tests.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Scan every hot-path source as text for the exact forbidden source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) [identities](../../../../glossary.md#term-identity-meanings):

  ```text
  io.github.pho001.synaptik.model.operation.Operation
  io/github/pho001/synaptik/model/operation/Operation
  io.github.pho001.synaptik.model.graph.CompiledNode
  io/github/pho001/synaptik/model/graph/CompiledNode
  ```

  The implementation may centralize this vocabulary in private test constants. Exact dotted
  identities catch imports and qualified Java references; slash identities cover binary-name
  spellings. Do not reject unrelated documentation or identifiers merely because they contain the
  words <span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant))</span>[`Operation`](../../../../glossary.md#operation) or `CompiledNode`.
- Keep the whole-module exact dependency assertion as the primary proof that Runtime cannot
  compile against Model or Compiler. The hot-source scan is the additional direct proof for the
  architecture rule's named execution boundary. Together, the absent Model/Compiler project edge
  and forbidden exact imports/qualified references prevent simple-name use from compiling.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Add focused synthetic failure-sensitivity coverage through private test helpers. At minimum,
  prove that the checker rejects:<span data-glossary-navigation="true"> ([sensitivity glossary](../../../../glossary.md#term-sensitivity))</span>
  - an added [Engine](../../../../glossary.md#term-engine) or concrete-[backend](../../../../glossary.md#backend) project [dependency](../../../../glossary.md#term-dependency);
  - an unclassified [Runtime](../../../../glossary.md#term-runtime) production source;
  - a hot-path `Operation` reference; and
  - a hot-path `CompiledNode` reference.
- Use fixed in-memory strings and path names for failure-sensitivity cases. Do not mutate the
  working tree, launch nested Gradle, copy repository sources, or create temporary production
  files.
- Preserve the existing three architecture suites and all Runtime module tests. The new suite
  complements their behavioral and bytecode-oriented checks; it does not move or duplicate those
  checks.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Update the architecture-test developer guide from “Runtime enforcement absent” to the exact
  implemented coverage. Do not claim that all architecture rules are now enforced.
- Finalize this task's evidence and status, then synchronize the Runtime master plan and roadmap.
  The historical Runtime 0011 audit artifact and completed 0011–0013 task evidence remain
  unchanged.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing `ARCHITECTURE.md`, an ADR, dependency rule, module responsibility, hot-path contract,
  or lifecycle<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- changing any Runtime production source, Runtime module test, Javadoc source, public API,
  behavior, package, or visibility<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- changing `modules/runtime/build.gradle.kts`, `testing/architecture-tests/build.gradle.kts`, root
  build/configuration, settings, Java version, dependency, plugin, or source-set structure<span data-glossary-navigation="true"> ([plugin glossary](../../../../glossary.md#term-plugin))</span>
- adding ArchUnit, ClassGraph, bytecode libraries, Gradle TestKit, a custom Gradle plugin, or any
  other dependency or general architecture-test framework<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- adding Runtime dependencies on Model, Compiler, Planning, Prepare, Engine, a concrete backend,
  or any new module<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- testing Runtime execution results, validation messages, resource ownership, cleanup,
  publication, transfer, scheduling, or concurrency behavior already owned by Runtime module tests<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership); [publication glossary](../../../../glossary.md#term-publication); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- enforcing every architecture rule named by `ARCHITECTURE.md`; this task closes only the audited
  Runtime dependency and hot-path gap
- scanning generated build outputs as the sole proof, depending on task execution order, or
  requiring Runtime classes on the architecture-test classpath<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath))</span>
- using an open-ended path [heuristic](../../../../glossary.md#term-heuristic) that lets a new hot class evade classification
- changing the [Runtime](../../../../glossary.md#term-runtime) 0011 historical `BLOCKING_GAP` verdict or rewriting its durable audit; the
  later remediation status belongs in 0014, the Runtime master plan, and roadmap
- implementing public result-value access, Prepare orchestration, Engine composition, a concrete
  backend, Trace run payloads, Config run policy, profiling, tuning, or another Runtime capability<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [run glossary](../../../../glossary.md#term-run))</span>
- creating Runtime 0015 or any later detailed specification
- unrelated refactoring, formatting, terminology, link, status, or documentation cleanup

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Architecture-test guide](../../../../developer-guide/architecture-tests.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [Developer-guide style](../../../../developer-guide/documentation/developer-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Runtime master plan](../master-plan.md)
- [Runtime contract closure audit](../runtime-contract-closure-audit.md)
- [Runtime 0011](0011-runtime-contract-closure-audit.md)
- [Runtime 0012](0012-run-state-shared-throwable-cleanup.md)
- [Runtime 0013](0013-general-architecture-status-correction.md)

## Architecture constraints

- `ARCHITECTURE.md` remains authoritative. Tests encode its existing Runtime rules; tests and
  planning documents do not become architecture authority.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [planning glossary](../../../../glossary.md#term-planning))</span>
- Runtime may depend only on current shared inward contracts and must not depend on Engine or a
  concrete backend. Concrete backend implementations may depend inward on Runtime, never the
  reverse.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- Runtime executes already-prepared schedules. Its direct traversal/actions must not consume
  Model <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>[`Operation`](../../../../glossary.md#operation) or `CompiledNode`, perform graph inspection, discover a backend, or select a
  route or kernel.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- Exact project-dependency enforcement is intentionally stricter than checking only two forbidden
  substrings. An unreviewed extra project edge is a boundary change and must fail closed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- The source manifest is an enforcement inventory, not a new production package taxonomy. Its
  hot/non-hot classification exists only in the test to prevent silent coverage gaps.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- Source inspection is appropriate because the architecture suite already reads repository
  Gradle/source structure, and the exact manifest makes file discovery explicit. The check must
  not rely on Runtime compilation order or a Runtime test dependency.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- If implementation evidence shows that a current source cannot be classified without changing
  the architecture meaning, or that the named rules require a production/build change, stop and
  report the conflict instead of weakening the test.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.testing.architecture` — owns repository-level architecture contract
  tests and already contains Config, Planning, and NN/Training dependency enforcement.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

Type placement:

- `io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest` — one
  package-private JUnit suite colocated with the other architecture tests because it validates
  repository structure rather than Runtime behavior.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>

No production package, [Runtime](../../../../glossary.md#term-runtime) test package, [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), or public surface changes.

## Affected files

Expected execution paths — exactly five:

- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/RuntimeDependencyAndHotPathContractTest.java`
- `docs/developer-guide/architecture-tests.md`
- `docs/planning/modules/runtime/tasks/0014-runtime-architecture-enforcement.md`
- `docs/planning/modules/runtime/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification:

- `AGENTS.md`, `ARCHITECTURE.md`, focused architecture explanations, and ADRs
- `modules/runtime/build.gradle.kts`, `testing/architecture-tests/build.gradle.kts`, root Gradle
  configuration, and <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`settings.gradle.kts`
- all [Runtime](../../../../glossary.md#term-runtime) production sources and Runtime [module](../../../../glossary.md#term-module) tests
- the existing Config, [Planning](../../../../glossary.md#term-planning), and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training architecture tests
- Runtime [API](../../../../glossary.md#term-application-programming-interface-api), Public API, glossary, [backend](../../../../glossary.md#backend) guide, generated Runtime Javadocs, and package pages
- the Runtime 0011 audit [artifact](../../../../glossary.md#term-artifact) and completed Runtime 0001–0013 task history
- [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), concrete-backend, [Trace](../../../../glossary.md#trace), Config, [conformance](../../../../glossary.md#term-conformance), and integration plans/sources

## Maximum [scope](../../../../glossary.md#term-scope)

Implementation may create or modify exactly the five expected paths above and no others. This
[Ready](../../../../glossary.md#term-ready) task file already exists and is finalized in place, so execution creates only the one test
source.

If implementation requires another Java/test file, a [fixture](../../../../glossary.md#term-fixture) file, a shared helper, a build file,
a [dependency](../../../../glossary.md#term-dependency), [Runtime](../../../../glossary.md#term-runtime) production/test source, an [API](../../../../glossary.md#term-application-programming-interface-api)/glossary/[Javadoc](../../../../glossary.md#term-javadoc) change, an [architecture
contract](../../../../glossary.md#architecture-contract) edit, or a sixth changed path, stop and propose a replan. Do not silently expand scope.

## Acceptance criteria

- The new architecture suite is package-private, uses only JDK/JUnit APIs already available to
  the test project, and adds no project or external dependency.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [JUnit glossary](../../../../glossary.md#term-junit))</span>
- Runtime's complete project-dependency line list is asserted equal, in order and visibility, to
  Config, Backend Contract, and Trace as three <span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace))</span>`implementation` dependencies.
- Any extra project [dependency](../../../../glossary.md#term-dependency)—including [Engine](../../../../glossary.md#term-engine) or a [concrete backend](../../../../glossary.md#concrete-backend)—fails the exact assertion.
- The test discovers all current Runtime production Java sources and proves that the explicit
  hot/non-hot manifests are disjoint and exhaustive.
- The exact five-file hot-path set is enforced. Adding, removing, renaming, or leaving a Runtime
  source unclassified fails with a diagnostic that identifies the inventory mismatch.
- Every hot-path source is rejected if it imports or uses the exact qualified [`Operation`](../../../../glossary.md#operation) or
  `CompiledNode` [identity](../../../../glossary.md#term-identity-meanings) in dotted or [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-name form.
- The checker does not reject unrelated Javadocs, string literals, or Java identifiers that only
  contain the words `Operation` or `CompiledNode`.
- Synthetic failure-sensitivity tests prove rejection of an Engine/concrete-backend edge, an
  unclassified source, <span data-glossary-navigation="true"> ([sensitivity glossary](../../../../glossary.md#term-sensitivity))</span>`Operation`, and `CompiledNode` without mutating repository files.
- Existing architecture tests remain unchanged and continue passing.
- No [Runtime](../../../../glossary.md#term-runtime) production/test/[Javadoc](../../../../glossary.md#term-javadoc) source or [Gradle](../../../../glossary.md#term-gradle)/build file changes.
- The architecture-test guide accurately lists Config, Planning, conditional NN/Training, and
  the new Runtime dependency/hot-path coverage while continuing to say that success proves only
  implemented focused assertions.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Runtime/Public APIs, glossary, backend guide, generated Runtime Javadocs, conformance tests, and
  integration tests receive reasoned no-change conclusions.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `ARCHITECTURE-ENFORCEMENT-001` is recorded resolved only after the new suite and final checkpoint
  pass. Runtime 0014 then becomes Complete and the Runtime milestone/roadmap area becomes Complete.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Historical Runtime 0011 remains an unchanged `BLOCKING_GAP` audit of its point-in-time evidence;
  completed 0012–0013 and their evidence remain unchanged.
- No later Runtime task or detailed specification is created. Prepare 0003 and downstream owners
  remain at their existing statuses.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- Focused, final combined checkpoint, Markdown, scope, status, history-preservation, link, and
  whitespace checks all pass.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Validation tier: [Runtime](../../../../glossary.md#term-runtime) capability-closure [checkpoint](../../../../glossary.md#term-checkpoint). Focused development may [run](../../../../glossary.md#term-run) the new suite
as needed. After executable test source stabilizes, run exactly one final combined repository,
architecture, and Runtime-[Javadoc](../../../../glossary.md#term-javadoc) checkpoint:

```bash
./gradlew :testing:architecture-tests:test \
  --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest
./gradlew test :testing:architecture-tests:test :modules:runtime:javadoc
```

The first command is focused development evidence. The second command is the sole final suite and
capability checkpoint; do not separately rerun Runtime or architecture suites around it. Record
suite/test totals from final [JUnit](../../../../glossary.md#term-junit) XML, including the new architecture suite.

The separate documentation-focused pass must inspect the final test, source [manifest](../../../../glossary.md#term-manifest), actual
Runtime build/source tree, generated Runtime Javadocs, architecture-test guide, audit, master plan,
and roadmap. It must finalize only the four permitted Markdown paths already in the five-path
[scope](../../../../glossary.md#term-scope). It reuses successful Java/checkpoint evidence unless it changes executable Java, which is
not authorized. Run:

```bash
python3 /tmp/validate_synaptik_markdown.py \
  docs/developer-guide/architecture-tests.md \
  docs/planning/modules/runtime/tasks/0014-runtime-architecture-enforcement.md \
  docs/planning/modules/runtime/master-plan.md \
  docs/planning/roadmap.md
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass reuses the Runtime Javadoc output from the combined checkpoint because
Java and Javadoc source changes are forbidden. It must not rerun the Javadoc task merely to
duplicate valid evidence. If inspection proves that output stale or absent, it may run
`./gradlew :modules:runtime:javadoc` once and must record the concrete reason.

Also verify mechanically:

- exactly five task-owned paths changed and no [Gradle](../../../../glossary.md#term-gradle)/production/Runtime-test path changed;
- the architecture suite contains the exact three approved dependency strings and exact five
  hot-path source paths;
- the discovered Runtime production inventory matches the test manifest;
- synthetic negative cases cover both forbidden types, [dependency](../../../../glossary.md#term-dependency) drift, and manifest drift;
- Runtime 0001–0013 task files and the audit [artifact](../../../../glossary.md#term-artifact) have no diff;
- task/master/roadmap agree on 0014 status and Runtime milestone closure;
- no Runtime 0015 task file or detailed later specification exists; and
- `ARCHITECTURE.md`, ADRs, dependencies, APIs, glossary, conformance, and integration paths have no
  diff.

[Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) is not triggered because no [concrete backend](../../../../glossary.md#concrete-backend) behavior changes. Integration is
not triggered independently because [Engine](../../../../glossary.md#term-engine) still exposes no end-to-end execution path; the root
checkpoint is sufficient repository evidence.

## Documentation-focused pass

After the implementation context has stabilized the test and recorded both Java commands, hand
the actual diff and exact evidence to a separate clean-context documentation-focused agent in the
same overall change. That agent must:

- read the documentation rules and General, Developer-guide, and [Planning](../../../../glossary.md#term-planning) profiles;
- independently inspect the new suite, current Runtime build/source inventory, audit finding, and
  generated Runtime Javadocs;
- update the architecture-test guide to describe exactly the new coverage and its limitations;
- finalize this task, [Runtime](../../../../glossary.md#term-runtime) master plan, and roadmap evidence/status;
- record reasoned no-change conclusions for Runtime/Public APIs, glossary, backend guide,
  Javadocs, architecture/ADRs, Gradle/dependencies, Runtime source/tests, conformance/integration,
  Prepare, Engine, concrete backends, Trace, Config, and tuning;<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace))</span>
- run the specified documentation/scope/status checks without repeating successful Java tests;
  and<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [scope glossary](../../../../glossary.md#term-scope))</span>
- mark 0014 Complete and the Runtime milestone Complete only after all evidence passes.

## Dependencies

- [Runtime](../../../../glossary.md#term-runtime) 0011 — Complete audit that identified `ARCHITECTURE-ENFORCEMENT-001`.
- Runtime 0012 — Complete cleanup remediation.
- Runtime 0013 — Complete architecture-status documentation remediation.
- Current Config, Planning, and conditional NN/Training architecture tests — preserved pattern and
  baseline.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Java 26 root [JUnit](../../../../glossary.md#term-junit) configuration — unchanged.

No new [module](../../../../glossary.md#term-module), library, build, architecture, [backend](../../../../glossary.md#backend), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), [Trace](../../../../glossary.md#trace), Config, or tuning
[dependency](../../../../glossary.md#term-dependency) is required.

## Follow-up tasks

None inside [Runtime](../../../../glossary.md#term-runtime) for the audited 0001–0010 contract milestone if this task passes. Public
result-value access, [Prepare](../../../../glossary.md#prepare) orchestration, [Engine](../../../../glossary.md#term-engine) composition, [concrete backend](../../../../glossary.md#concrete-backend) execution, typed
[run](../../../../glossary.md#term-run) tracing, Config run policy, and tuning retain their recorded downstream [owners](../../../../glossary.md#term-owner); they are not
Runtime 0015 tasks inferred from this closure.

Do not create another detailed task specification from this task.

## Architecture impact

Expected impact: None.

The task adds tests for existing authoritative rules. If implementation requires a new or changed
rule, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), hot-path definition, production annotation/marker, test-only export,
or architecture exception, stop and report the decision instead of editing `ARCHITECTURE.md`.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, focused Runtime/dependency architecture explanations,
documentation/planning rules and profiles, roadmap, Runtime master plan, completed Runtime
0011-0013, task 0014, current Runtime build/production/test inventories, current architecture-test
build/tests, architecture-test guide, Runtime/Public APIs, glossary, and Java 26 root build.

Implement Runtime 0014 exactly inside its five authorized paths. Add only one focused
RuntimeDependencyAndHotPathContractTest using JDK/JUnit repository-text inspection, the exact
Runtime Gradle project-dependency lock, exhaustive disjoint hot/non-hot source manifests, the exact
five-source hot-path Operation/CompiledNode exclusion, and in-memory negative sensitivity cases.
Add no dependency, build change, Runtime production/test change, general test framework,
architecture change, or later task. Stop on architecture, inventory, mechanism, affected-file, or
maximum-scope conflict.

Run the focused suite while developing and exactly one final combined repository/architecture/
Runtime-Javadoc checkpoint after executable Java stabilizes. Then hand the actual diff and exact
evidence to a separate clean-context documentation-focused agent in the same overall change. That
agent must inspect final source/tests, finalize the architecture-test guide and permitted planning
records, run Markdown/scope/status/whitespace checks, record reasoned no-change conclusions, and
reuse successful Java evidence unless executable behavior changes or it records a concrete stale-
evidence reason.

Do not mark 0014 Complete or the Runtime milestone Complete until both passes and every acceptance
criterion succeed. Do not create Runtime 0015 or another detailed specification.
```

## Local decisions

- The suite keeps its exact Gradle dependency lock, exhaustive source manifest, hot/non-hot
  classification, and forbidden identity vocabulary private to the architecture-test package.
  This makes the existing rules fail closed without adding Runtime or test-project dependencies.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [manifest glossary](../../../../glossary.md#term-manifest); [identity glossary](../../../../glossary.md#term-identity-meanings); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Known limitations

- The suite enforces the selected Runtime dependency and hot-path rules, not every architecture
  rule in the repository.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Source-manifest maintenance is intentional: a new Runtime production file must be reviewed and
  explicitly classified before architecture tests pass.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- Text inspection proves declared dependencies and forbidden type references. Runtime behavioral
  and bytecode mechanics remain covered by their existing module tests.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [module glossary](../../../../glossary.md#term-module))</span>
- Exact dependency order is a review lock for the current build surface, not a new semantic
  dependency-order architecture rule.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context: the focused command
  `./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.RuntimeDependencyAndHotPathContractTest`
  passed. The final combined checkpoint<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `./gradlew test :testing:architecture-tests:test :modules:runtime:javadoc` passed with 53
  actionable tasks: 3 executed and 50 up-to-date. No executable Java changed afterward.
- Final [JUnit](../../../../glossary.md#term-junit) XML records 3 tests for `RuntimeDependencyAndHotPathContractTest`, with 0 skipped,
  failures, or errors; the four architecture suites total 6 tests with the same result.
- Documentation context `/root/runtime0014_docs` inspected the final suite, exact Runtime Gradle
  dependency lines, 25-file production-source inventory and manifest, hot-path identities,
  synthetic negative cases, and generated Runtime Javadocs. It reused the successful Java/Javadoc
  evidence and did not rerun Java or Gradle tasks.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [manifest glossary](../../../../glossary.md#term-manifest); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation validation, final scope/status/history checks, and whitespace validation passed;
  detailed command results are recorded in the completion summary.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- `RuntimeDependencyAndHotPathContractTest` adds the single permitted architecture suite. It
  asserts the three ordered inward dependencies, disjoint and exhaustive source manifests, the
  five direct hot-path files, and exact dotted/slash [`Operation`](../../../../glossary.md#operation) and `CompiledNode` exclusions.
  In-memory checks prove dependency, manifest, and both forbidden-identity failure paths.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [manifest glossary](../../../../glossary.md#term-manifest); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The suite enforces existing [Runtime](../../../../glossary.md#term-runtime) rules only. It resolves
  `ARCHITECTURE-ENFORCEMENT-001`; it does not alter the historical Runtime 0011
  `BLOCKING_GAP` verdict or its point-in-time evidence.

## Completion summary

- Completed changes: added the focused dependency-free Runtime architecture suite; finalized the
  architecture-test guide and synchronized Runtime 0014, the Runtime master plan, and roadmap.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Files changed or created: `RuntimeDependencyAndHotPathContractTest.java`; this task;
  `docs/developer-guide/architecture-tests.md`; `docs/planning/modules/runtime/master-plan.md`;
  and `docs/planning/roadmap.md`.
- Tests and validation: implementation evidence reused exactly as recorded above. This clean
  documentation pass ran <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`python3 /tmp/validate_synaptik_markdown.py` for the four permitted
  Markdown paths, `git diff --check`, exact changed/untracked-path and status checks, source-
  manifest/identity checks, preserved-history checks for Runtime 0001-0013 and the 0011 audit,
  status synchronization, and later-specification absence checks; all passed.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [identity glossary](../../../../glossary.md#term-identity-meanings); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation-agent review: `/root/runtime0014_docs` independently finalized the affected guide
  and planning records without changing executable Java.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: the guide now states the exact Runtime suite coverage and its focused
  limit. Runtime/Public APIs, glossary, backend guide, and generated Runtime Javadocs need no
  change because public contracts and Javadoc source did not change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- No-change conclusions: architecture/ADRs, Gradle/dependencies, Runtime production/tests,
  backend conformance, integration, Prepare, Engine, concrete backends, Trace, Config, and tuning
  are unchanged. No backend behavior changed, and Engine has no end-to-end execution path, so
  conformance/integration were not separately triggered.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [Trace glossary](../../../../glossary.md#trace))</span>
- Unresolved issues: None for the selected Runtime 0001-0010 closure milestone. Runtime 0011
  remains its unchanged historical `BLOCKING_GAP` audit.
- Follow-up required: None. Prepare 0003 remains Draft without a detailed specification; no
  Runtime 0015 or later specification was created.

Status: Complete

# Task 0006B3: Engine-Facing Complete Compile Integration Port

## Status

Complete

## Goal

Make the already implemented complete Compiler pipeline callable by the future Engine module
through one narrow public-for-module-integration port, while keeping the implementation owner
`GraphCompiler` and its current compile entries package-private.

The current complete `GraphCompiler.compile(...)` returns public immutable `CompileArtifacts`, but
a class in another Java package cannot call it because `GraphCompiler`, the method, and its
`CompileTimeConstantGraph.Ingress` parameter are package-private. A public bridge placed in the
same Compiler package can legally supply empty constant ingress and delegate without exposing
those implementation types or changing architecture.

## Scope

- Add public final, non-instantiable
  `io.github.pho001.synaptik.compiler.GraphCompilationPort` as a cross-module integration SPI.
- Add exactly one public static `compile(...)` method with this logical parameter order:
  `CompileMode`, ordered forward `Tensor` outputs, optional `FunctionalGradientRequest`,
  `GraphOptimizationConfig`, `BackendIntent`, `PartitionScoringConfig`, ordered
  `BackendCapabilityProvider` values, and ordered `BackendAvailabilitySnapshot` values.
- Return the existing public immutable `CompileArtifacts` directly.
- Delegate exactly once to the existing nine-argument complete `GraphCompiler.compile(...)` with
  `CompileTimeConstantGraph.Ingress.empty()`.
- Keep `GraphCompiler`, its constructor, its graph-stage entry, and its complete entry
  package-private with their existing signatures and behavior.
- Preserve declaration-order validation, exception identity and wrapping, provider/snapshot
  encounter order, graph semantics, optimization, publication, planning, logical memory,
  diagnostics, and derivative behavior.
- Add a focused same-package delegation/shape test and a distinct-package source-level test proving
  that an integration caller can invoke the port using only public types.
- Document the port as cross-module SPI, not as the recommended Engine end-user API, through the
  mandatory separate documentation-focused pass.

## Out of scope

- Making `GraphCompiler` or either current compile method public.
- A public explicit compile-time logical-splat ingress; current constant machinery stays private.
- `CompileConfig`, `PrepareConfig`, `RunOptions`, defaults, builders, or other Config work.
- `CompiledGraph`, Engine construction, standard built-in composition, backend registration,
  Engine exceptions, user-facade signatures, or any Engine code.
- Prepare orchestration, prepared execution, Runtime execution, typed input binding, result access,
  host materialization, one-shot execution, backward convenience, or tuning integration.
- Tensor `execute`/`backward` methods, mutable gradients, hidden autograd state, or Model/Runtime
  dependencies.
- Compiler graph transformations, new gradient behavior, recurrent BPTT, Conv3d gradients,
  exact/relaxed algebra changes, vendor CPU routes, or BFLOAT16 OpenBLAS work.
- Changes to Config, Planning, Prepare, Runtime, Engine, Backend Contract, Model, CPU, Gradle,
  architecture contracts/tests, backend conformance, or integration tests.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Compiler,
  Compile artifacts, Engine, dependency rules, and compile lifecycle.
- [`docs/architecture/lifecycle.md`](../../../../architecture/lifecycle.md)
- [`docs/architecture/module-boundaries.md`](../../../../architecture/module-boundaries.md)
- [`docs/architecture/dependency-rules.md`](../../../../architecture/dependency-rules.md)
- [`docs/planning/planning-guide.md`](../../../planning-guide.md)
- [`docs/planning/modules/compiler/master-plan.md`](../master-plan.md)
- [`docs/planning/modules/engine/master-plan.md`](../../engine/master-plan.md)

## Architecture constraints

- Compiler remains the owner of capture, validation, transformation, autograd, publication,
  planning orchestration, logical memory, diagnostics, and immutable compile artifacts.
- Compiler must not depend on Runtime, Prepare, Engine, or a concrete backend.
- The new public type is a Compiler-owned integration port required by Java accessibility. It is
  technically callable by any module but must be documented as SPI and must not become an
  ordinary Engine user-facade parameter or return type.
- The port accepts only current public inward-layer contracts and retains no live provider or
  availability snapshot in `CompileArtifacts`.
- Compile creates no physical storage, prepared schedule, executable, runtime state, route,
  kernel, backend lowering, backend discovery, or tuning decision.
- Engine remains the future composition root. This task performs no standard or advanced backend
  composition and creates no service locator or process-global state.
- No `Operation` or `CompiledNode` crosses into Runtime, and this task changes no Runtime surface.
- With no explicit constant ingress, every reachable provenance-free forward leaf remains
  caller-bindable. Compiler-generated request constants continue through existing internals.
- If the bridge cannot delegate without changing compile semantics, exposing another private type,
  or adding an architecture rule, stop and report the concrete gap.

## Package impact

Existing production package:

- `io.github.pho001.synaptik.compiler` owns the package-private implementation, public artifacts,
  public functional requests, and the new narrow module-integration port. Co-location is required
  because Java package access lets the port call `GraphCompiler` without widening it.
- The port uses existing public types from `io.github.pho001.synaptik.config.compile`,
  `io.github.pho001.synaptik.model.tensor`,
  `io.github.pho001.synaptik.planning.capability`, and
  `io.github.pho001.synaptik.backend.contract`; it adds no package or dependency edge.

Test packages:

- `io.github.pho001.synaptik.compiler` verifies direct delegation and package-private preservation.
- `io.github.pho001.synaptik.compiler.spi` verifies cross-package source accessibility using only
  public types; no production `spi` subpackage is added.

No Engine, API facade, or additional production package is introduced.

## Affected files

Expected production and test files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilationPortTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/spi/GraphCompilationPortPublicShapeTest.java`

Expected documentation and planning files finalized in the same overall implementation change:

- `docs/api/compile-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`, limited to current Compile and Compile-artifacts status statements
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md`
- `docs/planning/roadmap.md`

`GraphCompiler.java` and its existing tests are expected to remain unchanged. No other file is
expected to change. The documentation pass must record a reasoned no-change conclusion for
architecture pages, Training API, Config/Prepare/Runtime/Engine/CPU/tuning plans, Gradle files,
architecture tests, backend conformance, and integration tests unless implementation exposes a
concrete contradiction.

## Maximum scope

This task may create or modify at most:

- one new Compiler production Java file;
- two new Compiler test Java files;
- two explanatory API Markdown files and one glossary Markdown file; and
- three planning Markdown files, including this task specification;
- nine paths total.

If `GraphCompiler.java`, another production type/package/module, a dependency/build file, or a
tenth path is required, stop and propose the smallest follow-up or architecture decision.

## Acceptance criteria

- `GraphCompilationPort` is public, final, non-instantiable, and contains exactly one public static
  `compile` method with the eight inputs listed in Scope and `CompileArtifacts` return type.
- `GraphCompiler` remains package-private and retains exactly its current two package-private
  compile entries and signatures.
- The port performs no graph or plan work itself and delegates exactly once to the existing
  complete entry with empty explicit constant ingress.
- Forward-only port compilation is artifact-equivalent to direct package-private complete
  compilation with the same inputs and empty ingress, including publications, ownership,
  partitions, logical memory, constants, diagnostics, and derivative metadata.
- Backward-capable requests preserve the existing explicit request and mode contract; no new
  autograd behavior or inferred target/seed policy is introduced.
- Null, invalid request/mode, no-eligible-backend, and provider-failure behavior preserves current
  declaration-order behavior and failure identity, with no Engine exception translation.
- The distinct-package test compiles and invokes the port without reflection, private imports, or
  Engine dependencies.
- Same-package reflection/source checks prove `GraphCompiler` and both of its entries remain
  package-private.
- Public Javadoc identifies module-integration status and documents constant-free leaf treatment,
  parameter constraints/order, result semantics/nullability, and expected failure classes.
- Public documentation keeps the port out of the recommended end-user surface and does not claim
  an operational Engine lifecycle.
- No production/test/build file outside `modules/compiler` changes, and no Gradle file changes.
- A separate documentation-focused clean context finalizes affected Javadoc, API/planning text,
  and glossary impact before completion.

## Tests / validation

Implementation-focused checks:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilationPortTest --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest
./gradlew :modules:compiler:test
```

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
git diff --check
```

Manual validation:

- inspect `javap -public` for `GraphCompilationPort` and confirm exactly one public compile method
  with only public parameter and return types;
- inspect declared modifiers for `GraphCompiler` and both current compile methods and confirm they
  remain package-private;
- inspect delegation to confirm one call with `CompileTimeConstantGraph.Ingress.empty()` and no
  other behavior;
- scan Compiler production imports and Gradle files for forbidden outward dependencies;
- validate changed Markdown links, anchors, headings, fences, trailing whitespace, and newline at
  end of file;
- confirm the implementation stays within nine paths and changes no code outside Compiler.

Repository-wide validation is deferred to the Engine lifecycle capability checkpoint or CI
because this task adds one module-local integration entry without changing dependencies,
architecture rules, shared build configuration, or executable semantics.

## Dependencies

- Compiler 0005 complete compile artifacts and package-private complete entry — Complete.
- Compiler 0006B2 current validated compile pipeline — Complete.
- Config 0001–0003 standalone `BackendIntent`, `CompileMode`, `GraphOptimizationConfig`, and
  `PartitionScoringConfig` contracts — Complete.
- Current public Planning capability provider and Backend Contract availability snapshot inputs —
  Complete.
- Engine-frontier source inspection proving the Java accessibility blocker and same-package bridge
  feasibility — Complete as planning evidence.

## Follow-up tasks

- CPU 0010F, required next: add the smallest supported CPU lifecycle integration adapter and
  production schedule assembly without leaking CPU internals.
- Engine 0001, required after CPU 0010F: establish advanced explicit composition and the first
  representation-level lifecycle seam.
- Engine 0002–0004, ordered: deterministic standard built-in composition ordinary users can
  construct without naming CPU, then typed logical binding/result access and host materialization.
- Engine 0005–0006, ordered: Engine-owned one-shot forward and scalar-objective backward
  convenience after typed binding/result/materialization exist.
- Engine 0007 and tools/tuning 0002, later optional integration; neither blocks this task.
- Config aggregate/default work, Compiler 0006C, and Compiler 0007 remain independent deferred
  work.

## Architecture impact

Expected impact: None.

This task realizes an existing Compiler-to-Engine accessibility boundary with a same-package
public SPI. It changes no module ownership, dependency direction, lifecycle stage, explicit
backend-registration rule, or user-facing Engine API. If implementation requires any such change,
stop and report it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository on Compiler task 0006B3.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/compiler/master-plan.md, and
docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md in full.

Implement task 0006B3 exactly as specified. Add the narrow Compiler-owned
GraphCompilationPort integration SPI in the existing Compiler package; keep GraphCompiler and
both current compile entries package-private. Do not implement any Engine facade, backend
composition, Config aggregate, typed execution, or other out-of-scope work. Stop and report if
the port cannot be added within the recorded architecture and maximum scope. Do not commit or push.

After implementation and Compiler validation, hand the diff and evidence to a separate clean
documentation-focused context. That pass must follow docs/developer-guide/documentation-rules.md,
finalize affected Javadoc/API/planning text and glossary impact in the same overall change, reuse
successful Java evidence unless executable code changes, and complete the specified documentation
validation.

Update this task with implementation notes, exact validation evidence, documentation-pass identity
and result, completion summary, and final status. Do not mark it Complete before every acceptance
criterion and the documentation pass succeed.
```

## Local decisions

- `GraphCompilationPort` remains in the Compiler root package because that package owns the
  complete compile pipeline and Java package access permits delegation without widening
  `GraphCompiler` or its explicit constant-ingress type.
- The port exposes exactly the existing complete result and supplies empty explicit forward
  constant ingress. It adds no overload, request aggregate, defaults, wrapper artifact, or
  exception translation.
- Documentation applies the General Style, API/Javadoc, Planning, and Example profiles. The port
  is described as cross-module SPI for future lifecycle composition, never as the recommended
  ordinary-user API.

## Known limitations

- The port cannot declare explicit forward constants; every reachable provenance-free forward
  leaf remains caller-bindable. This is the task's intentional constant-free boundary.
- No production capability-provider composition, Engine facade, preparation, execution, typed
  input binding, publication delivery, or failure translation exists yet. CPU 0010F is the next
  Draft operational-lifecycle frontier.

## Validation evidence

Implementation context `01a09f06-c837-79a0-b9f2-8e8a62a9ae01` supplied the following successful
evidence, reused by this documentation context because no executable Java changed afterward:

- `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCompilationPortTest --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest`
  passed all five new tests with zero failures, errors, or skips.
- `./gradlew :modules:compiler:test` passed 246 tests with zero failures, errors, or skips.
- `javap` confirmed that `GraphCompilationPort` is public and final with exactly one public static
  eight-input `compile` method whose parameter and return types are public.
- `javap -p` confirmed that `GraphCompiler` and both existing `compile` entries remain
  package-private and retain their signatures.
- Bytecode inspection confirmed exactly one `CompileTimeConstantGraph.Ingress.empty()` call and
  one `GraphCompiler.compile(...)` call, with no other port behavior.
- The forbidden outward-import scan and `git diff --check` passed in the implementation context.

Documentation context `01a09f0c-b9cc-79a0-8881-4d48818daa6e` independently reviewed the final
source, both new tests, the delegate/privacy implementation, architecture and documentation
contracts, Compiler planning state, affected API references, Training API, and glossary. It
selected General Style, API/Javadoc, Planning, and Example profiles and recorded these results:

- The prior `./gradlew :modules:compiler:javadoc` run completed successfully with seven actionable
  tasks: two executed and five up-to-date. Generated `GraphCompilationPort` Javadoc contains the
  finalized SPI, empty-ingress, caller-bindable-leaf, result, and failure contracts. The glossary
  correction changed no Java or Javadoc, so this successful evidence was reused without rerun.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/compile-api.md docs/api/public-api.md docs/glossary.md docs/planning/modules/compiler/master-plan.md docs/planning/modules/compiler/tasks/0006b3-public-constant-free-complete-compile-entry.md docs/planning/roadmap.md`
  validated all six changed Markdown files, including local links and anchors, headings, fences,
  line endings, trailing whitespace, and final newlines.
- Direct source/test/diff review confirmed exact delegation, public-only cross-package
  accessibility, artifact equivalence coverage, backward-request preservation, provider-failure
  identity, declaration-order validation, and package-private implementation ownership.
- API text scans found no remaining claim in the changed Compile/Public API pages that a public
  compile call is absent or that complete compilation is only package-private. They also confirm
  that the port is not presented as an Engine facade and that no Engine prepare/run support is
  claimed.
- `docs/api/training-api.md` remains unchanged because its package-private statement describes the
  reverse-mode implementation path, not accessibility of the complete constant-free delegate.
- `GraphCompiler.java` and its Javadoc remain unchanged and accurate: the implementation owner and
  both compile entries are still package-private, while the new public type delegates to them.
- `docs/glossary.md` required no new term. Its existing Compile and Compile-artifacts entries were
  corrected narrowly because their claims that public compilation remained wholly planned and no
  public entry produced `CompileArtifacts` became false when `GraphCompilationPort` was added.
  The revised entries retain the constant-free SPI and absent ordinary-user Engine boundaries.
- Architecture contracts/pages, Config, Prepare, Runtime, Engine, CPU, and tuning plans, Gradle
  files, architecture tests, backend conformance, integration tests, and all other modules remain
  unchanged by this documentation context because the port realizes an already planned boundary
  without changing ownership, dependencies, or executable behavior.
- Final scope accounting contains exactly nine task-owned changed paths: one production Java
  file, two test Java files, two API documents, one glossary document, and exactly three planning
  documents. The six pre-existing coordinator planning paths outside this task remain
  byte-for-byte unchanged by this context. Their final SHA-256 values remain
  `ff207648733f8c6f121536d5a37790e755563958681147749fe702b6ee366cf1` (CPU),
  `c233952dea792252fd54db18ee780a20d7787bc339d44d5ce9d74f31fbf75cd2` (Config),
  `ce8e5de4e460e6dc623d73abdb47667232e1dad26e79c39306f4b414260a7cb7` (Engine),
  `39a20aebc4baa96a5b326e8016e040c9053d40e7e493fbe91f8894d88a8f0d40` (Prepare),
  `9c8507d2c56bcd0b9679bff612eb7e5bb24b652a00fb8232d2acabd0a2a89c3b` (Runtime), and
  `c856a255b5780fb5ac2b6a6d2cca2e2bf421981cbf84e532ae9bc82bfcb12d25` (tuning). The total
  worktree contains those six protected paths plus the nine task-owned paths, and the Git index
  remains empty.
- `git diff --check`, final glossary/API/status scans, task-path and total-worktree accounting,
  Ready-state uniqueness, protected-path hashes, and direct final diff review passed after the
  completion update. Compiler 0006B3 is Complete; CPU 0010F is the next Draft frontier without a
  detailed task and was not marked Ready.

## Implementation notes

The implementation added one non-instantiable public final class with one public static method.
The method delegates directly to the existing complete compiler overload and supplies
`CompileTimeConstantGraph.Ingress.empty()`. It does not copy inputs, catch or translate failures,
retain collaborators, or perform compile work itself.

The independent documentation pass refined every public parameter, result, nullability,
ordering, empty-ingress, and caller-visible failure statement. It added one bounded current
zero-node example to the Compile API, synchronized the Public API inventory, corrected the two
stale glossary status statements, and kept the port out of the recommended ordinary-user
lifecycle. No Java or Javadoc changed during the glossary correction, so neither Java tests nor
Javadoc generation were repeated; their prior successful evidence was reused.

## Completion summary

- Completed changes: added and documented the narrow public constant-free Compiler integration
  port, preserved package-private compiler implementation entries and behavior, and synchronized
  Compiler task/master-plan/roadmap status with CPU 0010F as the next Draft frontier.
- Files changed or created: one Compiler production source, two Compiler tests, two API reference
  pages, one glossary page, and exactly three Compiler/roadmap planning documents; nine task-owned
  paths total.
- Tests and validation: reused the implementation context's five focused tests, 246-test Compiler
  suite, `javap`, bytecode, import, and whitespace evidence plus the prior successful Compiler
  Javadoc generation; validated Markdown links, anchors, headings, fences, whitespace, line
  endings, final newlines, glossary/API wording, scope, protected paths, status uniqueness, and
  the final diff.
- Documentation-agent review: clean documentation context
  `01a09f0c-b9cc-79a0-8881-4d48818daa6e` completed the independent review using the General Style,
  API/Javadoc, Planning, and Example profiles.
- Documentation impact: `compile-api.md`, `public-api.md`, and the existing glossary Compile and
  Compile-artifacts entries now distinguish the callable cross-module SPI from the absent
  ordinary-user Engine lifecycle; `training-api.md` remains accurate without change.
- Javadoc review: finalized SPI status, constant-free leaf treatment, all parameters, result
  semantics/nullability, ordering, and expected failure classes/identity.
- Glossary impact: corrected two stale status statements without adding a new term.
- Unresolved issues: None.
- Follow-up required: None for this task. CPU 0010F remains the next Draft operational frontier;
  its detailed task is intentionally not created here.

Status: Complete

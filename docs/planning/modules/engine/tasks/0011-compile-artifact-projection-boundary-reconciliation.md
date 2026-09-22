# Task 0011: Compile-Artifact Projection Boundary Reconciliation

## Status

Complete

Frontier verification at launch: the user explicitly authorized this drift repair ahead of the
normal roadmap reassessment, Engine 0010 and the required Prepare/CPU foundations were Complete,
and no other task was `Ready` or `In progress`. This executed as the sole active frontier.

## Change class

Class C — this repairs the Compiler-to-Prepare-to-concrete-backend module edge, changes a
supported CPU service-provider interface (SPI), and preserves the complete prepare/tuning
lifecycle across Prepare, CPU, Engine, and one Metal schedule-context consumer.

The brief exceeds the 15 KB planning target because one atomic Class C cutover must preserve and
validate the ordinary path, two tuning phases, fallback, producerless resources, and two backend
assemblers across four modules; splitting those coupled acceptance criteria would hide boundary
coverage.

## Goal

Make Engine the only concrete-composition owner that passes `CompileArtifacts` into shared
Prepare orchestration. Reuse `GraphPreparation`'s existing projection and
`CpuPartitionAnalysisInputs`; let CPU receive only stable Prepare/Model/Planning projections and
CPU-owned configuration, while preserving current ordinary preparation, producerless constants,
schedule assembly, OpenBLAS eligibility, tuning candidates, execution, numerics, and cleanup.

Remove the CPU production dependency on `modules/compiler`.

## Scope

- Lift `GraphPreparation`'s existing projection behind an artifact-owning Prepare operation so
  Engine can obtain the exact `PrepareContext` for CPU input/candidate production. Ordinary and
  tuning paths may invoke it separately but share one implementation and add no second projection.
- Keep complete orchestration in `GraphPreparation`: artifact validation, all contexts before
  backend work, analysis, assignment, finalization, producerless-resource coverage, schedule
  assembly/validation, rollback, and `PreparedExecution` construction.
- Replace `PreparedScheduleContext`'s `CompileArtifacts` component with the minimal immutable
  stable facts schedule assembly uses: planned/prepared partition association, graph values,
  source roles/constants, ordered publication value IDs, memory plan, and buffer assignments.
  Validate those projections inside Prepare before any concrete assembler sees them.
- Let the existing `PreparedScheduleAssembler` collaboration contribute CPU physical geometry
  for each required producerless published constant from stable value, logical-memory, and scalar
  arguments before assignment. Shared Prepare identifies, orders, validates, assigns, and
  transacts the complete set; CPU owns physical geometry and the initialized recipe.
- Change `CpuBackendIntegration` from artifact orchestration to artifact-free formation/exposure
  of the CPU analysis inputs and one `PartitionPreparation`, plus its schedule assembler,
  capabilities, storage/materialization operations, and tuning collaborators. Engine calls
  `GraphPreparation` for ordinary and tuned recipes.
- Change local- and complete-plan CPU tuning to consume the exact projected partition context,
  retain only projection/CPU association state, and return a selected/trial backend preparation
  for Engine to compose. They must not accept, retain, hash, or prepare `CompileArtifacts`, and
  must not construct a complete `PreparedExecution` themselves.
- Derive existing OpenBLAS boundary-storage facts from the exact `PrepareContext` rather than
  from `CompileArtifacts`; do not introduce a replacement CPU analysis-input aggregate.
- Preserve CPU complete-plan compatibility identity from the partition projection and existing
  CPU candidate/configuration identities. Engine retains exact compiled-graph association for
  the complete representative workflow.
- Adapt the Metal schedule assembler mechanically to the stable context. Add no Metal capability,
  route, composition, or production Compiler dependency.
- Move `backends/cpu -> modules/compiler` from `implementation` to `testImplementation`; keep
  compiler-backed CPU fixtures as tests only, and lock the production edge with architecture
  tests.
- After executable work stabilizes, complete the mandatory independent documentation/review pass.

## Non-goals

- changing compile output, graph transformation, partitioning, memory assignment, route choice,
  tuning ranking/cache policy, numerical results, Runtime execution, or resource ownership
- adding mixed-backend scheduling, zero-node execution, a second graph/partition projection,
  generic backend discovery, a registry, or a service locator
- changing ordinary or advanced Engine user APIs, Tensor/Training APIs, or public computation
  workflows
- changing `ARCHITECTURE.md`, a scoped normative contract, or an ADR; if implementation needs
  any such change, stop and return to planning
- repairing autograd documentation or any later contract drift in this task

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  compile artifacts flow through Prepare before backend analysis.
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  Engine composes; Prepare stages analysis/finalization; Runtime executes prepared schedules.
- [`ARCHITECTURE.md` heading `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Engine owns explicit
  composition, Prepare owns shared transition/validation, and CPU owns concrete lowering/routes.
- [`runtime-prepare-engine.md` heading `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — Prepare projects exact stable facts and must not expose `CompileArtifacts` to CPU.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine owns compile/prepare orchestration and explicit backend wiring.
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [`CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU receives shared contracts/configuration and owns lowering, routes, physical recipes, and
  finalization, but not Engine or Compiler orchestration.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

- `modules/prepare/.../GraphPreparation.java`, `PreparedScheduleContext.java`,
  `PreparedScheduleAssembler.java`, `PartitionPreparation.java`, and
  `ProducerlessPublishedConstantResource.java` — one projection, stable schedule facts, and
  complete orchestration.
- `backends/cpu/.../CpuBackendIntegration.java`, `CpuLocalWorkloadTuning.java`,
  `CpuCompletePlanTuning.java`, `CpuBackendComposition.java`, `CpuPartitionAnalysisInputs.java`,
  `CpuPartitionPreparer.java`, `CpuOpenBlasRouteSelector.java`, and
  `CpuPreparedScheduleAssembler.java` — projection-only lifecycle and tuning.
- `modules/engine/.../CpuEngineBackendComposition.java` and `AdvancedEngine.java` — ordinary and
  tuning orchestration plus exact compiled-graph association.
- `backends/metal/.../MetalNegPreparedScheduleAssembler.java` — stable-context adaptation only.
- `backends/cpu/build.gradle.kts` and `testing/architecture-tests/.../CpuDependencyContractTest.java`
  — remove and enforce the production Compiler edge.
- Focused `GraphPreparation`/`PreparedScheduleContext`, CPU integration/composition/local- and
  complete-plan tuning, Engine lifecycle/tuning, `MetalNegPreparedExecution`, CPU dependency, and
  Engine model-autotuning integration tests corresponding to the changed symbols.
- `docs/architecture/dependency-rules.md`, `runtime-prepare-backend-boundary.md`,
  `docs/api/public-api.md`, `runtime-api.md`, `docs/backend-guide/cpu-backend.md`,
  `partition-preparer.md`, and `docs/glossary.md` — reconcile current explanations.

No new top-level production type is expected. If a named production type must be added or moved,
stop and return the task to planning rather than creating a duplicate projection or broad facade.

## Acceptance criteria

- No CPU production source imports a Compiler package, no supported CPU signature names a
  Compiler type, and the CPU build has no production dependency on `modules/compiler`.
- Only Engine/Prepare code handles `CompileArtifacts` during current CPU composition; CPU analysis,
  schedule assembly, producerless contribution, and both tuning phases receive stable projected
  facts and CPU-owned configuration only.
- The existing `GraphPreparation` projection is the single source of partition-local nodes,
  values, logical requirements, constants, and topology for ordinary and tuning paths.
- `PreparedScheduleContext` exposes no Compiler aggregate. CPU and Metal assemblers consume only
  validated stable facts required for their current recipes.
- Ordinary, selected Phase-1, trial/final Phase-2, and fallback preparation all pass through
  shared `GraphPreparation`; no CPU path independently assembles a complete execution.
- Source-only published constants keep exact role detection, graph-value order, byte geometry,
  initialized-per-run representation, publication order, and zero schedule-occurrence behavior.
- Current sole non-empty CPU partition rejection, OpenBLAS eligibility/decision authentication,
  complete-plan candidate identity, stale/foreign association rejection, schedule steps,
  outputs, numerics, concurrency, rollback, and close behavior remain unchanged.
- Removing the artifact-taking `CpuBackendIntegration` preparation methods and both CPU tuning
  handoffs/preparation methods is an intentional source and binary SPI break; repository callers
  migrate atomically. No deprecated bridge preserves Compiler exposure. Ordinary/advanced Engine
  user APIs remain source-compatible.
- `PreparedScheduleContext` record construction/component reflection and the artifact-taking CPU
  lifecycle/tuning descriptors are intentionally incompatible. Current CPU tuning compatibility
  remains versioned with the same reuse-scope and within-batch identity behavior; its
  session-scoped canonical bytes have no before/after byte-continuity promise.
- Architecture tests distinguish CPU's allowed compiler-backed test fixtures from the forbidden
  production edge. Prepare stays concrete-backend-free; Metal stays production-Compiler-free.
- Javadocs fully document every changed input, result, failure, ownership, association, and
  compatibility rule; explanatory docs remove the obsolete direct CPU/Compiler rationale.

## Validation

```bash
./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :backends:metal:test
./gradlew :testing:architecture-tests:test :testing:integration-tests:test
./gradlew :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc :backends:metal:javadoc
./gradlew test
rg -n 'io\.github\.pho001\.synaptik\.compiler|CompileArtifacts' backends/cpu/src/main
rg -n '^(\s*)(api|implementation|compileOnly|runtimeOnly)\(project\(\":modules:compiler\"\)\)' backends/cpu/build.gradle.kts
rg -n 'testImplementation\(project\(\":modules:compiler\"\)\)' backends/cpu/build.gradle.kts
git diff --check
git status --short -uall
```

The source and production-configuration `rg` commands must return no matches; the
`testImplementation` command must return exactly one match. Validate every changed Markdown
link/anchor, heading uniqueness, fence balance, final newline, and targeted glossary terms using
the repository helper when available, otherwise record the manual substitute. Inspect the exact
changed-path list and public reflection tests for accidental API or scope expansion.

Repository-wide validation is required because this changes a dependency edge, shared Prepare
SPI, concrete-backend SPI, build configuration, and multiple modules.

## Dependencies and follow-up

- Engine 0010, Prepare 0003A/0005/0006, and CPU 0010F/0010H/0010I/0010J are Complete and supply
  the current projection, producerless, ownership, lifecycle, and tuning behavior being retained.
- This user-authorized drift-remediation exception precedes the normal roadmap reassessment.
  Execute it alone; do not parallelize overlapping Prepare, CPU, Engine, Metal, build, or docs.
- After completion, separately reconcile autograd documentation, then reassess and repair the
  remaining contract drift. Do not create either later detailed brief in this change.
- No ADR is required: this restores the already-authoritative boundary. A discovered need to
  change authority is a blocker, not implementation scope.

## Documentation and review impact

- Primary profile: planning for this brief; implementation uses API/Javadoc plus architecture,
  backend-guide, and general profiles for the affected public SPI and explanations.
- A separate clean documentation/review context is mandatory because the task changes public SPI,
  dependency direction in implementation, and a Class C cross-module boundary. It may reuse
  successful executable evidence unless it changes executable behavior.
- Target-search glossary entries for compile artifacts, Prepare context, graph preparation,
  producerless published constants, CPU tuning, and prepared schedule context. Update only terms
  whose current descriptions expose the drift; record reasoned no-change conclusions elsewhere.

## Result

Completed the boundary cutover without changing authority, Engine user APIs, execution behavior,
numerics, route policy, or resource ownership. `GraphPreparation` now exposes the sole stable
partition projection and remains the complete assignment/finalization/schedule/rollback owner.
`PreparedScheduleContext` carries validated Model/Planning/Prepare/Runtime facts rather than
`CompileArtifacts`; its assembler collaboration supplies CPU geometry for producerless published
constants while Prepare retains role detection, ordering, validation, and assignment.

CPU production now consumes only stable `PrepareContext` projections, shared preparation roles,
and CPU-owned configuration. Ordinary, local-tuning trial/selection, complete-plan trial/selection,
and fallback paths return a `PartitionPreparation` to Engine, which composes every complete recipe
through `GraphPreparation`. OpenBLAS storage facts are derived from the projected values and
logical requirements. Metal received only the stable-context mechanical adaptation. The CPU
Compiler dependency is now test-only and architecture tests reject production Compiler imports or
types.

The implementation and focused tests changed 36 paths across Prepare, CPU, Engine, Metal, the CPU
build, architecture/integration tests, explanatory documentation, glossary, and planning status.
No top-level production type, architecture/ADR change, or unrelated capability was added.

Validation completed successfully:

- `./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :backends:metal:test`
  (`BUILD SUCCESSFUL`, 2m39s);
- `./gradlew :testing:architecture-tests:test :testing:integration-tests:test`
  (`BUILD SUCCESSFUL` after correcting the integration test's reflection-only dependency use);
- `./gradlew :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc :backends:metal:javadoc`
  (`BUILD SUCCESSFUL`; 94 existing unrelated CPU constructor warnings remain);
- `./gradlew test` (`BUILD SUCCESSFUL`, 72 actionable tasks);
- production CPU Compiler import/type search returned no matches, the production Compiler
  dependency search returned no matches, and the test dependency search returned exactly one
  `testImplementation` match;
- changed-path, public-reflection, source-shape, glossary-term, Markdown link/anchor/fence,
  final-newline, whitespace, and `git diff --check` audits passed.

The mandatory independent targeted review inspected the final 36-path diff and the generated test
and Javadoc evidence. It corrected two Javadoc boundary descriptions without changing executable
behavior and found no remaining code, architecture, API, lifecycle, resource, or documentation
defect within task scope.

Documentation and Javadoc now explain artifact ownership, the stable schedule context, projected
tuning association, producerless contribution, and test-only dependency. The glossary adds the
prepared-schedule-context distinction and reconciles graph preparation and complete-plan terms.
Ordinary and advanced Engine APIs remain source-compatible. The artifact-taking CPU integration
and tuning descriptors plus the `PreparedScheduleContext` record descriptor are intentionally
source/binary incompatible; session-scoped CPU tuning bytes retain no before/after continuity
promise. No unresolved implementation issue remains. The separately authorized autograd
documentation repair remains follow-up outside this task and is not detailed here.

Status: Complete

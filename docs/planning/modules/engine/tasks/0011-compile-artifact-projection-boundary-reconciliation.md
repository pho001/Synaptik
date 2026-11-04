# Task 0011: [Compile](../../../../glossary.md#compile)-[Artifact](../../../../glossary.md#term-artifact) Projection Boundary Reconciliation

> **Editorial historical reading key.**
>
> This is a historical boundary cutover, not a new computation capability. Engine passes Compiler's
> complete artifacts into shared GraphPreparation; Prepare alone projects partition-local facts and
> validates complete orchestration. CPU receives PrepareContext plus CPU configuration, returns
> backend preparation contributions, and owns concrete lowering, OpenBLAS eligibility, geometry, and
> tuning alternatives—not Compiler orchestration. CPU and Metal schedule assemblers receive stable
> validated schedule facts rather than the Compiler aggregate. Producerless constants cross as logical
> value, requirement, and scalar facts; CPU contributes physical geometry and initialization recipes,
> while Prepare assigns slots and coordinates rollback before transferring resources to Runtime. See
> [current Prepare
> orchestration](../../../../reference/runtime/orchestration.md#current-prepare-orchestration-assignment-and-backend-finalization)
> and [compile
> artifacts](../../../../reference/compiler/artifacts.md#current-complete-artifact-compilation-and-integration-port).
> Preserve the intentional CPU SPI/record incompatibilities: ordinary Engine APIs stayed compatible,
> and test-only Compiler dependencies do not authorize production coupling.

## Status

Complete

Frontier verification at launch: the user explicitly authorized this drift repair ahead of the
normal roadmap reassessment, [Engine](../../../../glossary.md#term-engine) 0010 and the required [Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) foundations were Complete,
and no other task was [`Ready`](../../../../glossary.md#term-ready) or `In progress`. This executed as the sole active frontier.

## Change class

Class C — this repairs the [Compiler](../../../../glossary.md#term-compiler)-to-[Prepare](../../../../glossary.md#prepare)-to-concrete-[backend](../../../../glossary.md#backend) [module](../../../../glossary.md#term-module) edge, changes a
supported [CPU](../../../../glossary.md#term-cpu) service-provider interface ([SPI](../../../../glossary.md#term-service-provider-interface-spi)), and preserves the complete prepare/tuning
[lifecycle](../../../../glossary.md#lifecycle) across Prepare, CPU, [Engine](../../../../glossary.md#term-engine), and one [Metal](../../../../glossary.md#term-metal) schedule-context consumer.

The brief exceeds the 15 KB [planning](../../../../glossary.md#term-planning) [target](../../../../glossary.md#term-target) because one [atomic](../../../../glossary.md#resource-transaction-and-rollback) Class C cutover must preserve and
validate the ordinary path, two tuning phases, fallback, producerless resources, and two backend
assemblers across four modules; splitting those coupled acceptance criteria would hide boundary
coverage.

## Goal

Make [Engine](../../../../glossary.md#term-engine) the only concrete-composition [owner](../../../../glossary.md#term-owner) that passes `CompileArtifacts` into shared
[Prepare](../../../../glossary.md#prepare) orchestration. Reuse [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation)'s existing projection and
`CpuPartitionAnalysisInputs`; let [CPU](../../../../glossary.md#term-cpu) receive only stable Prepare/[Model](../../../../glossary.md#term-model)/[Planning](../../../../glossary.md#term-planning) projections and
CPU-owned configuration, while preserving current ordinary preparation, producerless [constants](../../../../glossary.md#term-constant),
schedule assembly, [OpenBLAS](../../../../glossary.md#term-openblas) eligibility, tuning candidates, execution, numerics, and cleanup.

Remove the CPU production [dependency](../../../../glossary.md#term-dependency) on `modules/compiler`.

## [Scope](../../../../glossary.md#term-scope)

- Lift [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation)'s existing projection behind an artifact-owning Prepare operation so
  Engine can obtain the exact <span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine))</span>[`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) for CPU input/candidate production. Ordinary and
  tuning paths may invoke it separately but share one implementation and add no second projection.
- Keep complete orchestration in `GraphPreparation`: artifact validation, all contexts before
  backend work, analysis, assignment, finalization, producerless-resource coverage, schedule
  assembly/validation, rollback, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [rollback glossary](../../../../glossary.md#term-rollback))</span>[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) construction.
- Replace [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext)'s `CompileArtifacts` component with the minimal immutable
  stable facts schedule assembly uses: planned/prepared partition association, graph values,
  source roles/constants, ordered publication value IDs, memory plan, and buffer assignments.
  Validate those projections inside Prepare before any concrete assembler sees them.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [prepared partition glossary](../../../../glossary.md#prepared-partition--preparedpartition); [graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Let the existing `PreparedScheduleAssembler` collaboration contribute CPU physical geometry
  for each required producerless published constant from stable value, logical-memory, and scalar
  arguments before assignment. Shared Prepare identifies, orders, validates, assigns, and
  transacts the complete set; CPU owns physical geometry and the initialized recipe.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [arguments glossary](../../../../glossary.md#term-argument))</span>
- Change `CpuBackendIntegration` from artifact orchestration to artifact-free formation/exposure
  of the CPU analysis inputs and one `PartitionPreparation`, plus its schedule assembler,
  capabilities, storage/materialization operations, and tuning collaborators. Engine calls<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
  `GraphPreparation` for ordinary and tuned recipes.
- Change local- and complete-plan CPU tuning to consume the exact projected partition context,
  retain only projection/CPU association state, and return a selected/trial backend preparation
  for Engine to compose. They must not accept, retain, hash, or prepare <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [hash glossary](../../../../glossary.md#term-hash))</span>`CompileArtifacts`, and
  must not construct a complete `PreparedExecution` themselves.
- Derive existing [OpenBLAS](../../../../glossary.md#term-openblas) boundary-storage facts from the exact `PrepareContext` rather than
  from `CompileArtifacts`; do not introduce a replacement [CPU](../../../../glossary.md#term-cpu) analysis-input aggregate.
- Preserve CPU complete-plan compatibility identity from the partition projection and existing
  CPU candidate/configuration identities. Engine retains exact compiled-graph association for
  the complete representative workflow.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Adapt the Metal schedule assembler mechanically to the stable context. Add no Metal capability,
  route, composition, or production Compiler dependency.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Move `backends/cpu -> modules/compiler` from `implementation` to `testImplementation`; keep
  compiler-backed CPU fixtures as tests only, and lock the production edge with architecture
  tests.
- After executable work stabilizes, complete the mandatory independent documentation/review pass.

## Non-goals

- changing compile output, graph transformation, partitioning, memory assignment, route choice,
  tuning ranking/cache policy, numerical results, Runtime execution, or resource ownership<span data-glossary-navigation="true"> ([ranking glossary](../../../../glossary.md#term-ranking); [resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph); [cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- adding mixed-backend scheduling, zero-node execution, a second graph/partition projection,
  generic backend discovery, a registry, or a service locator<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- changing ordinary or advanced Engine user APIs, Tensor/Training APIs, or public computation
  workflows<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Tensor glossary](../../../../glossary.md#tensor))</span>
- changing `ARCHITECTURE.md`, a scoped normative contract, or an ADR; if implementation needs
  any such change, stop and return to planning<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning))</span>
- repairing [autograd](../../../../glossary.md#autograd) documentation or any later contract drift in this task

## Contracts

- [`ARCHITECTURE.md` heading `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) —
  compile artifacts flow through Prepare before backend analysis.<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  Engine composes; Prepare stages analysis/finalization; Runtime executes prepared schedules.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [`ARCHITECTURE.md` heading `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Engine owns explicit
  composition, Prepare owns shared transition/validation, and CPU owns concrete lowering/routes.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lowering glossary](../../../../glossary.md#lowering))</span>
- [`runtime-prepare-engine.md` heading `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)
  and [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  — Prepare projects exact stable facts and must not expose `CompileArtifacts` to [CPU](../../../../glossary.md#term-cpu).
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine owns compile/prepare orchestration and explicit backend wiring.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [`CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU receives shared contracts/configuration and owns lowering, routes, physical recipes, and
  finalization, but not Engine or Compiler orchestration.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

- `modules/prepare/.../GraphPreparation.java`, `PreparedScheduleContext.java`,
  `PreparedScheduleAssembler.java`, `PartitionPreparation.java`, and
  `ProducerlessPublishedConstantResource.java` — one projection, stable schedule facts, and
  complete orchestration.
- `backends/cpu/.../CpuBackendIntegration.java`, `CpuLocalWorkloadTuning.java`,
  `CpuCompletePlanTuning.java`, `CpuBackendComposition.java`, `CpuPartitionAnalysisInputs.java`,
  `CpuPartitionPreparer.java`, `CpuOpenBlasRouteSelector.java`, and
  `CpuPreparedScheduleAssembler.java` — projection-only [lifecycle](../../../../glossary.md#lifecycle) and tuning.
- `modules/engine/.../CpuEngineBackendComposition.java` and `AdvancedEngine.java` — ordinary and
  tuning orchestration plus exact compiled-graph association.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- `backends/metal/.../MetalNegPreparedScheduleAssembler.java` — stable-context adaptation only.
- `backends/cpu/build.gradle.kts` and `testing/architecture-tests/.../CpuDependencyContractTest.java`
  — remove and enforce the production Compiler edge.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Focused [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation)/[`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext), CPU integration/composition/local- and
  complete-plan tuning, Engine lifecycle/tuning, <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>`MetalNegPreparedExecution`, CPU dependency, and
  Engine model-autotuning integration tests corresponding to the changed symbols.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [model glossary](../../../../glossary.md#term-model); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- `docs/architecture/dependency-rules.md`, `runtime-prepare-backend-boundary.md`,
  `docs/api/public-api.md`, `runtime-api.md`, `docs/backend-guide/cpu-backend.md`,
  `partition-preparer.md`, and `docs/glossary.md` — reconcile current explanations.

No new top-level production type is expected. If a named production type must be added or moved,
stop and return the task to [planning](../../../../glossary.md#term-planning) rather than creating a duplicate projection or broad [facade](../../../../glossary.md#term-facade).

## Acceptance criteria

- No CPU production source imports a Compiler package, no supported CPU signature names a
  Compiler type, and the CPU build has no production dependency on <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [signature glossary](../../../../glossary.md#term-signature); [dependency glossary](../../../../glossary.md#term-dependency))</span>`modules/compiler`.
- Only [Engine](../../../../glossary.md#term-engine)/[Prepare](../../../../glossary.md#prepare) code handles `CompileArtifacts` during current CPU composition; CPU analysis,
  schedule assembly, producerless contribution, and both tuning phases receive stable projected
  facts and CPU-owned configuration only.
- The existing [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) projection is the single source of partition-local nodes,
  values, logical requirements, constants, and topology for ordinary and tuning paths.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [nodes glossary](../../../../glossary.md#node); [constants glossary](../../../../glossary.md#term-constant); [topology glossary](../../../../glossary.md#term-topology))</span>
- [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext) exposes no Compiler aggregate. CPU and Metal assemblers consume only
  validated stable facts required for their current recipes.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- Ordinary, selected Phase-1, trial/final Phase-2, and fallback preparation all pass through
  shared `GraphPreparation`; no [CPU](../../../../glossary.md#term-cpu) path independently assembles a complete execution.
- Source-only published constants keep exact role detection, graph-value order, byte geometry,
  initialized-per-run representation, publication order, and zero schedule-occurrence behavior.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run); [publication glossary](../../../../glossary.md#term-publication); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Current sole non-empty CPU partition rejection, OpenBLAS eligibility/decision authentication,
  complete-plan candidate identity, stale/foreign association rejection, schedule steps,
  outputs, numerics, concurrency, rollback, and close behavior remain unchanged.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [authentication glossary](../../../../glossary.md#term-authentication); [complete-plan candidate glossary](../../../../glossary.md#complete-plan-candidate); [identity glossary](../../../../glossary.md#term-identity-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Removing the [artifact](../../../../glossary.md#term-artifact)-taking `CpuBackendIntegration` preparation methods and both CPU tuning
  handoffs/preparation methods is an intentional source and binary SPI break; repository callers
  migrate atomically. No deprecated bridge preserves Compiler exposure. Ordinary/advanced Engine
  user APIs remain source-compatible.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- `PreparedScheduleContext` record construction/component reflection and the artifact-taking CPU
  lifecycle/tuning descriptors are intentionally incompatible. Current CPU tuning compatibility
  remains versioned with the same reuse-scope and within-batch identity behavior; its
  session-scoped canonical bytes have no before/after byte-continuity promise.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [descriptors glossary](../../../../glossary.md#term-descriptor); [scope glossary](../../../../glossary.md#term-scope); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Architecture tests distinguish CPU's allowed compiler-backed test fixtures from the forbidden
  production edge. Prepare stays concrete-backend-free; Metal stays production-Compiler-free.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Javadocs fully document every changed input, result, failure, ownership, association, and
  compatibility rule; explanatory docs remove the obsolete direct CPU/Compiler rationale.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## [Validation](../../../../glossary.md#term-validation)

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
changed-path list and public reflection tests for accidental [API](../../../../glossary.md#term-application-programming-interface-api) or [scope](../../../../glossary.md#term-scope) expansion.

Repository-wide validation is required because this changes a [dependency edge](../../../../glossary.md#term-dependency-edge), shared [Prepare](../../../../glossary.md#prepare)
[SPI](../../../../glossary.md#term-service-provider-interface-spi), concrete-[backend](../../../../glossary.md#backend) SPI, build configuration, and multiple [modules](../../../../glossary.md#term-module).

## Dependencies and follow-up

- Engine 0010, Prepare 0003A/0005/0006, and CPU 0010F/0010H/0010I/0010J are Complete and supply
  the current projection, producerless, ownership, lifecycle, and tuning behavior being retained.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- This user-authorized drift-remediation exception precedes the normal roadmap reassessment.
  Execute it alone; do not parallelize overlapping Prepare, CPU, Engine, Metal, build, or docs.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- After completion, separately reconcile autograd documentation, then reassess and repair the
  remaining contract drift. Do not create either later detailed brief in this change.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd))</span>
- No ADR is required: this restores the already-authoritative boundary. A discovered need to
  change authority is a blocker, not implementation scope.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [scope glossary](../../../../glossary.md#term-scope))</span>

## Documentation and review impact

- Primary profile: planning for this brief; implementation uses API/Javadoc plus architecture,
  backend-guide, and general profiles for the affected public SPI and explanations.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- A separate clean documentation/review context is mandatory because the task changes public SPI,
  dependency direction in implementation, and a Class C cross-module boundary. It may reuse
  successful executable evidence unless it changes executable behavior.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>
- Target-search glossary entries for compile artifacts, Prepare context, graph preparation,
  producerless published constants, CPU tuning, and prepared schedule context. Update only terms
  whose current descriptions expose the drift; record reasoned no-change conclusions elsewhere.<span data-glossary-navigation="true"> ([Target glossary](../../../../glossary.md#term-target); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [Prepare glossary](../../../../glossary.md#prepare); [graph preparation glossary](../../../../glossary.md#graph-preparation--graphpreparation); [constants glossary](../../../../glossary.md#term-constant); [CPU glossary](../../../../glossary.md#term-cpu); [prepared schedule context glossary](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext))</span>

## Result

Completed the boundary cutover without changing authority, [Engine](../../../../glossary.md#term-engine) user APIs, execution behavior,
numerics, route policy, or resource [ownership](../../../../glossary.md#term-ownership). [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) now exposes the sole stable
[partition](../../../../glossary.md#partition) projection and remains the complete assignment/finalization/schedule/[rollback](../../../../glossary.md#term-rollback) [owner](../../../../glossary.md#term-owner).
[`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext) carries validated [Model](../../../../glossary.md#term-model)/[Planning](../../../../glossary.md#term-planning)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) facts rather than
`CompileArtifacts`; its assembler collaboration supplies [CPU](../../../../glossary.md#term-cpu) geometry for producerless published
[constants](../../../../glossary.md#term-constant) while Prepare retains role detection, ordering, [validation](../../../../glossary.md#term-validation), and assignment.

CPU production now consumes only stable [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) projections, shared preparation roles,
and CPU-owned configuration. Ordinary, local-tuning trial/selection, complete-plan trial/selection,
and fallback paths return a `PartitionPreparation` to Engine, which composes every complete recipe
through `GraphPreparation`. [OpenBLAS](../../../../glossary.md#term-openblas) storage facts are derived from the projected values and
logical requirements. [Metal](../../../../glossary.md#term-metal) received only the stable-context mechanical adaptation. The CPU
[Compiler](../../../../glossary.md#term-compiler) [dependency](../../../../glossary.md#term-dependency) is now test-only and architecture tests reject production Compiler imports or
types.

The implementation and focused tests changed 36 paths across Prepare, CPU, Engine, Metal, the CPU
build, architecture/integration tests, explanatory documentation, glossary, and planning status.
No top-level production type, architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) change, or unrelated capability was added.

Validation completed successfully:

- `./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :backends:metal:test`
  (`BUILD SUCCESSFUL`, 2m39s);
- `./gradlew :testing:architecture-tests:test :testing:integration-tests:test`
  (`BUILD SUCCESSFUL` after correcting the [integration test](../../../../glossary.md#term-integration-test)'s reflection-only dependency use);
- `./gradlew :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc :backends:metal:javadoc`
  (`BUILD SUCCESSFUL`; 94 existing unrelated CPU constructor warnings remain);
- `./gradlew test` (`BUILD SUCCESSFUL`, 72 actionable tasks);
- production CPU Compiler import/type search returned no matches, the production Compiler
  dependency search returned no matches, and the test dependency search returned exactly one
  `testImplementation` match;
- changed-path, public-reflection, source-shape, glossary-term, Markdown link/anchor/fence,
  final-newline, whitespace, and <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>`git diff --check` audits passed.

The mandatory independent targeted review inspected the final 36-path diff and the generated test
and [Javadoc](../../../../glossary.md#term-javadoc) evidence. It corrected two Javadoc boundary descriptions without changing executable
behavior and found no remaining code, architecture, [API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle), resource, or documentation
defect within task [scope](../../../../glossary.md#term-scope).

Documentation and Javadoc now explain [artifact](../../../../glossary.md#term-artifact) [ownership](../../../../glossary.md#term-ownership-meanings), the stable schedule context, projected
tuning association, producerless contribution, and test-only dependency. The glossary adds the
prepared-schedule-context distinction and reconciles [graph preparation](../../../../glossary.md#graph-preparation--graphpreparation) and complete-plan terms.
Ordinary and advanced Engine APIs remain source-compatible. The artifact-taking CPU integration
and tuning [descriptors](../../../../glossary.md#term-descriptor) plus the `PreparedScheduleContext` record descriptor are intentionally
source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata) incompatible; session-scoped CPU tuning bytes retain no before/after continuity
promise. No unresolved implementation issue remains. The separately authorized [autograd](../../../../glossary.md#autograd)
documentation repair remains follow-up outside this task and is not detailed here.

Status: Complete

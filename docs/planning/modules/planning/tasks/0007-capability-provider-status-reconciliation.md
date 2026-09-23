# Task 0007: Capability Provider Status Reconciliation

## Status

Ready

Frontier verification: Planning 0001–0006 and tuning 0004 are Complete, no task is `Ready` or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class B — this documentation-only task corrects public capability-provider and compile-workflow
status across a backend guide and API reference. It changes no executable behavior or authority,
but requires a clean implementation context and independent targeted documentation review.

## Goal

Describe the current CPU capability provider and its explicit consumption by the public Engine
compile workflow without inventing generic registration, a standalone compiler facade, or broader
backend support.

## Scope

- Correct `docs/backend-guide/capability-provider.md` to acknowledge the shipped public
  `CpuCapabilityProvider` and current public `Engine.compile(...)` workflow that supplies it
  explicitly through Engine composition.
- Preserve the internal boundary: `GraphCompilationPort` is a public cross-module SPI,
  `GraphCompiler` remains package-private, and no standalone ordinary compiler facade or public
  generic provider-registration workflow exists.
- Update the lifecycle/status text and illustrative-provider framing only where the old
  no-implementation/no-consumer milestone made them stale.
- Correct the matching stale production-provider statement in `docs/api/compile-api.md`.
- Synchronize this task, the Planning master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependencies, API, behavior, capability matrix, numerical
  support, scoring, preparation, runtime, execution, registration, discovery, or architecture
  authority change.
- No exhaustive rewrite of CPU operation coverage; link to the current CPU provider/backend guide
  rather than duplicating its large capability matrix.
- No generic backend registration, public graph-wide Planning facade, standalone compiler facade,
  cost-bearing scoring, Metal/CUDA provider, or mixed-backend Engine claim.
- No partition-preparer, CPU-backend, or unrelated status cleanup.

## Contracts

- [`ARCHITECTURE.md` headings `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Planning owns
  capability/ownership decisions, Compiler owns graph-wide orchestration, Engine owns explicit
  public composition, and concrete backends remain outward dependencies.
- [`foundational-modules.md` heading `Module responsibilities`](../../../../architecture/contracts/foundational-modules.md#module-responsibilities)
  — preserve Planning's provider query/selection responsibility.
- [`compiler-autograd.md` heading `modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)
  — preserve Compiler-owned graph-wide orchestration and its current public cross-module port.
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — preserve concrete backend ownership of capability providers and fail-closed support.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve current explicit CPU Engine composition and public lifecycle.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these five paths:

- `docs/backend-guide/capability-provider.md` — provider/consumer status, lifecycle, example
  framing, and validation status only.
- `docs/api/compile-api.md` — matching production-provider/consumer paragraph only.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/planning/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — Planning row and repository frontier only.

Read-only evidence:

- `CpuCapabilityProvider`, `CpuBackendIntegration.capabilityProvider()`,
  `CpuEngineBackendComposition.capabilityProviders()`, `Engine.compile(...)`,
  `AdvancedEngine.compile(...)`, and `GraphCompilationPort`.
- Focused CPU provider, Engine standard-composition/compile, and Compiler orchestration tests.
- Current public API CPU/Engine status and targeted glossary entries for capability provider,
  compile facade, Engine, and backend ownership.

## Acceptance criteria

- Both documents state that a production `CpuCapabilityProvider` exists and is explicitly composed
  into current public CPU Engine compilation.
- They distinguish ordinary public `Engine.compile(...)`, public cross-module
  `GraphCompilationPort`, and package-private `GraphCompiler`; no standalone compiler facade or
  generic provider registration is implied.
- The guide's illustrative provider remains clearly illustrative while no longer claiming that
  the repository has no real provider.
- The lifecycle and validation sections identify current compile consumption and concrete CPU
  provider tests without claiming generic public Planning orchestration or conformance beyond
  existing evidence.
- Only the five allowed paths change; no Java/Gradle files change; links, anchors, headings,
  fences, newlines, whitespace, exact path inventory, and `git diff --check` pass.

## Validation

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderPublicShapeTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.EngineStandardCompositionTest \
  --tests io.github.pho001.synaptik.engine.EngineTypedLifecycleTest
rg -n 'does not yet ship a provider implementation|supplies no production provider implementation|no .*public compile consumer|once concrete implementations exist' \
  docs/backend-guide/capability-provider.md docs/api/compile-api.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
git diff --check
```

The stale-text and executable/build-file inventories must be empty. Validate changed Markdown and
the exact five-path inventory with the repository's existing validator or an equivalent temporary
script outside the repository.

Repository-wide validation is deferred to CI because this task reconciles documentation with
already-tested current behavior and changes no code, dependency, or authority. Independent review
reuses successful focused-test evidence.

## Dependencies and follow-up

- Planning 0001–0006, the CPU provider/integration, Compiler public SPI, Engine public compile
  composition, and tuning 0004 are Complete.
- After completion, separately plan the partition-preparer guide drift; this task does not combine
  that Prepare/backend lifecycle work.

## Documentation and review impact

- The implementation applies the General, Backend Guide, API/Javadoc, and Planning profiles and
  performs a targeted glossary review.
- Independent documentation review is required because the task corrects public workflow and a
  durable cross-module capability/compile boundary.

## Result

Empty until execution.

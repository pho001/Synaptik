# Task 0007: Capability Provider Status Reconciliation

## Status

Complete

Frontier verification: [Planning](../../../../glossary.md#term-planning) 0001–0006 and tuning 0004 are Complete, no task is [`Ready`](../../../../glossary.md#term-ready) or
`In progress`, and the user authorized the next confirmed drift repair. This task is the sole
active frontier and its dependencies are satisfied.

## Change class

Class B — this documentation-only task corrects public capability-provider and [compile](../../../../glossary.md#compile)-workflow
status across a [backend](../../../../glossary.md#backend) guide and [API](../../../../glossary.md#term-application-programming-interface-api) reference. It changes no executable behavior or authority,
but requires a clean implementation context and independent targeted documentation review.

## Goal

Describe the current [CPU](../../../../glossary.md#term-cpu) capability provider and its explicit consumption by the public [Engine](../../../../glossary.md#term-engine)
[compile](../../../../glossary.md#compile) workflow without inventing generic registration, a standalone [compiler](../../../../glossary.md#term-compiler) [facade](../../../../glossary.md#term-facade), or broader
[backend](../../../../glossary.md#backend) support.

## [Scope](../../../../glossary.md#term-scope)

- Correct `docs/backend-guide/capability-provider.md` to acknowledge the shipped public
  `CpuCapabilityProvider` and current public `Engine.compile(...)` workflow that supplies it
  explicitly through Engine composition.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Preserve the internal boundary: `GraphCompilationPort` is a public cross-[module](../../../../glossary.md#term-module) [SPI](../../../../glossary.md#term-service-provider-interface-spi),
  `GraphCompiler` remains package-private, and no standalone ordinary compiler facade or public
  generic provider-registration workflow exists.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [facade glossary](../../../../glossary.md#term-facade))</span>
- Update the lifecycle/status text and illustrative-provider framing only where the old
  no-implementation/no-consumer milestone made them stale.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [framing glossary](../../../../glossary.md#term-framing))</span>
- Correct the matching stale production-provider statement in `docs/api/compile-api.md`.
- Synchronize this task, the [Planning](../../../../glossary.md#term-planning) master plan, and the roadmap after implementation and review.

## Non-goals

- No Java, Javadoc, tests, Gradle, dependencies, API, behavior, capability matrix, numerical
  support, scoring, preparation, runtime, execution, registration, discovery, or architecture
  authority change.<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [matrix glossary](../../../../glossary.md#term-review-matrix); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- No exhaustive rewrite of CPU operation coverage; link to the current CPU provider/backend guide
  rather than duplicating its large capability matrix.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- No generic backend registration, public graph-wide Planning facade, standalone compiler facade,
  cost-bearing scoring, Metal/CUDA provider, or mixed-backend Engine claim.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Planning glossary](../../../../glossary.md#term-planning); [facade glossary](../../../../glossary.md#term-facade); [compiler glossary](../../../../glossary.md#term-compiler); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [Engine glossary](../../../../glossary.md#term-engine))</span>
- No [partition](../../../../glossary.md#partition)-preparer, [CPU](../../../../glossary.md#term-cpu)-[backend](../../../../glossary.md#backend), or unrelated status cleanup.

## Contracts

- [`ARCHITECTURE.md` headings `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and [`Dependency rules`](../../../../../ARCHITECTURE.md#dependency-rules) — Planning owns
  capability/ownership decisions, Compiler owns graph-wide orchestration, Engine owns explicit
  public composition, and concrete backends remain outward dependencies.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Engine glossary](../../../../glossary.md#term-engine); [backends glossary](../../../../glossary.md#backend))</span>
- [`foundational-modules.md` heading `Module responsibilities`](../../../../architecture/contracts/foundational-modules.md#module-responsibilities)
  — preserve Planning's provider query/selection responsibility.
- [`compiler-autograd.md` heading `modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)
  — preserve Compiler-owned graph-wide orchestration and its current public cross-module port.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — preserve concrete backend ownership of capability providers and fail-closed support.<span data-glossary-navigation="true"> ([concrete backend glossary](../../../../glossary.md#concrete-backend); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve current explicit CPU Engine composition and public lifecycle.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Execution may change exactly these five paths:

- `docs/backend-guide/capability-provider.md` — provider/consumer status, lifecycle, example
  framing, and validation status only.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [framing glossary](../../../../glossary.md#term-framing); [validation glossary](../../../../glossary.md#term-validation))</span>
- `docs/api/compile-api.md` — matching production-provider/consumer paragraph only.
- this task brief — status and compact result evidence only.
- `docs/planning/modules/planning/master-plan.md` — task row and frontier only.
- `docs/planning/roadmap.md` — [Planning](../../../../glossary.md#term-planning) row and repository frontier only.

Read-only evidence:

- `CpuCapabilityProvider`, `CpuBackendIntegration.capabilityProvider()`,
  `CpuEngineBackendComposition.capabilityProviders()`, `Engine.compile(...)`,
  `AdvancedEngine.compile(...)`, and `GraphCompilationPort`.
- Focused [CPU](../../../../glossary.md#term-cpu) provider, [Engine](../../../../glossary.md#term-engine) standard-composition/[compile](../../../../glossary.md#compile), and [Compiler](../../../../glossary.md#term-compiler) orchestration tests.
- Current public API CPU/Engine status and targeted glossary entries for capability provider,
  compile facade, Engine, and backend ownership.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [backend ownership glossary](../../../../glossary.md#backend-ownership))</span>

## Acceptance criteria

- Both documents state that a production `CpuCapabilityProvider` exists and is explicitly composed
  into current public CPU Engine compilation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- They distinguish ordinary public `Engine.compile(...)`, public cross-[module](../../../../glossary.md#term-module)
  `GraphCompilationPort`, and package-private `GraphCompiler`; no standalone compiler facade or
  generic provider registration is implied.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [facade glossary](../../../../glossary.md#term-facade))</span>
- The guide's illustrative provider remains clearly illustrative while no longer claiming that
  the repository has no real provider.
- The lifecycle and validation sections identify current compile consumption and concrete CPU
  provider tests without claiming generic public Planning orchestration or conformance beyond
  existing evidence.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [validation glossary](../../../../glossary.md#term-validation); [compile glossary](../../../../glossary.md#compile); [Planning glossary](../../../../glossary.md#term-planning); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Only the five allowed paths change; no Java/Gradle files change; links, anchors, headings,
  fences, newlines, whitespace, exact path inventory, and <span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>`git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

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

Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task reconciles documentation with
already-tested current behavior and changes no code, [dependency](../../../../glossary.md#term-dependency), or authority. Independent review
reuses successful focused-test evidence.

## Dependencies and follow-up

- Planning 0001–0006, the CPU provider/integration, Compiler public SPI, Engine public compile
  composition, and tuning 0004 are Complete.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [Compiler glossary](../../../../glossary.md#term-compiler); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [Engine glossary](../../../../glossary.md#term-engine))</span>
- After completion, separately plan the partition-preparer guide drift; this task does not combine
  that Prepare/backend lifecycle work.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Targeted review found two additional compile-status defects outside this task's five-path
  allowlist: the glossary's [`Compile`](../../../../glossary.md#compile) entry still says the ordinary Engine lifecycle is planned,
  and `GraphCompilationPort` Javadoc still describes the Engine facade as future. Reconcile them
  together in a separate focused task rather than widening this review.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [facade glossary](../../../../glossary.md#term-facade); [widening glossary](../../../../glossary.md#term-widening))</span>

## Documentation and review impact

- The implementation applies the General, Backend Guide, API/Javadoc, and Planning profiles and
  performs a targeted glossary review.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Independent documentation review is required because the task corrects public workflow and a
  durable cross-module capability/compile boundary.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [compile glossary](../../../../glossary.md#compile))</span>

## Result

Implementation and independent targeted documentation review corrected the shipped [CPU](../../../../glossary.md#term-cpu) provider
and current [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile)-consumer status in the [backend](../../../../glossary.md#backend) guide and Compile [API](../../../../glossary.md#term-application-programming-interface-api) while preserving
the public-[SPI](../../../../glossary.md#term-service-provider-interface-spi)/package-private-[compiler](../../../../glossary.md#term-compiler) and no-generic-registration boundaries. Review also fixed
the [lifecycle](../../../../glossary.md#lifecycle) diagram so `GraphCompilationPort` is an entry to [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) rather than a stage
after [artifact](../../../../glossary.md#term-artifact) construction. The focused CPU provider command passed 28 tests and the focused
Engine command passed 36 tests, both with zero failures, errors, or skips; review reused that
evidence without rerunning Java tests. Stale-text, Java/[Gradle](../../../../glossary.md#term-gradle)-diff, exact five-path,
Markdown-link/anchor/fence/newline/whitespace, and `git diff --check` [validation](../../../../glossary.md#term-validation) passed. Targeted
glossary review found the separately scoped stale `Compile` entry and `GraphCompilationPort`
[Javadoc](../../../../glossary.md#term-javadoc) described above; neither changes this task's documentation-only verdict.

Status: Complete

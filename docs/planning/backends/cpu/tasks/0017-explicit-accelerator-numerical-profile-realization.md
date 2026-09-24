# Task 0017: Explicit ACCELERATOR Numerical-Profile Realization

## Status

Ready

Readiness verification: Model 0027 and Config 0006 are Complete. Engine 0018 is Complete at
implementation `ce7a7dfa` plus documentation/Javadoc remediation `07a01b9c`; its required
validation and independent Class C review passed. The integrated spine already separates CPU
plans, generated artifacts, OpenBLAS workloads, local tuning, and complete-plan tuning by profile.
CPU 0017 and Metal 0015 have disjoint backend behavior, source, test, and task-document ownership;
shared status and explanatory documents are reserved to the integration owner.

## Change class

Class C — this changes the supported CPU capability and executable Engine behavior for an entire
graph profile across portable generated code, OpenBLAS, tuning identities, forward execution, and
Compiler-generated backward graphs. It must prove that no arithmetic, route policy, hot path, or
cache boundary is relaxed accidentally.

## Goal

Admit `NumericalProfile.ACCELERATOR` everywhere the current CPU backend admits `STRICT_IEEE`, while
executing the exact same current CPU semantics and routes. A strict result is valid under the
Model-owned accelerator result set, so this first CPU realization adds permission and identity,
not relaxed math.

## Exact semantics and scope

- `CpuCapabilityProvider` must answer an `ACCELERATOR` query exactly as it answers the otherwise
  identical `STRICT_IEEE` query. Supported and rejected operation/type/Shape/layout/attribute/
  gradient rows remain identical; hardware, route availability, tuning, and workload size do not
  grant capability.
- `CpuPartitionPreparer` must accept both profiles and retain the selected profile in every existing
  `CpuPartitionPreparationPlan`, unit specialization, generated-artifact compatibility key,
  OpenBLAS workload, local-workload decision, and complete-plan decision.
- Portable scalar/vector/parallel/generated behavior, exact operation evaluation, OpenBLAS
  qualification and calls, fusion/decomposition/materialization, thresholds, workers, species,
  constants, and fallbacks remain byte-for-byte policy equivalents across profiles. No DAZ, FTZ,
  reassociation, approximate instruction, reduced precision, fast math, or value-dependent route is
  introduced.
- Existing profile-separated identity is authoritative: generator envelope `67`, OpenBLAS
  candidate/policy `2`, and local/complete value schemas `2` remain current unless implementation
  discovers an actual encoded-format change. Do not bump a version merely because capability is
  enabled. Old, malformed, and cross-profile identities continue to miss or reject safely.
- Runtime, sessions, run state/results, Trace, generated entry signatures, and hot loops remain
  profile-free. Profile selection stays cold and immutable.

## Non-goals

No new CPU operation, data type, route, provider, optimization, numerical relaxation, performance
claim, threshold/default change, public API, module edge, cache format, persistence enablement, or
Runtime/Trace field. Do not begin CPU 0011–0016, Model 0026, or any Metal work.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model owns
  result sets; the selected profile is cold and unchanged through plan/cache identity.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — `ACCELERATOR` contains every strict result and grants no generic fast math.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU owns truthful capability and route realization behind one backend identity.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  selection finishes before Runtime; the hot path executes fixed work.

If exact CPU behavior cannot be retained for an existing supported row, remove no strict capability
and do not narrow accelerator ad hoc; stop and report the row and counterexample.

## Dependencies and integration

- Depends on: Model 0027; Config 0006; Engine 0018; current CPU portable/OpenBLAS implementation
- Conflicts with: CPU capability, preparation, generated identity, OpenBLAS/tuning, Engine CPU
  integration, and CPU-specific guide/Javadoc scopes
- Parallel group: `numerical-profile-backends`
- Common base revision: `07a01b9c13c2ceea0922b9af82da3e6e08375306`
- Integration order: Any relative to Metal 0015; both executable commits precede shared-document
  reconciliation
- Integration validation: numerical-profile backend integration checkpoint
- Shared-document integration owner: Main planner

The CPU worker owns only CPU production/tests, `EngineNumericalProfileIntegrationTest`, this brief,
CPU package Javadocs, and `docs/backend-guide/cpu-backend.md`. It must not edit the roadmap, master
plans, root/scoped architecture, shared API/user/glossary/capability/preparer documents, or Metal
paths. Those shared files are reconciled once after both parallel branches land.

## Exact files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
  — remove the accelerator-wide rejection; preserve one identical occurrence predicate.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
  — remove the strict-only gate and continue projecting `context.numericalProfile()` unchanged.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java` and affected
  Javadocs in the two files above — describe exact dual-profile realization, not relaxed math.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
  — run the retained admitted/rejected capability matrix under both profiles.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
  and `internal/cache/CpuKernelSpecializationTest.java` — prove equal plans/executable bodies with
  distinct profile identities and successful accelerator preparation.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatchTest.java`
  plus the existing local/complete tuning public tests — retain deterministic same-profile encoding
  and cross-profile incompatibility at the current versions.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuNumericalProfileConformanceTest.java`
  — new bounded public capability/preparation/execution parity checkpoint.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineNumericalProfileIntegrationTest.java`
  — replace accelerator rejection with observable strict and accelerator forward/backward runs.
- `docs/backend-guide/cpu-backend.md` — current CPU profile behavior and unchanged routes/semantics.

## Tests and executable smoke

- Capability conformance must compare otherwise equal strict/accelerator queries over every current
  family represented in `CpuCapabilityProviderTest`, including rejected boundary rows; the two
  answers must always match.
- Prepare the same representative portable pointwise/reduction and eligible OpenBLAS MATMUL facts
  under both profiles. Plans must retain their exact profile; route, declarations, IR, strategy,
  workers, species, and operation semantics must match. Profile compatibility and structural keys
  must differ where required.
- Generate the same representative classes under both profiles. Strip only profile-bearing class
  identity/constant-pool names and prove the normalized executable method bodies have the same
  hash. Do not modify performance tests, thresholds, generated instructions, or expected results.
- The actual `EngineNumericalProfileIntegrationTest` must construct separate strict and accelerator
  CPU Engines. For each profile, execute and materialize a nontrivial forward tensor result, then
  execute an observable Compiler-generated backward path and materialize the objective and gradient
  values. Exercise a reusable session on the forward path. Expected represented values are exact
  and identical across profiles; a mere compile/not-throw assertion is insufficient.
- Exercise both portable and qualified-OpenBLAS conformance where the test fixture supplies the
  provider. No native probe is required: this task changes no native ABI/provider arithmetic.

## Acceptance criteria

1. CPU capability has exactly one current support matrix shared by both profiles; no rejected row
   becomes supported except by selecting the already supported row under `ACCELERATOR`.
2. Both profiles compile, prepare, execute, and materialize observable forward and backward CPU
   results with exact current semantics and no strict fallback or profile rewrite.
3. Plans and every generated/OpenBLAS/tuning/cache identity retain the requested profile. Same-
   profile encoding is deterministic; different profiles never share identity; current version
   numbers remain unless an actual wire change is demonstrated.
4. Normalized generated executable bodies match across profiles, and no arithmetic instruction,
   route/default, threshold, worker policy, or performance test changes.
5. Runtime/Trace and hot paths remain profile-free. No new public API, dependency, operation, type,
   provider, relaxed result, or performance claim.
6. Focused tests, Javadoc, conformance, actual Engine smoke, documentation validation, and
   independent Class C review pass.

## Validation

Worker validation:

```bash
./gradlew :backends:cpu:test :backends:cpu:javadoc
./gradlew :testing:backend-conformance:test --tests '*CpuNumericalProfileConformanceTest' --tests '*CpuOpenBlasRouteConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineNumericalProfileIntegrationTest'
git diff --check
```

Inspect generated CPU Javadocs and the materialized forward/objective/gradient values. The
integration owner runs the combined backend checkpoint, architecture/documentation checks, and full
build once after CPU 0017 and Metal 0015 are integrated.

## Documentation and review impact

The CPU worker updates CPU Javadocs and the CPU backend guide only. The integration owner updates
shared current-status wording. Independent Class C review must inspect capability-matrix identity,
exact arithmetic/route preservation, normalized generated bodies, OpenBLAS and cache separation,
forward/backward evidence, Runtime/Trace absence, and scope exclusions.

## Result

Empty until execution.

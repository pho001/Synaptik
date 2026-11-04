# Task 0017: Explicit [ACCELERATOR](../../../../glossary.md#term-accelerator) Numerical-Profile Realization

## Reader context — historical integration boundary

This task records an earlier numerical-profile interface: allowing the accelerator result set did
not force CPU to use relaxed arithmetic. CPU capability/preparation accepted the request, retained
it in identity, and executed the same strict realization; Runtime did not interpret profile policy.
That is the meaning of parity in this record, not proof that every native provider has identical
arithmetic. The profile names, identity versions and test outputs below remain historical. The
maintained integration reference now describes a profile-free CPU composition, so these old request
types are not instructions to add a profile parameter to today's API.

Current reading: [Supported lifecycle integration](../../../../backend-guide/cpu-backend.md#supported-lifecycle-integration).

## Status

Complete

Readiness verification: [Model](../../../../glossary.md#term-model) 0027 and Config 0006 are Complete. [Engine](../../../../glossary.md#term-engine) 0018 is Complete at
implementation `ce7a7dfa` plus documentation/[Javadoc](../../../../glossary.md#term-javadoc) remediation `07a01b9c`; its required
[validation](../../../../glossary.md#term-validation) and independent Class C review passed. The integrated spine already separates [CPU](../../../../glossary.md#term-cpu)
plans, generated [artifacts](../../../../glossary.md#term-artifact), [OpenBLAS](../../../../glossary.md#term-openblas) workloads, local tuning, and complete-plan tuning by profile.
CPU 0017 and [Metal](../../../../glossary.md#term-metal) 0015 have disjoint [backend](../../../../glossary.md#backend) behavior, source, test, and task-document [ownership](../../../../glossary.md#term-ownership-meanings);
shared status and explanatory documents are reserved to the integration [owner](../../../../glossary.md#term-owner).

## Change class

Class C — this changes the supported [CPU](../../../../glossary.md#term-cpu) capability and executable [Engine](../../../../glossary.md#term-engine) behavior for an entire
[graph](../../../../glossary.md#graph) profile across portable generated code, [OpenBLAS](../../../../glossary.md#term-openblas), tuning [identities](../../../../glossary.md#term-identity-meanings), forward execution, and
[Compiler](../../../../glossary.md#term-compiler)-generated backward graphs. It must prove that no arithmetic, route policy, [hot path](../../../../glossary.md#term-hot-path), or
[cache](../../../../glossary.md#term-cache) boundary is relaxed accidentally.

## Goal

Admit `NumericalProfile.ACCELERATOR` everywhere the current [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) admits `STRICT_IEEE`, while
executing the exact same current CPU semantics and routes. A strict result is valid under the
[Model](../../../../glossary.md#term-model)-owned [accelerator](../../../../glossary.md#term-accelerator) result set, so this first CPU realization adds permission and [identity](../../../../glossary.md#term-identity-meanings),
not relaxed math.

## Exact semantics and [scope](../../../../glossary.md#term-scope)

- `CpuCapabilityProvider` must answer an [`ACCELERATOR`](../../../../glossary.md#term-accelerator) query exactly as it answers the otherwise
  identical `STRICT_IEEE` query. Supported and rejected operation/type/Shape/layout/attribute/
  gradient rows remain identical; hardware, route availability, tuning, and workload size do not
  grant capability.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- `CpuPartitionPreparer` must accept both profiles and retain the selected profile in every existing
  `CpuPartitionPreparationPlan`, unit specialization, generated-artifact compatibility key,
  OpenBLAS workload, local-workload decision, and complete-plan decision.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [compatibility key glossary](../../../../glossary.md#cache-and-memoization); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Portable scalar/vector/parallel/generated behavior, exact operation evaluation, OpenBLAS
  qualification and calls, fusion/decomposition/materialization, thresholds, workers, species,
  constants, and fallbacks remain byte-for-byte policy equivalents across profiles. No DAZ, FTZ,
  reassociation, approximate instruction, reduced precision, fast math, or value-dependent route is
  introduced.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [evaluation glossary](../../../../glossary.md#term-evaluation); [OpenBLAS qualification glossary](../../../../glossary.md#openblas-qualification); [fusion glossary](../../../../glossary.md#term-fusion); [decomposition glossary](../../../../glossary.md#term-decomposition); [materialization glossary](../../../../glossary.md#materialization); [constants glossary](../../../../glossary.md#term-constant); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [reassociation glossary](../../../../glossary.md#term-reassociation); [precision glossary](../../../../glossary.md#term-precision))</span>
- Existing profile-separated [identity](../../../../glossary.md#term-identity-meanings) is authoritative: generator envelope `67`, OpenBLAS
  candidate/policy `2`, and local/complete value [schemas](../../../../glossary.md#term-schema) `2` remain current unless implementation
  discovers an actual encoded-format change. Do not bump a version merely because capability is
  enabled. Old, malformed, and cross-profile identities continue to miss or reject safely.
- Runtime, sessions, run state/results, Trace, generated entry signatures, and hot loops remain
  profile-free. Profile selection stays cold and immutable.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate); [Trace glossary](../../../../glossary.md#trace); [signatures glossary](../../../../glossary.md#term-signature); [immutable glossary](../../../../glossary.md#term-immutable))</span>

## Non-goals

No new [CPU](../../../../glossary.md#term-cpu) [operation](../../../../glossary.md#operation), [data type](../../../../glossary.md#data-type--datatype), route, provider, optimization, numerical relaxation, performance
claim, threshold/default change, public [API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module) edge, [cache](../../../../glossary.md#term-cache) format, [persistence](../../../../glossary.md#term-persistence) enablement, or
[Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) field. Do not begin CPU 0011–0016, [Model](../../../../glossary.md#term-model) 0026, or any [Metal](../../../../glossary.md#term-metal) work.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model owns
  result sets; the selected profile is cold and unchanged through plan/cache identity.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — [`ACCELERATOR`](../../../../glossary.md#term-accelerator) contains every strict result and grants no generic fast math.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — CPU owns truthful capability and route realization behind one backend identity.<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span> —
  selection finishes before Runtime; the hot path executes fixed work.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [hot path glossary](../../../../glossary.md#term-hot-path))</span>

If exact [CPU](../../../../glossary.md#term-cpu) behavior cannot be retained for an existing supported row, remove no strict capability
and do not narrow accelerator ad hoc; stop and report the row and counterexample.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; current [CPU](../../../../glossary.md#term-cpu) portable/[OpenBLAS](../../../../glossary.md#term-openblas) implementation
- Conflicts with: CPU capability, preparation, generated identity, OpenBLAS/tuning, Engine CPU
  integration, and CPU-specific guide/Javadoc scopes<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Parallel group: `numerical-profile-backends`
- Common base revision: `07a01b9c13c2ceea0922b9af82da3e6e08375306`
- Integration order: Any relative to Metal 0015; both executable commits precede shared-document
  reconciliation<span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Integration [validation](../../../../glossary.md#term-validation): numerical-profile [backend](../../../../glossary.md#backend) integration [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner

The CPU worker owns only CPU production/tests, `EngineNumericalProfileIntegrationTest`, this brief,
CPU package Javadocs, and `docs/backend-guide/cpu-backend.md`. It must not edit the roadmap, master
plans, root/scoped architecture, shared [API](../../../../glossary.md#term-application-programming-interface-api)/user/glossary/capability/preparer documents, or [Metal](../../../../glossary.md#term-metal)
paths. Those shared files are reconciled once after both parallel branches land.

## Exact files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
  — remove the accelerator-wide rejection; preserve one identical occurrence predicate.<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
  — remove the strict-only gate and continue projecting `context.numericalProfile()` unchanged.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java` and affected
  Javadocs in the two files above — describe exact dual-profile realization, not relaxed math.
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
  — run the retained admitted/rejected capability matrix under both profiles.<span data-glossary-navigation="true"> ([capability matrix glossary](../../../../glossary.md#review-tables-and-support-matrices))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [matrix glossary](../../../../glossary.md#term-review-matrix))</span>
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
  and `internal/cache/CpuKernelSpecializationTest.java` — prove equal plans/executable bodies with
  distinct profile identities and successful accelerator preparation.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasTuningBatchTest.java`
  plus the existing local/complete tuning public tests — retain deterministic same-profile encoding
  and cross-profile incompatibility at the current versions.
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuNumericalProfileConformanceTest.java`
  — new bounded public capability/preparation/execution parity checkpoint.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineNumericalProfileIntegrationTest.java`
  — replace accelerator rejection with observable strict and accelerator forward/backward runs.
- `docs/backend-guide/cpu-backend.md` — current [CPU](../../../../glossary.md#term-cpu) profile behavior and unchanged routes/semantics.

## Tests and executable [smoke](../../../../glossary.md#term-smoke-test)

- Capability conformance must compare otherwise equal strict/accelerator queries over every current
  family represented in <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>`CpuCapabilityProviderTest`, including rejected boundary rows; the two
  answers must always match.
- Prepare the same representative portable pointwise/reduction and eligible OpenBLAS MATMUL facts
  under both profiles. Plans must retain their exact profile; route, declarations, IR, strategy,
  workers, species, and operation semantics must match. Profile compatibility and structural keys
  must differ where required.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [pointwise glossary](../../../../glossary.md#term-pointwise); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [operation glossary](../../../../glossary.md#operation))</span>
- Generate the same representative classes under both profiles. Strip only profile-bearing class
  identity/constant-pool names and prove the normalized executable method bodies have the same
  hash. Do not modify performance tests, thresholds, generated instructions, or expected results.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [constant glossary](../../../../glossary.md#term-constant); [hash glossary](../../../../glossary.md#term-hash))</span>
- The actual `EngineNumericalProfileIntegrationTest` must construct separate strict and accelerator
  CPU Engines. For each profile, execute and materialize a nontrivial forward tensor result, then
  execute an observable Compiler-generated backward path and materialize the objective and gradient
  values. Exercise a reusable session on the forward path. Expected represented values are exact
  and identical across profiles; a mere compile/not-throw assertion is insufficient.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [objective glossary](../../../../glossary.md#term-objective); [gradient glossary](../../../../glossary.md#term-gradient); [compile glossary](../../../../glossary.md#compile))</span>
- Exercise both portable and qualified-OpenBLAS conformance where the test fixture supplies the
  provider. No native probe is required: this task changes no native ABI/provider arithmetic.<span data-glossary-navigation="true"> ([test fixture glossary](../../../../glossary.md#adr-ci-and-qualification); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>

## Acceptance criteria

1. CPU capability has exactly one current support matrix shared by both profiles; no rejected row
   becomes supported except by selecting the already supported row under <span data-glossary-navigation="true"> ([support matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [matrix glossary](../../../../glossary.md#term-review-matrix))</span>[`ACCELERATOR`](../../../../glossary.md#term-accelerator).
2. Both profiles compile, prepare, execute, and materialize observable forward and backward CPU
   results with exact current semantics and no strict fallback or profile rewrite.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare))</span>
3. Plans and every generated/OpenBLAS/tuning/cache identity retain the requested profile. Same-
   profile encoding is deterministic; different profiles never share identity; current version
   numbers remain unless an actual wire change is demonstrated.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
4. Normalized generated executable bodies match across profiles, and no arithmetic instruction,
   route/default, threshold, worker policy, or performance test changes.
5. Runtime/Trace and hot paths remain profile-free. No new public API, dependency, operation, type,
   provider, relaxed result, or performance claim.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [operation glossary](../../../../glossary.md#operation))</span>
6. Focused tests, Javadoc, conformance, actual Engine smoke, documentation validation, and
   independent Class C review pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :backends:cpu:test :backends:cpu:javadoc
./gradlew :testing:backend-conformance:test --tests '*CpuNumericalProfileConformanceTest' --tests '*CpuOpenBlasRouteConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineNumericalProfileIntegrationTest'
git diff --check
```

Inspect generated [CPU](../../../../glossary.md#term-cpu) Javadocs and the materialized forward/[objective](../../../../glossary.md#term-objective)/[gradient](../../../../glossary.md#term-gradient) values. The
integration [owner](../../../../glossary.md#term-owner) runs the combined [backend](../../../../glossary.md#backend) [checkpoint](../../../../glossary.md#term-checkpoint), architecture/documentation checks, and full
build once after CPU 0017 and [Metal](../../../../glossary.md#term-metal) 0015 are integrated.

## Documentation and review impact

The [CPU](../../../../glossary.md#term-cpu) worker updates CPU Javadocs and the CPU [backend](../../../../glossary.md#backend) guide only. The integration [owner](../../../../glossary.md#term-owner) updates
shared current-status wording. Independent Class C review must inspect capability-[matrix](../../../../glossary.md#term-review-matrix) [identity](../../../../glossary.md#term-identity-meanings),
exact arithmetic/route preservation, normalized generated bodies, [OpenBLAS](../../../../glossary.md#term-openblas) and [cache](../../../../glossary.md#term-cache) separation,
forward/backward evidence, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, and [scope](../../../../glossary.md#term-scope) exclusions.

## Result

Implementation, independent Class C review, combined [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) integration [validation](../../../../glossary.md#term-validation), and shared-
document reconciliation are complete.

- CPU capability now uses one exact predicate for `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); the retained
  admitted/rejected matrix executes under both profiles.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Preparation, OpenBLAS qualification, and OpenBLAS plan validation admit both profiles while
  retaining the request in plans, unit specializations, generated-artifact identity, workload
  identity, and local/complete tuning compatibility. Generator schema 67, OpenBLAS candidate and
  route-policy schema 2, and both tuning value schemas 2 remain unchanged.<span data-glossary-navigation="true"> ([OpenBLAS qualification glossary](../../../../glossary.md#openblas-qualification); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema))</span>
- Focused plan tests cover pointwise, exact floating reduction, and qualified MATMUL parity.
  Generated classes retain distinct profile-bearing identities while normalization of only the
  generated binary name produces the same SHA-256 class-body result.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [normalization glossary](../../../../glossary.md#term-normalization); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
- The bounded conformance checkpoint executes exact portable NEG results under both profiles; the
  native-free qualified OpenBLAS fixture executes exact MATMUL results under <span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [fixture glossary](../../../../glossary.md#term-fixture))</span>`ACCELERATOR`.
  Engine integration executes two reusable-session forward runs per profile and materializes<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
  `[0.3125, 8.75, 12.0]`, then executes Compiler-generated scalar backward graphs and materializes
  objective <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [scalar glossary](../../../../glossary.md#term-scalar); [graphs glossary](../../../../glossary.md#graph); [objective glossary](../../../../glossary.md#term-objective))</span>`6.0` and [gradient](../../../../glossary.md#term-gradient) `[5.0]` under both profiles.
- `./gradlew :backends:cpu:test :backends:cpu:javadoc` passed: 1,025 tests, zero failures/errors,
  28 existing conditional skips, and generated CPU Javadocs containing the final dual-profile
  contracts. The required focused conformance and Engine integration commands passed.
- The CPU guide/task Markdown audit found 21 local links with no missing targets, balanced fences,
  final newlines, and no trailing whitespace. <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>`git diff --check` passed before the result update
  and is repeated before commit.
- Independent Class C review found no remaining issue across capability/route parity, cold
  identity and unchanged versions, normalized generated bodies, OpenBLAS/tuning/cache separation,
  forward/backward/session evidence, Runtime/Trace/hot-path exclusions, documentation, and scope.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [scope glossary](../../../../glossary.md#term-scope))</span>
- The serial combined checkpoint passed: 1,025 CPU tests with zero failures/errors; 12 focused CPU
  conformance tests; one CPU Engine integration test; nine architecture tests; the full 87-task
  build; Runtime/Trace profile-free inspection; and baseline-aware documentation, Javadoc, status,
  and diff checks.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Implementation [commit](../../../../glossary.md#term-commit) `372a8b98` received final independent Class C `APPROVE` with zero findings.
  No limitation or unresolved issue remains.

Status: Complete

# Task 0008: Backend partition finalization glossary status reconciliation

## Status

Complete

## Change class

Class B — this documentation-only task corrects the current CPU status inside a shared lifecycle
term. It changes no contract or executable behavior, but receives a clean implementation context
and independent documentation review because the entry spans Prepare, CPU, and Engine composition.

## Goal

Reconcile only the stale CPU-status paragraph under `Backend partition finalization` with the
current public but unsupported CPU-private finalizer under `.internal` and its public
Engine-mediated composition, without changing the general finalization or prepared-resource
transaction contract.

## Scope

- State that the public finalizer remains an unsupported CPU-private implementation under
  `.internal` and is supplied through `CpuBackendIntegration` to current fixed ordinary and
  explicit advanced CPU-only Engine composition.
- Replace the obsolete exact-fused-FLOAT64 limitation with the current role: realize an already-
  selected CPU preparation plan across supported routes/families without route reselection or
  topology widening.
- Retain the current fact that CPU finalization returns no persistent prepared resources.
- Synchronize this task, the Prepare master plan, and the roadmap after independent review.

## Non-goals

- No Java, Javadoc, test, Gradle, API, behavior, dependency, architecture, or ownership change.
- No edit to the entry's general handoff, rollback, identity, transfer, or prohibited-work rules.
- No capability inventory, route-family matrix, operation list, tuning claim, or generic/mixed-
  backend composition claim.
- No edit to another glossary entry, guide, API page, ADR, or architecture contract.

## Contracts

- [`ARCHITECTURE.md` headings `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants)
  and [`Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing) —
  preserve Prepare/backend ownership and cold-finalization boundaries.
- [`runtime-prepare-engine.md` headings `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [`Prepare lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — preserve finalization, resource transfer, and current CPU-only Engine composition.
- [`backend-execution.md` heading `Concrete backend modules`](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  — CPU retains lowering, route, artifact, executable, and physical-mechanics ownership.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/glossary.md` — only the final current-CPU paragraph under `Backend partition finalization`.
- `docs/planning/modules/prepare/tasks/0008-backend-partition-finalization-glossary-status-reconciliation.md`
  — status and compact result evidence.
- `docs/planning/modules/prepare/master-plan.md` — task row and frontier summary only.
- `docs/planning/roadmap.md` — Prepare area, frontier, drift list, and next step only.

Read-only evidence:

- `CpuBackendIntegration.partitionPreparation()` and `CpuBackendComposition.preparation()`.
- `CpuPartitionFinalizer` type/finalization Javadoc and `finalizePartition(...)`.
- `Engine.standard()`, `AdvancedEngine.takeOwnership(...)`, and `CpuEngineBackendComposition`.
- Focused CPU finalizer/integration and Engine standard-composition tests.
- `docs/backend-guide/partition-preparer.md` focused finalization, resource, composition, and
  limitation paragraphs; `docs/backend-guide/cpu-backend.md` current integration/status sections.

No path outside the four-file implementation allowlist may change.

## Acceptance criteria

- The final paragraph accurately distinguishes the public but unsupported CPU-private finalizer
  under `.internal` from the package-private Engine composition and public Engine lifecycle that
  indirectly reach it through the supported CPU integration.
- It says the finalizer realizes only an already-selected plan and cannot reselect routes or widen
  topology; it does not reduce current CPU support to one fused FLOAT64 topology.
- It keeps CPU's empty persistent-resource result and does not imply executable/resource
  persistence, generic registration, mixed-owner execution, or new backend capability.
- The preceding general finalization/transaction paragraphs remain unchanged.
- Independent Class B documentation review confirms source/contract alignment and reasoned no-
  change conclusions for architecture, ADRs, guides, other glossary terms, Javadoc, tests, Gradle,
  and executable behavior.
- Only the four allowed paths change; brief stays at most 200 lines and 15 KB; Markdown links and
  anchors, headings, fences, whitespace, final newlines, exact path scope, and `git diff --check`
  validate.

## Validation

Implementation context:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.spi.CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.EngineStandardCompositionTest
python3 /tmp/validate_synaptik_markdown.py \
  docs/glossary.md \
  docs/planning/modules/prepare/tasks/0008-backend-partition-finalization-glossary-status-reconciliation.md \
  docs/planning/modules/prepare/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/prepare/tasks/0008-backend-partition-finalization-glossary-status-reconciliation.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The Java/Gradle scan must be empty and the path audit must equal the four-file allowlist. If the
temporary Markdown validator is absent, create an equivalent validator outside the repository.
Review reuses successful focused-test evidence unless executable behavior changes and reruns final
documentation, size, scope, and whitespace checks.

Repository-wide validation is deferred because this task changes one current-status paragraph and
planning state, with no executable, build, dependency, or architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-authorized after clean HEAD `baa626c6`; Prepare 0007, CPU 0010K,
  Engine 0013, and Compiler 0006B10 are Complete, so 0008 is the sole authorized `Ready` frontier.
- No product implementation or unrelated Draft, blocked, or review-needed task is authorized.

## Documentation and review impact

- Types: shared glossary status and planning state; apply General and Planning profiles plus the
  General profile's glossary rules.
- A clean implementation context and independent targeted documentation review are required.

## Result

Corrected only the final current-CPU paragraph under `Backend partition finalization` and
synchronized this task, the Prepare master plan, and the roadmap. Independent Class B review
corrected the brief's stale visibility wording and confirmed that the public
`CpuPartitionFinalizer` remains unsupported under `.internal`, while package-private
`CpuEngineBackendComposition` reaches it through `CpuBackendIntegration.partitionPreparation()`
for fixed ordinary and explicit advanced CPU-only composition. The finalizer realizes the
already-selected plan across current supported routes and operation families without reselection
or topology widening, and returns no persistent prepared resources. The focused implementation
tests passed with 17 finalizer and three integration tests plus six Engine tests, all without
failure or skip; review reused them because no executable code changed. A focused `javap` check
confirmed the finalizer is public and the Engine composition is package-private. General and
Planning profiles were applied; architecture, ADRs, guides, other glossary terms, Javadoc, tests,
Gradle, and executable behavior require no change because the edit reconciles documentation with
existing source and contracts. The Markdown validator accepted all four files; the brief measured
152 lines and 8,593 bytes, the Java/Gradle scan was empty, the
path audit contained exactly the four allowed files, and `git diff --check` passed. No unresolved
issue or follow-up remains.

Status: Complete

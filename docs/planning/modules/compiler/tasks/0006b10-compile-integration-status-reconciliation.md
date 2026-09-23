# Task 0006B10: Compile integration status reconciliation

## Status

Complete

## Change class

Class B — this documentation-only task corrects a public Compiler integration type and the shared
compile term. It changes no API or executable behavior, but requires a clean implementation context
and independent documentation review because the stale wording misstates the current public Engine
lifecycle.

## Goal

Reconcile `GraphCompilationPort` Javadoc and the glossary `Compile` entry with the current Engine
consumer while preserving the port's narrow cross-module SPI role and all current compile semantics.

## Scope

- Replace the future-Engine wording in `GraphCompilationPort` type Javadoc with the current role:
  lifecycle-composition SPI used by Engine, not an ordinary application-facing standalone compiler.
- Replace the stale glossary sentence that says the Engine compile lifecycle remains planned.
- Keep preparation and execution outside `GraphCompilationPort`; identify current public Engine
  compile/prepare/run composition without implying generic provider registration.
- Synchronize this task, the Compiler master plan, and the roadmap after independent review.

## Non-goals

- No Java signature, implementation, test, Gradle, behavior, dependency, or architecture change.
- No edit to `ARCHITECTURE.md`, scoped contracts, ADRs, API/backend guides, or other glossary entries.
- No claim that `GraphCompiler` is public, that applications should call the SPI directly, or that
  Engine supports arbitrary backend/provider registration.
- No repair of the separate `Backend partition finalization` glossary status paragraph.

## Contracts

- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) —
  compile creates immutable backend-neutral artifacts and no physical buffers.
- [`compiler-autograd.md` heading `Scope`](../../../../architecture/contracts/compiler-autograd.md#scope)
  — Compiler owns graph compilation and immutable compile artifacts, not preparation or execution.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — current Engine owns public lifecycle composition and fixed/advanced CPU-only integration.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java` — type
  Javadoc only.
- `docs/glossary.md` — heading `Compile` only.
- `docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md` —
  status and compact result evidence.
- `docs/planning/modules/compiler/master-plan.md` — task row and frontier summary only.
- `docs/planning/roadmap.md` — Compiler area, frontier, discovered-drift list, and next step only.

Read-only evidence:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/spi/GraphCompilationPortPublicShapeTest.java`.
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`.
- `docs/api/compile-api.md` focused `GraphCompilationPort` and `AdvancedEngine.compile(...)` sections.
- `docs/api/public-api.md` focused Compiler and Engine status sections.

No path outside the five-file implementation allowlist may change.

## Acceptance criteria

- Type Javadoc states that `GraphCompilationPort` is the current narrow integration SPI used by
  Engine and other lifecycle-composition code, while ordinary application code uses Engine.
- Type Javadoc still states that the port itself provides no Engine lifecycle, preparation, or
  execution.
- The glossary says current Engine composes compile, publication delivery, preparation, and
  execution, without claiming that those responsibilities belong to Compiler or the port.
- Package-private `GraphCompiler`, empty explicit constant ingress, compile artifacts, and all
  executable semantics remain unchanged.
- API guides and public API status require no edit because they already describe the current role.
- Independent Class B documentation review completes and records reasoned no-change conclusions
  for architecture, ADRs, other guides/glossary entries, tests, Gradle, and executable behavior.
- Only the five allowed paths change; brief stays at most 200 lines and 15 KB; Javadoc succeeds;
  Markdown links/anchors, headings, fences, whitespace, and final newlines validate; and
  `git diff --check` passes.

## Validation

Implementation context:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/glossary.md \
  docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md \
  docs/planning/modules/compiler/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/compiler/tasks/0006b10-compile-integration-status-reconciliation.md
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The path audit must equal the five-file allowlist. If the temporary Markdown validator is absent,
create an equivalent validator outside the repository. Review reuses successful executable
evidence unless it changes Java and reruns final documentation, scope, and whitespace checks.

Repository-wide validation is deferred because this task changes only status wording and type
Javadoc, with no executable, dependency, build, or architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-authorized after clean HEAD `e25b26d6`; current source and API guides
  already agree, so 0006B10 is the sole authorized `Ready` frontier.
- Follow-up: separately reconcile the stale `Backend partition finalization` glossary paragraph.
- Compiler 0006C/0007 and every unrelated Draft, blocked, or review-needed item remain unauthorized.

## Documentation and review impact

- Types: public SPI Javadoc, glossary terminology/status, and planning status; apply General,
  API/Javadoc, and Planning profiles, with the General profile's glossary-alignment rules.
- A clean implementation context and independent targeted documentation review are required.

## Result

Implementation corrected the public SPI Javadoc and shared `Compile` glossary entry against the
current fixed CPU-only Engine composition. Independent Class B review confirmed that Engine uses
`GraphCompilationPort`, ordinary applications use Engine, and fixed ordinary plus explicitly
owned advanced composition remain CPU-only. Review clarified that publication delivery belongs
to the ordinary lifecycle and corrected the read-only `AdvancedEngine` source path in this brief.

The focused public-shape test and Compiler Javadoc succeeded in the implementation context and
were reused because review changed no executable Java or final Javadoc. Final Markdown, brief-size,
exact five-path, and whitespace checks passed. The API guides require no edit because they already
state the current boundaries. Architecture and ADRs require no edit because ownership and module
edges did not change; other guides/glossary entries, tests, Gradle, and executable behavior require
no edit because this task only reconciles current documentation. The separately stale backend-
partition-finalization glossary status remains follow-up work.

Status: Complete

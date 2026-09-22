# Task 0006B9: Compile-artifacts contract reconciliation

## Status

Complete

## Change class

Class C — this documentation-only task changes the normative Compiler-owned artifact boundary.
Although it reconciles the contract with already implemented behavior and changes no Java, classify
upward because an incomplete architecture contract can misdirect later Prepare and Engine work.

## Goal

Reconcile the normative `CompileArtifacts` block with the exact current public eight-component
record while preserving its compile-time-only boundary and all current executable behavior.

## Scope

- Replace the stale five-component presentation under `Compile artifacts` with an exact semantic
  component list for mode, graph, partitions, memory, publication, constants, diagnostics, and
  derivatives.
- Keep the component descriptions aligned with current source, constructor tests, and the focused
  current sections of the Compile API and public-API status page.
- Prefer the semantic component list over embedding a Java record signature that would duplicate
  source syntax and become brittle, while still naming all eight current component types exactly.
- Preserve the existing prohibition on physical, prepared, backend-executable, kernel-route,
  runtime-workspace, residency, and mutable run-state fields.
- Complete the required independent review and synchronize this task, the Compiler master plan,
  and the roadmap after acceptance.

## Non-goals

- No Java source, test, Javadoc, Gradle, public API, or observable behavior change.
- No change to `ARCHITECTURE.md`, another scoped contract, an architecture decision record (ADR),
  an explanatory architecture page, an API guide, or the glossary.
- No change to automatic-differentiation semantics, compile lifecycle, ownership, dependencies,
  preparation, execution, or the status of unrelated work.
- No cleanup of unrelated stale documentation or planning status.

## Contracts

- [`ARCHITECTURE.md` heading `Authority, incorporation, and precedence`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  — the root incorporates exactly one owning scoped contract and wins on conflict.
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants)
  — `CompileArtifacts` is immutable compile-time output and Compiler allocates no physical buffer.
- [`compiler-autograd.md` heading `Scope`](../../../../architecture/contracts/compiler-autograd.md#scope)
  — this scoped contract owns compile artifacts, not physical resources or prepared execution.
- [`compiler-autograd.md` heading `Compile artifacts`](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)
  — the sole normative block to reconcile.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/architecture/contracts/compiler-autograd.md` — only heading `Compile artifacts`.
- `docs/planning/modules/compiler/tasks/0006b9-compile-artifacts-contract-reconciliation.md` —
  status and compact result evidence.
- `docs/planning/modules/compiler/master-plan.md` — task row and current-frontier summary only.
- `docs/planning/roadmap.md` — Compiler area and current-frontier summary only.

Read-only evidence:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java` — exact
  record components and constructor invariants.
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileArtifactsTest.java` —
  exact eight-component record-order test and constructor coverage.
- `docs/api/compile-api.md` heading `Current complete artifact compilation and integration port`
  and its `CompileArtifacts` subsection — exact current component and exclusion descriptions.
- `docs/api/public-api.md` heading `Current public contracts`, Compiler list and `Compile artifacts`
  glossary-style entry — exact current public boundary.

No path outside the four-file implementation allowlist may change.

## Acceptance criteria

- The normative block describes `CompileArtifacts` as one immutable compile-time recipe containing
  exactly these eight semantic components in current record order: exact `CompileMode`; exact final
  `CompiledGraphModel`; immutable graph-order `List<PlannedPartition>` membership for the maximal
  partitions; derived `LogicalMemoryPlan`; `PublicationPlan`; `CompileConstantPlan`;
  `CompileDiagnostics`; and `DerivativeGraphMetadata`.
- The wording distinguishes logical compile output from physical or executable state and retains
  every existing forbidden category: physical buffers, prepared or backend executable objects,
  concrete kernel routes, runtime workspaces, runtime residency, and mutable run state.
- The contract embeds no Java record signature and makes no new claim about constructor mechanics,
  automatic differentiation, preparation, execution, or backend behavior.
- Current source, the focused constructor test, Compile API, and public-API status page require no
  edits because they already agree on the exact eight-component public record.
- Root architecture, ADR, explanatory architecture, glossary, Javadoc, Java, tests, and Gradle
  remain unchanged; no unrelated status cleanup occurs.
- The clean documentation implementation and an independent targeted Class C review both complete.
  The reviewer inspects the final diff and read-only evidence, validates terminology and links,
  records reasoned no-change conclusions for every excluded documentation category, and does not
  rerun successful executable evidence unless executable behavior changes.
- Only the four allowed implementation paths change; task/master/roadmap status remains
  synchronized; the brief remains at most 200 lines and 15 KB; local links and anchors resolve;
  headings are unique; fences are balanced; files end with newlines and have no trailing
  whitespace; and `git diff --check` passes.

## Validation

Implementation context:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.CompileArtifactsTest
python3 /tmp/validate_synaptik_markdown.py \
  docs/architecture/contracts/compiler-autograd.md \
  docs/planning/modules/compiler/tasks/0006b9-compile-artifacts-contract-reconciliation.md \
  docs/planning/modules/compiler/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/compiler/tasks/0006b9-compile-artifacts-contract-reconciliation.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The Java/Gradle path scan must be empty and the complete path audit must equal the four-file
allowlist. If the temporary Markdown validator is absent, create an equivalent validator outside
the repository for local link targets and anchors, unique headings, balanced fences, trailing
whitespace, and final newlines.

The independent review reuses the focused successful test evidence, reruns the final Markdown,
size, scope, and `git diff --check` commands after its edits, and records any reason for repeating
the executable test.

Repository-wide validation: deferred because this task changes one scoped documentation block and
planning state, with no code, dependency, build, or executable architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-selected from clean HEAD `375d84be` after Compiler 0006B8 and the
  drift-remediation sequence through Engine 0012 completed. The source, focused constructor test,
  and two current API sections already agree; dependencies are satisfied and 0006B9 is the sole
  authorized `Ready` frontier.
- Compiler 0006C and 0007 remain `Draft` and unauthorized. This reconciliation grants no product
  implementation or follow-up capability.

## Documentation and review impact

- Primary document type: normative architecture contract; apply General and Architecture style.
  Planning files apply Planning style. There is no example, API, or Javadoc edit.
- Glossary impact: none; targeted `CompileArtifacts`, compile, prepare, and immutable-recipe terms
  retain their current meanings.
- A clean documentation implementation context and a separate independent targeted review context
  are mandatory because this Class C task edits a durable normative boundary.

## Result

- Clean documentation implementation replaced the stale five-component Java signature with the
  exact eight-component semantic recipe in current record order and retained every forbidden
  physical, prepared, backend-executable, and runtime-state category.
- Independent targeted Class C review against `b50e42cf` inspected the final four-path diff,
  `CompileArtifacts` source and constructor test, the focused Compile API and public-API sections,
  and the targeted glossary terms. It confirmed the exact eight types, order, roles, immutable
  partition membership, logical-memory derivation, and compile-time-only exclusions with no
  findings.
- `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.CompileArtifactsTest` passed (`BUILD SUCCESSFUL`) in the
  implementation context and was reused because review changed no executable behavior.
- Final review validation passed the four-file Markdown validator, task-size guardrail,
  Java/Gradle scan, exact four-path audit, and `git diff --check`.
- Root architecture, ADRs, explanatory architecture, API guides, glossary, Javadoc, Java, tests,
  Gradle, automatic-differentiation semantics, compile behavior, and unrelated status require no
  change because the existing evidence already agrees and this task only reconciles the owning
  normative block.
- No unresolved issues or follow-up are required; Compiler 0006C and 0007 remain unauthorized
  `Draft` work.

Status: Complete

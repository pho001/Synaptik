# Task 0006B9: [Compile](../../../../glossary.md#compile)-[artifacts](../../../../glossary.md#term-artifact) contract reconciliation

## Status

Complete

## Change class

Class C — this documentation-only task changes the [normative](../../../../glossary.md#term-normative) [Compiler](../../../../glossary.md#term-compiler)-owned [artifact](../../../../glossary.md#term-artifact) boundary.
Although it reconciles the contract with already implemented behavior and changes no Java, classify
upward because an incomplete [architecture contract](../../../../glossary.md#architecture-contract) can misdirect later [Prepare](../../../../glossary.md#prepare) and [Engine](../../../../glossary.md#term-engine) work.

## Goal

Reconcile the [normative](../../../../glossary.md#term-normative) `CompileArtifacts` block with the exact current public eight-component
record while preserving its [compile](../../../../glossary.md#compile)-time-only boundary and all current executable behavior.

## [Scope](../../../../glossary.md#term-scope)

- Replace the stale five-component presentation under [`Compile artifacts`](../../../../glossary.md#compile-artifacts) with an exact semantic
  component list for mode, graph, partitions, memory, publication, constants, diagnostics, and
  derivatives.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [constants glossary](../../../../glossary.md#term-constant); [derivatives glossary](../../../../glossary.md#term-derivative))</span>
- Keep the component descriptions aligned with current source, constructor tests, and the focused
  current sections of the Compile API and public-API status page.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Prefer the semantic component list over embedding a Java record signature that would duplicate
  source syntax and become brittle, while still naming all eight current component types exactly.<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [signature glossary](../../../../glossary.md#term-signature))</span>
- Preserve the existing prohibition on physical, prepared, backend-executable, kernel-route,
  runtime-workspace, residency, and mutable run-state fields.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [mutable glossary](../../../../glossary.md#term-mutable); [run glossary](../../../../glossary.md#term-run))</span>
- Complete the required independent review and synchronize this task, the Compiler master plan,
  and the roadmap after acceptance.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Non-goals

- No Java source, test, [Javadoc](../../../../glossary.md#term-javadoc), [Gradle](../../../../glossary.md#term-gradle), public [API](../../../../glossary.md#term-application-programming-interface-api), or observable behavior change.
- No change to `ARCHITECTURE.md`, another scoped contract, an architecture decision record (ADR),
  an explanatory architecture page, an API guide, or the glossary.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#adr-ci-and-qualification))</span><span data-glossary-navigation="true"> ([architecture decision record glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- No change to automatic-differentiation semantics, compile lifecycle, ownership, dependencies,
  preparation, execution, or the status of unrelated work.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No cleanup of unrelated stale documentation or [planning](../../../../glossary.md#term-planning) status.

## Contracts

- [`ARCHITECTURE.md` heading `Authority, incorporation, and precedence`](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  — the root incorporates exactly one owning scoped contract and wins on conflict.
- [`ARCHITECTURE.md` heading `Core invariants`](../../../../../ARCHITECTURE.md#core-invariants)
  — `CompileArtifacts` is [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)-time output and [Compiler](../../../../glossary.md#term-compiler) allocates no physical [buffer](../../../../glossary.md#term-buffer).
- [`compiler-autograd.md` heading `Scope`](../../../../architecture/contracts/compiler-autograd.md#scope)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
  — this scoped contract owns compile artifacts, not physical resources or prepared execution.<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- [`compiler-autograd.md` heading `Compile artifacts`](../../../../architecture/contracts/compiler-autograd.md#compile-artifacts)
  — the sole normative block to reconcile.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/architecture/contracts/compiler-autograd.md` — only heading [`Compile artifacts`](../../../../glossary.md#compile-artifacts).
- `docs/planning/modules/compiler/tasks/0006b9-compile-artifacts-contract-reconciliation.md` —
  status and compact result evidence.
- `docs/planning/modules/compiler/master-plan.md` — task row and current-frontier summary only.
- `docs/planning/roadmap.md` — [Compiler](../../../../glossary.md#term-compiler) area and current-frontier summary only.

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

- The [normative](../../../../glossary.md#term-normative) block describes `CompileArtifacts` as one immutable compile-time recipe containing
  exactly these eight semantic components in current record order: exact <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>[`CompileMode`](../../../../glossary.md#compile-mode--compilemode); exact final
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel); [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph)-order `List<PlannedPartition>` membership for the maximal
  partitions; derived `LogicalMemoryPlan`; `PublicationPlan`; `CompileConstantPlan`;
  `CompileDiagnostics`; and `DerivativeGraphMetadata`.
- The wording distinguishes logical compile output from physical or executable state and retains
  every existing forbidden category: physical buffers, prepared or backend executable objects,
  concrete kernel routes, runtime workspaces, runtime residency, and mutable run state.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel); [runtime glossary](../../../../glossary.md#term-runtime); [residency glossary](../../../../glossary.md#residency); [mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- The contract embeds no Java record signature and makes no new claim about constructor mechanics,
  automatic differentiation, preparation, execution, or backend behavior.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- Current source, the focused constructor test, Compile API, and public-API status page require no
  edits because they already agree on the exact eight-component public record.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Root architecture, ADR, explanatory architecture, glossary, Javadoc, Java, tests, and Gradle
  remain unchanged; no unrelated status cleanup occurs.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The clean documentation implementation and an independent targeted Class C review both complete.
  The reviewer inspects the final diff and read-only evidence, validates terminology and links,
  records reasoned no-change conclusions for every excluded documentation category, and does not
  rerun successful executable evidence unless executable behavior changes.
- Only the four allowed implementation paths change; task/master/roadmap status remains
  synchronized; the brief remains at most 200 lines and 15 KB; local links and anchors resolve;
  headings are unique; fences are balanced; files end with newlines and have no trailing
  whitespace; and `git diff --check` passes.

## [Validation](../../../../glossary.md#term-validation)

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

The Java/[Gradle](../../../../glossary.md#term-gradle) path [scan](../../../../glossary.md#term-cumulative-scan) must be empty and the complete path audit must equal the four-file
allowlist. If the temporary Markdown validator is absent, create an equivalent validator outside
the repository for local link [targets](../../../../glossary.md#term-target) and anchors, unique headings, balanced fences, trailing
whitespace, and final newlines.

The independent review reuses the focused successful test evidence, reruns the final Markdown,
size, [scope](../../../../glossary.md#term-scope), and `git diff --check` commands after its edits, and records any reason for repeating
the executable test.

Repository-wide validation: deferred because this task changes one scoped documentation block and
[planning](../../../../glossary.md#term-planning) state, with no code, [dependency](../../../../glossary.md#term-dependency), build, or executable architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-selected from clean HEAD `375d84be` after Compiler 0006B8 and the
  drift-remediation sequence through Engine 0012 completed. The source, focused constructor test,
  and two current API sections already agree; dependencies are satisfied and 0006B9 is the sole
  authorized <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>[`Ready`](../../../../glossary.md#term-ready) frontier.
- [Compiler](../../../../glossary.md#term-compiler) 0006C and 0007 remain `Draft` and unauthorized. This reconciliation grants no product
  implementation or follow-up capability.

## Documentation and review impact

- Primary document type: normative architecture contract; apply General and Architecture style.
  Planning files apply Planning style. There is no example, API, or Javadoc edit.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [architecture contract glossary](../../../../glossary.md#architecture-contract); [Planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Glossary impact: none; targeted `CompileArtifacts`, compile, prepare, and immutable-recipe terms
  retain their current meanings.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- A clean documentation implementation context and a separate independent targeted review context
  are mandatory because this Class C task edits a durable normative boundary.

## Result

- Clean documentation implementation replaced the stale five-component Java signature with the
  exact eight-component semantic recipe in current record order and retained every forbidden
  physical, prepared, backend-executable, and runtime-state category.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Independent targeted Class C review against `b50e42cf` inspected the final four-path diff,
  `CompileArtifacts` source and constructor test, the focused Compile API and public-API sections,
  and the targeted glossary terms. It confirmed the exact eight types, order, roles, immutable
  partition membership, logical-memory derivation, and compile-time-only exclusions with no
  findings.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [immutable glossary](../../../../glossary.md#term-immutable); [partition glossary](../../../../glossary.md#partition))</span>
- `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.CompileArtifactsTest` passed (`BUILD SUCCESSFUL`) in the
  implementation context and was reused because review changed no executable behavior.
- Final review validation passed the four-file Markdown validator, task-size guardrail,
  Java/Gradle scan, exact four-path audit, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Gradle glossary](../../../../glossary.md#term-gradle); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`git diff --check`.
- Root architecture, ADRs, explanatory architecture, API guides, glossary, Javadoc, Java, tests,
  Gradle, automatic-differentiation semantics, compile behavior, and unrelated status require no
  change because the existing evidence already agrees and this task only reconciles the owning
  normative block.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [normative glossary](../../../../glossary.md#term-normative))</span>
- No unresolved issues or follow-up are required; [Compiler](../../../../glossary.md#term-compiler) 0006C and 0007 remain unauthorized
  `Draft` work.

Status: Complete

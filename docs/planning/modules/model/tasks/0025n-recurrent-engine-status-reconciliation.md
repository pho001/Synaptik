# Task 0025N: Recurrent Engine status reconciliation

## Status

Complete

## Change class

Class C — this documentation-only task corrects one sentence in the normative recurrent-scan
contract. It changes no behavior or ownership, but requires clean implementation and independent
review because the sentence distinguishes the current Engine facade from future recurrent
execution and exception translation.

## Goal

State precisely that Engine already exists and owns lifecycle composition, while a public
recurrent-execution path and its exception translation remain future work.

## Scope

- Replace only the stale `future Engine facade` sentence in the recurrent runtime-validation
  paragraph.
- Preserve backend responsibility for validating the complete length vector before output mutation.
- Preserve Engine ownership of any future public exception translation without selecting an
  exception type.
- Synchronize this task, the Model master plan, and the roadmap after independent review.

## Non-goals

- No Java, Javadoc, tests, Gradle, API, behavior, operation semantics, backend capability,
  recurrent execution, backpropagation through time (BPTT), or exception-taxonomy implementation.
- No edit to another recurrent contract paragraph, architecture file, guide, API page, glossary,
  ADR, or planning area.
- No claim that a current backend executes RNN, GRU, or LSTM scans.

## Contracts

- [`ARCHITECTURE.md` heading `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  — Engine owns public lifecycle composition; Model owns operation semantics.
- [`recurrent-scan.md` heading `Valid lengths, traversal, and outputs`](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs)
  — the sole normative paragraph to reconcile.
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine is the current composition root and generic/mixed composition remains planned.

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/architecture/contracts/recurrent-scan.md` — only the final exception-translation sentence
  under `Valid lengths, traversal, and outputs`.
- `docs/planning/modules/model/tasks/0025n-recurrent-engine-status-reconciliation.md` — status and
  compact result evidence.
- `docs/planning/modules/model/master-plan.md` — task row and frontier summary only.
- `docs/planning/roadmap.md` — Model area, repository frontier, and next step only.

Read-only evidence:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java` and
  `AdvancedEngine.java` — current public facade and lifecycle composition.
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java`
  — current forward-only Compiler adoption.
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanSemanticsTest.java`
  — current operation semantics.
- Model master-plan recurrent-execution risk and focused public/compile API recurrent sections.

No path outside the four-file implementation allowlist may change.

## Acceptance criteria

- The contract no longer calls the Engine facade future.
- It states that Engine owns any future public recurrent-execution exception translation and that
  the recurrent contract does not select the public exception type.
- It does not imply current recurrent backend execution, current exception translation, BPTT, or
  a new Engine API.
- The rest of `Valid lengths, traversal, and outputs` remains byte-for-byte unchanged.
- Independent Class C review confirms wording against current authority/source/status and records
  reasoned no-change conclusions for root architecture, other scoped contracts, ADRs, API/guides,
  glossary, Javadoc, tests, Gradle, and executable behavior.
- Only the four allowed paths change; brief stays at most 200 lines and 15 KB; Markdown links and
  anchors, headings, fences, whitespace, final newlines, exact scope, and `git diff --check` pass.

## Validation

Implementation context:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest
python3 /tmp/validate_synaptik_markdown.py \
  docs/architecture/contracts/recurrent-scan.md \
  docs/planning/modules/model/tasks/0025n-recurrent-engine-status-reconciliation.md \
  docs/planning/modules/model/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/modules/model/tasks/0025n-recurrent-engine-status-reconciliation.md
git diff --name-only -- '*.java' '*.gradle' '*.gradle.kts'
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git diff --check
```

The Java/Gradle scan must be empty and the path audit must equal the four-file allowlist. If the
temporary Markdown validator is absent, create an equivalent validator outside the repository.
Review reuses successful focused-test evidence unless executable behavior changes and reruns final
documentation, size, scope, and whitespace checks.

Repository-wide validation is deferred because this task changes one status sentence and planning
state, with no executable, build, dependency, or architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-authorized after clean HEAD `5b9ef70c`; Model 0025E–0025F,
  Compiler 0006A, Engine 0013, and the documentation-drift sequence through Prepare 0008 are
  Complete, so 0025N is the sole authorized `Ready` frontier.
- Model 0026 and all unrelated Draft, blocked, or review-needed work remain unauthorized.

## Documentation and review impact

- Types: normative architecture contract and planning state; apply General, Architecture, and
  Planning profiles.
- A clean implementation context and independent targeted Class C review are mandatory.

## Result

- Replaced the stale final sentence under `Valid lengths, traversal, and outputs`: Engine is
  current, while public recurrent-execution exception translation remains future work and this
  contract selects no public exception type.
- Corrected this brief's stale heading label and anchor to the actual owning contract heading.
- Synchronized this brief, the Model master plan, and the roadmap at `Complete` after the
  mandatory independent Class C review.
- Changed only the four allowlisted Markdown paths; Java/Gradle scan was empty.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest` passed.
- `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest` passed.
- The independent Class C review confirmed current Engine lifecycle composition from `Engine` and
  `AdvancedEngine`, Compiler forward-only adoption, absent recurrent backend capability, and the
  fail-closed BPTT boundary. The wording implies no current recurrent execution, translation, or
  new API and selects no public exception type. It reused the successful focused tests because no
  executable behavior changed.
- The specified Markdown validator passed all four files; the brief remained within 200 lines and
  15 KB; exact-path audit and `git diff --check` passed.
- Documentation impact is limited to the normative status correction and synchronized planning
  evidence. Root architecture and other scoped contracts already assign lifecycle composition to
  Engine and execution to backends, so they need no edit. ADRs, API/guides, and the glossary
  already preserve the same current/future boundary and gain no decision, workflow, or term.
  Javadocs, tests, Gradle, and executable behavior need no change because this task changes no
  Java contract, build input, or behavior.
- Repository-wide validation remains deferred for the reason recorded above.

Status: Complete

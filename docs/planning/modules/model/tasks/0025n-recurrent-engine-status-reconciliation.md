# Task 0025N: Recurrent [Engine](../../../../glossary.md#term-engine) status reconciliation

**Editorial reading key (current context, not a status update).**

This completed task records a historical wording correction, not a recurrent-execution delivery:
Engine already composed public lifecycles, while public recurrent execution and exception
translation remained future work in that record. Model defines operation meaning; execution
providers realize supported routes without redefining it. That semantic responsibility is not
ownership of result storage. The [current fixed-scan
reference](../../../../reference/tensor/fixed-scans.md#fixed-recurrent-scan-expressions)
distinguishes storage-free construction from supported CPU execution; its later capabilities must
not be read back into this task.

A caller supplies Tensor arguments to `RecurrentScan`; construction checks descriptors—shape and
type descriptions—not the stored length values, and returns expressions without executing or owning
result buffers. At execution, the backend must validate every length value before writing any scan
output. A valid length counts active steps, not allocated time slots: a zero-length row retains its
initial state and has positive-zero dense outputs. Engine's lifecycle role is distinct from this
backend duty; the historical correction assigns Engine any future public exception translation
without choosing an exception type.

## Status

Complete

## Change class

Class C — this documentation-only task corrects one sentence in the [normative](../../../../glossary.md#term-normative) recurrent-scan
contract. It changes no behavior or [ownership](../../../../glossary.md#term-ownership-meanings), but requires clean implementation and independent
review because the sentence distinguishes the current [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade) from future recurrent
execution and exception translation.

## Goal

State precisely that [Engine](../../../../glossary.md#term-engine) already exists and owns [lifecycle](../../../../glossary.md#lifecycle) composition, while a public
recurrent-execution path and its exception translation remain future work.

## [Scope](../../../../glossary.md#term-scope)

- Replace only the stale `future Engine facade` sentence in the recurrent runtime-validation
  paragraph.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation))</span>
- Preserve [backend](../../../../glossary.md#backend) responsibility for validating the complete length [vector](../../../../glossary.md#term-vector) before output mutation.
- Preserve Engine ownership of any future public exception translation without selecting an
  exception type.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Synchronize this task, the [Model](../../../../glossary.md#term-model) master plan, and the roadmap after independent review.

## Non-goals

- No Java, Javadoc, tests, Gradle, API, behavior, operation semantics, backend capability,
  recurrent execution, backpropagation through time (BPTT), or exception-taxonomy implementation.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [operation glossary](../../../../glossary.md#operation); [backend capability glossary](../../../../glossary.md#backend-capability); [backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- No edit to another recurrent contract paragraph, architecture file, guide, API page, glossary,
  ADR, or planning area.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [planning glossary](../../../../glossary.md#term-planning))</span>
- No claim that a current [backend](../../../../glossary.md#backend) executes [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn), [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru), or [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) scans.

## Contracts

- [`ARCHITECTURE.md` heading `Module-ownership routing`](../../../../../ARCHITECTURE.md#module-ownership-routing)
  — Engine owns public lifecycle composition; Model owns operation semantics.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation))</span>
- [`recurrent-scan.md` heading `Valid lengths, traversal, and outputs`](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs)
  — the sole normative paragraph to reconcile.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>
- [`runtime-prepare-engine.md` heading `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — Engine is the current composition root and generic/mixed composition remains planned.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Files and symbols

Exact implementation allowlist:

- `docs/architecture/contracts/recurrent-scan.md` — only the final exception-translation sentence
  under `Valid lengths, traversal, and outputs`.
- `docs/planning/modules/model/tasks/0025n-recurrent-engine-status-reconciliation.md` — status and
  compact result evidence.
- `docs/planning/modules/model/master-plan.md` — task row and frontier summary only.
- `docs/planning/roadmap.md` — [Model](../../../../glossary.md#term-model) area, repository frontier, and next step only.

Read-only evidence:

- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java` and
  `AdvancedEngine.java` — current public [facade](../../../../glossary.md#term-facade) and [lifecycle](../../../../glossary.md#lifecycle) composition.
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java`
  — current forward-only Compiler adoption.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanSemanticsTest.java`
  — current operation semantics.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Model master-plan recurrent-execution risk and focused public/[compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) recurrent sections.

No path outside the four-file implementation allowlist may change.

## Acceptance criteria

- The contract no longer calls the [Engine](../../../../glossary.md#term-engine) [facade](../../../../glossary.md#term-facade) future.
- It states that Engine owns any future public recurrent-execution exception translation and that
  the recurrent contract does not select the public exception type.
- It does not imply current recurrent backend execution, current exception translation, BPTT, or
  a new Engine API.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The rest of `Valid lengths, traversal, and outputs` remains byte-for-byte unchanged.
- Independent Class C review confirms wording against current authority/source/status and records
  reasoned no-change conclusions for root architecture, other scoped contracts, ADRs, API/guides,
  glossary, Javadoc, tests, Gradle, and executable behavior.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Only the four allowed paths change; brief stays at most 200 lines and 15 KB; Markdown links and
  anchors, headings, fences, whitespace, final newlines, exact scope, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

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

The Java/[Gradle](../../../../glossary.md#term-gradle) scan must be empty and the path audit must equal the four-file allowlist. If the
temporary Markdown validator is absent, create an equivalent validator outside the repository.
Review reuses successful focused-test evidence unless executable behavior changes and reruns final
documentation, size, [scope](../../../../glossary.md#term-scope), and whitespace checks.

Repository-wide validation is deferred because this task changes one status sentence and [planning](../../../../glossary.md#term-planning)
state, with no executable, build, [dependency](../../../../glossary.md#term-dependency), or architecture-boundary change.

## Dependencies and follow-up

- Frontier verification: user-authorized after clean HEAD `5b9ef70c`; Model 0025E–0025F,
  Compiler 0006A, Engine 0013, and the documentation-drift sequence through Prepare 0008 are
  Complete, so 0025N is the sole authorized <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare))</span>[`Ready`](../../../../glossary.md#term-ready) frontier.
- [Model](../../../../glossary.md#term-model) 0026 and all unrelated Draft, blocked, or review-needed work remain unauthorized.

## Documentation and review impact

- Types: normative architecture contract and planning state; apply General, Architecture, and
  Planning profiles.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [architecture contract glossary](../../../../glossary.md#architecture-contract); [planning glossary](../../../../glossary.md#term-planning))</span>
- A clean implementation context and independent targeted Class C review are mandatory.

## Result

- Replaced the stale final sentence under `Valid lengths, traversal, and outputs`: Engine is
  current, while public recurrent-execution exception translation remains future work and this
  contract selects no public exception type.
- Corrected this brief's stale heading label and anchor to the actual owning contract heading.
- Synchronized this brief, the [Model](../../../../glossary.md#term-model) master plan, and the roadmap at `Complete` after the
  mandatory independent Class C review.
- Changed only the four allowlisted Markdown paths; Java/[Gradle](../../../../glossary.md#term-gradle) scan was empty.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanSemanticsTest` passed.
- `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.RecurrentScanCompilerTest` passed.
- The independent Class C review confirmed current [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) composition from `Engine` and
  `AdvancedEngine`, Compiler forward-only adoption, absent recurrent backend capability, and the
  fail-closed BPTT boundary. The wording implies no current recurrent execution, translation, or
  new API and selects no public exception type. It reused the successful focused tests because no
  executable behavior changed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [backend capability glossary](../../../../glossary.md#backend-capability); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- The specified Markdown validator passed all four files; the brief remained within 200 lines and
  15 KB; exact-path audit and `git diff --check` passed.
- Documentation impact is limited to the normative status correction and synchronized planning
  evidence. Root architecture and other scoped contracts already assign lifecycle composition to
  Engine and execution to backends, so they need no edit. ADRs, API/guides, and the glossary
  already preserve the same current/future boundary and gain no decision, workflow, or term.
  Javadocs, tests, Gradle, and executable behavior need no change because this task changes no
  Java contract, build input, or behavior.<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative); [planning glossary](../../../../glossary.md#term-planning); [backends glossary](../../../../glossary.md#backend))</span>
- Repository-wide [validation](../../../../glossary.md#term-validation) remains deferred for the reason recorded above.

Status: Complete

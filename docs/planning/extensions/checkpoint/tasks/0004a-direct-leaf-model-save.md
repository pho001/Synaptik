# Task 0004A: Direct leaf-model save

## Status

Complete — implementation, independent review, and configured coordinator gate passed on 2026-10-14.

## Change class

Class C — conservatively treat exact native/heap state capture and the cross-owner recovery
fixture as a high-risk boundary. Use a clean executor and independent targeted review; preserve
existing ownership, execution, persistence, and numerical contracts without changing them.

## Goal

Make ordinary `ModelCheckpoint.save` work on a fully bound provenance-free leaf-state model,
including a trained live tree and a previously loaded tree, without caller expression wrappers.
Checkpoint selects raw-preserving outputs; existing public Engine compute materializes them.

## Scope

- In `ModelArtifactCapture.capture`, keep original NN descriptors, complete preflight, and exact
  source Tensor-identity deduplication in first-encounter order. After preflight, request one
  `.contiguous()` output per unique provenance-free leaf; retain existing expression outputs.
  Deduplicate by original Tensor identity, never by the fresh output wrapper or descriptor.
- Use exactly one existing `Engine.compute(List<Tensor>, uniqueBytes)` call for nonempty state;
  preserve original ordinal mapping and returned type/Shape/byte-length validation.
- Add direct-save and loaded-re-save coverage, remove `snapshotForCheckpoint` from the private
  recovery fixture, and update affected Javadocs and explanatory API text in the same change.

## Non-goals

No generic zero-node Engine/Prepare support, lower-layer production edits, manual storage reads
in capture, numeric identity arithmetic/casts, new public API, wire/schema/fingerprint changes,
new dependencies, storage ownership changes, source-alias metadata, lazy load, Training restore
facade, durable Training Checkpoint adapter, backend capability expansion, or unrelated refactor.
Do not edit `THIRD_PARTY_NOTICES.md` or the CPU clean Java oracle user modification.

## Contracts

Start at the root scope index; read these exact authoritative headings, not broad history:

- [Root — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Extensions/training — `extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint):
  once-only NN snapshot, preflight before Engine/payload allocation, identity order, bounded public
  materialization, detached candidate, strict target-defined alias installation, unchanged v1.
- [Foundational — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
  specifically [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)
  and [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor):
  immutable provenance/descriptors and exact raw movement, including `CONTIGUOUS`.
- [Runtime/Prepare/Engine — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine):
  existing public composition and lifecycle, no alternate execution pipeline.

The selected request expressions are an internal materialization detail, not replacements for NN
state or schema descriptors. If a contract conflict or unsupported required movement is found,
stop and report the exact heading/evidence; do not broaden Engine, Prepare, or backend scope.

## Dependencies and integration

- Depends on: Checkpoint 0004; CPU 0024; Training 0004 (completed fixture baseline).
- Conflicts with: None; no overlapping active task is authorized.
- Parallel group: None; serial implementation followed by independent review.
- Common base revision: `5d09f6c373d3200003bcc6efb08467a1c634da73` (main and origin/main).
- Integration order: coordinator transfers this planning packet → executor implementation →
  independent targeted review/documentation → coordinator integration and final gate.
- Integration validation: coordinator runs `./gradlew test javadoc --max-workers=2 --console=plain`
  once on the integrated change with the source-matched `SYNAPTIK_METAL_TEST_LIBRARY` configured,
  then bounded links/anchors and `git diff --check`; reuse worker evidence unless code changes.
- Shared-document integration owner: main coordinator for this brief, Checkpoint master, roadmap;
  executor/reviewer own the targeted explanatory/Javadoc changes serially in the executor worktree.

Frontier verification: Checkpoint 0004, CPU 0024, and Training 0004 each record final Complete
results, successful focused/module and combined gates, and independent review. Current source
contains the public save/load facade, restored-heap execution support, and private recovery test.
The user explicitly selected this Checkpoint-only repair and workaround removal. The reconciled
master/roadmap authorize only 0004A here; 0005–0007 remain Draft. Main and origin/main match the
base above; `/private/tmp/synaptik-checkpoint-leaf.oCW897` is clean at that base. Coordinator
commits/transfers planning edits before launch and rechecks base, status, and inactive conflicts.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/model/ModelArtifactCapture.java`
  — `capture`, original descriptor preflight, `unique`/`positions`/`ordinals`, compute outputs.
- Same package `ModelCheckpoint.java` — save Javadoc; no signature/load behavior changes.
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/model/ModelArtifactTest.java`
  and `ModelCheckpointTest.java` — raw-bit, alias, limit, direct save/load/re-save regressions.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java`
  — `Fixture.capture`, alternative artifact save, remove `snapshotForCheckpoint` and dead imports;
  direct trained save/load and loaded-tree re-save coverage, retaining recovery/fault assertions.
- `docs/api/model-checkpoint-api.md` — replace the save-time limitation with tested bounded behavior;
  targeted review of `docs/api/training-api.md` and `docs/glossary.md` only where affected.
- Read-only references: `Tensor.provenance`/`contiguous`, `Engine.compute`, model codec/restoration,
  relevant NN state APIs, existing raw heap execution tests. No added/moved production types.

## Acceptance criteria

- A fully bound direct leaf model saves without caller `.contiguous()` wrappers or a private view.
  Save → inspect → strict load → direct loaded-tree re-save → load preserves original ordered
  paths/kinds/types/Shapes, schema fingerprint, payload bits, and target-defined alias topology.
- Test all seven types: FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, INT64, BOOL, on both native
  and matching primitive-array heap leaf storage. Use exact canonical bytes, not tolerances:
  floating signed zeros, subnormals, infinities, and distinct NaN payload/sign patterns; low types
  retain all supplied 16 bits; signed integral extremes/patterns and canonical BOOL remain exact.
- Cover scalar and zero-extent leaf state, mixed leaf/expression state, and existing expression
  capture. Empty NN state still makes no Engine call, including with a closed Engine.
- Original identity dedup/order and once-only state export remain visible in implementation and
  focused evidence. Shared parameter/buffer and repeated buffer aliases retain distinct per-path
  payloads; distinct equal-descriptor Tensors are not collapsed. Capture never mutates bindings.
- Fingerprint, static Shape/checked length, entry, payload, aggregate (including every alias),
  manifest, and final-file limits reject before Engine or payload-sized allocation. Retain
  monotonic large-list preflight tests and closed-Engine rejection evidence for invalid leaf state.
- The private recovery fixture saves the actual quiescent trained tree through ordinary save,
  strict-loads and re-saves a loaded heap tree before any native parameter-copy step, and continues
  like the uninterrupted CPU session. Remove the workaround and all its call sites; retain
  canonical parameter comparison, pending accumulation/momentum, and failure-discard checks.
- Existing corruption, alias conflict, digest, unbound-target, no-partial-install, and publication
  tests stay green. No v1 envelope or metadata encoding, public API, or dependency changes.
- Required independent review finalizes accurate Javadocs, explains the internal raw movement
  without claiming generic leaf Engine support or durable Training recovery, and records evidence.

## Validation

Worker focused development checks, then one final affected-module run after executable edits:

```bash
./gradlew :extensions:checkpoint:test --tests '*ModelArtifactTest' --tests '*ModelCheckpointTest'
./gradlew :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest'
./gradlew :extensions:checkpoint:check :testing:integration-tests:check
```

After final Javadocs, generate `:extensions:checkpoint:javadoc` and
`:testing:integration-tests:javadoc`; validate owned Markdown links/anchors and `git diff --check`.
Independent review reuses successful executable evidence unless it changes code or identifies a
specific uncovered risk. Coordinator owns the one configured repository gate above; no backend
conformance or dependency-test edit is expected because those contracts remain unchanged.

## Documentation and review impact

Use `docs/developer-guide/documentation-rules.md`, general style, and API/Javadoc profile for the
API guide/Javadocs. Remove the current direct-save limitation only after regressions pass; explain
caller quiescence and retained bounds. Review changed implementation/test Javadocs, public save
Javadoc, Training API recovery text, and targeted glossary anchors for checkpoint/contiguous.
Record accurate unchanged-Javadoc/glossary conclusions where applicable. A separate clean
documentation/review context is mandatory for the user-visible fix and Class C capture boundary;
it reviews the actual diff and tests, not just the executor summary. No normative edits or ADR
are planned because the selected implementation preserves the current contracts.

## Result

- Capture now selects raw-preserving leaf outputs after full preflight, preserving original
  descriptors, Tensor-identity order, one bounded Engine call, bindings, and the v1 wire.
- Direct native/heap save/load/re-save covers all seven types and exact bits, scalar/empty state,
  mixed expressions, aliases, and rejection bounds. Private recovery saves the actual trained
  tree and re-saves loaded heap state before native copying; the expression workaround is removed.
- Changed code/tests: `ModelArtifactCapture`, `ModelCheckpoint`, `ModelArtifactTest`,
  `ModelCheckpointTest`, and `PrivateTrainingRestoreIntegrationTest` in the paths listed above.
- Worker commands in Validation passed: focused checkpoint tests, focused private recovery,
  affected-module checks, and affected Javadocs. Initial integration check failed only for missing
  native-library configuration; the configured integration check passed.
- Final coordinator command:
  `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
  passed in 6s (105 tasks: 10 executed, 58 from cache, 37 up-to-date); source-matched native unchanged.
  Current XML reports: checkpoint 29 tests/0 skipped/0 failures; integration 125/1 skipped/0 failures.
- Independent targeted review found no blocking findings and finalized affected Javadocs,
  `docs/api/model-checkpoint-api.md`, `docs/api/training-api.md`,
  `docs/architecture/training-graph.md`, and checkpoint glossary text; 317 links and diff checks passed.
- Closure updates only this brief, Checkpoint/Training masters, and roadmap; 117 planning links/anchors and
  `git diff --check` passed. No normative contract, ADR, dependency, or public API change.
- No unresolved acceptance issues or required follow-up. Generic zero-node Engine support and
  durable Training recovery remain outside scope; 0005–0007 stay Draft pending a separate plan.
- Status: Complete

# Task 0004A: Direct [leaf](../../../../glossary.md#term-leaf)-[model](../../../../glossary.md#term-model) save

## Status

Complete — implementation, independent review, and configured coordinator gate passed on 2026-10-14.

## Change class

Class C — conservatively treat exact native/[heap](../../../../glossary.md#term-heap) state [capture](../../../../glossary.md#term-capture-meanings) and the cross-[owner](../../../../glossary.md#term-owner) recovery
[fixture](../../../../glossary.md#term-fixture) as a high-risk boundary. Use a clean executor and independent targeted review; preserve
existing [ownership](../../../../glossary.md#term-ownership-meanings), execution, [persistence](../../../../glossary.md#term-persistence), and numerical contracts without changing them.

## Goal

Make ordinary `ModelCheckpoint.save` work on a fully bound [provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf)-state [model](../../../../glossary.md#term-model),
including a trained live tree and a previously loaded tree, without caller expression wrappers.
[Checkpoint](../../../../glossary.md#term-checkpoint) selects raw-preserving outputs; existing public [Engine](../../../../glossary.md#term-engine) compute materializes them.

## [Scope](../../../../glossary.md#term-scope)

- In `ModelArtifactCapture.capture`, keep original NN descriptors, complete preflight, and exact
  source Tensor-identity deduplication in first-encounter order. After preflight, request one<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [descriptors glossary](../../../../glossary.md#term-descriptor); [preflight glossary](../../../../glossary.md#term-preflight); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
  `.contiguous()` output per unique provenance-free leaf; retain existing expression outputs.
  Deduplicate by original Tensor identity, never by the fresh output wrapper or descriptor.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Use exactly one existing `Engine.compute(List<Tensor>, uniqueBytes)` call for nonempty state;
  preserve original ordinal mapping and returned type/Shape/byte-length validation.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Add direct-save and loaded-re-save coverage, remove `snapshotForCheckpoint` from the private
  recovery fixture, and update affected Javadocs and explanatory API text in the same change.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

No generic zero-[node](../../../../glossary.md#node) [Engine](../../../../glossary.md#term-engine)/[Prepare](../../../../glossary.md#prepare) support, lower-layer production edits, manual storage reads
in [capture](../../../../glossary.md#term-capture-meanings), numeric [identity](../../../../glossary.md#term-identity-meanings) arithmetic/casts, new public [API](../../../../glossary.md#term-application-programming-interface-api), wire/[schema](../../../../glossary.md#term-schema)/[fingerprint](../../../../glossary.md#term-fingerprint) changes,
new dependencies, storage [ownership](../../../../glossary.md#term-ownership-meanings) changes, source-[alias](../../../../glossary.md#aliasing-and-arena) [metadata](../../../../glossary.md#term-metadata), [lazy](../../../../glossary.md#term-lazy-evaluation) load, Training restore
[facade](../../../../glossary.md#term-facade), durable Training [Checkpoint](../../../../glossary.md#term-checkpoint) adapter, [backend capability](../../../../glossary.md#backend-capability) expansion, or unrelated refactor.
Do not edit `THIRD_PARTY_NOTICES.md` or the [CPU](../../../../glossary.md#term-cpu) clean Java [oracle](../../../../glossary.md#term-test-oracle) user modification.

## Contracts

Start at the root [scope](../../../../glossary.md#term-scope) index; read these exact authoritative headings, not broad history:

- [Root — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>, and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Extensions/training — `extensions/checkpoint`](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint):
  once-only NN snapshot, preflight before Engine/payload allocation, identity order, bounded public
  materialization, detached candidate, strict target-defined alias installation, unchanged v1.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [preflight glossary](../../../../glossary.md#term-preflight); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [detached glossary](../../../../glossary.md#detached-model-results); [target glossary](../../../../glossary.md#term-target); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- [Foundational — `modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>,
  specifically [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
  and [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor):
  immutable provenance/descriptors and exact raw movement, including <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous).
- [Runtime/Prepare/Engine — `modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>:
  existing public composition and lifecycle, no alternate execution pipeline.

The selected request expressions are an internal [materialization](../../../../glossary.md#materialization) detail, not replacements for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
state or [schema](../../../../glossary.md#term-schema) [descriptors](../../../../glossary.md#term-descriptor). If a contract conflict or unsupported required movement is found,
stop and report the exact heading/evidence; do not broaden [Engine](../../../../glossary.md#term-engine), [Prepare](../../../../glossary.md#prepare), or [backend](../../../../glossary.md#backend) scope.

## Dependencies and integration

- Depends on: [Checkpoint](../../../../glossary.md#term-checkpoint) 0004; [CPU](../../../../glossary.md#term-cpu) 0024; Training 0004 (completed [fixture](../../../../glossary.md#term-fixture) baseline).
- Conflicts with: None; no overlapping active task is authorized.
- Parallel group: None; serial implementation followed by independent review.
- Common base revision: `5d09f6c373d3200003bcc6efb08467a1c634da73` (main and origin/main).
- Integration order: coordinator transfers this planning packet → executor implementation →
  independent targeted review/documentation → coordinator integration and final gate.
- Integration [validation](../../../../glossary.md#term-validation): coordinator runs `./gradlew test javadoc --max-workers=2 --console=plain`
  once on the integrated change with the source-matched `SYNAPTIK_METAL_TEST_LIBRARY` configured,
  then bounded links/anchors and `git diff --check`; reuse worker evidence unless code changes.
- Shared-document integration owner: main coordinator for this brief, Checkpoint master, roadmap;
  executor/reviewer own the targeted explanatory/Javadoc changes serially in the executor worktree.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Frontier verification: Checkpoint 0004, CPU 0024, and Training 0004 each record final Complete
results, successful focused/[module](../../../../glossary.md#term-module) and combined gates, and independent review. Current source
contains the public save/load [facade](../../../../glossary.md#term-facade), restored-[heap](../../../../glossary.md#term-heap) execution support, and private recovery test.
The user explicitly selected this Checkpoint-only repair and workaround removal. The reconciled
master/roadmap authorize only 0004A here; 0005–0007 remain Draft. Main and origin/main match the
base above; `/private/tmp/synaptik-checkpoint-leaf.oCW897` is clean at that base. Coordinator
[commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/transfers [planning](../../../../glossary.md#term-planning) edits before launch and rechecks base, status, and inactive conflicts.

## Files and symbols

- `extensions/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/model/ModelArtifactCapture.java`
  — [`capture`](../../../../glossary.md#term-capture-meanings), original [descriptor](../../../../glossary.md#term-descriptor) [preflight](../../../../glossary.md#term-preflight), `unique`/`positions`/`ordinals`, compute outputs.
- Same package `ModelCheckpoint.java` — save [Javadoc](../../../../glossary.md#term-javadoc); no [signature](../../../../glossary.md#term-signature)/load behavior changes.
- `extensions/checkpoint/src/test/java/io/github/pho001/synaptik/checkpoint/model/ModelArtifactTest.java`
  and `ModelCheckpointTest.java` — raw-bit, [alias](../../../../glossary.md#aliasing-and-arena), limit, direct save/load/re-save regressions.
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java`
  — `Fixture.capture`, alternative [artifact](../../../../glossary.md#term-artifact) save, remove `snapshotForCheckpoint` and dead imports;
  direct trained save/load and loaded-tree re-save coverage, retaining recovery/fault assertions.
- `docs/api/model-checkpoint-api.md` — replace the save-time limitation with tested bounded behavior;
  targeted review of `docs/api/training-api.md` and `docs/glossary.md` only where affected.
- Read-only references: `Tensor.provenance`/[`contiguous`](../../../../glossary.md#term-contiguous), `Engine.compute`, model codec/restoration,
  relevant NN state APIs, existing raw heap execution tests. No added/moved production types.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [codec glossary](../../../../glossary.md#term-codec); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [heap glossary](../../../../glossary.md#term-heap))</span>

## Acceptance criteria

- A fully bound direct [leaf](../../../../glossary.md#term-leaf) [model](../../../../glossary.md#term-model) saves without caller `.contiguous()` wrappers or a private view.
  Save → inspect → strict load → direct loaded-tree re-save → load preserves original ordered
  paths/kinds/types/Shapes, schema fingerprint, payload bits, and target-defined alias topology.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [Shapes glossary](../../../../glossary.md#shape); [schema glossary](../../../../glossary.md#term-schema); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [target glossary](../../../../glossary.md#term-target); [alias glossary](../../../../glossary.md#aliasing-and-arena); [topology glossary](../../../../glossary.md#term-topology))</span>
- Test all seven types: FLOAT64, FLOAT32, BFLOAT16, FLOAT16, INT32, INT64, BOOL, on both native
  and matching primitive-array heap leaf storage. Use exact canonical bytes, not tolerances:
  floating signed zeros, subnormals, infinities, and distinct NaN payload/sign patterns; low types
  retain all supplied 16 bits; signed integral extremes/patterns and canonical BOOL remain exact.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [heap glossary](../../../../glossary.md#term-heap); [canonical glossary](../../../../glossary.md#term-canonical-form); [subnormals glossary](../../../../glossary.md#term-subnormal); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Cover scalar and zero-extent leaf state, mixed leaf/expression state, and existing expression
  capture. Empty NN state still makes no Engine call, including with a closed Engine.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-structural-capture))</span><span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Original identity dedup/order and once-only state export remain visible in implementation and
  focused evidence. Shared parameter/buffer and repeated buffer aliases retain distinct per-path
  payloads; distinct equal-descriptor Tensors are not collapsed. Capture never mutates bindings.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [buffer glossary](../../../../glossary.md#term-buffer); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensors glossary](../../../../glossary.md#tensor); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Fingerprint, static Shape/checked length, entry, payload, aggregate (including every alias),
  manifest, and final-file limits reject before Engine or payload-sized allocation. Retain
  monotonic large-list preflight tests and closed-Engine rejection evidence for invalid leaf state.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- The private recovery fixture saves the actual quiescent trained tree through ordinary save,
  strict-loads and re-saves a loaded heap tree before any native parameter-copy step, and continues
  like the uninterrupted CPU session. Remove the workaround and all its call sites; retain
  canonical parameter comparison, pending accumulation/momentum, and failure-discard checks.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [quiescent glossary](../../../../glossary.md#term-quiescent); [CPU glossary](../../../../glossary.md#term-cpu); [momentum glossary](../../../../glossary.md#term-momentum))</span>
- Existing corruption, alias conflict, digest, unbound-target, no-partial-install, and publication
  tests stay green. No v1 envelope or metadata encoding, public API, or dependency changes.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [publication glossary](../../../../glossary.md#term-publication); [metadata glossary](../../../../glossary.md#term-metadata); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Required independent review finalizes accurate Javadocs, explains the internal raw movement
  without claiming generic leaf Engine support or durable Training recovery, and records evidence.

## [Validation](../../../../glossary.md#term-validation)

Worker focused development checks, then one final affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run) after executable edits:

```bash
./gradlew :extensions:checkpoint:test --tests '*ModelArtifactTest' --tests '*ModelCheckpointTest'
./gradlew :testing:integration-tests:test --tests '*PrivateTrainingRestoreIntegrationTest'
./gradlew :extensions:checkpoint:check :testing:integration-tests:check
```

After final Javadocs, generate `:extensions:checkpoint:javadoc` and
`:testing:integration-tests:javadoc`; validate owned Markdown links/anchors and `git diff --check`.
Independent review reuses successful executable evidence unless it changes code or identifies a
specific uncovered risk. Coordinator owns the one configured repository gate above; no [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance) or [dependency](../../../../glossary.md#term-dependency)-test edit is expected because those contracts remain unchanged.

## Documentation and review impact

Use `docs/developer-guide/documentation-rules.md`, general style, and [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) profile for the
API guide/Javadocs. Remove the current direct-save limitation only after regressions pass; explain
caller [quiescence](../../../../glossary.md#term-quiescence) and retained bounds. Review changed implementation/test Javadocs, public save
Javadoc, Training API recovery text, and targeted glossary anchors for [checkpoint](../../../../glossary.md#term-checkpoint)/[contiguous](../../../../glossary.md#term-contiguous).
Record accurate unchanged-Javadoc/glossary conclusions where applicable. A separate clean
documentation/review context is mandatory for the user-visible fix and Class C [capture](../../../../glossary.md#term-capture-meanings) boundary;
it reviews the actual diff and tests, not just the executor summary. No [normative](../../../../glossary.md#term-normative) edits or [ADR](../../../../glossary.md#term-architecture-decision-record-adr)
are planned because the selected implementation preserves the current contracts.

## Result

- Capture now selects raw-preserving leaf outputs after full preflight, preserving original
  descriptors, Tensor-identity order, one bounded Engine call, bindings, and the v1 wire.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [leaf glossary](../../../../glossary.md#term-leaf); [preflight glossary](../../../../glossary.md#term-preflight); [descriptors glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [Engine glossary](../../../../glossary.md#term-engine); [bindings glossary](../../../../glossary.md#term-binding))</span>
- Direct native/heap save/load/re-save covers all seven types and exact bits, scalar/empty state,
  mixed expressions, aliases, and rejection bounds. Private recovery saves the actual trained
  tree and re-saves loaded heap state before native copying; the expression workaround is removed.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Changed code/tests: `ModelArtifactCapture`, `ModelCheckpoint`, `ModelArtifactTest`,
  `ModelCheckpointTest`, and `PrivateTrainingRestoreIntegrationTest` in the paths listed above.
- Worker commands in Validation passed: focused checkpoint tests, focused private recovery,
  affected-module checks, and affected Javadocs. Initial integration check failed only for missing
  native-library configuration; the configured integration check passed.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module))</span>
- Final coordinator command:
  `SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc --max-workers=2 --console=plain`
  passed in 6s (105 tasks: 10 executed, 58 from cache, 37 up-to-date); source-matched native unchanged.
  Current XML reports: checkpoint 29 tests/0 skipped/0 failures; integration 125/1 skipped/0 failures.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Independent targeted review found no blocking findings and finalized affected Javadocs,
  `docs/api/model-checkpoint-api.md`, `docs/api/training-api.md`,
  `docs/architecture/training-graph.md`, and [checkpoint](../../../../glossary.md#term-checkpoint) glossary text; 317 links and diff checks passed.
- Closure updates only this brief, Checkpoint/Training masters, and roadmap; 117 [planning](../../../../glossary.md#term-planning) links/anchors and
  `git diff --check` passed. No [normative](../../../../glossary.md#term-normative) contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), or public [API](../../../../glossary.md#term-application-programming-interface-api) change.
- No unresolved acceptance issues or required follow-up. Generic zero-node Engine support and
  durable Training recovery remain outside scope; 0005–0007 stay Draft pending a separate plan.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [scope glossary](../../../../glossary.md#term-scope))</span>
- Status: Complete

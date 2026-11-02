# Task Compiler0012: General-scan cold source reconstruction

## Status

Complete — main accepted implementation, independent review and final gates on 2026-10-23 (~05:42 UTC).
User explicitly approved NN0030's limited existing-owner direction and first Compiler step
(“ok, pust se do toho”, recorded 2026-10-23). Prior NN0030 approval covered Draft planning only.
Lovelace's binding pre-code APPROVE closes both P2 findings, zero remaining blockers:
`/private/tmp/synaptik-compiler0012-review.yky0FE/final-review.md` (correction-only follow-up).
Main accepted frozen diff SHA-256 `3bb992423f23aad218c3cb4e1c8767a3a9e2a7c4678b413f7d2471d921bb8825`.
Main reverified 2026-10-23 approximately 05:18 UTC: HEAD=origin/main at the base below, empty index,
exact 259-path accepted frozen base preserved with no unrelated hash changes; Complete predecessors,
current masters and accepted reviews/gates verified; limited user approval recorded, no active
conflicts/source writer, sole plan writer, Parallel group None. This records historical Ready
authorization before one clean Compiler executor launched; Compiler0012 was the sole frontier.
Current Compiler/NN/Checkpoint frontiers: None; no successor authorized.

## Change class

Class C — scoped reconstruction, constructor/preflight and cross-module consumer boundary.
Clean implementation thread and independent targeted code/documentation review are mandatory.

## Goal and scope

Extend existing GraphCompilationPort.captureSource/reconstructSource to reconstruct bounded
general SCAN with ordinary public Tensor/Scan constructors, no storage or execution. Keep the
flat outer-source inventory unchanged except general SCAN; body-only mappings must not leak out.
Model PURE validity/dtypes, cold constructor support, AD/execution and persistence are separate.
Low TANH and F32 RELU must round-trip cold without provider support. No all-PURE copying promise.

Production scope: existing Compiler owners/private helpers. No added/moved production type or API.
STOP before any new class/type: give exact name/package, missing responsibility, why existing
owners/private helpers cannot suffice, alternatives and ownership/duplication cost. Avoid god classes.
One downstream test-only exception: GraphCheckpointTest pins safe consumer rejection; no Checkpoint
or NN production changes. Body attachment preservation is the later serial Checkpoint capability.

## Non-goals

No T-unroll, new executor/session/DAG/AD layer/dependency, padding/row skipping, fixed BPTT or
builder redirection, body wire/save/load/version, low/vector/parallel/OpenBLAS execution widening,
runtime layout/backend filter, numerical-policy change, native build or performance campaign.

## Contracts

Read AGENTS.md and the affected source/tests; use the root scope index, not predecessor histories.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules` — unchanged edges/owners.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle): `Compile lifecycle`,
  `Bounded general-scan cold reconstruction` — exact flat/body inventories and complete preflight.
- [Scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Traversal, empty domains and declared qualification`, `Qualification, performance and migration gates`.
- [Consumers](../../../../architecture/contracts/extensions-training.md#owned-tensor-models):
  `Owned tensor Models`, `Generic graph source and values`, `Canonical graph-model version 3`.
- [ADR0046](../../../../design/decisions/0046-general-scan-cold-source-and-owned-model-direction.md)
  explains the approved direction, not implementation completion. STOP on contract ambiguity.

## Dependencies and integration

- Depends on: NN0030 approved decision; Compiler0011, Model0034/0035, Planning0008, Prepare0010,
  CPU0030–0033 and low-level join; Compiler0008/0009, Checkpoint0008–0011, NN0028/0029,
  NN0022B and Training0007/0008 — Complete baseline, verified from current masters/results and
  main's accepted baseline/authority/consumer review; Lovelace final pre-code APPROVE is recorded above.
- Conflicts with: cold capture/reconstruction, scoped scan semantics, Checkpoint normalization/wire,
  NN acquisition/lifecycle and shared authority/docs writers; serialize, no concurrent writer.
- Parallel group: None.
- Common base revision: HEAD=main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the exact 259-path accepted frozen base (257 prior paths plus two new plans/ADR); index empty,
  preserve all. Only verified Ready/frontier status metadata changes after the accepted diff above.
- Integration order: independent pre-code APPROVE → main Ready verification → clean Compiler
  implementation → independent code/docs APPROVE → main consumer/architecture checkpoint →
  later separately briefed Checkpoint copy/rebind → NN workflow qualification → final program join.
- Integration validation: main's consumer/architecture checkpoint below; no CPU full/native gate here.
- Shared-document integration owner: main, exclusive shared planning/authority owner.

## Files and symbols

Paths use `io/github/pho001/synaptik/` beneath each Java root.

- `modules/compiler/src/main/java/.../compiler/SourceGraphReconstruction.java`: preflight,
  additional, validateLeaves, reconstruct; scope-aware private helpers, not a new reconstruction layer.
- Same package: GraphCompilationPort.SourceCapture/cold Javadoc, GraphCapture.captureSource/captureBody,
  ScanBodyInference.capture/infer/validateResiduals and CapturedGraphInference.inferAndValidate.
- Compiler tests: SourceGraphCaptureTest, SourceGraphReconstructionTest, GeneralScanCompilerTest,
  ScanBodyInferenceTest; update old GENERAL cold rejection assertions only, retain AD exclusions.
- Read Model tensor ScanBody/Scan/ScanResult, Tensor and constructor helpers/attrs used below;
  graph CompiledNode/CompiledScanBody/CompiledGraphModel. No Model production changes.
- `modules/checkpoint/src/test/java/.../checkpoint/graph/GraphCheckpointTest.java`: test-only consumer
  rejection. Read GraphSource.capture/copy, GraphArtifactCodec.encode/opcode and GraphCheckpoint.
- Read NN OwnedModelSource acquisition/forward and ModelExecutionState.rejectScanTraining;
  fixed RecurrentScanKind containsScan/rejectScanTraining stays unchanged.

## Constructor-backed inventory

The owning Compiler contract is exhaustive. Current public constructors/attribute classes were
inspected, not inferred from CPU support. All cases preserve raw full descriptors and Model guards.

| Scope/signature | Reconstruction and guard |
|---|---|
| Flat outer source | Existing closed mappings unchanged; add general SCAN with body only. In particular flat DIV/scalar/SUM/EXPAND/CONTIGUOUS/CAST still reject. |
| Body reused mappings | Existing flat mappings except fixed/nested scans; same guards unless overridden below. MATMUL retains existing vector/batched/promotion support; rank-two execution is not a cold restriction. PERMUTE remains rank-two `[1,0]`. |
| Body tensor ADD/SUB/MUL/DIV | Two inputs, NoOperationAttrs; add/sub/mul/div(Tensor). Numeric ADD/SUB/MUL (including integral body SUB), floating DIV, Model promotion/broadcast. |
| Body scalar ADD/SUB/MUL/DIV | One input, ScalarValueAttrs; typed ScalarValue overload, exact dtype/value. Numeric first three, floating DIV; no double conversion. |
| Body NEG/RELU/TANH/SIGMOID | One input, NoOperationAttrs; existing floating constructor. |
| Body SUM | One numeric input; NoOperationAttrs→sum(); AxisReductionAttrs→sum(axis,keep); MultiAxisReductionAttrs→sum(int[],keep); SumToShapeAttrs→sumToShape(Shape). Preserve normalized axis order/empty axes/keep; no masked form. |
| Body RESHAPE/EXPAND | TargetShapeAttrs, Shape overload; normalized static target, equal checked counts / exact broadcast. Reshape scalar/zero extents allowed; do not reinterpret a stored target as raw `-1` inference. |
| Body EXPAND_DIMS/SQUEEZE | AxisTransformAttrs; any valid normalized axis, singleton squeeze. |
| Body CONTIGUOUS / identity CAST | NoOperationAttrs→contiguous(); CastAttrs with target=input dtype→cast(dtype). Both preserve fresh producer semantics; no storage copy; CAST does not alias input. |
| Constants | Explicit outer source input/invariant positions or scalar attrs. No free body leaf, hidden constant producer, host-backed formal or new sidecar. |

Other PURE families may be valid Model sources but have no selected cold mapping here. Existing
flat mappings are reused, not generalized (e.g. convolution geometry, GATHER/SLICE/CONCAT guards).
Cross-dtype CAST and arbitrary permutations are outside the selected closure, not invalid Model.

## Acceptance criteria

1. Whole outer graph plus every body/value/phase passes static descriptor/count/layout arithmetic,
   exact raw inference, no deferred obligations, formal/header/role/residual checks before **any**
   formal/producer allocation. Include late/dead outer scans, T=0, invalid body nodes, unused values,
   overflow, missing/malformed attachments and scope errors. Pin no Tensor-ID consumption on known
   preflight failures. No physical resource/byte/alignment limits; those belong to Checkpoint/backend.
   Cold bodies require empty demandedOutputIndices and body.graph.outputs EXACTLY the first-encounter
   unique concatenation of nextCarry + emissions + residuals, in that order. Explicit no-ID-consumption
   tests reject nonempty demand and extra/reordered/missing role outputs before ANY formal/producer,
   including late/dead outer scans and T=0. Keep these checks cold-only, not ordinary compilation.
2. Reconstruct one fresh distinct canonical formal per declared slot, including unused formals;
   body nodes once, one ScanBody and Scan.apply per occurrence. Map all canonical ports, including
   latent/unrequested residuals, repeated nextCarry/emission roles and shared uses. Preserve both
   traversal flags, repeated actual positions, multiple occurrences/template reuse and independent
   local IDs even when numerically equal to outer IDs; retain no old formals/templates or actuals
   except the explicitly supplied outer substitutions.
3. Outer-only substitutions accept full-descriptor-compatible produced/repeated Tensors with opaque
   upstream provenance. Keep ordinary provenance for later compile/AD. Scope tables are call-local,
   discarded on failure/return; no persistent map or ValueId-to-Tensor AD conversion.
4. Capture/reconstruct/recapture agree on raw graph/descriptors/roles/ordered roots for every inventory
   row and applicable Model dtype. Include valid low TANH, F32 RELU, broadcast/reduction/layout/identity
   cast, rank-two MATMUL/transpose, explicit constants, T=0/empties and differing static T with O(body)
   object counts. Preserve old flat rejection and unresolved MATMUL/CONCAT/view round trips.
   Include positive repeated-role cases proving first-encounter-unique body output order and empty
   raw demand survive round trips; ordinary compiled attachments may still carry nonempty demand.
5. Compiler-only stage leaves Checkpoint/NN fail-closed. Add focused GraphCheckpointTest cases for
   GENERAL copy/rebind/save (both directions, final/residual-root coverage), with otherwise valid named
   state and source boundaries. Existing four-arg CompiledNode must reject missing body during
   GraphSource normalization, before it returns: no silent lossy result. A storage spy must observe
   zero host segment reads; dead/absent state must not win error precedence. Existing target bytes
   and directory entries stay unchanged, absent target remains absent. Rebind must not reach replacement
   admission. No production consumer guard needed or authorized. Future Checkpoint task replaces this
   temporary rejection with attachment-preserving copy/rebind and independent wire rejection.
6. Existing fixed scan source/ports, ordinary Compiler AD and fixed NN early training rejection remain
   unchanged. No CPU/backend qualification or file-byte compatibility claim is widened.

## Validation

Worker focused development (extend existing named tests rather than add a production type):

```bash
./gradlew :modules:compiler:test --tests '*SourceGraphCaptureTest' --tests '*SourceGraphReconstructionTest' --tests '*GeneralScanCompilerTest' --tests '*ScanBodyInferenceTest'
./gradlew :extensions:checkpoint:test --tests '*GraphCheckpointTest'
```

After final executable/Javadoc edits, one module gate:

```bash
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

Main after independent implementation review runs one consumer/architecture checkpoint:

```bash
./gradlew :extensions:checkpoint:test
./gradlew :extensions:nn:test --tests '*OwnedModelSourceTest' --tests '*OwnedModelTrainingTest' --tests '*RuntimeLengthScanModelTest'
./gradlew :testing:architecture-tests:test
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
```

Reuse focused successful worker evidence; no full CPU/native test or benchmark now. One source-matched
native-enabled root `test javadoc` belongs to the FINAL future NN program join, not every prerequisite;
no dependency/backend/execution change justifies repeating that root gate for Compiler-only source work.

## Documentation and review impact

Authority cutover precedes Java: root plus Compiler/scan/extension owners, ADR0046 and focused
module-boundaries/training-graph/general-differentiable-scan explanations and glossary are staged.
Update GraphCompilationPort/SourceGraphReconstruction Javadoc and affected helper contracts in the
implementation; update docs/api/compile-api.md's cold boundary; review docs/api/graph-checkpoint-api.md
and user-guide/running-models.md without
claiming Compiler-only work delivers Checkpoint/NN support. No user workflow example until its gate.
Independent pre-code architecture/inventory review and later targeted code/docs review are mandatory,
in clean threads; review actual source/tests and task-only diff, reuse tests unless code changes.

## Result

Status: Complete — main accepted the unchanged reviewed source and all recorded gates, 2026-10-23.

- Implemented bounded GENERAL cold reconstruction in existing SourceGraphReconstruction only; no new production type/API/dependency, AD/execution or backend capability.
- Seven changed files: SourceGraphReconstruction.java (executable), GraphCompilationPort.java (Javadoc), SourceGraphReconstructionTest.java, GeneralScanCompilerTest.java, GraphCheckpointTest.java, docs/api/compile-api.md and docs/api/graph-checkpoint-api.md.
- Approved baseline/authority preparation and historical Ready verification above preceded implementation. Final independent code/docs APPROVE, zero P1/P2, no reviewer edits: `/private/tmp/synaptik-compiler0012-final-review.d6gsha/final-review.md`.
- Frozen task-only diff SHA-256 `ee968ff19f0c210367c8c1b1c27ce7ce27f077f8abce7ffc6d7fe49453c8db68`; all seven reviewed hashes unchanged through gates.
- Recorded focused commands: Compiler 32 + Checkpoint 10 = 42 passing tests, subsets of final gates, not additional tests.
- `./gradlew :modules:compiler:test :modules:compiler:javadoc` ran once after review: PASS, 46 suites/347 tests; Javadoc zero warnings.
- `./gradlew :extensions:checkpoint:test :extensions:nn:test --tests '*OwnedModelSourceTest' --tests '*OwnedModelTrainingTest' --tests '*RuntimeLengthScanModelTest' :testing:architecture-tests:test`: PASS; Checkpoint 6 suites/47 tests, NN 3/31, architecture 11/25.
- `./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'`: PASS, 1 suite/17 tests. Final unique gates total 467 tests, zero failures/errors/skips.
- Worker logs/results: `/private/tmp/synaptik-compiler0012-impl.wnir9h/` compiler-focused-4.log, checkpoint-focused-3.log, compiler-module-gate.log, module-gate-result.md and module-gate-counts.log; main logs: `/private/tmp/synaptik-compiler0012-coordination.8rCsJ1/` consumer-checkpoint.log and engine-checkpoint.log.
- Independent Javadoc/API/explanation review finalized the drafts; helper contracts and targeted glossary remained accurate, no new terminology/example or further edits. Links/anchors and whitespace passed.
- Main verified unchanged 263-path source snapshot before/after gates, HEAD=origin/main at the recorded base, empty index and preserved unrelated dirty baseline. Existing FFM/incubator consumer warnings only; no failure.
- No unresolved acceptance issue. Checkpoint/NN remain fail-closed; attachment-preserving Checkpoint copy/rebind is next, separately briefed, then NN qualification. No new Ready brief or successor launch.
- No native build, performance campaign or root rerun; one source-matched native root test/Javadoc remains at the final future NN join.
- Size guardrail exception: retaining this cohesive Class C acceptance/Ready packet plus final evidence slightly exceeds 200 lines/15 KB; no accepted scope was rewritten.

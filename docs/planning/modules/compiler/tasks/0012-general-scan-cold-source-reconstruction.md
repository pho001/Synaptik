# Task Compiler0012: General-[scan](../../../../glossary.md#term-cumulative-scan) cold source reconstruction

> **Editorial reading key — not part of the historical record.**
>
> Cold reconstruction rebuilds expressions through selected public constructors; it is not the
> same admission gate as general-Scan inference or differentiation. The caller supplies positional
> outer substitutions, while each rebuilt body gets fresh local formals and a call-local identity
> table. Body-only constructors do not automatically become legal flat source constructors.
> Borrowed leaves carry no transferred storage ownership, and successful reconstruction does not
> establish Checkpoint wire support or NN execution.
>
> Current reading: [raw source capture and positional reconstruction](../../../../reference/compiler/cold-source.md#current-cold-source-capture-and-reconstruction); [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad).

## Status

Complete — main accepted implementation, independent review and final gates on 2026-10-23 (~05:42 UTC).
User explicitly approved NN0030's limited existing-[owner](../../../../glossary.md#term-owner) direction and first [Compiler](../../../../glossary.md#term-compiler) step
(“ok, pust se do toho”, recorded 2026-10-23). Prior NN0030 approval covered Draft [planning](../../../../glossary.md#term-planning) only.
Lovelace's [binding](../../../../glossary.md#term-binding) pre-code APPROVE closes both P2 findings, zero remaining blockers:
`/private/tmp/synaptik-compiler0012-review.yky0FE/final-review.md` (correction-only follow-up).
Main accepted frozen diff [SHA-256](../../../../glossary.md#term-sha-256) `3bb992423f23aad218c3cb4e1c8767a3a9e2a7c4678b413f7d2471d921bb8825`.
Main reverified 2026-10-23 approximately 05:18 UTC: HEAD=origin/main at the base below, empty index,
exact 259-path accepted frozen base preserved with no unrelated [hash](../../../../glossary.md#term-hash) changes; Complete predecessors,
current masters and accepted reviews/gates verified; limited user approval recorded, no active
conflicts/source writer, sole plan writer, Parallel group None. This records historical [Ready](../../../../glossary.md#term-ready)
authorization before one clean Compiler executor launched; Compiler0012 was the sole frontier.
Current Compiler/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Checkpoint](../../../../glossary.md#term-checkpoint) frontiers: None; no successor authorized.

## Change class

Class C — scoped reconstruction, constructor/[preflight](../../../../glossary.md#term-preflight) and cross-[module](../../../../glossary.md#term-module) consumer boundary.
Clean implementation thread and independent targeted code/documentation review are mandatory.

## Goal and [scope](../../../../glossary.md#term-scope)

Extend existing GraphCompilationPort.captureSource/reconstructSource to reconstruct bounded
[general SCAN](../../../../glossary.md#general-differentiable-scan) with ordinary public [Tensor](../../../../glossary.md#tensor)/[Scan](../../../../glossary.md#term-cumulative-scan) constructors, no storage or execution. Keep the
flat outer-source inventory unchanged except general SCAN; body-only mappings must not leak out.
[Model](../../../../glossary.md#term-model) [PURE](../../../../glossary.md#term-pure) validity/[dtypes](../../../../glossary.md#data-type--datatype), cold constructor support, [AD](../../../../glossary.md#autograd)/execution and [persistence](../../../../glossary.md#term-persistence) are separate.
Low [TANH](../../../../glossary.md#term-tanh) and [F32](../../../../glossary.md#floating-point-precision-and-rounding) RELU must [round-trip](../../../../glossary.md#schema-serialization-and-checkpoint) cold without provider support. No all-PURE copying promise.

Production scope: existing [Compiler](../../../../glossary.md#term-compiler) [owners](../../../../glossary.md#term-owner)/private helpers. No added/moved production type or [API](../../../../glossary.md#term-application-programming-interface-api).
STOP before any new class/type: give exact name/package, missing responsibility, why existing
owners/private helpers cannot suffice, alternatives and [ownership](../../../../glossary.md#term-ownership-meanings)/duplication cost. Avoid god classes.
One downstream test-only exception: GraphCheckpointTest pins safe consumer rejection; no [Checkpoint](../../../../glossary.md#term-checkpoint)
or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) production changes. Body attachment preservation is the later serial Checkpoint capability.

## Non-goals

No T-unroll, new executor/session/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag)/[AD](../../../../glossary.md#autograd) layer/[dependency](../../../../glossary.md#term-dependency), [padding](../../../../glossary.md#padding)/row skipping, fixed [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) or
builder redirection, body wire/save/load/version, low/[vector](../../../../glossary.md#term-vector)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas) execution [widening](../../../../glossary.md#term-widening),
[runtime](../../../../glossary.md#term-runtime) [layout](../../../../glossary.md#layout)/[backend](../../../../glossary.md#backend) filter, numerical-policy change, native build or performance campaign.

## Contracts

Read AGENTS.md and the affected source/tests; use the root [scope](../../../../glossary.md#term-scope) index, not predecessor histories.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules` — unchanged edges/[owners](../../../../glossary.md#term-owner).
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>: `Compile lifecycle`,
  `Bounded general-scan cold reconstruction` — exact flat/body inventories and complete [preflight](../../../../glossary.md#term-preflight).
- [Scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>:
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
  NN acquisition/lifecycle and shared authority/docs writers; serialize, no concurrent writer.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None.
- Common base revision: HEAD=main=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus the exact 259-path accepted frozen base (257 prior paths plus two new plans/ADR); index empty,
  preserve all. Only verified Ready/frontier status metadata changes after the accepted diff above.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Ready glossary](../../../../glossary.md#term-ready); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Integration order: independent pre-code APPROVE → main Ready verification → clean Compiler
  implementation → independent code/docs APPROVE → main consumer/architecture checkpoint →
  later separately briefed Checkpoint copy/rebind → NN workflow qualification → final program join.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Integration [validation](../../../../glossary.md#term-validation): main's consumer/architecture [checkpoint](../../../../glossary.md#term-checkpoint) below; no [CPU](../../../../glossary.md#term-cpu) full/native gate here.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main, exclusive shared [planning](../../../../glossary.md#term-planning)/authority owner.

## Files and symbols

Paths use `io/github/pho001/synaptik/` beneath each Java root.

- `modules/compiler/src/main/java/.../compiler/SourceGraphReconstruction.java`: preflight,
  additional, validateLeaves, reconstruct; scope-aware private helpers, not a new reconstruction layer.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [scope glossary](../../../../glossary.md#term-scope))</span>
- Same package: GraphCompilationPort.SourceCapture/cold Javadoc, GraphCapture.captureSource/captureBody,
  ScanBodyInference.capture/infer/validateResiduals and CapturedGraphInference.inferAndValidate.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Compiler tests: SourceGraphCaptureTest, SourceGraphReconstructionTest, GeneralScanCompilerTest,
  ScanBodyInferenceTest; update old GENERAL cold rejection assertions only, retain AD exclusions.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [AD glossary](../../../../glossary.md#autograd))</span>
- Read Model tensor ScanBody/Scan/ScanResult, Tensor and constructor helpers/attrs used below;
  graph CompiledNode/CompiledScanBody/CompiledGraphModel. No Model production changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [tensor glossary](../../../../glossary.md#tensor); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [graph glossary](../../../../glossary.md#graph); [CompiledGraphModel glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- `modules/checkpoint/src/test/java/.../checkpoint/graph/GraphCheckpointTest.java`: test-only consumer
  rejection. Read GraphSource.capture/copy, GraphArtifactCodec.encode/opcode and GraphCheckpoint.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode))</span>
- Read NN OwnedModelSource acquisition/forward and ModelExecutionState.rejectScanTraining;
  fixed RecurrentScanKind containsScan/rejectScanTraining stays unchanged.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## Constructor-backed inventory

The owning [Compiler](../../../../glossary.md#term-compiler) contract is exhaustive. Current public constructors/attribute classes were
inspected, not inferred from [CPU](../../../../glossary.md#term-cpu) support. All cases preserve raw full [descriptors](../../../../glossary.md#term-descriptor) and [Model](../../../../glossary.md#term-model) guards.

| [Scope](../../../../glossary.md#term-scope)/[signature](../../../../glossary.md#term-signature) | Reconstruction and [guard](../../../../glossary.md#term-guard) |
|---|---|
| Flat outer source | Existing closed mappings unchanged; add [general SCAN](../../../../glossary.md#general-differentiable-scan) with body only. In particular flat DIV/[scalar](../../../../glossary.md#term-scalar)/SUM/EXPAND/[CONTIGUOUS](../../../../glossary.md#term-contiguous)/CAST still reject. |
| Body reused mappings | Existing flat mappings except fixed/nested scans; same guards unless overridden below. [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) retains existing [vector](../../../../glossary.md#term-vector)/batched/promotion support; [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) execution is not a cold restriction. PERMUTE remains rank-two `[1,0]`. |
| Body [tensor](../../../../glossary.md#tensor) ADD/SUB/MUL/DIV | Two inputs, [NoOperationAttrs](../../../../glossary.md#nooperationattrs); add/sub/mul/div(Tensor). Numeric ADD/SUB/MUL (including integral body SUB), floating DIV, Model promotion/broadcast. |
| Body scalar ADD/SUB/MUL/DIV | One input, ScalarValueAttrs; typed [ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) overload, exact [dtype](../../../../glossary.md#data-type--datatype)/value. Numeric first three, floating DIV; no double conversion. |
| Body NEG/RELU/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid) | One input, NoOperationAttrs; existing floating constructor. |
| Body SUM | One numeric input; NoOperationAttrs→sum(); AxisReductionAttrs→sum([axis](../../../../glossary.md#term-axis),keep); MultiAxisReductionAttrs→sum(int[],keep); SumToShapeAttrs→sumToShape([Shape](../../../../glossary.md#shape)). Preserve [normalized axis](../../../../glossary.md#normalized-axis) order/empty axes/keep; no masked form. |
| Body RESHAPE/EXPAND | TargetShapeAttrs, Shape overload; normalized static [target](../../../../glossary.md#term-target), equal checked counts / exact broadcast. Reshape scalar/zero extents allowed; do not reinterpret a stored target as raw `-1` [inference](../../../../glossary.md#term-inference). |
| Body EXPAND_DIMS/SQUEEZE | AxisTransformAttrs; any valid normalized axis, [singleton](../../../../glossary.md#term-singleton) squeeze. |
| Body CONTIGUOUS / [identity](../../../../glossary.md#term-identity-meanings) CAST | NoOperationAttrs→contiguous(); CastAttrs with target=input dtype→cast(dtype). Both preserve fresh [producer](../../../../glossary.md#term-producer) semantics; no storage copy; CAST does not [alias](../../../../glossary.md#aliasing-and-arena) input. |
| [Constants](../../../../glossary.md#term-constant) | Explicit outer source input/[invariant](../../../../glossary.md#term-invariant) positions or scalar attrs. No free body [leaf](../../../../glossary.md#term-leaf), hidden constant producer, [host](../../../../glossary.md#term-host)-backed formal or new sidecar. |

Other [PURE](../../../../glossary.md#term-pure) families may be valid Model sources but have no selected cold mapping here. Existing
flat mappings are reused, not generalized (e.g. [convolution](../../../../glossary.md#term-channels-first-convolution-layers) geometry, [GATHER](../../../../glossary.md#gather)/[SLICE](../../../../glossary.md#slice)/CONCAT guards).
Cross-dtype CAST and arbitrary permutations are outside the selected closure, not invalid Model.

## Acceptance criteria

1. Whole outer graph plus every body/value/phase passes static descriptor/count/layout arithmetic,
   exact raw inference, no deferred obligations, formal/header/role/residual checks before <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>**any**
   formal/producer allocation. Include late/dead outer scans, T=0, invalid body nodes, unused values,
   overflow, missing/malformed attachments and scope errors. Pin no Tensor-ID consumption on known
   preflight failures. No physical resource/byte/alignment limits; those belong to Checkpoint/backend.
   Cold bodies require empty demandedOutputIndices and body.graph.outputs EXACTLY the first-encounter
   unique concatenation of nextCarry + emissions + residuals, in that order. Explicit no-ID-consumption
   tests reject nonempty demand and extra/reordered/missing role outputs before ANY formal/producer,
   including late/dead outer scans and T=0. Keep these checks cold-only, not ordinary compilation.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [nodes glossary](../../../../glossary.md#node); [overflow glossary](../../../../glossary.md#term-overflow); [scope glossary](../../../../glossary.md#term-scope); [Tensor glossary](../../../../glossary.md#tensor); [preflight glossary](../../../../glossary.md#term-preflight); [alignment glossary](../../../../glossary.md#term-alignment); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [backend glossary](../../../../glossary.md#backend); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
2. Reconstruct one fresh distinct canonical formal per declared slot, including unused formals;
   body nodes once, one ScanBody and Scan.apply per occurrence. Map all canonical ports, including
   latent/unrequested residuals, repeated nextCarry/emission roles and shared uses. Preserve both
   traversal flags, repeated actual positions, multiple occurrences/template reuse and independent
   local IDs even when numerically equal to outer IDs; retain no old formals/templates or actuals
   except the explicitly supplied outer substitutions.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Scan glossary](../../../../glossary.md#term-cumulative-scan); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
3. Outer-only substitutions accept full-descriptor-compatible produced/repeated Tensors with opaque
   upstream provenance. Keep ordinary provenance for later compile/AD. Scope tables are call-local,
   discarded on failure/return; no persistent map or ValueId-to-Tensor AD conversion.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque); [provenance glossary](../../../../glossary.md#provenance); [compile glossary](../../../../glossary.md#compile); [AD glossary](../../../../glossary.md#autograd); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot))</span>
4. Capture/reconstruct/recapture agree on raw graph/descriptors/roles/ordered roots for every inventory
   row and applicable Model dtype. Include valid low TANH, F32 RELU, broadcast/reduction/layout/identity
   cast, rank-two MATMUL/transpose, explicit constants, T=0/empties and differing static T with O(body)
   object counts. Preserve old flat rejection and unresolved MATMUL/CONCAT/view round trips.
   Include positive repeated-role cases proving first-encounter-unique body output order and empty
   raw demand survive round trips; ordinary compiled attachments may still carry nonempty demand.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [Model glossary](../../../../glossary.md#term-model); [dtype glossary](../../../../glossary.md#data-type--datatype); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [identity glossary](../../../../glossary.md#term-identity-meanings); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [constants glossary](../../../../glossary.md#term-constant); [view glossary](../../../../glossary.md#term-view-meanings); [carry glossary](../../../../glossary.md#term-carry))</span>
5. Compiler-only stage leaves Checkpoint/NN fail-closed. Add focused GraphCheckpointTest cases for
   GENERAL copy/rebind/save (both directions, final/residual-root coverage), with otherwise valid named
   state and source boundaries. Existing four-arg CompiledNode must reject missing body during
   GraphSource normalization, before it returns: no silent lossy result. A storage spy must observe
   zero host segment reads; dead/absent state must not win error precedence. Existing target bytes
   and directory entries stay unchanged, absent target remains absent. Rebind must not reach replacement
   admission. No production consumer guard needed or authorized. Future Checkpoint task replaces this
   temporary rejection with attachment-preserving copy/rebind and independent wire rejection.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [normalization glossary](../../../../glossary.md#term-normalization); [host glossary](../../../../glossary.md#term-host); [target glossary](../../../../glossary.md#term-target); [admission glossary](../../../../glossary.md#term-admission); [guard glossary](../../../../glossary.md#term-guard))</span>
6. Existing fixed scan source/ports, ordinary Compiler AD and fixed NN early training rejection remain
   unchanged. No CPU/backend qualification or file-byte compatibility claim is widened.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused development (extend existing named tests rather than add a production type):

```bash
./gradlew :modules:compiler:test --tests '*SourceGraphCaptureTest' --tests '*SourceGraphReconstructionTest' --tests '*GeneralScanCompilerTest' --tests '*ScanBodyInferenceTest'
./gradlew :extensions:checkpoint:test --tests '*GraphCheckpointTest'
```

After final executable/[Javadoc](../../../../glossary.md#term-javadoc) edits, one [module](../../../../glossary.md#term-module) gate:

```bash
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

Main after independent implementation review runs one consumer/architecture [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
./gradlew :extensions:checkpoint:test
./gradlew :extensions:nn:test --tests '*OwnedModelSourceTest' --tests '*OwnedModelTrainingTest' --tests '*RuntimeLengthScanModelTest'
./gradlew :testing:architecture-tests:test
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
```

Reuse focused successful worker evidence; no full [CPU](../../../../glossary.md#term-cpu)/native test or benchmark now. One source-matched
native-enabled root `test javadoc` belongs to the FINAL future [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) program join, not every prerequisite;
no [dependency](../../../../glossary.md#term-dependency)/[backend](../../../../glossary.md#backend)/execution change justifies repeating that root gate for [Compiler](../../../../glossary.md#term-compiler)-only source work.

## Documentation and review impact

Authority cutover precedes Java: root plus [Compiler](../../../../glossary.md#term-compiler)/[scan](../../../../glossary.md#term-cumulative-scan)/extension [owners](../../../../glossary.md#term-owner), ADR0046 and focused
[module](../../../../glossary.md#term-module)-boundaries/training-[graph](../../../../glossary.md#graph)/general-[differentiable](../../../../glossary.md#derivative-and-chain-rule)-scan explanations and glossary are staged.
Update GraphCompilationPort/SourceGraphReconstruction [Javadoc](../../../../glossary.md#term-javadoc) and affected helper contracts in the
implementation; update docs/api/[compile](../../../../glossary.md#compile)-api.md's cold boundary; review docs/api/graph-[checkpoint](../../../../glossary.md#term-checkpoint)-api.md
and user-guide/running-models.md without
claiming Compiler-only work delivers Checkpoint/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) support. No user workflow example until its gate.
Independent pre-code architecture/inventory review and later targeted code/docs review are mandatory,
in clean threads; review actual source/tests and task-only diff, reuse tests unless code changes.

## Result

Status: Complete — main accepted the unchanged reviewed source and all recorded gates, 2026-10-23.

- Implemented bounded GENERAL [cold reconstruction](../../../../glossary.md#term-cold-reconstruction) in existing SourceGraphReconstruction only; no new production type/[API](../../../../glossary.md#term-application-programming-interface-api)/[dependency](../../../../glossary.md#term-dependency), [AD](../../../../glossary.md#autograd)/execution or [backend capability](../../../../glossary.md#backend-capability).
- Seven changed files: SourceGraphReconstruction.java (executable), GraphCompilationPort.java ([Javadoc](../../../../glossary.md#term-javadoc)), SourceGraphReconstructionTest.java, GeneralScanCompilerTest.java, GraphCheckpointTest.java, docs/api/[compile](../../../../glossary.md#compile)-api.md and docs/api/[graph](../../../../glossary.md#graph)-[checkpoint](../../../../glossary.md#term-checkpoint)-api.md.
- Approved baseline/authority preparation and historical [Ready](../../../../glossary.md#term-ready) verification above preceded implementation. Final independent code/docs APPROVE, zero P1/P2, no reviewer edits: `/private/tmp/synaptik-compiler0012-final-review.d6gsha/final-review.md`.
- Frozen task-only diff [SHA-256](../../../../glossary.md#term-sha-256) `ee968ff19f0c210367c8c1b1c27ce7ce27f077f8abce7ffc6d7fe49453c8db68`; all seven reviewed [hashes](../../../../glossary.md#term-hash) unchanged through gates.
- Recorded focused commands: [Compiler](../../../../glossary.md#term-compiler) 32 + Checkpoint 10 = 42 passing tests, subsets of final gates, not additional tests.
- `./gradlew :modules:compiler:test :modules:compiler:javadoc` ran once after review: PASS, 46 suites/347 tests; Javadoc zero warnings.
- `./gradlew :extensions:checkpoint:test :extensions:nn:test --tests '*OwnedModelSourceTest' --tests '*OwnedModelTrainingTest' --tests '*RuntimeLengthScanModelTest' :testing:architecture-tests:test`: PASS; Checkpoint 6 suites/47 tests, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 3/31, architecture 11/25.
- `./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'`: PASS, 1 suite/17 tests. Final unique gates total 467 tests, zero failures/errors/skips.
- Worker [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/results: `/private/tmp/synaptik-compiler0012-impl.wnir9h/` compiler-focused-4.[log](../../../../glossary.md#term-log), checkpoint-focused-3.log, compiler-[module](../../../../glossary.md#term-module)-gate.log, module-gate-result.md and module-gate-counts.log; main logs: `/private/tmp/synaptik-compiler0012-coordination.8rCsJ1/` consumer-checkpoint.log and [engine](../../../../glossary.md#term-engine)-checkpoint.log.
- Independent Javadoc/API/explanation review finalized the drafts; helper contracts and targeted glossary remained accurate, no new terminology/example or further edits. Links/anchors and whitespace passed.
- Main verified unchanged 263-path source [snapshot](../../../../glossary.md#term-snapshot) before/after gates, HEAD=origin/main at the recorded base, empty index and preserved unrelated dirty baseline. Existing [FFM](../../../../glossary.md#term-ffm)/incubator consumer warnings only; no failure.
- No unresolved acceptance issue. Checkpoint/NN remain [fail-closed](../../../../glossary.md#term-fail-closed); attachment-preserving Checkpoint copy/[rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) is next, separately briefed, then NN [qualification](../../../../glossary.md#term-qualification). No new Ready brief or successor launch.
- No native build, performance campaign or root rerun; one source-matched native root test/Javadoc remains at the final future NN join.
- Size guardrail exception: retaining this cohesive Class C acceptance/Ready packet plus final evidence slightly exceeds 200 lines/15 KB; no accepted [scope](../../../../glossary.md#term-scope) was rewritten.

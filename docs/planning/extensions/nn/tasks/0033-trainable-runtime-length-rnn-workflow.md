# Task 0033: Trainable runtime-length RNN Model workflow

## Status

Complete — MAIN accepted 2026-11-01 after independent core/init and numerical/persistence
APPROVE (zero open P1/P2), 566 NN plus 54 integration tests without failures/errors/skips,
Model/NN Javadocs and the public F32/F64 lifecycle plus exact documented example.

## Change class

Class C — public composition semantics, owned state/lifecycle and persistence/training workflow.
Use a clean implementer and an independent code/documentation reviewer.

## Goal and scope

Deliver the ordinary F32/F64 runtime-length tanh-RNN journey through existing general Scan,
Model, Engine, Training and GraphCheckpoint owners: construction, compute, configureTraining,
repeated changed-input/length train, save, close, loadGraph, compute and freshly configured train.
Keep ordinary statically unrolled `.rnn` unchanged; this is the explicit `rnnScan` recipe cutover,
not NN0032's separate all-valid migration or a higher-order compatibility waiver.

## Accepted API and composition decision

Add exactly this composition convenience, returning the existing semantic result carrier:
```java
public RecurrentScanResult forwardGeneralScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, RecurrentDirection direction);
```
Admit homogeneous configured FLOAT32/FLOAT64, static input [T,B,F], state [B,H] and non-gradient
INT64[B] lengths, T/B >= 0, F/H > 0, bias on/off and both directions. Reject unsupported dtype or
geometry before automatic initialization/body construction. Read no actual storage or length
values; preserve produced-actual provenance and caller-coordinated parameter replacement.

Reuse the existing descriptor-only `initializeForScanIfNeeded` gate and snapshot current weights
once in declaration order. Factor one package-private stateless `RnnCell.composeStep` shared by
ordinary forward and the general body, preserving projection/broadcast prevalidation before any
expression prefix and the existing bias/ADD order. No new parameter carrier, dummy transition,
temporary cell, recapture, parameter mutation or public initialization hook.

Create canonical storage-free formals for carry [B,H], step [B,F], and invariants inputWeight,
hiddenWeight, optional bias, retaining dtype/gradient flags. Use the shared next state as both
next carry and emission. Bind actual weights as outer invariants to Scan.apply with rowCarryCount=1,
sequenceRowAxes=[ROW(0)] and traversal from direction. RecurrentScanResult component names express
semantic roles: general outputs are producer port1 and finalHidden port0; fixed ordering stays 0/1.
Update only carrier Javadoc, never manufacture old order/layout with graph operations.

Existing direct `forwardScan` remains the fixed inference API. The builder's explicit rnnScan
uses the general method for F32/F64; retained low/fixed paths and GRU/LSTM remain unchanged.
Document the selected exact output-layout and backend qualification cutover. Final-hidden
selection must understand the declared producer ports: general Scan carry 0 versus the fixed
scan final-hidden port 1; do not retain the old hardcoded selector. BuildWithLengths allows
training only for sources supported by ordinary general-Scan AD, not a blanket fixed-scan waiver.
No shape/dtype conversion or CPU fallback is implicit. Padding affects scan activity only, not
an automatic downstream objective mask. Maintain explicit all-zero-length final-state behavior.

## Contracts

- [Root lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle) and Core invariants,
  Module-ownership routing, Dependency rules, Testing requirements.
- [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
  modules/training and optimizer/training lifecycle: one state set, existing owners, fresh inputs,
  borrowed Engine and cached preparations, no retained loss/optimizer recipe in saved artifacts.
- [Length-aware scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan):
  homogeneous F32/F64 first-order qualification, explicit roles/activity and stable carry shape.
- [Cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
  and [Checkpoint v5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5).

## Dependencies and integration

- Depends on: Compiler0018 and Checkpoint0014 Complete; current static Model/Training/NN baseline.
- Conflicts with: NN0032, RnnCell/RnnSequence/ModelBuilder/source-state or shared NN contract changes;
  serialize; do not migrate `.rnn`, GRU/LSTM or fixed kernels incidentally.
- Parallel group: None.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus MAIN's exact
  post-Checkpoint0014 accepted source manifest, recorded in `nn0033-base.json` before launch.
- Integration order: approved bounded API/authority -> isolated implementation -> independent review
  -> MAIN affected NN/consumer gate and public numerical smoke.
- Integration validation: NN module tests/Javadoc, named runtime-length Model/integration tests;
  one final production root checkpoint after all selected fixes.
- Shared-document integration owner: MAIN.

MAIN Ready verification: Compiler0018 and Checkpoint0014 are Complete; v5 passed 619 affected
tests/Javadocs and loaded F32/F64 generic Model numerical proof. Root/owning NN authority and
ADR0058 permit the exact method/builder exception below; independent review APPROVED. No
competing NN/Model/Compiler/Checkpoint source writer is active. Freeze the dirty source snapshot;
MAIN alone owns shared documents and later integration. Code workers run no mid-flight checks.

## Files and symbols

- Existing nn.layers RnnSequence/RnnCell: explicit general-Scan composition and shared cell equation;
  no new production type or package. Model RecurrentScanResult: semantic-component Javadoc only.
- Existing nn.module ModelBuilder: explicit rnnScan route, finalHidden selector and accurate Javadoc.
- ModelExecutionState/OwnedModelSource only if a concrete owned lifecycle gap is demonstrated;
  do not create another cache/session or change Training ownership.
- Existing NN and integration tests plus focused new numerical runtime-length workflow cases.
- MAIN owns root/scoped/API/user-guide/master/roadmap integration and exact admission record.

## Acceptance criteria

- F32/F64 builder and direct general composition match independent scalar/unrolled numerical
  forward and first-order gradients for nonuniform lengths, both direct traversal directions,
  length zero and changed inputs/lengths. Padded NaNs cannot participate in inactive transitions.
- Repeated SGD updates change the expected parameters, preserve caller-owned weights and use
  fresh inputs without recapture/reprepare; invalid lengths cause no partial parameter update
  and a valid subsequent call remains usable under current lifecycle rules.
- Save before compute and after training; close original, loadGraph, compare inference and
  reconfigure/train restored weights. Objectives and optimizer state are new, not exact resumed.
- Construction/cold forward/save create no Engine. State paths, seeds/initialization order,
  aliases, caller/borrowed Engine ownership, result detachment, exclusion and close remain sound.
- Original `.rnn`, GRU/LSTM, direct fixed forwardScan, low fixed paths and NN0032 compatibility
  tests retain their prior supported behavior. Unsupported saved/engine domains fail truthfully.
- Direct composition covers eager/supplied/automatic/strict-loaded cells, bias on/off, both
  directions/dtypes and T=0/B=0. Preserve seed bytes, initialization order, stable paths/wrappers,
  bound-cell reuse, concurrent first-call publication and retry after invalid descriptors.
- Verify exact general/fixed producer ports, finalHidden across retained families and mixed
  stacks, dense lastTimeStep versus shortened final hidden, and later parameter replacement
  without rebinding earlier occurrences.
- Numerical cases include joint final/emission gradients, requested initial/input/weight/bias
  roles, non-square F/H, B=1 and all-zero/full/nonuniform lengths. All-zero lengths preserve raw
  initial state, positive-zero emissions and zero recurrent-parameter gradients; not every valid
  SGD call must change recurrent parameters. Invalid lengths preserve optimizer state as well.
- V5 is selected for new general RNN sources. Old v3 fixed-RNN files remain fixed, never migrated;
  retained fixed and mixed fixed/general sources keep pre-setup training rejection, including
  callback/Engine/native-copy/cache preservation. Preserve retained old-version selection.
- Delete incidental exact public-method inventory tests rather than repinning a method-name set;
  retain actual fixed-signature, producer, error-order and numerical compatibility tests.
- Qualify changed canonical general output layouts and narrower proved actual-read domain
  explicitly; no erased view metadata, automatic copy or silent old-pass/new-reject regression.
- No duplicated cell mathematics, workaround flag, synthetic placeholder, body interpreter,
  per-step resource construction or new execution owner. Humans can follow each named method.

## Validation

Skip build/lint/tests/formatters mid-flight. MAIN runs focused and affected NN/consumer validation
once sources and independent review stabilize, then a real throwaway public Model train/save/load
scenario in both CPU dtypes. Root/native validation belongs to the final source-matched release
checkpoint; reuse earlier unaffected evidence.

## Documentation and review impact

Independent review must cover public API/compatibility, cell-equation reuse, initialization,
exact ports/roles, state ownership, changed input/length binding, numerical oracle independence,
codec selection and failure atomicity. MAIN updates the owning NN contract, public API and
running-models guide in the same integrated change. Do not mark Draft as capability delivered.

## Result

Implemented in existing RnnCell/RnnSequence/ModelBuilder and RecurrentScanResult Javadoc, with
six affected test classes (two new). The 2621-file dirty base and ten integrated path hashes are
recorded in `nn0033-base.json` and `nn0033-integrated-files.json`; main counterparts matched
the fork base, preserving main-only Model Javadoc and user changes.

One package-private composeStep shares ordinary forward and Scan mathematics, prevalidation and
bias/ADD order. Existing initialization/publication, Model caches, Training and Checkpoint owners
are reused; ModelExecutionState/OwnedModelSource needed no changes. Builder cutover is only
explicit F32/F64 rnnScan; finalHidden uses actual general/fixed producer ports.

Independent core review found one P2: canonical emission/latent geometry could overflow after
automatic parameter publication. Corrected with existing checked Shape/layout helpers before
initialization/body IDs, including actual input count and both transposed-weight histories.
Regression cases preserve IDs/unbound reservations, accept large legal empty geometry and retry
with a different valid width. Final core and separate consumer/oracle reviews APPROVED.

MAIN JDK26 gate: Model Javadoc, NN tests/Jar/Javadoc, and six named integration classes —
PASS (`artifact://104`): 566 NN tests and 54 integration tests, zero failures/errors/skips.
The integration selection covered TrainableRuntimeLengthRnn, RuntimeLengthScanModel,
PaddedNumericSequenceInference, RuntimeLengthRecurrentNn, GeneralScanOwnedModel and
FluentModelBuilder. Independent scalar/BPTT cases qualify joint final/emission derivatives,
all requested roles, directions, bias modes, non-square widths, empty/inactive domains and
retained fixed-v3/mixed-source behavior. NN momentum coverage checks failed-length optimizer
slots/counter/parameter atomicity and valid reuse.

Unchanged public RnnProductionSmoke first observed both old F32/F64 fixed-builder training
rejections. After cutover, [T=3,B=2,F=2] -> RNN H=3 -> finalHidden -> linear1 matched independent
scalar/finite-difference expectations for all five parameter groups across changed lengths,
padded NaNs and four updates. Invalid lengths preserved state/next optimizer step. Save before
compute/after train, close/loadGraph, fresh [2,0] inference and newly configured step1 passed.
Logs: `rnn-production-before.log`, `rnn-production-after.log`; no optional Vector module needed.

The exact running-models Java example was extracted and run: `shape [2,1]: true`, then
`new optimizer step: 1` (`nn0033-documentation-example.log`). A missing closing class brace
in its first draft was corrected before this successful run. Throwaway sources are removed;
numerical logs remain. Full root/release acceptance, arbitrary PURE body composition, higher-order
Scan and ordinary .rnn migration are not claimed by this bounded completion.

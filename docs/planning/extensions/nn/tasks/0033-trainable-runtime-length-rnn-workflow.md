# Task 0033: Trainable [runtime](../../../../glossary.md#term-runtime)-length [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) [Model](../../../../glossary.md#term-model) workflow

## Status

Complete — MAIN accepted 2026-11-01 after independent core/init and numerical/[persistence](../../../../glossary.md#term-persistence)
APPROVE (zero open P1/P2), 566 [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) plus 54 integration tests without failures/errors/skips,
[Model](../../../../glossary.md#term-model)/NN Javadocs and the public [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [lifecycle](../../../../glossary.md#lifecycle) plus exact documented example.

## Change class

Class C — public composition semantics, owned state/[lifecycle](../../../../glossary.md#lifecycle) and [persistence](../../../../glossary.md#term-persistence)/training workflow.
Use a clean implementer and an independent code/documentation reviewer.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver the ordinary [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [runtime](../../../../glossary.md#term-runtime)-length [tanh](../../../../glossary.md#term-tanh)-[RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) journey through existing [general Scan](../../../../glossary.md#general-differentiable-scan),
[Model](../../../../glossary.md#term-model), [Engine](../../../../glossary.md#term-engine), Training and GraphCheckpoint [owners](../../../../glossary.md#term-owner): construction, compute, configureTraining,
repeated changed-input/length train, save, close, loadGraph, compute and freshly configured train.
Keep ordinary statically [unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) `.rnn` unchanged; this is the explicit `rnnScan` recipe cutover,
not NN0032's separate all-valid migration or a higher-order compatibility waiver.

## Accepted [API](../../../../glossary.md#term-application-programming-interface-api) and composition decision

Add exactly this composition convenience, returning the existing semantic result [carrier](../../../../glossary.md#term-storage-carrier):
```java
public RecurrentScanResult forwardGeneralScan(
        Tensor input, Tensor validLengths, Tensor initialHidden, RecurrentDirection direction);
```
Admit [homogeneous](../../../../glossary.md#term-homogeneous) configured [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64), static input [T,B,F], state [B,H] and non-[gradient](../../../../glossary.md#term-gradient)
[INT64](../../../../glossary.md#data-type--datatype)[B] lengths, T/B >= 0, F/H > 0, [bias](../../../../glossary.md#term-bias) on/off and both directions. Reject unsupported [dtype](../../../../glossary.md#data-type--datatype) or
geometry before automatic initialization/body construction. Read no actual storage or length
values; preserve produced-actual [provenance](../../../../glossary.md#provenance) and caller-coordinated [parameter](../../../../glossary.md#term-parameter) replacement.

Reuse the existing [descriptor](../../../../glossary.md#term-descriptor)-only `initializeForScanIfNeeded` gate and [snapshot](../../../../glossary.md#term-snapshot) current [weights](../../../../glossary.md#term-weight)
once in declaration order. Factor one package-private stateless `RnnCell.composeStep` shared by
ordinary forward and the general body, preserving projection/broadcast prevalidation before any
expression prefix and the existing bias/ADD order. No new parameter carrier, dummy transition,
temporary cell, recapture, parameter mutation or public initialization hook.

Create [canonical](../../../../glossary.md#term-canonical-form) storage-free formals for [carry](../../../../glossary.md#term-carry) [B,H], step [B,F], and invariants inputWeight,
hiddenWeight, optional bias, retaining dtype/gradient flags. Use the shared next state as both
next carry and emission. Bind actual weights as outer invariants to [Scan](../../../../glossary.md#term-cumulative-scan).apply with rowCarryCount=1,
sequenceRowAxes=[ROW(0)] and traversal from direction. RecurrentScanResult component names express
semantic roles: general outputs are [producer](../../../../glossary.md#term-producer) port1 and finalHidden port0; fixed ordering stays 0/1.
Update only carrier [Javadoc](../../../../glossary.md#term-javadoc), never manufacture old order/[layout](../../../../glossary.md#layout) with [graph](../../../../glossary.md#graph) operations.

Existing direct `forwardScan` remains the fixed [inference](../../../../glossary.md#term-inference) API. The builder's explicit rnnScan
uses the general method for [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding); retained low/fixed paths and [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) remain unchanged.
Document the selected exact output-layout and [backend](../../../../glossary.md#backend) [qualification](../../../../glossary.md#term-qualification) cutover. Final-hidden
selection must understand the declared producer ports: [general Scan](../../../../glossary.md#general-differentiable-scan) carry 0 versus the fixed
scan final-hidden port 1; do not retain the old hardcoded selector. BuildWithLengths allows
training only for sources supported by ordinary general-Scan [AD](../../../../glossary.md#autograd), not a blanket fixed-scan waiver.
No [shape](../../../../glossary.md#shape)/dtype conversion or [CPU](../../../../glossary.md#term-cpu) fallback is implicit. [Padding](../../../../glossary.md#padding) affects scan activity only, not
an automatic downstream [objective](../../../../glossary.md#term-objective) [mask](../../../../glossary.md#term-boolean-mask). Maintain explicit all-zero-length final-state behavior.

## Contracts

- [Root lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> and Core invariants,
  Module-ownership routing, Dependency rules, Testing requirements.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>,
  modules/training and optimizer/training lifecycle: one state set, existing owners, fresh inputs,
  borrowed Engine and cached preparations, no retained loss/optimizer recipe in saved artifacts.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [Engine glossary](../../../../glossary.md#term-engine); [loss glossary](../../../../glossary.md#loss); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- [Length-aware scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>:
  homogeneous F32/F64 first-order qualification, explicit roles/activity and stable carry shape.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [qualification glossary](../../../../glossary.md#term-qualification); [carry glossary](../../../../glossary.md#term-carry); [shape glossary](../../../../glossary.md#shape))</span>
- [Cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([Cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction))</span>
  and [Checkpoint v5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5)<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>.

## Dependencies and integration

- Depends on: Compiler0018 and Checkpoint0014 Complete; current static [Model](../../../../glossary.md#term-model)/Training/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) baseline.
- Conflicts with: NN0032, RnnCell/RnnSequence/ModelBuilder/source-state or shared NN contract changes;
  serialize; do not migrate `.rnn`, [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) or fixed [kernels](../../../../glossary.md#kernel) incidentally.
- Parallel group: None.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus MAIN's exact
  post-Checkpoint0014 accepted source manifest, recorded in <span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>`nn0033-base.json` before launch.
- Integration order: approved bounded API/authority -> isolated implementation -> independent review
  -> MAIN affected NN/consumer gate and public numerical smoke.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [smoke glossary](../../../../glossary.md#term-smoke-test))</span>
- Integration validation: NN module tests/Javadoc, named runtime-length Model/integration tests;
  one final production root checkpoint after all selected fixes.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

MAIN [Ready](../../../../glossary.md#term-ready) verification: Compiler0018 and Checkpoint0014 are Complete; v5 passed 619 affected
tests/Javadocs and loaded [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) generic Model numerical proof. Root/owning NN authority and
ADR0058 permit the exact method/builder exception below; independent review APPROVED. No
competing NN/Model/[Compiler](../../../../glossary.md#term-compiler)/[Checkpoint](../../../../glossary.md#term-checkpoint) source writer is active. Freeze the dirty source [snapshot](../../../../glossary.md#term-snapshot);
MAIN alone owns shared documents and later integration. Code workers [run](../../../../glossary.md#term-run) no mid-flight checks.

## Files and symbols

- Existing nn.layers RnnSequence/RnnCell: explicit general-Scan composition and shared cell equation;
  no new production type or package. Model RecurrentScanResult: semantic-component Javadoc only.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [Model glossary](../../../../glossary.md#term-model))</span>
- Existing nn.[module](../../../../glossary.md#term-module) ModelBuilder: explicit rnnScan route, finalHidden selector and accurate [Javadoc](../../../../glossary.md#term-javadoc).
- ModelExecutionState/OwnedModelSource only if a concrete owned lifecycle gap is demonstrated;
  do not create another cache/session or change Training ownership.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [cache glossary](../../../../glossary.md#term-cache); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and integration tests plus focused new numerical [runtime](../../../../glossary.md#term-runtime)-length workflow cases.
- MAIN owns root/scoped/[API](../../../../glossary.md#term-application-programming-interface-api)/user-guide/master/roadmap integration and exact [admission](../../../../glossary.md#term-admission) record.

## Acceptance criteria

- F32/F64 builder and direct general composition match independent scalar/unrolled numerical
  forward and first-order gradients for nonuniform lengths, both direct traversal directions,
  length zero and changed inputs/lengths. Padded NaNs cannot participate in inactive transitions.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Repeated SGD updates change the expected parameters, preserve caller-owned weights and use
  fresh inputs without recapture/reprepare; invalid lengths cause no partial parameter update
  and a valid subsequent call remains usable under current lifecycle rules.<span data-glossary-navigation="true"> ([SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [parameters glossary](../../../../glossary.md#term-parameter); [weights glossary](../../../../glossary.md#term-weight); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Save before compute and after training; close original, loadGraph, compare inference and
  reconfigure/train restored weights. Objectives and optimizer state are new, not exact resumed.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [optimizer state glossary](../../../../glossary.md#term-optimizer-state))</span>
- Construction/cold forward/save create no Engine. State paths, seeds/initialization order,
  aliases, caller/borrowed Engine ownership, result detachment, exclusion and close remain sound.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Original `.rnn`, GRU/LSTM, direct fixed forwardScan, low fixed paths and NN0032 compatibility
  tests retain their prior supported behavior. Unsupported saved/engine domains fail truthfully.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Direct composition covers eager/supplied/automatic/strict-loaded cells, bias on/off, both
  directions/dtypes and T=0/B=0. Preserve seed bytes, initialization order, stable paths/wrappers,
  bound-cell reuse, concurrent first-call publication and retry after invalid descriptors.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [bias glossary](../../../../glossary.md#term-bias); [dtypes glossary](../../../../glossary.md#data-type--datatype); [seed glossary](../../../../glossary.md#term-seed); [publication glossary](../../../../glossary.md#term-publication); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- Verify exact general/fixed producer ports, finalHidden across retained families and mixed
  stacks, dense lastTimeStep versus shortened final hidden, and later parameter replacement
  without rebinding earlier occurrences.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Numerical cases include joint final/emission gradients, requested initial/input/weight/bias
  roles, non-square F/H, B=1 and all-zero/full/nonuniform lengths. All-zero lengths preserve raw
  initial state, positive-zero emissions and zero recurrent-parameter gradients; not every valid
  SGD call must change recurrent parameters. Invalid lengths preserve optimizer state as well.
- V5 is selected for new general RNN sources. Old v3 fixed-RNN files remain fixed, never migrated;
  retained fixed and mixed fixed/general sources keep pre-setup training rejection, including
  callback/Engine/native-copy/cache preservation. Preserve retained old-version selection.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [callback glossary](../../../../glossary.md#term-callback); [cache glossary](../../../../glossary.md#term-cache))</span>
- Delete incidental exact public-method inventory tests rather than repinning a method-name set;
  retain actual fixed-signature, producer, error-order and numerical compatibility tests.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Qualify changed canonical general output layouts and narrower proved actual-read domain
  explicitly; no erased view metadata, automatic copy or silent old-pass/new-reject regression.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [regression glossary](../../../../glossary.md#term-regression))</span>
- No duplicated cell mathematics, workaround flag, synthetic placeholder, body interpreter,
  per-step resource construction or new execution owner. Humans can follow each named method.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## [Validation](../../../../glossary.md#term-validation)

Skip build/lint/tests/formatters mid-flight. MAIN runs focused and affected [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/consumer validation
once sources and independent review stabilize, then a real throwaway public [Model](../../../../glossary.md#term-model) train/save/load
scenario in both [CPU](../../../../glossary.md#term-cpu) [dtypes](../../../../glossary.md#data-type--datatype). Root/native validation belongs to the final source-matched [release](../../../../glossary.md#term-release)
[checkpoint](../../../../glossary.md#term-checkpoint); reuse earlier unaffected evidence.

## Documentation and review impact

Independent review must cover public [API](../../../../glossary.md#term-application-programming-interface-api)/compatibility, cell-equation reuse, initialization,
exact ports/roles, state [ownership](../../../../glossary.md#term-ownership-meanings), changed input/length [binding](../../../../glossary.md#term-binding), [numerical oracle](../../../../glossary.md#adr-ci-and-qualification) independence,
[codec](../../../../glossary.md#term-codec) selection and [failure atomicity](../../../../glossary.md#term-failure-atomicity). MAIN updates the owning [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) contract, public API and
running-models guide in the same integrated change. Do not mark Draft as capability delivered.

## Result

Implemented in existing RnnCell/RnnSequence/ModelBuilder and RecurrentScanResult [Javadoc](../../../../glossary.md#term-javadoc), with
six affected test classes (two new). The 2621-file dirty base and ten integrated path [hashes](../../../../glossary.md#term-hash) are
recorded in `nn0033-base.json` and `nn0033-integrated-files.json`; main counterparts matched
the [fork](../../../../glossary.md#term-fork-meanings) base, preserving main-only [Model](../../../../glossary.md#term-model) Javadoc and user changes.

One package-private composeStep shares ordinary forward and [Scan](../../../../glossary.md#term-cumulative-scan) mathematics, prevalidation and
[bias](../../../../glossary.md#term-bias)/ADD order. Existing initialization/[publication](../../../../glossary.md#term-publication), Model caches, Training and [Checkpoint](../../../../glossary.md#term-checkpoint) [owners](../../../../glossary.md#term-owner)
are reused; ModelExecutionState/OwnedModelSource needed no changes. Builder cutover is only
explicit [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) rnnScan; finalHidden uses actual general/fixed [producer](../../../../glossary.md#term-producer) ports.

Independent core review found one P2: [canonical](../../../../glossary.md#term-canonical-form) emission/latent geometry could [overflow](../../../../glossary.md#term-overflow) after
automatic [parameter](../../../../glossary.md#term-parameter) publication. Corrected with existing checked [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout) helpers before
initialization/body IDs, including actual input count and both transposed-[weight](../../../../glossary.md#term-weight) histories.
[Regression](../../../../glossary.md#term-regression) cases preserve IDs/unbound reservations, accept large legal empty geometry and retry
with a different valid width. Final core and separate consumer/[oracle](../../../../glossary.md#term-test-oracle) reviews APPROVED.

MAIN JDK26 gate: Model Javadoc, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tests/Jar/Javadoc, and six named integration classes —
PASS (`artifact://104`): 566 NN tests and 54 integration tests, zero failures/errors/skips.
The integration selection covered TrainableRuntimeLengthRnn, RuntimeLengthScanModel,
PaddedNumericSequenceInference, RuntimeLengthRecurrentNn, GeneralScanOwnedModel and
FluentModelBuilder. Independent [scalar](../../../../glossary.md#term-scalar)/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) cases qualify joint final/emission [derivatives](../../../../glossary.md#term-derivative),
all requested roles, directions, bias modes, non-square widths, empty/inactive domains and
retained fixed-v3/mixed-source behavior. NN [momentum](../../../../glossary.md#term-momentum) coverage checks failed-length [optimizer](../../../../glossary.md#term-optimizer)
slots/counter/parameter [atomicity](../../../../glossary.md#term-atomicity) and valid reuse.

Unchanged public RnnProductionSmoke first observed both old F32/F64 fixed-builder training
rejections. After cutover, [T=3,B=2,F=2] -> [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) H=3 -> finalHidden -> linear1 matched independent
scalar/[finite-difference](../../../../glossary.md#term-finite-difference-gradient-check) expectations for all five parameter groups across changed lengths,
padded NaNs and four updates. Invalid lengths preserved state/next [optimizer step](../../../../glossary.md#optimizer-and-stochastic-gradient-descent). Save before
compute/after train, close/loadGraph, fresh [2,0] [inference](../../../../glossary.md#term-inference) and newly configured step1 passed.
[Logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings): `rnn-production-before.log`, `rnn-production-after.log`; no optional [Vector](../../../../glossary.md#term-vector) [module](../../../../glossary.md#term-module) needed.

The exact running-models Java example was extracted and [run](../../../../glossary.md#term-run): `shape [2,1]: true`, then
`new optimizer step: 1` (`nn0033-documentation-example.log`). A missing closing class brace
in its first draft was corrected before this successful run. Throwaway sources are removed;
numerical logs remain. Full root/[release](../../../../glossary.md#term-release) acceptance, arbitrary [PURE](../../../../glossary.md#term-pure) body composition, higher-order
Scan and ordinary .rnn migration are not claimed by this bounded completion.

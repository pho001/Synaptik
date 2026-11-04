# Task 0077: [F64](../../../../glossary.md#floating-point-precision-and-rounding) [finite](../../../../glossary.md#term-finite)-[softmax](../../../../glossary.md#softmax--log-softmax) [admission](../../../../glossary.md#term-admission) parity

## Status

Complete — independently reviewed and MAIN accepted after source-matched targeted gates and the
unchanged public after scenario. [F64](../../../../glossary.md#floating-point-precision-and-rounding) now shares [F32](../../../../glossary.md#floating-point-precision-and-rounding)'s [finite](../../../../glossary.md#term-finite)-input domain; no mathematical emitter,
[ABI](../../../../glossary.md#term-application-binary-interface-abi), algorithm version or [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings) changed.

## Change class

Class C — value-dependent [admission](../../../../glossary.md#term-admission), numerical-domain and pre-write failure guarantees.
Clean implementer and independent code/documentation review; no new mathematical algorithm.

## Goal and observed result

[CPU](../../../../glossary.md#term-cpu) [F32](../../../../glossary.md#floating-point-precision-and-rounding) [SOFTMAX](../../../../glossary.md#softmax--log-softmax) and LOG_SOFTMAX accept [finite](../../../../glossary.md#term-finite) `[-MAX,+MAX]`, publish `[+0,1]` / `[-infinity,+0]`
and produce the ordinary finite/zero actual backward values. CPU [F64](../../../../glossary.md#floating-point-precision-and-rounding) rejects both at the extra
`value-maximum` finite-shift pass. Parent retained `softmax-parity-before.log` and unchanged
`SoftmaxParitySmoke.java`; do not rerun merely to confirm that observed rejection.

[Model](../../../../glossary.md#term-model) already defines negative-[infinity](../../../../glossary.md#term-infinity) working shifts from finite subtraction [overflow](../../../../glossary.md#term-overflow), [EXP](../../../../glossary.md#term-expx)
[positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and negative-infinity [log](../../../../glossary.md#term-log) probability. The generated typed implementation exists;
CPU's extra F64 [validation](../../../../glossary.md#term-validation) narrows it unnecessarily. Max is one finite represented contributor,
so one working shift is zero, every other is nonpositive, shifted EXP is in [0,1], and the
[all-contributors-once](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) sum is positive/finite. This is not unshifted EXP or a new approximation.

## Bounded change and non-goals

Admit all [finite](../../../../glossary.md#term-finite) [F64](../../../../glossary.md#floating-point-precision-and-rounding) inputs like [F32](../../../../glossary.md#floating-point-precision-and-rounding), including negative-[infinity](../../../../glossary.md#term-infinity) working shifts. Reuse existing
finite input [validation](../../../../glossary.md#term-validation) and unchanged stable emitters/backward rules. Remove the now-dead F64
maximum/second shift-validation traversal; keep checked addresses and validation before writes.
Preserve nonfinite-input rejection and complete active-domain/failure ordering. No special-case
input tuple, swallowed exception, result substitution or [tolerance](../../../../glossary.md#term-tolerance)/policy object.

Do not expand [BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[FP16](../../../../glossary.md#term-fp16) capability or alter retained BF16 shift rules. No new production
[API](../../../../glossary.md#term-application-programming-interface-api)/type/package/[dependency](../../../../glossary.md#term-dependency), cast, [materialization](../../../../glossary.md#materialization), fallback, numerical mode or [kernel](../../../../glossary.md#kernel). Generated
bytes/[ABI](../../../../glossary.md#term-application-binary-interface-abi) are unchanged; do not arbitrarily bump the global generator [schema](../../../../glossary.md#term-schema). Review whether
any actual cached [admission](../../../../glossary.md#term-admission) fact needs separate invalidation; current validators [run](../../../../glossary.md#term-run) at invocation.
The independent review found the matching obsolete F64 shift rejection in
CpuSoftmaxReferenceKernel.evaluateDouble. Synchronize that [oracle](../../../../glossary.md#term-test-oracle)'s domain/[Javadoc](../../../../glossary.md#term-javadoc) without
changing its arithmetic, compensation, traversal or BF16 rules. Update CpuSoftmaxEmitter's
domain Javadoc only. There is no cached successful value-domain fact: unchanged generated classes
always run with the current invocation validator, so all algorithm/[identity](../../../../glossary.md#term-identity-meanings)/ABI versions stay.

## Contracts

- [Root invariants](../../../../../ARCHITECTURE.md#core-invariants): one numerical semantics,
  backend-owned hard eligibility and honest unsupported domains.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [Model finite-softmax semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [finite glossary](../../../../glossary.md#term-finite); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>:
  shifted sites and overflow result are already defined; the CPU owner below owns narrower admission.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner); [admission glossary](../../../../glossary.md#term-admission))</span>
- [CPU0077 admission](../../../../architecture/contracts/backend-execution.md#cpu0077-finite-softmax-admission-parity)
  and [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  accepted exact correction, unchanged runtime validation/publication/resource ownership.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- [Production target](../../../production-release-goal.md#java-cpu-target)<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span> and reviewed
  [support matrix](../../../../backend-guide/production-support.md)<span data-glossary-navigation="true"> ([support matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span> require equal [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) domains.

## Dependencies and integration

- Depends on: NN0033 Complete, existing stable-[softmax](../../../../glossary.md#softmax--log-softmax)/actual backward baseline, reviewed [scope](../../../../glossary.md#term-scope).
- Conflicts with: CpuSoftmaxInputValidator, CpuSoftmaxReferenceKernel and CpuSoftmaxEmitter Javadoc;
  InputValidator/Reference/StableReductionSoftmaxStageA/SoftmaxFiniteConformance and actual Engine
  backward/loss consumer tests. These files are exclusively CPU0077; low-lock successor rebases.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Engine glossary](../../../../glossary.md#term-engine); [loss glossary](../../../../glossary.md#loss))</span>
- Parallel group: None; disjoint read-only successor review is allowed, no other [CPU](../../../../glossary.md#term-cpu) source writer.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus MAIN's exact post-freeze
  source/authority manifest (<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>`cpu0077-base.json`) in the recorded isolated worktree.
- Integration order: authority/scope approval -> isolated code/review -> unchanged public after
  reproducer and affected validator/CPU/conformance/consumer gates -> later final release join.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [release glossary](../../../../glossary.md#term-release))</span>
- Integration validation: targeted finite-softmax tests/Javadoc and actual public both-dtype
  forward/Compiler-backward, then source-matched final repository checkpoint.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [finite glossary](../../../../glossary.md#term-finite); [dtype glossary](../../../../glossary.md#data-type--datatype); [Compiler glossary](../../../../glossary.md#term-compiler); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

## Files and symbols

- Existing CpuSoftmaxInputValidator.bind/validateDouble, CpuSoftmaxReferenceKernel.evaluateDouble
  and domain Javadocs; CpuSoftmaxEmitter Javadoc only.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Existing CpuSoftmaxInputValidatorTest, CpuSoftmaxReferenceTest,
  CpuStableReductionSoftmaxStageATest, CpuSoftmaxFiniteConformanceTest and actual Engine
  backward/loss consumer tests. Add a focused behavioral integration class if needed.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [loss glossary](../../../../glossary.md#loss))</span>
- Generated softmax/loss/backward fusion executable bodies remain unchanged. A demonstrated
  dependency requiring a change stops for MAIN review before widening scope.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [fusion glossary](../../../../glossary.md#term-fusion); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening); [scope glossary](../../../../glossary.md#term-scope))</span>
- MAIN owns root/scoped/[CPU](../../../../glossary.md#term-cpu) guide/[matrix](../../../../glossary.md#term-matrix-meanings)/changelog/result updates.

## Acceptance criteria

- Both kinds/[dtypes](../../../../glossary.md#data-type--datatype): `[-MAX,+MAX,+MAX]` gives `[+0,0.5,0.5]` /
  `[-infinity,-log(2),-log(2)]`; a selected maximal component gives a finite objective with
  nonzero actual gradients. Preserve ordinary derivative signed-zero behavior.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [objective glossary](../../../../glossary.md#term-objective); [gradients glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Ordinary finite controls, tied maxima, width one, non-last axis, offsets/strides, heap/segment
  carriers and empty non-selected dimensions retain their current semantics.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [strides glossary](../../../../glossary.md#term-stride); [heap glossary](../../../../glossary.md#term-heap); [dimensions glossary](../../../../glossary.md#dimension))</span>
- All nonfinite inputs still reject before consumer writes/submission, including invalid final
  values in later slices and negative infinity produced upstream; prove parallel failure atomicity.<span data-glossary-navigation="true"> ([negative infinity glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
- Bind once, change values valid -> invalid -> valid, and verify current values are rescanned.
- Dense/index cross-entropy actual backward at finite extremes preserves zero-target and
  ignore-index guards; the formulas do not change.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Retain current low behavior without expansion; do not turn a low lock into a domain shortcut.
- Existing softmax saved-output backward/fusion and relevant loss consumers retain their own
  guards/algorithm; no cross-entropy or attention formula is changed by analogy.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [fusion glossary](../../../../glossary.md#term-fusion); [loss glossary](../../../../glossary.md#loss))</span>
- New permanent tests catch consumer-visible value-domain or failure-atomicity defects, not
  diagnostic wording/source text/wiring. Delete incidental pins, never update message snapshots.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity))</span>
- The independent review explicitly confirms no changed arithmetic sites or unnecessary work.

## [Validation](../../../../glossary.md#term-validation) and documentation

No build/lint/tests/formatters mid-flight. MAIN runs targeted gates once corrected sources and
independent review stabilize, then the unchanged public after [smoke](../../../../glossary.md#term-smoke-test). Record exact skips and
limits; do not relabel historical failures. Public [CPU](../../../../glossary.md#term-cpu) domain text must match implementation.

MAIN [Ready](../../../../glossary.md#term-ready) verification: authorized CPU source frontier 0077; all predecessors and scoped design
approval satisfied; no active writer conflicts. No new [dependency](../../../../glossary.md#term-dependency)/[ownership](../../../../glossary.md#term-ownership-meanings)/algorithm decision
needs a separate [ADR](../../../../glossary.md#term-architecture-decision-record-adr); coordinated numerical authority plus [conformance](../../../../glossary.md#term-conformance)/integration validation is
the applicable Class C record. Independent source/[Javadoc](../../../../glossary.md#term-javadoc) review precedes the coordinator gate.

## Result

Status: Complete.

Eight declared existing source/test files integrated against cpu0077-base.json with no unrelated
changes. Independent final source/[Javadoc](../../../../glossary.md#term-javadoc)/test-design APPROVE, zero P1/P2: removed obsolete [F64](../../../../glossary.md#floating-point-precision-and-rounding)
shift-domain checks/dead production traversal, synchronized the reference [admission](../../../../glossary.md#term-admission), retained all
represented-input/address/prewrite checks and low behavior. Emitter executable body unchanged.

MAIN's final gate passed: 45 tests in six [CPU](../../../../glossary.md#term-cpu), two [conformance](../../../../glossary.md#term-conformance) and one [Engine](../../../../glossary.md#term-engine) integration suite,
zero failures/errors/skips; CPU Javadoc/Jar passed. Covers tied extremes/nonzero actual [gradients](../../../../glossary.md#term-gradient),
[axis](../../../../glossary.md#term-axis)/[stride](../../../../glossary.md#term-stride)/offset/[carrier](../../../../glossary.md#term-storage-carrier) controls, current-value rescanning, later-invalid parallel [failure
atomicity](../../../../glossary.md#term-failure-atomicity), produced [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) and actual dense/index cross-entropy backward guards.
The unchanged actual public after program passed [F32](../../../../glossary.md#floating-point-precision-and-rounding)/F64 x [SOFTMAX](../../../../glossary.md#softmax--log-softmax)/LOG_SOFTMAX forward/[Compiler](../../../../glossary.md#term-compiler)
backward without [Vector](../../../../glossary.md#term-vector)-[module](../../../../glossary.md#term-module) flags; original F64 rejection remains in before evidence.

Evidence: cpu0077-integrated-files.json, cpu0077-gate-counts.json/XML and
softmax-parity-before.[log](../../../../glossary.md#term-log)/softmax-parity-after.log in the coordinator [snapshot](../../../../glossary.md#term-snapshot); [Gradle](../../../../glossary.md#term-gradle) artifact130.
The completed throwaway Java program was removed. This closes the bounded parity blocker, not
wider CPU/[Metal](../../../../glossary.md#term-metal) [qualification](../../../../glossary.md#term-qualification), resource gates or final [release](../../../../glossary.md#term-release)/root readiness.

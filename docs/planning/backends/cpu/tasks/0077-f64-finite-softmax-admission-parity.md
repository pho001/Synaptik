# Task 0077: F64 finite-softmax admission parity

## Status

Complete — independently reviewed and MAIN accepted after source-matched targeted gates and the
unchanged public after scenario. F64 now shares F32's finite-input domain; no mathematical emitter,
ABI, algorithm version or cache identity changed.

## Change class

Class C — value-dependent admission, numerical-domain and pre-write failure guarantees.
Clean implementer and independent code/documentation review; no new mathematical algorithm.

## Goal and observed result

CPU F32 SOFTMAX and LOG_SOFTMAX accept finite `[-MAX,+MAX]`, publish `[+0,1]` / `[-infinity,+0]`
and produce the ordinary finite/zero actual backward values. CPU F64 rejects both at the extra
`value-maximum` finite-shift pass. Parent retained `softmax-parity-before.log` and unchanged
`SoftmaxParitySmoke.java`; do not rerun merely to confirm that observed rejection.

Model already defines negative-infinity working shifts from finite subtraction overflow, EXP
positive zero and negative-infinity log probability. The generated typed implementation exists;
CPU's extra F64 validation narrows it unnecessarily. Max is one finite represented contributor,
so one working shift is zero, every other is nonpositive, shifted EXP is in [0,1], and the
all-contributors-once sum is positive/finite. This is not unshifted EXP or a new approximation.

## Bounded change and non-goals

Admit all finite F64 inputs like F32, including negative-infinity working shifts. Reuse existing
finite input validation and unchanged stable emitters/backward rules. Remove the now-dead F64
maximum/second shift-validation traversal; keep checked addresses and validation before writes.
Preserve nonfinite-input rejection and complete active-domain/failure ordering. No special-case
input tuple, swallowed exception, result substitution or tolerance/policy object.

Do not expand BF16/FP16 capability or alter retained BF16 shift rules. No new production
API/type/package/dependency, cast, materialization, fallback, numerical mode or kernel. Generated
bytes/ABI are unchanged; do not arbitrarily bump the global generator schema. Review whether
any actual cached admission fact needs separate invalidation; current validators run at invocation.
The independent review found the matching obsolete F64 shift rejection in
CpuSoftmaxReferenceKernel.evaluateDouble. Synchronize that oracle's domain/Javadoc without
changing its arithmetic, compensation, traversal or BF16 rules. Update CpuSoftmaxEmitter's
domain Javadoc only. There is no cached successful value-domain fact: unchanged generated classes
always run with the current invocation validator, so all algorithm/identity/ABI versions stay.

## Contracts

- [Root invariants](../../../../../ARCHITECTURE.md#core-invariants): one numerical semantics,
  backend-owned hard eligibility and honest unsupported domains.
- [Model finite-softmax semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics):
  shifted sites and overflow result are already defined; the CPU owner below owns narrower admission.
- [CPU0077 admission](../../../../architecture/contracts/backend-execution.md#cpu0077-finite-softmax-admission-parity)
  and [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  accepted exact correction, unchanged runtime validation/publication/resource ownership.
- [Production target](../../../production-release-goal.md#java-cpu-target) and reviewed
  [support matrix](../../../../backend-guide/production-support.md) require equal F32/F64 domains.

## Dependencies and integration

- Depends on: NN0033 Complete, existing stable-softmax/actual backward baseline, reviewed scope.
- Conflicts with: CpuSoftmaxInputValidator, CpuSoftmaxReferenceKernel and CpuSoftmaxEmitter Javadoc;
  InputValidator/Reference/StableReductionSoftmaxStageA/SoftmaxFiniteConformance and actual Engine
  backward/loss consumer tests. These files are exclusively CPU0077; low-lock successor rebases.
- Parallel group: None; disjoint read-only successor review is allowed, no other CPU source writer.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus MAIN's exact post-freeze
  source/authority manifest (`cpu0077-base.json`) in the recorded isolated worktree.
- Integration order: authority/scope approval -> isolated code/review -> unchanged public after
  reproducer and affected validator/CPU/conformance/consumer gates -> later final release join.
- Integration validation: targeted finite-softmax tests/Javadoc and actual public both-dtype
  forward/Compiler-backward, then source-matched final repository checkpoint.
- Shared-document integration owner: MAIN.

## Files and symbols

- Existing CpuSoftmaxInputValidator.bind/validateDouble, CpuSoftmaxReferenceKernel.evaluateDouble
  and domain Javadocs; CpuSoftmaxEmitter Javadoc only.
- Existing CpuSoftmaxInputValidatorTest, CpuSoftmaxReferenceTest,
  CpuStableReductionSoftmaxStageATest, CpuSoftmaxFiniteConformanceTest and actual Engine
  backward/loss consumer tests. Add a focused behavioral integration class if needed.
- Generated softmax/loss/backward fusion executable bodies remain unchanged. A demonstrated
  dependency requiring a change stops for MAIN review before widening scope.
- MAIN owns root/scoped/CPU guide/matrix/changelog/result updates.

## Acceptance criteria

- Both kinds/dtypes: `[-MAX,+MAX,+MAX]` gives `[+0,0.5,0.5]` /
  `[-infinity,-log(2),-log(2)]`; a selected maximal component gives a finite objective with
  nonzero actual gradients. Preserve ordinary derivative signed-zero behavior.
- Ordinary finite controls, tied maxima, width one, non-last axis, offsets/strides, heap/segment
  carriers and empty non-selected dimensions retain their current semantics.
- All nonfinite inputs still reject before consumer writes/submission, including invalid final
  values in later slices and negative infinity produced upstream; prove parallel failure atomicity.
- Bind once, change values valid -> invalid -> valid, and verify current values are rescanned.
- Dense/index cross-entropy actual backward at finite extremes preserves zero-target and
  ignore-index guards; the formulas do not change.
- Retain current low behavior without expansion; do not turn a low lock into a domain shortcut.
- Existing softmax saved-output backward/fusion and relevant loss consumers retain their own
  guards/algorithm; no cross-entropy or attention formula is changed by analogy.
- New permanent tests catch consumer-visible value-domain or failure-atomicity defects, not
  diagnostic wording/source text/wiring. Delete incidental pins, never update message snapshots.
- The independent review explicitly confirms no changed arithmetic sites or unnecessary work.

## Validation and documentation

No build/lint/tests/formatters mid-flight. MAIN runs targeted gates once corrected sources and
independent review stabilize, then the unchanged public after smoke. Record exact skips and
limits; do not relabel historical failures. Public CPU domain text must match implementation.

MAIN Ready verification: authorized CPU source frontier 0077; all predecessors and scoped design
approval satisfied; no active writer conflicts. No new dependency/ownership/algorithm decision
needs a separate ADR; coordinated numerical authority plus conformance/integration validation is
the applicable Class C record. Independent source/Javadoc review precedes the coordinator gate.

## Result

Status: Complete.

Eight declared existing source/test files integrated against cpu0077-base.json with no unrelated
changes. Independent final source/Javadoc/test-design APPROVE, zero P1/P2: removed obsolete F64
shift-domain checks/dead production traversal, synchronized the reference admission, retained all
represented-input/address/prewrite checks and low behavior. Emitter executable body unchanged.

MAIN's final gate passed: 45 tests in six CPU, two conformance and one Engine integration suite,
zero failures/errors/skips; CPU Javadoc/Jar passed. Covers tied extremes/nonzero actual gradients,
axis/stride/offset/carrier controls, current-value rescanning, later-invalid parallel failure
atomicity, produced infinities and actual dense/index cross-entropy backward guards.
The unchanged actual public after program passed F32/F64 x SOFTMAX/LOG_SOFTMAX forward/Compiler
backward without Vector-module flags; original F64 rejection remains in before evidence.

Evidence: cpu0077-integrated-files.json, cpu0077-gate-counts.json/XML and
softmax-parity-before.log/softmax-parity-after.log in the coordinator snapshot; Gradle artifact130.
The completed throwaway Java program was removed. This closes the bounded parity blocker, not
wider CPU/Metal qualification, resource gates or final release/root readiness.

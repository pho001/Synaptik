# [CPU](../../../../glossary.md#term-cpu) 0021 — Repair composed value [validation](../../../../glossary.md#term-validation) and [loss](../../../../glossary.md#loss) [bindings](../../../../glossary.md#term-binding)

## Status

Complete

## Goal and class

Class C [CPU](../../../../glossary.md#term-cpu) preparation/execution correctness under existing capability and [lifecycle](../../../../glossary.md#lifecycle) contracts.
Close project-review [C2](../../../../glossary.md#term-c2)/C3/C4 without [widening](../../../../glossary.md#term-widening) [operation](../../../../glossary.md#operation) domains or changing shared [Runtime](../../../../glossary.md#term-runtime).

## Confirmed defects and owned [scope](../../../../glossary.md#term-scope)

- `CpuSoftmaxInputValidator` sends FLOAT16 through its wide reader: short arrays fail a cast and
  native short storage is read as FLOAT32. Add exact F16 decode and the declared working precision.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [FLOAT32 glossary](../../../../glossary.md#term-float32); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [precision glossary](../../../../glossary.md#term-precision))</span>
- `CpuPreparedExecutable` scans softmax/loss/mask values while all partition children are bound,
  before earlier units produce intermediates. Retain metadata/carrier/span checks in binding;
  move required value validation to invocation before consumer writes, after producers. Inspect
  canonical-BOOL and attention-mask checks too; retain cold-prebound validation mechanics and
  avoid per-element dispatch/allocation. Index/scatter execution validation is an existing seam.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [mask glossary](../../../../glossary.md#term-boolean-mask); [partition glossary](../../../../glossary.md#partition); [metadata glossary](../../../../glossary.md#term-metadata); [carrier glossary](../../../../glossary.md#term-storage-carrier); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `CpuPartitionPreparationPlan.unitAccessesAgree` compares loss bindings with rank-zero generated
  cache projections. Use actual occurrence <span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`CpuLossIr.boundaryAccesses` while preserving [cache](../../../../glossary.md#term-cache) [IR](../../../../glossary.md#term-intermediate-representation-ir).

Owned [CPU](../../../../glossary.md#term-cpu) files are those classes and smallest related bound-call/validator [owners](../../../../glossary.md#term-owner) plus focused
CPU tests. Add [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/public CPU [Engine](../../../../glossary.md#term-engine) integration tests in new focused files when
needed. Do not edit Engine mixed-[Metal](../../../../glossary.md#term-metal) [fixture](../../../../glossary.md#term-fixture) or [Compiler](../../../../glossary.md#term-compiler) tests owned by parallel repair tasks.
Review CPU [softmax](../../../../glossary.md#softmax--log-softmax) guide: its [finite](../../../../glossary.md#term-finite)-only route is a deliberate limitation, not undefined [Model](../../../../glossary.md#term-model)
special-value semantics. No unrelated reference-[kernel](../../../../glossary.md#kernel) or [activation](../../../../glossary.md#term-activation)-emitter change.

## Required reading

AGENTS; root [scope](../../../../glossary.md#term-scope) index, Core invariants, [ownership](../../../../glossary.md#term-ownership-meanings)/[dependency](../../../../glossary.md#term-dependency) rules and Testing; scoped
`backend-execution.md` headings [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module), [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes, Profile-free [backend
capability](../../../../glossary.md#backend-capability) and [identity](../../../../glossary.md#term-identity-meanings); `runtime-prepare-engine.md` [Run](../../../../glossary.md#term-run) [lifecycle](../../../../glossary.md#lifecycle) and modules/[runtime](../../../../glossary.md#term-runtime); [Model](../../../../glossary.md#term-model)
SoftmaxKind/LossKind and [mask](../../../../glossary.md#term-boolean-mask) contracts; affected [lowering](../../../../glossary.md#lowering)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/[binding](../../../../glossary.md#term-binding) and focused tests.

## Acceptance and [validation](../../../../glossary.md#term-validation)

- FLOAT16 SOFTMAX/LOG_SOFTMAX execute valid nonempty heap/native inputs with offsets and strided
  geometry. Provider→prepare→execute tests must not only call the decoder directly.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [heap glossary](../../../../glossary.md#term-heap); [prepare glossary](../../../../glossary.md#prepare))</span>
- `[0,1000].exp().softmax(0)` rejects the produced [infinity](../../../../glossary.md#term-infinity) just as direct `[1,+Inf]` does;
  consumer output remains untouched. Cover computed index-loss targets/masks and repeated runs
  so stale workspace contents cannot stand in for producer values.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [targets glossary](../../../../glossary.md#term-target); [masks glossary](../../../../glossary.md#term-boolean-mask); [producer glossary](../../../../glossary.md#term-producer))</span>
- Composed non-scalar MSE and categorical losses prepare/run correctly, including strided
  boundaries; retain rejection of forged incompatible bindings and generated-cache projection.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [MSE glossary](../../../../glossary.md#mean-squared-error--mse); [run glossary](../../../../glossary.md#term-run); [bindings glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache))</span>
- Validation occurs once on the orchestrator before worker submission/output mutation, not in
  element loops, with no new hot-path allocation or Runtime policy.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Focused regressions then one CPU test/Javadoc run; selected conformance/integration tests.
  No historical performance/five-fork matrix or full repository run in worker.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [fork glossary](../../../../glossary.md#term-fork-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Review/update affected Javadoc and explanation. Independent Class C review checks source,
  placement, domain preservation and semantic/conformance evidence.
- [Commit](../../../../glossary.md#term-commit) scoped files, no push/plans; report files/tests/doc impact and completion status.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0019/0020 Complete; project-review [C2](../../../../glossary.md#term-c2)/C3/C4.
- Conflicts with: CPU [prepare](../../../../glossary.md#prepare)/executable/validator writers and CPU 0022 implementation.
- Parallel group: PROJECT-REVIEW-2026-10-13-REPAIRS (other [module](../../../../glossary.md#term-module) [owners](../../../../glossary.md#term-owner) only).
- Common base revision: production `728680ea` plus [planning](../../../../glossary.md#term-planning)-only briefs.
- Integration order: 0021 before 0022; otherwise independent of module repairs.
- Integration validation: focused/module evidence, independent review, combined repository
  checkpoint after ordered integration.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): coordinator; worker owns focused CPU explanation only.

Launch verified 2026-10-13: diagnostic counterexamples and source agree; current contracts suffice;
no CPU writer is active. User-authorized repair, complete predecessors, clean context/worktree.

## Integration record

Implementation `ca21c80c` integrated as `c0559e2b` after independent Class C APPROVE and the
coordinator's [softmax](../../../../glossary.md#softmax--log-softmax) glossary correction. [CPU](../../../../glossary.md#term-cpu): 1,026 passed / 28 optional skips, no failures;
two focused [conformance](../../../../glossary.md#term-conformance) and six public integration tests passed. [Javadoc](../../../../glossary.md#term-javadoc) passed with 94
pre-existing warnings; diff checks passed. [Metadata](../../../../glossary.md#term-metadata)/value-[scan](../../../../glossary.md#term-cumulative-scan) timing and [materialization](../../../../glossary.md#materialization) order
were independently inspected. No unresolved code finding; the report records the bounded [mask](../../../../glossary.md#term-boolean-mask)
and materialized-[BOOL](../../../../glossary.md#data-type--datatype) sentinel coverage limitations. The group's combined [checkpoint](../../../../glossary.md#term-checkpoint) passed
after serial 0022; successful worker evidence was reused.

Completion checkpoint: [integrated review evidence](../../../project-review-2026-10-13-findings.md#final-integrated-checkpoint).
Required independent code/Javadoc/documentation review and the combined `test javadoc` passed.
No unresolved finding within this repair's [scope](../../../../glossary.md#term-scope).

Status: Complete

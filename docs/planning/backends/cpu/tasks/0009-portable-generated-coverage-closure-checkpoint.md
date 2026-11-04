# Task 0009: Portable [CPU](../../../../glossary.md#term-cpu) Coverage and Implementation-Closure [Checkpoint](../../../../glossary.md#term-checkpoint)

## Status

Complete

## Goal

Close portable [CPU](../../../../glossary.md#term-cpu) coverage with the least costly maintainable implementation for each family:
direct optimized typed Java where the algorithm is static, bounded generated code where [fusion](../../../../glossary.md#term-fusion) or
specialized compute justifies it, and exact rejection where no route is supported. This supersedes
the unfinished program of universal generated-versus-clean-Java structural closure.

## [Scope](../../../../glossary.md#term-scope)

Completed 0009A--C/D1/D1A/D2/D3/D4 remain historical completed evidence; they are not migration
[targets](../../../../glossary.md#term-target) merely because a later family adopts direct Java. D4 and E1 completed retained
generated-route decisions, so parent 0009D is Complete and E1A [masked reduction](../../../../glossary.md#masked-reduction) is Complete.
The 0009E parent remains a master-plan summary. E1B is Complete: its three material numerical
[owners](../../../../glossary.md#term-owner), E1B1 [log-sum-exp](../../../../glossary.md#log-sum-exp), E1B2 corrected statistics, and E1B3 norms, retain bounded generated
execution because direct replacement would increase implementation and verification work. E1C
[softmax](../../../../glossary.md#softmax--log-softmax)-style and E2 [normalization](../../../../glossary.md#term-normalization) are Complete retained generated routes. E3 separately retains
the bounded generated [MSE](../../../../glossary.md#mean-squared-error--mse), dense categorical, and index categorical [loss](../../../../glossary.md#loss) routes after complete
non-performance cost decisions. Parent 0009F, including F1, F2, and F3, is Complete. 0009G
completed the final support, correctness, hygiene, and inventory accounting [checkpoint](../../../../glossary.md#term-checkpoint), closing
[CPU](../../../../glossary.md#term-cpu) 0009 without a universal strategy, structural-[oracle](../../../../glossary.md#term-test-oracle), or performance claim.

The default classification, subject to the active family's code review, is:

| Family | Planned implementation direction |
|---|---|
| Bounded [fused](../../../../glossary.md#specialization-and-fusion) [pointwise](../../../../glossary.md#term-pointwise) DAGs and genuinely dynamic fused/specialized compositions | Retain bounded generation. |
| Dense [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), Conv inner loops, and similarly compute-intensive integral generation | Retain generation when it remains the simpler specialized route. |
| Aggregate | Retain the current supported generated route: its exact numerical [matrix](../../../../glossary.md#term-matrix-meanings) is not a cheap static-loop migration. |
| [Scan](../../../../glossary.md#term-cumulative-scan), ordering, fold | Use a route-decision gate; prefer [finite](../../../../glossary.md#term-finite) direct typed Java only where it is genuinely cheaper to implement and verify. |
| General reductions, normalizations, loss | Begin with bounded 0009E1 partial integral route decision; masked, advanced, softmax, normalization, and loss remain ordered summaries. |
| [Pooling](../../../../glossary.md#term-pooling), attention, [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [norm](../../../../glossary.md#term-norm) | Decide per family in 0009F: migrate orchestration/static loops; retain an already-good generated compute [kernel](../../../../glossary.md#kernel) when replacement increases work. |

For every direct route, [prepare](../../../../glossary.md#prepare) cold-selects one typed array/segment, dense/general, [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector)
where useful, and range-owning entry. No generic per-[element](../../../../glossary.md#term-element) [runtime](../../../../glossary.md#term-runtime) [dispatch](../../../../glossary.md#term-dispatch) is allowed. Existing
clean-Java algorithms are the first reference. `legacy/pre-rewrite` is read-only supplemental
evidence for capability, observable behavior, algorithms, and tests; source copying, legacy
packages/dependencies, runtime coupling, and shortcuts remain forbidden.

## Out of [scope](../../../../glossary.md#term-scope)

Architecture/[module](../../../../glossary.md#term-module)-boundary changes, a universal structural verifier, mandatory generated-vs-Java
performance studies, a benchmark ratio gate for migration, and a big-bang generated-code rewrite.

## Architecture references and constraints

`ARCHITECTURE.md` remains authoritative: [CPU](../../../../glossary.md#term-cpu) [prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering) and [kernel](../../../../glossary.md#kernel) selection; [Runtime](../../../../glossary.md#term-runtime)
executes prepared work; parallelism is outside [element](../../../../glossary.md#term-element) loops; prepared recipes are [immutable](../../../../glossary.md#term-immutable);
direct and retained generated loops receive proportionate hygiene review. This plan changes neither
that contract nor CPU [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings).

## Acceptance criteria

- Every advertised [CPU](../../../../glossary.md#term-cpu) form is direct Java, retained bounded generated code, or an exact rejection.
- Every supported form has semantic and invocation coverage; direct and retained hot paths have
  proportionate source/call-site hygiene for allocation, boxing, reflection, string/map dispatch,
  synchronization, and per-element virtual dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [synchronization glossary](../../../../glossary.md#term-synchronization); [element glossary](../../../../glossary.md#term-element))</span>
- Generated implementation, schema/inventory/evidence, and structural scaffolding are removed only
  after its direct replacement passes semantics/conformance and no selected route references it.
  No permanent parallel implementation survives without an explicit reason.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Existing performance evidence and known NON_PASSING facts remain reported. Fresh representative
  profiling is optional/final and never blocks correctness closure unless it exposes a concrete
  correctness regression.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- There is no requirement for universal generated-vs-clean structural equivalence or exhaustive
  new performance evidence.

## Tests / [validation](../../../../glossary.md#term-validation)

Each active migration runs focused [lowering](../../../../glossary.md#lowering), [prepare](../../../../glossary.md#prepare), invocation, semantics, [canary](../../../../glossary.md#term-canary), and retirement
tests plus a proportionate hot-loop inspection. The final [checkpoint](../../../../glossary.md#term-checkpoint) verifies support accounting,
exact rejections, selected routes, stale generated evidence removal, documentation, and hygiene.
It reports existing performance facts but does not require fresh benchmark closure. This parent
[planning](../../../../glossary.md#term-planning) revision validates Markdown/status consistency and `git diff --check`; it runs no [Gradle](../../../../glossary.md#term-gradle).

## Dependencies and follow-up tasks

- 0009D is Complete: D1 aggregate, D2 scan, D3 ordering, and D4 fold are retained-generated-route
  decisions. D4 retains all 32 fold rows because a direct replacement would cost more to implement
  and verify.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- 0009E1 is Complete: it retained the finite typed generated partial integral route because direct
  migration would increase selection, binding, replacement, and retirement-proof work without a
  performance gate. E1A is Complete: it retained the complete generated FLOAT64/FLOAT32/BFLOAT16
  masked SUM/MEAN route because direct Java would duplicate typed carrier/layout, directional
  mask, selected exact-state, binding, invocation, inventory, and retirement work. E1B1 is
  Complete as a retained generated log-sum-exp route, E1B2 is Complete as a retained generated
  corrected statistical route, and E1B3 is Complete as a retained generated L1/L2 norm route.
  E1B, E1C, E2, and E3 are Complete retained-generated-route decisions. Parent 0009F is
  Complete, including F1, F2, and F3.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [binding glossary](../../../../glossary.md#term-binding); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [MEAN glossary](../../../../glossary.md#term-mean); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [mask glossary](../../../../glossary.md#term-boolean-mask); [log-sum-exp glossary](../../../../glossary.md#log-sum-exp); [L1 glossary](../../../../glossary.md#term-l1); [L2 norm glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions))</span>
- 0009F: Complete parent for per-family hybrid decisions covering MATMUL/convolution, pooling,
  attention, and batch normalization. <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>[F1](0009f1-matmul-and-convolution-route-decisions.md),
  [F2 pooling](0009f2-pooling-route-decisions.md), and [F3 attention and BatchNorm](0009f3-attention-and-batch-normalization-route-decisions.md) are Complete.
- 0009G is Complete: its [final support, correctness, hygiene, and inventory checkpoint](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  reconciles all 120 live operation forms with the readable ledger and records selected strategy
  facts only where they are meaningful.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>

## Architecture impact

Expected impact: None. Stop without edits if inspection proves a conflict with the authoritative
contract.

## Implementation prompt

```text
Work only on the current detailed child. Choose direct typed Java for the static family unless a
bounded generated route is demonstrably integral and cheaper to retain. Keep selection cold and
range ownership outside element loops. Use legacy only as read-only behavioral/test evidence; never
copy its implementation. Retire generated routes only after direct replacement evidence passes.
No universal structural verifier or performance gate. Do not commit or push.
```

## Local decisions

The [canonical](../../../../glossary.md#term-canonical-form) generated inventory and completed evidence remain useful historical accounting, but
they are no longer a requirement to prove every generated row against a clean-Java classfile.

## Known limitations

[CPU](../../../../glossary.md#term-cpu) 0008I [loss](../../../../glossary.md#loss) [fork](../../../../glossary.md#term-fork-meanings) 0 remains NON_PASSING (19 of 792 above 1.15; worst
1.3861164205039096); user-closed forks 1--4 remain historical facts. No result is promoted by this
[planning](../../../../glossary.md#term-planning) revision.

## [Validation](../../../../glossary.md#term-validation) evidence

D4's specified eight-class command passed on 2026-09-09: 131 tests, zero failures, errors, or
skips. E1's specified six-class command passed on 2026-09-09: 110 tests, zero failures, errors, or
skips. E1 retained its generated partial/combine implementation after source and test inspection,
not performance evidence; it made no Java change. Its clean documentation-focused finalization
reused that evidence, verified the current unselected generated candidate and whole-cell fallback,
and synchronized the four [planning](../../../../glossary.md#term-planning) records without rerunning Java. E1B3's implementation context
ran its recorded 13-class focused command on 2026-09-09; the XML reports record 168 tests, zero
failures or errors, and one expected opt-in [persistence](../../../../glossary.md#term-persistence)-evidence skip. Its documentation-focused
finalization reused that evidence because no executable Java changed, inspected the [norm](../../../../glossary.md#term-norm) route and
its evidence [owners](../../../../glossary.md#term-owner), and synchronized the four planning records without [benchmarking](../../../../glossary.md#term-benchmarking) or running a
repository-wide suite.

0009G's specified focused command passed after its final test/resource edits;
`:backends:cpu:test` passed with 931 tests, 28 skipped, and zero failures or errors,
`:backends:cpu:javadoc` passed, and root `test` passed on 2026-09-10. Its documentation-focused
pass made Markdown-only status/evidence edits and reused those results.

## Implementation notes

None.

## Completion summary

[CPU](../../../../glossary.md#term-cpu) 0009 is Complete. Parent 0009D is Complete: D1 through D4 are
retained-generated-route decisions; the D4 fold result preserves the full 32-row generated family
because a complete direct Java replacement would increase total implementation and verification
work. CPU 0009E1 and E1A are Complete retained-generated-route decisions; E1A retains the
complete masked SUM/[MEAN](../../../../glossary.md#term-mean) route because a direct replacement would increase implementation and
verification work. E1B is Complete: E1B1 retains generated [log-sum-exp](../../../../glossary.md#log-sum-exp), E1B2 retains generated
corrected statistics, and E1B3 retains generated [L1](../../../../glossary.md#term-l1)/[L2](../../../../glossary.md#term-l2) norms. E1C retains the complete stable
[SOFTMAX](../../../../glossary.md#softmax--log-softmax)/LOG_SOFTMAX generated route. E2 retains the complete static Layer/RMS generated route;
E3 separately retains the complete bounded generated [MSE](../../../../glossary.md#mean-squared-error--mse), dense categorical, and index
categorical routes. Parent F and F1--F3 are Complete. 0009G reconciles every live [operation](../../../../glossary.md#operation) form
with the readable ledger, exact generated/rejected accounting, meaningful selected strategies,
and existing evidence [owners](../../../../glossary.md#term-owner). It does not promote partial structural or performance evidence.

Status: Complete

# Task 0009: Portable CPU Coverage and Implementation-Closure Checkpoint

## Status

Ready

## Goal

Close portable CPU coverage with the least costly maintainable implementation for each family:
direct optimized typed Java where the algorithm is static, bounded generated code where fusion or
specialized compute justifies it, and exact rejection where no route is supported. This supersedes
the unfinished program of universal generated-versus-clean-Java structural closure.

## Scope

Completed 0009A--C/D1/D1A/D2/D3/D4 remain historical completed evidence; they are not migration
targets merely because a later family adopts direct Java. D4 and E1 completed retained
generated-route decisions, so parent 0009D is Complete and E1A masked reduction is Complete.
The 0009E parent remains a master-plan summary. E1B is split only by material numerical ownership:
E1B1 is Complete as a retained generated log-sum-exp route; E1B2 statistics is the sole next
Ready detailed child, while E1B3 norms and E1C--E3 remain Draft
summaries.

The default classification, subject to the active family's code review, is:

| Family | Planned implementation direction |
|---|---|
| Bounded fused pointwise DAGs and genuinely dynamic fused/specialized compositions | Retain bounded generation. |
| Dense MATMUL, Conv inner loops, and similarly compute-intensive integral generation | Retain generation when it remains the simpler specialized route. |
| Aggregate | Retain the current supported generated route: its exact numerical matrix is not a cheap static-loop migration. |
| Scan, ordering, fold | Use a route-decision gate; prefer finite direct typed Java only where it is genuinely cheaper to implement and verify. |
| General reductions, normalizations, loss | Begin with bounded 0009E1 partial integral route decision; masked, advanced, softmax, normalization, and loss remain ordered summaries. |
| Pooling, attention, batch norm | Decide per family in 0009F: migrate orchestration/static loops; retain an already-good generated compute kernel when replacement increases work. |

For every direct route, prepare cold-selects one typed array/segment, dense/general, scalar/vector
where useful, and range-owning entry. No generic per-element runtime dispatch is allowed. Existing
clean-Java algorithms are the first reference. `legacy/pre-rewrite` is read-only supplemental
evidence for capability, observable behavior, algorithms, and tests; source copying, legacy
packages/dependencies, runtime coupling, and shortcuts remain forbidden.

## Out of scope

Architecture/module-boundary changes, a universal structural verifier, mandatory generated-vs-Java
performance studies, a benchmark ratio gate for migration, and a big-bang generated-code rewrite.

## Architecture references and constraints

`ARCHITECTURE.md` remains authoritative: CPU prepare owns lowering and kernel selection; Runtime
executes prepared work; parallelism is outside element loops; prepared recipes are immutable;
direct and retained generated loops receive proportionate hygiene review. This plan changes neither
that contract nor CPU module ownership.

## Acceptance criteria

- Every advertised CPU form is direct Java, retained bounded generated code, or an exact rejection.
- Every supported form has semantic and invocation coverage; direct and retained hot paths have
  proportionate source/call-site hygiene for allocation, boxing, reflection, string/map dispatch,
  synchronization, and per-element virtual dispatch.
- Generated implementation, schema/inventory/evidence, and structural scaffolding are removed only
  after its direct replacement passes semantics/conformance and no selected route references it.
  No permanent parallel implementation survives without an explicit reason.
- Existing performance evidence and known NON_PASSING facts remain reported. Fresh representative
  profiling is optional/final and never blocks correctness closure unless it exposes a concrete
  correctness regression.
- There is no requirement for universal generated-vs-clean structural equivalence or exhaustive
  new performance evidence.

## Tests / validation

Each active migration runs focused lowering, prepare, invocation, semantics, canary, and retirement
tests plus a proportionate hot-loop inspection. The final checkpoint verifies support accounting,
exact rejections, selected routes, stale generated evidence removal, documentation, and hygiene.
It reports existing performance facts but does not require fresh benchmark closure. This parent
planning revision validates Markdown/status consistency and `git diff --check`; it runs no Gradle.

## Dependencies and follow-up tasks

- 0009D is Complete: D1 aggregate, D2 scan, D3 ordering, and D4 fold are retained-generated-route
  decisions. D4 retains all 32 fold rows because a direct replacement would cost more to implement
  and verify.
- 0009E1 is Complete: it retained the finite typed generated partial integral route because direct
  migration would increase selection, binding, replacement, and retirement-proof work without a
  performance gate. E1A is Complete: it retained the complete generated FLOAT64/FLOAT32/BFLOAT16
  masked SUM/MEAN route because direct Java would duplicate typed carrier/layout, directional
  mask, selected exact-state, binding, invocation, inventory, and retirement work. E1B is a
  summary parent because its shared geometry has distinct numerical owners. E1B1 is Complete as a
  retained generated log-sum-exp route; E1B2 statistics is the sole next Ready detailed child, followed
  by E1B3 norms, E1C softmax-style, E2 normalization, and E3 loss.
- 0009F: per-family hybrid decision children for MATMUL/convolution/pooling/attention/batch norm.
- 0009G: final support, correctness, hygiene, inventory, and documentation checkpoint.

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

The canonical generated inventory and completed evidence remain useful historical accounting, but
they are no longer a requirement to prove every generated row against a clean-Java classfile.

## Known limitations

CPU 0008I loss fork 0 remains NON_PASSING (19 of 792 above 1.15; worst
1.3861164205039096); user-closed forks 1--4 remain historical facts. No result is promoted by this
planning revision.

## Validation evidence

D4's specified eight-class command passed on 2026-09-09: 131 tests, zero failures, errors, or
skips. E1's specified six-class command passed on 2026-09-09: 110 tests, zero failures, errors, or
skips. E1 retained its generated partial/combine implementation after source and test inspection,
not performance evidence; it made no Java change. Its clean documentation-focused finalization
reused that evidence, verified the current unselected generated candidate and whole-cell fallback,
and synchronized the four planning records without rerunning Java.

## Implementation notes

None.

## Completion summary

CPU 0009 remains Ready. Parent 0009D is Complete: D1 through D4 are retained-generated-route
decisions; the D4 fold result preserves the full 32-row generated family because a complete direct
Java replacement would increase total implementation and verification work. CPU 0009E1 and E1A
are Complete retained generated-route decisions; E1A retains the complete masked SUM/MEAN route
because a direct replacement would increase implementation and verification work. E1B is a Draft
parent split by distinct numerical ownership; E1B1 is Complete as a retained generated
log-sum-exp route, while E1B2 is the sole next Ready detailed child.

Status: Ready

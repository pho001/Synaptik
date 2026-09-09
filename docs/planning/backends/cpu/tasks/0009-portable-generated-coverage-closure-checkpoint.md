# Task 0009: Portable CPU Coverage and Implementation-Closure Checkpoint

## Status

Ready

## Goal

Close portable CPU coverage with the least costly maintainable implementation for each family:
direct optimized typed Java where the algorithm is static, bounded generated code where fusion or
specialized compute justifies it, and exact rejection where no route is supported. This supersedes
the unfinished program of universal generated-versus-clean-Java structural closure.

## Scope

Completed 0009A--C/D1/D1A/D2 remain historical completed evidence; they are not migration targets
merely because a later family adopts direct Java. D2 has completed its retained generated-route
decision; D3 is the sole detailed Ready frontier.

The default classification, subject to the active family's code review, is:

| Family | Planned implementation direction |
|---|---|
| Bounded fused pointwise DAGs and genuinely dynamic fused/specialized compositions | Retain bounded generation. |
| Dense MATMUL, Conv inner loops, and similarly compute-intensive integral generation | Retain generation when it remains the simpler specialized route. |
| Aggregate | Retain the current supported generated route: its exact numerical matrix is not a cheap static-loop migration. |
| Scan, ordering, fold | Use a route-decision gate; prefer finite direct typed Java only where it is genuinely cheaper to implement and verify. |
| General reductions, normalizations, loss | Split into bounded direct-Java summary children in 0009E. |
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

- 0009D: D1 aggregate and D2 scan (Complete retained-generated-route decisions), then detailed D3
  ordering and D4 fold (summary only).
- 0009E: separate summary children for static direct-Java reductions, normalization, and loss.
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

D2's focused seven-class command passed 65 tests with zero failures, errors, or skips. It
validated the 160-row scan matrix's generation, invocation, independent-reference semantics,
range ownership, and inventory join. Documentation-focused context `/root/cpu_0009d2_docs`
completed the corresponding planning review/static checks; no behavior changed.

## Implementation notes

None.

## Completion summary

CPU 0009 remains Ready. D1 and D2 are Complete retained-generated-route decisions. D3 is the sole
detailed Ready frontier; D4 remains summary-only.

Status: Ready

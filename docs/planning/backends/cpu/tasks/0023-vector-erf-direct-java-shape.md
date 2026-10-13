# CPU 0023 — Align vector ERF with the direct Java shape

## Status

Draft

## Goal and class

Class C generated hot-path follow-up. Bring FLOAT32/FLOAT64 vector ERF polynomial evaluation
into the same specialized constant-Horner shape as the clean Java oracle, or document a concrete
technical reason and evidence for any retained deviation. Preserve numerical meaning and support.
This Draft is not implementation authorization and does not reopen historical benchmark campaigns.

## Evidence and limits

CPU 0022's bounded diagnostic measured generated/direct vector ERF median ratios of 3.276754
(FLOAT32) and 3.509958 (FLOAT64): one process, 4,097 elements, preferred species, 3,000 warm-up
pairs and seven measured pairs. Both paths use the same polynomial algorithm and one scalar tail.
The generated path calls `CpuVectorMath.erf`, whose coefficient arrays are traversed in Horner
loops; the direct Java vector helper spells out constant Horner steps. Those production helpers
and loops predate 0022. The measured slowdown's cause, before/after regression status and other
environments are not established. Performance parity must not be claimed from this evidence.

## Bounded future packet

Read AGENTS generated-code discipline, root scope/invariants, backend-execution CPU routes and
identity, Model ERF/GELU contracts, and 0022's final oracle/tests/diagnostic evidence. Keep dtype,
coefficient precision, operation order, saturation, NaN/infinity/zero and tail behavior unchanged.
Inspect the generated entry and helper bodies together; an unchanged Class-File call alone does
not establish an equivalent hot path. Review dependent GELU without expanding other families.

Add structural and semantic regressions, assess compatibility identity explicitly, and run focused
generated/direct paired diagnostics after stabilization plus the affected module/Javadoc once.
No five-fork matrix, universal speed claim, new public strategy, tuning policy or numerical change.
If a shape correction does not close the gap, report the remaining evidence and investigate only
the demonstrated bottleneck; do not substitute a global benchmark program for a scoped decision.

## Dependencies and integration

- Depends on: CPU 0022 Complete and explicit selection of this Draft.
- Conflicts with: CPU activation/vector/helper/schema and generated-evidence writers.
- Parallel group: None until a Ready packet selects disjoint ownership.
- Common base revision: select after the project review closes.
- Integration order: oracle/shape correction → focused validation → independent Class C review.
- Integration validation: affected CPU tests/Javadoc, exact semantic/class checks, code-shape
  inspection, bounded timing evidence and documentation; no automatic repository-wide campaign.
- Shared-document integration owner: future coordinator.

Independent review must distinguish a demonstrated current gap from an unproved causal explanation
or regression claim. Record any remaining gap explicitly rather than labeling parity as passing.

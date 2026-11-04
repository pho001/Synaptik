# [CPU](../../../../glossary.md#term-cpu) 0023 — Align [vector](../../../../glossary.md#term-vector) [ERF](../../../../glossary.md#term-erf) with the direct Java [shape](../../../../glossary.md#shape)

## Status

Draft

## Goal and class

Class C generated hot-path follow-up. Bring [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [vector](../../../../glossary.md#term-vector) [ERF](../../../../glossary.md#term-erf) polynomial [evaluation](../../../../glossary.md#term-evaluation)
into the same specialized [constant](../../../../glossary.md#term-constant)-Horner [shape](../../../../glossary.md#shape) as the clean Java [oracle](../../../../glossary.md#term-test-oracle), or document a concrete
technical reason and evidence for any retained deviation. Preserve numerical meaning and support.
This Draft is not implementation authorization and does not reopen historical benchmark campaigns.

## Evidence and limits

[CPU](../../../../glossary.md#term-cpu) 0022's bounded diagnostic measured generated/direct [vector](../../../../glossary.md#term-vector) [ERF](../../../../glossary.md#term-erf) [median](../../../../glossary.md#term-median) ratios of 3.276754
([FLOAT32](../../../../glossary.md#term-float32)) and 3.509958 ([FLOAT64](../../../../glossary.md#term-float64)): one process, 4,097 [elements](../../../../glossary.md#term-element), preferred species, 3,000 [warm-up](../../../../glossary.md#benchmark-measurement-vocabulary)
pairs and seven measured pairs. Both paths use the same polynomial algorithm and one [scalar tail](../../../../glossary.md#term-scalar-tail).
The generated path calls `CpuVectorMath.erf`, whose coefficient arrays are traversed in Horner
loops; the direct Java vector helper spells out [constant](../../../../glossary.md#term-constant) Horner steps. Those production helpers
and loops predate 0022. The measured slowdown's cause, before/after [regression](../../../../glossary.md#term-regression) status and other
environments are not established. Performance parity must not be claimed from this evidence.

## Bounded future packet

Read AGENTS generated-code discipline, root [scope](../../../../glossary.md#term-scope)/invariants, [backend](../../../../glossary.md#backend)-execution [CPU](../../../../glossary.md#term-cpu) routes and
[identity](../../../../glossary.md#term-identity-meanings), [Model](../../../../glossary.md#term-model) [ERF](../../../../glossary.md#term-erf)/[GELU](../../../../glossary.md#gaussian-error-linear-unit--gelu) contracts, and 0022's final [oracle](../../../../glossary.md#term-test-oracle)/tests/diagnostic evidence. Keep [dtype](../../../../glossary.md#data-type--datatype),
coefficient [precision](../../../../glossary.md#term-precision), [operation](../../../../glossary.md#operation) order, [saturation](../../../../glossary.md#term-saturation), [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/zero and tail behavior unchanged.
Inspect the generated entry and helper bodies together; an unchanged Class-File call alone does
not establish an equivalent [hot path](../../../../glossary.md#term-hot-path). Review dependent GELU without expanding other families.

Add structural and semantic regressions, assess [compatibility identity](../../../../glossary.md#identity-meanings-and-algebraic-identities) explicitly, and [run](../../../../glossary.md#term-run) focused
generated/direct paired diagnostics after stabilization plus the affected [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) once.
No five-[fork](../../../../glossary.md#term-fork-meanings) [matrix](../../../../glossary.md#term-matrix-meanings), universal speed claim, new public strategy, tuning policy or numerical change.
If a [shape](../../../../glossary.md#shape) correction does not close the gap, report the remaining evidence and investigate only
the demonstrated bottleneck; do not substitute a global benchmark program for a scoped decision.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0022 Complete and explicit selection of this Draft.
- Conflicts with: CPU [activation](../../../../glossary.md#term-activation)/[vector](../../../../glossary.md#term-vector)/helper/[schema](../../../../glossary.md#term-schema) and generated-evidence writers.
- Parallel group: None until a [Ready](../../../../glossary.md#term-ready) packet selects disjoint [ownership](../../../../glossary.md#term-ownership-meanings).
- Common base revision: select after the project review closes.
- Integration order: [oracle](../../../../glossary.md#term-test-oracle)/[shape](../../../../glossary.md#shape) correction → focused [validation](../../../../glossary.md#term-validation) → independent Class C review.
- Integration validation: affected CPU tests/Javadoc, exact semantic/class checks, code-shape
  inspection, bounded timing evidence and documentation; no automatic repository-wide campaign.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): future coordinator.

Independent review must distinguish a demonstrated current gap from an unproved causal explanation
or [regression](../../../../glossary.md#term-regression) claim. Record any remaining gap explicitly rather than labeling parity as passing.

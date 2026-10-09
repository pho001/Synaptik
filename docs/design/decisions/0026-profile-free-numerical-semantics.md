# ADR 0026: Profile-free numerical semantics

## Status

Accepted — 2026-10-08, on completion of [Task 0032](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
in the main worktree. This record explains the decision; the root and incorporated scoped
contracts remain authoritative. Acceptance does not assert a commit, push, or remote publication.

## Context

A *numerical profile* selected a graph-wide Model allowed-result set before Task 0032.
`STRICT_IEEE` and `ACCELERATOR` traveled through cold compilation, capability queries,
preparation, and backend identities. The current
[root invariants](../../../ARCHITECTURE.md#core-invariants) and incorporated
[profile-free Model semantics](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)
now specify the cutover contract; an architecture decision record (ADR) does
not supply that authority. The former strict set was not universally bit-exact, and the accelerator
name is not a speed guarantee. Yet a global choice conflates exact stored-value behavior with
arithmetic latitude and carries profile identity across
otherwise unrelated layers.

The historical pre-cutover text had concrete semantic conflicts that the cutover resolved:

- Before cutover, [`ScatterReduction.MUL`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java)
  promised one order-independent exact abstract floating product with final result-format rounding.
  [`AxisScatterKind`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java)
  separately permitted an `ACCELERATOR FLOAT32` per-step rounded tree for the same reduction. The
  profile-free contract replaces the abstract-product promise for floating `MUL`.
- Before cutover, [`BinaryComparisonKind`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java)
  permitted `ACCELERATOR FLOAT32` comparisons to read subnormals as signed zero. The
  stored-value comparison rule still requires route-specific conformance evidence.

## Decision drivers

- Keep one Model-owned meaning for each operation without a replacement global mode or backend-
  defined semantics.
- Preserve exact structure, represented-value selection, conversions, and state while locating
  floating arithmetic latitude at named family, dtype, and observable-site boundaries.
- Retain existing truthful provider capability, low-precision routes, and safety checks without
  converting numerical samples into production policy or new support claims.
- Make finite numerical validation practical without granting a tolerance-based compiler rewrite.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Keep both profiles | Preserves current API and proof model | Retains cold identity and the split between stored-value and arithmetic rules. |
| Rename `ACCELERATOR` to `FAST` or `RELAXED` | Small surface change | Renames rather than resolves the global semantic split; implies performance or blanket relaxation. |
| Introduce a global tolerance or runtime certificate | One apparent numerical gate | Cannot protect special values, winner selection, exact casts, or rewrite legality; adds production policy. |
| Remove selectors; define family/dtype rules and test-only tolerances | Separates exact boundaries from arithmetic qualification | Required coordinated API/identity migration and independent route evidence; Task 0032 completed both. |

The table compares semantic choices, not measured route performance.

## Decision

Remove `STRICT_IEEE` and `ACCELERATOR` selectors, their public selection API, and their cold
transport/identity fields without replacing them with another global profile. Model continues to
own each operation's family formula, contributor domain, guards, special-value behavior, allowed
working precision, rounding, and observable result rules. Backend support remains an occurrence-
specific boolean question, not a consequence of removing a selector.

Structure and discrete results remain exact: dtype and promotion, shapes/layouts/axes, mapped
contributors, bounds and indices, masks and predicates, ordering and stability, graph-random state,
saved values, and publication. Declared casts and raw movement preserve their existing conversion
or represented-bit rules. `WHERE` makes an exact Boolean branch choice, then applies the declared
cast if promotion requires one; same-type selection preserves the chosen representation. Exact
represented-value rules remain for `ABS`, `NEG`, `SIGN`, `FLOOR`, `CEIL`, and `RELU` where their
family contracts specify them. No neighboring arithmetic DAZ/FTZ freedom leaks into these
boundaries.

Floating comparisons and extrema observe stored numeric values, including represented
subnormals. They retain their family NaN, opposite-signed-zero, tie, winner, and index rules;
comparison has no epsilon and extrema cannot invent a candidate. For example, a positive minimum
subnormal is greater than `+0` when on the left, less when on the right, and unequal in both
orders; `MIN` selects `+0` and `MAX` selects the subnormal. This is a semantic example;
route qualification is separate from the Model rule.

The accepted permission is per *named floating arithmetic primitive site* in the
applicable Model family formula, not a blanket operation-output rule:

| Arithmetic-site dtype | Input DAZ | Result FTZ |
|---|---|---|
| `FLOAT32` | Permitted at named primitive inputs | Permitted at named primitive results |
| `BFLOAT16` | Permitted at named primitive inputs | Permitted at named primitive results |
| `FLOAT16` | Permitted at named primitive inputs, including multiplication | Permitted at named primitive results, including multiplication |
| `FLOAT64` | Forbidden | Forbidden |

Denormals-are-zero (DAZ) reads a subnormal arithmetic-site input as same-signed zero without
changing its stored value. Flush-to-zero (FTZ) may map a finite subnormal arithmetic-site result
to zero, subject to that family's existing signed-zero contract; it creates no universal choice
of zero sign. These permissions never apply to stored comparisons, extrema/winner selection, raw
movement, casts, selection, indices, masks, or state, nor to the exact unary `ABS`, `NEG`, `SIGN`,
`FLOOR`, `CEIL`, and `RELU` sites. Arithmetic NaN class/domain, infinity, signed zero, overflow,
and underflow rules remain explicit per family; no generic finite tolerance decides a
special-value result. Composite formulas and generated gradients inherit only the rules of their
named arithmetic primitives, not a blanket final-output allowance. For example, `BFLOAT16`
`0x0001 * 128` may yield zero through input DAZ even though the exact low product is normal
`0x0080`; `FLOAT16` multiplication has the same permission, not a requirement to zero or a claim
about a particular backend's zeroing site. Family formulas and contributor domains remain intact.

For floating scatter `MUL`, the base and every addressed update contribute exactly once, but a
permitted rounded multiplication tree need not equal a once-rounded exact abstract product. With
`FLOAT32` factors `[maxFinite, 2, +0]`, an overflow-first tree can produce NaN from
`infinity * +0`, whereas an exact product rounded once gives `+0`. The revised contract permits the
former where the family/dtype arithmetic rules qualify it; it does not permit omitted or duplicate
contributors. Integral fixed-width modular `MUL`, exact target mapping, and raw unaddressed cells
remain unchanged. This conceptual example does not certify a backend route.

`BFLOAT16` and `FLOAT16` remain distinct storage types with their exact promotion and cast rules.
For a qualified homogeneous low-precision arithmetic occurrence, working values and accumulators
are `FLOAT32`, and each declared low output has one final round-to-nearest, ties-to-even narrowing;
there is no hidden low pre-narrowing of inputs, contributors, accumulators, or saved values. This
is a Model semantic obligation; route qualification is separate. Direct mixed-
low execution is not newly admitted; an explicit `FLOAT32` cast remains the boundary.

Finite accuracy thresholds belong only in tests, scoped by family, dtype, backend, and size (and
route where routes differ), with an explicit oracle, metric, threshold, and cancellation boundary.
No threshold is selected by this ADR. NaN/infinity, signed zero, subnormal/underflow, domains,
empty cases, and exact boundaries receive separate tests. There is no public accuracy envelope,
runtime tolerance, proof evaluator, or numerical certificate. A passing sample or tolerance cannot
license Compiler's guarded identity rewrites, `x * 0`, `x - x`, reassociation, contraction, or
fusion across observable or special-value boundaries. Unproved transformations keep the old path.
Compiler-generated gradients remain ordinary public Tensor operations under the same single
Model semantics as forward operations. Their saved outputs, masks, indices, statistics, and state
are exact stored values; backward preflight and entire generated-topology capability stay
fail-closed. Removing the graph-wide profile does not create a gradient-only numerical policy.

## Rationale and evidence limits

The exact/stored boundary is independently useful: a bit-preserving move of low word `0x0001`
must remain a move even when arithmetic on that value underflows. The current
[CPU generated scatter regression](../../../backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java)
contains a `maxFinite * 2 * 0` witness for its exact `+0` result; that route-local result does not
exclude a NaN-producing rounded tree under the revised rule. The
[Metal stored-subnormal regression](../../../backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalStoredSubnormalProviderNativeTest.java)
records custom-program raw movement, cast, comparison/extrema, and arithmetic observations. These
tests are evidence about their specified routes, not universal Model rules or proof for other
routes. Task 0032's integrated validation and independent review are recorded in its Result.

Historical readiness evidence in the [Model 0032 brief](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
reported that Metal custom low-word `0x0001` survived cast/raw movement, while `BFLOAT16` minimum
subnormal times `128` produced zero even though its exact low result is normal `0x0080`. Final
low-result FTZ alone could not explain that observation; input DAZ or earlier loss could. The
precise zeroing site remains unproved, and is not needed to define the allowed-result set.
`FLOAT16` minimum subnormal times one had a distinct result, so no uniform low-type device
behavior follows. Those original route-local observations alone did not qualify opaque
MPSGraph, other shapes, or other routes; the later integrated conformance evidence is separate.

## Consequences and completed cutover

The benefit is one visible exact boundary and no graph-wide numerical switch. The costs are loss
of a caller-selected strict mode, changed floating results (notably comparison of subnormals and
scatter-product specials), new family/dtype qualification, and coordinated identity/API migration.
No cross-backend bitwise reproducibility or performance improvement is promised.

The cutover reconciled the three Javadocs above, affected Model formulas, the glossary and user
guides, Compiler's seven guarded rewrites and their existing phase/descriptor/output/gradient
guards, generated gradients, and focused tests. It projected the actual pre-cutover `ACCELERATOR`
CPU/Metal provider **true and false** answers to 254 profile-free occurrence rows, retaining
independent `BFLOAT16`/`FLOAT16` queries and negative predicates. This representative ledger is
drift evidence, not enumeration or numerical certification. The implementation retains the fixed
Metal low `CUSTOM_PROGRAM` route, eligible `FLOAT32` MPSGraph/custom routes, transfers, Training,
tuning, and fail-closed unsupported occurrences. Only numerical-profile and numerical-
certification state was removed; source/package/manifest authentication, native application binary
interface (ABI) and export checks, decoder and preflight validation, bounds, ownership,
stale-identity rejection, and fail-before-mutation rules remain. Coordinated identities are CPU
generator schema 69, Metal program schema 20 with a 124-byte header, Metal identity 30, and
native ABI 7 with thirteen exports. CPU schema 68 and Metal schema 19/identity 29 are historical.

The revised [root invariants](../../../ARCHITECTURE.md#core-invariants),
[backend route qualification](../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
and [low-precision capability evidence](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
specify route- and occurrence-scoped conformance: exact structure/stored/special checks plus
explicit finite test oracles, metrics, and thresholds. Task 0032 passed the integrated gate. The
cutover does not infer provider support from samples, silently admit an unqualified route, or weaken
source-bound route evidence, structural dominance, native ABI/source
authentication, fixed cold selection, or the ban on runtime fallback and retry.

The approved rule moved [Model 0032](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
into execution. Its route/site conformance, finite/special/exact tests, documentation, and
independent Class C review are complete in the main worktree. The source-matched signed local
Metal package has SHA-256 `77f683d4331a688423a5a5e14cb462e3d9bafdf1aaab3ae5963c20dfadb3d318`;
historical schema-19 evidence does not certify schema-20 programs. The MPSGraph scalar `+0`
guard's performance effect remains unmeasured, a nonblocking follow-up. Git history records
publication separately from this decision record.

## Relationship to earlier decisions

This decision supersedes [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)'s
two-profile and recursive numerical-floor choice and [ADR 0023](0023-low-precision-accelerator-parity.md)'s
profile-indexed low numerical envelope/proof choice. [ADR 0019](0019-explicit-numerical-profiles.md)
was already superseded by 0021; its profile rationale remains historical. ADR 0023's provider
ledger and append-only identity rationale remain relevant. [ADR 0025](0025-custom-only-metal-low-precision.md)'s
custom-only low route remains in force; its profile-dependent trace/identity provisions are
superseded by this decision. ADR 0025 is not wholly superseded.

## Related documentation

- [Architecture authority and scope index](../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
- [Current architecture explanation](../../architecture/current-architecture-plan.md)
- [Backend capability and safety contract](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)
- [Profile-free Compiler compile identity](../../architecture/contracts/compiler-autograd.md#profile-free-compile-identity)
- [Compiler-owned automatic differentiation](../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Profile-free Engine/Prepare lifecycle](../../architecture/contracts/runtime-prepare-engine.md#profile-free-engine-and-prepare-lifecycle)

# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0019: Explicit Numerical Profiles


## Editorial reading context

A numerical profile was a graph-wide selector for permitted floating results. It was not a benchmark sampling profile or a speed promise. These selectors and their transport are historical; the supersession chain leads to operation-local numerical semantics. Keep signed-zero, NaN and empty-domain assertions scoped to the rule that actually authorized them.

Use the [profile-removal decision](0026-profile-free-numerical-semantics.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Superseded by [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md) — 2026-09-25

[ADR](../../glossary.md#term-architecture-decision-record-adr) 0019 remains historical context for the two [graph](../../glossary.md#graph)-wide [identities](../../glossary.md#term-identity-meanings), monotonicity, [ownership](../../glossary.md#term-ownership-meanings),
and [fail-closed](../../glossary.md#term-fail-closed) capability decisions. Its closed [operation](../../glossary.md#operation)-row table and rejection of a generic
bounded primitive rule no longer govern.

## Context

Synaptik's [Model](../../glossary.md#term-model) operations already define family-specific numerical contracts. Some require exact
represented behavior; some deliberately leave [NaN](../../glossary.md#term-nan) payloads or intermediate [precision](../../glossary.md#term-precision) unspecified;
and [contraction](../../glossary.md#contraction-dimension) families already permit [reassociation and fused multiply-add](../../glossary.md#rounding-sites-fma-and-aggregate-floor). Several [accelerator](../../glossary.md#term-accelerator)
routes cannot satisfy particular current [`FLOAT32`](../../glossary.md#term-float32) [subnormal](../../glossary.md#term-subnormal) or reduction results. A fresh [Metal](../../glossary.md#term-metal)
[MATMUL](../../glossary.md#matrix-multiplication--matmul) probe also published [positive zero](../../glossary.md#nan-infinity-and-signed-zero) for the one-term product `+0.0f * -1.0f`, whose
otherwise-permitted result is [negative zero](../../glossary.md#nan-infinity-and-signed-zero). Treating these differences as generic fast math would
also admit term loss, unrelated [graph](../../glossary.md#graph) rewrites, gross special-value errors, and behavior that no
[operation](../../glossary.md#operation) contract authorizes.

The [foundational numerical-profile contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
is the sole [normative](../../glossary.md#term-normative) definition of the profiles and operation-family allowed-result sets. This
record explains why that bounded design was selected.

## Decision drivers

- Preserve every operation's current contract as the strict baseline without falsely promising
  universal bitwise identity, correct rounding, Java <span data-glossary-navigation="true"> ([identity glossary](../../glossary.md#term-identity-meanings); [correct rounding glossary](../../glossary.md#term-correct-rounding))</span>`strictfp`, or fixed instructions.
- Make any wider result set explicit, opt-in, [graph](../../glossary.md#graph)-wide, and [operation](../../glossary.md#operation)-specific.
- Bound denormals-are-zero (DAZ), flush-to-zero (FTZ), reassociation, and FMA permissions tightly
  enough to exclude term loss, arbitrary rewrites, tolerance acceptance, and gross special-value
  errors.<span data-glossary-navigation="true"> ([denormals-are-zero glossary](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [DAZ glossary](../../glossary.md#term-daz); [flush-to-zero glossary](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [FTZ glossary](../../glossary.md#term-ftz); [reassociation glossary](../../glossary.md#term-reassociation); [FMA glossary](../../glossary.md#term-fused-multiply-add-fma); [tolerance glossary](../../glossary.md#term-tolerance))</span>
- Keep semantic authority in Model while Config declares only identity and Planning asks only
  profile-qualified capability questions.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [Planning glossary](../../glossary.md#term-planning))</span>
- Select all execution behavior before the Runtime hot path and keep backend preparation
  responsible for truthful realization.<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [hot path glossary](../../glossary.md#term-hot-path); [backend glossary](../../glossary.md#backend))</span>
- Prevent cache, specialization, generated-artifact, or tuning decisions from crossing profile
  boundaries.<span data-glossary-navigation="true"> ([cache glossary](../../glossary.md#term-cache); [specialization glossary](../../glossary.md#term-specialization); [artifact glossary](../../glossary.md#term-artifact))</span>

## Options considered

### One fast-math [boolean](../../glossary.md#data-type--datatype)

A boolean cannot say which [operation](../../glossary.md#operation) family or transformation it permits. It would encourage each
[backend](../../glossary.md#backend) to invent a different meaning and would conflate [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz), [reassociation](../../glossary.md#term-reassociation), [contraction](../../glossary.md#contraction-dimension),
approximation, reduced [precision](../../glossary.md#term-precision), and algebraic rewriting. Rejected.

### Generic [tolerance](../../glossary.md#term-tolerance) or [ULP](../../glossary.md#term-unit-in-the-last-place-ulp) budget

Tolerance-based acceptance does not define special-value, [signed-zero](../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../glossary.md#term-nan), term-membership, or
selected-bit behavior. It would also incorrectly give transcendental and other unlisted families a
blanket approximation allowance. Rejected.

### Per-step reduction exact-zero sign choice

Allowing either sign after every exact-zero reduction addition is broader than the observed
difference requires. Intermediate zero sign does not create a new final nonzero value or
classification for the permitted reduction trees, while exposing the choice at every [node](../../glossary.md#node) makes
the allowed-result set harder to audit. Rejected.

### Per-product or per-intermediate [contraction](../../glossary.md#contraction-dimension) exact-zero sign choice

Allowing either sign for every exact-zero product, addition, or [FMA](../../glossary.md#term-fused-multiply-add-fma) result is broader than the
observed [MATMUL](../../glossary.md#matrix-multiplication--matmul) [publication](../../glossary.md#term-publication) difference requires. It would expose choices inside a contraction even
though final-publication freedom already admits the required result. Rejected.

### Shared [MATMUL](../../glossary.md#matrix-multiplication--matmul) and [convolution](../../glossary.md#term-channels-first-convolution-layers) [publication](../../glossary.md#term-publication) choice

The old profile table grouped MATMUL with [CONV2D](../../glossary.md#convolution--conv2d) and [CONV3D](../../glossary.md#conv3d--ncdhw) because all three shared existing
[reassociation](../../glossary.md#term-reassociation)/[FMA](../../glossary.md#term-fused-multiply-add-fma) plus [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) permissions. Extending that shared row would relax convolution
without evidence and would obscure that the observed mismatch is specific to a completed MATMUL
cell. Rejected.

### Per-[backend](../../glossary.md#backend) numerical modes

Backend-specific semantics would make the same [graph](../../glossary.md#graph) request mean different things depending on
availability or [owner](../../glossary.md#term-owner) selection. Hardware and route capability must answer a [Model](../../glossary.md#term-model)-owned semantic
question, not define it. Rejected.

### [Metal](../../glossary.md#term-metal)-specific reduction exception

A Metal-only rule would let a [backend](../../glossary.md#backend) redefine [Model](../../glossary.md#term-model) semantics and make one [graph](../../glossary.md#graph) request depend
on [owner](../../glossary.md#term-owner) selection. The [operation](../../glossary.md#operation) family must own any justified result freedom, and every backend
must separately qualify support before advertising it. Rejected.

### Per-[operation](../../glossary.md#operation) or per-[node](../../glossary.md#node) caller selection

Fine-grained selection would complicate [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), [ownership](../../glossary.md#term-ownership-meanings), [fusion](../../glossary.md#term-fusion) boundaries, [cache](../../glossary.md#term-cache) [identity](../../glossary.md#term-identity-meanings),
and user reasoning before there is evidence that such complexity is necessary. It could also let
an implementation erase observable intermediate values through accidental cross-node [contraction](../../glossary.md#contraction-dimension).
Rejected.

### Two [graph](../../glossary.md#graph)-wide profiles with bounded [operation](../../glossary.md#operation) rows

One graph-wide [identity](../../glossary.md#term-identity-meanings) can preserve the existing result set or opt into a precisely enumerated
superset. [Model](../../glossary.md#term-model) remains the single semantic [owner](../../glossary.md#term-owner) while later layers transport and realize the
choice. Selected.

## Decision

Synaptik defines two [graph](../../glossary.md#graph) numerical-profile [identities](../../glossary.md#term-identity-meanings). `STRICT_IEEE` denotes each [operation](../../glossary.md#operation)'s
exact current allowed-result set, including all family-specific promises and freedoms.
[`ACCELERATOR`](../../glossary.md#term-accelerator) is an opt-in superset: a strict result is always valid, and an additional result is
valid only when the sole [normative](../../glossary.md#term-normative) table explicitly permits the transformation for that named
operation family and [`FLOAT32`](../../glossary.md#term-float32).

The bounded additions are [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) and zero-sign freedom for listed arithmetic, all-and-only-term
[binary](../../glossary.md#term-binary-descriptor-and-metadata)-tree [reassociation](../../glossary.md#term-reassociation) for `SUM`, [`MEAN`](../../glossary.md#term-mean), and `SUM_TO_SHAPE`, existing [contraction](../../glossary.md#contraction-dimension)-scoped
reassociation/[FMA](../../glossary.md#term-fused-multiply-add-fma) plus DAZ/FTZ for [MATMUL](../../glossary.md#matrix-multiplication--matmul) and [convolution](../../glossary.md#term-channels-first-convolution-layers), a MATMUL-only final-[publication](../../glossary.md#term-publication)
exact-zero sign choice, DAZ-normalized floating comparisons, bounded [extrema](../../glossary.md#saturation-pole-and-extremum) ties, and
arithmetic-step-only DAZ/FTZ for cumulative scans.

For an accelerator `FLOAT32` MATMUL cell, the declared [contraction dimension](../../glossary.md#contraction-dimension) still supplies every
pairwise multiplication term exactly once. One complete otherwise-permitted [evaluation](../../glossary.md#term-evaluation) may use
the family's existing reassociation, corresponding multiply-add FMA placements, `FLOAT32`
[rounding](../../glossary.md#term-rounding), and row-scoped DAZ/FTZ. Only when that complete nonempty result is exact zero may
publication choose either zero sign. Every standalone product and every addition or FMA result
before publication keeps its existing sign rule. The rule includes a one-term contraction without
adding a positive-zero [accumulator](../../glossary.md#accumulator-type) or turning its multiplication into an FMA. Empty contraction
remains [positive zero](../../glossary.md#nan-infinity-and-signed-zero); [finite](../../glossary.md#term-finite) nonzero results and [NaN](../../glossary.md#term-nan)/[infinity](../../glossary.md#term-infinity) classification do not widen.
[CONV2D](../../glossary.md#convolution--conv2d) and [CONV3D](../../glossary.md#conv3d--ncdhw) retain only their previous reassociation/FMA plus DAZ/FTZ profile rule.

For an accelerator `FLOAT32` `SUM` or arithmetic `SUM_TO_SHAPE` cell with at least two terms,
either zero sign is permitted only when the selected tree's root addition is exact zero. For
`MEAN`, the declared positive count remains a mandatory `FLOAT32` divisor and either sign is
permitted only when that quotient is exact zero. Intermediate exact-zero additions receive no new
sign freedom. One-term, point-domain, and equal-[Shape](../../glossary.md#shape) copy forms remain bit-preserving; empty SUM
remains positive zero and zero-count MEAN remains NaN. FMA can contract only a corresponding
multiply/add in a declared contraction. It cannot fuse arbitrary graph [nodes](../../glossary.md#node) or erase an
observable intermediate. Movement, selection, classification, [Boolean](../../glossary.md#data-type--datatype) logic, casts, indexing,
ordering, arg-extrema, unlisted operations, and every type other than `FLOAT32` receive no
relaxation.

Neither profile permits term dropping, reciprocal substitution, algebraic [identity](../../glossary.md#term-identity-meanings) rewriting,
arbitrary reduced [precision](../../glossary.md#term-precision), cross-node contraction, [tolerance](../../glossary.md#term-tolerance)-based acceptance, or a generic
transcendental error budget. In particular, `maxFinite / maxFinite -> 0`,
`+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and `TANH(NaN) -> +1` remain invalid. Movement,
[`CONTIGUOUS`](../../glossary.md#term-contiguous), `CAST`, and selected `WHERE` values do not inherit DAZ/FTZ or NaN-[payload](../../glossary.md#term-nan-payload) freedom from
neighboring arithmetic.

The choice is graph-wide and cold. Config owns only the [immutable](../../glossary.md#term-immutable) declarative identity. [Engine](../../glossary.md#term-engine)
construction selects one value, defaulting to `STRICT_IEEE`, and the current propagation spine
carries it through [Planning](../../glossary.md#term-planning), [Compiler](../../glossary.md#term-compiler), [Prepare](../../glossary.md#prepare), and [backend](../../glossary.md#backend) plan/[cache](../../glossary.md#term-cache) identity. Planning asks
profile-qualified capability questions without interpreting semantics. [Runtime](../../glossary.md#term-runtime) executes the
prepared recipe without profile policy or a hot-path [lookup](../../glossary.md#term-lookup).

The selected profile is part of relevant [backend route](../../glossary.md#backend-route), [specialization](../../glossary.md#term-specialization), generated-[artifact](../../glossary.md#term-artifact),
local-tuning, complete-plan-tuning, and cache-[compatibility identity](../../glossary.md#identity-meanings-and-algebraic-identities) before a backend may advertise
relaxed capability. [CPU](../../glossary.md#term-cpu) now advertises both profiles with identical exact behavior. [Metal](../../glossary.md#term-metal)'s common
domain includes its exact [unary](../../glossary.md#term-unary)/affine/[canonicalization](../../glossary.md#occurrence-identity-and-canonical-form)/indexing/[BOOL](../../glossary.md#data-type--datatype)/movement/replacement/fold/
aggregate rows and no-[gradient](../../glossary.md#term-gradient) promoted [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) MATMUL. Accelerator Metal additionally
advertises only the FLOAT32 and low-precision rows admitted by the current capability provider and
the parity boundary in [ADR](../../glossary.md#term-architecture-decision-record-adr) 0023. Direct [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) mixing remains unsupported, and strict
floating MATMUL remains [fail-closed](../../glossary.md#term-fail-closed). No backend may infer permission merely from the profile
identity.

[Trace payload](../../glossary.md#trace-payload--tracepayload) changes remain deferred. The current propagation spine adds no [trace](../../glossary.md#trace) field because
the profile is cold prepared identity rather than per-[run state](../../glossary.md#run-state--runstate); later observability requires a
separately coordinated Trace architecture update if operational evidence shows it is needed.

## Rationale

A disposable bounded raw-bit comparison covered 1,823 one- through four-term [MATMUL](../../glossary.md#matrix-multiplication--matmul) cases,
168,461 ordered [reassociation](../../glossary.md#term-reassociation) trees, 1,895,389 legal unfused/[FMA](../../glossary.md#term-fused-multiply-add-fma) [evaluation](../../glossary.md#term-evaluation) plans, and 3,232
baseline root states. The operands covered both zero signs, [cancellation](../../glossary.md#term-cancellation), [subnormal](../../glossary.md#term-subnormal) and normal
boundaries, [gradual underflow](../../glossary.md#term-gradual-underflow), [infinities](../../glossary.md#nan-infinity-and-signed-zero), and NaNs under row-scoped [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz). For every case,
final-[publication](../../glossary.md#term-publication)-only freedom produced the same allowed publication set as hypothetical
per-product and per-intermediate exact-zero sign freedom followed by that publication rule. The
one-term `+0.0f * -1.0f` witness retained raw multiplication bits `0x80000000` while publication
admitted both `0x80000000` and `0x00000000`. A rounded-[underflow](../../glossary.md#term-underflow) zero remained distinguishable
from exact zero, and FTZ retained its existing either-sign rule.

The MATMUL rule is tied to exact zero rather than a [tolerance](../../glossary.md#term-tolerance), so it does not admit a nearby
nonzero value. It changes neither term membership nor [NaN](../../glossary.md#term-nan)/[infinity](../../glossary.md#term-infinity) classification and cannot add
an [accumulator](../../glossary.md#accumulator-type) [identity](../../glossary.md#term-identity-meanings) or a new FMA. Keeping the rule in the [Model](../../glossary.md#term-model)-owned MATMUL row rather than
naming [Metal](../../glossary.md#term-metal) also preserves one semantic question for every [backend capability](../../glossary.md#backend-capability) check. Keeping
[convolution](../../glossary.md#term-channels-first-convolution-layers) separate avoids an unsupported relaxation of a neighboring [contraction](../../glossary.md#contraction-dimension) family.

## Consequences

### Positive

- Current strict behavior and existing [backend](../../glossary.md#backend) evidence remain valid.
- Accelerator permission is auditable by operation family rather than inferred from hardware or a
  performance objective.<span data-glossary-navigation="true"> ([Accelerator glossary](../../glossary.md#term-accelerator); [operation glossary](../../glossary.md#operation))</span>
- A backend may incrementally support either profile and may use strict results under both.
- Gross errors and semantic leakage into movement, selection, casting, classification, or other
  data types remain excluded.
- Strict MATMUL, both convolution families, empty contractions, and all product/intermediate sign
  rules remain unchanged.<span data-glossary-navigation="true"> ([MATMUL glossary](../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../glossary.md#term-channels-first-convolution-layers))</span>
- [Graph](../../glossary.md#graph)-wide cold selection keeps [Runtime](../../glossary.md#term-runtime) free of numerical-policy branching.

### Negative and limits

- [Backends](../../glossary.md#backend) must qualify capability and [cache](../../glossary.md#term-cache) [identity](../../glossary.md#term-identity-meanings) per profile before exposing new routes.
- [`ACCELERATOR`](../../glossary.md#term-accelerator) does not automatically unlock unlisted unary, transcendental, normalization, loss,
  attention, pooling, scatter/fold, random, or recurrent operations.<span data-glossary-navigation="true"> ([unary glossary](../../glossary.md#term-unary); [normalization glossary](../../glossary.md#term-normalization); [loss glossary](../../glossary.md#loss); [pooling glossary](../../glossary.md#term-pooling))</span>
- A future operation relaxation requires another coordinated architecture update with a named
  finite-domain and special-value envelope.<span data-glossary-navigation="true"> ([operation glossary](../../glossary.md#operation); [finite glossary](../../glossary.md#term-finite))</span>
- Per-[node](../../glossary.md#node) profiles and [trace](../../glossary.md#trace) reporting remain unavailable.

### Follow-up

[CPU](../../glossary.md#term-cpu) task 0017 and [Metal](../../glossary.md#term-metal) task 0015 completed the first bounded [backend](../../glossary.md#backend) realizations with
profile-specific [conformance](../../glossary.md#term-conformance) evidence. Later operations remain independently gated by the sole
[normative](../../glossary.md#term-normative) table, [operation](../../glossary.md#operation)-specific evidence, and [fail-closed](../../glossary.md#term-fail-closed) capability.

## Related documentation

- [Authoritative foundational numerical-profile contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Architecture authority index](../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
- [Task 0027](../../planning/modules/model/tasks/0027-explicit-numerical-profile-semantic-authority.md)
- [Engine task 0018](../../planning/modules/engine/tasks/0018-explicit-numerical-profile-propagation-spine.md)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine))</span>
- [Task 0029](../../planning/modules/model/tasks/0029-accelerator-matmul-exact-zero-sign-freedom.md)

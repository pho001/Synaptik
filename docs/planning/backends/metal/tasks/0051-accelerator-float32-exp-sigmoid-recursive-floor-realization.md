# Task 0051: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid) Recursive-Floor Realization

## Reader context

This blocked attempt separates three questions: can the Java/native boundary describe a candidate, does its algorithm meet the then-required full-domain result set, and did a bounded device sample reject it? An 83-word pass answered only the last question. It did not turn a MPSGraph selector into a complete-domain proof. The consumed invocation and rejected candidates remain history.

Model owned the formula and rounding choices; Metal owned candidate preparation and execution. Later profile-free routes have different evidence rules, so this task's status does not describe current EXP/SIGMOID support.

- [Current numerical-site and capability explanation](../../../../backend-guide/metal-backend.md#numerical-semantics-and-capability) — Explains current semantics without certifying this historical attempt.
- [Current independently typed numerical domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Locates later supported routes separately from these blocked candidates.

## Status

Blocked

The completed 83-word invocation is [regression](../../../../glossary.md#term-regression)/rejection evidence only, not complete-domain proof.
Gate 1B now leaves every previously sampled route ineligible: [opaque](../../../../glossary.md#term-opaque) direct [EXP](../../../../glossary.md#term-expx) and any composition
inheriting it are `DOMAIN-BLOCKED`, while direct/custom [SIGMOID](../../../../glossary.md#term-sigmoid) and the original custom EXP retain
their numerical failures. This remediation is [planning](../../../../glossary.md#term-planning)/evidence only. Capability, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings),
routes, and production remain unchanged, and the consumed one-shot [oracle](../../../../glossary.md#term-test-oracle) must not be rerun.

## Change class

Class C — an eventual implementation would be the first executable [Metal](../../../../glossary.md#term-metal) consumer of the completed
recursive numerical-profile contract. This [planning](../../../../glossary.md#term-planning)/evidence remediation changes no
profile-qualified capability, typed native [schema](../../../../glossary.md#term-schema)/[lowering](../../../../glossary.md#lowering), workload or tuning [identity](../../../../glossary.md#term-identity-meanings), [prepared
execution](../../../../glossary.md#prepared-execution--preparedexecution), generated-[gradient](../../../../glossary.md#term-gradient) reachability, public [Engine](../../../../glossary.md#term-engine) behavior, [Model](../../../../glossary.md#term-model) semantics, public [API](../../../../glossary.md#term-application-programming-interface-api),
[Runtime](../../../../glossary.md#term-runtime) policy, or native [ABI](../../../../glossary.md#term-application-binary-interface-abi) export.

## Goal

Define the gates for eventually adding exactly [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) `UnaryElementwiseKind.EXP` and
[`SIGMOID`](../../../../glossary.md#term-sigmoid) under `NumericalProfile.ACCELERATOR`; `STRICT_IEEE` remains false for both.

No current candidate is eligible for timing or production. Direct [MPSGraph](../../../../glossary.md#term-mpsgraph) [EXP](../../../../glossary.md#term-expx) passed the bounded
[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) but lacks an authoritative all-[binary32](../../../../glossary.md#floating-point-precision-and-rounding)/[rank](../../../../glossary.md#term-rank)-`1..16` contract and is `DOMAIN-BLOCKED`; the
sampled MPSGraph stable SIGMOID composition inherits that blocker. Direct/custom SIGMOID and the
original custom EXP also retain their recorded numerical failures. A future successor may stage the
auditable integer custom EXP and stable SIGMOID designs below only after their complete-domain
proof [artifacts](../../../../glossary.md#term-artifact) pass independent review, then [run](../../../../glossary.md#term-run) a newly authorized one-shot [oracle](../../../../glossary.md#term-test-oracle) under that
successor's own contract.

## Authorized capability domain

An [occurrence](../../../../glossary.md#term-occurrence) is supported only when all of the following hold:

- the [graph](../../../../glossary.md#graph) profile is exactly [`ACCELERATOR`](../../../../glossary.md#term-accelerator);
- kind is exactly parameterless [`EXP`](../../../../glossary.md#term-expx) or [`SIGMOID`](../../../../glossary.md#term-sigmoid) with `NoOperationAttrs.INSTANCE`;
- there is exactly one input and one output;
- both [descriptors](../../../../glossary.md#term-descriptor) are exactly [`FLOAT32`](../../../../glossary.md#term-float32), fully static positive [rank](../../../../glossary.md#term-rank) `1..16`, and every extent is
  positive with checked element and byte geometry;<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [element glossary](../../../../glossary.md#term-element))</span>
- input and output [Shapes](../../../../glossary.md#shape) and `requiresGrad` flags are equal; and
- input and output [layouts](../../../../glossary.md#layout) are [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset non-[views](../../../../glossary.md#term-view-meanings).

Both operations consume and produce canonical state. They may compose inside the existing complete
maximal accelerator [partition](../../../../glossary.md#partition) with current canonical [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), reduction, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), exact/common, and
layout operations only where the existing state rules already permit it. No affine-view operand is
newly accepted. Every other [unary](../../../../glossary.md#term-unary) kind and every strict `EXP`/`SIGMOID` occurrence remains false.

## [Model](../../../../glossary.md#term-model) result-set gate

The candidate gate implements the completed Model [accelerator](../../../../glossary.md#term-accelerator) result set, without assigning a
whole-output elementary envelope to composite [sigmoid](../../../../glossary.md#term-sigmoid). For each represented [EXP](../../../../glossary.md#term-expx) input, it evaluates
the exact represented [binary32](../../../../glossary.md#floating-point-precision-and-rounding) value with high [precision](../../../../glossary.md#term-precision), rounds once to the [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero)
binary32 reference `r`, and requires:

- [NaN](../../../../glossary.md#term-nan) input produces NaN; payload, sign, and quieting remain free;
- `EXP(-infinity)=+0`, `EXP(+infinity)=+infinity`, and either zero maps to `+1`;
- a [subnormal](../../../../glossary.md#term-subnormal) input may remain represented or be read as same-[signed zero](../../../../glossary.md#term-signed-zero) ([DAZ](../../../../glossary.md#term-daz));
- an ordinary [finite](../../../../glossary.md#term-finite) non-subnormal `r` admits only finite results whose inclusive
  ordered-binary32 distance from <span data-glossary-navigation="true"> ([ordered-binary32 distance glossary](../../../../glossary.md#ulp-and-numerical-tolerance))</span>`r` is at most five;
- a nonzero subnormal `r` admits exactly `r`, [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), or [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) ([FTZ](../../../../glossary.md#term-ftz)); and
- zero, [infinity](../../../../glossary.md#term-infinity), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), sign, and class checks remain independent of distance.

For each SIGMOID input, the gate instead enumerates the finite union of Model's recursive stable
formula results. It starts from the represented input plus its same-[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) DAZ alternative
when subnormal, applies the exact sign [guard](../../../../glossary.md#term-guard), and evaluates the selected negative
`EXP(x) -> ADD(1) -> DIV(exp,denominator)` or nonnegative
`NEG -> EXP -> ADD(1) -> DIV(1,denominator)` branch. At EXP it enumerates exactly the elementary
set above. At every explicit arithmetic site it enumerates same-signed-zero operand DAZ, one
binary32 [round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-even [operation](../../../../glossary.md#operation), and exact-subnormal-or-either-signed-zero result FTZ.
A direct or custom sigmoid candidate passes only when its represented output belongs to this union,
with independent NaN/infinity/zero-sign/range checks. It is never checked against a final
whole-sigmoid distance ceiling.

Ordered distance uses the Model key: unsigned `~bits` for a sign-set word and unsigned
`bits ^ 0x80000000` otherwise, followed by mathematical absolute key difference.

## Gate 1B: complete-domain proof prerequisite

Gate 1B is independent of the bounded [numerical oracle](../../../../glossary.md#adr-ci-and-qualification). A numerical pass can reject an incorrect
proof, but cannot prove that a candidate's outputs are a subset of the [Model](../../../../glossary.md#term-model) set for all
`2^32` input words, every positive [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank) `1..16` [Shape](../../../../glossary.md#shape), and every checked [extent](../../../../glossary.md#term-extent). An [opaque](../../../../glossary.md#term-opaque)
route requires an authoritative, deployment-version-applicable contract covering raw-word special
classes, [DAZ](../../../../glossary.md#term-daz), [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) reference/distance, [underflow](../../../../glossary.md#term-underflow)/[overflow](../../../../glossary.md#term-overflow), shape-dependent algorithms,
and exact output [publication](../../../../glossary.md#term-publication). Selector names, [SDK](../../../../glossary.md#term-sdk) declarations, [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), and final-output agreement
are insufficient.

| Candidate from the consumed invocation | Gate 1A/[API](../../../../glossary.md#term-application-programming-interface-api) | Gate 1B | Bounded numerical verdict | Consequence |
|---|---|---|---|---|
| direct [MPSGraph](../../../../glossary.md#term-mpsgraph) [EXP](../../../../glossary.md#term-expx) | `PASS` | `DOMAIN-BLOCKED` — no authoritative all-[binary32](../../../../glossary.md#floating-point-precision-and-rounding)/rank-`1..16` selector contract | `PASS` on 83 words | [regression](../../../../glossary.md#term-regression) evidence only; not eligible for timing or production |
| original custom EXP | `PASS` | `DOMAIN-BLOCKED` — no reviewed complete source/[constant](../../../../glossary.md#term-constant)/error proof | `FAIL` | rejected; remains removed |
| direct MPSGraph [SIGMOID](../../../../glossary.md#term-sigmoid) | `PASS` | `DOMAIN-BLOCKED` — opaque direct formula/algorithm | `FAIL` | rejected; remains removed |
| original custom stable SIGMOID | `PASS` | `DOMAIN-BLOCKED` — no complete primitive/formula proof | `FAIL` | rejected; remains removed |
| MPSGraph stable SIGMOID composition | `PASS` | `DOMAIN-BLOCKED` — inherits direct MPSGraph EXP's blocker | `PASS` on 83 words | regression evidence only; not eligible for timing or production |

No candidate from the consumed invocation is `DOMAIN-PASS`. Consequently the prior timing stop is
not only an instrumentation stop: even with [trace](../../../../glossary.md#trace) tooling installed, Gate 3 cannot begin until a
new candidate passes Gate 1B and a separately authorized fresh numerical gate.

### Successor-only auditable custom candidate design

`CUSTOM_EXP_CR_U32_V1` is a fixed typed [pointwise](../../../../glossary.md#term-pointwise) [kernel](../../../../glossary.md#kernel) over raw [`uint32`](../../../../glossary.md#index-mappings-and-cost-dominance) words, not a generalized
transcendental framework. It calls no [MPSGraph](../../../../glossary.md#term-mpsgraph) selector, [Metal](../../../../glossary.md#term-metal) transcendental, native `float`
[operation](../../../../glossary.md#operation), or [opaque](../../../../glossary.md#term-opaque) library routine:

1. Decode sign/exponent/fraction bits and handle every NaN, both infinities, and both zeros by
   explicit raw-word branches. Emit one fixed quiet NaN for every NaN, <span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [quiet NaN glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`+0` for [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero),
   `+infinity` for [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero), and `+1` for either zero.
2. Choose the represented finite input, never the optional DAZ alternative. This is one allowed
   Model input path, including represented subnormal inputs.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
3. Convert that exact binary rational into a frozen signed multi-limb fixed-point representation.
   Use only explicitly implemented <span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic))</span>`uint32` add/subtract/multiply/shift/compare operations with
   checked shift counts, proved carry bounds, and no signed overflow or implementation-defined
   behavior.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
4. Select `k` with certified outward rational bounds for `ln(2)/256`, then enclose
   `r = x - k * ln(2)/256` without treating the irrational constant as exactly represented. Frozen
   outward-rounded bounds, a 256-entry `2^(j/256)` interval table, and a fixed polynomial for
   `exp(r)` produce a directed lower/upper result interval. The proof generator selects and freezes
   limb width, polynomial degree, coefficients, evaluation order, and every truncation direction;
   the kernel may not substitute compiler floating arithmetic.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [compiler glossary](../../../../glossary.md#term-compiler))</span>
5. If the interval identifies one correctly rounded binary32 result, pack that raw word directly.
   Every unresolved rounding-boundary input uses a frozen sorted exception table keyed by the exact
   input word. Table generation must prove exhaustiveness and uniqueness; an unbounded or
   unexplained exception set rejects the candidate.<span data-glossary-navigation="true"> ([correctly rounded glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [rounding glossary](../../../../glossary.md#term-rounding))</span>
6. Handle overflow and gradual underflow through proved exact raw thresholds and integer packing.
   Emit the correctly rounded normal/subnormal/zero/infinity word directly, so the chosen route uses
   no output FTZ. Correct rounding is ordered distance <span data-glossary-navigation="true"> ([gradual underflow glossary](../../../../glossary.md#term-gradual-underflow); [Correct rounding glossary](../../../../glossary.md#term-correct-rounding))</span>`0`, a strict subset of the Model's inclusive
   five-step ordinary-result allowance; exact subnormal `r` is one allowed [subnormal](../../../../glossary.md#term-subnormal) result.

`CUSTOM_SIGMOID_STABLE_U32_V1` composes only reviewed raw-word primitives. It handles [NaN](../../../../glossary.md#term-nan) explicitly,
uses the raw sign [guard](../../../../glossary.md#term-guard), and evaluates the negative branch
`EXP(x) -> ADD_RNE(1,e) -> DIV_RNE(e,d)` or nonnegative branch
`NEG_BITS(x) -> EXP -> ADD_RNE(1,e) -> DIV_RNE(1,d)`. [`EXP`](../../../../glossary.md#term-expx) is exactly
`CUSTOM_EXP_CR_U32_V1`; `ADD_RNE` and `DIV_RNE` are fixed total software-[binary32](../../../../glossary.md#floating-point-precision-and-rounding) integer
algorithms with explicit NaN/[infinity](../../../../glossary.md#term-infinity)/zero rules, [normalization](../../../../glossary.md#term-normalization), guard/round/sticky bits, and
[ties-to-even](../../../../glossary.md#term-ties-to-even) packing. They choose represented operands and exact non-[FTZ](../../../../glossary.md#term-ftz) results, one legal
[DAZ](../../../../glossary.md#term-daz)/RNE/FTZ path at each [Model](../../../../glossary.md#term-model) site. Every intermediate is materialized as a raw binary32 word in a
guarded [buffer](../../../../glossary.md#term-buffer); the proof never relies on opaque [fusion](../../../../glossary.md#term-fusion) or excess [precision](../../../../glossary.md#term-precision).

The reviewable proof package for these two candidates has mandatory, independently checked
obligations:

1. freeze source, generator, coefficient/table/exception bytes, proof checker, compiler options,
   resource ledger, and SHA-256 identities before any device process;<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
2. prove raw classification and all special outputs, including both zeros, both infinities, every
   NaN word, overflow, underflow, and exact subnormal publication;<span data-glossary-navigation="true"> ([underflow glossary](../../../../glossary.md#term-underflow); [publication glossary](../../../../glossary.md#term-publication))</span>
3. prove every multi-limb primitive, carry bound, shift precondition, table index, and buffer offset;
   for element count <span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>$N=\prod_{i=0}^{rank-1}d_i$, prove checked positive multiplication and the
   launch map $\ell=gid_x+W(gid_y+Hgid_z)$ covers every integer $0\le\ell<N$ exactly once and no
   other thread writes, for every resource-feasible positive rank-<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16` [Shape](../../../../glossary.md#shape);
4. prove range-reduction uniqueness and coverage, outward interval validity, approximation and
   truncation error, exception-table exhaustiveness, and final round-to-nearest-even packing;<span data-glossary-navigation="true"> ([round-to-nearest glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
5. independently enumerate all `2^32` raw inputs with a separate at-least-256-bit reference/checker,
   recording class counts, maximum ordered distance and witness, unresolved interval count,
   exception count, and a certificate digest; EXP must have zero failures and maximum distance<span data-glossary-navigation="true"> ([certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions); [digest glossary](../../../../glossary.md#term-digest))</span>
   `<=5`;
6. prove `ADD_RNE` and `DIV_RNE` against their complete raw-word truth tables or equivalent checked
   partitions, then prove stable SIGMOID site-by-site recursive membership for all <span data-glossary-navigation="true"> ([SIGMOID glossary](../../../../glossary.md#term-sigmoid))</span>`2^32` inputs,
   including exact branch selection and materialized intermediate words;
7. establish that the Metal source uses only the proved unsigned operations, fixed constants, and
   safe compile settings, with no optimizer-enabled floating reassociation or hidden intrinsic;<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [optimizer glossary](../../../../glossary.md#term-optimizer); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
8. independently review the source-to-certificate correspondence and rerun the proof checker from a
   clean environment before assigning `DOMAIN-PASS`; and
9. keep capability false and stage both candidates outside the repository until all obligations
   pass.

This document can statically prove the current direct/inherited routes are domain-blocked, preserve
the recorded numerical failures, and define a sufficient auditable proof shape. It cannot claim the
new designs are `DOMAIN-PASS` until concrete source/[constants](../../../../glossary.md#term-constant)/certificates exist and are reviewed.
That work requires a new explicitly authorized successor task. Because Task 0051's sole [device](../../../../glossary.md#term-device)
invocation is consumed, the successor must own any fresh compiled-device [regression](../../../../glossary.md#term-regression) [oracle](../../../../glossary.md#term-test-oracle); it must
not rerun, amend, or relabel the Task-0051 invocation.

## Consumed candidate staging and exactly one [oracle](../../../../glossary.md#term-test-oracle)

The completed capability-false invocation staged these historical implementations:

1. direct [MPSGraph](../../../../glossary.md#term-mpsgraph) `exponentWithTensor` for [`EXP`](../../../../glossary.md#term-expx);
2. one disposable custom [Metal](../../../../glossary.md#term-metal) `float` EXP [kernel](../../../../glossary.md#kernel);
3. direct MPSGraph [sigmoid](../../../../glossary.md#term-sigmoid) for `SIGMOID`;
4. one disposable custom stable-branch sigmoid kernel; and
5. one MPSGraph stable sigmoid composition using explicit negation/exponential/addition/division
   sites and graph constants.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant))</span>

The disposable oracle was outside the repository and exported no [ABI](../../../../glossary.md#term-application-binary-interface-abi). It compiled all five
candidates first, then performed exactly one real-[device](../../../../glossary.md#term-device) process invocation with one default-[device
context](../../../../glossary.md#device-submission-synchronization-and-execution-context), direct supplied [targets](../../../../glossary.md#term-target), recorded optimization/fast-math state, input-preservation and
target-[canary](../../../../glossary.md#term-canary) checks, and the same frozen corpus for every candidate. There was no retry, second
context, optimization-level [matrix](../../../../glossary.md#term-matrix-meanings), route fallback, or repeated oracle. That invocation is now
consumed permanently.

The frozen corpus is the ordered de-duplicated union of:

```text
00000000 80000000 00000001 80000001
007fffff 807fffff 00800000 80800000 00800001 80800001
3f000000 bf000000 3f800000 bf800000 40000000 c0000000
7f7fffff ff7fffff 7f800000 ff800000
7fc12345 ffc54321 7fa12345 ffa54321
```

and each listed base word plus its immediately adjacent raw words when [finite](../../../../glossary.md#term-finite):

```text
c2d00000 c2cff1b5 c2cfcccd c2ce0000 c2b40000 c2b00000 c2af0000
c1a00000 c0e80000 bf900000 bf800000 bf000000
3f000000 3f800000 40e80000 41a00000 42a00000 42b00000 42b17218
42b20000 42d00000
```

Those words cover signed zeros, signed minimum/maximum [subnormals](../../../../glossary.md#term-subnormal), the normal boundary, ordinary
small/large values, retained historical `-88`, `-7.25`, and `-1.125` observations, EXP [underflow](../../../../glossary.md#term-underflow) and
[overflow](../../../../glossary.md#term-overflow) neighborhoods, maximum finite values, both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and multiple quiet/[signaling NaN](../../../../glossary.md#term-signaling-nan)
signs/payloads. The reference evaluates the exact represented [binary32](../../../../glossary.md#floating-point-precision-and-rounding) value with at least 160
significant decimal digits, performs stable-branch sigmoid, and resolves final binary32
[round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-even by comparing adjacent exact binary32 values rather than by double [rounding](../../../../glossary.md#term-rounding).

The raw oracle report names every historical candidate, corpus [SHA-256](../../../../glossary.md#term-sha-256), output SHA-256, maximum
ordered distance and its input, [FTZ](../../../../glossary.md#term-ftz) count, class/sign/range verdicts, canary/input/[Shape](../../../../glossary.md#shape) verdicts,
and bounded `PASS`/`FAIL`. A candidate could proceed only with both `DOMAIN-PASS` and numerical
`PASS`; none did. The task Result retains the unedited raw verdict text and SHA-256; all disposable
[artifacts](../../../../glossary.md#term-artifact) were removed.

## One bounded route adjudication

**Not authorized:** no current candidate is `DOMAIN-PASS`, the Task-0051 [oracle](../../../../glossary.md#term-test-oracle) is consumed, and
supported [trace](../../../../glossary.md#trace) tooling is also unavailable. This workload did not [run](../../../../glossary.md#term-run) and must not run under Task
0051.

The unexecuted historical adjudication specification was:

```text
profile = ACCELERATOR
input = canonical caller FLOAT32 [1_048_576]
outputs = EXP(input), SIGMOID(input)
optimization = production MPSGraphCompilationDescriptor default
reducedPrecisionFastMath = None with readback
corpus = frozen edge prefix, then Float.floatToRawIntBits((((i * 17) % 257) - 128) / 16.0f)
```

It would have prepared every surviving route before timing and exact-checked it against the
recursive-floor gate.
It would have used four alternating-order [warmup](../../../../glossary.md#term-warmup) rounds and eight alternating-order retained rounds
in one process; each retained [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) would run to a 25 ms floor with a 1,000,000-execution ceiling.
Nothing would be discarded or retried. The hot boundary would be [synchronous](../../../../glossary.md#term-synchronous) route execution plus
result closure only, excluding preparation, upload, download, oracle checks, and reporting.

It would have recorded every raw batch duration, iteration count, normalized duration, and
descriptive [median](../../../../glossary.md#term-median). After uncaptured timing, `MTLCaptureManager` queue [capture](../../../../glossary.md#term-capture-meanings) would cover exactly
one hot logical execution of each passing route in the same workload. Supported installed tooling
would have to read each trace's [SHA-256](../../../../glossary.md#term-sha-256), actual compute-[dispatch](../../../../glossary.md#term-dispatch) records, and peak
transient/temporary [Metal](../../../../glossary.md#term-metal) resource bytes above steady prepared inputs/outputs. A custom resource
ledger is only a cross-check; framework-internal work cannot be relabeled as a single [backend](../../../../glossary.md#backend) call
or reported only as [opaque](../../../../glossary.md#term-opaque). Missing actual dispatch or peak temporary facts blocks capability.

The corrected Task-0051 selection result is empty:

1. remove every candidate without both Gate-1B `DOMAIN-PASS` and numerical `PASS`;
2. all five historical candidates are removed at step 1, so Task 0051 has no baseline, alternative,
   timing population, tie, route winner, or production decision; and
3. a new explicitly authorized successor must predeclare its own complete-domain candidates,
   one-shot numerical gate, all-survivor cost rule, and deterministic tie-break before measurement.

Any future decision is fixed in production, never a [runtime](../../../../glossary.md#term-runtime) threshold, tuning [matrix](../../../../glossary.md#term-matrix-meanings), retry,
fallback, or input-dependent choice, and removes every unselected disposable candidate [artifact](../../../../glossary.md#term-artifact).

## Typed [schema](../../../../glossary.md#term-schema) and [identity](../../../../glossary.md#term-identity-meanings) cutover

- Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version `4`, the fixed 160-byte record, statuses, signatures, complete-plan
  wrapper schema <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`1`, and exactly thirteen exports unchanged.
- Advance [node](../../../../glossary.md#node) schema `11 -> 12`; retain wires `1..19`; append no-attribute [canonical](../../../../glossary.md#term-canonical-form) [unary](../../../../glossary.md#term-unary)
  `EXP=20` and `SIGMOID=21`; require `UINT32_MAX` second-input/axis sentinels, zero auxiliary,
  zero attribute count/payload, equal input/output Shape, and FLOAT32 canonical state.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Shape glossary](../../../../glossary.md#shape); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- Java and native [preflight](../../../../glossary.md#term-preflight) independently reject strict profile use, stale schema 11, unknown wire
  `22+`, wrong attrs/sentinel/padding/type/state/topology/Shape/rank, and malformed or overflowed
  geometry before resource creation.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [rank glossary](../../../../glossary.md#term-rank))</span>
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and tuning codec together <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`12 -> 13`. Hash profile, schema, ordered unary topology,
  descriptors/states, feeds/targets, splats, ABI, and route facts. Version-12 bytes safely miss or
  reject; no alias, dual decoder, migration shim, or fallback remains.<span data-glossary-navigation="true"> ([Hash glossary](../../../../glossary.md#term-hash); [descriptors glossary](../../../../glossary.md#term-descriptor); [targets glossary](../../../../glossary.md#term-target); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

If adjudication selects direct [MPSGraph](../../../../glossary.md#term-mpsgraph) for both operations, the existing whole-[partition](../../../../glossary.md#partition)
`MPSGRAPH` candidate is the only production route for them and the existing NEG-only custom
candidate domain remains unchanged. Do not retain candidate enum values for discarded probes.

## Production [lifecycle](../../../../glossary.md#lifecycle) and [gradients](../../../../glossary.md#term-gradient)

Preserve current whole-[partition](../../../../glossary.md#partition) analysis, stable [node](../../../../glossary.md#node)/value/feed/[target](../../../../glossary.md#term-target) order, complete candidate
[batches](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), authenticated optional decisions, declarations-before-assignment, transactional
finalization, context child [leases](../../../../glossary.md#term-lease), [immutable](../../../../glossary.md#term-immutable) splats, direct assigned targets, [run](../../../../glossary.md#term-run)-owned outputs and
address workspace, [synchronous](../../../../glossary.md#term-synchronous) invocation, reuse, [concurrency](../../../../glossary.md#term-concurrency) isolation, [rollback](../../../../glossary.md#term-rollback), close
rejection, [publication](../../../../glossary.md#term-publication), transfer, tracing, and session-scoped tuning behavior. Add no hot-path
numerical check, retry, fallback, secondary route, per-node invocation, [host](../../../../glossary.md#term-host) staging, or hidden
[materialization](../../../../glossary.md#materialization).

[Compiler](../../../../glossary.md#term-compiler) generation is unchanged. Prove a seeded [`EXP`](../../../../glossary.md#term-expx) gradient is all [Metal](../../../../glossary.md#term-metal) under
[`ACCELERATOR`](../../../../glossary.md#term-accelerator): forward `EXP` and generated `seed * output` use only the new EXP wire and existing
[canonical](../../../../glossary.md#term-canonical-form) MUL. Inspect and execute the generated [topology](../../../../glossary.md#term-topology), not a handwritten substitute.

Attempt the generated seeded [`SIGMOID`](../../../../glossary.md#term-sigmoid) gradient
`seed * output * (one - output)` without [widening](../../../../glossary.md#term-widening) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) capability. If the current compiler emits
an affine/[view](../../../../glossary.md#term-view-meanings) or other unsupported boundary, keep explicit mixed [ownership](../../../../glossary.md#term-ownership-meanings) and document exact
partition [owners](../../../../glossary.md#term-owner)/transfers; do not broaden affine-to-binary or [scalar](../../../../glossary.md#term-scalar) capability. If the existing
canonical/splat domain already closes it, prove the all-Metal topology instead.

## Non-goals

No strict [EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid); other [unary](../../../../glossary.md#term-unary) kind; [scalar](../../../../glossary.md#term-scalar) [operation](../../../../glossary.md#operation); new [dtype](../../../../glossary.md#data-type--datatype); [rank](../../../../glossary.md#term-rank) zero; zero/dynamic [Shape](../../../../glossary.md#shape);
[view](../../../../glossary.md#term-view-meanings) ingress; [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/affine capability [widening](../../../../glossary.md#term-widening); [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) [gradient](../../../../glossary.md#term-gradient) change; [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace)
profile state; async work; [device](../../../../glossary.md#term-device) selection; pooling; package/discovery change; [ABI](../../../../glossary.md#term-application-binary-interface-abi)/export/status;
persistent tuning format; runtime numerical check; retry; fallback; benchmark [matrix](../../../../glossary.md#term-review-matrix); generalized
custom-unary framework; retained unselected [kernel](../../../../glossary.md#kernel); or final repository-wide build.

## Dependencies and integration

- Depends on: completed and independently approved [Model](../../../../glossary.md#term-model) 0030/0031 at remediation
  `97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`; Metal 0050 packaged baseline; retained Metal 0016
  device evidence; Compiler unary gradients; Config 0006; Engine 0018.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [device glossary](../../../../glossary.md#term-device); [Compiler glossary](../../../../glossary.md#term-compiler); [unary glossary](../../../../glossary.md#term-unary); [gradients glossary](../../../../glossary.md#term-gradient); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Original execution base revision: `621555a500355d4f0a4b32a54bf7d828331e12f0`; Gate-1B
  remediation base: `6efe3b859ebcb84ffdc05e8925d7bbb9da52782d`.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None.
- Integration order reached: original planning; capability-false historical staging; exactly one
  consumed oracle; artifact removal; Gate-1B evidence correction. A new explicitly authorized
  successor must own concrete custom source/certificates, independent complete-domain review, its
  own one-shot oracle, instrumentation-enabled adjudication, route freeze, production work, and
  independent Class C review.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [oracle glossary](../../../../glossary.md#term-test-oracle); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer.

## Files and symbols

Expected production [owners](../../../../glossary.md#term-owner):

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`,
  `MetalNegPartitionPreparer`, and affected package Javadocs;
- `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`, `MetalNegTuningCodec`, and current
  decision/complete-plan compatibility tests;
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m` and its README;
- focused Metal capability/schema/native/prepared/candidate/codec tests, backend conformance, and
  public Engine integration;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Metal](../../../../glossary.md#term-metal)/[backend](../../../../glossary.md#backend)/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary explanatory docs, this task, Metal master, and roadmap.

No probe, [oracle](../../../../glossary.md#term-test-oracle) output, route report, custom [kernel](../../../../glossary.md#kernel), benchmark [artifact](../../../../glossary.md#term-artifact), or [cache](../../../../glossary.md#term-cache) remains unless the
selection rule chooses it as production.

## Acceptance criteria

1. The consumed 83-word oracle remains exact regression/rejection evidence with unchanged raw
   hashes/verdicts; it is not complete-domain certification and is never rerun.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [regression glossary](../../../../glossary.md#term-regression))</span>
2. Every historical candidate has a separate Gate-1B verdict. Direct EXP and its MPSGraph stable
   SIGMOID composition are <span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [SIGMOID glossary](../../../../glossary.md#term-sigmoid))</span>`DOMAIN-BLOCKED`; the original custom EXP and both failed SIGMOID
   candidates retain their numerical failures.
3. The successor-only custom EXP and stable SIGMOID designs name all source, fixed-point,
   special-value, approximation, exception, arithmetic-site, full-raw-domain, rank/launch,
   certificate, and independent-review obligations without claiming those unbuilt candidates pass.<span data-glossary-navigation="true"> ([fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic); [rank glossary](../../../../glossary.md#term-rank); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
4. No timing, trace capture, route decision, production staging, build, native invocation, or test
   runs in this remediation.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
5. Capability, [schema](../../../../glossary.md#term-schema) 11, wires `1..19`, version-twelve identities, ABI 4, thirteen exports,
   production code, and every strict/excluded row remain unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
6. Task/master/roadmap wording, Markdown, path [scope](../../../../glossary.md#term-scope), [artifact](../../../../glossary.md#term-artifact) absence, and `git diff --check` pass;
   the remediation is committed for independent planning/evidence review.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## [Validation](../../../../glossary.md#term-validation)

Validate only the [planning](../../../../glossary.md#term-planning)/evidence remediation:

```bash
git diff --check
```

Also validate the three changed Markdown files for links, anchors, fences, final newlines, trailing
whitespace, status consistency, exact retained [oracle](../../../../glossary.md#term-test-oracle) [hashes](../../../../glossary.md#term-hash), no Task-0052 task-file edit, and zero
production/native/test/generated [artifacts](../../../../glossary.md#term-artifact). Do not [run](../../../../glossary.md#term-run) a build, test, native process, timing
workload, [trace](../../../../glossary.md#trace) [capture](../../../../glossary.md#term-capture-meanings), or [device](../../../../glossary.md#term-device) oracle.

## Documentation and review impact

Synchronize only this task's Gate-1B correction and the Task-0051 summaries in the [Metal](../../../../glossary.md#term-metal) master plan
and roadmap. Preserve the original raw [oracle](../../../../glossary.md#term-test-oracle) evidence, every historical blocked record, and
Task-0052 content under independent review. A future successor's independent Class C review must
inspect concrete custom source/[constants](../../../../glossary.md#term-constant)/certificates, complete-domain proofs, its newly authorized
one-shot numerical evidence, instrumentation-backed cost evidence, fixed route, false exclusions,
stale-data rejection, [lifecycle](../../../../glossary.md#lifecycle), and removed [artifacts](../../../../glossary.md#term-artifact).

## Result

Blocked before complete-domain authorization, route timing, or capability. [Planning](../../../../glossary.md#term-planning) completed at
`3a81874528b9a3cbcca4a5643c121bc164c2f75a`; this Gate-1B remediation corrects the interpretation of
its evidence. Production capability remained false throughout, and no Java, native, [schema](../../../../glossary.md#term-schema),
[identity](../../../../glossary.md#term-identity-meanings), test, guide, [Javadoc](../../../../glossary.md#term-javadoc), [ABI](../../../../glossary.md#term-application-binary-interface-abi), export, package, or generated [artifact](../../../../glossary.md#term-artifact) changed.

Exactly one fresh Apple M3 Max numerical-[oracle](../../../../glossary.md#term-test-oracle) process invocation ran with the frozen 83-word
corpus, [MPSGraph](../../../../glossary.md#term-mpsgraph)'s production-default optimization level `1`,
`MPSGraphReducedPrecisionFastMathNone` (`0`), custom `MTLMathModeSafe` (`0`), direct supplied
[targets](../../../../glossary.md#term-target), input preservation, and target canaries. Oracle source [SHA-256](../../../../glossary.md#term-sha-256) was
`b25723c322d9a3078272ea0e79d090d296312186b0e1563b8021f6b7c67e865a`; executable SHA-256 was
`35432d9124478d01a1729ab01d4d2a10778fbf6a8bc87a79b02b0c8b6b881e8f`; raw [device](../../../../glossary.md#term-device)-output SHA-256
was `3b762b96b3a1a57aecb97384362069ac2cc9187d271103bc333e841be4274e01`. The 200-digit exact
reference/recursive-membership validator SHA-256 was
`e851ad5c9feabec8282052d458cd40b1efeacddada7723f1292dd2025060794f`; its raw verdict SHA-256 was
`6ac29b44e9a1a693f0e4822ddf6165ef690151f16b63b7f1110434b007be17db`.

The raw recursive verdict was:

```text
REFERENCE_PRECISION_DIGITS=200
CORPUS_COUNT=83
CORPUS_SHA256=06e9137c3b6a9ba147f0a0b73c7face1f5b8cf15ef150964040b64f9be2cad25
RAW_OUTPUT_SHA256=3b762b96b3a1a57aecb97384362069ac2cc9187d271103bc333e841be4274e01
CANDIDATE=direct_exp VERDICT=PASS MAX_ORDERED_DISTANCE=1 MAX_INPUT=3f800000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=0
CANDIDATE=custom_exp VERDICT=FAIL MAX_ORDERED_DISTANCE=53 MAX_INPUT=42b17217 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=9
FAIL name=custom_exp input=c19fffff actual=310da44b allowed={822977600, 822977601, 822977602, 822977603, 822977604, 822977605, 822977606, 822977607, 822977608, 822977609, 822977599}
FAIL name=custom_exp input=419fffff actual=4de7581e allowed={1307007010, 1307007011, 1307007012, 1307007013, 1307007014, 1307007015, 1307007016, 1307007017, 1307007018, 1307007019, 1307007020}
FAIL name=custom_exp input=41a00001 actual=4de7585a allowed={1307007072, 1307007073, 1307007074, 1307007075, 1307007076, 1307007077, 1307007078, 1307007068, 1307007069, 1307007070, 1307007071}
FAIL name=custom_exp input=429fffff actual=792abb5b allowed={2032843635, 2032843636, 2032843637, 2032843638, 2032843639, 2032843640, 2032843641, 2032843642, 2032843643, 2032843644, 2032843645}
FAIL name=custom_exp input=42a00001 actual=792abc0c allowed={2032843808, 2032843809, 2032843810, 2032843811, 2032843812, 2032843813, 2032843814, 2032843815, 2032843816, 2032843806, 2032843807}
FAIL name=custom_exp input=42afffff actual=7ef88243 allowed={2130215488, 2130215478, 2130215479, 2130215480, 2130215481, 2130215482, 2130215483, 2130215484, 2130215485, 2130215486, 2130215487}
FAIL name=custom_exp input=42b00000 actual=7ef88299 allowed={2130215602, 2130215603, 2130215604, 2130215605, 2130215606, 2130215607, 2130215608, 2130215609, 2130215610, 2130215611, 2130215612}
FAIL name=custom_exp input=42b00001 actual=7ef88346 allowed={2130215726, 2130215727, 2130215728, 2130215729, 2130215730, 2130215731, 2130215732, 2130215733, 2130215734, 2130215735, 2130215736}
CANDIDATE=direct_sigmoid VERDICT=FAIL RECURSIVE_MEMBERSHIP=FAIL MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=2
FAIL name=direct_sigmoid input=bf900000 actual=3e7af78f allowed=3e7af78b,3e7af78c,3e7af78e,3e7af790,3e7af791,3e7af793,3e7af794,3e7af796
FAIL name=direct_sigmoid input=bf900001 actual=3e7af78f allowed=3e7af789,3e7af78b,3e7af78c,3e7af78e,3e7af790,3e7af791,3e7af793,3e7af794,3e7af796
CANDIDATE=custom_sigmoid VERDICT=FAIL RECURSIVE_MEMBERSHIP=FAIL MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=1
FAIL name=custom_sigmoid input=c19fffff actual=310da44b allowed=310da43f,310da440,310da441,310da442,310da443,310da444,310da445,310da446,310da447,310da448,310da449
CANDIDATE=composed_sigmoid VERDICT=PASS RECURSIVE_MEMBERSHIP=PASS MAX_SET_SIZE=11 MAX_SET_INPUT=c0000000 FTZ_COUNT=16 CLASS_SIGN_RANGE=PASS FAILURES=0
PASSING_CANDIDATES=direct_exp,composed_sigmoid
ORACLE_VERDICT=PASS
```

The composed [SIGMOID](../../../../glossary.md#term-sigmoid) numerical validator selected captured device results from the negative or
nonnegative stable branch with the exact raw-bit sign [guard](../../../../glossary.md#term-guard), then checked membership in the sampled
recursive site-by-site union. Direct SIGMOID was not given a final elementary distance envelope.
These are bounded numerical facts only: direct [EXP](../../../../glossary.md#term-expx) and composed SIGMOID passed 83 words, while the
original custom EXP/custom SIGMOID/direct SIGMOID failed. Direct EXP remains `DOMAIN-BLOCKED`, so
the composition inheriting it is also `DOMAIN-BLOCKED`; neither sampled pass is certified for all
[binary32](../../../../glossary.md#floating-point-precision-and-rounding) inputs, ranks `1..16`, or positive checked extents.

The required actual [GPU](../../../../glossary.md#term-gpu) [dispatch](../../../../glossary.md#term-dispatch) and peak transient-memory instrumentation was unavailable before
timing. The installed command-line tools produced these raw results:

```text
$ xcrun --sdk macosx xctrace list templates
xcrun: error: unable to find utility \"xctrace\", not a developer tool or in PATH
exit=72

$ /usr/bin/xctrace list templates
xcode-select: error: tool 'xctrace' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance
exit=1
```

`MTLCaptureManager` can describe a GPU-[trace](../../../../glossary.md#trace) destination, but this workstation has no supported
installed trace reader capable of extracting actual framework-internal compute dispatches and peak
transient resource bytes. A production-equivalent Level-1/fast-math-none adjudication harness
compiled successfully (source SHA-256
`0ad83c650d00c24c74e225c7d7502fee2c293f521d68747f538ceecd859d3b12`, executable SHA-256
`4fb094ec5ea9472af53adca8272bb5aafcdbe6d08e59f32c52f4f8768503a762`) but was deliberately never
[run](../../../../glossary.md#term-run), so there is no timing, inferred [backend](../../../../glossary.md#backend)-call count, [opaque](../../../../glossary.md#term-opaque)-memory substitute, route winner, or
capability authorization. All disposable oracle, validator, raw-output, verdict, adjudicator source,
and executable artifacts were removed after this evidence was recorded.

The static remediation adds no candidate source and consumes no device action. It proves only the
current Gate-1B blocker/failure classification and freezes the proof obligations for
`CUSTOM_EXP_CR_U32_V1` and `CUSTOM_SIGMOID_STABLE_U32_V1`. Concrete generation, exhaustive
[certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) production, independent proof review, and any compiled-device [regression](../../../../glossary.md#term-regression) require a new
explicitly authorized successor task with its own one-shot rule. Even after such a successor obtains
`DOMAIN-PASS` and numerical `PASS`, actual GPU dispatch and peak transient-memory facts remain
mandatory before timing can select a route or capability can change.

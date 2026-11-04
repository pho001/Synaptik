# Task 0053: Certified [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) Custom [EXP](../../../../glossary.md#term-expx) and Stable [SIGMOID](../../../../glossary.md#term-sigmoid)

## Reader context

This preserves an unfinished certification route, not proof that no Metal implementation is possible. The custom shader manipulates integer binary32 encodings; its formal model and independent checker are different participants. Exhaustive checker agreement did not supply the missing theorem connecting the candidate to mathematical exponential.

The constructive-real bridge and source-to-proof obligations are blockers for this candidate. Dormant schema vocabulary did not authorize execution. Later profile-free qualification is separate: current EXP/SIGMOID support does not retroactively grant `DOMAIN-PASS` to these proofs or revive their device authorization.

- [Current numerical implementations and original carriers](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Separates executable current routes from the blocked proof candidate.
- [Current historical-certificate interpretation](../../../../backend-guide/metal-backend.md#evidence-composition) — Explains why old certificates alone cannot qualify changed source.

## Status

Blocked

The retained [finite](../../../../glossary.md#term-finite) [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) and integer-only raw-word candidate passed their structural,
algebraic, source-audit, and independent all-`2^32` corroboration [checkpoints](../../../../glossary.md#term-checkpoint), but no pinned
no-axiom constructive-real exponential bridge is available to prove the required Taylor
enclosures, exponential [identities](../../../../glossary.md#term-identity-meanings), [range reduction](../../../../glossary.md#range-reduction-and-approximation), reconstruction error, and [binary32](../../../../glossary.md#floating-point-precision-and-rounding) [rounding](../../../../glossary.md#term-rounding).
The independent checkpoint review found no P0/P1/P2 issue and explicitly did not grant
`DOMAIN-PASS`. Production capability, compiled-[MSL](../../../../glossary.md#term-metal-shading-language-msl) audit, and [device](../../../../glossary.md#term-device) execution therefore remain
[fail-closed](../../../../glossary.md#term-fail-closed). Work may [resume](../../../../glossary.md#schema-serialization-and-checkpoint) only when that external proof prerequisite exists; Task 0057 is the
active serialized [Metal](../../../../glossary.md#term-metal) frontier.

The concrete custom candidate uses raw-bit classification and integer/[fixed-point arithmetic](../../../../glossary.md#fixed-point-arithmetic) only;
the native boundary may load and store raw words but may not rely on ungrounded [GPU](../../../../glossary.md#term-gpu) [FLOAT32](../../../../glossary.md#term-float32)
rounding, [fused](../../../../glossary.md#specialization-and-fusion)-multiply-add, [contraction](../../../../glossary.md#contraction-dimension), [denormal](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow), or fast-math behavior. [MPSGraph](../../../../glossary.md#term-mpsgraph) direct
[`EXP`](../../../../glossary.md#term-expx)/[`SIGMOID`](../../../../glossary.md#term-sigmoid) and the inherited stable composition remain structural catalog candidates only.
Selector presence, headers, bounded [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), and timing grant no correctness approval, so no
MPSGraph candidate may be forced unless it independently passes the same complete-domain proof.

Production capability remains gated on retained source/[constants](../../../../glossary.md#term-constant)/manifests, proof-[kernel](../../../../glossary.md#kernel) replay,
an independent exhaustive or partitioned all-word checker with zero unresolved inputs, exact
source-to-proof/[compiler](../../../../glossary.md#term-compiler)-site audit, [rank](../../../../glossary.md#term-rank)-`1..16` launch proof, recursive stable-SIGMOID proof,
one proof-first device [smoke](../../../../glossary.md#term-smoke-test), and independent Class C approval. A single correct custom survivor
needs no cost comparison and may become the one [immutable](../../../../glossary.md#term-immutable) production route. Local timing is never
[run](../../../../glossary.md#term-run) and has no [qualification](../../../../glossary.md#term-qualification), route-selection, tuning-[identity](../../../../glossary.md#term-identity-meanings), or [cache](../../../../glossary.md#term-cache) authority.

## Change class and [scope](../../../../glossary.md#term-scope)

Class C — execution adds [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) under
`NumericalProfile.ACCELERATOR` only if the proof and review gates close. It consumes existing
[schema](../../../../glossary.md#term-schema)-13 [operation](../../../../glossary.md#operation) wires and introduces no public knob, hot selection, fallback, retry, or
[autotuning](../../../../glossary.md#autotuning--model-autotuning) path.

In scope for this execution:

- one auditable custom `EXP` candidate over raw [binary32](../../../../glossary.md#floating-point-precision-and-rounding) words;
- one stable custom `SIGMOID` candidate proved recursively from an exact [guard](../../../../glossary.md#term-guard) and `NEG`, the proved
  `EXP`, and explicit `ADD`/`DIV` sites;
- any [MPSGraph](../../../../glossary.md#term-mpsgraph) or composed alternative that independently earns complete-domain `DOMAIN-PASS`
  before candidate freeze; and
- one proof-first oracle, survivor adjudication under ADR 0022, one fixed-route decision only when
  authorized, then a separately reviewed production implementation.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

Out of scope now and later unless a new task says otherwise: strict-profile `EXP`/`SIGMOID`, any
other [unary](../../../../glossary.md#term-unary) kind, [rank](../../../../glossary.md#term-rank) zero, dynamic or zero extents, [views](../../../../glossary.md#term-view-meanings), non-`FLOAT32` payloads, [gradients](../../../../glossary.md#term-gradient) that
require unsupported [topology](../../../../glossary.md#term-topology), [runtime](../../../../glossary.md#term-runtime) checking, retries, fallback, input-dependent routing,
general transcendental frameworks, [Model](../../../../glossary.md#term-model) changes, [Compiler](../../../../glossary.md#term-compiler) rewrites, async/[device](../../../../glossary.md#term-device)-selection work,
native [ABI](../../../../glossary.md#term-application-binary-interface-abi) exports, Task-0052 files/evidence, and reuse of a removed Task-0051 implementation.

## Supersession and preserved history

Task 0051 remains the [immutable](../../../../glossary.md#term-immutable) blocked record for its five historical candidates and sole 83-word
Apple M3 Max invocation:

- [opaque](../../../../glossary.md#term-opaque) direct [MPSGraph](../../../../glossary.md#term-mpsgraph) [`EXP`](../../../../glossary.md#term-expx) and its inherited stable composition passed that bounded sample but
  remain <span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>`DOMAIN-BLOCKED`;
- the original custom `EXP`, custom [`SIGMOID`](../../../../glossary.md#term-sigmoid), and direct `SIGMOID` retain their recorded numerical
  failures; and
- its historical plan stopped at a trace prerequisite, but this successor does not inherit
  full-Xcode tracing as an unconditional blocker. Supported actual observation is needed only to
  establish structural facts for an opaque survivor.<span data-glossary-navigation="true"> ([trace glossary](../../../../glossary.md#trace))</span>

Task 0053 owns only new source, new certificates, a new frozen candidate [manifest](../../../../glossary.md#term-manifest), and one new
successor [oracle](../../../../glossary.md#term-test-oracle) after the prerequisites. It may reuse historical words as [regression](../../../../glossary.md#term-regression) witnesses in
a materially new successor corpus, but it must not execute the Task-0051 source, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), candidate
implementations, wrapper, or 83-word invocation again.

## Exact authorized [occurrence](../../../../glossary.md#term-occurrence) domain

A future occurrence is eligible only when all of the following are true:

- profile is exactly [`ACCELERATOR`](../../../../glossary.md#term-accelerator);
- kind is parameterless `UnaryElementwiseKind.EXP` or [`SIGMOID`](../../../../glossary.md#term-sigmoid) with
  `NoOperationAttrs.INSTANCE`;
- there is exactly one input and one output;
- both [descriptors](../../../../glossary.md#term-descriptor) are [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32);
- input and output have the same fully static positive [rank](../../../../glossary.md#term-rank)-`1..16` [Shape](../../../../glossary.md#shape) and matching
  `requiresGrad == false`;
- element and byte geometry, launch products, offsets, and buffer bounds are checked before native
  resource creation; and<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- the occurrence, topology, feeds, targets, ownership, state, and all sentinels/padding pass the
  then-current Java and native preflight before candidate creation.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [targets glossary](../../../../glossary.md#term-target); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [padding glossary](../../../../glossary.md#padding); [preflight glossary](../../../../glossary.md#term-preflight))</span>

Every strict occurrence and every excluded descriptor remains capability-false. Passing a bounded
rank or value [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) never narrows or expands this domain.

## Exact [EXP](../../../../glossary.md#term-expx) result-set obligation

Interpret an input word `x` as [binary32](../../../../glossary.md#floating-point-precision-and-rounding). The complete proof must use the disjoint and exhaustive raw
[partition](../../../../glossary.md#partition) below over all `2^32` words, with counts and coverage checked by the proof [kernel](../../../../glossary.md#kernel):

1. positive and negative NaNs: [exponent](../../../../glossary.md#term-exponent) `0xff`, fraction nonzero;
2. positive and [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero): exponent `0xff`, fraction zero;
3. positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero): exponent and fraction zero;
4. positive and negative [subnormals](../../../../glossary.md#term-subnormal): exponent zero, fraction nonzero; and
5. positive and negative [finite](../../../../glossary.md#term-finite) normals: exponent in `1..254`.

For a finite subnormal input, the [Model](../../../../glossary.md#term-model) input alternatives are the represented value and its
same-[signed zero](../../../../glossary.md#term-signed-zero). For every other finite input, the represented value is the sole input alternative.
The candidate may choose one documented allowed [DAZ](../../../../glossary.md#term-daz) path, but the checker must construct the full
alternative set and prove the chosen result belongs to it.

For each chosen finite input value, evaluate mathematical `exp` with a certified directed [enclosure](../../../../glossary.md#interval-arithmetic-and-directed-rounding)
and define `r` as its [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) binary32 [round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-even result. The allowed result is:

- any binary32 [NaN](../../../../glossary.md#term-nan) for a NaN input; [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), sign, signaling/quiet state are abstract;
- exactly [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) for negative infinity;
- exactly [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) for positive infinity;
- exactly `0x3f800000` for either signed zero;
- for a finite, nonzero, non-subnormal `r`, only a finite nonnegative word whose inclusive ordered
  distance from `r` is at most `5`;
- for a nonzero subnormal `r`, exactly `r`, positive zero, or negative zero; and
- for an `r` that is zero or infinity because of correctly rounded underflow or overflow, the exact
  required class and sign, independent of the distance rule.

Ordered distance is mathematical absolute difference of the unsigned monotone keys

```text
key(bits) = sign_set(bits) ? unsigned(~bits) : unsigned(bits ^ 0x80000000)
distance(a,b) = abs_integer(key(a) - key(b)).
```

It is never a wrapping machine subtraction. NaN, zero, [infinity](../../../../glossary.md#term-infinity), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), result sign,
and result class are checked before and independently from the ordinary finite distance. The final
ordinary exact-EXP bound is inclusive: `distance(actual,r) <= 5`.

## Stable [SIGMOID](../../../../glossary.md#term-sigmoid) recursive obligation

`SIGMOID` has no final elementary [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp)/distance envelope. Its proof must follow the exact stable
site [graph](../../../../glossary.md#graph) for every raw input and every [Model](../../../../glossary.md#term-model) alternative.

After explicit [NaN](../../../../glossary.md#term-nan) classification, apply the exact raw sign [guard](../../../../glossary.md#term-guard). The selected branch is:

```text
negative-sign branch:
    e = EXP(x)
    d = ADD(1.0f, e)
    y = DIV(e, d)

nonnegative-sign branch:
    n = NEG(x)
    e = EXP(n)
    d = ADD(1.0f, e)
    y = DIV(1.0f, d)
```

`NEG` is exact sign-bit inversion with its special-value behavior proved. [`EXP`](../../../../glossary.md#term-expx) is exactly a
`DOMAIN-PASS` instance of the custom candidate proved above, not a selector with analogous output.
At each `ADD` and `DIV` site, the proof independently enumerates same-[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) [DAZ](../../../../glossary.md#term-daz) alternatives
for [subnormal](../../../../glossary.md#term-subnormal) operands, performs exactly one [binary32](../../../../glossary.md#floating-point-precision-and-rounding) [round-to-nearest](../../../../glossary.md#floating-point-precision-and-rounding)-even [operation](../../../../glossary.md#operation), and admits an
exact nonzero subnormal result or either [signed zero](../../../../glossary.md#term-signed-zero) as that site's [FTZ](../../../../glossary.md#term-ftz) alternatives. NaN-producing
or NaN-consuming sites use the Model's abstract NaN class.

The proof must show exact branch selection, exact contributor/operand [identity](../../../../glossary.md#term-identity-meanings), and explicit
binary32 [materialization](../../../../glossary.md#materialization) between sites. A [fused](../../../../glossary.md#specialization-and-fusion) implementation may pass only if it proves the same
observable site boundaries and allowed [rounding](../../../../glossary.md#term-rounding) set; an [opaque](../../../../glossary.md#term-opaque) [fusion](../../../../glossary.md#term-fusion) assertion fails closed. The
final `SIGMOID` word must be a member of the recursively constructed site-by-site union. Agreement
with a high-[precision](../../../../glossary.md#term-precision) whole-sigmoid value or final [tolerance](../../../../glossary.md#term-tolerance) envelope is not evidence.

## Candidate ledger and freeze rule

| Candidate class | Current Gate 1B status | Future [admission](../../../../glossary.md#term-admission) rule |
|---|---|---|
| `CUSTOM_EXP_CERTIFIED_V1` | `DOMAIN-PENDING` | exact source/[constants](../../../../glossary.md#term-constant)/site [manifest](../../../../glossary.md#term-manifest) plus complete checked proof and independent approval |
| `CUSTOM_SIGMOID_STABLE_CERTIFIED_V1` | `DOMAIN-PENDING` | recursive proof using the exact approved [EXP](../../../../glossary.md#term-expx) instance and proved [guard](../../../../glossary.md#term-guard)/NEG/ADD/DIV sites |
| current [opaque](../../../../glossary.md#term-opaque) direct [MPSGraph](../../../../glossary.md#term-mpsgraph) EXP | `DOMAIN-BLOCKED` | remains blocked; selector name, [SDK](../../../../glossary.md#term-sdk) declaration, and [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) cannot admit it |
| current opaque direct MPSGraph [SIGMOID](../../../../glossary.md#term-sigmoid) | `DOMAIN-BLOCKED` | remains blocked and is not a stable-formula proof |
| Task-0051 MPSGraph stable composition | `DOMAIN-BLOCKED` | inherits its opaque EXP [dependency](../../../../glossary.md#term-dependency) and remains historical only |
| future MPSGraph or composed alternative | absent | may be added before freeze only with its own independently checked complete-domain proof for every primitive, [shape](../../../../glossary.md#shape), site, [materialization](../../../../glossary.md#materialization), and deployment version |

No route family is a baseline. Before the [oracle](../../../../glossary.md#term-test-oracle), [commit](../../../../glossary.md#term-commit) one [immutable](../../../../glossary.md#term-immutable) candidate manifest naming
every considered route, [operation](../../../../glossary.md#operation), source/proof [identity](../../../../glossary.md#term-identity-meanings), and verdict. A candidate without
`DOMAIN-PASS` is recorded but is not compiled or [run](../../../../glossary.md#term-run). Adding, removing, or changing source,
constants, [compiler](../../../../glossary.md#term-compiler) flags, proof assumptions, or a candidate after freeze invalidates the oracle
authorization and requires a newly reviewed successor plan; it never permits a retry.

## Gate 0: pinned offline execution boundary

The pinned [toolchain](../../../../glossary.md#term-toolchain) is locally installed and no longer blocks source/proof work. Command Line
Tools do not provide a standalone [Metal](../../../../glossary.md#term-metal) [compiler](../../../../glossary.md#term-compiler) or [MSL](../../../../glossary.md#term-metal-shading-language-msl) standard-library source capable of
grounding hardware [FLOAT32](../../../../glossary.md#term-float32) site semantics. The candidate therefore uses integer/[fixed-point
arithmetic](../../../../glossary.md#fixed-point-arithmetic) over raw [binary32](../../../../glossary.md#floating-point-precision-and-rounding) words and an explicit integer packer; any compiler-created FLOAT32
arithmetic, conversion, [FMA](../../../../glossary.md#term-fused-multiply-add-fma), [contraction](../../../../glossary.md#contraction-dimension), or [opaque](../../../../glossary.md#term-opaque) intrinsic rejects the candidate.

The repository must retain the concrete candidate source/[constants](../../../../glossary.md#term-constant) and all [artifacts](../../../../glossary.md#term-artifact) in the proof
package below. A pinned offline proof command must succeed, a second implementation-independent
all-raw-word or complete [partition](../../../../glossary.md#partition) checker must succeed with zero unresolved inputs, and
independent read-only review must issue `DOMAIN-PASS` before capability or a [device](../../../../glossary.md#term-device) [run](../../../../glossary.md#term-run) is enabled.

Full Xcode and [trace](../../../../glossary.md#trace) tooling are not prerequisites for the closed source-owned route. Exact
source-bound [dispatch](../../../../glossary.md#term-dispatch) and route-owned temporary-byte declarations are admissible under [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0022.
The opaque [MPSGraph](../../../../glossary.md#term-mpsgraph) structural candidates remain `DOMAIN-BLOCKED` unless separately proved; they
are not baselines and cannot be forced. No local timing, comparison, benchmark, or trace [capture](../../../../glossary.md#term-capture-meanings) is
authorized.

Source, proof, [host](../../../../glossary.md#term-host) integration, and [fail-closed](../../../../glossary.md#term-fail-closed) tests may be implemented while this gate is being
checked. Until it passes, capability remains false, production preparation exposes no candidate,
and no device process or numerical [smoke](../../../../glossary.md#term-smoke-test) runs.

## Gate 1A: concrete source and structural audit

The custom candidates must be fixed typed [kernels](../../../../glossary.md#kernel), not a general transcendental subsystem. Before
Gate 1B review, retain reviewable source that:

- classifies raw words before arithmetic and contains explicit [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/zero/[subnormal](../../../../glossary.md#term-subnormal) paths;
- calls no [Metal](../../../../glossary.md#term-metal) [`exp`](../../../../glossary.md#term-expx), `exp2`, [`sigmoid`](../../../../glossary.md#term-sigmoid), fast transcendental, MPSGraph selector, or other opaque
  numerical routine inside the custom proof boundary;<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- freezes range-reduction representation, table dimensions, exact table bytes, polynomial or other
  approximation, coefficient bytes, evaluation order, exception records, and output packer;<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- uses only operations enumerated in the source model and has checked shifts, carries, conversions,
  indices, offsets, and buffer lengths;<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- compiles with safe math, fast/relaxed math disabled, and an explicit [contraction](../../../../glossary.md#contraction-dimension) policy;
- contains no undefined signed overflow, implementation-defined conversion, unchecked table read,
  hidden allocation, or data-dependent fallback; and<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- preserves exact input/output [Shape](../../../../glossary.md#shape) and one-output [pointwise](../../../../glossary.md#term-pointwise) [ownership](../../../../glossary.md#term-ownership-meanings) for every admitted [rank](../../../../glossary.md#term-rank).

The [artifact manifest](../../../../glossary.md#artifact-manifest-and-source-provenance) must enumerate every [binary32](../../../../glossary.md#floating-point-precision-and-rounding) `ADD`, `SUB`, `MUL`, `DIV`, conversion, and
[`FMA`](../../../../glossary.md#term-fused-multiply-add-fma) site by stable source identifier. For each site it records operand interpretation, [DAZ](../../../../glossary.md#term-daz) choice,
[rounding](../../../../glossary.md#term-rounding) mode, contraction, intermediate [materialization](../../../../glossary.md#materialization), [FTZ](../../../../glossary.md#term-ftz) choice, and exceptional-class rule.
An integer-only section records an explicit empty hardware-[FLOAT32](../../../../glossary.md#term-float32)/FMA set and proves from source and
[compiler](../../../../glossary.md#term-compiler) inspection that the set is truly empty. An unlisted compiler-created FMA, contraction,
conversion, or intrinsic rejects the candidate.

## Gate 1B: machine-checkable complete-domain proof

Gate 1B precedes and is independent of the [numerical oracle](../../../../glossary.md#adr-ci-and-qualification). The proof must cover every raw input
and every authorized [Shape](../../../../glossary.md#shape), not a [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) or an error estimate copied from a generator [log](../../../../glossary.md#term-log).

### Mandatory [EXP](../../../../glossary.md#term-expx) theorems

The checked package proves:

1. raw-class [partition](../../../../glossary.md#partition) disjointness, exact counts, and union equal to all `2^32` words;
2. [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/zero/[subnormal](../../../../glossary.md#term-subnormal) behavior and the represented/same-[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) [DAZ](../../../../glossary.md#term-daz) alternatives;
3. exact conversion from binary32 to the candidate's integer/fixed-point or proved floating
   representation, including sign, shift, carry, and width bounds;<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic); [carry glossary](../../../../glossary.md#term-carry))</span>
4. unique range-reduction quotient selection and a directed enclosure for the residual, without
   treating an irrational constant such as <span data-glossary-navigation="true"> ([enclosure glossary](../../../../glossary.md#interval-arithmetic-and-directed-rounding); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [constant glossary](../../../../glossary.md#term-constant))</span>`ln(2)` as exact;
5. every table-index formula is total and in bounds, every table interval encloses its mathematical
   value, and every scaling exponent/shift is in range;<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent))</span>
6. the polynomial or other approximation is valid over the entire certified residual interval,
   with approximation, coefficient-rounding, evaluation, truncation, and reconstruction errors
   composed in the correct direction;<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [evaluation glossary](../../../../glossary.md#term-evaluation); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
7. every actual binary32 RNE/FMA site has the exact declared one-rounding semantics, and every
   possible DAZ/FTZ alternative is represented at the corresponding Model site;<span data-glossary-navigation="true"> ([FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [FTZ glossary](../../../../glossary.md#term-ftz); [Model glossary](../../../../glossary.md#term-model))</span>
8. exact adjacent-word overflow and underflow thresholds, gradual-subnormal packing, tie-to-even
   boundaries, and infinity/zero publication;<span data-glossary-navigation="true"> ([overflow and underflow glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [publication glossary](../../../../glossary.md#term-publication))</span>
9. every hard-to-round/exception table is sorted, unique, in bounds, and exhaustively covers all
   unresolved intervals; any unresolved word rejects the proof; and
10. for every finite allowed input path, the final result has the required class/sign and either the
    exact subnormal/zero/infinity membership or inclusive ordinary ordered distance <span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>`<=5`.

### Mandatory launch and memory theorems

For [rank](../../../../glossary.md#term-rank) `r in 1..16`, positive [dimensions](../../../../glossary.md#dimension) `d[0..r-1]`, and
`N = product(d[i])`, prove checked [element](../../../../glossary.md#term-element) and byte products and the launch mapping

```text
linear(x,y,z) = x + W * (y + H * z)
```

for the exact [host](../../../../glossary.md#term-host) grid construction. The proof must show products fit their types, every integer
`0 <= linear < N` has exactly one grid [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), excess grid threads write nothing, each admitted
thread reads and writes exactly one in-bounds element, and no 32-bit [truncation](../../../../glossary.md#term-truncation-toward-zero) occurs. The theorem
must cover every resource-feasible positive rank-`1..16` [Shape](../../../../glossary.md#shape), not only the future rank witnesses.

### Mandatory [SIGMOID](../../../../glossary.md#term-sigmoid) theorems

The checked package proves:

1. total [NaN](../../../../glossary.md#term-nan) handling and the exact raw sign [guard](../../../../glossary.md#term-guard), including both signed zeros and [infinities](../../../../glossary.md#nan-infinity-and-signed-zero);
2. `NEG` source-bit behavior for every non-NaN word;
3. recursive use of the exact approved [EXP](../../../../glossary.md#term-expx) theorem for the selected branch;
4. each `ADD`/`DIV` implementation against complete raw-word partitions, including exceptional
   classes, DAZ, one RNE operation, FTZ, signed zeros, and materialized intermediates;<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [operation glossary](../../../../glossary.md#operation); [FTZ glossary](../../../../glossary.md#term-ftz))</span>
5. no contributor/site is omitted, duplicated, reordered, fused without proof, or replaced by a
   whole-function approximation; and<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
6. for every one of the `2^32` inputs, the published word belongs to the Model's recursively
   enumerated branch result set.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

### Retained proof [artifacts](../../../../glossary.md#term-artifact)

Before the [oracle](../../../../glossary.md#term-test-oracle), retain under a dedicated Task-0053 evidence directory, with [SHA-256](../../../../glossary.md#term-sha-256) [identities](../../../../glossary.md#term-identity-meanings):

- exact `.metal` source and [host](../../../../glossary.md#term-host) staging source;
- machine-readable candidate, raw-partition, FLOAT32-site, constants, table, polynomial, exception,
  launch, and resource manifests;<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- [constant](../../../../glossary.md#term-constant)/coefficient generator inputs and deterministic generated outputs;
- the formal source [model](../../../../glossary.md#term-model) and theorem sources;
- proof-[kernel](../../../../glossary.md#kernel) outputs/certificates and a single offline proof driver;
- an independent exhaustive `2^32`-word checker, its source, exact-reference configuration, partition
  counts, class counts, maximum ordered distance and witness, unresolved count, exception count,
  and deterministic report;
- compiler version/options, source-to-model/site correspondence audit, and compiled-function
  reflection or equivalent instruction/site audit; and<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- one top-level [manifest](../../../../glossary.md#term-manifest) [binding](../../../../glossary.md#term-binding) every byte, tool version, command, assumption, and result.

Evidence remains reviewable after execution. Only compiled binaries, raw temporary caches, and
unneeded transient workspaces are disposable.

### Acceptable proof tools

The numerical and bit-level theorems must terminate in a small proof [kernel](../../../../glossary.md#kernel). Accepted primary paths
are either:

- a pinned **[Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) 4** toolchain checking the raw-word, IEEE/RNE, interval, approximation, indexing,
  and launch theorems; or<span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [IEEE glossary](../../../../glossary.md#term-ieee))</span>
- pinned **[Coq](../../../../glossary.md#proof-assistant-and-model-checking-tools) + [Flocq](../../../../glossary.md#proof-assistant-and-model-checking-tools)**, with **[Gappa](../../../../glossary.md#proof-assistant-and-model-checking-tools)** allowed to generate interval/[rounding](../../../../glossary.md#term-rounding) certificates that
  `coqc` checks.

**[Sollya](../../../../glossary.md#high-precision-numerical-tools)** may generate approximation coefficients and exact rational interval obligations;
**[MPFR](../../../../glossary.md#high-precision-numerical-tools)/[MPFI](../../../../glossary.md#high-precision-numerical-tools)** with [directed rounding](../../../../glossary.md#interval-arithmetic-and-directed-rounding) may drive the independent adaptive-[precision](../../../../glossary.md#term-precision) reference and
all-word cross-check. They are generators/checkers, not substitutes for a proof-kernel result.
An [SMT](../../../../glossary.md#proof-assistant-and-model-checking-tools) solver, [CBMC](../../../../glossary.md#proof-assistant-and-model-checking-tools)/[Kani](../../../../glossary.md#proof-assistant-and-model-checking-tools) [run](../../../../glossary.md#term-run), [compiler](../../../../glossary.md#term-compiler) optimization report, Python/decimal script, random test,
spreadsheet, [ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), or `PASS` [log](../../../../glossary.md#term-log) is corroboration only unless every relied-on result is emitted
as a [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) replayed by the accepted Lean/Coq kernel.

The future implementation must pin exact tool/library versions and [distribution](../../../../glossary.md#term-distribution) [hashes](../../../../glossary.md#term-hash) before use;
this plan intentionally does not guess versions or certificate bytes.

### [Fail-closed](../../../../glossary.md#term-fail-closed) independent review

The independent reviewer is read-only and reruns the proof from a clean, network-disabled
environment. The review rejects the candidate on any:

- missing or hash-mismatched artifact, mutable download, unpinned tool/library, nondeterministic
  generator, or unreproducible report;<span data-glossary-navigation="true"> ([hash glossary](../../../../glossary.md#term-hash); [artifact glossary](../../../../glossary.md#term-artifact); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- `sorry`, `admit`, candidate-specific axiom, unchecked foreign [callback](../../../../glossary.md#term-callback), proof timeout, solver
  `unknown`, or unreviewed [trusted assumption](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions);
- raw-word partition gap/overlap/count mismatch, unproved DAZ/FTZ branch, unresolved interval,
  ambiguous rounding boundary, or maximum distance greater than <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [rounding glossary](../../../../glossary.md#term-rounding))</span>`5`;
- range-reduction gap, invalid constant enclosure, table/index/shift/carry/buffer bound gap,
  duplicate exception key, or incomplete exception proof;<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [enclosure glossary](../../../../glossary.md#interval-arithmetic-and-directed-rounding); [carry glossary](../../../../glossary.md#term-carry); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- source/model/constants/compiler-options mismatch, unmodeled FLOAT32/FMA/contraction site, opaque
  numerical call, or compiler-created operation not present in the site manifest;<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [contraction glossary](../../../../glossary.md#contraction-dimension); [opaque glossary](../../../../glossary.md#term-opaque); [operation glossary](../../../../glossary.md#operation); [manifest glossary](../../../../glossary.md#term-manifest))</span>
- [SIGMOID](../../../../glossary.md#term-sigmoid) final-envelope [argument](../../../../glossary.md#term-argument) in place of recursive site membership; or
- launch proof that omits a rank, positive checked extent, resource-feasible element count, exact
  one-writer coverage, or in-bounds access.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent); [element glossary](../../../../glossary.md#term-element))</span>

Any rejection assigns that candidate `DOMAIN-BLOCKED`, records the first failure, and forbids the
[device](../../../../glossary.md#term-device) [oracle](../../../../glossary.md#term-test-oracle). There is no guessed [certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions), reviewer waiver, post-oracle proof repair, narrowed
domain, or conditional `DOMAIN-PASS`.

## Gate 2: one new proof-first [numerical oracle](../../../../glossary.md#adr-ci-and-qualification)

This task explicitly authorizes exactly one future real-[device](../../../../glossary.md#term-device) numerical process invocation, but
only after Gate 0, Gate 1A, Gate 1B, independent approval, and [immutable](../../../../glossary.md#term-immutable) candidate/corpus manifests
all pass. A failed process, crash, mismatch, or control failure is recorded as the result; there is
no retry, second optimization level, second device/context, amended corpus, or rerun.

The successor corpus is generated and [hash](../../../../glossary.md#term-hash)-frozen before execution from:

- fixed representatives of every raw class/sign, both zeros/infinities, quiet/signaling NaNs, and
  minimum/maximum subnormals/normals;<span data-glossary-navigation="true"> ([infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Task-0051 failure and boundary words as historical regression witnesses, without running its
  source or reproducing its old 83-word-only invocation;
- every proved [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow) and normal/[subnormal](../../../../glossary.md#term-subnormal) threshold neighbor required by the manifests;
- predecessor/exact/successor words at every range-reduction quotient and table-index transition;
- binary32 words nearest each approximation-interval endpoint and every certified critical/error
  point;<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- deterministic stratified exponent/fraction/sign samples bound to a published seed and generator
  hash; and<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [seed glossary](../../../../glossary.md#term-seed))</span>
- positive [canonical](../../../../glossary.md#term-canonical-form) [rank](../../../../glossary.md#term-rank) witnesses `1..16`, with checked [Shapes](../../../../glossary.md#shape) and identical per-candidate values.

The exact generated word list, Shapes, order, generator, and [SHA-256](../../../../glossary.md#term-sha-256) are reviewed and frozen before
launch. The single process uses one default-[device context](../../../../glossary.md#device-submission-synchronization-and-execution-context), compiles every `DOMAIN-PASS` candidate
before creating the [command buffer](../../../../glossary.md#term-command-buffer), disables relaxed/fast math, records [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) settings, uses
one command buffer where structurally possible, supplies direct [targets](../../../../glossary.md#term-target), snapshots inputs, guards
outputs, and prints raw words plus control [metadata](../../../../glossary.md#term-metadata). Every candidate sees the same ordered values and
Shapes.

An independent offline validator uses adaptive directed [MPFR](../../../../glossary.md#high-precision-numerical-tools)/[MPFI](../../../../glossary.md#high-precision-numerical-tools) enclosures for [EXP](../../../../glossary.md#term-expx) and the formal
recursive site [model](../../../../glossary.md#term-model) for [SIGMOID](../../../../glossary.md#term-sigmoid). It records per candidate: source/proof/corpus/output [identities](../../../../glossary.md#term-identity-meanings),
class/sign failures, maximum ordinary ordered distance and witness, [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) observations, recursive
membership, every rank/control verdict, and first failure. The bounded [oracle](../../../../glossary.md#term-test-oracle) is [regression](../../../../glossary.md#term-regression) evidence
only; it can reject but cannot establish `DOMAIN-PASS`.

## Gate 3: survivor adjudication and conditional controlled comparison

Only candidates with Gate-1 `DOMAIN-PASS` and Gate-2 numerical `PASS` survive. For each [operation](../../../../glossary.md#operation):

1. zero survivors keeps capability false;
2. one survivor needs no comparative cost gate;
3. multiple survivors may be selected without timing only when they have identical proven
   semantics/domain and one strictly structurally dominates every alternative: no more compute
   dispatches, no more route-owned temporary bytes, and at least one strict improvement; and
4. multiple nondominated survivors leave route selection pending an explicitly authorized,
   controlled comparative environment and protocol.<span data-glossary-navigation="true"> ([nondominated glossary](../../../../glossary.md#compiler-transformations-and-dominance))</span>

Closed custom routes may establish [dispatch](../../../../glossary.md#term-dispatch) and route-owned temporary-byte facts from exact
source-bound declarations under [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0022. [Opaque](../../../../glossary.md#term-opaque) routes require supported actual-internal
observation; [backend](../../../../glossary.md#backend) calls, [graph](../../../../glossary.md#graph) [nodes](../../../../glossary.md#node), command [buffers](../../../../glossary.md#term-buffer), inferred [fusion](../../../../glossary.md#term-fusion), estimates, and process
[RSS](../../../../glossary.md#term-resident-set-size-rss) are not substitutes. Structural comparison is not a weighted score and cannot trade more
dispatches for fewer bytes. Timing on an uncontrolled local developer [device](../../../../glossary.md#term-device) may be retained as
diagnostic history only and never qualifies a route, selects a route, changes tuning [identity](../../../../glossary.md#term-identity-meanings),
populates a production decision, or breaks a structural tie.

## Gate 4: fixed route and future production cutover

Encode exactly one [immutable](../../../../glossary.md#term-immutable) production route only after Gate 3 authorizes it, with no threshold,
retry, fallback, [runtime](../../../../glossary.md#term-runtime) timing, or input-dependent choice. A route decision still does not
authorize production by itself. A future implementation revision must update all callers
atomically, prove [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/[preflight](../../../../glossary.md#term-preflight)/[lifecycle](../../../../glossary.md#lifecycle) behavior, [run](../../../../glossary.md#term-run) focused consumer-observable
verification, and pass independent Class C review.

Task 0055 has landed schema 13, `EXP=55`, `SIGMOID=64`, and [backend](../../../../glossary.md#backend)-local identity version 14 as
structural vocabulary only. Task 0053 must consume those stable values without advancing them
unless its production change introduces a genuinely new identity fact. No dual decoder, [alias](../../../../glossary.md#aliasing-and-arena),
migration shim, or stale-byte acceptance is permitted.

## Future production and [lifecycle](../../../../glossary.md#lifecycle) proof

A selected implementation must preserve whole-[partition](../../../../glossary.md#partition) analysis, [canonical](../../../../glossary.md#term-canonical-form) state, stable ordered
[nodes](../../../../glossary.md#node)/values/feeds/[targets](../../../../glossary.md#term-target), declarations-before-assignment, transactional finalization, [immutable](../../../../glossary.md#term-immutable)
splats, direct assigned targets, context child [leases](../../../../glossary.md#term-lease), [run](../../../../glossary.md#term-run)-owned outputs/address workspace,
[synchronous](../../../../glossary.md#term-synchronous) invocation, reuse, concurrent/session isolation, [rollback](../../../../glossary.md#term-rollback), close rejection,
[publication](../../../../glossary.md#term-publication), transfer, [trace](../../../../glossary.md#trace), and tuning behavior.

Focused proof must cover capability positives/negatives, Java/native [schema](../../../../glossary.md#term-schema) agreement, stale
[identity](../../../../glossary.md#term-identity-meanings) rejection, all [rank](../../../../glossary.md#term-rank)/geometry boundaries, mixed partitions with existing canonical
operations, repeated/concurrent/session isolation, and real public [Engine](../../../../glossary.md#term-engine) forward execution without
[CPU](../../../../glossary.md#term-cpu) [owner](../../../../glossary.md#term-owner) or skip. Generated [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) [gradients](../../../../glossary.md#term-gradient) remain outside capability unless their
actual [compiler](../../../../glossary.md#term-compiler) [topology](../../../../glossary.md#term-topology) is already closed by supported canonical operations; do not handwrite a
substitute or widen [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/affine/[scalar](../../../../glossary.md#term-scalar) capability.

## Dependencies and integration

- Depends on: completed and independently approved Model 0030/0031; Task 0051's preserved blocked
  record; Complete Task 0055's ABI-5/schema-13 foundation; Complete Task 0056's canonical route
  identity/catalog; Compiler unary inference/gradient capture; Config 0006; and Engine 0018.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [unary glossary](../../../../glossary.md#term-unary); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [Planning](../../../../glossary.md#term-planning) base: current clean Task-0056 completion plus cast-reason remediation `fa9387e0`.
- Proof tools: pinned [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) `4.34.1`, [Sollya](../../../../glossary.md#high-precision-numerical-tools) `8.0`, [MPFI](../../../../glossary.md#high-precision-numerical-tools) `1.5.4`, [MPFR](../../../../glossary.md#high-precision-numerical-tools) `4.2.2`, [GMP](../../../../glossary.md#high-precision-numerical-tools) `6.3.0`.
- Supersedes: Task 0051 for all new EXP/SIGMOID source, proof, device smoke, route, schema,
  capability, and production work; Task 0051's historical source and oracle are never rerun.<span data-glossary-navigation="true"> ([EXP glossary](../../../../glossary.md#term-expx); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [device glossary](../../../../glossary.md#term-device); [smoke glossary](../../../../glossary.md#term-smoke-test); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Independent of: Task 0052 candidates, proof, oracle, raw timing history, and production kernels;
  it shares only ADR 0022's corrected decision-authority rule and the current custom-program
  lifecycle.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/tuning/
  lifecycle/public Engine edit and shared numerical-profile documentation.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec))</span>
- Parallel group: none.
- Integration [owner](../../../../glossary.md#term-owner): this serialized Task-0053 execution.

## Execution files and [artifacts](../../../../glossary.md#term-artifact)

Retain a dedicated Task-0053 evidence directory containing exact custom `.metal` source, generated
embedded-source bytes, [constant](../../../../glossary.md#term-constant)/table generator inputs and outputs, machine-readable manifests,
formal [model](../../../../glossary.md#term-model) and theorem sources, proof-[kernel](../../../../glossary.md#kernel) output, the independent checker and deterministic
report, launch proof, source-to-proof/[compiler](../../../../glossary.md#term-compiler)-site audit, tool [identities](../../../../glossary.md#term-identity-meanings), and a top-level
[SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest). Production cutover may update the existing [Metal](../../../../glossary.md#term-metal) capability, [lowering](../../../../glossary.md#lowering), [canonical](../../../../glossary.md#term-canonical-form)
prepared-route/candidate [identity](../../../../glossary.md#term-identity-meanings), custom-program native [lifecycle](../../../../glossary.md#lifecycle), package [metadata](../../../../glossary.md#term-metadata), focused
[conformance](../../../../glossary.md#term-conformance)/integration tests, authoritative [architecture contract](../../../../glossary.md#architecture-contract), explanatory guides, this task,
the Metal master, and roadmap.

## Acceptance criteria

1. Task 0051 remains an unchanged historical record and its source/[oracle](../../../../glossary.md#term-test-oracle) is never rerun.
2. The custom source is integer/fixed-point over raw words with explicit specials, DAZ/FTZ,
   overflow/underflow, range reduction, exact-byte table, polynomial, and bit packer; its compiler
   audit reports no hardware FLOAT32/FMA/contraction site.<span data-glossary-navigation="true"> ([fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [range reduction glossary](../../../../glossary.md#range-reduction-and-approximation); [compiler glossary](../../../../glossary.md#term-compiler); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
3. Pinned Lean replay checks the complete raw partition, exact decode, thresholds/quotient/table/
   residual/polynomial/reconstruction bounds, tie-even packer, inclusive EXP distance <span data-glossary-navigation="true"> ([Lean glossary](../../../../glossary.md#proof-assistant-and-model-checking-tools); [partition glossary](../../../../glossary.md#partition); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [EXP glossary](../../../../glossary.md#term-expx))</span>`<=5`, and
   exact rank-<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>`1..16` launch/index coverage without `sorry`, `admit`, or candidate-specific axioms.
4. An implementation-independent [MPFR](../../../../glossary.md#high-precision-numerical-tools)/[MPFI](../../../../glossary.md#high-precision-numerical-tools) checker covers all `2^32` input words directly or by
   checked complete partitions and records zero unresolved words, correct class/sign behavior, and
   maximum ordinary distance at most five.
5. Stable [SIGMOID](../../../../glossary.md#term-sigmoid) is proved recursively from exact [NaN](../../../../glossary.md#term-nan)/sign classification, `NEG`, the approved
   EXP, and materialized integer implementations of the exact FLOAT32 `ADD`/`DIV` sites including
   their DAZ/RNE/FTZ alternatives; no whole-function envelope substitutes for site membership.
6. MPSGraph direct EXP/SIGMOID and the structural stable composition remain catalog candidates but
   are unavailable to forcing and production unless they independently satisfy the same proof.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
7. One correct custom survivor requires no timing or cost comparison and becomes one immutable
   cold prepared route; there is no runtime switch, fallback, retry, autotune, or public route knob.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [runtime glossary](../../../../glossary.md#term-runtime))</span>
8. Schema 13 remains the native node schema. Any private candidate/compatibility/route identity
   change is a clean exact bump with no alias or dual decoder. Strict, gradients, excluded Shapes,
   types, layouts, and attributes remain capability-false.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [node glossary](../../../../glossary.md#node); [identity glossary](../../../../glossary.md#term-identity-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [gradients glossary](../../../../glossary.md#term-gradient); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout))</span>
9. Focused proof replay, source/site audit, Java/native/conformance tests, and a no-skip public
   Engine Metal smoke pass. No local performance benchmark and no final full repository build run.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [smoke glossary](../../../../glossary.md#term-smoke-test); [run glossary](../../../../glossary.md#term-run))</span>
10. Independent Class C review approves the proof package and production cutover. Documentation,
    Javadocs, links, fences, final newlines, status synchronization, and <span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>`git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the single offline proof driver from a network-disabled environment, the independent checker,
the retained source/header/site audit, focused [Metal](../../../../glossary.md#term-metal) Java/native/[conformance](../../../../glossary.md#term-conformance) tests, native build and
export audit, package verification when package bytes change, and one no-skip public [Engine](../../../../glossary.md#term-engine) [device](../../../../glossary.md#term-device)
[smoke](../../../../glossary.md#term-smoke-test) after `DOMAIN-PASS`. Run no Task-0051 [oracle](../../../../glossary.md#term-test-oracle), timing, benchmark, numerical performance
comparison, or final full repository build.

## Documentation and review impact

This Class C capability and native-boundary change requires synchronized updates to the
authoritative root/[backend](../../../../glossary.md#backend)-execution contract, current [Metal](../../../../glossary.md#term-metal) explanatory guide/Javadocs, native
README/package [metadata](../../../../glossary.md#term-metadata), task/master/roadmap status, backend [conformance](../../../../glossary.md#term-conformance), and public [Engine](../../../../glossary.md#term-engine)
integration evidence. An independent reviewer must replay the proof and [target](../../../../glossary.md#term-target) the capability,
source/proof [identity](../../../../glossary.md#term-identity-meanings), [compiler](../../../../glossary.md#term-compiler)-site, prepared-route, native [lifecycle](../../../../glossary.md#lifecycle), and no-fallback boundaries.

## Result

Execution is Blocked after the retained [checkpoint](../../../../glossary.md#term-checkpoint) from exact clean base `fa9387e0`. One shared integer core is expanded
into the [MSL](../../../../glossary.md#term-metal-shading-language-msl) NSString and included directly by the C checker [model](../../../../glossary.md#term-model), eliminating the duplicate
algorithm. Its signed-magnitude conversion, shifts, products, additions/subtractions, table index,
and pack paths are explicitly guarded or [saturating](../../../../glossary.md#saturation-pole-and-extremum) outside the admitted range. Retained evidence
also includes [MPFR](../../../../glossary.md#high-precision-numerical-tools)-generated table and class/one-[rounding](../../../../glossary.md#term-rounding) thresholds, a [Sollya](../../../../glossary.md#high-precision-numerical-tools) degree-six
polynomial interval, structural [Lean](../../../../glossary.md#proof-assistant-and-model-checking-tools) launch/classification proofs, source/header/[host](../../../../glossary.md#term-host) audits, and
deterministic manifests. The partitioned checker visited all
`2^32` raw words with exact class counts, zero unresolved inputs, zero [EXP](../../../../glossary.md#term-expx)/stable-[SIGMOID](../../../../glossary.md#term-sigmoid)/site
failures, and maximum ordinary EXP ordered distance one at input `0x33800000`
(`0x3f800000` candidate versus `0x3f800001` reference). That PASS is independent corroboration,
not proof-root authorization.

The source and dormant host mappings are integrated [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). Native creation returns
unsupported-[operation](../../../../glossary.md#operation) status for `EXP=55` and `SIGMOID=64`; the candidate source is not appended to
the active Task-0052 library, and Java capability, catalog availability, prepared-route [identity](../../../../glossary.md#term-identity-meanings),
[schema](../../../../glossary.md#term-schema), [ABI](../../../../glossary.md#term-application-binary-interface-abi), and public behavior are unchanged. No [device](../../../../glossary.md#term-device) execution, timing, benchmark, route
selection, or production approval occurred.

The no-axiom Lean checkpoint now retains an exact Nat/Int definition of the candidate, proves
nearest-even division error and monotonicity, monotonic [fixed-point](../../../../glossary.md#fixed-point-arithmetic) decode/shift behavior, and
boundary examples, and [kernel](../../../../glossary.md#kernel)-replays the generated table intervals as exact rational enclosures of
the positive roots satisfying `y^128 = 2^i`. A generated 32,778-record sign/[exponent](../../../../glossary.md#term-exponent)/quotient
[certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) is checked in the kernel for endpoint quotients and same-exponent intervals; generic
Lean theorems prove that its [contiguous](../../../../glossary.md#term-contiguous) positive and negative interval lists cover their complete
active magnitude ranges and are disjoint. MPFR and Sollya remain generators/corroboration only.

That retained [finite](../../../../glossary.md#term-finite) certificate is not sufficient for a sound mathematical-EXP theorem in the
pinned environment. No constructive-real or equivalent rational-Cauchy exponential development is
available to bridge a checked rational [Taylor enclosure](../../../../glossary.md#range-reduction-and-approximation) to mathematical `exp`, prove
`exp(x + y) = exp(x) * exp(y)` and the `ln(2)/128` reduction identity, and [carry](../../../../glossary.md#term-carry) those semantics
through the [Q48](../../../../glossary.md#fixed-point-arithmetic)/[Q32](../../../../glossary.md#fixed-point-arithmetic) polynomial, reconstruction, and [binary32](../../../../glossary.md#floating-point-precision-and-rounding) rounding result. The exact external
blocker is a pinned no-axiom library or formal lemma set providing that bridge. Source-to-Lean
semantic equivalence and the stable-SIGMOID materialized ADD/DIV RNE-or-[FTZ](../../../../glossary.md#term-ftz) membership theorem also
remain open.

The open obligations are recorded in
`backends/metal/evidence/0053/manifests/proof-status.json`. The compiled-MSL site audit remains
unavailable without a standalone [compiler](../../../../glossary.md#term-compiler). An independent read-only checkpoint review reported no
P0/P1/P2 findings and accepted these [artifacts](../../../../glossary.md#term-artifact) as [fail-closed](../../../../glossary.md#term-fail-closed) evidence only, explicitly not
`DOMAIN-PASS`. Required independent domain approval and the proof-first device [smoke](../../../../glossary.md#term-smoke-test) remain blocked.
EXP/SIGMOID are not certified, available, executable, or capability-true.

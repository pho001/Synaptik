# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0026: [Profile-free numerical semantics](../../glossary.md#profile-free-numerical-semantics)


## Editorial reading context

Model defines what each operation means; a backend provider separately states whether it implements a particular occurrence. Removing a graph-wide profile does not turn every occurrence into a supported one. DAZ reads a tiny arithmetic operand as signed zero; FTZ may zero a tiny result. Neither changes raw movement, stored comparisons or an exact branch choice.

Use the [semantics and capability at preparation](../../architecture/runtime-prepare-backend-boundary.md#numerical-semantics-at-the-boundary) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-10-08, on completion of [Task 0032](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
in the main worktree. This record explains the decision; the root and incorporated scoped
contracts remain authoritative. Acceptance does not assert a [commit](../../glossary.md#term-commit), push, or remote [publication](../../glossary.md#term-publication).

## Context

A *numerical profile* selected a [graph](../../glossary.md#graph)-wide [Model](../../glossary.md#term-model) allowed-result set before Task 0032.
`STRICT_IEEE` and [`ACCELERATOR`](../../glossary.md#term-accelerator) traveled through cold [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), capability queries,
preparation, and [backend](../../glossary.md#backend) [identities](../../glossary.md#term-identity-meanings). The current
[root invariants](../../../ARCHITECTURE.md#core-invariants) and incorporated
[profile-free Model semantics](../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)
now specify the cutover contract; an [architecture decision record](../../glossary.md#term-architecture-decision-record-adr) ([ADR](../../glossary.md#adr-ci-and-qualification)) does
not supply that authority. The former strict set was not universally bit-exact, and the accelerator
name is not a speed guarantee. Yet a global choice conflates exact stored-value behavior with
arithmetic latitude and carries profile [identity](../../glossary.md#term-identity-meanings) across
otherwise unrelated layers.

The historical pre-cutover text had concrete semantic conflicts that the cutover resolved:

- Before cutover, [`ScatterReduction.MUL`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/ScatterReduction.java)
  promised one order-independent exact abstract floating product with final result-format rounding.<span data-glossary-navigation="true"> ([rounding glossary](../../glossary.md#term-rounding))</span>
  [`AxisScatterKind`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/index/AxisScatterKind.java)
  separately permitted an `ACCELERATOR FLOAT32` per-step rounded tree for the same reduction. The
  profile-free contract replaces the abstract-product promise for floating `MUL`.
- Before cutover, [`BinaryComparisonKind`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/elementwise/comparison/BinaryComparisonKind.java)
  permitted `ACCELERATOR FLOAT32` comparisons to read subnormals as signed zero. The
  stored-value comparison rule still requires route-specific conformance evidence.<span data-glossary-navigation="true"> ([subnormals glossary](../../glossary.md#term-subnormal); [signed zero glossary](../../glossary.md#term-signed-zero); [conformance glossary](../../glossary.md#term-conformance))</span>

## Decision drivers

- Keep one Model-owned meaning for each operation without a replacement global mode or backend-
  defined semantics.<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model); [operation glossary](../../glossary.md#operation); [backend glossary](../../glossary.md#backend))</span>
- Preserve exact structure, represented-value selection, conversions, and state while locating
  floating arithmetic latitude at named family, dtype, and observable-site boundaries.<span data-glossary-navigation="true"> ([dtype glossary](../../glossary.md#data-type--datatype))</span>
- Retain existing truthful provider capability, low-precision routes, and safety checks without
  converting numerical samples into production policy or new support claims.<span data-glossary-navigation="true"> ([precision glossary](../../glossary.md#term-precision); [samples glossary](../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Make [finite](../../glossary.md#term-finite) numerical [validation](../../glossary.md#term-validation) practical without granting a [tolerance](../../glossary.md#term-tolerance)-based [compiler](../../glossary.md#term-compiler) rewrite.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Keep both profiles | Preserves current [API](../../glossary.md#term-application-programming-interface-api) and proof [model](../../glossary.md#term-model) | Retains cold [identity](../../glossary.md#term-identity-meanings) and the split between stored-value and arithmetic rules. |
| Rename [`ACCELERATOR`](../../glossary.md#term-accelerator) to `FAST` or `RELAXED` | Small surface change | Renames rather than resolves the global semantic split; implies performance or blanket relaxation. |
| Introduce a global [tolerance](../../glossary.md#term-tolerance) or [runtime](../../glossary.md#term-runtime) [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) | One apparent numerical gate | Cannot protect special values, winner selection, exact casts, or rewrite legality; adds production policy. |
| Remove selectors; define family/[dtype](../../glossary.md#data-type--datatype) rules and test-only tolerances | Separates exact boundaries from arithmetic [qualification](../../glossary.md#term-qualification) | Required coordinated API/identity migration and independent route evidence; Task 0032 completed both. |

The table compares semantic choices, not measured route performance.

## Decision

Remove `STRICT_IEEE` and [`ACCELERATOR`](../../glossary.md#term-accelerator) selectors, their public selection [API](../../glossary.md#term-application-programming-interface-api), and their cold
transport/[identity](../../glossary.md#term-identity-meanings) fields without replacing them with another global profile. [Model](../../glossary.md#term-model) continues to
own each [operation](../../glossary.md#operation)'s family formula, contributor domain, guards, special-value behavior, allowed
working [precision](../../glossary.md#term-precision), [rounding](../../glossary.md#term-rounding), and observable result rules. [Backend](../../glossary.md#backend) support remains an [occurrence](../../glossary.md#term-occurrence)-
specific [boolean](../../glossary.md#data-type--datatype) question, not a consequence of removing a selector.

Structure and discrete results remain exact: [dtype](../../glossary.md#data-type--datatype) and promotion, [shapes](../../glossary.md#shape)/[layouts](../../glossary.md#layout)/[axes](../../glossary.md#term-axis), mapped
contributors, bounds and indices, [masks](../../glossary.md#term-boolean-mask) and predicates, ordering and stability, [graph](../../glossary.md#graph)-random state,
saved values, and [publication](../../glossary.md#term-publication). Declared casts and raw movement preserve their existing conversion
or [represented-bit](../../glossary.md#storage-mapping-contiguity-and-injectivity) rules. `WHERE` makes an exact Boolean branch choice, then applies the declared
cast if promotion requires one; same-type selection preserves the chosen representation. Exact
represented-value rules remain for `ABS`, `NEG`, `SIGN`, `FLOOR`, `CEIL`, and `RELU` where their
family contracts specify them. No neighboring arithmetic [DAZ](../../glossary.md#term-daz)/[FTZ](../../glossary.md#term-ftz) freedom leaks into these
boundaries.

Floating comparisons and [extrema](../../glossary.md#saturation-pole-and-extremum) observe stored numeric values, including represented
[subnormals](../../glossary.md#term-subnormal). They retain their family [NaN](../../glossary.md#term-nan), opposite-[signed-zero](../../glossary.md#nan-infinity-and-signed-zero), tie, winner, and index rules;
comparison has no [epsilon](../../glossary.md#epsilon) and extrema cannot invent a candidate. For example, a positive minimum
subnormal is greater than `+0` when on the left, less when on the right, and unequal in both
orders; `MIN` selects `+0` and `MAX` selects the subnormal. This is a semantic example;
route [qualification](../../glossary.md#term-qualification) is separate from the Model rule.

The accepted permission is per *named floating arithmetic [primitive site](../../glossary.md#rounding-sites-fma-and-aggregate-floor)* in the
applicable Model family formula, not a blanket operation-output rule:

| Arithmetic-site dtype | Input DAZ | Result FTZ |
|---|---|---|
| [`FLOAT32`](../../glossary.md#term-float32) | Permitted at named primitive inputs | Permitted at named primitive results |
| [`BFLOAT16`](../../glossary.md#term-bfloat16) | Permitted at named primitive inputs | Permitted at named primitive results |
| [`FLOAT16`](../../glossary.md#term-float16) | Permitted at named primitive inputs, including multiplication | Permitted at named primitive results, including multiplication |
| [`FLOAT64`](../../glossary.md#term-float64) | Forbidden | Forbidden |

[Denormals-are-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (DAZ) reads a subnormal arithmetic-site input as same-[signed zero](../../glossary.md#term-signed-zero) without
changing its stored value. [Flush-to-zero](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) (FTZ) may map a [finite](../../glossary.md#term-finite) subnormal arithmetic-site result
to zero, subject to that family's existing signed-zero contract; it creates no universal choice
of zero sign. These permissions never apply to stored comparisons, extrema/winner selection, raw
movement, casts, selection, indices, masks, or state, nor to the exact [unary](../../glossary.md#term-unary) `ABS`, `NEG`, `SIGN`,
`FLOOR`, `CEIL`, and `RELU` sites. Arithmetic NaN class/domain, [infinity](../../glossary.md#term-infinity), signed zero, [overflow](../../glossary.md#term-overflow),
and [underflow](../../glossary.md#term-underflow) rules remain explicit per family; no generic finite [tolerance](../../glossary.md#term-tolerance) decides a
special-value result. Composite formulas and generated [gradients](../../glossary.md#term-gradient) inherit only the rules of their
named arithmetic primitives, not a blanket final-output allowance. For example, `BFLOAT16`
`0x0001 * 128` may yield zero through input DAZ even though the exact low product is normal
`0x0080`; `FLOAT16` multiplication has the same permission, not a requirement to zero or a claim
about a particular backend's zeroing site. Family formulas and contributor domains remain intact.

For floating scatter `MUL`, the base and every addressed update contribute exactly once, but a
permitted rounded multiplication tree need not equal a once-rounded exact abstract product. With
`FLOAT32` factors `[maxFinite, 2, +0]`, an overflow-first tree can produce NaN from
`infinity * +0`, whereas an exact product rounded once gives `+0`. The revised contract permits the
former where the family/dtype arithmetic rules qualify it; it does not permit omitted or duplicate
contributors. Integral fixed-width modular `MUL`, exact [target](../../glossary.md#term-target) mapping, and raw unaddressed cells
remain unchanged. This conceptual example does not certify a [backend route](../../glossary.md#backend-route).

`BFLOAT16` and `FLOAT16` remain distinct storage types with their exact promotion and cast rules.
For a qualified [homogeneous](../../glossary.md#term-homogeneous) low-precision arithmetic occurrence, working values and accumulators
are `FLOAT32`, and each declared low output has one final [round-to-nearest, ties-to-even](../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../glossary.md#term-narrowing);
there is no hidden low pre-narrowing of inputs, contributors, accumulators, or saved values. This
is a Model semantic obligation; route qualification is separate. Direct mixed-
low execution is not newly admitted; an explicit `FLOAT32` cast remains the boundary.

Finite accuracy thresholds belong only in tests, scoped by family, dtype, backend, and size (and
route where routes differ), with an explicit [oracle](../../glossary.md#term-test-oracle), [metric](../../glossary.md#term-metric), threshold, and [cancellation](../../glossary.md#term-cancellation) boundary.
No threshold is selected by this [ADR](../../glossary.md#term-architecture-decision-record-adr). NaN/infinity, signed zero, subnormal/underflow, domains,
empty cases, and exact boundaries receive separate tests. There is no public accuracy envelope,
[runtime](../../glossary.md#term-runtime) tolerance, proof evaluator, or numerical [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions). A passing [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) or tolerance cannot
license [Compiler](../../glossary.md#term-compiler)'s guarded identity rewrites, `x * 0`, `x - x`, [reassociation](../../glossary.md#term-reassociation), [contraction](../../glossary.md#contraction-dimension), or
[fusion](../../glossary.md#term-fusion) across observable or special-value boundaries. Unproved transformations keep the old path.
Compiler-generated gradients remain ordinary public [Tensor](../../glossary.md#tensor) operations under the same single
Model semantics as forward operations. Their saved outputs, masks, indices, statistics, and state
are exact stored values; backward [preflight](../../glossary.md#term-preflight) and entire generated-[topology](../../glossary.md#term-topology) capability stay
[fail-closed](../../glossary.md#term-fail-closed). Removing the graph-wide profile does not create a gradient-only numerical policy.

## Rationale and evidence limits

The exact/stored boundary is independently useful: a bit-preserving move of low word `0x0001`
must remain a move even when arithmetic on that value underflows. The current
[CPU generated scatter regression](../../../modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScatterGeneratedKernelTest.java)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [regression glossary](../../glossary.md#term-regression))</span>
contains a `maxFinite * 2 * 0` witness for its exact `+0` result; that route-local result does not
exclude a [NaN](../../glossary.md#term-nan)-producing rounded tree under the revised rule. The
[Metal stored-subnormal regression](../../../modules/backend/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalStoredSubnormalProviderNativeTest.java)
records custom-program raw movement, cast, comparison/[extrema](../../glossary.md#saturation-pole-and-extremum), and arithmetic observations. These
tests are evidence about their specified routes, not universal [Model](../../glossary.md#term-model) rules or proof for other
routes. Task 0032's integrated [validation](../../glossary.md#term-validation) and independent review are recorded in its Result.

Historical readiness evidence in the [Model 0032 brief](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
reported that [Metal](../../glossary.md#term-metal) custom low-word `0x0001` survived cast/raw movement, while [`BFLOAT16`](../../glossary.md#term-bfloat16) minimum
[subnormal](../../glossary.md#term-subnormal) times `128` produced zero even though its exact low result is normal `0x0080`. Final
low-result [FTZ](../../glossary.md#term-ftz) alone could not explain that observation; input [DAZ](../../glossary.md#term-daz) or earlier [loss](../../glossary.md#loss) could. The
precise zeroing site remains unproved, and is not needed to define the allowed-result set.
[`FLOAT16`](../../glossary.md#term-float16) minimum subnormal times one had a distinct result, so no uniform low-type [device](../../glossary.md#term-device)
behavior follows. Those original route-local observations alone did not qualify [opaque](../../glossary.md#term-opaque)
[MPSGraph](../../glossary.md#term-mpsgraph), other [shapes](../../glossary.md#shape), or other routes; the later integrated [conformance](../../glossary.md#term-conformance) evidence is separate.

## Consequences and completed cutover

The benefit is one visible exact boundary and no [graph](../../glossary.md#graph)-wide numerical switch. The costs are [loss](../../glossary.md#loss)
of a caller-selected strict mode, changed floating results (notably comparison of [subnormals](../../glossary.md#term-subnormal) and
scatter-product specials), new family/[dtype](../../glossary.md#data-type--datatype) [qualification](../../glossary.md#term-qualification), and coordinated [identity](../../glossary.md#term-identity-meanings)/[API](../../glossary.md#term-application-programming-interface-api) migration.
No cross-[backend](../../glossary.md#backend) bitwise [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance) or performance improvement is promised.

The cutover reconciled the three Javadocs above, affected [Model](../../glossary.md#term-model) formulas, the glossary and user
guides, [Compiler](../../glossary.md#term-compiler)'s seven guarded rewrites and their existing phase/[descriptor](../../glossary.md#term-descriptor)/output/[gradient](../../glossary.md#term-gradient)
guards, generated gradients, and focused tests. It projected the actual pre-cutover [`ACCELERATOR`](../../glossary.md#term-accelerator)
[CPU](../../glossary.md#term-cpu)/[Metal](../../glossary.md#term-metal) provider **true and false** answers to 254 profile-free [occurrence](../../glossary.md#term-occurrence) rows, retaining
independent [`BFLOAT16`](../../glossary.md#term-bfloat16)/[`FLOAT16`](../../glossary.md#term-float16) queries and negative predicates. This representative ledger is
drift evidence, not enumeration or numerical certification. The implementation retains the fixed
Metal low `CUSTOM_PROGRAM` route, eligible [`FLOAT32`](../../glossary.md#term-float32) [MPSGraph](../../glossary.md#term-mpsgraph)/custom routes, transfers, Training,
tuning, and [fail-closed](../../glossary.md#term-fail-closed) unsupported occurrences. Only numerical-profile and numerical-
certification state was removed; source/package/[manifest](../../glossary.md#term-manifest) [authentication](../../glossary.md#term-authentication), native [application binary
interface](../../glossary.md#term-application-binary-interface-abi) ([ABI](../../glossary.md#application-binary-interface-and-native-bridge)) and export checks, decoder and [preflight](../../glossary.md#term-preflight) [validation](../../glossary.md#term-validation), bounds, [ownership](../../glossary.md#term-ownership-meanings),
stale-identity rejection, and fail-before-mutation rules remain. Coordinated identities are CPU
generator [schema](../../glossary.md#term-schema) 69, Metal program schema 20 with a 124-byte header, Metal identity 30, and
native ABI 7 with thirteen exports. CPU schema 68 and Metal schema 19/identity 29 are historical.

The revised [root invariants](../../../ARCHITECTURE.md#core-invariants),
[backend route qualification](../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling)<span data-glossary-navigation="true"> ([backend route glossary](../../glossary.md#backend-route))</span>,
and [low-precision capability evidence](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)<span data-glossary-navigation="true"> ([precision glossary](../../glossary.md#term-precision))</span>
specify route- and occurrence-scoped [conformance](../../glossary.md#term-conformance): exact structure/stored/special checks plus
explicit [finite](../../glossary.md#term-finite) test oracles, [metrics](../../glossary.md#machine-learning-and-hyperparameters), and thresholds. Task 0032 passed the integrated gate. The
cutover does not infer provider support from [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters), silently admit an unqualified route, or weaken
source-bound route evidence, [structural dominance](../../glossary.md#term-structural-dominance-index-mappings-and-cost-dominance), native ABI/source
authentication, fixed cold selection, or the ban on [runtime](../../glossary.md#term-runtime) fallback and retry.

The approved rule moved [Model 0032](../../planning/modules/model/tasks/0032-profile-free-numerical-semantics-reset.md)
into execution. Its route/site conformance, finite/special/exact tests, documentation, and
independent Class C review are complete in the main worktree. The source-matched signed local
Metal package has [SHA-256](../../glossary.md#term-sha-256) `77f683d4331a688423a5a5e14cb462e3d9bafdf1aaab3ae5963c20dfadb3d318`;
historical schema-19 evidence does not certify schema-20 programs. The MPSGraph [scalar](../../glossary.md#term-scalar) `+0`
[guard](../../glossary.md#term-guard)'s performance effect remains unmeasured, a nonblocking follow-up. Git history records
[publication](../../glossary.md#term-publication) separately from this decision record.

## Relationship to earlier decisions

This decision supersedes [ADR 0021](0021-total-recursive-accelerator-numerical-floor.md)'s
two-profile and recursive numerical-floor choice and [ADR 0023](0023-low-precision-accelerator-parity.md)'s
profile-indexed low numerical envelope/proof choice. [ADR 0019](0019-explicit-numerical-profiles.md)
was already superseded by 0021; its profile rationale remains historical. [ADR](../../glossary.md#term-architecture-decision-record-adr) 0023's provider
ledger and append-only [identity](../../glossary.md#term-identity-meanings) rationale remain relevant. [ADR 0025](0025-custom-only-metal-low-precision.md)'s
custom-only low route remains in force; its profile-dependent [trace](../../glossary.md#trace)/identity provisions are
superseded by this decision. ADR 0025 is not wholly superseded.

## Related documentation

- [Architecture authority and scope index](../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../glossary.md#term-scope))</span>
- [Current architecture explanation](../../architecture/current-architecture-plan.md)
- [Backend capability and safety contract](../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)<span data-glossary-navigation="true"> ([Backend capability glossary](../../glossary.md#backend-capability))</span>
- [Profile-free Compiler compile identity](../../architecture/contracts/compiler-autograd.md#profile-free-compile-identity)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [compile glossary](../../glossary.md#compile); [identity glossary](../../glossary.md#term-identity-meanings))</span>
- [Compiler-owned automatic differentiation](../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../glossary.md#autograd))</span>
- [Profile-free Engine/Prepare lifecycle](../../architecture/contracts/runtime-prepare-engine.md#profile-free-engine-and-prepare-lifecycle)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [Prepare glossary](../../glossary.md#prepare); [lifecycle glossary](../../glossary.md#lifecycle))</span>

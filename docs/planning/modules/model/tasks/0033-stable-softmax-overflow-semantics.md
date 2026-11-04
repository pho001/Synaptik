# [Model](../../../../glossary.md#term-model) 0033 — Decide stable [softmax](../../../../glossary.md#softmax--log-softmax) [overflow](../../../../glossary.md#term-overflow) semantics

**Editorial reading key (current context, not a status update).**

This historical task records the decision, implementation scope and reported evidence for replacing
literal unshifted exponentiation on finite softmax slices; it is not a current status update. In the
current [boundary
explanation](../../../../architecture/softmax-numerical-semantics.md#responsibilities-and-boundaries),
Model defines meaning; CPU realizes its supported route rather than redefining it. SoftmaxKind
describes the operation, while runtime tensor values reach CPU validation after producers execute
and before consumer writes. This semantic responsibility does not establish storage ownership.

A normalization slice groups values along the selected axis; finite excludes NaN and infinity.
Working precision means calculation format, not necessarily output storage format. The current
[worked
scenarios](../../../../architecture/softmax-numerical-semantics.md#scenario-that-follows-the-boundary)
explain why FLOAT32 `[1000, 1000]` yields `[0.5, 0.5]`, and why LOG_SOFTMAX need not be the
logarithm of rounded SOFTMAX output. That finite rule does not extend to non-finite slices: Model
retains separate special-value semantics, while CPU rejects such inputs before output mutation. Nor
does it authorize analogous changes to attention or loss formulas; the recorded tests, artifact
identity and timing evidence belong to this task’s historical scope.

**Mathematical reading illustration.** For a nonempty finite real slice $x_1,\ldots,x_n$ with
positive integer $n$, let $m=\max_{1\le j\le n}x_j$ and $Z=\sum_{j=1}^{n}\exp(x_j-m)$. Probabilities
are $p_i=\exp(x_i-m)/Z$ for $1\le i\le n$. At $n=2$ and $x=(1000,1000)$, both shifts are zero and
$Z=2$, so $p=(1/2,1/2)$. This explains the finite shift; non-finite inputs use a different branch,
and the specified working-site rounding remains authoritative.

## Status

Complete

## Goal and class

Class C numerical-contract and generated-code change. Resolve review G1: the current [CPU](../../../../glossary.md#term-cpu)
implementation avoids intermediate [EXP](../../../../glossary.md#term-expx) [overflow](../../../../glossary.md#term-overflow), while [Model](../../../../glossary.md#term-model)'s literal wording can require a
different output class. The [owner](../../../../glossary.md#term-owner) accepted stable [finite](../../../../glossary.md#term-finite) [softmax](../../../../glossary.md#softmax--log-softmax) after the final review report
and authorized the implementation in the follow-up conversation.

## Concrete conflict

For [contiguous](../../../../glossary.md#term-contiguous) [FLOAT32](../../../../glossary.md#term-float32) `[1000,1000]`, [axis](../../../../glossary.md#term-axis) 0, the current [CPU](../../../../glossary.md#term-cpu) provider/[finite](../../../../glossary.md#term-finite)-shift validator
admit execution. `CpuSoftmaxEmitter` subtracts the maximum, so [EXP](../../../../glossary.md#term-expx) sees `[0,0]` and [softmax](../../../../glossary.md#softmax--log-softmax)
returns `[0.5,0.5]`. `SoftmaxKind` and the foundational composite row instead describe EXP
of each unshifted value, a rounded sum and division: EXP overflows to [infinity](../../../../glossary.md#term-infinity) and infinity
divided by infinity is [NaN](../../../../glossary.md#term-nan). The alternative-algorithm clause requires special-value preservation;
finite-input [qualification](../../../../glossary.md#term-qualification) alone does not authorize changing that result class.

This was source-confirmed by an independent read-only review. Existing CPU tests use a stable
[oracle](../../../../glossary.md#term-test-oracle), not a demonstration that the literal contract authorizes the result. No new numerical
[run](../../../../glossary.md#term-run) was performed in that cross-check. CPU 0021's storage/[validation](../../../../glossary.md#term-validation) fixes and CPU 0022's
[activation](../../../../glossary.md#term-activation)-site repairs are separate and do not settle G1.

## Accepted decision and [scope](../../../../glossary.md#term-scope)

Explicitly define stable [finite](../../../../glossary.md#term-finite) [softmax](../../../../glossary.md#softmax--log-softmax) and [log-softmax](../../../../glossary.md#softmax--log-softmax) without
incidental unshifted-[EXP](../../../../glossary.md#term-expx) [overflow](../../../../glossary.md#term-overflow), retaining precise input [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity), empty, [dtype](../../../../glossary.md#data-type--datatype), [accumulator](../../../../glossary.md#accumulator-type)
and [publication](../../../../glossary.md#term-publication) rules. Preserve the current stable [CPU](../../../../glossary.md#term-cpu) algorithm where conforming; do not
replace useful finite probabilities with NaNs solely to reproduce accidental intermediate overflow.

The [owner](../../../../glossary.md#term-owner)'s earlier response deferred the decision until the review report. That report was
delivered before the later authorization. For an admitted finite [slice](../../../../glossary.md#slice) `[1000,1000]`, SOFTMAX
produces `[0.5,0.5]` and LOG_SOFTMAX produces two values near `-ln(2)`.

Define maximum selection, shifted exponentials, sum, probability/[log](../../../../glossary.md#term-log)-probability, working
[precision](../../../../glossary.md#term-precision), [narrowing](../../../../glossary.md#term-narrowing), accepted/rejected value domains and exact special cases. Check the current
[FLOAT32](../../../../glossary.md#term-float32) and [FLOAT16](../../../../glossary.md#term-float16) CPU emitter's double locals against the root's declared working sites and
correct demonstrated mismatches without weakening the root low-precision rule. Review attention
and [loss](../../../../glossary.md#loss) references explicitly; do not generalize this permission to other composites by analogy.

Non-goals: new dtype/[layout](../../../../glossary.md#layout)/provider/route support, public [API](../../../../glossary.md#term-application-programming-interface-api), numerical profile, production
[tolerance](../../../../glossary.md#term-tolerance) [registry](../../../../glossary.md#term-registry), broad performance campaign or unrelated attention/loss algorithm rewrite.

## Contracts and implementation packet

Read `ARCHITECTURE.md` headings `Authority, incorporation, and precedence`, `Core invariants`,
`Scope-indexed normative contracts`, `Documentation authority and architecture updates`, and
`Testing requirements`; `docs/architecture/contracts/foundational-modules.md` headings
`modules/model`, [`Profile-free numerical semantics`](../../../../glossary.md#profile-free-numerical-semantics) (especially `Floating arithmetic primitives`,
[`Aggregate floor`](../../../../glossary.md#term-aggregate-floor), `Composite inheritance`, `Model low-precision contract`); and
`docs/architecture/contracts/backend-execution.md` heading `CPU backend routes`.
If an applicable contract is ambiguous, stop and report it. Read `SoftmaxKind`, current [CPU](../../../../glossary.md#term-cpu)
emitter/validator/reference and focused tests.
Update root and the one owning scoped contract, family [Javadoc](../../../../glossary.md#term-javadoc), numerical explanation and a
significant [ADR](../../../../glossary.md#term-architecture-decision-record-adr) in one reviewed change. Add independent contract examples and [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)
regressions for [finite](../../../../glossary.md#term-finite) [EXP](../../../../glossary.md#term-expx) [overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow), [cancellation](../../../../glossary.md#term-cancellation)-relevant offsets, both kinds and [dtypes](../../../../glossary.md#data-type--datatype),
plus [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity)/empty behavior. Review consumers and compatibility [identities](../../../../glossary.md#term-identity-meanings) explicitly.
Keep test-only tolerances separate from class/[guard](../../../../glossary.md#term-guard) semantics and retain clean Java code [shape](../../../../glossary.md#shape).

Files and symbols: `SoftmaxKind.java`; `CpuSoftmaxEmitter.java`,
`CpuSoftmaxReferenceKernel.java`, `CpuSoftmaxInputValidator.java` when correction is required;
their focused tests; relevant backend conformance/integration tests; `docs/api/tensor-api.md`;
and targeted numerical/CPU explanation and glossary entries. The implementation context owns
the root, scoped contract, ADR, source/tests and affected [API](../../../../glossary.md#term-application-programming-interface-api) docs. The main coordinator owns
the roadmap and master-plan status, which are separate write scopes.

Acceptance: finite `[1000,1000]` produces finite normalized outputs; offset and underflow cases
show LOG_SOFTMAX need not equal a rounded `log(softmax)`; all four admitted floating dtypes are
covered; [Model](../../../../glossary.md#term-model)'s special-value/empty rules and CPU rejection-before-write behavior are distinct;
generated and clean Java sites align with the declared working [precision](../../../../glossary.md#term-precision); no per-[element](../../../../glossary.md#term-element)
allocation or [dispatch](../../../../glossary.md#term-dispatch) is introduced. Record the result of attention/[loss](../../../../glossary.md#loss) and CPU/[Metal](../../../../glossary.md#term-metal) [identity](../../../../glossary.md#term-identity-meanings)
review, including a reasoned no-change conclusion if appropriate.

Worker [validation](../../../../glossary.md#term-validation) after focused development tests:

```bash
./gradlew :modules:model:test :backends:cpu:test :testing:backend-conformance:test :modules:model:javadoc :backends:cpu:javadoc --max-workers=2 --console=plain
```

The coordinator runs affected integration selectors and one repository [checkpoint](../../../../glossary.md#term-checkpoint) if executable
changes span Model and CPU; it reuses the worker's successful [module](../../../../glossary.md#term-module) evidence. Generated emitter
changes require class-shape inspection against direct Java. An independent Class C review inspects
the final code, contract, tests, Javadoc, explanatory docs, glossary, and links; it does not rerun
successful Java suites without a concrete reason.

## Dependencies and integration

- Depends on: [owner](../../../../glossary.md#term-owner) decision above; [Model](../../../../glossary.md#term-model) 0032 Complete; review G1.
- Conflicts with: Model numerical semantics, [softmax](../../../../glossary.md#softmax--log-softmax) code/oracles and shared numerical docs.
- Parallel group: None; contract-first serial work.
- Common base revision: `4e5eb7baca6c16ddb02a05518d569e90aa557b76`.
- Integration order: decision/contract → scoped consumers and [conformance](../../../../glossary.md#term-conformance) → independent review.
- Integration validation: focused Model/CPU/conformance tests and Javadoc, affected integration
  selectors, and one repository-wide <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`./gradlew test javadoc --max-workers=2 --console=plain`
  checkpoint if executable changes cross module boundaries.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration owner: main coordinator for roadmap/master-plan status.

Frontier verification (2026-10-13): Model 0032 records `Status: Complete`; the final project
review delivered G1 and the owner then selected stable [finite](../../../../glossary.md#term-finite) semantics. Model 0033 is the sole
active Model frontier, no conflicting writer is active, and the base above is the clean [source
revision](../../../../glossary.md#term-source-revision) before these [planning](../../../../glossary.md#term-planning) edits. The implementation and coordinator write scopes above are
disjoint; contract and [backend](../../../../glossary.md#backend) changes are serial within this one task.

## Result

Status: Complete. Stable [finite](../../../../glossary.md#term-finite) [SOFTMAX](../../../../glossary.md#softmax--log-softmax) and LOG_SOFTMAX now have explicit [Model](../../../../glossary.md#term-model) authority,
explanatory documentation and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0029. [CPU](../../../../glossary.md#term-cpu) generated and direct bodies use the declared
working [precision](../../../../glossary.md#term-precision), retain the prior non-finite result boundary and CPU pre-write rejection,
and bind version-two [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings). [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) and Model special-case examples
cover the accepted finite and exceptional domains; current generated inventory and dependent
[digest](../../../../glossary.md#term-digest) fixtures were refreshed without changing other families.

The affected Model, CPU and backend-conformance tests, CPU integration selector, one full
repository `test` [checkpoint](../../../../glossary.md#term-checkpoint), and repository [`javadoc`](../../../../glossary.md#term-javadoc) checkpoint passed. A bounded one-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)
paired v2 diagnostic checked raw generated/direct equality for both kinds at [FLOAT32](../../../../glossary.md#term-float32) and
[FLOAT16](../../../../glossary.md#term-float16); its four observed generated/direct [median](../../../../glossary.md#term-median) ratios were 0.999465–1.021311x. This is
local evidence, not a universal performance guarantee. The initial independent Class C review
identified missing performance evidence and special-case examples; both were added and the
independent re-review returned `APPROVE` with no blocking findings. G1 is closed. No unresolved
task-specific follow-up remains; CPU 0023 and Training 0003A are separate Draft work.

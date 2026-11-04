# Task 0031: STRICT [Unary](../../../../glossary.md#term-unary) Numerical Baseline

## Status

Complete

## Change class

Class C — this closes previously deferred public numerical semantics for every current [unary](../../../../glossary.md#term-unary) kind
and becomes a prerequisite for the cross-[module](../../../../glossary.md#term-module) recursive [`ACCELERATOR`](../../../../glossary.md#term-accelerator) contract. Execution is
[Model](../../../../glossary.md#term-model)-owned documentation and [Javadoc](../../../../glossary.md#term-javadoc) only; it changes no executable path or [backend capability](../../../../glossary.md#backend-capability).

## Goal

Define the complete `STRICT_IEEE` allowed-result baseline for all nineteen
`UnaryElementwiseKind` values before Task 0030 broadens [`ACCELERATOR`](../../../../glossary.md#term-accelerator). Preserve the public
mathematical [targets](../../../../glossary.md#term-target) and current [CPU](../../../../glossary.md#term-cpu) results while making [finite](../../../../glossary.md#term-finite) accuracy, formula sites, domains,
special classes, [signed zero](../../../../glossary.md#term-signed-zero), and [subnormal](../../../../glossary.md#term-subnormal) behavior explicit enough for [conformance](../../../../glossary.md#term-conformance).

## [Scope](../../../../glossary.md#term-scope)

- Close `ABS`, `NEG`, `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), `LOG1P`, [`EXP`](../../../../glossary.md#term-expx), `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`,
  `FLOOR`, `CEIL`, `SIGN`, `RELU`, [`SIGMOID`](../../../../glossary.md#term-sigmoid), [`TANH`](../../../../glossary.md#term-tanh), [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu),
  `GELU_TANH_APPROXIMATION`, and `SILU` together; no kind remains deferred.
- Define ordered finite-representation distance for BFLOAT16, FLOAT32, and FLOAT64 with the
  sign-aware monotonic raw-bit key. The correctly rounded exact reference has distance zero and
  every bound is inclusive. Distance does not replace separate domain, range, NaN, infinity,
  overflow, underflow, subnormal, or signed-zero requirements.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [FLOAT64 glossary](../../../../glossary.md#term-float64); [correctly rounded glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Retain represented inputs exactly under `STRICT_IEEE`: no denormals-are-zero (DAZ),
  flush-to-zero (FTZ), reciprocal estimate, reduced precision, unlisted reassociation, fused
  operation, or cross-node contraction.<span data-glossary-navigation="true"> ([denormals-are-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [DAZ glossary](../../../../glossary.md#term-daz); [flush-to-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [FTZ glossary](../../../../glossary.md#term-ftz); [precision glossary](../../../../glossary.md#term-precision); [reassociation glossary](../../../../glossary.md#term-reassociation); [fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
- Make exact/discrete results [backend](../../../../glossary.md#backend)-independent: represented sign edit for `ABS`/`NEG`, one exact
  ties-to-even same-format division for <span data-glossary-navigation="true"> ([ties-to-even glossary](../../../../glossary.md#term-ties-to-even))</span>`RECIPROCAL`, exact `FLOOR`/`CEIL`, exact sign
  classification, and family-extrema <span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>`max(x,+0)` for [ReLU](../../../../glossary.md#term-relu).
- Use correctly rounded exact same-format references and inclusive ordered-distance ceilings of
  two for `LOG`, `LOG1P`, `EXP`, and `EXPM1`; one for `SQRT`; and five for `TANH`, in every
  accepted floating format. `LOG1P` and `EXPM1` reference the exact real function rather than a
  separately rounded addition or subtraction.
- Define `ERF` from the exact real error-function integral. Its finite result must be finite,
  same-signed, nonzero for nonzero input, and inside `[-1,1]`, with inclusive error
  `max(A, R*abs(reference))`: `A=R=2^-7` for [BFLOAT16](../../../../glossary.md#term-bfloat16), `2e-5` for [FLOAT32](../../../../glossary.md#term-float32), and `2e-7` for
  FLOAT64. Those bounds are Model semantics supported by retained CPU conformance, not an
  implementation identity.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Define `RSQRT`, `SIGMOID`, both GELU forms, and `SILU` recursively from Model-owned primitive
  result sets. A realization evaluates in the result format or, for BFLOAT16/FLOAT32, one wider
  format followed by one final ties-to-even narrowing. Every constant rounds once before use;
  every named negation, addition, multiplication, and division is one ties-to-even site; no
  reassociation or contraction is allowed.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [constant glossary](../../../../glossary.md#term-constant))</span>
- Enumerate every composite data [dependency](../../../../glossary.md#term-dependency). `RSQRT` is square root then division. Sigmoid and SiLU
  use their sign-stable exponential branches. Exact GELU names <span data-glossary-navigation="true"> ([SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu))</span>`0.5*x`, `sqrt(2)`, division,
  `ERF`, addition, and final multiplication. Tanh GELU names `x*x`, `x^2*x`, `0.044715*x^3`,
  addition, `2/pi`, [square root](../../../../glossary.md#norms-elementary-functions-and-probability-distributions), multiplication, `TANH`, addition, `0.5*x`, and final
  multiplication. No composite receives a whole-result strict or accelerator envelope.<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator))</span>
- State every top-level finite domain, signed-zero, infinity, and NaN-class result. NaN payload,
  quieting, and sign remain unspecified, but NaN classification is mandatory.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Reconcile the Model-owned numerical-profile contract, unary kind and Tensor Javadocs, public
  Tensor API reference, glossary, Model capability/master plans, Task 0030 evidence, and roadmap.
  Keep executable/API/capability behavior unchanged.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Non-goals

No executable Java/native statement, enum value/order, public method, [signature](../../../../glossary.md#term-signature), attribute,
[operation](../../../../glossary.md#operation) construction, [gradient](../../../../glossary.md#term-gradient), [backend capability](../../../../glossary.md#backend-capability), route, [schema](../../../../glossary.md#term-schema), wire, [ABI](../../../../glossary.md#term-application-binary-interface-abi), [cache](../../../../glossary.md#term-cache), tuning,
hot-path policy, [conformance](../../../../glossary.md#term-conformance) test, or unrelated historical completed/blocked task rewrite.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
- [CPU 0005H retained unary evidence](../../../backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>
- [CPU 0005I retained FLOAT32 vector evidence](../../../backends/cpu/tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [vector glossary](../../../../glossary.md#term-vector))</span>

If retained [CPU](../../../../glossary.md#term-cpu) behavior cannot be shown to lie inside a proposed strict allowed-result set, stop
rather than tighten the public contract or change execution.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0018T1 and 0019A; completed [CPU](../../../../glossary.md#term-cpu) 0005H/0005I evidence; Model 0027–0029.
- Conflicts with: Task 0030 remediation and every concurrent numerical-profile contract, unary
  Javadoc, Tensor API, glossary, Model master, or roadmap edit.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Parallel group: None.
- Common base revision: `4bd26e72a4f3aa71a6607269991f1b96b880812f`.
- Review-remediation base revision: `2d95ab71683698753c9ac64d9373fde8301c9404`.
- Integration order: first; Task 0030 remediation depends on this completed baseline.
- Integration [validation](../../../../glossary.md#term-validation): focused Model documentation/[Javadoc](../../../../glossary.md#term-javadoc) and architecture [checkpoint](../../../../glossary.md#term-checkpoint).
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner.

All dependencies are Complete and retained evidence is present. Task 0031 remains Complete while
this same-review remediation closes its [finite](../../../../glossary.md#term-finite)-result omission; Task 0030 remains `Review needed`
until independent rereview accepts the corrected prerequisite and [accelerator](../../../../glossary.md#term-accelerator) contract.

## Files and symbols

- `ARCHITECTURE.md` and `docs/architecture/contracts/foundational-modules.md` — strict unary
  ownership and normative allowed-result baseline.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [normative glossary](../../../../glossary.md#term-normative))</span>
- `UnaryElementwiseKind` and the nineteen [`Tensor`](../../../../glossary.md#tensor) unary-method Javadocs — exact formulas,
  references, bounds, domain, special classes, signed zero, and subnormals.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal))</span>
- `docs/api/tensor-api.md`, `docs/glossary.md`, and Model capabilities — public explanation and
  terminology.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- ADR 0021, Task 0030, this task, Model master, and roadmap — dependency, evidence, and review
  frontier.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Acceptance criteria

- All nineteen unary kinds and every accepted BFLOAT16/FLOAT32/FLOAT64 type have one complete,
  backend-independent strict allowed-result assignment and complete exceptional/signed-zero rules.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [backend glossary](../../../../glossary.md#backend); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Exact mathematical references, ordered keys for all three formats, inclusive primitive bounds,
  and [`ERF`](../../../../glossary.md#term-erf) absolute/relative coefficients are unambiguous.
- Current CPU FLOAT32/FLOAT64 scalar and vector results remain admitted by retained conformance
  evidence, without making an implementation or coefficient table semantic authority.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- `RSQRT`, sigmoid, GELU, tanh GELU, and SiLU recursively enumerate every primitive, constant, and
  one-round arithmetic site in the native/one-wider union; no whole-operation envelope is used.<span data-glossary-navigation="true"> ([sigmoid glossary](../../../../glossary.md#term-sigmoid); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [tanh glossary](../../../../glossary.md#term-tanh); [SiLU glossary](../../../../glossary.md#sigmoid-linear-unit--silu); [constant glossary](../../../../glossary.md#term-constant); [operation glossary](../../../../glossary.md#operation))</span>
- Unsupported [BFLOAT16](../../../../glossary.md#term-bfloat16) execution still has complete [finite](../../../../glossary.md#term-finite) semantics without implying capability.
- Completed Task 0031 remains the explicit strict prerequisite for Task 0030 remediation and
  re-review.
- No executable/[API](../../../../glossary.md#term-application-programming-interface-api)/capability/[schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache)/hot-path behavior changes.

## [Validation](../../../../glossary.md#term-validation)

```bash
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
git diff --check
```

Inspect rendered [unary](../../../../glossary.md#term-unary)/[Tensor](../../../../glossary.md#tensor) Javadocs; validate changed Markdown links, anchors, fences, and
newlines; compare every [finite](../../../../glossary.md#term-finite) bound and special-value row with retained [CPU](../../../../glossary.md#term-cpu) 0005H/0005I evidence;
and confirm the Java diff contains documentation comments only.

## Documentation and review impact

The task is documentation/[Javadoc](../../../../glossary.md#term-javadoc)-only but closes a durable public numerical contract. Independent
Class C review reused Task 0030's successful executable evidence because this task changes no
executable statement.

## Result

Completed the documentation/[Javadoc](../../../../glossary.md#term-javadoc)-only strict [unary](../../../../glossary.md#term-unary) decision for all nineteen current kinds and
all accepted [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) types. The [normative](../../../../glossary.md#term-normative) contract now owns exact mathematical
references; ordered-representation keys and inclusive distance bounds of two, one, and five for
the applicable primitives; [`ERF`](../../../../glossary.md#term-erf) absolute/relative coefficients `2^-7`, `2e-5`, and `2e-7`; exact
special/domain/[subnormal](../../../../glossary.md#term-subnormal) rules; and recursive native/one-wider composite site [graphs](../../../../glossary.md#graph). No [CPU](../../../../glossary.md#term-cpu),
library, coefficient table, or route defines semantics.

Retained CPU 0005H/0005I [conformance](../../../../glossary.md#term-conformance) supplies the bound rationale and remains admitted: the
one-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) logarithmic/exponential contract converts to two ordered steps at a [binade](../../../../glossary.md#term-binade) boundary,
2.5-ULP [tanh](../../../../glossary.md#term-tanh) converts to five, square-root evidence fits one, and direct-reference error-function
gates supply the FLOAT32/FLOAT64 coefficients. The BFLOAT16 coefficient dominates one BFLOAT16
[narrowing](../../../../glossary.md#term-narrowing) plus the retained FLOAT32 error. Unary kind, [Tensor](../../../../glossary.md#tensor), public [API](../../../../glossary.md#term-application-programming-interface-api), glossary, capability,
[Model](../../../../glossary.md#term-model) master, Task 0030, and roadmap wording are synchronized. No executable statement, API [shape](../../../../glossary.md#shape),
capability, route, [schema](../../../../glossary.md#term-schema), [cache](../../../../glossary.md#term-cache), tuning, or hot-path behavior changed.

Focused verification passed:
`./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test` was
`BUILD SUCCESSFUL` with 23 actionable tasks (three executed, twenty up-to-date). Markdown
[validation](../../../../glossary.md#term-validation) covered ten changed documents, 481 local links/anchors, fences, and final newlines;
`git diff --check` also passed.

The initial strict baseline is `a08db430f2d01f8250b74b65bea488bbc06b5860`; the complete
[backend](../../../../glossary.md#backend)-independent result-set remediation is
`97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`. Independent Class C rereview after `97cb9d11`
returned `APPROVE` with zero findings. Status remains `Complete`.

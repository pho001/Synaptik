# Task 0031: STRICT Unary Numerical Baseline

## Status

Complete

## Change class

Class C — this closes previously deferred public numerical semantics for every current unary kind
and becomes a prerequisite for the cross-module recursive `ACCELERATOR` contract. Execution is
Model-owned documentation and Javadoc only; it changes no executable path or backend capability.

## Goal

Define the complete `STRICT_IEEE` allowed-result baseline for all nineteen
`UnaryElementwiseKind` values before Task 0030 broadens `ACCELERATOR`. Preserve the public
mathematical targets and current CPU results while making finite accuracy, formula sites, domains,
special classes, signed zero, and subnormal behavior explicit enough for conformance.

## Scope

- Close `ABS`, `NEG`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`,
  `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`, `GELU`,
  `GELU_TANH_APPROXIMATION`, and `SILU` together; no kind remains deferred.
- Define ordered finite-representation distance for BFLOAT16, FLOAT32, and FLOAT64 with the
  sign-aware monotonic raw-bit key. The correctly rounded exact reference has distance zero and
  every bound is inclusive. Distance does not replace separate domain, range, NaN, infinity,
  overflow, underflow, subnormal, or signed-zero requirements.
- Retain represented inputs exactly under `STRICT_IEEE`: no denormals-are-zero (DAZ),
  flush-to-zero (FTZ), reciprocal estimate, reduced precision, unlisted reassociation, fused
  operation, or cross-node contraction.
- Make exact/discrete results backend-independent: represented sign edit for `ABS`/`NEG`, one exact
  ties-to-even same-format division for `RECIPROCAL`, exact `FLOOR`/`CEIL`, exact sign
  classification, and family-extrema `max(x,+0)` for ReLU.
- Use correctly rounded exact same-format references and inclusive ordered-distance ceilings of
  two for `LOG`, `LOG1P`, `EXP`, and `EXPM1`; one for `SQRT`; and five for `TANH`, in every
  accepted floating format. `LOG1P` and `EXPM1` reference the exact real function rather than a
  separately rounded addition or subtraction.
- Define `ERF` from the exact real error-function integral. Its finite result must be finite,
  same-signed, nonzero for nonzero input, and inside `[-1,1]`, with inclusive error
  `max(A, R*abs(reference))`: `A=R=2^-7` for BFLOAT16, `2e-5` for FLOAT32, and `2e-7` for
  FLOAT64. Those bounds are Model semantics supported by retained CPU conformance, not an
  implementation identity.
- Define `RSQRT`, `SIGMOID`, both GELU forms, and `SILU` recursively from Model-owned primitive
  result sets. A realization evaluates in the result format or, for BFLOAT16/FLOAT32, one wider
  format followed by one final ties-to-even narrowing. Every constant rounds once before use;
  every named negation, addition, multiplication, and division is one ties-to-even site; no
  reassociation or contraction is allowed.
- Enumerate every composite data dependency. `RSQRT` is square root then division. Sigmoid and SiLU
  use their sign-stable exponential branches. Exact GELU names `0.5*x`, `sqrt(2)`, division,
  `ERF`, addition, and final multiplication. Tanh GELU names `x*x`, `x^2*x`, `0.044715*x^3`,
  addition, `2/pi`, square root, multiplication, `TANH`, addition, `0.5*x`, and final
  multiplication. No composite receives a whole-result strict or accelerator envelope.
- State every top-level finite domain, signed-zero, infinity, and NaN-class result. NaN payload,
  quieting, and sign remain unspecified, but NaN classification is mandatory.
- Reconcile the Model-owned numerical-profile contract, unary kind and Tensor Javadocs, public
  Tensor API reference, glossary, Model capability/master plans, Task 0030 evidence, and roadmap.
  Keep executable/API/capability behavior unchanged.

## Non-goals

No executable Java/native statement, enum value/order, public method, signature, attribute,
operation construction, gradient, backend capability, route, schema, wire, ABI, cache, tuning,
hot-path policy, conformance test, or unrelated historical completed/blocked task rewrite.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
- [CPU 0005H retained unary evidence](../../../backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md)
- [CPU 0005I retained FLOAT32 vector evidence](../../../backends/cpu/tasks/0005i-float32-vector-parity-and-vector-emission-boundary.md)

If retained CPU behavior cannot be shown to lie inside a proposed strict allowed-result set, stop
rather than tighten the public contract or change execution.

## Dependencies and integration

- Depends on: Model 0018T1 and 0019A; completed CPU 0005H/0005I evidence; Model 0027–0029.
- Conflicts with: Task 0030 remediation and every concurrent numerical-profile contract, unary
  Javadoc, Tensor API, glossary, Model master, or roadmap edit.
- Parallel group: None.
- Common base revision: `4bd26e72a4f3aa71a6607269991f1b96b880812f`.
- Review-remediation base revision: `2d95ab71683698753c9ac64d9373fde8301c9404`.
- Integration order: first; Task 0030 remediation depends on this completed baseline.
- Integration validation: focused Model documentation/Javadoc and architecture checkpoint.
- Shared-document integration owner: Main planner.

All dependencies are Complete and retained evidence is present. Task 0031 remains Complete while
this same-review remediation closes its finite-result omission; Task 0030 remains `Review needed`
until independent rereview accepts the corrected prerequisite and accelerator contract.

## Files and symbols

- `ARCHITECTURE.md` and `docs/architecture/contracts/foundational-modules.md` — strict unary
  ownership and normative allowed-result baseline.
- `UnaryElementwiseKind` and the nineteen `Tensor` unary-method Javadocs — exact formulas,
  references, bounds, domain, special classes, signed zero, and subnormals.
- `docs/api/tensor-api.md`, `docs/glossary.md`, and Model capabilities — public explanation and
  terminology.
- ADR 0021, Task 0030, this task, Model master, and roadmap — dependency, evidence, and review
  frontier.

## Acceptance criteria

- All nineteen unary kinds and every accepted BFLOAT16/FLOAT32/FLOAT64 type have one complete,
  backend-independent strict allowed-result assignment and complete exceptional/signed-zero rules.
- Exact mathematical references, ordered keys for all three formats, inclusive primitive bounds,
  and `ERF` absolute/relative coefficients are unambiguous.
- Current CPU FLOAT32/FLOAT64 scalar and vector results remain admitted by retained conformance
  evidence, without making an implementation or coefficient table semantic authority.
- `RSQRT`, sigmoid, GELU, tanh GELU, and SiLU recursively enumerate every primitive, constant, and
  one-round arithmetic site in the native/one-wider union; no whole-operation envelope is used.
- Unsupported BFLOAT16 execution still has complete finite semantics without implying capability.
- Completed Task 0031 remains the explicit strict prerequisite for Task 0030 remediation and
  re-review.
- No executable/API/capability/schema/cache/hot-path behavior changes.

## Validation

```bash
./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test
git diff --check
```

Inspect rendered unary/Tensor Javadocs; validate changed Markdown links, anchors, fences, and
newlines; compare every finite bound and special-value row with retained CPU 0005H/0005I evidence;
and confirm the Java diff contains documentation comments only.

## Documentation and review impact

The task is documentation/Javadoc-only but closes a durable public numerical contract. Independent
Class C review reused Task 0030's successful executable evidence because this task changes no
executable statement.

## Result

Completed the documentation/Javadoc-only strict unary decision for all nineteen current kinds and
all accepted BFLOAT16/FLOAT32/FLOAT64 types. The normative contract now owns exact mathematical
references; ordered-representation keys and inclusive distance bounds of two, one, and five for
the applicable primitives; `ERF` absolute/relative coefficients `2^-7`, `2e-5`, and `2e-7`; exact
special/domain/subnormal rules; and recursive native/one-wider composite site graphs. No CPU,
library, coefficient table, or route defines semantics.

Retained CPU 0005H/0005I conformance supplies the bound rationale and remains admitted: the
one-ULP logarithmic/exponential contract converts to two ordered steps at a binade boundary,
2.5-ULP tanh converts to five, square-root evidence fits one, and direct-reference error-function
gates supply the FLOAT32/FLOAT64 coefficients. The BFLOAT16 coefficient dominates one BFLOAT16
narrowing plus the retained FLOAT32 error. Unary kind, Tensor, public API, glossary, capability,
Model master, Task 0030, and roadmap wording are synchronized. No executable statement, API shape,
capability, route, schema, cache, tuning, or hot-path behavior changed.

Focused verification passed:
`./gradlew :modules:model:test :modules:model:javadoc :testing:architecture-tests:test` was
`BUILD SUCCESSFUL` with 23 actionable tasks (three executed, twenty up-to-date). Markdown
validation covered ten changed documents, 481 local links/anchors, fences, and final newlines;
`git diff --check` also passed.

The initial strict baseline is `a08db430f2d01f8250b74b65bea488bbc06b5860`; the complete
backend-independent result-set remediation is
`97cb9d116bd85ee6a0dfbf2e9b70d32604633c85`. Independent Class C rereview after `97cb9d11`
returned `APPROVE` with zero findings. Status remains `Complete`.

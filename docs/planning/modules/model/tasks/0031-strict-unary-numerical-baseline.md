# Task 0031: STRICT Unary Numerical Baseline

## Status

Ready

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
- Define an ordered floating-representation distance for finite same-format results. The reference
  has distance zero and a bound is inclusive. Distance does not replace separate NaN, infinity,
  domain, overflow, underflow, or signed-zero rules.
- Retain represented inputs exactly under `STRICT_IEEE`: no denormals-are-zero (DAZ),
  flush-to-zero (FTZ), reciprocal estimate, reduced precision, reassociation, or cross-node
  contraction. Ordinary primitive arithmetic uses same-format round-to-nearest, ties-to-even.
- Define strict ordinary finite outputs as the exact union of the current selected scalar and vector
  realizations, not as the loose conformance tolerances used to test them. Exact rows use the
  represented result of `ABS`, `NEG`, one typed `+1 / x`, `FLOOR`, `CEIL`, `SIGN`, and
  `max(x,+0)` for ReLU. `SQRT` uses its correctly rounded scalar reference and the retained vector
  result at distance at most one from that reference. NaN payload/sign are unspecified; NaN
  classification is mandatory.
- `LOG`, `LOG1P`, `EXP`, and `EXPM1` use their selected scalar `StrictMath` result, with exact
  widening and one final narrowing for binary32, or a retained vector result at distance at most
  two from that scalar reference. `TANH` similarly uses its selected scalar result or a retained
  vector result at distance at most five. These are inclusive scalar-relative route bounds, not
  claims about the exact mathematical reference.
- Define `RSQRT` against the retained first-class `1 / sqrt(x)` scalar realization: exact widening
  of a represented binary32 input to binary64, binary64 `StrictMath.sqrt`, one binary64 division,
  and one final ties-to-even binary32 narrowing; the retained vector result may be at most two
  ordered binary32 representations from that scalar reference. Record the corresponding binary64
  realization without inventing an exact-mathematical proof.
- Define strict `ERF` and exact GELU as the union of their selected scalar Cephes-derived
  realization and selected typed vector realization, each with its fixed coefficients, branches,
  operation order, and special corrections. Define `SIGMOID`, tanh GELU, and `SILU` by their
  selected stable scalar formulas and fixed evaluation order. The retained
  `max(2e-5, 2e-5 * abs(reference))` and `max(2e-7, 2e-7 * abs(reference))` test tolerances remain
  backend qualification evidence, not new public strict result envelopes. In particular they
  cannot admit FTZ or a wrong special class.
- Enumerate every primitive site in the three first-class composite activation formulas. Exact
  GELU uses typed `0.5`, typed `1`, typed `sqrt(2)`, division, `ERF`, addition, and multiplication.
  Tanh GELU additionally names typed `2`, typed `pi`, typed `0.044715`, `x*x`, the resulting
  `x^3`, additions, multiplications, division/root for `sqrt(2/pi)`, and `TANH`. SiLU uses its
  sign-stable `SIGMOID`/`EXP`, additions, division, and final multiplication. Constants are rounded
  once to the result format where the formula names a typed constant.
- Reconcile the Model-owned numerical-profile contract, unary kind and Tensor Javadocs, public
  Tensor API reference, glossary, Model master, and roadmap. Keep Task 0030 review-blocked until
  this prerequisite is Complete, then make its dependency and baseline explicit.

## Non-goals

No executable Java/native statement, enum value/order, public method, signature, attribute,
operation construction, gradient, backend capability, route, schema, wire, ABI, cache, tuning,
hot-path policy, conformance test, or historical completed/blocked task rewrite.

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
- Integration order: first; Task 0030 remediation depends on this completed baseline.
- Integration validation: focused Model documentation/Javadoc and architecture checkpoint.
- Shared-document integration owner: Main planner.

All dependencies are Complete and the retained evidence is present at the common base. Task 0031
is the sole authorized repository frontier; Task 0030 remains `Review needed`, but final review is
blocked until this prerequisite completes and its remediation rests on the explicit baseline.

## Files and symbols

- `docs/architecture/contracts/foundational-modules.md` — strict unary allowed-result baseline.
- `UnaryElementwiseKind` and the nineteen `Tensor` unary-method Javadocs — exact formulas,
  references, bounds, domain, special classes, signed zero, and subnormals.
- `docs/api/tensor-api.md` and `docs/glossary.md` — public explanation and terminology.
- This task, Model master, and roadmap — status, evidence, and dependency frontier.

## Acceptance criteria

- All nineteen unary kinds have one explicit strict allowed-result assignment and complete
  exceptional/signed-zero treatment; no accuracy or formula-site decision is deferred.
- Current CPU FLOAT32/FLOAT64 scalar and vector results remain admitted by evidence-backed bounds.
- Ordered distance, reference zero, inclusive bounds, ordinary-finite scope, and zero/NaN/infinity
  exclusions are unambiguous.
- GELU, tanh GELU, and SiLU formulas enumerate their primitive exponent and constant sites; no
  whole-operation envelope is mistaken for Task 0030 accelerator recursion.
- Task 0030 names completed 0031 as a prerequisite before its own remediation can remain
  `Review needed`.
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
Class C review is mandatory and may reuse Task 0030's successful executable evidence because this
task changes no executable statement.

## Result

Pending execution.

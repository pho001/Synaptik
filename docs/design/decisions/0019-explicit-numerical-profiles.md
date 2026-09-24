# ADR 0019: Explicit Numerical Profiles

## Status

Accepted — 2026-09-24

## Context

Synaptik's Model operations already define family-specific numerical contracts. Some require exact
represented behavior; some deliberately leave NaN payloads or intermediate precision unspecified;
and contraction families already permit reassociation and fused multiply-add. Several accelerator
routes cannot satisfy particular current `FLOAT32` subnormal or reduction results, but treating all
such differences as generic fast math would also admit term loss, unrelated graph rewrites, gross
special-value errors, and behavior that no operation contract authorizes.

The [foundational numerical-profile contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
is the sole normative definition of the profiles and operation-family allowed-result sets. This
record explains why that bounded design was selected.

## Decision drivers

- Preserve every operation's current contract as the strict baseline without falsely promising
  universal bitwise identity, correct rounding, Java `strictfp`, or fixed instructions.
- Make any wider result set explicit, opt-in, graph-wide, and operation-specific.
- Bound denormals-are-zero (DAZ), flush-to-zero (FTZ), reassociation, and FMA permissions tightly
  enough to exclude term loss, arbitrary rewrites, tolerance acceptance, and gross special-value
  errors.
- Keep semantic authority in Model while Config declares only identity and Planning asks only
  profile-qualified capability questions.
- Select all execution behavior before the Runtime hot path and keep backend preparation
  responsible for truthful realization.
- Prevent cache, specialization, generated-artifact, or tuning decisions from crossing profile
  boundaries.

## Options considered

### One fast-math boolean

A boolean cannot say which operation family or transformation it permits. It would encourage each
backend to invent a different meaning and would conflate DAZ/FTZ, reassociation, contraction,
approximation, reduced precision, and algebraic rewriting. Rejected.

### Generic tolerance or ULP budget

Tolerance-based acceptance does not define special-value, signed-zero, NaN, term-membership, or
selected-bit behavior. It would also incorrectly give transcendental and other unlisted families a
blanket approximation allowance. Rejected.

### Per-step reduction exact-zero sign choice

Allowing either sign after every exact-zero reduction addition is broader than the observed
difference requires. Intermediate zero sign does not create a new final nonzero value or
classification for the permitted reduction trees, while exposing the choice at every node makes
the allowed-result set harder to audit. Rejected.

### Per-backend numerical modes

Backend-specific semantics would make the same graph request mean different things depending on
availability or owner selection. Hardware and route capability must answer a Model-owned semantic
question, not define it. Rejected.

### Metal-specific reduction exception

A Metal-only rule would let a backend redefine Model semantics and make one graph request depend
on owner selection. The operation family must own any justified result freedom, and every backend
must separately qualify support before advertising it. Rejected.

### Per-operation or per-node caller selection

Fine-grained selection would complicate compilation, ownership, fusion boundaries, cache identity,
and user reasoning before there is evidence that such complexity is necessary. It could also let
an implementation erase observable intermediate values through accidental cross-node contraction.
Rejected.

### Two graph-wide profiles with bounded operation rows

One graph-wide identity can preserve the existing result set or opt into a precisely enumerated
superset. Model remains the single semantic owner while later layers transport and realize the
choice. Selected.

## Decision

Synaptik defines two graph numerical-profile identities. `STRICT_IEEE` denotes each operation's
exact current allowed-result set, including all family-specific promises and freedoms.
`ACCELERATOR` is an opt-in superset: a strict result is always valid, and an additional result is
valid only when the sole normative table explicitly permits the transformation for that named
operation family and `FLOAT32`.

The bounded additions are DAZ/FTZ and zero-sign freedom for listed arithmetic, all-and-only-term
binary-tree reassociation for `SUM`, `MEAN`, and `SUM_TO_SHAPE`, existing contraction-scoped
reassociation/FMA plus DAZ/FTZ for `MATMUL` and convolution, DAZ-normalized floating comparisons,
bounded extrema ties, and arithmetic-step-only DAZ/FTZ for cumulative scans. For an accelerator
`FLOAT32` `SUM` or arithmetic `SUM_TO_SHAPE` cell with at least two terms, either zero sign is
permitted only when the selected tree's root addition is exact zero. For `MEAN`, the declared
positive count remains a mandatory `FLOAT32` divisor and either sign is permitted only when that
quotient is exact zero. Intermediate exact-zero additions receive no new sign freedom. One-term,
point-domain, and equal-Shape copy forms remain bit-preserving; empty SUM remains positive zero and
zero-count MEAN remains NaN. FMA can contract only a corresponding multiply/add in a declared
contraction. It cannot fuse arbitrary graph nodes or erase an observable intermediate. Movement,
selection, classification, Boolean logic, casts, indexing, ordering, arg-extrema, unlisted
operations, and every type other than `FLOAT32` receive no relaxation.

Neither profile permits term dropping, reciprocal substitution, algebraic identity rewriting,
arbitrary reduced precision, cross-node contraction, tolerance-based acceptance, or a generic
transcendental error budget. In particular, `maxFinite / maxFinite -> 0`,
`+infinity / maxFinite -> NaN`, `RELU(NaN) -> +0`, and `TANH(NaN) -> +1` remain invalid. Movement,
`CONTIGUOUS`, `CAST`, and selected `WHERE` values do not inherit DAZ/FTZ or NaN-payload freedom from
neighboring arithmetic.

The choice is graph-wide and cold. Config owns only the immutable declarative identity. Engine
construction selects one value, defaulting to `STRICT_IEEE`, and the current propagation spine
carries it through Planning, Compiler, Prepare, and backend plan/cache identity. Planning asks
profile-qualified capability questions without interpreting semantics. Runtime executes the
prepared recipe without profile policy or a hot-path lookup.

The selected profile is part of relevant backend route, specialization, generated-artifact,
local-tuning, complete-plan-tuning, and cache-compatibility identity before a backend may advertise
relaxed capability. CPU now advertises both profiles with identical exact behavior. Metal
advertises exact canonical FLOAT32 `ABS` under both profiles, retains its strict
NEG/affine/`CONTIGUOUS` matrix, and separately advertises accelerator tensor-binary
`ADD`/`SUB`/`MUL`/`DIV` plus canonical `SUM`/`MEAN`/`SUM_TO_SHAPE` reductions after bounded
conformance. Unsupported pairs still fail closed; no backend may infer permission merely from the
identity.

Trace payload changes remain deferred. The current propagation spine adds no trace field because
the profile is cold prepared identity rather than per-run state; later observability requires a
separately coordinated Trace architecture update if operational evidence shows it is needed.

## Rationale

A disposable bounded raw-bit comparison covered signed zeros, cancellation, subnormals,
infinities, and NaNs across 2,427 one- through four-term multisets and 147,471 permitted ordered
binary trees. The final-result rule and a hypothetical rule allowing either sign after every
exact-zero addition produced identical raw-bit output sets for both SUM and the mandatory
positive-count MEAN quotient. Result-level freedom is therefore sufficient; per-step freedom adds
no required result.

The rule is tied to exact zero rather than a tolerance, so it does not admit a nearby nonzero
value. It changes neither term membership nor NaN/infinity classification, and it cannot turn a
copy into arithmetic by adding an identity. Keeping the rule in the Model-owned operation row
rather than naming Metal also preserves one semantic question for every backend capability check.

## Consequences

### Positive

- Current strict behavior and existing backend evidence remain valid.
- Accelerator permission is auditable by operation family rather than inferred from hardware or a
  performance objective.
- A backend may incrementally support either profile and may use strict results under both.
- Gross errors and semantic leakage into movement, selection, casting, classification, or other
  data types remain excluded.
- Graph-wide cold selection keeps Runtime free of numerical-policy branching.

### Negative and limits

- Backends must qualify capability and cache identity per profile before exposing new routes.
- `ACCELERATOR` does not automatically unlock unlisted unary, transcendental, normalization, loss,
  attention, pooling, scatter/fold, random, or recurrent operations.
- A future operation relaxation requires another coordinated architecture update with a named
  finite-domain and special-value envelope.
- Per-node profiles and trace reporting remain unavailable.

### Follow-up

CPU task 0017 and Metal task 0015 completed the first bounded backend realizations with
profile-specific conformance evidence. Later operations remain independently gated by the sole
normative table, operation-specific evidence, and fail-closed capability.

## Related documentation

- [Authoritative foundational numerical-profile contract](../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Architecture authority index](../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
- [Task 0027](../../planning/modules/model/tasks/0027-explicit-numerical-profile-semantic-authority.md)
- [Engine task 0018](../../planning/modules/engine/tasks/0018-explicit-numerical-profile-propagation-spine.md)

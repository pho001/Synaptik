# Task 0058: Remaining Elementwise Arithmetic

## Status

Active

Planned from clean base `118aedca`. The planning revision is recorded before any production edit.

## Change class

Class C. This is one serial Metal capability, structural-route, native exact-kernel, mixed-program
resource/lifecycle, public Engine, test, evidence, and documentation change. It changes no public
API or Model semantics.

## Goal

Complete the current Metal structural MPSGraph recipes for the remaining elementwise arithmetic
wires while admitting production capability only where the complete Model result set is proved:

- `38 TENSOR_POW`;
- `46 SCALAR_ADD`, `47 SCALAR_SUB`, `48 SCALAR_MUL`, `49 SCALAR_DIV`, and `50 SCALAR_POW`;
- `52 RECIPROCAL`, `53 LOG`, `54 LOG1P`, `56 EXPM1`, `57 ERF`, `58 SQRT`, `59 RSQRT`;
- `60 FLOOR`, `61 CEIL`, `62 SIGN`, `63 RELU`; and
- `65 TANH`, `66 GELU`, `67 GELU_TANH_APPROXIMATION`, and `68 SILU`.

Wires `55 EXP` and `64 SIGMOID` remain owned by blocked Task 0053 and are untouched. Selector
presence, executable creation, or bounded device output is structural evidence only and never
`DOMAIN-PASS`.

## Frozen capability gates

### Gate A: exact raw discrete unary

Implement `FLOAT32 FLOOR`, `CEIL`, `SIGN`, and `RELU` through fixed raw-`uint32` custom kernels.
After the independent all-word proof/checker and native/public-path checks pass, admit the same
canonical positive-rank domain under both numerical profiles:

- one `FLOAT32` input and one same-Shape `FLOAT32` output;
- resolved dense-contiguous, zero-offset, non-view, fully static rank `1..16` descriptors;
- equal input/output `requiresGrad`; and
- checked element, byte, dispatch, and resource geometry.

`FLOOR` and `CEIL` preserve signed zero and signed infinity, map NaN to NaN, and return the exact
integral-valued binary32 result. `SIGN` preserves signed zero, maps every nonzero finite or infinite
value to same-signed one, and maps NaN to NaN. `RELU` is `max(x,+0)`: every negative value and
either zero becomes positive zero, positive values pass through bit-for-bit, and NaN remains NaN.
NaN payload/sign/signaling preservation may be used but is not promised beyond the Model NaN class.

The proof artifact must cover all `2^32` binary32 input words, not a sampled corpus. It must use an
independent reference/class partition, prove that the partitions cover the entire word space without
overlap or omission, check every transition boundary, and record reproducible source, command, and
digest. Production code and checker must not share one unchecked transform implementation.

### Gate B: scalar basic arithmetic

Assess `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, and `SCALAR_DIV` only under `ACCELERATOR
FLOAT32`. Promotion is allowed only if the implementation is a literal recursive member of the
already-approved tensor `ADD/SUB/MUL/DIV` result set:

- materialize the exact scalar attribute bits into a declared full-Shape internal tensor with a
  fixed raw integer splat kernel;
- invoke the corresponding already-approved ordered two-placeholder tensor primitive once, with the
  input on the left and scalar tensor on the right;
- preserve same Shape, canonical layout, rank `1..16`, exact `FLOAT32 ScalarValueAttrs`, and equal
  input/output `requiresGrad`; and
- reuse Task 0015's one-site ACCELERATOR DAZ/FTZ/rounding set without adding a final tolerance,
  algebraic rewrite, constant folding, scalar MPSGraph selector assumption, fallback, or special
  dispatch.

The historical Task-0011 constant-tensor selector path remains blocker evidence: especially
`maxFinite/maxFinite -> 0` and `+infinity/maxFinite -> NaN` are outside the approved set. A route
that cannot demonstrate the same two-runtime-tensor primitive semantics stays capability false.
`STRICT_IEEE` scalar arithmetic stays false because the binary primitives are not strict-approved.

### Gate C: reciprocal

Assess `RECIPROCAL` only under `ACCELERATOR FLOAT32`. Promotion requires a declared exact
full-Shape `+1.0f` raw splat followed by the approved ordered tensor `DIV(1,x)` primitive. The
result must therefore be a recursive member of the Model `1/x` set at the single division site,
including signed zero, infinity, NaN class, DAZ, FTZ, and zero-sign rules. Strict capability stays
false. If the approved primitive cannot be reused without a different site, hidden constant fold,
or result-set widening, `RECIPROCAL` remains false.

### Gate D: structural-only blockers

The following remain capability false in both profiles during Task 0058:

- `TENSOR_POW` and `SCALAR_POW`: irreducible power sites lack a complete-domain proof;
- `LOG`, `LOG1P`, `EXPM1`, `ERF`, `SQRT`, and `TANH`: the elementary/site result sets lack
  complete-domain MPSGraph or custom proof; `LOG1P` and `EXPM1` are first-class operations and
  cannot be justified by rounded `1+x`/`exp(x)-1` identities;
- `RSQRT`: transitively blocked on `SQRT` plus its required rounded division site;
- `GELU` and `GELU_TANH_APPROXIMATION`: every named constant, arithmetic, root, ERF/TANH site is
  recursively observable and lacks complete proof; and
- `SILU`: transitively blocked on Task-0053 `EXP` plus all guarded arithmetic sites.

No direct opaque selector, ordinary-value smoke, final-output tolerance, or successful graph create
may relax these blockers.

## Structural MPSGraph recipes

Implement and native-validate every catalog recipe in scope while keeping route structure separate
from capability:

- direct power for `TENSOR_POW` and exact-data scalar constant plus ordered power for `SCALAR_POW`;
- exact-data scalar constant plus ordered binary selector for scalar `ADD/SUB/MUL/DIV` structural
  candidates;
- direct selectors for `RECIPROCAL`, `LOG`, `ERF`, `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`,
  `RELU`, and `TANH`;
- `LOG1P = log(1+x)` and `EXPM1 = exp(x)-1` only as the catalog's structural compositions;
- the catalog formulas `0.5*x*(1+erf(x/sqrt(2)))` for `GELU`, the fixed tanh approximation for
  `GELU_TANH_APPROXIMATION`, and `x*sigmoid(x)` for `SILU`.

Blocked recipes are package-private/raw-ABI structural candidates only. Production Planning and
preparation reject them before native creation. Their tests force the MPSGraph route through the
bounded internal test seam, prove exact selector/operand/attribute/shape wiring and lifecycle, and
must not reinterpret outputs as correctness evidence.

## Custom catalog and mixed-program design

- Add one systematic Task-0058 exact custom family for `FLOOR/CEIL/SIGN/RELU`, exact scalar splat,
  and, only after Gates B/C pass, scalar arithmetic/reciprocal recipes.
- Retain explicit pending states for unsafe operations with closed reasons distinguishing unproved
  power, irreducible elementary, and recursive-site obligations. Do not group them under an
  implication that source absence alone is the blocker.
- Keep `CUSTOM_PROGRAM` as stable private route wire `3`. Newly true operations select it
  deterministically in production; eligible singleton direct MPSGraph candidates remain
  independently forceable only package-private.
- A scalar or reciprocal custom program declares the splat tensor as an internal value and resource,
  runs one exact raw splat step, then one cold nested MPSGraph tensor primitive. Every internal value
  participates in liveness; there is no hidden buffer, host materialization, per-node route switch,
  repartitioning, or retry.
- Mixed programs may combine Task-0052 comparisons/extrema/scans, Task-0057 BOOL/WHERE, the four new
  exact raw unary nodes, approved scalar/reciprocal nodes, and existing exact nested MPSGraph nodes
  under one preparation-fixed recipe.

## Schema, identity, ABI, and lifecycle

Retain node schema 14, route-bearing image bytes, operation wires `1..115`, type wires `1..6`,
attribute wires `0..41`, identity version 15, ABI 5, and exactly thirteen exports. The existing
registered node/attribute encodings already represent every Task-0058 operation. A version bump is
permitted only if implementation proves that encoded bytes genuinely must change; if so, perform one
clean cutover with no compatibility reader, alias, or dual decoder.

Preserve transactional finalization, complete resource declaration, context child leases,
reverse/attempt-all rollback, isolated per-run workspaces, direct assigned-output publication,
synchronous execution, reuse, concurrency, close-vs-run ordering, trace identity, authenticated
forced candidates, and exact no-fallback/no-retry behavior. No timing, benchmarking, runtime route
selection, cache-policy change, or autotuning is authorized.

## Implementation order

1. Freeze this plan and the reviewer inventory before production edits.
2. Implement the independent raw-word formulas and all-word checker for Gate A; land catalog source
   ownership and explicit blocker reasons.
3. Add Java capability/lowering/preflight and native custom execution for Gate A, including direct
   MPSGraph structural candidates.
4. Implement the exact declared splat resource and nested primitive route; adjudicate Gates B and C
   against Task 0015 and Task 0011 evidence. Promote only gates whose recursive membership passes.
5. Implement every remaining direct/composed MPSGraph structural recipe without capability widening.
6. Complete mixed-program liveness, native malformed controls, candidate forcing, lifecycle,
   publication, trace, and public Engine coverage.
7. Build/sign/package the local dylib, run focused verification, update contracts/plans, and obtain
   independent cumulative Class C review. Remediate every P0/P1/P2 finding before completion.

## Required verification

- Independent all-`2^32` checker for each exact raw unary, partition-coverage proof, boundary corpus,
  source/result digests, and agreement with the production Metal/C raw transform.
- Native build and exact export audit; ABI remains 5 with thirteen exports and manifest schema 14.
- Focused Metal capability, catalog-count, schema/malformed-image, native exact-output, forced-route,
  preparation/finalization, resource/liveness, publication, and no-fallback tests.
- Exact corpus includes both zeros; minimum/largest subnormals; adjacent values around `-1`, `0`,
  `+1`, and every exponent/integer transition; minimum normals; largest exactly integral and first
  fractional-transition magnitudes; finite extrema; infinities; and signed quiet/signaling NaNs with
  multiple payloads.
- For every promoted scalar/reciprocal operation: exact scalar/splat raw bits, ordered operands,
  Task-0011 gross-error controls, Task-0015 DAZ/FTZ membership, rank `1` and `16`, gradients,
  repeated/fan-out/mixed use, and strict/profile rejection.
- For every blocked operation: catalog recipe and raw forced MPSGraph create/run/close coverage on
  representative safe ordinary inputs, plus public capability and forced production-preparation
  rejection before native creation. Output observations are non-authoritative structural evidence.
- Public no-skip `Engine` smoke against the fresh dylib for every newly true operation, sole Metal
  ownership, mixed `CUSTOM_PROGRAM` raw/splat/nested-MPSGraph execution, exact outputs, direct
  publication, repeated and concurrent independent sessions, caller-input preservation, close/run,
  and recovery after pre-invocation rejection.
- Backend conformance matrix with exact final true/false counts, both profiles, rank-zero/rank-17,
  dtype/layout/attrs/Shape/gradient exclusions, and explicit unchanged false rows for wires `38`,
  `50`, `53..59`, and `65..68` except any Gate-C `52` promotion.
- Focused Javadoc, architecture, package, documentation, and `git diff --check`. Do not run timing,
  benchmarks, or a final full repository build.

## Dependencies and conflicts

Depends on Complete Tasks 0015, 0052, 0055, 0056, and 0057; Task-0011 blocker evidence; current
Model numerical-profile/result-set contracts; current Compiler forward and first-order capture; and
current Prepare/Runtime/Engine lifecycle. Task 0053 remains Blocked and retains sole ownership of
`EXP`/`SIGMOID` proof/source scope.

Conflicts with every concurrent Metal capability, schema, native preflight, selector lowering,
custom source, catalog/reason, candidate/codec/route, preparation/finalization, resource/liveness,
publication/trace, test/package, and shared Metal documentation edit, and with any resumed Task-0053
production/proof integration. Parallel group: none. One implementation owner integrates all shared
files serially.

## Architecture and documentation

Update the Metal backend contract/guide, native README, affected package/type Javadocs, master plan,
roadmap, route-evidence/catalog records, and this task's reproducible proof evidence. No ADR is
required unless implementation changes the frozen schema/identity/ABI or public architecture.

## Completion gate

Complete only after every scoped structural recipe exists, every unsupported row remains fail-closed,
all admitted operations have complete result-set proof and production/public execution, the exact
raw checker and lifecycle/resource obligations pass, the local package is verified, documentation is
current, commits are clean, and an independent cumulative Class C review approves with zero
P0/P1/P2 findings. A failed Gate B or C is a recorded exact blocker, not permission to narrow input
values or weaken Model semantics.

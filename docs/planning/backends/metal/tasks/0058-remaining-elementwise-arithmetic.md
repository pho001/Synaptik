# Task 0058: Remaining Elementwise Arithmetic

## Status

Complete

Planned from clean base `118aedca` at `6181299a`, reviewed at `c6145b09`, implemented at
`13a2e540`, and documented at `ac261e88`.

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

### Gate B: scalar basic arithmetic — blocked after plan review

`SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, and `SCALAR_DIV` remain capability false in both
profiles during Task 0058. The intended ACCELERATOR composition fails a public primitive
precondition: existing tensor `ADD/SUB/MUL/DIV` capability requires left input, right input, and
output `requiresGrad` flags to be equal, while a semantic scalar constant is non-grad and scalar
operations must preserve either source gradient flag. Marking a constant grad-eligible, silently
bypassing the public primitive capability, or narrowing scalar support to no-grad occurrences would
not prove the complete scalar occurrence domain.

The historical Task-0011 constant-tensor selector path remains independent blocker evidence:
`maxFinite/maxFinite -> 0` and `+infinity/maxFinite -> NaN` are outside the approved set. No
production splat, nested primitive, or custom scalar arithmetic route is added. The structural
MPSGraph recipe still binds the distinct scalar wire, exact `ScalarValueAttrs` type/raw bits,
ordered input/scalar operands, and output Shape in the schema image and fingerprint; this prevents
scalar/tensor identity collision or hidden operand folding without claiming correctness.

### Gate C: reciprocal — blocked after plan review

`RECIPROCAL` remains capability false in both profiles during Task 0058. A semantic `DIV(1,x)`
composition has the same gradient mismatch: the exact `+1.0f` constant is non-grad, while existing
tensor `DIV` capability requires both operands and output to have equal `requiresGrad`. The direct
selector also retains Task-0006 special-class failures. No legal complete-domain recursive route is
therefore available, and production may not bypass primitive capability or narrow the occurrence
domain.

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
preparation reject them before native creation. Their explicit nonproduction raw-image fixture
bypasses capability only inside tests, forces the MPSGraph route, and proves exact
selector/operand/attribute/Shape wiring and lifecycle. Output observations are never correctness
evidence. Image/digest tests distinguish every scalar wire and raw attribute bit pattern from each
other and from the corresponding tensor primitive.

## Custom catalog and mixed-program design

- Add one systematic Task-0058 exact custom family only for `FLOOR/CEIL/SIGN/RELU`.
- Retain explicit pending states for unsafe operations with closed reasons distinguishing scalar
  gradient/primitive mismatch, unproved power, irreducible elementary, and recursive-site
  obligations. Do not group them under an implication that source absence alone is the blocker.
- Keep `CUSTOM_PROGRAM` as stable private route wire `3`. The four newly true operations select it
  deterministically in production; eligible singleton direct MPSGraph candidates remain
  independently forceable only package-private.
- Mixed programs may combine Task-0052 comparisons/extrema/scans, Task-0057 BOOL/WHERE, the four new
  exact raw unary nodes, and existing exact nested MPSGraph nodes under one preparation-fixed
  recipe. There is no scalar/reciprocal splat or hidden internal primitive.

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
4. Record Gates B/C fail-closed from the reviewed gradient/public-primitive and historical numerical
   evidence; implement their scalar/direct structural recipes only.
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
- For blocked scalar/reciprocal rows: exact scalar type/raw-bit/operand-order image and fingerprint
  drift tests, Task-0011/0006 blocker controls, capability rejection for both gradient flags and
  profiles, and production-preparation rejection before native creation.
- For every blocked operation: catalog recipe and explicit nonproduction raw-image forced MPSGraph
  create/run/close coverage on representative safe ordinary inputs. The fixture is inaccessible to
  production and output observations are non-authoritative structural evidence.
- Public no-skip `Engine` smoke against the fresh dylib for all four newly true operations, sole
  Metal ownership, mixed `CUSTOM_PROGRAM` raw/nested-MPSGraph execution, exact outputs, direct
  publication, repeated and concurrent independent sessions, caller-input preservation, close/run,
  and recovery after pre-invocation rejection.
- Backend conformance matrix with exactly `45 true / 70 false = 115`, both profiles,
  rank-zero/rank-17, dtype/layout/attrs/Shape/gradient exclusions, and explicit unchanged false
  rows for wires `38`, `46..50`, `52..59`, and `65..68`.
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
P0/P1/P2 findings. Gates B and C are recorded exact blockers; they do not permit occurrence
narrowing, synthetic gradient metadata, or a hidden primitive-capability bypass.

## Implementation evidence

Implementation `13a2e540` lands all 21 scoped structural recipes and the four exact raw production
operations. Schema 14, identity 15, ABI 5, route wire 3, and the thirteen exports are unchanged.
The structural registry now has `62 executable / 53 nonexecutable`; the production capability
ledger is exactly `45 true / 70 false = 115`.

The catalog records exact custom availability only for `FLOOR`, `CEIL`, `SIGN`, and `RELU`, for
new totals `27 AVAILABLE / 88 PENDING / 0 UNAVAILABLE_WITH_PROOF`. Scalar `ADD/SUB/MUL/DIV` and
`RECIPROCAL` retain explicit gradient/public-primitive blockers; tensor/scalar power retains an
unproved-power blocker; `LOG`, `LOG1P`, `EXPM1`, `ERF`, `SQRT`, `RSQRT`, and `TANH` retain
irreducible elementary blockers; and `GELU`, `GELU_TANH_APPROXIMATION`, and `SILU` retain recursive
site blockers. `EXP` and `SIGMOID` remain nonexecutable and Task-0053-owned.

[`evidence/0058/proof.txt`](../evidence/0058/proof.txt) records the reproducible all-word checker
source and production-source SHA-256 digests, compiler identity, exact command, partition counts,
boundary count, output digests, and the result:

```text
words=4294967296 failures=0 partition_failures=0 covered=4294967296
partitions nan=16777214 infinity=2 zero=2 below_one=2130706430 integral=1778384894 fractional=369098754
digest=be418f313ebe8f30 boundary_words=2590 boundary_digest=8e08446771842252
```

Focused Metal capability/catalog/schema/malformed-image/route/fingerprint/native-output tests,
backend conformance, and the no-skip public Engine integration test passed against the fresh
dylib. The Engine case executed all four operations under both profiles through sole Metal
ownership and `CUSTOM_KERNEL` trace identity, with exact outputs, direct publication, repeated and
concurrent independent sessions, caller-input preservation, recovery after pre-invocation
rejection, and close rejection. Mixed raw custom and nested MPSGraph execution and independently
forced direct MPSGraph candidates passed. Structural raw fixtures created, ran, and closed every
blocked recipe without granting production ownership.

The focused Metal Javadoc generation, final native build, fixed ad-hoc signature, local package,
native verifier, and Gradle package verifier passed. The packaged Mach-O retains ABI 5, schema 14
manifest metadata, the fixed identifier, and exactly thirteen audited exports. No timing,
benchmark, autotuning, fallback, retry, or final full repository build was run.

The independent cumulative Class C rereview approved the clean checkpoint with zero P0/P1/P2
findings. It verified the exhaustive Gate-A proof, public/lifecycle/custom route, fail-closed
Gates B/C/D and Task-0053 isolation, `45/70` capability, `62/53` structural registry,
schema 14, identity 15, ABI 5, route wire 3, and thirteen exports.

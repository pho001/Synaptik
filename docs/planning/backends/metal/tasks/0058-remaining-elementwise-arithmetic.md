# Task 0058: Remaining Elementwise Arithmetic

## Status

Active

The original bounded implementation completed through `5760901b`. A follow-on authorization
reopens Task 0058 for the exact ACCELERATOR no-gradient scalar/reciprocal subset below.

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

### Original Gate B: scalar basic arithmetic — blocked before the extension

`SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, and `SCALAR_DIV` remained capability false in both
profiles during the original bounded implementation because its scope required the complete
gradient-bearing scalar domain while the exact semantic constant was non-gradient. Marking the
constant grad-eligible or bypassing the public primitive was prohibited. The historical Task-0011
constant-tensor selector evidence also showed results outside the approved set.

The later authorization explicitly narrows production to no-gradient input and output occurrences,
where the no-gradient constant satisfies the primitive precondition. It supersedes this blocker
only for FLOAT32 ACCELERATOR scalar `ADD/SUB/MUL/DIV`. The schema image and workload fingerprint
still bind the distinct scalar wire, exact `ScalarValueAttrs` type/raw bits, ordered input/scalar
operands, and Shape; no tensor identity collision or hidden folding is permitted.

### Original Gate C: reciprocal — blocked before the extension

`RECIPROCAL` remained capability false in both profiles during the original bounded implementation
because the exact no-gradient `+1.0f` numerator could not satisfy the then-required complete
gradient-bearing domain, while the direct selector retained Task-0006 special-class failures. The
later authorization supersedes this blocker only for no-gradient FLOAT32 ACCELERATOR occurrences
and requires the separately approved exact `+1.0f / input` division composition; the direct
selector remains unauthorized.

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

## Authorized no-gradient scalar/reciprocal extension

Admit wires `46 SCALAR_ADD`, `47 SCALAR_SUB`, `48 SCALAR_MUL`, `49 SCALAR_DIV`, and
`52 RECIPROCAL` only for canonical positive-rank FLOAT32 ACCELERATOR occurrences whose input and
output both have `requiresGrad=false`. Wire `50 SCALAR_POW` and every other blocker remain false.
Materialize each semantic scalar directly as one exact four-byte raw FLOAT32 MPSGraph constant
with Shape `[1]`; for reciprocal the raw word is `+1.0f`. The constant is no-gradient and the
already-approved binary MPSGraph primitive broadcasts it against the input, with no Java/native
full-Shape expansion, repeated-byte allocation, synthetic gradient metadata, hidden primitive
bypass, or occurrence reinterpretation.

Each lowering has exactly one approved tensor primitive site and no constant folding:
`input op scalar` for scalar ADD/SUB/MUL/DIV, preserving source operand order, and `1/input` through
tensor DIV for RECIPROCAL. The schema/program and workload identity must bind source wire, exact
`ScalarValueAttrs` type/raw bits where present, rank-one `[1]` constant Shape, operand order,
four-byte payload, input Shape, and lowered primitive opcode. Retain schema 14 and identity 15
because existing encoded schema/candidate bytes do not change; extend the existing version-15
workload digest with these derived lowering facts.

Prove special scalar metadata for both zeros, signed subnormals, infinities, and NaNs; subtraction
and division operand order; division by both zero signs; stale-identity rejection; public
Metal-only Engine execution; and rejection of every `requiresGrad=true` occurrence before native
creation. Update the capability ledger from `45/70` to exactly `50 true / 65 false`, preserve
`62 executable / 53 nonexecutable` structural coverage, run focused verification without timing or
a final full build, and obtain a fresh independent cumulative Class C approval before returning
this task to Complete.

### Extension implementation checkpoint

Implementation `5ab9c44c` admits exactly wires `46..49` and `52` for ACCELERATOR canonical
positive-rank FLOAT32 no-gradient input/output occurrences. Wire `50` and every other prior blocker
remain false. The capability ledger is `50 true / 65 false`; structural coverage remains
`62 executable / 53 nonexecutable`. The MPSGraph catalog is
`75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; custom state remains
`27 AVAILABLE / 88 PENDING / 0 UNAVAILABLE_WITH_PROOF`.

Native lowering creates each scalar directly from its exact four raw bytes at Shape `[1]` and lets
the one approved binary MPSGraph primitive broadcast it. Scalar arithmetic retains
`input op scalar`; reciprocal uses exact `+1.0f` primary and input secondary for one division.
Java and native preflight independently reject gradient-bearing values, rank-zero values, and a
non-FLOAT32 scalar attribute. Schema 14, identity 15, ABI 5, route wire 3, and thirteen exports are
unchanged. The existing version-15 workload digest now also binds the source wire, primitive wire,
operand order, FLOAT32 type, rank-one extent, four-byte payload, exact scalar bits, and logical
input Shape; encoded schema and candidate-decision bytes did not change.

Focused capability/catalog/schema/route/fingerprint/native-malformed tests, backend conformance,
and the public Metal-only Engine scenario passed against the rebuilt native library. The public
scenario exercises both scalar zero signs, signed subnormals, infinities, quiet/signaling NaNs,
SUB/DIV operand order, both zero denominators, reciprocal, repeated execution, input preservation,
sole Metal ownership, and pre-native gradient rejection. Native build, fixed ad-hoc signing, and
local package publication passed. No timing, benchmark, autotuning, fallback, retry, or final full
repository build ran. Final package/export/Javadoc/documentation checks and the independent
cumulative Class C rereview remain before completion.

# Task 0058: Remaining [Elementwise](../../../../glossary.md#term-elementwise) Arithmetic

## Status

Complete

The original bounded implementation completed through `5760901b`. The authorized [ACCELERATOR](../../../../glossary.md#term-accelerator)
no-[gradient](../../../../glossary.md#term-gradient) [scalar](../../../../glossary.md#term-scalar)/reciprocal extension completed at implementation `5ab9c44c`, contracts/evidence
`63c070cc`, and independent cumulative Class C approval with zero P0/P1/P2.

## Change class

Class C. This is one serial [Metal](../../../../glossary.md#term-metal) capability, structural-route, native exact-[kernel](../../../../glossary.md#kernel), mixed-program
resource/[lifecycle](../../../../glossary.md#lifecycle), public [Engine](../../../../glossary.md#term-engine), test, evidence, and documentation change. It changes no public
[API](../../../../glossary.md#term-application-programming-interface-api) or [Model](../../../../glossary.md#term-model) semantics.

## Goal

Complete the current [Metal](../../../../glossary.md#term-metal) structural [MPSGraph](../../../../glossary.md#term-mpsgraph) recipes for the remaining [elementwise](../../../../glossary.md#term-elementwise) arithmetic
wires while admitting production capability only where the complete [Model](../../../../glossary.md#term-model) result set is proved:

- `38 TENSOR_POW`;
- `46 SCALAR_ADD`, `47 SCALAR_SUB`, `48 SCALAR_MUL`, `49 SCALAR_DIV`, and `50 SCALAR_POW`;
- `52 RECIPROCAL`, `53 LOG`, `54 LOG1P`, `56 EXPM1`, `57 ERF`, `58 SQRT`, `59 RSQRT`;
- `60 FLOOR`, `61 CEIL`, `62 SIGN`, `63 RELU`; and
- `65 TANH`, `66 GELU`, `67 GELU_TANH_APPROXIMATION`, and `68 SILU`.

Wires `55 EXP` and `64 SIGMOID` remain owned by blocked Task 0053 and are untouched. Selector
presence, executable creation, or bounded [device](../../../../glossary.md#term-device) output is structural evidence only and never
`DOMAIN-PASS`.

## Frozen capability gates

### Gate A: exact raw discrete [unary](../../../../glossary.md#term-unary)

Implement `FLOAT32 FLOOR`, `CEIL`, `SIGN`, and `RELU` through fixed raw-[`uint32`](../../../../glossary.md#index-mappings-and-cost-dominance) custom [kernels](../../../../glossary.md#kernel).
After the independent all-word proof/checker and native/public-path checks pass, admit the same
[canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) domain under both numerical profiles:

- one [`FLOAT32`](../../../../glossary.md#term-float32) input and one same-[Shape](../../../../glossary.md#shape) `FLOAT32` output;
- resolved dense-[contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings), fully static rank `1..16` [descriptors](../../../../glossary.md#term-descriptor);
- equal input/output `requiresGrad`; and
- checked [element](../../../../glossary.md#term-element), byte, [dispatch](../../../../glossary.md#term-dispatch), and resource geometry.

`FLOOR` and `CEIL` preserve [signed zero](../../../../glossary.md#term-signed-zero) and signed [infinity](../../../../glossary.md#term-infinity), map [NaN](../../../../glossary.md#term-nan) to NaN, and return the exact
integral-valued [binary32](../../../../glossary.md#floating-point-precision-and-rounding) result. `SIGN` preserves signed zero, maps every nonzero [finite](../../../../glossary.md#term-finite) or infinite
value to same-signed one, and maps NaN to NaN. `RELU` is `max(x,+0)`: every negative value and
either zero becomes [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), positive values pass through bit-for-bit, and NaN remains NaN.
[NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign/signaling preservation may be used but is not promised beyond the [Model](../../../../glossary.md#term-model) NaN class.

The proof [artifact](../../../../glossary.md#term-artifact) must cover all `2^32` binary32 input words, not a sampled corpus. It must use an
independent reference/class [partition](../../../../glossary.md#partition), prove that the partitions cover the entire word space without
overlap or omission, check every transition boundary, and record reproducible source, command, and
[digest](../../../../glossary.md#term-digest). Production code and checker must not share one unchecked transform implementation.

### Original Gate B: [scalar](../../../../glossary.md#term-scalar) basic arithmetic — blocked before the extension

`SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, and `SCALAR_DIV` remained capability false in both
profiles during the original bounded implementation because its [scope](../../../../glossary.md#term-scope) required the complete
[gradient](../../../../glossary.md#term-gradient)-bearing scalar domain while the exact semantic [constant](../../../../glossary.md#term-constant) was non-gradient. Marking the
constant grad-eligible or bypassing the public primitive was prohibited. The historical Task-0011
constant-[tensor](../../../../glossary.md#tensor) selector evidence also showed results outside the approved set.

The later authorization explicitly narrows production to no-gradient input and output occurrences,
where the no-gradient constant satisfies the primitive precondition. It supersedes this blocker
only for [FLOAT32](../../../../glossary.md#term-float32) [ACCELERATOR](../../../../glossary.md#term-accelerator) scalar `ADD/SUB/MUL/DIV`. The [schema](../../../../glossary.md#term-schema) image and workload [fingerprint](../../../../glossary.md#term-fingerprint)
still bind the distinct scalar wire, exact `ScalarValueAttrs` type/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), ordered input/scalar
operands, and [Shape](../../../../glossary.md#shape); no tensor [identity](../../../../glossary.md#term-identity-meanings) collision or hidden folding is permitted.

### Original Gate C: reciprocal — blocked before the extension

`RECIPROCAL` remained capability false in both profiles during the original bounded implementation
because the exact no-[gradient](../../../../glossary.md#term-gradient) `+1.0f` numerator could not satisfy the then-required complete
gradient-bearing domain, while the direct selector retained Task-0006 special-class failures. The
later authorization supersedes this blocker only for no-gradient [FLOAT32](../../../../glossary.md#term-float32) [ACCELERATOR](../../../../glossary.md#term-accelerator) occurrences
and requires the separately approved exact `+1.0f / input` division composition; the direct
selector remains unauthorized.

### Gate D: structural-only blockers

The following remain capability false in both profiles during Task 0058:

- `TENSOR_POW` and `SCALAR_POW`: irreducible power sites lack a complete-domain proof;
- [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, and [`TANH`](../../../../glossary.md#term-tanh): the elementary/site result sets lack
  complete-domain MPSGraph or custom proof; <span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>`LOG1P` and `EXPM1` are first-class operations and
  cannot be justified by rounded `1+x`/`exp(x)-1` [identities](../../../../glossary.md#term-identity-meanings);
- `RSQRT`: transitively blocked on `SQRT` plus its required rounded division site;
- [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) and `GELU_TANH_APPROXIMATION`: every named constant, arithmetic, root, ERF/TANH site is
  recursively observable and lacks complete proof; and<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `SILU`: transitively blocked on Task-0053 [`EXP`](../../../../glossary.md#term-expx) plus all guarded arithmetic sites.

No direct [opaque](../../../../glossary.md#term-opaque) selector, ordinary-value [smoke](../../../../glossary.md#term-smoke-test), final-output [tolerance](../../../../glossary.md#term-tolerance), or successful [graph](../../../../glossary.md#graph) create
may relax these blockers.

## Structural [MPSGraph](../../../../glossary.md#term-mpsgraph) recipes

Implement and native-validate every catalog recipe in [scope](../../../../glossary.md#term-scope) while keeping route structure separate
from capability:

- direct power for `TENSOR_POW` and exact-data [scalar](../../../../glossary.md#term-scalar) [constant](../../../../glossary.md#term-constant) plus ordered power for `SCALAR_POW`;
- exact-data scalar constant plus ordered [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) selector for scalar `ADD/SUB/MUL/DIV` structural
  candidates;
- direct selectors for `RECIPROCAL`, [`LOG`](../../../../glossary.md#term-log), [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, `FLOOR`, `CEIL`, `SIGN`,
  `RELU`, and [`TANH`](../../../../glossary.md#term-tanh);
- `LOG1P = log(1+x)` and `EXPM1 = exp(x)-1` only as the catalog's structural compositions;
- the catalog formulas `0.5*x*(1+erf(x/sqrt(2)))` for [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), the fixed tanh approximation for
  `GELU_TANH_APPROXIMATION`, and `x*sigmoid(x)` for `SILU`.

Blocked recipes are package-private/raw-[ABI](../../../../glossary.md#term-application-binary-interface-abi) structural candidates only. Production [Planning](../../../../glossary.md#term-planning) and
preparation reject them before native creation. Their explicit nonproduction raw-image [fixture](../../../../glossary.md#term-fixture)
bypasses capability only inside tests, forces the MPSGraph route, and proves exact
selector/operand/attribute/[Shape](../../../../glossary.md#shape) wiring and [lifecycle](../../../../glossary.md#lifecycle). Output observations are never correctness
evidence. Image/[digest](../../../../glossary.md#term-digest) tests distinguish every scalar wire and raw attribute bit pattern from each
other and from the corresponding [tensor](../../../../glossary.md#tensor) primitive.

## Custom catalog and mixed-program design

- Add one systematic Task-0058 exact custom family only for `FLOOR/CEIL/SIGN/RELU`.
- Retain explicit pending states for unsafe operations with closed reasons distinguishing scalar
  gradient/primitive mismatch, unproved power, irreducible elementary, and recursive-site
  obligations. Do not group them under an implication that source absence alone is the blocker.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Keep `CUSTOM_PROGRAM` as stable private route wire `3`. The four newly true operations select it
  deterministically in production; eligible singleton direct MPSGraph candidates remain
  independently forceable only package-private.<span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- Mixed programs may combine Task-0052 comparisons/extrema/scans, Task-0057 BOOL/WHERE, the four new
  exact raw unary nodes, and existing exact nested MPSGraph nodes under one preparation-fixed
  recipe. There is no scalar/reciprocal splat or hidden internal primitive.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BOOL glossary](../../../../glossary.md#data-type--datatype); [unary glossary](../../../../glossary.md#term-unary); [nodes glossary](../../../../glossary.md#node))</span>

## [Schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [ABI](../../../../glossary.md#term-application-binary-interface-abi), and [lifecycle](../../../../glossary.md#lifecycle)

Retain [node](../../../../glossary.md#node) schema 14, route-bearing image bytes, [operation](../../../../glossary.md#operation) wires `1..115`, type wires `1..6`,
attribute wires `0..41`, identity version 15, ABI 5, and exactly thirteen exports. The existing
registered node/attribute encodings already represent every Task-0058 operation. A version bump is
permitted only if implementation proves that encoded bytes genuinely must change; if so, perform one
clean cutover with no compatibility reader, [alias](../../../../glossary.md#aliasing-and-arena), or dual decoder.

Preserve transactional finalization, complete resource declaration, context child [leases](../../../../glossary.md#term-lease),
reverse/attempt-all [rollback](../../../../glossary.md#term-rollback), isolated per-[run](../../../../glossary.md#term-run) workspaces, direct assigned-output [publication](../../../../glossary.md#term-publication),
[synchronous](../../../../glossary.md#term-synchronous) execution, reuse, [concurrency](../../../../glossary.md#term-concurrency), close-vs-run ordering, [trace](../../../../glossary.md#trace) identity, authenticated
forced candidates, and exact no-fallback/no-retry behavior. No timing, [benchmarking](../../../../glossary.md#term-benchmarking), [runtime](../../../../glossary.md#term-runtime) route
selection, [cache](../../../../glossary.md#term-cache)-policy change, or [autotuning](../../../../glossary.md#autotuning--model-autotuning) is authorized.

## Implementation order

1. Freeze this plan and the reviewer inventory before production edits.
2. Implement the independent raw-word formulas and all-word checker for Gate A; land catalog source
   ownership and explicit blocker reasons.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
3. Add Java capability/lowering/preflight and native custom execution for Gate A, including direct
   MPSGraph structural candidates.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [preflight glossary](../../../../glossary.md#term-preflight))</span>
4. Record Gates B/C fail-closed from the reviewed gradient/public-primitive and historical numerical
   evidence; implement their scalar/direct structural recipes only.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [gradient glossary](../../../../glossary.md#term-gradient); [scalar glossary](../../../../glossary.md#term-scalar))</span>
5. Implement every remaining direct/composed [MPSGraph](../../../../glossary.md#term-mpsgraph) structural recipe without capability [widening](../../../../glossary.md#term-widening).
6. Complete mixed-program liveness, native malformed controls, candidate forcing, lifecycle,
   publication, trace, and public Engine coverage.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [lifecycle glossary](../../../../glossary.md#lifecycle); [publication glossary](../../../../glossary.md#term-publication); [trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine))</span>
7. Build/sign/package the local dylib, run focused verification, update contracts/plans, and obtain
   independent cumulative Class C review. Remediate every P0/P1/P2 finding before completion.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [run glossary](../../../../glossary.md#term-run))</span>

## Required verification

- Independent all-`2^32` checker for each exact raw unary, partition-coverage proof, boundary corpus,
  source/result digests, and agreement with the production Metal/C raw transform.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [partition glossary](../../../../glossary.md#partition); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Native build and exact export audit; [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains 5 with thirteen exports and [manifest](../../../../glossary.md#term-manifest) [schema](../../../../glossary.md#term-schema) 14.
- Focused Metal capability, catalog-count, schema/malformed-image, native exact-output, forced-route,
  preparation/finalization, resource/liveness, publication, and no-fallback tests.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [publication glossary](../../../../glossary.md#term-publication))</span>
- Exact corpus includes both zeros; minimum/largest [subnormals](../../../../glossary.md#term-subnormal); adjacent values around `-1`, `0`,
  `+1`, and every exponent/integer transition; minimum normals; largest exactly integral and first
  fractional-transition magnitudes; finite extrema; infinities; and signed quiet/signaling NaNs with
  multiple payloads.<span data-glossary-navigation="true"> ([exponent glossary](../../../../glossary.md#term-exponent); [finite glossary](../../../../glossary.md#term-finite); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- For blocked scalar/reciprocal rows: exact scalar type/raw-bit/operand-order image and fingerprint
  drift tests, Task-0011/0006 blocker controls, capability rejection for both gradient flags and
  profiles, and production-preparation rejection before native creation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- For every blocked operation: catalog recipe and explicit nonproduction raw-image forced MPSGraph
  create/run/close coverage on representative safe ordinary inputs. The fixture is inaccessible to
  production and output observations are non-authoritative structural evidence.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [run glossary](../../../../glossary.md#term-run); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Public no-skip [`Engine`](../../../../glossary.md#term-engine) smoke against the fresh dylib for all four newly true operations, sole
  Metal ownership, mixed <span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [dylib glossary](../../../../glossary.md#term-dylib); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>`CUSTOM_PROGRAM` raw/nested-MPSGraph execution, exact outputs, direct
  publication, repeated and concurrent independent sessions, caller-input preservation, close/run,
  and recovery after pre-invocation rejection.
- [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) [matrix](../../../../glossary.md#term-matrix-meanings) with exactly `45 true / 70 false = 115`, both profiles,
  rank-zero/rank-17, dtype/layout/attrs/Shape/gradient exclusions, and explicit unchanged false
  rows for wires <span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [rank glossary](../../../../glossary.md#term-rank); [dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [Shape glossary](../../../../glossary.md#shape))</span>`38`, `46..50`, `52..59`, and `65..68`.
- Focused [Javadoc](../../../../glossary.md#term-javadoc), architecture, package, documentation, and `git diff --check`. Do not run timing,
  benchmarks, or a final full repository build.

## Dependencies and conflicts

Depends on Complete Tasks 0015, 0052, 0055, 0056, and 0057; Task-0011 blocker evidence; current
[Model](../../../../glossary.md#term-model) numerical-profile/result-set contracts; current [Compiler](../../../../glossary.md#term-compiler) forward and first-order [capture](../../../../glossary.md#term-capture-meanings); and
current [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle). Task 0053 remains Blocked and retains sole [ownership](../../../../glossary.md#term-ownership-meanings) of
[`EXP`](../../../../glossary.md#term-expx)/[`SIGMOID`](../../../../glossary.md#term-sigmoid) proof/source [scope](../../../../glossary.md#term-scope).

Conflicts with every concurrent [Metal](../../../../glossary.md#term-metal) capability, [schema](../../../../glossary.md#term-schema), native [preflight](../../../../glossary.md#term-preflight), selector [lowering](../../../../glossary.md#lowering),
custom source, catalog/reason, candidate/[codec](../../../../glossary.md#term-codec)/route, preparation/finalization, resource/[liveness](../../../../glossary.md#term-liveness),
[publication](../../../../glossary.md#term-publication)/[trace](../../../../glossary.md#trace), test/package, and shared Metal documentation edit, and with any resumed Task-0053
production/proof integration. Parallel group: none. One implementation [owner](../../../../glossary.md#term-owner) integrates all shared
files serially.

## Architecture and documentation

Update the [Metal](../../../../glossary.md#term-metal) [backend](../../../../glossary.md#backend) contract/guide, native README, affected package/type Javadocs, master plan,
roadmap, route-evidence/catalog records, and this task's reproducible proof evidence. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr) is
required unless implementation changes the frozen [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings)/[ABI](../../../../glossary.md#term-application-binary-interface-abi) or public architecture.

## Completion gate

Complete only after every scoped structural recipe exists, every unsupported row remains [fail-closed](../../../../glossary.md#term-fail-closed),
all admitted operations have complete result-set proof and production/public execution, the exact
raw checker and [lifecycle](../../../../glossary.md#lifecycle)/resource obligations pass, the local package is verified, documentation is
current, [commits](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are clean, and an independent cumulative Class C review approves with zero
P0/P1/P2 findings. Gates B and C are recorded exact blockers; they do not permit [occurrence](../../../../glossary.md#term-occurrence)
[narrowing](../../../../glossary.md#term-narrowing), synthetic [gradient](../../../../glossary.md#term-gradient) [metadata](../../../../glossary.md#term-metadata), or a hidden primitive-capability bypass.

## Implementation evidence

Implementation `13a2e540` lands all 21 scoped structural recipes and the four exact raw production
operations. [Schema](../../../../glossary.md#term-schema) 14, [identity](../../../../glossary.md#term-identity-meanings) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, route wire 3, and the thirteen exports are unchanged.
The structural [registry](../../../../glossary.md#term-registry) now has `62 executable / 53 nonexecutable`; the production [capability
ledger](../../../../glossary.md#capability-ledger) is exactly `45 true / 70 false = 115`.

The catalog records exact custom availability only for `FLOOR`, `CEIL`, `SIGN`, and `RELU`, for
new totals `27 AVAILABLE / 88 PENDING / 0 UNAVAILABLE_WITH_PROOF`. [Scalar](../../../../glossary.md#term-scalar) `ADD/SUB/MUL/DIV` and
`RECIPROCAL` retain explicit [gradient](../../../../glossary.md#term-gradient)/public-primitive blockers; [tensor](../../../../glossary.md#tensor)/scalar power retains an
unproved-power blocker; [`LOG`](../../../../glossary.md#term-log), `LOG1P`, `EXPM1`, [`ERF`](../../../../glossary.md#term-erf), `SQRT`, `RSQRT`, and [`TANH`](../../../../glossary.md#term-tanh) retain
irreducible elementary blockers; and [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu), `GELU_TANH_APPROXIMATION`, and `SILU` retain recursive
site blockers. [`EXP`](../../../../glossary.md#term-expx) and [`SIGMOID`](../../../../glossary.md#term-sigmoid) remain nonexecutable and Task-0053-owned.

[`evidence/0058/proof.txt`](../evidence/0058/proof.txt) records the reproducible all-word checker
source and production-source [SHA-256](../../../../glossary.md#term-sha-256) digests, [compiler](../../../../glossary.md#term-compiler) identity, exact command, [partition](../../../../glossary.md#partition) counts,
boundary count, output digests, and the result:

```text
words=4294967296 failures=0 partition_failures=0 covered=4294967296
partitions nan=16777214 infinity=2 zero=2 below_one=2130706430 integral=1778384894 fractional=369098754
digest=be418f313ebe8f30 boundary_words=2590 boundary_digest=8e08446771842252
```

Focused [Metal](../../../../glossary.md#term-metal) capability/catalog/schema/malformed-image/route/[fingerprint](../../../../glossary.md#term-fingerprint)/native-output tests,
[backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and the no-skip public [Engine](../../../../glossary.md#term-engine) [integration test](../../../../glossary.md#term-integration-test) passed against the fresh
[dylib](../../../../glossary.md#term-dylib). The Engine case executed all four operations under both profiles through sole Metal
[ownership](../../../../glossary.md#term-ownership-meanings) and `CUSTOM_KERNEL` [trace](../../../../glossary.md#trace) identity, with exact outputs, direct [publication](../../../../glossary.md#term-publication), repeated and
concurrent independent sessions, caller-input preservation, recovery after pre-invocation
rejection, and close rejection. Mixed raw custom and nested [MPSGraph](../../../../glossary.md#term-mpsgraph) execution and independently
forced direct MPSGraph candidates passed. Structural raw fixtures created, ran, and closed every
blocked recipe without granting production ownership.

The focused Metal [Javadoc](../../../../glossary.md#term-javadoc) generation, final native build, fixed ad-hoc [signature](../../../../glossary.md#term-signature), local package,
native verifier, and [Gradle](../../../../glossary.md#term-gradle) package verifier passed. The packaged [Mach-O](../../../../glossary.md#term-mach-o) retains ABI 5, schema 14
[manifest](../../../../glossary.md#term-manifest) [metadata](../../../../glossary.md#term-metadata), the fixed identifier, and exactly thirteen audited exports. No timing,
benchmark, [autotuning](../../../../glossary.md#autotuning--model-autotuning), fallback, retry, or final full repository build was [run](../../../../glossary.md#term-run).

The independent cumulative Class C rereview approved the clean [checkpoint](../../../../glossary.md#term-checkpoint) with zero P0/P1/P2
findings. It verified the exhaustive Gate-A proof, public/[lifecycle](../../../../glossary.md#lifecycle)/custom route, [fail-closed](../../../../glossary.md#term-fail-closed)
Gates B/C/D and Task-0053 isolation, `45/70` capability, `62/53` structural registry,
schema 14, identity 15, ABI 5, route wire 3, and thirteen exports.

## Authorized no-[gradient](../../../../glossary.md#term-gradient) [scalar](../../../../glossary.md#term-scalar)/reciprocal extension

Admit wires `46 SCALAR_ADD`, `47 SCALAR_SUB`, `48 SCALAR_MUL`, `49 SCALAR_DIV`, and
`52 RECIPROCAL` only for [canonical](../../../../glossary.md#term-canonical-form) positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32) [ACCELERATOR](../../../../glossary.md#term-accelerator) occurrences whose input and
output both have `requiresGrad=false`. Wire `50 SCALAR_POW` and every other blocker remain false.
Materialize each semantic scalar directly as one exact four-byte raw FLOAT32 [MPSGraph](../../../../glossary.md#term-mpsgraph) [constant](../../../../glossary.md#term-constant)
with [Shape](../../../../glossary.md#shape) `[1]`; for reciprocal the raw word is `+1.0f`. The constant is no-gradient and the
already-approved [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) MPSGraph primitive broadcasts it against the input, with no Java/native
full-Shape expansion, repeated-byte allocation, synthetic gradient [metadata](../../../../glossary.md#term-metadata), hidden primitive
bypass, or [occurrence](../../../../glossary.md#term-occurrence) reinterpretation.

Each [lowering](../../../../glossary.md#lowering) has exactly one approved [tensor](../../../../glossary.md#tensor) [primitive site](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor) and no [constant folding](../../../../glossary.md#term-constant-folding):
`input op scalar` for scalar ADD/SUB/MUL/DIV, preserving source operand order, and `1/input` through
tensor DIV for RECIPROCAL. The [schema](../../../../glossary.md#term-schema)/program and workload [identity](../../../../glossary.md#term-identity-meanings) must bind source wire, exact
`ScalarValueAttrs` type/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) where present, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[1]` constant Shape, operand order,
four-byte [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), input Shape, and lowered primitive [opcode](../../../../glossary.md#term-opcode). Retain schema 14 and identity 15
because existing encoded schema/candidate bytes do not change; extend the existing version-15
workload [digest](../../../../glossary.md#term-digest) with these derived lowering facts.

Prove special scalar metadata for both zeros, signed [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and NaNs; subtraction
and division operand order; division by both zero signs; stale-identity rejection; public
[Metal](../../../../glossary.md#term-metal)-only [Engine](../../../../glossary.md#term-engine) execution; and rejection of every `requiresGrad=true` occurrence before native
creation. Update the [capability ledger](../../../../glossary.md#capability-ledger) from `45/70` to exactly `50 true / 65 false`, preserve
`62 executable / 53 nonexecutable` structural coverage, [run](../../../../glossary.md#term-run) focused verification without timing or
a final full build, and obtain a fresh independent cumulative Class C approval before returning
this task to Complete.

### Extension implementation evidence

Implementation `5ab9c44c` admits exactly wires `46..49` and `52` for [ACCELERATOR](../../../../glossary.md#term-accelerator) [canonical](../../../../glossary.md#term-canonical-form)
positive-[rank](../../../../glossary.md#term-rank) [FLOAT32](../../../../glossary.md#term-float32) no-[gradient](../../../../glossary.md#term-gradient) input/output occurrences. Wire `50` and every other prior blocker
remain false. The [capability ledger](../../../../glossary.md#capability-ledger) is `50 true / 65 false`; structural coverage remains
`62 executable / 53 nonexecutable`. The [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog is
`75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; custom state remains
`27 AVAILABLE / 88 PENDING / 0 UNAVAILABLE_WITH_PROOF`.

Native [lowering](../../../../glossary.md#lowering) creates each [scalar](../../../../glossary.md#term-scalar) directly from its exact four raw bytes at [Shape](../../../../glossary.md#shape) `[1]` and lets
the one approved [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) MPSGraph primitive broadcast it. Scalar arithmetic retains
`input op scalar`; reciprocal uses exact `+1.0f` primary and input secondary for one division.
Java and native [preflight](../../../../glossary.md#term-preflight) independently reject gradient-bearing values, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) values, and a
non-FLOAT32 scalar attribute. [Schema](../../../../glossary.md#term-schema) 14, [identity](../../../../glossary.md#term-identity-meanings) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, route wire 3, and thirteen exports are
unchanged. The existing version-15 workload [digest](../../../../glossary.md#term-digest) now also binds the source wire, primitive wire,
operand order, FLOAT32 type, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [extent](../../../../glossary.md#term-extent), four-byte [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), exact scalar bits, and logical
input Shape; encoded schema and candidate-decision bytes did not change.

Focused capability/catalog/schema/route/[fingerprint](../../../../glossary.md#term-fingerprint)/native-malformed tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance),
and the public [Metal](../../../../glossary.md#term-metal)-only [Engine](../../../../glossary.md#term-engine) scenario passed against the rebuilt [native library](../../../../glossary.md#term-native-library). The public
scenario exercises both scalar zero signs, signed [subnormals](../../../../glossary.md#term-subnormal), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), quiet/signaling NaNs,
SUB/DIV operand order, both zero denominators, reciprocal, repeated execution, input preservation,
sole Metal [ownership](../../../../glossary.md#term-ownership-meanings), and pre-native gradient rejection. Native build, fixed [ad-hoc signing](../../../../glossary.md#term-ad-hoc-signing), local
package [publication](../../../../glossary.md#term-publication), native and [Gradle](../../../../glossary.md#term-gradle) package verification, the exact thirteen-export audit,
Metal [Javadoc](../../../../glossary.md#term-javadoc), a targeted ten-file/617-link Markdown path/anchor/fence/newline/whitespace check, and
final diff/clean-worktree [validation](../../../../glossary.md#term-validation) passed. No timing, benchmark, [autotuning](../../../../glossary.md#autotuning--model-autotuning), fallback, retry, or
final full repository build ran.

The independent cumulative Class C review of `5ab9c44c`/`63c070cc` returned `APPROVE` with zero
P0/P1/P2. It verified exact four-byte Shape `[1]` raw [constants](../../../../glossary.md#term-constant), operand order, the no-gradient
ACCELERATOR-only domain, identity/fingerprint coverage, `50/65` capability, `62/53` structural
coverage, unchanged blocked rows, and public Engine semantics and [lifecycle](../../../../glossary.md#lifecycle).

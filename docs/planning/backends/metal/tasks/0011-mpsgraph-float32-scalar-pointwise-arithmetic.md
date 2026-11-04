# Task 0011: [MPSGraph](../../../../glossary.md#term-mpsgraph) [FLOAT32](../../../../glossary.md#term-float32) [Scalar](../../../../glossary.md#term-scalar) [Pointwise](../../../../glossary.md#term-pointwise) Arithmetic

## Reader context

“Scalar” here meant an exact FLOAT32 attribute used as the ordered right operand, not a public
rank-zero input Tensor. Its proposed native constant would belong to the reusable executable;
it would not be borrowed caller storage or uploaded as a fresh feed each run.

Compiler captured that attribute and Metal would encode its bits and lower the operation.
The tensor descriptor still described a positive-rank input. Reusing a familiar arithmetic
selector did not establish that constant-specialized execution preserved tensor-tensor semantics:
the recorded counterexamples blocked this proposal.
The [current numerical domains](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains)
describe later scalar arithmetic, including its actual rank-zero boundary, rather than treating
the prospective schema below as an implemented migration.

## Status

Blocked

The mandatory real-[device](../../../../glossary.md#term-device) probe and independent blocker review completed [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). No
production, test, native, or probe changes remain. [Metal](../../../../glossary.md#term-metal) 0008 stays Complete, and 0006, 0007,
0009, and 0010 remain separate blockers.

## Blocking evidence

On an Apple M3 Max, real [MPSGraph](../../../../glossary.md#term-mpsgraph) [scalar](../../../../glossary.md#term-scalar) executables ran at optimization levels `0` and `1` with
reduced-[precision](../../../../glossary.md#term-precision) fast math disabled. Three independently compiled executables per level ran every
case eight times using exact raw embedded [constants](../../../../glossary.md#term-constant), caller-supplied direct [targets](../../../../glossary.md#term-target), canaries,
feed/target permutations, and raw input [identity](../../../../glossary.md#term-identity-meanings) controls.

The bounded [matrix](../../../../glossary.md#term-matrix-meanings) performed 77,824 scalar checks and found 17,328 mismatches. Reproducible
counterexamples include:

- embedded scalar `ADD +0` mapped input `-0` to `-0` in all 48 executions, while both the strict
  FLOAT32 oracle and the completed tensor-tensor <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [oracle glossary](../../../../glossary.md#term-test-oracle); [tensor glossary](../../../../glossary.md#tensor))</span>`ADD` route produce `+0`;
- positive minimum-[subnormal](../../../../glossary.md#term-subnormal) `ADD +0` flushed to zero;
- positive minimum-subnormal `DIV +0` produced [NaN](../../../../glossary.md#term-nan) instead of [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero);
- maximum-[finite](../../../../glossary.md#term-finite) `DIV` maximum-finite produced zero instead of one; and
- positive infinity `DIV` maximum-finite produced NaN instead of positive infinity.

The constant-bit, direct-target, [canary](../../../../glossary.md#term-canary), [permutation](../../../../glossary.md#permutation), repetition, and identity controls exclude
constant construction, upload/download, [binding](../../../../glossary.md#term-binding), overwrite, and executable-instability errors.
The independent review confirmed these are [Model](../../../../glossary.md#term-model)-visible arithmetic failures. Model permits no
relaxation here beyond [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) and signaling state, neither of which explains any
counterexample. Unblocking requires an exact replacement route or an explicit Model numerical
contract change. This task authorizes neither a custom [kernel](../../../../glossary.md#kernel), fallback, nor relaxation.

## Change class

Class C — this extends [Metal](../../../../glossary.md#term-metal) capability, the private typed native [schema](../../../../glossary.md#term-schema), candidate [identity](../../../../glossary.md#term-identity-meanings),
whole-[partition](../../../../glossary.md#partition) [lowering](../../../../glossary.md#lowering), persistent [MPSGraph](../../../../glossary.md#term-mpsgraph) resources, and public [Engine](../../../../glossary.md#term-engine) execution while
preserving the existing architecture, [ABI](../../../../glossary.md#term-application-binary-interface-abi) exports, storage types, and transfer contract.

## Goal

Add the four one-input [`FLOAT32`](../../../../glossary.md#term-float32) [scalar](../../../../glossary.md#term-scalar) [pointwise](../../../../glossary.md#term-pointwise) operations
`ScalarElementwiseKind.ADD`, `SUB`, `MUL`, and `DIV` with exact `ScalarValueAttrs`. Lower each
attribute to one embedded exact-bit [MPSGraph](../../../../glossary.md#term-mpsgraph) [constant](../../../../glossary.md#term-constant) and the corresponding already-used arithmetic
selector. Keep the capability forward-only and independent of every blocked [Metal](../../../../glossary.md#term-metal) branch.

## Inventory and [scope](../../../../glossary.md#term-scope) decision

The current [Model](../../../../glossary.md#term-model) [pointwise](../../../../glossary.md#term-pointwise) inventory contains seven `BinaryArithmeticKind` values, eight
`ScalarElementwiseKind` values, and nineteen `UnaryElementwiseKind` values, plus comparison,
classification, [BOOL](../../../../glossary.md#data-type--datatype) logic, selection, and cast families. [Metal](../../../../glossary.md#term-metal) 0005 already completed [tensor](../../../../glossary.md#tensor)
`ADD/SUB/MUL/DIV` and `NEG`; Metal 0008 completed five non-arithmetic affine transforms.

The safest remaining coherent [slice](../../../../glossary.md#slice) is [scalar](../../../../glossary.md#term-scalar) `ADD/SUB/MUL/DIV`: Model gives each one the same
ordered arithmetic meaning as its completed 0005 tensor counterpart, [Compiler](../../../../glossary.md#term-compiler) already [captures](../../../../glossary.md#term-capture-meanings) the
one-input typed-attribute [occurrence](../../../../glossary.md#term-occurrence), and the installed [SDK](../../../../glossary.md#term-sdk) exposes
`additionWithPrimaryTensor:secondaryTensor:name:`, `subtractionWithPrimaryTensor:secondaryTensor:name:`,
`multiplicationWithPrimaryTensor:secondaryTensor:name:`, and
`divisionWithPrimaryTensor:secondaryTensor:name:`. The SDK also exposes
`constantWithData:shape:dataType:` for an exact four-byte [`FLOAT32`](../../../../glossary.md#term-float32) [constant](../../../../glossary.md#term-constant).

[Binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/scalar `MIN` and `MAX` and `CLAMP` remain outside this task because their exact [NaN](../../../../glossary.md#term-nan),
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and represented-value ordering requires a separate [extrema](../../../../glossary.md#saturation-pole-and-extremum) probe after 0010's
comparison failure. Binary/scalar `POW` and every remaining [unary](../../../../glossary.md#term-unary) kind stay outside because they
are transcendental or unary numerical work adjacent to blocked 0006. This task does not rename or
repackage any failed 0006 or 0010 [operation](../../../../glossary.md#operation).

## Exact capability domain

Every admitted [occurrence](../../../../glossary.md#term-occurrence) has exactly one input and one output, one of the four listed kinds, and
an exact `ScalarValueAttrs` whose [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) is [`FLOAT32`](../../../../glossary.md#term-float32). Input and output must have identical
`FLOAT32` [descriptors](../../../../glossary.md#term-descriptor) except for value [identity](../../../../glossary.md#term-identity-meanings): fully static positive [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank) `1..16`,
[canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) zero-offset non-[view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout), and equal `requiresGrad` flags. Checked
[element](../../../../glossary.md#term-element) and byte geometry is mandatory.

The [scalar](../../../../glossary.md#term-scalar) is the ordered right operand: `input + scalar`, `input - scalar`, `input * scalar`, or
`input / scalar`. Its exact raw 32 bits are semantic [schema](../../../../glossary.md#term-schema) and candidate identity; no conversion
through `double`, text, [host](../../../../glossary.md#term-host) [Tensor](../../../../glossary.md#tensor) storage, or a public [graph](../../../../glossary.md#graph) input is allowed. Existing 0005
[pointwise](../../../../glossary.md#term-pointwise) [nodes](../../../../glossary.md#node) may precede or follow a scalar node. Existing 0008 affine nodes remain terminal
under their current logical-view restriction.

Capability and analysis [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for `MIN`, `MAX`, `POW`, `CLAMP`, every [unary](../../../../glossary.md#term-unary) except existing
`NEG`, comparisons, classification, [BOOL](../../../../glossary.md#data-type--datatype), `WHERE`, `CAST`, reductions, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), non-FLOAT32 scalar
attributes, wrong attrs/[arity](../../../../glossary.md#term-arity)/result [metadata](../../../../glossary.md#term-metadata), arbitrary views, scalar/zero/dynamic Shapes, rank
above sixteen, unchecked geometry, or malformed [topology](../../../../glossary.md#term-topology).

## Mandatory real-[device](../../../../glossary.md#term-device) probe and [fail-closed](../../../../glossary.md#term-fail-closed) gate

Before production edits, [compile](../../../../glossary.md#compile) and [run](../../../../glossary.md#term-run) a disposable Apple-silicon probe with [MPSGraph](../../../../glossary.md#term-mpsgraph)
optimization levels `0` and `1`, reduced-[precision](../../../../glossary.md#term-precision) fast math disabled, at least three independently
compiled executables per level, and at least eight executions per case. Use production-style
[`FLOAT32`](../../../../glossary.md#term-float32) placeholders, an exact four-byte `[1]` [constant](../../../../glossary.md#term-constant) from
`constantWithData:shape:dataType:`, the four named arithmetic selectors, caller-supplied
[canary](../../../../glossary.md#term-canary)-filled direct [targets](../../../../glossary.md#term-target), and feed/target [permutation](../../../../glossary.md#permutation) checks.

The probe must cover both input and [scalar](../../../../glossary.md#term-scalar) positions with positive/negative minimum and ordinary
[subnormals](../../../../glossary.md#term-subnormal), both signed zeros, minimum/maximum [finite](../../../../glossary.md#term-finite) values, ordinary exact and inexact finite
values, both [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and quiet/signaling NaNs of both signs and multiple payloads. It must prove:

- exact scalar raw-bit construction for finite values, signed zeros, subnormals, and infinities;
- ordered `SUB` and `DIV`, all four special-value tables, and NaN classification without imposing a
  NaN payload or signaling-state promise absent from Model;<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Model glossary](../../../../glossary.md#term-model))</span>
- raw-bit results for uniquely fixed exact cases, including `±min-subnormal` with `+0`, `-0`, `+1`,
  and `-1`, plus strict FLOAT32 reference results for ordinary arithmetic;
- unchanged output [Shape](../../../../glossary.md#shape), [rank](../../../../glossary.md#term-rank)-`1` and rank-`16` cases, stable mixed chains with existing 0005
  nodes, complete direct writes, untouched outside canaries, permutations, repetition, and
  executable independence; and<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node))</span>
- raw input identity controls so an arithmetic result cannot be mistaken for upload/download or
  target-binding corruption.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>

If any selector, constant byte, subnormal, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), special-value, Shape, direct-target,
permutation, repetition, or independence gate fails, remove the probe and mark 0011 Blocked. Do
not substitute a custom [kernel](../../../../glossary.md#kernel), [host](../../../../glossary.md#term-host)/[CPU](../../../../glossary.md#term-cpu) fallback, per-run scalar feed, numerical relaxation, or
contract change.

## Native [schema](../../../../glossary.md#term-schema), candidates, and [lifecycle](../../../../glossary.md#lifecycle)

Keep native [ABI](../../../../glossary.md#term-application-binary-interface-abi) version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte [node](../../../../glossary.md#node)
schema from version 2 to version 3. Preserve [operation](../../../../glossary.md#operation) wires `1..10`; assign `11..14` to [scalar](../../../../glossary.md#term-scalar)
`ADD/SUB/MUL/DIV` in that order and attribute wire `4` to `FLOAT32_SCALAR`. A scalar record has one
input, `second_input = UINT32_MAX`, `attribute_count = 1`, `axis = UINT32_MAX`, `reserved = 0`, and
the exact unsigned raw 32 bits in the low half of `attribute_values[0]` with its high half zero.
Every other unused [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) cell is zero.

Java [preflight](../../../../glossary.md#term-preflight) validates exact [`FLOAT32`](../../../../glossary.md#term-float32) [descriptors](../../../../glossary.md#term-descriptor), flags, kind/attribute pairing, and geometry.
Native creation validates schema sentinels, payload width, [Shapes](../../../../glossary.md#shape), [topology](../../../../glossary.md#term-topology), and geometry, then
creates one `[1]` `MPSDataTypeFloat32` [constant](../../../../glossary.md#term-constant) from those exact four bytes and applies the ordered
arithmetic selector. The constant is executable-owned prepared state: it adds no [graph](../../../../glossary.md#graph) boundary,
[Metal](../../../../glossary.md#term-metal) [buffer](../../../../glossary.md#term-buffer), resource declaration, per-[run](../../../../glossary.md#term-run) upload, workspace, or hot-path branch.

Bump candidate, compatibility/workload, route-policy, and [backend](../../../../glossary.md#backend)-local [codec](../../../../glossary.md#term-codec) versions. [Identity](../../../../glossary.md#term-identity-meanings)
includes ordered typed nodes, descriptors, exact scalar bits and roles, schema/ABI, topology,
splats, and live [target](../../../../glossary.md#term-target). Old decisions [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). The custom [singleton](../../../../glossary.md#term-singleton)-NEG route remains exact;
any [partition](../../../../glossary.md#partition) containing a scalar node uses the [MPSGraph](../../../../glossary.md#term-mpsgraph) route.

Analysis still declares all buffers and workspace before assignment. Finalization transactionally
creates one reusable Shape-specialized executable. Cold [binding](../../../../glossary.md#term-binding) validates representations; hot
execution performs one [synchronous](../../../../glossary.md#term-synchronous) native submission. Existing [rollback](../../../../glossary.md#term-rollback), autorelease-pool,
closure, repeated/concurrent run, and independent-session rules remain unchanged.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Metal glossary](../../../../glossary.md#term-metal); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend))</span>
  — Metal owns MPSGraph lowering, native schema, resources, and materialization.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [schema glossary](../../../../glossary.md#term-schema); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  — current forward capture is consumed without a backward or training claim.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Tensor API](../../../../api/tensor-api.md#scalar-arithmetic-and-clamp-expressions)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span> — exact typed
  scalar attributes and ordered arithmetic meanings remain Model-owned.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

If implementation requires a shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), new [dependency](../../../../glossary.md#term-dependency), [ABI](../../../../glossary.md#term-application-binary-interface-abi) export/version,
per-[run](../../../../glossary.md#term-run) [scalar](../../../../glossary.md#term-scalar) resource, or transfer change, stop and mark 0011 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model scalar semantics and Compiler forward capture;
  Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not 0006/0007/0009/0010 or
  Model 0026.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Engine glossary](../../../../glossary.md#term-engine); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- **Conflicts with:** every concurrent Metal capability, preparation, native schema, candidate/codec,
  materialization, or Engine Metal integration change.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization))</span>
- **Parallel group:** None.
- **Common base revision:** `45225a7a6c4219ff61913319d454589a5d50e909`.
- **Integration order:** mandatory probe → schema and exact scalar carrier → capability/topology →
  candidates/lifecycle → Engine proof → documentation and independent review.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- **Integration [validation](../../../../glossary.md#term-validation):** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer and candidate/codec
  types — exact scalar domain, schema v3, topology, geometry, and invalidation.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [scalar glossary](../../../../glossary.md#term-scalar); [schema glossary](../../../../glossary.md#term-schema); [topology glossary](../../../../glossary.md#term-topology))</span>
- [Metal](../../../../glossary.md#term-metal) finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — embedded constant creation, ordered selectors, reusable resource, and unchanged hot lifecycle.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability and real behavior proofs.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Acceptance criteria

- Capability/preparation admit exactly the four listed scalar operations and preserve every existing
  route, affine restriction, FLOAT32 storage/publication, and FLOAT32-only cross-owner transfer.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [publication glossary](../../../../glossary.md#term-publication); [owner glossary](../../../../glossary.md#term-owner))</span>
- The disposable probe passes every constant-bit, arithmetic, subnormal, signed-zero, special-value,
  Shape, direct-target, canary, permutation, repetition, and independence gate before production.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [subnormal glossary](../../../../glossary.md#term-subnormal); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [canary glossary](../../../../glossary.md#term-canary); [permutation glossary](../../../../glossary.md#permutation))</span>
- [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains exactly thirteen exports; [schema](../../../../glossary.md#term-schema) v3 retains 160-byte records, uses only wires
  `11..14` and [scalar](../../../../glossary.md#term-scalar) attribute wire `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover each scalar kind and raw attribute class, wrong attrs/types/flags,
  topology and geometry boundaries, rollback, close, reuse, concurrency, and independent sessions.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- A [Metal](../../../../glossary.md#term-metal)-only public [Engine](../../../../glossary.md#term-engine) [run](../../../../glossary.md#term-run) composes caller-input scalar `ADD/SUB/MUL/DIV` with existing 0005
  nodes in one partition, publishes direct exact outputs, repeats across sessions, and proves no CPU
  owner or per-run scalar upload.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- No extrema, POW, failed unary/predicate family, BOOL, CAST, reduction, MATMUL, custom kernel,
  backward/training, transfer widening, FLOAT16, BFLOAT16, or performance claim is introduced.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [unary glossary](../../../../glossary.md#term-unary); [BOOL glossary](../../../../glossary.md#data-type--datatype); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [kernel glossary](../../../../glossary.md#kernel); [widening glossary](../../../../glossary.md#term-widening); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Independent Class C review returns APPROVE before completion.

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Scalar*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Scalar*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier,
thirteen exports, no custom [scalar](../../../../glossary.md#term-scalar) [kernel](../../../../glossary.md#kernel)/per-[run](../../../../glossary.md#term-run) scalar feed, unchanged contracts, and no excluded-family claim.

## Documentation and review impact

Update [Metal](../../../../glossary.md#term-metal) Javadocs, package documentation, [backend](../../../../glossary.md#backend)/native guides, targeted public [scope](../../../../glossary.md#term-scope) text, and
glossary status only after evidence passes. [Model](../../../../glossary.md#term-model)/[Tensor](../../../../glossary.md#tensor) semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, [scalar](../../../../glossary.md#term-scalar) raw-bit handling, [schema](../../../../glossary.md#term-schema) and
compatibility invalidation, [lifecycle](../../../../glossary.md#lifecycle)/resource non-expansion, [Engine](../../../../glossary.md#term-engine) evidence, and all exclusions.

## Result

Blocked. The required optimization-level `0`/`1`, three-executable-per-level, eight-[run](../../../../glossary.md#term-run) [matrix](../../../../glossary.md#term-matrix-meanings)
found 17,328 mismatches in 77,824 [scalar](../../../../glossary.md#term-scalar) checks, including [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [subnormal](../../../../glossary.md#term-subnormal), [finite](../../../../glossary.md#term-finite) division,
and [infinity](../../../../glossary.md#term-infinity) division failures. Exact [constants](../../../../glossary.md#term-constant), direct [targets](../../../../glossary.md#term-target), canaries, permutations, repeated
runs, and [identity](../../../../glossary.md#term-identity-meanings) controls passed. Independent blocker review confirmed no existing [Model](../../../../glossary.md#term-model)
relaxation admits the results. No production, native, test, or probe changes remain.

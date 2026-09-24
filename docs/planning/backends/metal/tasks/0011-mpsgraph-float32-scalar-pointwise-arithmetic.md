# Task 0011: MPSGraph FLOAT32 Scalar Pointwise Arithmetic

## Status

Blocked

The mandatory real-device probe and independent blocker review completed fail closed. No
production, test, native, or probe changes remain. Metal 0008 stays Complete, and 0006, 0007,
0009, and 0010 remain separate blockers.

## Blocking evidence

On an Apple M3 Max, real MPSGraph scalar executables ran at optimization levels `0` and `1` with
reduced-precision fast math disabled. Three independently compiled executables per level ran every
case eight times using exact raw embedded constants, caller-supplied direct targets, canaries,
feed/target permutations, and raw input identity controls.

The bounded matrix performed 77,824 scalar checks and found 17,328 mismatches. Reproducible
counterexamples include:

- embedded scalar `ADD +0` mapped input `-0` to `-0` in all 48 executions, while both the strict
  FLOAT32 oracle and the completed tensor-tensor `ADD` route produce `+0`;
- positive minimum-subnormal `ADD +0` flushed to zero;
- positive minimum-subnormal `DIV +0` produced NaN instead of positive infinity;
- maximum-finite `DIV` maximum-finite produced zero instead of one; and
- positive infinity `DIV` maximum-finite produced NaN instead of positive infinity.

The constant-bit, direct-target, canary, permutation, repetition, and identity controls exclude
constant construction, upload/download, binding, overwrite, and executable-instability errors.
The independent review confirmed these are Model-visible arithmetic failures. Model permits no
relaxation here beyond NaN payload and signaling state, neither of which explains any
counterexample. Unblocking requires an exact replacement route or an explicit Model numerical
contract change. This task authorizes neither a custom kernel, fallback, nor relaxation.

## Change class

Class C — this extends Metal capability, the private typed native schema, candidate identity,
whole-partition lowering, persistent MPSGraph resources, and public Engine execution while
preserving the existing architecture, ABI exports, storage types, and transfer contract.

## Goal

Add the four one-input `FLOAT32` scalar pointwise operations
`ScalarElementwiseKind.ADD`, `SUB`, `MUL`, and `DIV` with exact `ScalarValueAttrs`. Lower each
attribute to one embedded exact-bit MPSGraph constant and the corresponding already-used arithmetic
selector. Keep the capability forward-only and independent of every blocked Metal branch.

## Inventory and scope decision

The current Model pointwise inventory contains seven `BinaryArithmeticKind` values, eight
`ScalarElementwiseKind` values, and nineteen `UnaryElementwiseKind` values, plus comparison,
classification, BOOL logic, selection, and cast families. Metal 0005 already completed tensor
`ADD/SUB/MUL/DIV` and `NEG`; Metal 0008 completed five non-arithmetic affine transforms.

The safest remaining coherent slice is scalar `ADD/SUB/MUL/DIV`: Model gives each one the same
ordered arithmetic meaning as its completed 0005 tensor counterpart, Compiler already captures the
one-input typed-attribute occurrence, and the installed SDK exposes
`additionWithPrimaryTensor:secondaryTensor:name:`, `subtractionWithPrimaryTensor:secondaryTensor:name:`,
`multiplicationWithPrimaryTensor:secondaryTensor:name:`, and
`divisionWithPrimaryTensor:secondaryTensor:name:`. The SDK also exposes
`constantWithData:shape:dataType:` for an exact four-byte `FLOAT32` constant.

Binary/scalar `MIN` and `MAX` and `CLAMP` remain outside this task because their exact NaN,
signed-zero, and represented-value ordering requires a separate extrema probe after 0010's
comparison failure. Binary/scalar `POW` and every remaining unary kind stay outside because they
are transcendental or unary numerical work adjacent to blocked 0006. This task does not rename or
repackage any failed 0006 or 0010 operation.

## Exact capability domain

Every admitted occurrence has exactly one input and one output, one of the four listed kinds, and
an exact `ScalarValueAttrs` whose `ScalarValue` is `FLOAT32`. Input and output must have identical
`FLOAT32` descriptors except for value identity: fully static positive Shape, rank `1..16`,
canonical dense-contiguous zero-offset non-view layout, and equal `requiresGrad` flags. Checked
element and byte geometry is mandatory.

The scalar is the ordered right operand: `input + scalar`, `input - scalar`, `input * scalar`, or
`input / scalar`. Its exact raw 32 bits are semantic schema and candidate identity; no conversion
through `double`, text, host Tensor storage, or a public graph input is allowed. Existing 0005
pointwise nodes may precede or follow a scalar node. Existing 0008 affine nodes remain terminal
under their current logical-view restriction.

Capability and analysis fail closed for `MIN`, `MAX`, `POW`, `CLAMP`, every unary except existing
`NEG`, comparisons, classification, BOOL, `WHERE`, `CAST`, reductions, MATMUL, non-FLOAT32 scalar
attributes, wrong attrs/arity/result metadata, arbitrary views, scalar/zero/dynamic Shapes, rank
above sixteen, unchecked geometry, or malformed topology.

## Mandatory real-device probe and fail-closed gate

Before production edits, compile and run a disposable Apple-silicon probe with MPSGraph
optimization levels `0` and `1`, reduced-precision fast math disabled, at least three independently
compiled executables per level, and at least eight executions per case. Use production-style
`FLOAT32` placeholders, an exact four-byte `[1]` constant from
`constantWithData:shape:dataType:`, the four named arithmetic selectors, caller-supplied
canary-filled direct targets, and feed/target permutation checks.

The probe must cover both input and scalar positions with positive/negative minimum and ordinary
subnormals, both signed zeros, minimum/maximum finite values, ordinary exact and inexact finite
values, both infinities, and quiet/signaling NaNs of both signs and multiple payloads. It must prove:

- exact scalar raw-bit construction for finite values, signed zeros, subnormals, and infinities;
- ordered `SUB` and `DIV`, all four special-value tables, and NaN classification without imposing a
  NaN payload or signaling-state promise absent from Model;
- raw-bit results for uniquely fixed exact cases, including `±min-subnormal` with `+0`, `-0`, `+1`,
  and `-1`, plus strict FLOAT32 reference results for ordinary arithmetic;
- unchanged output Shape, rank-`1` and rank-`16` cases, stable mixed chains with existing 0005
  nodes, complete direct writes, untouched outside canaries, permutations, repetition, and
  executable independence; and
- raw input identity controls so an arithmetic result cannot be mistaken for upload/download or
  target-binding corruption.

If any selector, constant byte, subnormal, signed-zero, special-value, Shape, direct-target,
permutation, repetition, or independence gate fails, remove the probe and mark 0011 Blocked. Do
not substitute a custom kernel, host/CPU fallback, per-run scalar feed, numerical relaxation, or
contract change.

## Native schema, candidates, and lifecycle

Keep native ABI version 4, statuses, and exactly thirteen exports. Bump the fixed 160-byte node
schema from version 2 to version 3. Preserve operation wires `1..10`; assign `11..14` to scalar
`ADD/SUB/MUL/DIV` in that order and attribute wire `4` to `FLOAT32_SCALAR`. A scalar record has one
input, `second_input = UINT32_MAX`, `attribute_count = 1`, `axis = UINT32_MAX`, `reserved = 0`, and
the exact unsigned raw 32 bits in the low half of `attribute_values[0]` with its high half zero.
Every other unused payload cell is zero.

Java preflight validates exact `FLOAT32` descriptors, flags, kind/attribute pairing, and geometry.
Native creation validates schema sentinels, payload width, Shapes, topology, and geometry, then
creates one `[1]` `MPSDataTypeFloat32` constant from those exact four bytes and applies the ordered
arithmetic selector. The constant is executable-owned prepared state: it adds no graph boundary,
Metal buffer, resource declaration, per-run upload, workspace, or hot-path branch.

Bump candidate, compatibility/workload, route-policy, and backend-local codec versions. Identity
includes ordered typed nodes, descriptors, exact scalar bits and roles, schema/ABI, topology,
splats, and live target. Old decisions fail closed. The custom singleton-NEG route remains exact;
any partition containing a scalar node uses the MPSGraph route.

Analysis still declares all buffers and workspace before assignment. Finalization transactionally
creates one reusable Shape-specialized executable. Cold binding validates representations; hot
execution performs one synchronous native submission. Existing rollback, autorelease-pool,
closure, repeated/concurrent run, and independent-session rules remain unchanged.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Planning
  selects ownership; Metal owns truthful lowering and Runtime receives prepared work.
- [Concrete backend modules and Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  — Metal owns MPSGraph lowering, native schema, resources, and materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle) —
  route and resources are fixed before direct hot execution.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — current forward capture is consumed without a backward or training claim.
- [Tensor API](../../../../api/tensor-api.md#scalar-arithmetic-and-clamp-expressions) — exact typed
  scalar attributes and ordered arithmetic meanings remain Model-owned.

If implementation requires a shared contract, public API, new dependency, ABI export/version,
per-run scalar resource, or transfer change, stop and mark 0011 Blocked.

## Dependencies and integration

- **Depends on:** Metal 0005 and 0008; current Model scalar semantics and Compiler forward capture;
  Engine 0017; Compiler 0006B7/0006B11; Prepare 0008; Runtime 0016. Not 0006/0007/0009/0010 or
  Model 0026.
- **Conflicts with:** every concurrent Metal capability, preparation, native schema, candidate/codec,
  materialization, or Engine Metal integration change.
- **Parallel group:** None.
- **Common base revision:** `45225a7a6c4219ff61913319d454589a5d50e909`.
- **Integration order:** mandatory probe → schema and exact scalar carrier → capability/topology →
  candidates/lifecycle → Engine proof → documentation and independent review.
- **Integration validation:** native build/export audit; focused Metal, conformance, real Engine,
  Compiler-contract, architecture, Javadoc, Markdown, and whitespace checks.
- **Shared-document integration owner:** Main planner.

## Files and symbols

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNativeApi`, preparer and candidate/codec
  types — exact scalar domain, schema v3, topology, geometry, and invalidation.
- Metal finalization/execution types and `native/metal-macos-arm64/src/synaptik_metal_foundation.m`
  — embedded constant creation, ordered selectors, reusable resource, and unchanged hot lifecycle.
- Focused Metal/conformance/Compiler/Engine tests and affected Metal/native guides and Javadocs —
  fail-closed capability and real behavior proofs.

## Acceptance criteria

- Capability/preparation admit exactly the four listed scalar operations and preserve every existing
  route, affine restriction, FLOAT32 storage/publication, and FLOAT32-only cross-owner transfer.
- The disposable probe passes every constant-bit, arithmetic, subnormal, signed-zero, special-value,
  Shape, direct-target, canary, permutation, repetition, and independence gate before production.
- ABI v4 retains exactly thirteen exports; schema v3 retains 160-byte records, uses only wires
  `11..14` and scalar attribute wire `4`, and rejects malformed/stale records and decisions.
- Fake and real tests cover each scalar kind and raw attribute class, wrong attrs/types/flags,
  topology and geometry boundaries, rollback, close, reuse, concurrency, and independent sessions.
- A Metal-only public Engine run composes caller-input scalar `ADD/SUB/MUL/DIV` with existing 0005
  nodes in one partition, publishes direct exact outputs, repeats across sessions, and proves no CPU
  owner or per-run scalar upload.
- No extrema, POW, failed unary/predicate family, BOOL, CAST, reduction, MATMUL, custom kernel,
  backward/training, transfer widening, FLOAT16, BFLOAT16, or performance claim is introduced.
- Independent Class C review returns APPROVE before completion.

## Validation

```bash
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./gradlew :backends:metal:test --tests '*Metal*Scalar*' --tests '*Metal*MpsGraph*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :modules:compiler:test --tests '*Scalar*' :backends:metal:javadoc :testing:architecture-tests:test
git diff --check
```

Also validate balanced Markdown fences, local links/anchors, exactly one Ready Metal frontier,
thirteen exports, no custom scalar kernel/per-run scalar feed, unchanged contracts, and no excluded-family claim.

## Documentation and review impact

Update Metal Javadocs, package documentation, backend/native guides, targeted public scope text, and
glossary status only after evidence passes. Model/Tensor semantics and architecture contracts remain
unchanged. Independent Class C review inspects capability truth, scalar raw-bit handling, schema and
compatibility invalidation, lifecycle/resource non-expansion, Engine evidence, and all exclusions.

## Result

Blocked. The required optimization-level `0`/`1`, three-executable-per-level, eight-run matrix
found 17,328 mismatches in 77,824 scalar checks, including signed-zero, subnormal, finite division,
and infinity division failures. Exact constants, direct targets, canaries, permutations, repeated
runs, and identity controls passed. Independent blocker review confirmed no existing Model
relaxation admits the results. No production, native, test, or probe changes remain.

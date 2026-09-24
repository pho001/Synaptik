# Task 0026: Profile-Common Canonical FLOAT32 SOFTMAX

## Status

Ready

## Planning decision

The broad todo `Implementovat Metal 0018 softmax a normalizace` becomes one new serial Metal task,
not a restart or renumbering of historical blocked Metal 0018. This task implements only first-class
`SoftmaxKind.SOFTMAX`. It does not implement `LOG_SOFTMAX`, layer normalization, RMS normalization,
or batch normalization.

The smallest coherent slice is canonical static positive-Shape `FLOAT32` SOFTMAX forward with the
same finite policy under `STRICT_IEEE` and `ACCELERATOR`, plus only the existing Compiler-generated
SOFTMAX gradient graph under `ACCELERATOR`. There is no backend gradient operation, selector,
fusion, rewrite, or semantic kind.

There is deliberately no selected-axis upper bound. A width limit such as 16 would be arbitrary:
Model defines no such limit, the direct MPSGraph selector performs no selector-list expansion, and
the existing `MAX_SELECTOR_EXPANSION=16` belongs only to `UNFOLD_AXIS`. Admit every strictly
positive selected extent whose complete static Shape, element count, byte count, and native integer
conversions pass the existing checked Metal geometry. Rank remains bounded to 1..16 by the current
Metal ABI dimension table.

## Change class

Class C — this adds a floating numerical selector to both graph-wide profile matrices, evolves the
private node schema and all route identities, and proves public Engine forward execution plus
accelerator generated-gradient closure. One smallest one-run real-device numerical gate is required
before production edits because the active SDK documents the SOFTMAX operation and axis but gives no
finite accuracy bound. No Shape, width, axis, profile, optimization, context, or repetition matrix
is permitted.

## Coordination metadata

- Depends on: Metal 0025 Complete at final reviewed revision
  `f88066e3ad0547987bb03b2d18ed2813f97de223`; Model 0016I–0016J; Compiler 0005B/0005C; CPU 0007E
  only as independent numerical-policy precedent; Config 0006; Engine 0018; current completed Metal
  lifecycle, profile, schema, and opaque-route contracts.
- Planning base: exact clean reviewed Metal 0025 revision
  `f88066e3ad0547987bb03b2d18ed2813f97de223`. Implementation begins only from the committed revision
  containing this plan.
- Conflicts with: any Metal capability, schema, native-preflight, candidate/codec, materialization,
  or public Engine scope; any blocked Metal 0016–0018 restart; another normalization, loss,
  attention, fusion, or shared semantic task.
- Parallel group: None.
- Integration order: one fail-closed direct-selector numerical gate; schema/identity; capability and
  Java/native preflight; native lowering; focused forward and generated-gradient composition;
  current documentation; independent lean Class C review.
- Integration validation: focused Metal/conformance/real-Engine proof, native export/link audit,
  architecture checks, Javadoc, one final build, Markdown/diff validation.
- Shared-document integration owner: Metal 0026 implementer.

## Model and profile contract

Model already owns the complete public meaning:

- `SoftmaxKind.SOFTMAX` is one first-class one-input/one-output operation paired exactly with
  `SoftmaxAttrs`;
- `SoftmaxAttrs.axis()` is already normalized, non-negative, and must be in range for the concrete
  input Shape;
- the output preserves the exact logical Shape, floating data type, and `requiresGrad` eligibility;
- for one selected-axis slice, the ideal result is
  `exp(x_i) / sum_j(exp(x_j))`, preserving every logical position; and
- Model deliberately defines no temperature, mask, scale, epsilon, algorithm switch, reduction
  order, finite-precision decomposition, or NaN/infinity/empty-axis policy.

`STRICT_IEEE` means that exact current per-operation Model contract; it is not a universal
correct-rounding promise. `ACCELERATOR` adds no normalization result freedom. The architecture's
profile table explicitly leaves normalization unrelaxed, so this task uses one identical forward
capability and finite policy under both profiles. For every fixed occurrence, Metal strict
capability is therefore a subset of Metal accelerator capability without widening SOFTMAX results.
Any future accelerator-only DAZ/FTZ, approximation, or special-value envelope requires a preceding
Model/architecture task and is not part of 0026.

## Exact capability domain

Admit one occurrence exactly when all of the following hold:

- kind and attributes are exactly `SoftmaxKind.SOFTMAX + SoftmaxAttrs`;
- there is one ordered input and one fresh output;
- both descriptors are `FLOAT32`, have the same exact Shape and `requiresGrad`, and have canonical
  dense-contiguous, zero-offset, non-view layouts;
- the Shape is fully static, rank 1..16, and every extent is strictly positive;
- the normalized axis is in `[0, rank)`; and
- checked element count, byte count, dimension-table cells, and every native `NSUInteger`/`NSInteger`
  conversion are representable before native resource creation.

The selected extent may be 1, 17, or any other positive representable value. Do not use
`MAX_SELECTOR_EXPANSION`, add a softmax width constant, or infer a width cap from the one-run probe.
The probe establishes one concrete undocumented numerical property of the direct selector; the
SDK's axis-generic selector declaration and shape-specialized compilation establish structural
applicability.

Reject `LOG_SOFTMAX`; wrong kind or attributes; wrong arity; non-FLOAT32 or cross-type descriptors;
scalar, dynamic, zero-extent, rank-above-16, axis-out-of-range, Shape-mismatched, layout-mismatched,
or `requiresGrad`-mismatched occurrences; affine views; aliases; overflow; and every unlisted
normalization family before native creation.

## Metal-private finite numerical policy

For every structurally admitted slice whose represented inputs are all finite, define the Task 0026
acceptance oracle as follows:

1. Convert each exact FLOAT32 input value to binary64.
2. Find the slice maximum in binary64.
3. Evaluate `StrictMath.exp(x_i - maximum)` in binary64 and accumulate the terms with a compensated
   binary64 sum in increasing selected-axis coordinate order.
4. Divide each binary64 exponential by that sum and narrow once to FLOAT32 with Java's
   round-to-nearest-ties-to-even conversion.
5. Require each Metal output to be finite, in `[0.0f, 1.0f]`, and at ordered FLOAT32 distance at
   most four result ULPs from that independently calculated reference value.

The four-ULP bound matches the completed CPU FLOAT32 softmax accuracy envelope but does not require
CPU's private three-pass algorithm, compensation strategy, raw-bit result, or rejection machinery.
It is a Metal-private executable policy, not new Model semantics and not a cross-backend equality
claim. The input raw bits must remain unchanged and the output must be fresh canonical storage.
Cellwise oracle comparison is the complete accuracy gate; do not add a redundant independently tuned
sum-to-one epsilon.

Signed-zero and subnormal inputs are finite and remain inside this finite gate. A reference result
may narrow to a subnormal or zero. No DAZ/FTZ permission is added. NaN and either infinity are
outside the advertised finite accuracy envelope because Model has not defined their SOFTMAX edge
policy. Task 0026 neither value-scans nor rejects them, freezes their observed MPSGraph result, or
claims payload/sign behavior; descriptor capability has no value predicate. All permanent numerical
acceptance data is finite. Zero selected extent and every other zero extent are structurally
rejected.

## One smallest real-device gate

The active macOS 27.0 SDK's `MPSGraphActivationOps.h` declares
`softMaxWithTensor:axis:name:` but states no finite error bound. That is one concrete undocumented
property necessary for the advertised four-ULP behavior, so exactly one disposable real-device run
is mandatory before production edits.

Use one process, Metal context/queue, MPSGraph, direct SOFTMAX node, executable, and execution. Mirror
production rather than creating a favorable probe mode: leave the compilation descriptor at its
production default optimization level, set and read back
`MPSGraphReducedPrecisionFastMathNone` where available, compile one direct target, and set
`MPSGraphOptionsNone` on the executable. Use one canonical `[4,8]` FLOAT32 input and axis one with
these four frozen finite slices:

```text
[1, 2, 3, -1, 0.5, -2, 4, 0]
[999997, 999998, 999999, 1000000, 1000001, 1000002, 1000003, 1000004]
[0, -80, -90, -100, -103, -104, -110, -120]
[+0, -0, +minSubnormal, -minSubnormal, +maxFinite, +maxFinite, -maxFinite, 1]
```

Compare all 32 cells to the independent policy above, require finite `[0,1]` results and unchanged
input raw bits, then remove probe source and binary immediately. This one execution covers ordinary
finite values, a large common offset, strongly skewed subnormal/zero probability results, signed
zeros, subnormal inputs, equal maxima, and the complete finite range. Do not probe another width,
axis, rank, profile, optimization level, context, repetition, NaN, or infinity.

If any cell fails, mark Task 0026 `Blocked`, retain only exact evidence, remove the probe, and make no
production change. Do not narrow capability to the probe width, weaken the tolerance, move SOFTMAX
to accelerator-only, decompose it, or introduce fallback.

## Compiler-owned gradient closure

Compiler production remains unchanged. `ReductionGradientRules` already expands first-order
SOFTMAX input contribution from the exact forward output `y` and incoming gradient `g` as:

```text
y * (g - sum(g * y, axis, keepDimensions=true))
```

Under `ACCELERATOR`, current Metal capability already owns the required canonical FLOAT32 `MUL`,
`SUB`, and keep-dimensions `SUM` occurrences, including their existing operation-local profile
semantics. After adding profile-common SOFTMAX forward, an explicitly seeded first-order graph in
the admitted canonical domain must therefore form one maximal Metal MPSGraph partition and execute
without CPU ownership or transfer.

This is composition of the compiler-generated graph, not a direct softmax-gradient backend
semantic. Although the SDK declares `softMaxGradientWithIncomingGradient:sourceTensor:axis:name:`,
Task 0026 must not call it, add a gradient wire, recognize/fuse the formula, save hidden forward
state, or change Compiler. Strict Metal does not own binary arithmetic or SUM, so Task 0026 claims
strict SOFTMAX forward only, not a strict all-Metal backward graph. Accelerator gradient results are
governed node-by-node by existing SOFTMAX, binary, and reduction contracts; no new fused-gradient
accuracy promise is introduced.

## Native schema and identity cutover

- Retain native ABI version 4, every C signature/status, the fixed 160-byte node record, and exactly
  thirteen exports.
- Evolve node schema `11 -> 12`; retain operation wires `1..19` and append only `SOFTMAX=20`.
- Reuse typed `AXIS=3`. Encode one semantic axis field with `attributeCount=1`, absent second-input
  sentinel, zero auxiliary, and every attribute-value/padding cell zero. Add no attribute kind.
- Define Java `NodeKind.SOFTMAX(20, 1, AXIS, CANONICAL, false)` and lower through the existing
  axis-node constructor. Java and native schema/preflight independently authenticate kind,
  profile, FLOAT32 types, canonical states, one-input topology, equal positive Shapes, rank, axis,
  and all zero/sentinel cells before graph construction.
- Native lowering calls only
  `[graph softMaxWithTensor:first axis:(NSInteger)node.axis name:nil]` and verifies the exact output
  Shape before compilation.
- Reject unknown wire `21+`, stale schema 11, wrong discriminator/count/sentinel/padding, profile,
  type/state/topology/Shape/rank/axis, and overflow before resource creation.
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and tuning-codec identities together from version 12 to 13. No alias or migration
  reader remains. Identity must distinguish normalized SOFTMAX axis in addition to existing
  profile, schema, topology, descriptors/states, feeds/targets, ABI, and splat facts.
- Keep the existing whole-partition MPSGraph candidate, direct target publication, binding order,
  leases, rollback, reuse, concurrency, close/quiescence, cleanup, and session isolation. Add no
  workspace, custom kernel, late route, retry, host result, or CPU fallback.

## Exact production and proof touchpoints

Expected Metal-private production scope:

- `MetalCapabilityProvider.java`;
- `MetalMpsGraphProgram.java`;
- `MetalNativeApi.java`;
- `MetalNegPartitionPreparer.java` — handle SOFTMAX before its trailing generic axis-transform cast;
- `MetalNegRouteCandidateGenerator.java`, `MetalNegTuningBatch.java`, and
  `MetalNegTuningCodec.java` for the version-13 cutover;
- Metal `package-info.java`; and
- `native/metal-macos-arm64/src/synaptik_metal_foundation.m`.

Compiler, Model, Config, Planning, Prepare, Runtime, Backend Contract, Trace, Engine, Training, CPU,
and transfer production are read-only.

Lean permanent proof:

- extend `MetalCapabilityProviderTest` with one common-profile valid occurrence, one positive
  selected-width-17 occurrence proving there is no selector-expansion cap, and independent exact
  exclusions for LOG_SOFTMAX/kind, type, rank/axis/Shape/layout/gradient state;
- extend `MetalMpsGraphRawAbiNativeTest` for schema 12, wire 20, AXIS 3, exact 160-byte cells,
  malformed/stale rejection, and native/Java parity;
- add one focused `MetalMpsGraphSoftmaxNativeTest` as the numerical owner for the frozen finite
  corpus, four-ULP oracle, `[0,1]` classification, output freshness, and input preservation;
- update current schema assertions without duplicating generic lifecycle matrices;
- extend `MetalNegRouteCandidateGeneratorTest` for axis-separated identity and version-12
  rejection;
- extend one backend-conformance maximal-region test with the accelerator generated-gradient
  formula; and
- extend one CPU-free public Engine integration method for both-profile forward plus one
  ACCELERATOR explicitly seeded first-order gradient. Assert numerical outputs/gradient and Metal
  ownership; do not add a direct gradient route test or claim strict backward closure.

## Documentation at implementation

After executable behavior stabilizes, synchronize only current claims in `ARCHITECTURE.md`, Tensor
and Compile API references, backend-execution and module-boundary contracts, Metal backend and
partition-preparer guides, glossary, Metal native README, Metal package documentation, this task,
Metal master plan, and roadmap. Do not edit Model's semantic plan to present the Metal-private
finite policy as public semantics.

## Explicit exclusions

- `SoftmaxKind.LOG_SOFTMAX`: the active SDK exposes no direct log-softmax selector; SOFTMAX followed
  by LOG is not the first-class stable log-probability contract, and its generated gradient needs
  currently unsupported EXP.
- Layer normalization and RMS normalization: their exact trailing normalized Shape, epsilon,
  population/mean-square accumulation, constant-slice, NaN/infinity, signed-zero, and overflow
  contracts are not established by selector headers; generated gradients also require unsupported
  scalar arithmetic and RSQRT closure.
- Batch normalization: inference is a wider five-input arbitrary-channel slice and training has
  five outputs, while current Metal preparation admits one output per node.
- BFLOAT16, FLOAT64, FLOAT16, integral/BOOL, dynamic, scalar, zero-extent, rank-above-16, affine-view,
  noncanonical, aliased, or cross-type forms.
- Masks, scales, temperature, bias, causal forms, losses, attention, sparse/adaptive variants,
  decomposed-softmax recognition, fusion, direct gradient selector, hidden saved state, or training
  API.
- Model/Compiler/shared semantic changes, accelerator normalization relaxation, DAZ/FTZ permission,
  special-value promise, custom kernel, workspace, transfer widening, retry, fallback, or
  performance claim.

## Acceptance and review

- The one mandatory probe either passes the exact finite gate before production or blocks the task
  without production changes.
- Capability and execution admit exactly the canonical FLOAT32 SOFTMAX domain under both profiles,
  with no selected-axis upper bound beyond positive checked geometry and the independent rank-16
  ABI limit.
- Direct finite forward output satisfies the frozen four-ULP Metal-private oracle, remains finite
  and in `[0,1]`, preserves input bits, and publishes fresh canonical storage.
- ACCELERATOR explicit seeded backward closes as one Metal partition through the unchanged Compiler
  formula and current MUL/SUB/SUM routes. No backend gradient semantic or strict backward claim is
  added.
- Schema 12/wire 20/AXIS 3 and version-13 identities fail closed while ABI 4, 160-byte records,
  thirteen exports, ingress/publication, FLOAT32 transfer, and lifecycle remain exact.
- LOG_SOFTMAX and every general normalization remain closed.
- Focused proof, native export/link audit, architecture checks, Javadoc, one final build,
  synchronized current documentation, clean diff, and independent lean Class C `APPROVE` with zero
  findings close the task.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
- [Model softmax semantics](../../../modules/model/tasks/0016i-softmax-semantic-kinds-and-attributes.md)
- [Model softmax Tensor expressions](../../../modules/model/tasks/0016j-softmax-tensor-expressions.md)
- [Compiler softmax inference and gradients](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [CPU stable softmax precedent](../../../backends/cpu/tasks/0007e-portable-stable-softmax-and-log-softmax-coverage.md)

## Architecture impact

Expected impact: None. This task extends only the existing backend-private typed MPSGraph program,
capability, native lowering, identities, and tests. Stop and replan if implementation requires a
new public/shared semantic, dependency, module, resource kind, profile relaxation, or fallback.

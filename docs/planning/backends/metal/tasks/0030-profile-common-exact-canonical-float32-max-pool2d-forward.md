# Task 0030: Profile-Common Exact Canonical FLOAT32 MAX_POOL2D Forward

## Status

Ready

## Change class

Class C — this task adds a numerical operation to both Metal profile matrices, evolves the
backend-private native schema and route identities, and changes the generated native graph. Use one
clean implementation context and an independent lean Class C review.

## Goal

Implement only first-class `Pool2dKind.MAX_POOL2D` forward execution for canonical, positive, fully
static rank-four NCHW `FLOAT32` tensors under both numerical profiles. Preserve every Model-valid
checked geometry and exact maximum-selection semantic while gradient-bearing occurrences fail closed.

## Scope

- Admit exactly `MAX_POOL2D + MaxPool2dAttrs`, one input and one output, when both descriptors are
  canonical dense contiguous, zero-offset, non-view `FLOAT32` NCHW rank-four tensors with positive static extents and `requiresGrad=false`.
- Independently rederive checked `[N,C,Hout,Wout]` geometry and require exact output Shape, type,
  canonical layout, and no-gradient metadata. Admit the complete documented Model-valid attribute domain that fits checked Java/native geometry; add no probe-derived limit.
- Lower one native node through `maxPooling2DWithSourceTensor:descriptor:name:` and
  `MPSGraphPooling2DOpDescriptor`, with `MPSGraphPoolingReturnIndicesNone` and one result.
- Add schema 12, wire `MAX_POOL2D=20`, discriminator `POOL2D=7`, and version-thirteen identities
  while retaining ABI 4, the 160-byte record, and exactly thirteen exports.
- Keep current whole-partition MPSGraph preparation, storage, publication, reuse, rollback, concurrency, cleanup, and session isolation unchanged.

## Non-goals

- `AVERAGE_POOL2D`; Conv1d/2d/3d; Pool1d or Pool3d; adaptive/global pooling; another layout or type
- `requiresGrad=true`, pooling backward, gradient selectors, indices, a second output, unpooling, recognition, fusion, or hidden state
- affine/view inputs, dynamic or zero extents, noncanonical layouts, transfer widening, workspace, custom kernels, fallback, retries, performance claims, or shared production changes
- a geometry, padding, dilation, ceil, rank, profile, optimization, context, or repetition device matrix; documented structural mappings are authorized and tested without numerical probing

## Contracts

- [`ARCHITECTURE.md` — Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence)
  requires plans to preserve the root and incorporated scoped contracts.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  leaves pooling unrelaxed; strict and accelerator therefore use the identical Model allowed set.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  MPSGraph lowering, native integration, storage, and materialization.
- [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  requires profile-qualified capability and compatible route identities.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps lowering in Metal analysis/finalization and Runtime free of route selection.
- [Model MAX_POOL2D](../../../modules/model/tasks/0020a-nchw-max-pool2d-semantics-and-tensor-expression.md)
  owns NCHW geometry, padding exclusion, all-padding negative infinity, NaN, signed-zero, and
  first-logical-winner meaning.

If implementation needs a profile relaxation, public/shared semantic, module edge, new resource
kind, multi-output lifecycle, or fallback, stop and replan.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0020A; Compiler 0005D/0006B11; CPU 0008G exact-policy precedent; Config 0006; Engine 0018; current Metal lifecycle.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/materialization/public Engine scope; any 0016–0018 restart; convolution, average/1D/3D pooling, backward, multi-output, or shared semantic work.
- Parallel group: None.
- Common base revision: `030d4c561cc1a836d94a8c01c41ffef86a8c807b`; execution starts only from the committed planning revision containing this brief.
- Integration order: sole one-execution raw-winner gate; schema/identity; capability and Java/native preflight; native lowering; focused proof; current docs; independent review.
- Integration validation: focused Metal/conformance/one real-Engine checkpoint, native export audit, architecture checks, Javadoc, one final build, Markdown and diff validation.
- Shared-document integration owner: Metal 0030 implementer.

All declared predecessors are Complete, blocked 0026/0027 have no active write scope, this contract
is independent of their selectors, and 0030 is the sole authorized Metal frontier.

## Documented structural authorization

The installed `MPSGraphPoolingOps.h` documents a rank-four source, descriptor-owned layout, a
one-result max-pooling selector, kernel, X/Y stride and dilation, explicit four-side padding,
`ceilMode`, and optional indices. `MPSGraphCore.h` defines `NCHW` and explicit padding.

Map Model width to X, height to Y, and symmetric padding to equal left/right and top/bottom values.
Set explicit padding, NCHW, exact geometry, `ceilMode`, and no indices. These declarations plus
Model's formula authorize all admitted geometry. Java/native preflight independently recomputes
effective kernels, padded numerators, literal floor/ceil output extents, conversions, counts, and
exact output Shape. Structural tests cover non-square/padded/dilated/ceil/all-padding geometry without a device matrix.

The one-output selector matches Model's 1 -> 1 signature. Compiler reconstructs max-pool winners
through ordinary public operations and exposes no saved index result. Do not call the return-indices
or gradient selectors.

## Exact numerical contract

Padding is excluded; an all-padding window returns exact negative infinity. Otherwise NaN wins,
then ordinary maximum applies with positive zero above negative zero. Equal candidates keep the
first height-then-width sample. Preserve the selected non-NaN `FLOAT32` word exactly.

Model does not promise NaN payload, sign, or signaling preservation. Require NaN class for a
selected quiet or signaling NaN without inventing payload identity. Multiple-NaN first-winner order
remains Model/Compiler gradient policy, is not distinguishable in allowed forward bits, and justifies no extra probe.

Pooling is unlisted in the accelerator-relaxation table. Both profiles use this exact policy: no
DAZ, FTZ, tolerance, alternate signed zero, or finite-value approximation.

## Sole pre-production numerical gate

Before any production edit, run one disposable Objective-C program exactly once: one process, one
device/queue context, graph, executable, direct max-pool node, target, and synchronous execution.
Use production default compilation settings once; vary no profile, optimization, option, context, binding, or repetition.

Use canonical input `[1,1,1,18]`, output `[1,1,1,9]`, and attrs
`(kernelHeight=1, kernelWidth=2, strideHeight=1, strideWidth=2, paddingHeight=0,
paddingWidth=0, dilationHeight=1, dilationWidth=1, ceilMode=false)`. Its nine disjoint windows are:

```text
[1, 2]
[-0, +0]
[+0, -0]
[+0, +minSubnormal]
[-1, -minSubnormal]
[qNaN-A, 3]
[3, qNaN-B]
[sNaN-A, 3]
[3, sNaN-B]
```

Freeze `qNaN-A=0x7fc12345`, `qNaN-B=0xffc23456`, `sNaN-A=0x7fa12345`, and
`sNaN-B=0xffa23456`. Read every output raw word.
Require exact `2`, positive-zero, positive-zero, positive minimum subnormal, and negative minimum
subnormal words for the first five windows; require NaN class for each last-four word because Model
leaves payload/sign/signaling unspecified. Require exact output Shape, unchanged input words, and no
guard corruption. This defends ordinary selection; signed-zero ordering in both operand orders;
positive/negative subnormal comparison and selected-word preservation; quiet/signaling NaN
propagation from first and later positions; source immutability; and exact publication.

Do not add padding/dilation/ceil cases or any matrix/repeat. On any failure, remove source/binary,
mark 0030 Blocked with exact evidence, and make no production/schema/capability change. Do not
narrow geometry or values, weaken the oracle, move pooling to accelerator-only, use indices,
decompose, retry, or add fallback. A pass authorizes implementation; independent review reuses it
unless it identifies a new concrete risk and first replans the one-execution policy.

## Typed schema and identity cutover

- Append `NodeKind.MAX_POOL2D(20, 1, POOL2D, CANONICAL, false)` and `AttributeKind.POOL2D(7)`;
  advance node schema `11 -> 12` and retain wires `1..19` unchanged.
- Encode `first_input=input`, absent second-input sentinel, `output=output`, absent axis sentinel,
  `auxiliary=ceilMode ? 1 : 0`, `attributeCount=8`, and attribute values in this order:
  kernel height/width, stride height/width, padding height/width, dilation height/width. Require every
  remaining cell zero.
- Java and native validation independently authenticate kind/attrs/profile, one-input/one-output
  topology, canonical state, FLOAT32, rank four, positive static Shapes, no-grad flags, geometry,
  exact output Shape, sentinels, payload cells, and checked native conversions before graph creation.
- Advance workload signature, exact default policy, candidate schema, compatibility schema,
  route-policy, and codec together `12 -> 13`. Identity includes profile, schema, operation/attribute
  wires, all geometry plus ceil mode, topology, descriptors/states, feeds/targets, ABI, and splats.
  Version 12 safely misses or rejects; add no alias or migration.
- Retain ABI 4, 160 bytes, thirteen exports, the current MPSGraph candidate, and all existing
  resource/lifecycle rules.

## Files and symbols

Production remains Metal-private: `MetalCapabilityProvider`, `MetalMpsGraphProgram` (`NodeKind`,
`AttributeKind`, `Node`, `SCHEMA_VERSION`), `MetalNativeApi.MpsGraphExecutableAbi`,
`MetalNegPartitionPreparer`, `MetalNegRouteCandidateGenerator`, `MetalNegTuningBatch`,
`MetalNegTuningCodec`, Metal `package-info.java`, and
`native/metal-macos-arm64/src/synaptik_metal_foundation.m`.

Extend existing capability, raw-ABI, prepared-execution, identity/codec,
`MetalNegCapabilityPartitionConformanceTest`, and `EngineExplicitCompositionMetalIntegrationTest`
owners. Add one focused native max-pool numerical owner only if no existing native test can own the
single frozen corpus without duplicating lifecycle matrices. Model, Compiler, CPU, Config, Planning,
Prepare, Runtime, Backend Contract, Trace, Engine, Training, and transfer production are read-only.

## Acceptance criteria

1. The sole gate passes before production edits or blocks the task without them.
2. Both profiles admit exactly the no-grad canonical positive static rank-four FLOAT32 domain and
   every documented valid attribute geometry; all exclusions reject before native graph creation.
3. Direct one-output lowering preserves the exact Model result policy without returned indices.
4. Schema 12/wire 20/POOL2D 7 and identities 13 fail closed while ABI 4, 160 bytes, thirteen exports,
   transfer, publication, and lifecycle remain unchanged.
5. Focused structural, ABI, capability, conformance, prepared-lifecycle, and one CPU-free public
   Engine proof pass without another winner corpus or device matrix.
6. Current documentation, clean diff, one final build, and independent lean Class C review pass.

## Validation

Worker validation uses the task-owned one-run gate, then focused Metal capability/raw-ABI/prepared/
identity suites, `MetalNegCapabilityPartitionConformanceTest`, one ordinary CPU-free
`EngineExplicitCompositionMetalIntegrationTest` forward, native build/export audit, Metal Javadoc,
architecture tests, and `git diff --check`. Run `./gradlew build` exactly once after stabilization.
The independent review reuses successful executable evidence and performs no device rerun unless a
new concrete risk first changes this plan.

## Documentation and review impact

After executable behavior stabilizes, update current Metal/backend/API/glossary status only where
implemented behavior changes it. Model and Compiler semantics remain unchanged. Class C requires an
independent targeted implementation/documentation review.

## Result

Empty until execution.

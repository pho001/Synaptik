# Task 0033: Profile-Common Canonical FLOAT32 No-Grad Unmasked Noncausal Default-Scale SDPA Forward

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add first-class scaled dot-product attention (SDPA) to
both Metal profile matrices, evolve backend-private schema and route identities, and change native
graph construction. This planning-only blocker changes no executable behavior and ran no probe.

## Goal

Assess the smallest coherent first-class Metal attention slice: canonical positive static
rank-four `FLOAT32`, no-grad, unmasked, noncausal, default-scale, one-output
`SCALED_DOT_PRODUCT_ATTENTION` under both numerical profiles. MPSGraph exposes a direct selector,
but attention has no operation-specific `ACCELERATOR` relaxation. Its documentation leaves
independent FTZ, accumulation, stable-softmax, special-class, and shape-algorithm questions. One
execution cannot close them and the required matrix is prohibited, so Task 0033 is Blocked before
any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` with
  `new ScaledDotProductAttentionAttrs(Optional.empty(), false)`, ordered
  `[query, key, value]`, and only output slot zero.
- Report identical exact capability under `STRICT_IEEE` and `ACCELERATOR`; accelerator adds no
  attention result freedom and may not remove strict capability.
- Require `FLOAT32`, `requiresGrad=false`, fully static canonical dense-contiguous zero-offset
  non-view descriptors with exact Shapes:

  ```text
  query [B,H,L,E]   key [B,H,S,E]   value [B,H,S,E]   output [B,H,L,E]
  ```

- Require strictly positive static `B/H/L/S/E` with checked conversions. Rank is exactly four;
  batch/head broadcasting and `Ev != E` are excluded beyond the documented selector surface.
- Resolve absent scale once from positive static `E`. Implementation must prove the single
  FLOAT32 conversion from semantic `1 / sqrt(E)`; Metal receives no new rounding freedom.
- Preserve whole-partition lifecycle, fresh canonical non-aliasing output, FLOAT32-only transfer,
  and the profile-free Runtime/Trace boundary.

## Non-goals

- Explicit BOOL/additive mask, causal mode, explicit scale, batch/head expansion, grouped-query,
  multi-query, `Ev != E`, other ranks, or empty/dynamic extents.
- Weights/two-output publication, saved intermediates, `requiresGrad=true`, pullbacks, or training.
- Dropout, random state, cache, sinks, bias, packed/sparse attention, quantization, other types,
  mixed promotion, casts, or views.
- Recognizing MATMUL/mask/SOFTMAX/MATMUL composition; borrowed row freedoms; host repair, fallback,
  retry, tolerance, or value-dependent capability.
- Any Shape/value/option/optimization matrix or narrowing to one passing geometry.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  leaves attention unrelaxed under `ACCELERATOR`; other-row permissions are not compositional.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route/resource selection cold and Runtime free of operation-policy lookup.
- [Model attention](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md) and
  [weights output](../../../modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md)
  own kind, attributes, Shape, mask, arity, numerical, and metadata contracts.
- [Compiler attention gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
  owns pullbacks; [CPU attention](../../cpu/tasks/0008h-portable-scaled-dot-product-attention-execution.md)
  is atomic-policy precedent, not Metal numerical authorization.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019E/0023F; Compiler 0005D/0006B11; CPU 0008H precedent;
  Config 0006; Engine 0018. Metal 0021/0026 are evidence, not functional dependencies.
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/
  materialization/public Engine scope; attention masks/causal/weights/backward; any 0016–0018 restart.
- Parallel group: None.
- Common base revision: `a4784bc9f1c55a747af4fc72e32690370e0587c5`.
- Integration order: no implementation before a recorded complete-domain unblock.
- Integration validation: Markdown/link/anchor/fence/whitespace validation and `git diff --check`;
  no executable validation or probe while Blocked.
- Shared-document integration owner: Task 0033 planner; later Class C work needs a new owner/review.

## Exact Model, Compiler, and CPU boundary

Model's absent scale means `1 / sqrt(E)` after positive binding and `causal=false` adds no
restriction. Output zero is `[...,L,Ev]`, promoted query/key/value type, unresolved construction
layout, and query/key/value gradient-request OR. This candidate fixes every descriptor to
`FLOAT32`, rank four, same batch/head, `Ev=E`, and no-grad.

Model requires FLOAT32 score/output accumulation while permitting corresponding FMA,
reassociation, and stable softmax that preserves special cases: eligible NaN produces NaN,
positive-infinity ties split unit weight, all-negative-infinity produces positive-zero output, and
eligible zero-weight value NaN participates. These atomic attention rules are not composed rows.

Compiler gradients require the exact two-output occurrence and canonical weights in slot one, so
one-output fails closed and this candidate requires all descriptors `requiresGrad=false`. CPU
0008H likewise executes one atomic schema row and forbids recognizing a decomposition, but its
algorithm and evidence do not prove MPSGraph.

## Installed MPSGraph surface and availability

The macOS 27.0 SDK's `MPSGraphMatrixMultiplicationOps.h` declares output-only direct SDPA
selectors. The unmasked `scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:`
selector and additive-mask sibling are available from macOS 15.0 and document
`softmax(scale * QK^T + M)V` for rank-four `[B,H,N,F]`, but no weights or numerical contract.

`MPSGraphSDPADescriptor` and its selector require macOS 27.0. The runtime is macOS 26.6.2, while
the native bridge records minimum macOS 26.0 with SDK 27.0, so that path is unavailable here. A
conditional implementation may use only the macOS-15 unmasked selector and must fail before graph
construction when unavailable.

Both mask surfaces mismatch Model: MPSGraph documents an additive numeric mask and the macOS-27
descriptor makes mask and causal mutually exclusive. Model requires exact BOOL selection, permits
explicit-mask/causal AND, and excludes masked specials before arithmetic. No sentinel/composed
conversion may be inferred; this candidate is unmasked and noncausal.

## Documentation-first blocker

Attention is unrelaxed in the normative profile table. Metal 0021 MATMUL, 0015/0020
binary/reduction, and blocked 0026 standalone SOFTMAX permissions are operation-scoped. The direct
selector avoids structural composition but inherits none of their result freedoms.

The direct-selector headers independently omit:

1. DAZ/FTZ behavior for both contractions, scaling, exponential, probability, and publication;
2. FLOAT32 accumulation, term membership, intermediate format, reduced precision, reassociation,
   and corresponding-FMA guarantees;
3. a stable-softmax finite error/result-set bound;
4. eligible NaN, positive-infinity tie, all-negative-infinity, signed-zero, and zero-weight
   exceptional-value guarantees; and
5. one numerical envelope across every positive `B/H/L/S/E` and opaque shape-dependent algorithm.

`MPSGraphReducedPrecisionFastMathNone` closes none of these. Blocked 0026 direct SOFTMAX also
returned zero where its finite oracle required subnormal `0x0008ec28`; that is relevant evidence,
not direct SDPA proof. One execution proves at most one property for one compiled geometry, not the
other independent gaps or algorithms. The prohibited matrix and arbitrary one-geometry narrowing
are not routes. No probe ran and no schema, identity, capability, lowering, native, test, or source
change exists.

## Conditional schema only

Task 0033 changes and reserves no schema. A separately authorized direct-selector implementation
could advance node schema `11 -> 12`, append `SCALED_DOT_PRODUCT_ATTENTION=20`, add
`AttributeKind.ATTENTION=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and codec identities together `12 -> 13`, retaining wires `1..19`, ABI v4, the
160-byte record, and thirteen exports.

Its conditional record uses `firstInput=query`, `secondInput=key`, `auxiliary=value`, one output,
`attributeCount=2`, absent-axis sentinel, values `[0]=0` for default scale and `[1]=0` for
noncausal, and zero cells `2..15`. Java/native preflight rejects every other form; native
independently validates canonical inputs, exact rank-four Shapes/output, sentinels, checked scale
conversion, availability, and result Shape/type. `requiresGrad` stays Java/workload-only.

While Blocked, schema 11, wires `1..19`, attribute kinds `0..6`, version-twelve identities, ABI v4,
and thirteen exports remain unchanged and unreserved. A custom kernel needs its own typed route.

## Exact unblocking options

Unblocking requires one separately authorized complete-domain path:

1. authoritative Apple documentation closing FLOAT32 accumulation/reduced precision, DAZ/FTZ,
   finite stable-softmax accuracy, special classes, and shape-algorithm applicability for the exact
   unmasked selector; only one genuinely remaining property may receive a later one-run gate;
2. an exact custom Metal kernel with complete proof of scale derivation, all contraction terms,
   gradual underflow, stable normalization, special classes, and every positive candidate
   geometry—ordinary MSL float is insufficient under retained FTZ evidence; or
3. a preceding Model/architecture decision defining an attention-specific accelerator envelope
   covered by a documented/proved implementation without importing standalone row relaxations.

Afterward, structural proof must cover duplicate Java/native arity, type, Shape, layout, payload,
availability, conversion, alias, result Shape, stale identity, lifecycle, and CPU-free Engine
checks; it cannot substitute for numerical proof.

## Acceptance criteria and validation

- Record the exact candidate, exclusions, runtime/mask boundaries, profile rule, independent gaps,
  unblocking paths, and unreserved conditional schema with no executable changes.
- Keep Task 0033 `Blocked`, expose no Ready Metal frontier, and preserve earlier evidence.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, and
  final diff. Run no probe, test, formatter, build, schema reservation, or production edit.

## Documentation and review impact

Only this task, the Metal master plan, and roadmap change. Architecture, APIs, backend/native
guides, glossary, Javadoc, and sources remain accurate because behavior does not change. Later
Class C implementation needs synchronized documentation and independent targeted review.

## Result

Planning from clean `a4784bc9f1c55a747af4fc72e32690370e0587c5` found the direct selector structurally usable only
for the narrow unmasked output form, but its documentation leaves multiple independent exact
attention gaps. No probe ran and no executable file changed. Task 0033 is Blocked; no Metal task is
Ready.

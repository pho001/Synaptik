# Task 0033: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) No-Grad Unmasked Noncausal Default-Scale SDPA Forward

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add first-class [scaled dot-product attention](../../../../glossary.md#scaled-dot-product-attention) (SDPA) to
both [Metal](../../../../glossary.md#term-metal) profile [matrices](../../../../glossary.md#term-matrix-meanings), evolve [backend](../../../../glossary.md#backend)-private [schema](../../../../glossary.md#term-schema) and route [identities](../../../../glossary.md#term-identity-meanings), and change native
[graph](../../../../glossary.md#graph) construction. This [planning](../../../../glossary.md#term-planning)-only blocker changes no executable behavior and ran no probe.

## Goal

Assess the smallest coherent first-class [Metal](../../../../glossary.md#term-metal) attention [slice](../../../../glossary.md#slice): [canonical](../../../../glossary.md#term-canonical-form) positive static
[rank](../../../../glossary.md#term-rank)-four [`FLOAT32`](../../../../glossary.md#term-float32), no-grad, unmasked, noncausal, default-scale, one-output
`SCALED_DOT_PRODUCT_ATTENTION` under both numerical profiles. [MPSGraph](../../../../glossary.md#term-mpsgraph) exposes a direct selector,
but attention has no [operation](../../../../glossary.md#operation)-specific [`ACCELERATOR`](../../../../glossary.md#term-accelerator) relaxation. Its documentation leaves
independent [FTZ](../../../../glossary.md#term-ftz), accumulation, stable-[softmax](../../../../glossary.md#softmax--log-softmax), special-class, and [shape](../../../../glossary.md#shape)-algorithm questions. One
execution cannot close them and the required [matrix](../../../../glossary.md#term-matrix-meanings) is prohibited, so Task 0033 is Blocked before
any probe or production edit.

## Candidate domain if unblocked

- Admit exactly `ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION` with
  `new ScaledDotProductAttentionAttrs(Optional.empty(), false)`, ordered
  `[query, key, value]`, and only output slot zero.
- Report identical exact capability under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); accelerator adds no
  attention result freedom and may not remove strict capability.
- Require [`FLOAT32`](../../../../glossary.md#term-float32), `requiresGrad=false`, fully static canonical dense-contiguous zero-offset
  non-view descriptors with exact Shapes:<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shapes glossary](../../../../glossary.md#shape))</span>

  ```text
  query [B,H,L,E]   key [B,H,S,E]   value [B,H,S,E]   output [B,H,L,E]
  ```

- Require strictly positive static `B/H/L/S/E` with checked conversions. Rank is exactly four;
  batch/head broadcasting and <span data-glossary-navigation="true"> ([Rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>`Ev != E` are excluded beyond the documented selector surface.
- Resolve absent scale once from positive static `E`. Implementation must prove the single
  FLOAT32 conversion from semantic `1 / sqrt(E)`; [Metal](../../../../glossary.md#term-metal) receives no new [rounding](../../../../glossary.md#term-rounding) freedom.
- Preserve whole-partition lifecycle, fresh canonical non-aliasing output, FLOAT32-only transfer,
  and the profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lifecycle glossary](../../../../glossary.md#lifecycle); [aliasing glossary](../../../../glossary.md#term-aliasing); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>

## Non-goals

- Explicit BOOL/additive mask, causal mode, explicit scale, batch/head expansion, grouped-query,
  multi-query, <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>`Ev != E`, other ranks, or empty/dynamic extents.
- [Weights](../../../../glossary.md#term-weight)/two-output [publication](../../../../glossary.md#term-publication), saved intermediates, `requiresGrad=true`, pullbacks, or training.
- Dropout, random state, cache, sinks, bias, packed/sparse attention, quantization, other types,
  mixed promotion, casts, or views.<span data-glossary-navigation="true"> ([Dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [cache glossary](../../../../glossary.md#term-cache); [bias glossary](../../../../glossary.md#term-bias); [views glossary](../../../../glossary.md#term-view-meanings))</span>
- Recognizing MATMUL/mask/SOFTMAX/MATMUL composition; borrowed row freedoms; host repair, fallback,
  retry, tolerance, or value-dependent capability.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [host glossary](../../../../glossary.md#term-host); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
- Any [Shape](../../../../glossary.md#shape)/value/option/optimization [matrix](../../../../glossary.md#term-matrix-meanings) or [narrowing](../../../../glossary.md#term-narrowing) to one passing geometry.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [owner glossary](../../../../glossary.md#term-owner); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  leaves attention unrelaxed under [`ACCELERATOR`](../../../../glossary.md#term-accelerator); other-row permissions are not compositional.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  keeps route/resource selection cold and Runtime free of operation-policy lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Model attention](../../../modules/model/tasks/0019e-scaled-dot-product-attention.md) and
  [weights output](../../../modules/model/tasks/0023f-scaled-dot-product-attention-weights-output.md)<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight))</span>
  own kind, attributes, Shape, mask, arity, numerical, and metadata contracts.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [mask glossary](../../../../glossary.md#term-boolean-mask); [arity glossary](../../../../glossary.md#term-arity); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- [Compiler attention gradients](../../../modules/compiler/tasks/0005d-attention-convolution-pooling-and-loss-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  owns pullbacks; [CPU attention](../../cpu/tasks/0008h-portable-scaled-dot-product-attention-execution.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  is atomic-policy precedent, not Metal numerical authorization.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019E/0023F; Compiler 0005D/0006B11; CPU 0008H precedent;
  Config 0006; Engine 0018. Metal 0021/0026 are evidence, not functional dependencies.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal capability/schema/native-preflight/candidate/codec/
  materialization/public Engine scope; attention masks/causal/weights/backward; any 0016–0018 restart.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [masks glossary](../../../../glossary.md#term-boolean-mask); [weights glossary](../../../../glossary.md#term-weight))</span>
- Parallel group: None.
- Common base revision: `a4784bc9f1c55a747af4fc72e32690370e0587c5`.
- Integration order: no implementation before a recorded complete-domain unblock.
- Integration [validation](../../../../glossary.md#term-validation): Markdown/link/anchor/fence/whitespace validation and `git diff --check`;
  no executable validation or probe while Blocked.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0033 planner; later Class C work needs a new owner/review.

## Exact [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), and [CPU](../../../../glossary.md#term-cpu) boundary

Model's absent scale means `1 / sqrt(E)` after positive [binding](../../../../glossary.md#term-binding) and `causal=false` adds no
restriction. Output zero is `[...,L,Ev]`, promoted query/key/value type, unresolved construction
[layout](../../../../glossary.md#layout), and query/key/value [gradient](../../../../glossary.md#term-gradient)-request OR. This candidate fixes every [descriptor](../../../../glossary.md#term-descriptor) to
[`FLOAT32`](../../../../glossary.md#term-float32), [rank](../../../../glossary.md#term-rank) four, same [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/head, `Ev=E`, and no-grad.

Model requires FLOAT32 score/output accumulation while permitting corresponding [FMA](../../../../glossary.md#term-fused-multiply-add-fma),
[reassociation](../../../../glossary.md#term-reassociation), and stable [softmax](../../../../glossary.md#softmax--log-softmax) that preserves special cases: eligible [NaN](../../../../glossary.md#term-nan) produces NaN,
positive-[infinity](../../../../glossary.md#term-infinity) ties split unit [weight](../../../../glossary.md#term-weight), all-negative-infinity produces positive-zero output, and
eligible zero-weight value NaN participates. These [atomic](../../../../glossary.md#resource-transaction-and-rollback) attention rules are not composed rows.

Compiler gradients require the exact two-output [occurrence](../../../../glossary.md#term-occurrence) and [canonical](../../../../glossary.md#term-canonical-form) weights in slot one, so
one-output fails closed and this candidate requires all descriptors `requiresGrad=false`. CPU
0008H likewise executes one atomic [schema](../../../../glossary.md#term-schema) row and forbids recognizing a [decomposition](../../../../glossary.md#term-decomposition), but its
algorithm and evidence do not prove [MPSGraph](../../../../glossary.md#term-mpsgraph).

## Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) surface and availability

The macOS 27.0 [SDK](../../../../glossary.md#term-sdk)'s `MPSGraphMatrixMultiplicationOps.h` declares output-only direct SDPA
selectors. The unmasked `scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:`
selector and additive-[mask](../../../../glossary.md#term-boolean-mask) sibling are available from macOS 15.0 and document
`softmax(scale * QK^T + M)V` for [rank](../../../../glossary.md#term-rank)-four `[B,H,N,F]`, but no [weights](../../../../glossary.md#term-weight) or numerical contract.

`MPSGraphSDPADescriptor` and its selector require macOS 27.0. The [runtime](../../../../glossary.md#term-runtime) is macOS 26.6.2, while
the [native bridge](../../../../glossary.md#term-native-bridge) records minimum macOS 26.0 with SDK 27.0, so that path is unavailable here. A
conditional implementation may use only the macOS-15 unmasked selector and must fail before [graph](../../../../glossary.md#graph)
construction when unavailable.

Both mask surfaces mismatch [Model](../../../../glossary.md#term-model): MPSGraph documents an additive numeric mask and the macOS-27
[descriptor](../../../../glossary.md#term-descriptor) makes mask and causal mutually exclusive. Model requires exact [BOOL](../../../../glossary.md#data-type--datatype) selection, permits
explicit-mask/causal AND, and excludes masked specials before arithmetic. No sentinel/composed
conversion may be inferred; this candidate is unmasked and noncausal.

## Documentation-first blocker

Attention is unrelaxed in the [normative](../../../../glossary.md#term-normative) profile table. [Metal](../../../../glossary.md#term-metal) 0021 [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), 0015/0020
[binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/reduction, and blocked 0026 standalone [SOFTMAX](../../../../glossary.md#softmax--log-softmax) permissions are [operation](../../../../glossary.md#operation)-scoped. The direct
selector avoids structural composition but inherits none of their result freedoms.

The direct-selector headers independently omit:

1. [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) behavior for both contractions, scaling, exponential, probability, and [publication](../../../../glossary.md#term-publication);
2. FLOAT32 accumulation, term membership, intermediate format, reduced precision, reassociation,
   and corresponding-FMA guarantees;<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [precision glossary](../../../../glossary.md#term-precision); [reassociation glossary](../../../../glossary.md#term-reassociation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma))</span>
3. a stable-softmax [finite](../../../../glossary.md#term-finite) error/result-set bound;
4. eligible NaN, positive-infinity tie, all-negative-infinity, signed-zero, and zero-weight
   exceptional-value guarantees; and<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [weight glossary](../../../../glossary.md#term-weight))</span>
5. one numerical envelope across every positive `B/H/L/S/E` and [opaque](../../../../glossary.md#term-opaque) [shape](../../../../glossary.md#shape)-dependent algorithm.

`MPSGraphReducedPrecisionFastMathNone` closes none of these. Blocked 0026 direct SOFTMAX also
returned zero where its finite [oracle](../../../../glossary.md#term-test-oracle) required [subnormal](../../../../glossary.md#term-subnormal) `0x0008ec28`; that is relevant evidence,
not direct SDPA proof. One execution proves at most one property for one compiled geometry, not the
other independent gaps or algorithms. The prohibited [matrix](../../../../glossary.md#term-matrix-meanings) and arbitrary one-geometry [narrowing](../../../../glossary.md#term-narrowing)
are not routes. No probe ran and no [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), capability, [lowering](../../../../glossary.md#lowering), native, test, or source
change exists.

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0033 changes and reserves no schema. A separately authorized direct-selector implementation
could advance [node](../../../../glossary.md#node) schema `11 -> 12`, append `SCALED_DOT_PRODUCT_ATTENTION=20`, add
`AttributeKind.ATTENTION=7`, and advance workload, exact-policy, candidate, compatibility,
route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`, retaining wires `1..19`, [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4, the
160-byte record, and thirteen exports.

Its conditional record uses `firstInput=query`, `secondInput=key`, `auxiliary=value`, one output,
`attributeCount=2`, absent-[axis](../../../../glossary.md#term-axis) sentinel, values `[0]=0` for default scale and `[1]=0` for
noncausal, and zero cells `2..15`. Java/native [preflight](../../../../glossary.md#term-preflight) rejects every other form; native
independently validates [canonical](../../../../glossary.md#term-canonical-form) inputs, exact [rank](../../../../glossary.md#term-rank)-four [Shapes](../../../../glossary.md#shape)/output, sentinels, checked scale
conversion, availability, and result Shape/type. `requiresGrad` stays Java/workload-only.

While Blocked, schema 11, wires `1..19`, attribute kinds `0..6`, version-twelve identities, ABI v4,
and thirteen exports remain unchanged and unreserved. A custom [kernel](../../../../glossary.md#kernel) needs its own typed route.

## Exact unblocking options

Unblocking requires one separately authorized complete-domain path:

1. authoritative Apple documentation closing FLOAT32 accumulation/reduced precision, DAZ/FTZ,
   finite stable-softmax accuracy, special classes, and shape-algorithm applicability for the exact
   unmasked selector; only one genuinely remaining property may receive a later one-run gate;<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [precision glossary](../../../../glossary.md#term-precision); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [finite glossary](../../../../glossary.md#term-finite); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [run glossary](../../../../glossary.md#term-run))</span>
2. an exact custom Metal kernel with complete proof of scale derivation, all contraction terms,
   gradual underflow, stable normalization, special classes, and every positive candidate
   geometry—ordinary MSL float is insufficient under retained FTZ evidence; or<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [kernel glossary](../../../../glossary.md#kernel); [contraction glossary](../../../../glossary.md#contraction-dimension); [gradual underflow glossary](../../../../glossary.md#term-gradual-underflow); [normalization glossary](../../../../glossary.md#term-normalization); [MSL glossary](../../../../glossary.md#term-metal-shading-language-msl))</span>
3. a preceding Model/architecture decision defining an attention-specific accelerator envelope
   covered by a documented/proved implementation without importing standalone row relaxations.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [accelerator glossary](../../../../glossary.md#term-accelerator))</span>

Afterward, structural proof must cover duplicate Java/native [arity](../../../../glossary.md#term-arity), type, [Shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication),
availability, conversion, [alias](../../../../glossary.md#aliasing-and-arena), result Shape, stale [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine)
checks; it cannot substitute for numerical proof.

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

- Record the exact candidate, exclusions, runtime/mask boundaries, profile rule, independent gaps,
  unblocking paths, and unreserved conditional schema with no executable changes.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [mask glossary](../../../../glossary.md#term-boolean-mask); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep Task 0033 `Blocked`, expose no [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier, and preserve earlier evidence.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, and
  final diff. Run no probe, test, formatter, build, schema reservation, or production edit.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

## Documentation and review impact

Only this task, the [Metal](../../../../glossary.md#term-metal) master plan, and roadmap change. Architecture, APIs, [backend](../../../../glossary.md#backend)/native
guides, glossary, [Javadoc](../../../../glossary.md#term-javadoc), and sources remain accurate because behavior does not change. Later
Class C implementation needs synchronized documentation and independent targeted review.

## Result

[Planning](../../../../glossary.md#term-planning) from clean `a4784bc9f1c55a747af4fc72e32690370e0587c5` found the direct selector structurally usable only
for the narrow unmasked output form, but its documentation leaves multiple independent exact
attention gaps. No probe ran and no executable file changed. Task 0033 is Blocked; no [Metal](../../../../glossary.md#term-metal) task is
[Ready](../../../../glossary.md#term-ready).

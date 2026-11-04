# Task 0035: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) No-Grad Explicit-State [Dropout](../../../../glossary.md#dropout--inverted-dropout)

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add explicit-state random execution under both [Metal](../../../../glossary.md#term-metal)
profiles, local [INT64](../../../../glossary.md#data-type--datatype), genuine multi-output [lifecycle](../../../../glossary.md#lifecycle), private [schema](../../../../glossary.md#term-schema)/route [identities](../../../../glossary.md#term-identity-meanings), and an exact
custom native route. This [planning](../../../../glossary.md#term-planning)-only blocker changes no executable behavior and ran no probe.

## Goal

Assess the smallest nondegenerate coherent [Metal](../../../../glossary.md#term-metal) random [slice](../../../../glossary.md#slice): exact `INITIAL_STATE` plus [canonical](../../../../glossary.md#term-canonical-form)
positive static [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [`FLOAT32`](../../../../glossary.md#term-float32), no-grad, explicit-state [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) under both profiles. Direct
[MPSGraph](../../../../glossary.md#term-mpsgraph) dropout hides randomness and returns one output; [opaque](../../../../glossary.md#term-opaque) [Philox](../../../../glossary.md#graph-rng-state--graphrngstate) state does not establish
[Model](../../../../glossary.md#term-model)'s exact `INT64[2]` key/counter, mandatory [mask](../../../../glossary.md#term-boolean-mask)/nextState, or `counter + N`. Current Metal
cannot represent zero-input [nodes](../../../../glossary.md#node), three-output nodes, or [INT64](../../../../glossary.md#data-type--datatype), so Task 0035 is Blocked pre-probe.

## Candidate domain if unblocked

- Admit exact zero-input/one-output `GraphRngKind.INITIAL_STATE` and two-input/three-output
  `DropoutKind.DROPOUT` roles `[value,state] -> [output,keepMask,nextState]`.
- Report identical exact capability under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); random operations receive
  no accelerator relaxation.
- Require [canonical](../../../../glossary.md#term-canonical-form) dense zero-offset non-[view](../../../../glossary.md#term-view-meanings) [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [`FLOAT32`](../../../../glossary.md#term-float32) value/output [Shape](../../../../glossary.md#shape) `[N]`, exact
  local `INT64 Shape[2]` state, same-Shape canonical [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), and distinct non-[aliasing](../../../../glossary.md#term-aliasing) outputs.
- Require `requiresGrad=false` for value/output; mask and both state positions are always
  non-differentiable. No-grad removes backward construction, not either mandatory auxiliary output.<span data-glossary-navigation="true"> ([differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Accept every raw key/counter pair and the complete [finite](../../../../glossary.md#term-finite) [binary64](../../../../glossary.md#floating-point-precision-and-rounding) probability domain `[0,1)`,
  including signed zeros. Each row-major element consumes one draw; next state keeps key and adds<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element))</span>
  `N` to counter modulo `2^64`. One versioned Metal-private exact algorithm/policy must replay the
  same output, mask, and next state from identical inputs without mutable session RNG.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [mutable glossary](../../../../glossary.md#term-mutable); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>

## Non-goals

- A `p=0` identity route, fixed seed, fixed passing Shape, selected probability, selected value
  corpus, or other degenerate special case in place of the complete admitted occurrence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `requiresGrad=true`, mask-consuming pullback, training-session integration, fusion,
  branching/chaining widening, or attention dropout.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [fusion glossary](../../../../glossary.md#term-fusion); [widening glossary](../../../../glossary.md#term-widening); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- FLOAT64, BFLOAT16, integral/BOOL values, scalar/empty/dynamic Shapes, other ranks/layouts, views,
  negative strides, in-place execution, or unresolved geometry.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [scalar glossary](../../../../glossary.md#term-scalar); [layouts glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [strides glossary](../../../../glossary.md#term-stride))</span>
- Other distributions, public algorithm selection, key derivation, split/fold-in, certification,
  cross-backend bitstream, public INT64 ingress/publication/transfer, host repair/fallback/retry, or
  hidden/default/session RNG.<span data-glossary-navigation="true"> ([distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [backend glossary](../../../../glossary.md#backend); [INT64 glossary](../../../../glossary.md#data-type--datatype); [publication glossary](../../../../glossary.md#term-publication); [host glossary](../../../../glossary.md#term-host); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- A [seed](../../../../glossary.md#term-seed), [Shape](../../../../glossary.md#shape), probability, optimization, context, or repetition [matrix](../../../../glossary.md#term-matrix-meanings).

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  give random operations no [`ACCELERATOR`](../../../../glossary.md#term-accelerator) relaxation.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  keeps route/resource selection cold and Runtime free of operation-policy lookup.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [Model](../../../../glossary.md#term-model) [explicit RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
  and [dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span> own state,
  probability, output-role, draw-count, replay, and ideal-value semantics.
- [Compiler stochastic gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradients glossary](../../../../glossary.md#term-gradient))</span>
  own pullbacks; [CPU random coverage](../../cpu/tasks/0006d-portable-explicit-state-rng-and-dropout.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> is exact private precedent only.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019B/0019B1; Compiler 0003/0005C/0006B11; CPU 0006D
  precedent; Config 0006; Engine 0018. Earlier Metal blockers are independent evidence.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/
  local-type/materialization/public Engine scope; multi-output, INT64, random, or gradient work; any
  0016–0018 restart.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [INT64 glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Parallel group: None.
- Common base revision: `43863c4203fadc9281ab27653594f4720eed2639`.
- Integration order: no implementation before an exact custom-kernel and multi-output/INT64 design
  is Ready. While Blocked, validate only Markdown and <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [Ready glossary](../../../../glossary.md#term-ready))</span>`git diff --check`; [run](../../../../glossary.md#term-run) no executable probe.
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0035 planner; later Class C work needs a new owner/review.

## Exact [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), and [CPU](../../../../glossary.md#term-cpu) boundary

`INITIAL_STATE` is `0 -> 1`: its exact output is non-[gradient](../../../../glossary.md#term-gradient) `INT64 Shape[2]` containing raw
unsigned key then counter words. [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) is `2 -> 3`: the value output preserves input floating
type, exact [Shape](../../../../glossary.md#shape), and gradient eligibility; the hidden keep [mask](../../../../glossary.md#term-boolean-mask) is same-Shape [BOOL](../../../../glossary.md#data-type--datatype); next state is
non-gradient `INT64 Shape[2]`. Both auxiliary outputs are mandatory even when only value slot zero
is a public [graph](../../../../glossary.md#graph) output.

For logical [row-major](../../../../glossary.md#term-row-major) position `i`, one abstract `u_i` in `[0,1)` is consumed and keep is exactly
`u_i >= probability`. A dropped result is represented [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) even for [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan),
or [infinity](../../../../glossary.md#term-infinity). A kept [signed zero](../../../../glossary.md#term-signed-zero) or infinity preserves sign; a kept NaN remains NaN without a [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)
promise; [finite](../../../../glossary.md#term-finite) kept values have inverted-dropout meaning `input / (1 - probability)` under a
conforming finite-[precision](../../../../glossary.md#term-precision) realization. Signed `p=0` keeps every [element](../../../../glossary.md#term-element) but still draws `N` times,
creates all outputs, and advances state; it is not an [identity](../../../../glossary.md#term-identity-meanings) rewrite or implementation [slice](../../../../glossary.md#slice).

Compiler [capture](../../../../glossary.md#term-capture-meanings) of only the value output still retains the zero-input state source, explicit state
edge, and all three dropout outputs. Forward [DCE](../../../../glossary.md#term-dead-code-elimination-dce) preserves a live multi-output [occurrence](../../../../glossary.md#term-occurrence)
indivisibly. A [differentiable](../../../../glossary.md#derivative-and-chain-rule) value would require the exact same-occurrence mask through
`where(mask, gradient / (1 - probability), +0)` without resampling; this candidate excludes it.

CPU 0006D proves one conforming [backend](../../../../glossary.md#backend)-private realization: exact raw state, five ordered dropout
boundaries, one global-logical-ordinal draw per element, [canonical](../../../../glossary.md#term-canonical-form) mask, modulo advancement,
versioned [SplitMix64](../../../../glossary.md#graph-rng-state--graphrngstate)-based mapping, and an explicit [binary64](../../../../glossary.md#floating-point-precision-and-rounding) threshold/scaling order. Its algorithm
is CPU-private and cannot be imported as a portable Model or [Metal](../../../../glossary.md#term-metal) contract.

## Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) surface and blocker

The installed [SDK](../../../../glossary.md#term-sdk)'s `MPSGraphRandomOps.h` declares `dropoutTensor:rate:name:` and a rate-[tensor](../../../../glossary.md#tensor)
overload. Both return only the dropped tensor and expose no [seed](../../../../glossary.md#term-seed)/state input, keep [mask](../../../../glossary.md#term-boolean-mask), or next
state. The header says removed values become zero but does not establish inverted scaling, exact
[binary64](../../../../glossary.md#floating-point-precision-and-rounding) probability handling, [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity), [subnormal](../../../../glossary.md#term-subnormal), or replay semantics. Hidden
provider randomness is incompatible with explicit [graph](../../../../glossary.md#graph) state.

The same header exposes [Philox](../../../../glossary.md#graph-rng-state--graphrngstate) state constructors and stateful random/uniform calls returning
`[randomValues, updatedState]`; equal state replays the MPSGraph stream. That is explicit MPSGraph
[state threading](../../../../glossary.md#state-threading), but not [Model](../../../../glossary.md#term-model) state. The constructor accepts counter-low, counter-high, and key,
while Model fixes two words `[key, counter]`. The header does not specify the state tensor's type,
[Shape](../../../../glossary.md#shape), serialized [layout](../../../../glossary.md#layout), [FLOAT32](../../../../glossary.md#term-float32) [element](../../../../glossary.md#term-element)-to-counter mapping, or updated-state advancement. It also
requires state created through the Philox surface, so an arbitrary Model `INT64[2]` edge is not an
authorized substitute.

Composing stateful uniform, comparison, selection, and division would still expose MPSGraph's
[opaque](../../../../glossary.md#term-opaque) next state rather than exact `[key, counter + N]`, must narrow or otherwise prove comparison
against binary64 `p`, and inherits unproved special/subnormal/scaling behavior. These structural and
independent semantic gaps cannot be closed by one execution. No probe ran.

## Current [Metal](../../../../glossary.md#term-metal) representation boundary

`MetalCapabilityProvider` rejects output count other than one; `MetalNegPartitionPreparer` requires
one output per [node](../../../../glossary.md#node); [schema](../../../../glossary.md#term-schema)-v11 records have one output index and require a first input. Native
[MPSGraph](../../../../glossary.md#term-mpsgraph) create/[run](../../../../glossary.md#term-run) rejects zero feeds. The Java/native type vocabulary is only [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), and
[BOOL](../../../../glossary.md#data-type--datatype); local [buffer](../../../../glossary.md#term-buffer) [validation](../../../../glossary.md#term-validation), workload [identity](../../../../glossary.md#term-identity-meanings), ingress/[materialization](../../../../glossary.md#materialization), and transfer contain no
[INT64](../../../../glossary.md#data-type--datatype) state path. The custom NEG route is fixed to one FLOAT32 input and one FLOAT32 output.

Prepared MPSGraph execution is reusable and synchronized but owns no [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate). [`RunState`](../../../../glossary.md#run-state--runstate) has no
[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) stream and the context UUID is tuning identity only. Neither repairs explicit key/counter or
the per-node single-output schema.

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0035 changes and reserves no schema. If its exact custom route were separately specified,
proved, and landed first from the current state, an append-only design could advance [node](../../../../glossary.md#node) schema
`11 -> 12`, append `INITIAL_STATE=20` and `DROPOUT=21`, add typed random attributes beginning at
`7`, add local [`INT64`](../../../../glossary.md#data-type--datatype) after existing value-type wires `1..3`, and advance workload, exact-policy,
candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`.

That schema must represent zero inputs for `INITIAL_STATE`, three distinct ordered outputs for
[`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout), raw key/counter and [binary64](../../../../glossary.md#floating-point-precision-and-rounding) probability bits, exact state/[mask](../../../../glossary.md#term-boolean-mask)/value types and [Shapes](../../../../glossary.md#shape),
and complete unused-cell sentinels. The current 160-byte one-output record cannot be retained by
assertion; any kind-specific use of existing cells for extra output indices needs an exact duplicate
Java/native [layout](../../../../glossary.md#layout) and [validation](../../../../glossary.md#term-validation) proof. Local INT64 allocation/[binding](../../../../glossary.md#term-binding) is mandatory even if public
INT64 ingress, [publication](../../../../glossary.md#term-publication), and transfer remain absent.

An exact custom route also needs its own typed native pipeline [lifecycle](../../../../glossary.md#lifecycle). If implemented by adding
three create/[release](../../../../glossary.md#term-release)/[run](../../../../glossary.md#term-run) exports like the custom-NEG precedent, it would conditionally advance [ABI](../../../../glossary.md#term-application-binary-interface-abi)
`4 -> 5` and thirteen exports to sixteen; another exact design must state its complete replacement
contract. None of these numbers, fields, candidates, symbols, or layouts is reserved here.

Tasks 0033 and 0034 conditionally name the same next schema/wire/attribute/version values and reserve
none. The first separately authorized implementation owns actual append-only identities; every
other task must rebase. While all remain Blocked, schema 11, wires `1..19`, attribute kinds `0..6`,
version-twelve identities, ABI v4, the 160-byte record, and thirteen exports remain unchanged.

## Exact unblock and proof prerequisite

Task 0035 can unblock only after a separately authorized design provides an exact [Metal](../../../../glossary.md#term-metal)-private
custom [kernel](../../../../glossary.md#kernel) and the complete multi-output/local-[INT64](../../../../glossary.md#data-type--datatype) [lifecycle](../../../../glossary.md#lifecycle). The proof must fix and identify:

1. the counter-to-word algorithm, key mapping, uniform conversion, exact binary64 threshold rule,
   one-draw-per-logical-index mapping, and modulo state transition;<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
2. exact output arithmetic for finite values, signed zero, every subnormal path, overflow,
   infinities, and NaN classification, using integer/bit-level handling wherever Metal floating
   behavior is insufficiently specified;<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal); [overflow glossary](../../../../glossary.md#term-overflow); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
3. canonical BOOL bytes, one next-state write, deterministic grid-independent logical ordinals,
   same-state replay, branching semantics, and absence of persistent/session RNG;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
4. zero-input initialization, ordered five-boundary dropout binding, distinct buffers, complete
   input/output and output/output overlap rejection before mutation, and reusable lifecycle; and<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [binding glossary](../../../../glossary.md#term-binding); [buffers glossary](../../../../glossary.md#term-buffer))</span>
5. duplicate Java/native kind, attribute, arity, type, Shape, layout, gradient, sentinel, payload,
   stale-identity, resource, publication, and CPU-free Engine validation.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [validation glossary](../../../../glossary.md#term-validation))</span>

Independent [host](../../../../glossary.md#term-host) [vectors](../../../../glossary.md#term-vector)/oracles must derive expected words, threshold boundaries, [masks](../../../../glossary.md#term-boolean-mask), value
bits, and state words without production helpers. A later real-[device](../../../../glossary.md#term-device) [smoke](../../../../glossary.md#term-smoke-test) may prove plumbing but
cannot replace the complete algorithm and arithmetic proof. Standardizing a portable cross-[backend](../../../../glossary.md#backend)
bitstream instead requires a preceding [Model](../../../../glossary.md#term-model)/architecture decision; Metal cannot create it.

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

- Record the exact Model/Compiler/CPU contract, installed MPSGraph mismatch, current Metal
  zero-input/multi-output/INT64 boundary, nondegenerate candidate, conditional schema, and unblock.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [INT64 glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep Task 0035 `Blocked`, expose no [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, scope,
  and final diff. Run no probe, test, formatter, build, schema reservation, or production edit.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Run glossary](../../../../glossary.md#term-run))</span>

## Result

[Planning](../../../../glossary.md#term-planning) from clean `43863c4203fadc9281ab27653594f4720eed2639` found no conforming direct [MPSGraph](../../../../glossary.md#term-mpsgraph)
route. Direct [dropout](../../../../glossary.md#dropout--inverted-dropout) hides randomness and returns one result; [opaque](../../../../glossary.md#term-opaque) [Philox](../../../../glossary.md#graph-rng-state--graphrngstate) state cannot establish
[Model](../../../../glossary.md#term-model)'s exact two-word state or transition. Current [Metal](../../../../glossary.md#term-metal) cannot encode the required [topology](../../../../glossary.md#term-topology) or
[INT64](../../../../glossary.md#data-type--datatype) state. No probe or executable change occurred. Task 0035 is Blocked; no Metal task is [Ready](../../../../glossary.md#term-ready).

# Task 0035: Profile-Common Canonical FLOAT32 No-Grad Explicit-State Dropout

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add explicit-state random execution under both Metal
profiles, local INT64, genuine multi-output lifecycle, private schema/route identities, and an exact
custom native route. This planning-only blocker changes no executable behavior and ran no probe.

## Goal

Assess the smallest nondegenerate coherent Metal random slice: exact `INITIAL_STATE` plus canonical
positive static rank-one `FLOAT32`, no-grad, explicit-state `DROPOUT` under both profiles. Direct
MPSGraph dropout hides randomness and returns one output; opaque Philox state does not establish
Model's exact `INT64[2]` key/counter, mandatory mask/nextState, or `counter + N`. Current Metal
cannot represent zero-input nodes, three-output nodes, or INT64, so Task 0035 is Blocked pre-probe.

## Candidate domain if unblocked

- Admit exact zero-input/one-output `GraphRngKind.INITIAL_STATE` and two-input/three-output
  `DropoutKind.DROPOUT` roles `[value,state] -> [output,keepMask,nextState]`.
- Report identical exact capability under `STRICT_IEEE` and `ACCELERATOR`; random operations receive
  no accelerator relaxation.
- Require canonical dense zero-offset non-view rank-one `FLOAT32` value/output Shape `[N]`, exact
  local `INT64 Shape[2]` state, same-Shape canonical BOOL mask, and distinct non-aliasing outputs.
- Require `requiresGrad=false` for value/output; mask and both state positions are always
  non-differentiable. No-grad removes backward construction, not either mandatory auxiliary output.
- Accept every raw key/counter pair and the complete finite binary64 probability domain `[0,1)`,
  including signed zeros. Each row-major element consumes one draw; next state keeps key and adds
  `N` to counter modulo `2^64`. One versioned Metal-private exact algorithm/policy must replay the
  same output, mask, and next state from identical inputs without mutable session RNG.

## Non-goals

- A `p=0` identity route, fixed seed, fixed passing Shape, selected probability, selected value
  corpus, or other degenerate special case in place of the complete admitted occurrence.
- `requiresGrad=true`, mask-consuming pullback, training-session integration, fusion,
  branching/chaining widening, or attention dropout.
- FLOAT64, BFLOAT16, integral/BOOL values, scalar/empty/dynamic Shapes, other ranks/layouts, views,
  negative strides, in-place execution, or unresolved geometry.
- Other distributions, public algorithm selection, key derivation, split/fold-in, certification,
  cross-backend bitstream, public INT64 ingress/publication/transfer, host repair/fallback/retry, or
  hidden/default/session RNG.
- A seed, Shape, probability, optimization, context, or repetition matrix.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of profile-indexed meaning and requires fail-closed backend support.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  give random operations no `ACCELERATOR` relaxation.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route/resource selection cold and Runtime free of operation-policy lookup.
- Model [explicit RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)
  and [dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md) own state,
  probability, output-role, draw-count, replay, and ideal-value semantics.
- [Compiler stochastic gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  own pullbacks; [CPU random coverage](../../cpu/tasks/0006d-portable-explicit-state-rng-and-dropout.md) is exact private precedent only.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0019B/0019B1; Compiler 0003/0005C/0006B11; CPU 0006D
  precedent; Config 0006; Engine 0018. Earlier Metal blockers are independent evidence.
- Conflicts with: every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/
  local-type/materialization/public Engine scope; multi-output, INT64, random, or gradient work; any
  0016–0018 restart.
- Parallel group: None.
- Common base revision: `43863c4203fadc9281ab27653594f4720eed2639`.
- Integration order: no implementation before an exact custom-kernel and multi-output/INT64 design
  is Ready. While Blocked, validate only Markdown and `git diff --check`; run no executable probe.
- Shared-document integration owner: Task 0035 planner; later Class C work needs a new owner/review.

## Exact Model, Compiler, and CPU boundary

`INITIAL_STATE` is `0 -> 1`: its exact output is non-gradient `INT64 Shape[2]` containing raw
unsigned key then counter words. `DROPOUT` is `2 -> 3`: the value output preserves input floating
type, exact Shape, and gradient eligibility; the hidden keep mask is same-Shape BOOL; next state is
non-gradient `INT64 Shape[2]`. Both auxiliary outputs are mandatory even when only value slot zero
is a public graph output.

For logical row-major position `i`, one abstract `u_i` in `[0,1)` is consumed and keep is exactly
`u_i >= probability`. A dropped result is represented positive zero even for negative zero, NaN,
or infinity. A kept signed zero or infinity preserves sign; a kept NaN remains NaN without a payload
promise; finite kept values have inverted-dropout meaning `input / (1 - probability)` under a
conforming finite-precision realization. Signed `p=0` keeps every element but still draws `N` times,
creates all outputs, and advances state; it is not an identity rewrite or implementation slice.

Compiler capture of only the value output still retains the zero-input state source, explicit state
edge, and all three dropout outputs. Forward DCE preserves a live multi-output occurrence
indivisibly. A differentiable value would require the exact same-occurrence mask through
`where(mask, gradient / (1 - probability), +0)` without resampling; this candidate excludes it.

CPU 0006D proves one conforming backend-private realization: exact raw state, five ordered dropout
boundaries, one global-logical-ordinal draw per element, canonical mask, modulo advancement,
versioned SplitMix64-based mapping, and an explicit binary64 threshold/scaling order. Its algorithm
is CPU-private and cannot be imported as a portable Model or Metal contract.

## Installed MPSGraph surface and blocker

The installed SDK's `MPSGraphRandomOps.h` declares `dropoutTensor:rate:name:` and a rate-tensor
overload. Both return only the dropped tensor and expose no seed/state input, keep mask, or next
state. The header says removed values become zero but does not establish inverted scaling, exact
binary64 probability handling, signed-zero, NaN/infinity, subnormal, or replay semantics. Hidden
provider randomness is incompatible with explicit graph state.

The same header exposes Philox state constructors and stateful random/uniform calls returning
`[randomValues, updatedState]`; equal state replays the MPSGraph stream. That is explicit MPSGraph
state threading, but not Model state. The constructor accepts counter-low, counter-high, and key,
while Model fixes two words `[key, counter]`. The header does not specify the state tensor's type,
Shape, serialized layout, FLOAT32 element-to-counter mapping, or updated-state advancement. It also
requires state created through the Philox surface, so an arbitrary Model `INT64[2]` edge is not an
authorized substitute.

Composing stateful uniform, comparison, selection, and division would still expose MPSGraph's
opaque next state rather than exact `[key, counter + N]`, must narrow or otherwise prove comparison
against binary64 `p`, and inherits unproved special/subnormal/scaling behavior. These structural and
independent semantic gaps cannot be closed by one execution. No probe ran.

## Current Metal representation boundary

`MetalCapabilityProvider` rejects output count other than one; `MetalNegPartitionPreparer` requires
one output per node; schema-v11 records have one output index and require a first input. Native
MPSGraph create/run rejects zero feeds. The Java/native type vocabulary is only FLOAT32, INT32, and
BOOL; local buffer validation, workload identity, ingress/materialization, and transfer contain no
INT64 state path. The custom NEG route is fixed to one FLOAT32 input and one FLOAT32 output.

Prepared MPSGraph execution is reusable and synchronized but owns no RNG state. `RunState` has no
RNG stream and the context UUID is tuning identity only. Neither repairs explicit key/counter or
the per-node single-output schema.

## Conditional schema only

Task 0035 changes and reserves no schema. If its exact custom route were separately specified,
proved, and landed first from the current state, an append-only design could advance node schema
`11 -> 12`, append `INITIAL_STATE=20` and `DROPOUT=21`, add typed random attributes beginning at
`7`, add local `INT64` after existing value-type wires `1..3`, and advance workload, exact-policy,
candidate, compatibility, route-policy, and codec identities together `12 -> 13`.

That schema must represent zero inputs for `INITIAL_STATE`, three distinct ordered outputs for
`DROPOUT`, raw key/counter and binary64 probability bits, exact state/mask/value types and Shapes,
and complete unused-cell sentinels. The current 160-byte one-output record cannot be retained by
assertion; any kind-specific use of existing cells for extra output indices needs an exact duplicate
Java/native layout and validation proof. Local INT64 allocation/binding is mandatory even if public
INT64 ingress, publication, and transfer remain absent.

An exact custom route also needs its own typed native pipeline lifecycle. If implemented by adding
three create/release/run exports like the custom-NEG precedent, it would conditionally advance ABI
`4 -> 5` and thirteen exports to sixteen; another exact design must state its complete replacement
contract. None of these numbers, fields, candidates, symbols, or layouts is reserved here.

Tasks 0033 and 0034 conditionally name the same next schema/wire/attribute/version values and reserve
none. The first separately authorized implementation owns actual append-only identities; every
other task must rebase. While all remain Blocked, schema 11, wires `1..19`, attribute kinds `0..6`,
version-twelve identities, ABI v4, the 160-byte record, and thirteen exports remain unchanged.

## Exact unblock and proof prerequisite

Task 0035 can unblock only after a separately authorized design provides an exact Metal-private
custom kernel and the complete multi-output/local-INT64 lifecycle. The proof must fix and identify:

1. the counter-to-word algorithm, key mapping, uniform conversion, exact binary64 threshold rule,
   one-draw-per-logical-index mapping, and modulo state transition;
2. exact output arithmetic for finite values, signed zero, every subnormal path, overflow,
   infinities, and NaN classification, using integer/bit-level handling wherever Metal floating
   behavior is insufficiently specified;
3. canonical BOOL bytes, one next-state write, deterministic grid-independent logical ordinals,
   same-state replay, branching semantics, and absence of persistent/session RNG;
4. zero-input initialization, ordered five-boundary dropout binding, distinct buffers, complete
   input/output and output/output overlap rejection before mutation, and reusable lifecycle; and
5. duplicate Java/native kind, attribute, arity, type, Shape, layout, gradient, sentinel, payload,
   stale-identity, resource, publication, and CPU-free Engine validation.

Independent host vectors/oracles must derive expected words, threshold boundaries, masks, value
bits, and state words without production helpers. A later real-device smoke may prove plumbing but
cannot replace the complete algorithm and arithmetic proof. Standardizing a portable cross-backend
bitstream instead requires a preceding Model/architecture decision; Metal cannot create it.

## Acceptance criteria and validation

- Record the exact Model/Compiler/CPU contract, installed MPSGraph mismatch, current Metal
  zero-input/multi-output/INT64 boundary, nondegenerate candidate, conditional schema, and unblock.
- Keep Task 0035 `Blocked`, expose no Ready Metal frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, scope,
  and final diff. Run no probe, test, formatter, build, schema reservation, or production edit.

## Result

Planning from clean `43863c4203fadc9281ab27653594f4720eed2639` found no conforming direct MPSGraph
route. Direct dropout hides randomness and returns one result; opaque Philox state cannot establish
Model's exact two-word state or transition. Current Metal cannot encode the required topology or
INT64 state. No probe or executable change occurred. Task 0035 is Blocked; no Metal task is Ready.

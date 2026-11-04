# Task 0037: Profile-Common [Canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) No-Grad FORWARD RNN_TANH

## Reader context

This recurrent neural network (RNN) candidate was one operation with two mandatory outputs,
not a host loop invoking a child graph each time step. Shapes described time, batch, input
and hidden dimensions. The validLengths buffer instead supplied execution-time values that
could change without rebuilding the prepared work.

Model required invalid transitions not to run; calculating everything and masking published
results afterward would not satisfy that rule, especially with exceptional inactive inputs.
Compiler captured the complete occurrence without reading lengths. Metal would need to validate
all lengths and storage access before mutation, but this task implemented no such route.
The [current fixed-recurrent and runtime-length qualification](../../../../backend-guide/production-support.md#metal-production-target)
distinguishes later evidence from this original forward-only, no-probe assessment.

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add first-class recurrent execution under both [Metal](../../../../glossary.md#term-metal)
profiles, caller-input [INT64](../../../../glossary.md#data-type--datatype), genuine five-input/two-output [lifecycle](../../../../glossary.md#lifecycle), private [schema](../../../../glossary.md#term-schema) [identities](../../../../glossary.md#term-identity-meanings),
and an exact custom native route. This [planning](../../../../glossary.md#term-planning)-only blocker changes no executable behavior and ran
no probe.

## Goal

Assess the smallest coherent fixed recurrent-[scan](../../../../glossary.md#term-cumulative-scan) [slice](../../../../glossary.md#slice): [bias](../../../../glossary.md#term-bias)-free `RNN_TANH`, `FORWARD`, [canonical](../../../../glossary.md#term-canonical-form)
positive fully static [`FLOAT32`](../../../../glossary.md#term-float32), and no [gradients](../../../../glossary.md#term-gradient) under both profiles. Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) recurrent
selectors cannot consume [Model](../../../../glossary.md#term-model)'s [runtime](../../../../glossary.md#term-runtime) [`INT64`](../../../../glossary.md#data-type--datatype) valid lengths, atomically validate them, skip
padded recurrence, publish exact positive-zero [padding](../../../../glossary.md#padding), or return `finalHidden`. Their documented
numerics also leave several independent unrelaxed gaps. Task 0037 is therefore Blocked pre-probe.

## Candidate domain if unblocked

- Admit exactly `RecurrentScanKind.RNN_TANH + RecurrentDirection.FORWARD`, ordered
  `[input,validLengths,initialHidden,inputWeight,hiddenWeight] -> [outputs,finalHidden]`.
- Report identical capability under `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator); recurrent scans receive no
  accelerator relaxation.
- Require [canonical](../../../../glossary.md#term-canonical-form) dense [contiguous](../../../../glossary.md#term-contiguous), zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [Shapes](../../../../glossary.md#shape) with positive static
  `time`, [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), `inputSize`, and `hiddenSize`: [`FLOAT32`](../../../../glossary.md#term-float32) input `[T,N,I]`, non-[gradient](../../../../glossary.md#term-gradient) [`INT64`](../../../../glossary.md#data-type--datatype)
  valid lengths `[N]`, initial hidden `[N,H]`, input [weight](../../../../glossary.md#term-weight) `[H,I]`, hidden weight `[H,H]`, output
  `[T,N,H]`, and final hidden `[N,H]`.
- Require `requiresGrad=false` for every floating input and both outputs. Use [Compiler](../../../../glossary.md#term-compiler)
  `FORWARD_ONLY`; no saved gates, [backward graph](../../../../glossary.md#backward-graph), or hidden training output exists.
- Accept every [runtime](../../../../glossary.md#term-runtime) length `0 <= L[n] <= T` without rebuilding the prepared executable. Valid
  coordinates store the next hidden state, padded coordinates store exact positive zero, and a
  zero-length row returns the exact initial hidden state.<span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Validate the complete length vector and every representation access before any output mutation.
  Invalid coordinates execute no recurrent dot product, activation, or state update.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [dot product glossary](../../../../glossary.md#term-dot-product); [activation glossary](../../../../glossary.md#term-activation))</span>

## Non-goals

- [Bias](../../../../glossary.md#term-bias), `REVERSE`, GRU, LSTM, bidirectionality, stacking, configurable activation, projection,
  peephole, recurrent dropout, fusion, saved intermediates, gradients, backpropagation through time,
  or training.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [activation glossary](../../../../glossary.md#term-activation); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [fusion glossary](../../../../glossary.md#term-fusion); [gradients glossary](../../../../glossary.md#term-gradient); [backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Full-length-only values, one time step, hidden size one, fixed lengths, fixed Shape, selected
  corpus, host <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>`long[]` [specialization](../../../../glossary.md#term-specialization), or any other degenerate [narrowing](../../../../glossary.md#term-narrowing).
- BFLOAT16, FLOAT64, FLOAT16, zero/dynamic extents, noncanonical layouts, views, in-place execution,
  fallback, CPU execution, host repair, per-run graph rebuilding, or per-time-step graph expansion.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT16 glossary](../../../../glossary.md#term-float16); [layouts glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [run glossary](../../../../glossary.md#term-run); [graph glossary](../../../../glossary.md#graph))</span>
- A direction, length, [Shape](../../../../glossary.md#shape), optimization, context, or repetition [matrix](../../../../glossary.md#term-matrix-meanings).

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of operation meaning and requires fail-closed backend support.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [operation glossary](../../../../glossary.md#operation); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backend glossary](../../../../glossary.md#backend))</span>
- [Fixed recurrent scan](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
  owns the flat multi-output operation, Shapes, runtime-length validation, traversal, output roles,
  lifecycle ownership, and no-invalid-work invariant.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [runtime glossary](../../../../glossary.md#term-runtime); [validation glossary](../../../../glossary.md#term-validation); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  list recurrent families with no relaxation in either profile.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  keeps route/resource selection cold and Runtime free of recurrent interpretation.
- [Model](../../../../glossary.md#term-model) [fixed recurrent scans](../../../modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  and [expression correction](../../../modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md)
  own public signatures and semantics. Compiler <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Compiler glossary](../../../../glossary.md#term-compiler))</span>[forward adoption and BPTT boundary](../../../modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
  owns capture, inference, layout closure, and backward rejection.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [layout glossary](../../../../glossary.md#layout))</span>

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0025E/0025F; Compiler 0006A; Config 0006; Engine 0018.
  Current CPU has no recurrent execution precedent; earlier Metal blockers are independent evidence.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Conflicts with: every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/
  input-type/materialization/public Engine scope; recurrent, multi-input, multi-output, INT64,
  custom-kernel, or gradient work; any 0016–0018 restart.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [preflight glossary](../../../../glossary.md#term-preflight); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope); [INT64 glossary](../../../../glossary.md#data-type--datatype); [kernel glossary](../../../../glossary.md#kernel); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Parallel group: None.
- Common base revision: `c939143a612cd7ce1c3768ba0720928511f10475`.
- Integration order: no implementation before an exact custom recurrent route and complete
  five-input/two-output/INT64 design are Ready. While Blocked, run no device gate.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [run glossary](../../../../glossary.md#term-run); [device glossary](../../../../glossary.md#term-device))</span>
- Integration validation: Markdown links/anchors/fences/newlines, whitespace, planning-only scope,
  and final diff only.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [planning glossary](../../../../glossary.md#term-planning))</span>
- Shared-document integration owner: Task 0037 planner; later Class C work needs a new owner and
  independent review.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

## Exact [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), and [CPU](../../../../glossary.md#term-cpu) boundary

For each valid [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), the [bias](../../../../glossary.md#term-bias)-free transition is:

```text
h' = tanh((x @ inputWeight^T) + (h @ hiddenWeight^T))
```

Model fixes the displayed association, [time-major](../../../../glossary.md#term-time-major) geometry, `FORWARD` valid-prefix traversal,
positive-zero [padding](../../../../glossary.md#padding), and final-state role. All floating roles have one exact type; valid lengths
are ordinary non-[gradient](../../../../glossary.md#term-gradient) [runtime](../../../../glossary.md#term-runtime) `INT64[N]` input. The output gradient flag is the OR of the
floating inputs, so this candidate's all-false inputs produce all-false outputs.

Compiler [captures](../../../../glossary.md#term-capture-meanings) one [identity](../../../../glossary.md#term-identity-meanings)-distinct five-input/two-output [node](../../../../glossary.md#node), revalidates its complete static
[descriptors](../../../../glossary.md#term-descriptor) without reading length values, and canonicalizes eligible static result [layouts](../../../../glossary.md#layout). A live
sibling keeps the whole multi-output [occurrence](../../../../glossary.md#term-occurrence). `FORWARD_ONLY` is supported, while every original
recurrent occurrence is rejected in `FORWARD_AND_BACKWARD` or `TRAINING_STEP` before gradient
allocation. No recurrent gradient or [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) formula exists.

Current CPU has no first-class recurrent capability, [lowering](../../../../glossary.md#lowering), route, or numerical policy to reuse.
Existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) sequence containers statically unroll [host](../../../../glossary.md#term-host) `long[]` lengths and return compact lists;
the recurrent contract explicitly forbids redirecting them to this runtime-length [operation](../../../../glossary.md#operation).

## Installed [MPSGraph](../../../../glossary.md#term-mpsgraph) surface and blocker

Installed `MPSGraphRNNOps.h` declares single-gate [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) selectors on macOS 12.3+ and [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) on
macOS 13.0+. Single-gate RNN accepts [time-major](../../../../glossary.md#term-time-major) source, input/recurrent [weights](../../../../glossary.md#term-weight), optional [bias](../../../../glossary.md#term-bias), and
initial state, but no sequence lengths. Its [mask](../../../../glossary.md#term-boolean-mask) is recurrent [dropout](../../../../glossary.md#dropout--inverted-dropout), not validity; it cannot
validate lengths, skip padded transitions, or define zero-padded outputs.

With training disabled, RNN returns only hidden sequence `[T,N,H]`; training adds preactivation
sequence `[T,N,H]`, not [Model](../../../../glossary.md#term-model)'s `finalHidden[N,H]`. Post-masking still executes forbidden padded
recurrence. Dynamic final-state [gather](../../../../glossary.md#gather) adds an `L=0` branch but cannot repair [atomic](../../../../glossary.md#resource-transaction-and-rollback) [validation](../../../../glossary.md#term-validation) or
skipped work. The selector cannot implement the candidate regardless of a numerical probe.

The header does not authorize [contraction](../../../../glossary.md#contraction-dimension) algorithm, [FLOAT32](../../../../glossary.md#term-float32) accumulation/[rounding](../../../../glossary.md#term-rounding), add association,
[tanh](../../../../glossary.md#term-tanh) special/[subnormal](../../../../glossary.md#term-subnormal) behavior, or per-step state [materialization](../../../../glossary.md#materialization)/order across all [Shapes](../../../../glossary.md#shape).
Recurrent scans have no profile relaxation. Task 0006 observed standalone MPSGraph tanh map [negative
zero](../../../../glossary.md#nan-infinity-and-signed-zero) to [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) and [quiet NaN](../../../../glossary.md#nan-infinity-and-signed-zero) to positive one; that does not prove a shared implementation,
but forbids assuming compatibility. One [run](../../../../glossary.md#term-run) cannot close the independent gaps, so no probe ran.

## Adjacent [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) mismatches

GRU can select plausible fixed controls but lacks lengths/final hidden, returns packed training
state, and documents `z*h + (1-z)*candidate`, not [Model](../../../../glossary.md#term-model)'s fixed association.

LSTM can align fixed controls, but its cell/training outputs are full sequence/packed gate state,
not `[outputs,finalHidden,finalCell]`; it also lacks length [validation](../../../../glossary.md#term-validation) and skipped padded work.

## Current [Metal](../../../../glossary.md#term-metal) representation boundary

`MetalCapabilityProvider` rejects any [occurrence](../../../../glossary.md#term-occurrence) whose output count is not one, and
`MetalNegPartitionPreparer` requires one output per [node](../../../../glossary.md#node). [Schema](../../../../glossary.md#term-schema) v11 has two ordinary input indices,
one [operation](../../../../glossary.md#operation)-specific auxiliary index, one output index, and fixed input-count [metadata](../../../../glossary.md#term-metadata). The
candidate requires five ordered inputs and two distinct ordered outputs.

The Java/native [MPSGraph](../../../../glossary.md#term-mpsgraph) type vocabulary is `FLOAT32=1`, `INT32=2`, and `BOOL=3`; no [INT64](../../../../glossary.md#data-type--datatype) wire,
caller-input typing, [materialization](../../../../glossary.md#materialization), or supported transfer path exists for `validLengths`. The
custom NEG route is fixed to one [FLOAT32](../../../../glossary.md#term-float32) input and one FLOAT32 output. Existing reusable execution
cannot infer or specialize valid lengths in [Runtime](../../../../glossary.md#term-runtime) or [Engine](../../../../glossary.md#term-engine).

## Conditional [schema](../../../../glossary.md#term-schema) only

Task 0037 changes and reserves no schema. If its exact route were separately specified, proved, and
landed first from the current state, an append-only design could advance [node](../../../../glossary.md#node) schema `11 -> 12`,
append `RNN_TANH=20` and `AttributeKind.RECURRENT=7`, add `INT64=4` after existing value-type wires
`1..3`, and advance workload, exact-policy, candidate, compatibility, route-policy, and [codec](../../../../glossary.md#term-codec)
[identities](../../../../glossary.md#term-identity-meanings) together `12 -> 13`.

That schema must represent five ordered inputs, two distinct ordered outputs, `FORWARD`, exact types
and [Shapes](../../../../glossary.md#shape), [gradient](../../../../glossary.md#term-gradient) policy, and complete unused-cell sentinels. It must add caller [INT64](../../../../glossary.md#data-type--datatype)
input/[binding](../../../../glossary.md#term-binding)/[materialization](../../../../glossary.md#materialization) support, not merely a [backend](../../../../glossary.md#backend)-local INT64 temporary. The current
160-byte one-output record cannot be retained by assertion; any use of existing cells for extra
input/output indices needs an exact duplicate Java/native [layout](../../../../glossary.md#layout) and [validation](../../../../glossary.md#term-validation) proof.

An exact custom route needs its own typed multi-[buffer](../../../../glossary.md#term-buffer) native pipeline [lifecycle](../../../../glossary.md#lifecycle). If implemented by
adding create/[release](../../../../glossary.md#term-release)/[run](../../../../glossary.md#term-run) exports like the custom-NEG precedent, it would conditionally advance [ABI](../../../../glossary.md#term-application-binary-interface-abi)
`4 -> 5` and thirteen exports to sixteen; another design must state its complete replacement
contract. None of these numbers, fields, candidates, symbols, or layouts is reserved here. Tasks
0026–0027 and 0030–0036 conditionally name the same next values and reserve none; the first
separately authorized implementation owns them and every other plan rebases.

## Exact custom-route prerequisite

Unblocking requires a separately authorized [Metal](../../../../glossary.md#term-metal)-private recurrent [kernel](../../../../glossary.md#kernel) and [lifecycle](../../../../glossary.md#lifecycle) that:

1. validates every [`INT64`](../../../../glossary.md#data-type--datatype) length, bound, access, conversion, and overlap before mutating either
   output, then reports failure without partial published results;
2. executes the fixed forward transition only for `t < L[n]`, writes exact positive-zero padding,
   preserves initial hidden for <span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>`L[n]=0`, and publishes both ordered outputs;
3. implements and proves the complete profile-common contraction, addition, tanh, state-update,
   exceptional-value, signed-zero, and subnormal allowed-result contract for every admitted Shape;<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [tanh glossary](../../../../glossary.md#term-tanh); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal))</span>
4. prepares one bounded reusable occurrence-level executable, declares all resources before slot
   assignment, and creates no graph, node, executable body, or object graph per time step; and<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node))</span>
5. duplicates Java/native validation for kind, direction, arity, type, Shape, layout, gradient,
   payload, sentinel, identity, resource, binding, publication, and CPU-free ownership.<span data-glossary-navigation="true"> ([arity glossary](../../../../glossary.md#term-arity); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

A future direct [API](../../../../glossary.md#term-application-programming-interface-api) is relevant only if it supplies the same complete [runtime](../../../../glossary.md#term-runtime)-length, output-role,
skipped-work, [validation](../../../../glossary.md#term-validation), and numerical contract. Do not substitute full lengths, a post-[mask](../../../../glossary.md#term-boolean-mask),
final-state slicing, [host](../../../../glossary.md#term-host) [specialization](../../../../glossary.md#term-specialization), a one-cell [Shape](../../../../glossary.md#shape), fallback, or a weakened [oracle](../../../../glossary.md#term-test-oracle).

## Acceptance criteria and [validation](../../../../glossary.md#term-validation)

- Record the exact smallest candidate, Model/Compiler/CPU boundary, installed RNN blocker, adjacent
  GRU/LSTM mismatches, current Metal arity/type boundary, conditional schema, and custom-route proof.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [arity glossary](../../../../glossary.md#term-arity); [schema glossary](../../../../glossary.md#term-schema))</span>
- Keep Task 0037 `Blocked`, expose no [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, scope,
  and final diff. Run no probe, test, formatter, build, schema reservation, or production edit.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Run glossary](../../../../glossary.md#term-run))</span>

## Result

[Planning](../../../../glossary.md#term-planning) from clean `c939143a612cd7ce1c3768ba0720928511f10475` found no conforming direct
[MPSGraph](../../../../glossary.md#term-mpsgraph) route. The selector lacks [Model](../../../../glossary.md#term-model)'s [runtime](../../../../glossary.md#term-runtime) lengths, [atomic](../../../../glossary.md#resource-transaction-and-rollback) [validation](../../../../glossary.md#term-validation), skipped padded work,
zero [padding](../../../../glossary.md#padding), and final hidden output, while its numerics leave multiple unrelaxed gaps. Current
[Metal](../../../../glossary.md#term-metal) cannot encode the required [INT64](../../../../glossary.md#data-type--datatype) five-input/two-output [occurrence](../../../../glossary.md#term-occurrence). No probe or executable
change occurred. Task 0037 is Blocked; no Metal task is [Ready](../../../../glossary.md#term-ready).

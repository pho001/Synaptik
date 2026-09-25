# Task 0037: Profile-Common Canonical FLOAT32 No-Grad FORWARD RNN_TANH

## Status

Blocked

## Change class

Class C if unblocked — the candidate would add first-class recurrent execution under both Metal
profiles, caller-input INT64, genuine five-input/two-output lifecycle, private schema identities,
and an exact custom native route. This planning-only blocker changes no executable behavior and ran
no probe.

## Goal

Assess the smallest coherent fixed recurrent-scan slice: bias-free `RNN_TANH`, `FORWARD`, canonical
positive fully static `FLOAT32`, and no gradients under both profiles. Installed MPSGraph recurrent
selectors cannot consume Model's runtime `INT64` valid lengths, atomically validate them, skip
padded recurrence, publish exact positive-zero padding, or return `finalHidden`. Their documented
numerics also leave several independent unrelaxed gaps. Task 0037 is therefore Blocked pre-probe.

## Candidate domain if unblocked

- Admit exactly `RecurrentScanKind.RNN_TANH + RecurrentDirection.FORWARD`, ordered
  `[input,validLengths,initialHidden,inputWeight,hiddenWeight] -> [outputs,finalHidden]`.
- Report identical capability under `STRICT_IEEE` and `ACCELERATOR`; recurrent scans receive no
  accelerator relaxation.
- Require canonical dense contiguous, zero-offset, non-view Shapes with positive static
  `time`, `batch`, `inputSize`, and `hiddenSize`: `FLOAT32` input `[T,N,I]`, non-gradient `INT64`
  valid lengths `[N]`, initial hidden `[N,H]`, input weight `[H,I]`, hidden weight `[H,H]`, output
  `[T,N,H]`, and final hidden `[N,H]`.
- Require `requiresGrad=false` for every floating input and both outputs. Use Compiler
  `FORWARD_ONLY`; no saved gates, backward graph, or hidden training output exists.
- Accept every runtime length `0 <= L[n] <= T` without rebuilding the prepared executable. Valid
  coordinates store the next hidden state, padded coordinates store exact positive zero, and a
  zero-length row returns the exact initial hidden state.
- Validate the complete length vector and every representation access before any output mutation.
  Invalid coordinates execute no recurrent dot product, activation, or state update.

## Non-goals

- Bias, `REVERSE`, GRU, LSTM, bidirectionality, stacking, configurable activation, projection,
  peephole, recurrent dropout, fusion, saved intermediates, gradients, backpropagation through time,
  or training.
- Full-length-only values, one time step, hidden size one, fixed lengths, fixed Shape, selected
  corpus, host `long[]` specialization, or any other degenerate narrowing.
- BFLOAT16, FLOAT64, FLOAT16, zero/dynamic extents, noncanonical layouts, views, in-place execution,
  fallback, CPU execution, host repair, per-run graph rebuilding, or per-time-step graph expansion.
- A direction, length, Shape, optimization, context, or repetition matrix.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps Model
  as sole owner of operation meaning and requires fail-closed backend support.
- [Fixed recurrent scan](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  owns the flat multi-output operation, Shapes, runtime-length validation, traversal, output roles,
  lifecycle ownership, and no-invalid-work invariant.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  list recurrent families with no relaxation in either profile.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) owns
  truthful capability, lowering, native integration, storage, and materialization, not meaning.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps route/resource selection cold and Runtime free of recurrent interpretation.
- Model [fixed recurrent scans](../../../modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  and [expression correction](../../../modules/model/tasks/0025f-recurrent-scan-expression-namespace-correction.md)
  own public signatures and semantics. Compiler [forward adoption and BPTT boundary](../../../modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)
  owns capture, inference, layout closure, and backward rejection.

## Dependencies and integration

- Depends on: Metal 0025 Complete; Model 0025E/0025F; Compiler 0006A; Config 0006; Engine 0018.
  Current CPU has no recurrent execution precedent; earlier Metal blockers are independent evidence.
- Conflicts with: every concurrent Metal capability/schema/native-ABI/preflight/candidate/codec/
  input-type/materialization/public Engine scope; recurrent, multi-input, multi-output, INT64,
  custom-kernel, or gradient work; any 0016–0018 restart.
- Parallel group: None.
- Common base revision: `c939143a612cd7ce1c3768ba0720928511f10475`.
- Integration order: no implementation before an exact custom recurrent route and complete
  five-input/two-output/INT64 design are Ready. While Blocked, run no device gate.
- Integration validation: Markdown links/anchors/fences/newlines, whitespace, planning-only scope,
  and final diff only.
- Shared-document integration owner: Task 0037 planner; later Class C work needs a new owner and
  independent review.

## Exact Model, Compiler, and CPU boundary

For each valid coordinate, the bias-free transition is:

```text
h' = tanh((x @ inputWeight^T) + (h @ hiddenWeight^T))
```

Model fixes the displayed association, time-major geometry, `FORWARD` valid-prefix traversal,
positive-zero padding, and final-state role. All floating roles have one exact type; valid lengths
are ordinary non-gradient runtime `INT64[N]` input. The output gradient flag is the OR of the
floating inputs, so this candidate's all-false inputs produce all-false outputs.

Compiler captures one identity-distinct five-input/two-output node, revalidates its complete static
descriptors without reading length values, and canonicalizes eligible static result layouts. A live
sibling keeps the whole multi-output occurrence. `FORWARD_ONLY` is supported, while every original
recurrent occurrence is rejected in `FORWARD_AND_BACKWARD` or `TRAINING_STEP` before gradient
allocation. No recurrent gradient or BPTT formula exists.

Current CPU has no first-class recurrent capability, lowering, route, or numerical policy to reuse.
Existing NN sequence containers statically unroll host `long[]` lengths and return compact lists;
the recurrent contract explicitly forbids redirecting them to this runtime-length operation.

## Installed MPSGraph surface and blocker

Installed `MPSGraphRNNOps.h` declares single-gate RNN and LSTM selectors on macOS 12.3+ and GRU on
macOS 13.0+. Single-gate RNN accepts time-major source, input/recurrent weights, optional bias, and
initial state, but no sequence lengths. Its mask is recurrent dropout, not validity; it cannot
validate lengths, skip padded transitions, or define zero-padded outputs.

With training disabled, RNN returns only hidden sequence `[T,N,H]`; training adds preactivation
sequence `[T,N,H]`, not Model's `finalHidden[N,H]`. Post-masking still executes forbidden padded
recurrence. Dynamic final-state gather adds an `L=0` branch but cannot repair atomic validation or
skipped work. The selector cannot implement the candidate regardless of a numerical probe.

The header does not authorize contraction algorithm, FLOAT32 accumulation/rounding, add association,
tanh special/subnormal behavior, or per-step state materialization/order across all Shapes.
Recurrent scans have no profile relaxation. Task 0006 observed standalone MPSGraph tanh map negative
zero to positive zero and quiet NaN to positive one; that does not prove a shared implementation,
but forbids assuming compatibility. One run cannot close the independent gaps, so no probe ran.

## Adjacent GRU and LSTM mismatches

GRU can select plausible fixed controls but lacks lengths/final hidden, returns packed training
state, and documents `z*h + (1-z)*candidate`, not Model's fixed association.

LSTM can align fixed controls, but its cell/training outputs are full sequence/packed gate state,
not `[outputs,finalHidden,finalCell]`; it also lacks length validation and skipped padded work.

## Current Metal representation boundary

`MetalCapabilityProvider` rejects any occurrence whose output count is not one, and
`MetalNegPartitionPreparer` requires one output per node. Schema v11 has two ordinary input indices,
one operation-specific auxiliary index, one output index, and fixed input-count metadata. The
candidate requires five ordered inputs and two distinct ordered outputs.

The Java/native MPSGraph type vocabulary is `FLOAT32=1`, `INT32=2`, and `BOOL=3`; no INT64 wire,
caller-input typing, materialization, or supported transfer path exists for `validLengths`. The
custom NEG route is fixed to one FLOAT32 input and one FLOAT32 output. Existing reusable execution
cannot infer or specialize valid lengths in Runtime or Engine.

## Conditional schema only

Task 0037 changes and reserves no schema. If its exact route were separately specified, proved, and
landed first from the current state, an append-only design could advance node schema `11 -> 12`,
append `RNN_TANH=20` and `AttributeKind.RECURRENT=7`, add `INT64=4` after existing value-type wires
`1..3`, and advance workload, exact-policy, candidate, compatibility, route-policy, and codec
identities together `12 -> 13`.

That schema must represent five ordered inputs, two distinct ordered outputs, `FORWARD`, exact types
and Shapes, gradient policy, and complete unused-cell sentinels. It must add caller INT64
input/binding/materialization support, not merely a backend-local INT64 temporary. The current
160-byte one-output record cannot be retained by assertion; any use of existing cells for extra
input/output indices needs an exact duplicate Java/native layout and validation proof.

An exact custom route needs its own typed multi-buffer native pipeline lifecycle. If implemented by
adding create/release/run exports like the custom-NEG precedent, it would conditionally advance ABI
`4 -> 5` and thirteen exports to sixteen; another design must state its complete replacement
contract. None of these numbers, fields, candidates, symbols, or layouts is reserved here. Tasks
0026–0027 and 0030–0036 conditionally name the same next values and reserve none; the first
separately authorized implementation owns them and every other plan rebases.

## Exact custom-route prerequisite

Unblocking requires a separately authorized Metal-private recurrent kernel and lifecycle that:

1. validates every `INT64` length, bound, access, conversion, and overlap before mutating either
   output, then reports failure without partial published results;
2. executes the fixed forward transition only for `t < L[n]`, writes exact positive-zero padding,
   preserves initial hidden for `L[n]=0`, and publishes both ordered outputs;
3. implements and proves the complete profile-common contraction, addition, tanh, state-update,
   exceptional-value, signed-zero, and subnormal allowed-result contract for every admitted Shape;
4. prepares one bounded reusable occurrence-level executable, declares all resources before slot
   assignment, and creates no graph, node, executable body, or object graph per time step; and
5. duplicates Java/native validation for kind, direction, arity, type, Shape, layout, gradient,
   payload, sentinel, identity, resource, binding, publication, and CPU-free ownership.

A future direct API is relevant only if it supplies the same complete runtime-length, output-role,
skipped-work, validation, and numerical contract. Do not substitute full lengths, a post-mask,
final-state slicing, host specialization, a one-cell Shape, fallback, or a weakened oracle.

## Acceptance criteria and validation

- Record the exact smallest candidate, Model/Compiler/CPU boundary, installed RNN blocker, adjacent
  GRU/LSTM mismatches, current Metal arity/type boundary, conditional schema, and custom-route proof.
- Keep Task 0037 `Blocked`, expose no Ready Metal frontier, and preserve every earlier result.
- Synchronize master/roadmap; validate Markdown links, anchors, fences, newlines, whitespace, scope,
  and final diff. Run no probe, test, formatter, build, schema reservation, or production edit.

## Result

Planning from clean `c939143a612cd7ce1c3768ba0720928511f10475` found no conforming direct
MPSGraph route. The selector lacks Model's runtime lengths, atomic validation, skipped padded work,
zero padding, and final hidden output, while its numerics leave multiple unrelaxed gaps. Current
Metal cannot encode the required INT64 five-input/two-output occurrence. No probe or executable
change occurred. Task 0037 is Blocked; no Metal task is Ready.

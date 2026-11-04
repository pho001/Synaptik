# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0058: Trainable [runtime](../../glossary.md#term-runtime)-length [RNN](../../glossary.md#term-recurrent-neural-network-rnn) composition


## Editorial reading context

NN composes the ordinary RNN cell equation into a general Scan; Compiler supplies first-order reverse construction and Engine executes the prepared result. Parameters are bound as invariant inputs, not copied per time step. The application still supplies values and lengths. Fixed scans, static unrolling and general scans have different compatibility boundaries despite similar recurrent mathematics.

Use the [general Scan roles and differentiation](../../architecture/general-differentiable-scan.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status and date

Accepted design — 2026-11-01 under continuous production delivery. Independent bounded
[API](../../glossary.md#term-application-programming-interface-api)/authority review APPROVE, zero P1/P2. Implementation and consumer acceptance belong to
[NN0033](../../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md),
after accepted Compiler0018 and Checkpoint0014. This is not an implementation-completion claim.

## Context

The existing fixed [runtime](../../glossary.md#term-runtime)-length [RNN](../../glossary.md#term-recurrent-neural-network-rnn) [API](../../glossary.md#term-application-programming-interface-api) is [inference](../../glossary.md#term-inference)-only. [General Scan](../../glossary.md#general-differentiable-scan) already owns ordinary
pure bodies, runtime row activity and first-order reverse construction. Replacing the fixed API
would also alter its output [layout](../../glossary.md#layout), broader resolved-read [qualification](../../glossary.md#term-qualification) and retained low paths.
The production [target](../../glossary.md#term-target) needs trainable [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) runtime-length RNN Models without a second executor.

## Options and decision

- Changing fixed forwardScan globally would obscure compatibility and affect unrelated callers.
- A mode flag, temporary cell/dummy transition or recapture-and-replace path adds branching,
  allocation and duplicate ownership/construction; rejected.<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- Add explicit RnnSequence.forwardGeneralScan and use it for [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) builder rnnScan; selected.

The [owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
specifies exact bounds, initialization, [carrier](../../glossary.md#term-storage-carrier) roles and compatibility. Reuse [descriptor](../../glossary.md#term-descriptor)-only
[atomic](../../glossary.md#resource-transaction-and-rollback) cell initialization and one package-private cell-step equation shared with ordinary
forward, preserving [validation](../../glossary.md#term-validation) and [bias](../../glossary.md#term-bias)/ADD order. Build the general body directly from
storage-free [carry](../../glossary.md#term-carry)/step/[parameter](../../glossary.md#term-parameter) formals and bind actual parameters as invariants.

Return the existing RecurrentScanResult by semantic component names, not by pretending every
[producer](../../glossary.md#term-producer) uses the same slot order. Update final-hidden selection to use actual general/fixed
roles; do not add [graph](../../glossary.md#graph) operations or copy data to imitate old [metadata](../../glossary.md#term-metadata).

## Rationale

A compiled ordinary body with fixed [carry](../../glossary.md#term-carry) [shape](../../glossary.md#shape)/[dtype](../../glossary.md#data-type--datatype) is the established [Scan](../../glossary.md#general-differentiable-scan) model; see
[JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html). Synaptik retains its
own explicit row-activity, [binding](../../glossary.md#term-binding), numerical and [lifecycle](../../glossary.md#lifecycle) contracts, not JAX [serialization](../../glossary.md#term-serialization) or
arbitrary control-flow semantics. A small composition method exposes that existing capability
without duplicating recurrence mathematics or introducing a new execution architecture.

## Consequences

[F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) rnnScan builder sources intentionally use [canonical](../../glossary.md#term-canonical-form) general output [layouts](../../glossary.md#layout) and v5
[persistence](../../glossary.md#term-persistence) after [qualification](../../glossary.md#term-qualification). Direct fixed forwardScan, ordinary statically [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) recurrence,
[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm), old fixed files and retained low construction remain unchanged. Any source containing
a fixed [scan](../../glossary.md#term-fixed-recurrent-scan) still rejects training before setup; no higher-order compatibility waiver follows.

The added public method requires substantive bounds/numerical/initialization/[ownership](../../glossary.md#term-ownership-meanings) tests,
not a repinned exact-method inventory. Consumer acceptance includes changed lengths, inactive
NaNs, zero-length state [identity](../../glossary.md#term-identity-meanings), actual recurrent [gradients](../../glossary.md#term-gradient) and repeated updates, save/close/load
and newly configured training. No exact [optimizer](../../glossary.md#term-optimizer)/session-[resume](../../glossary.md#schema-serialization-and-checkpoint) claim is added.

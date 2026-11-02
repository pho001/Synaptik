# ADR 0058: Trainable runtime-length RNN composition

## Status and date

Accepted design — 2026-11-01 under continuous production delivery. Independent bounded
API/authority review APPROVE, zero P1/P2. Implementation and consumer acceptance belong to
[NN0033](../../planning/extensions/nn/tasks/0033-trainable-runtime-length-rnn-workflow.md),
after accepted Compiler0018 and Checkpoint0014. This is not an implementation-completion claim.

## Context

The existing fixed runtime-length RNN API is inference-only. General Scan already owns ordinary
pure bodies, runtime row activity and first-order reverse construction. Replacing the fixed API
would also alter its output layout, broader resolved-read qualification and retained low paths.
The production target needs trainable F32/F64 runtime-length RNN Models without a second executor.

## Options and decision

- Changing fixed forwardScan globally would obscure compatibility and affect unrelated callers.
- A mode flag, temporary cell/dummy transition or recapture-and-replace path adds branching,
  allocation and duplicate ownership/construction; rejected.
- Add explicit RnnSequence.forwardGeneralScan and use it for F32/F64 builder rnnScan; selected.

The [owning NN contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)
specifies exact bounds, initialization, carrier roles and compatibility. Reuse descriptor-only
atomic cell initialization and one package-private cell-step equation shared with ordinary
forward, preserving validation and bias/ADD order. Build the general body directly from
storage-free carry/step/parameter formals and bind actual parameters as invariants.

Return the existing RecurrentScanResult by semantic component names, not by pretending every
producer uses the same slot order. Update final-hidden selection to use actual general/fixed
roles; do not add graph operations or copy data to imitate old metadata.

## Rationale

A compiled ordinary body with fixed carry shape/dtype is the established Scan model; see
[JAX lax.scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html). Synaptik retains its
own explicit row-activity, binding, numerical and lifecycle contracts, not JAX serialization or
arbitrary control-flow semantics. A small composition method exposes that existing capability
without duplicating recurrence mathematics or introducing a new execution architecture.

## Consequences

F32/F64 rnnScan builder sources intentionally use canonical general output layouts and v5
persistence after qualification. Direct fixed forwardScan, ordinary statically unrolled recurrence,
GRU/LSTM, old fixed files and retained low construction remain unchanged. Any source containing
a fixed scan still rejects training before setup; no higher-order compatibility waiver follows.

The added public method requires substantive bounds/numerical/initialization/ownership tests,
not a repinned exact-method inventory. Consumer acceptance includes changed lengths, inactive
NaNs, zero-length state identity, actual recurrent gradients and repeated updates, save/close/load
and newly configured training. No exact optimizer/session-resume claim is added.

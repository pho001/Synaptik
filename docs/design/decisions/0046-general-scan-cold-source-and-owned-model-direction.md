# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0046: Bounded general-[scan](../../glossary.md#general-differentiable-scan) cold source and [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) direction


## Editorial reading context

An owned Model must copy and rebind its source without losing the attached Scan body. Compiler reconstructs supported source descriptions; Checkpoint preserves scoped attachments; NN then acquires private state. Successful backend execution alone does not prove that this cold, non-executing reconstruction is supported. The decision sequences those owners rather than adding an NN-local clone or runner.

Use the [cold reconstruction versus execution](../../architecture/general-differentiable-scan.md#bounded-cold-persistence) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted direction — 2026-10-23, explicit user approval of limited NN0030 and the first [Compiler](../../glossary.md#term-compiler)
prerequisite. Independent pre-code authority/inventory review is required before implementation.
This record explains [the authority root](../../../ARCHITECTURE.md); it is not another contract.

## Context and decision drivers

[General scan](../../glossary.md#general-differentiable-scan) already has source/[IR](../../glossary.md#term-intermediate-representation-ir), first-order [Compiler](../../glossary.md#term-compiler) differentiation and bounded [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) [CPU](../../glossary.md#term-cpu)
execution. [Owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions).[graph](../../glossary.md#graph)/defineTensor acquisition instead depends on cold Compiler constructors
and [Checkpoint](../../glossary.md#term-checkpoint) [normalization](../../glossary.md#term-normalization). The latter currently omits scanBody; CompiledNode rejects that omission
before [host](../../glossary.md#term-host) copying. Execution capability alone cannot safely unlock owned acquisition.
Preserve one ordinary [Tensor](../../glossary.md#tensor) graph, private [weight](../../glossary.md#term-weight) set and existing [Engine](../../glossary.md#term-engine)/[TrainingSession](../../glossary.md#training-session--trainingsession) [owners](../../glossary.md#term-owner);
separate [Model](../../glossary.md#term-model) semantics, reconstructibility, execution [qualification](../../glossary.md#term-qualification) and wire compatibility.

## Options considered

- Keep owned general scans rejected: safe but does not deliver the requested ordinary [Model](../../glossary.md#term-model) workflow.
- Add an [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-local clone/runner/session: duplicates [Compiler](../../glossary.md#term-compiler)/[ownership](../../glossary.md#term-ownership-meanings) responsibilities; rejected.
- Extend bounded Compiler constructors, then Checkpoint in-memory copy/rebind, then qualify NN:
  selected; avoids a second pipeline while retaining fail-closed consumer and wire boundaries.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../glossary.md#term-checkpoint); [rebind glossary](../../glossary.md#capture-graph-ports-and-reconstruction); [fail-closed glossary](../../glossary.md#term-fail-closed))</span>

## Decision and rationale

Use existing [Model](../../glossary.md#term-model).[graph](../../glossary.md#graph)/defineTensor → compute/configureTraining/train. First tested execution
is all-valid [canonical](../../glossary.md#term-canonical-form) static [homogeneous](../../glossary.md#term-homogeneous) [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding); no [padding](../../glossary.md#padding) or fixed-[scan](../../glossary.md#term-fixed-recurrent-scan) [BPTT](../../glossary.md#term-backpropagation-through-time-bptt). Cold [admission](../../glossary.md#term-admission)
uses the exact [Compiler inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler))</span>,
not a [CPU](../../glossary.md#term-cpu) [dtype](../../glossary.md#data-type--datatype)/[layout](../../glossary.md#layout) predicate or an all-[PURE](../../glossary.md#term-pure) promise. Preserve unchanged flat outer admission
except GENERAL; explicit body-only mappings use current public constructors and applicable Model
dtypes. Whole outer/all-body [preflight](../../glossary.md#term-preflight) precedes all formals/[producers](../../glossary.md#term-producer); tables are call-local per [scope](../../glossary.md#term-scope).

## Consequences and follow-up

[Compiler](../../glossary.md#term-compiler)-only work must keep [Checkpoint](../../glossary.md#term-checkpoint)/[NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) rejected before [host](../../glossary.md#term-host)/file/setup effects; a test pins
existing missing-attachment rejection. A later Checkpoint capability preserves attachments/local
IDs for copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) and independently rejects bodies in the closed v1/v2/v3 wire before effects.
Only then qualify NN private acquisition, cached compute and first-order training. This costs
serial consumer [validation](../../glossary.md#term-validation) and intentionally leaves some valid [PURE](../../glossary.md#term-pure) bodies cold-unsupported.
No new production type/[API](../../glossary.md#term-application-programming-interface-api)/[dependency](../../glossary.md#term-dependency) or [runtime](../../glossary.md#term-runtime) [owner](../../glossary.md#term-owner) is selected. Physical byte/resource limits
stay with Checkpoint/[backend](../../glossary.md#backend); raw unresolved outer [layouts](../../glossary.md#layout) stay valid. Fixed [scan](../../glossary.md#term-fixed-recurrent-scan) training guards,
builders and formats remain. Padded NN, low/[vector](../../glossary.md#term-vector)/parallel/[OpenBLAS](../../glossary.md#term-openblas) execution and body [persistence](../../glossary.md#term-persistence)
require separate decisions. Final program join owns one native root test/[Javadoc](../../glossary.md#term-javadoc) checkpoint.

## Related documentation

[NN0030 decision](../../planning/extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md),
[Compiler0012 brief](../../planning/modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md),
[scan scopes](../../architecture/general-differentiable-scan.md#source-expressions-and-independent-scopes)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#general-differentiable-scan))</span>,
[owned Model contract](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([owned Model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions))</span>.

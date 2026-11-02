# ADR 0046: Bounded general-scan cold source and owned Model direction

## Status

Accepted direction — 2026-10-23, explicit user approval of limited NN0030 and the first Compiler
prerequisite. Independent pre-code authority/inventory review is required before implementation.
This record explains [the authority root](../../../ARCHITECTURE.md); it is not another contract.

## Context and decision drivers

General scan already has source/IR, first-order Compiler differentiation and bounded F32/F64 CPU
execution. Owned Model.graph/defineTensor acquisition instead depends on cold Compiler constructors
and Checkpoint normalization. The latter currently omits scanBody; CompiledNode rejects that omission
before host copying. Execution capability alone cannot safely unlock owned acquisition.
Preserve one ordinary Tensor graph, private weight set and existing Engine/TrainingSession owners;
separate Model semantics, reconstructibility, execution qualification and wire compatibility.

## Options considered

- Keep owned general scans rejected: safe but does not deliver the requested ordinary Model workflow.
- Add an NN-local clone/runner/session: duplicates Compiler/ownership responsibilities; rejected.
- Extend bounded Compiler constructors, then Checkpoint in-memory copy/rebind, then qualify NN:
  selected; avoids a second pipeline while retaining fail-closed consumer and wire boundaries.

## Decision and rationale

Use existing Model.graph/defineTensor → compute/configureTraining/train. First tested execution
is all-valid canonical static homogeneous F32/F64; no padding or fixed-scan BPTT. Cold admission
uses the exact [Compiler inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction),
not a CPU dtype/layout predicate or an all-PURE promise. Preserve unchanged flat outer admission
except GENERAL; explicit body-only mappings use current public constructors and applicable Model
dtypes. Whole outer/all-body preflight precedes all formals/producers; tables are call-local per scope.

## Consequences and follow-up

Compiler-only work must keep Checkpoint/NN rejected before host/file/setup effects; a test pins
existing missing-attachment rejection. A later Checkpoint capability preserves attachments/local
IDs for copy/rebind and independently rejects bodies in the closed v1/v2/v3 wire before effects.
Only then qualify NN private acquisition, cached compute and first-order training. This costs
serial consumer validation and intentionally leaves some valid PURE bodies cold-unsupported.
No new production type/API/dependency or runtime owner is selected. Physical byte/resource limits
stay with Checkpoint/backend; raw unresolved outer layouts stay valid. Fixed scan training guards,
builders and formats remain. Padded NN, low/vector/parallel/OpenBLAS execution and body persistence
require separate decisions. Final program join owns one native root test/Javadoc checkpoint.

## Related documentation

[NN0030 decision](../../planning/extensions/nn/tasks/0030-general-scan-owned-model-admission-decision.md),
[Compiler0012 brief](../../planning/modules/compiler/tasks/0012-general-scan-cold-source-reconstruction.md),
[scan scopes](../../architecture/general-differentiable-scan.md#source-expressions-and-independent-scopes),
[owned Model contract](../../architecture/contracts/extensions-training.md#owned-tensor-models).

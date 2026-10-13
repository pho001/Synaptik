# Implementation Roadmap

## Purpose and authority

This is the active index for area status, dependencies and authorized frontiers, not an
architecture contract. [ARCHITECTURE.md](../../ARCHITECTURE.md) and its incorporated scoped
contracts govern implementation. Read the owning master plan and compact Ready brief for scope;
completed evidence belongs in linked task records and Git history.

## Execution policy

- Each independent workstream may expose one authorized Ready frontier when dependencies are met.
- Verify status, exact contracts, base, conflicts, write/semantic ownership, integration order and
  validation before launch. Use isolated worktrees for concurrent writers.
- Table order is priority, not a hidden dependency. Draft is not execution authorization.
- Classes B/C use clean implementation contexts and the required independent review. Reuse
  successful focused evidence; run one combined checkpoint after ordered integration.
- Follow the [planning guide](planning-guide.md); never derive architecture from old task prose.

## Current priority: repository review repairs

The user authorized [project review 2026-10-13](project-review-2026-10-13.md) against production
base `728680ea`. Its [finding report](project-review-2026-10-13-findings.md) distinguishes
reproduced/source-confirmed bugs, coverage regressions, planning drift, intentional limitations,
and unmeasured risks. Repair existing behavior before starting new capability.

| Workstream | Authorized task | Current state | Dependency / integration |
|---|---|---|---|
| Compiler | [0006D](modules/compiler/tasks/0006d-autograd-audit-correctness-repairs.md) | Review needed | Correct batched matvec gradients and fail-closed derivative paths. |
| Prepare | [0009](modules/prepare/tasks/0009-complete-resource-rollback-on-duplicate.md) | Review needed | Complete rollback of all returned unique resources. |
| Training | [0002C](extensions/training/tasks/0002c-adaptive-update-operation-order.md) | Review needed | Preserve documented Adam arithmetic order. |
| NN/build | [0025B](extensions/nn/tasks/0025b-public-model-dependency-export.md) | Review needed | Export existing public Model edge; isolated consumer validation. |
| Tuning | [0005](tools/tuning/tasks/0005-model-plan-cache-encoding-bound.md) | Review needed | Preflight aggregate cache size before encoding allocation. |
| CPU | [0021](backends/cpu/tasks/0021-composed-value-validation-and-loss-bindings.md) | Review needed | F16 softmax, produced-value validation and composed loss bindings. |
| CPU successor | [0022](backends/cpu/tasks/0022-activation-extremes-and-low-working-precision.md) | In progress | After reviewed/integrated 0021: activation extremes and low F32 working sites; no simultaneous CPU writer. |
| Metal | [0078](backends/metal/tasks/0078-native-boundary-and-low-value-audit-repairs.md) | Review needed | Native metadata/context and low exact-value repairs; source-matched device validation. |
| Metal successor | [0079](backends/metal/tasks/0079-scalar-provider-preflight-alignment.md) | In progress | After reviewed/integrated 0078: align advertised scalar RELU/MUL with preflight and prove actual-device execution. |
| Engine/tests | [0019](modules/engine/tasks/0019-lifecycle-and-mixed-owner-test-repair.md) | Review needed | Deterministic copy/close, genuine CPU↔Metal tests and tuning Javadoc. Final device run uses integrated Metal. |

Parallel group: `PROJECT-REVIEW-2026-10-13-REPAIRS`. Coordinator owns roadmap, master plans,
report and final checkpoint. Exact launch bases and scopes are in each brief. Required reviews
precede integration; shared authoritative version facts are integrated serially.

## Area map

No row below grants implementation permission beyond the active repair briefs above.

| Area | Delivered baseline / remaining gate |
|---|---|
| [Model](modules/model/master-plan.md) | Complete through 0032, active profile-free semantics and FLOAT16. Obsolete 0026 is Superseded. |
| [Trace](modules/trace/master-plan.md) | Envelope/IDs and preparation/invocation/low-route facts delivered; broader payloads/encoding Draft. No numerical-profile DTO. |
| [Backend Contract](modules/backend-contract/master-plan.md) | Complete through 0004; reopen only for a concrete shared need. |
| [Config](modules/config/master-plan.md) | Compile leaves and two-phase tuning policy delivered. Cost consumer/aggregate/run configuration Draft; no numerical-profile API. |
| [Planning](modules/planning/master-plan.md) | Complete through 0007; cost model awaits a concrete consumer and units. Partition size is not fusion size. |
| [Runtime](modules/runtime/master-plan.md) | Complete through 0016; per-run isolation and prepared-resource leases remain current. |
| [Compiler](modules/compiler/master-plan.md) | Complete through 0006B11 plus active repair. Conv3d adjoints 0006C and exact rewrites 0007 Draft; BPTT unsupported. |
| [Prepare](modules/prepare/master-plan.md) | Complete through 0008 plus active repair; opaque backend collaboration, complete resource transaction. |
| [OpenBLAS provider](backends/openblas-provider/master-plan.md) | 0001–0003 Complete. Optional direct BF16 0004 Blocked on ABI and one-final-narrowing evidence; F32/F64 unaffected. |
| [CPU](backends/cpu/master-plan.md) | Complete through 0020 plus repairs. 0010M retained scalar/1/1/1 defaults; vendor peers remain use-case/ABI gated. |
| [Engine](modules/engine/master-plan.md) | Public composition, sessions, compute/backward, host materialization and bounded tuning delivered; profile spine is historical after Model 0032. |
| [Metal](backends/metal/master-plan.md) | Complete through 0077 plus repair. Current occurrence support is independently typed; hosted GPU CI 0047 remains blocked. |
| [CUDA](backends/cuda/master-plan.md) | Marker/native scaffold only; no executable CUDA backend. 0001 Draft. |
| [ONNX](extensions/onnx/master-plan.md) | Marker only; no importer/exporter. Area Draft; first mapping task not yet defined. |
| [Data](extensions/data/master-plan.md) | No module. 0001 architecture decision precedes batching; current plan excludes sampler/epoch lifecycle. |
| [NN](extensions/nn/master-plan.md) | State/layers/static recurrence/lazy initialization/convolution delivered. Runtime scan/backend/Data program 0021B–0024 Draft. |
| [Text](extensions/text/master-plan.md) | No module. Depends on Data architecture, then selected tokenizer/artifact/batching scope. |
| [Vision](extensions/vision/master-plan.md) | No module. Shares Data 0001 architecture decision before decoder/preprocessing work. |
| [Training](extensions/training/master-plan.md) | Sessions, SGD/Adam/AdamW groups and in-memory state delivered. Exact-resume ownership decision 0003A precedes 0003. |
| [Checkpoint](extensions/checkpoint/master-plan.md) | No module/codec. Architecture and model artifact precede optional Training adapter; existing Engine host access is available. |
| [Benchmarks](tools/benchmarks/master-plan.md) | Fixed CPU/Metal report-only harnesses delivered; broader workload suites Draft. No universal performance claim. |
| [Tuning](tools/tuning/master-plan.md) | Local/complete-plan tuning and read-only inspection delivered, plus cache-bound repair. |
| [CLI](tools/cli/master-plan.md) | Marker only; commands await selected concrete Engine/diagnostic workflow. |

## Proposed next sequence after repair closure

1. Close each independent review and integrate scoped repairs. Run one source-matched combined
   test/Javadoc checkpoint, explicitly recording native execution vs skips.
   Resolve the separate [Model 0033 softmax decision](modules/model/tasks/0033-stable-softmax-overflow-semantics.md)
   before claiming numerical-contract closure: current stable computation and literal EXP-overflow
   wording disagree. It is Draft, not an implicit authorization to change numerics.
2. Keep follow-up focused on a real user capability. Recommended next planning decision:
   Training 0003A selects exact-resume participants and caller-owned progress/sampler handoff.
   Do not invent a DataLoader or silently broaden Data to satisfy an unnamed prerequisite.
3. If durable save/load is the next product priority, prepare Checkpoint 0001's module decision
   and model-only 0002–0004. Existing Engine materialization is not a missing prerequisite.
4. In a separately selected stream, Data 0001 authorizes Data/Text/Vision architecture before
   batching or text/image APIs. Runtime recurrent adoption still needs a concrete backend route.

Steps 2–4 are proposals, not Ready implementation tasks. The Training/Checkpoint order is:
`Training 0003A → 0003` and `Checkpoint 0001 → 0004`, then
`Training 0004 → Checkpoint 0005 → Checkpoint 0006`. No adapter-to-prerequisite cycle remains.

## Explicit non-goals and unresolved decisions

- Do not restart historical profile/certificate/proof gates as prerequisites for current work.
- Do not equate operation-kind coverage with all dtypes/layouts/gradients/strategies or speed.
- No broad performance campaign, new native provider, automatic backend discovery, persistence
  module, rollback-mode change, or new numerical policy is authorized by this review.
- GPU CI provisioning, large-model aggregate memory policy, new vendor routes, recurrent execution
  and missing adjoints retain their explicit decisions and owning plans.

## History

The 2026-10-13 reconciliation removed duplicated completion narratives and stale global-frontier
claims from this index. Completed and historically blocked task evidence remains in its original
briefs and Git history. Future updates change current status, dependencies and next gates only.

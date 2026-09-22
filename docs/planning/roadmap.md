# Implementation Roadmap

## Purpose and authority

This roadmap is the active index for project order, area status, blockers, and the current
implementation frontier. It is not an architecture contract. [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
is authoritative and wins if a planning document conflicts with it.

Read an area's linked master plan for its ordered queue and the current task brief for executable
scope. Completed-task narratives and validation evidence do not belong in this index.

## Execution policy

- Advance one active frontier at a time by default.
- Before launch, the planner verifies that the task is `Ready`, is at the authorized frontier,
  and has satisfied dependencies, then records that verification in the task brief.
- Follow task-table order within an area and roadmap order across areas unless an explicit
  exception records non-overlapping dependencies and files plus an integration plan.
- Parallel or out-of-order work is never implied by a task being independent or by a master plan
  existing. Record each exception in the owning master plan and this roadmap while it is active.
- Allowed and forbidden dependencies remain defined only by `ARCHITECTURE.md`.

## Ordered project areas

The order is the default delivery sequence. “Next gate” identifies the first unfinished decision
or task boundary; it does not promote Draft work to Ready.

| Order | Project area | Current status | Entry or next gate |
|---:|---|---|---|
| 1 | [`modules/model`](modules/model/master-plan.md) | Selected scope Complete through 0025L; 0026 Draft | Select 0026 only when IEEE FLOAT16 and mixed-precision semantics become current. |
| 2 | [`modules/trace`](modules/trace/master-plan.md) | In progress, deliberately interleaved; 0001–0002 Complete, 0003–0008 Draft | Resume 0003 only after its producer vocabulary is stable; no Trace task is Ready. |
| 3 | [`modules/backend-contract`](modules/backend-contract/master-plan.md) | Complete through 0004 | Reopen only for a concrete shared-contract need. |
| 4 | [`modules/config`](modules/config/master-plan.md) | In progress, interleaved; 0001–0003 and 0006A–0006B Complete; 0004–0006 and 0007–0008 Draft | 0004 waits for a concrete cost-bearing Planning consumer; no Config task is Ready. |
| 5 | [`modules/planning`](modules/planning/master-plan.md) | Complete through 0006 | A separate reassessment must select any later cost-bearing frontier. |
| 6 | [`modules/runtime`](modules/runtime/master-plan.md) | Complete through 0016 | No Runtime task is Ready. |
| 7 | [`modules/compiler`](modules/compiler/master-plan.md) | Complete through 0006B7; 0006C and 0007 Draft | 0006C needs proved Conv3d adjoint expressibility; relaxed 0007 work also waits for Config 0006. |
| 8 | [`modules/prepare`](modules/prepare/master-plan.md) | Complete through 0006 | No Prepare task is Ready. |
| 9 | [`backends/openblas-provider`](backends/openblas-provider/master-plan.md) | Required baseline Complete; optional 0004 Blocked and deferred | Resume 0004 only when both direct-BFLOAT16 application binary interface (ABI) and one-final-narrowing proofs exist. |
| 10 | [`backends/cpu`](backends/cpu/master-plan.md) | Mainline Complete through 0010J; 0007A1D Review needed; 0010D1 and 0011 Blocked | No CPU task is Ready; vendor peers 0012–0015 and integrations 0016–0017 remain Draft. |
| 11 | [`modules/engine`](modules/engine/master-plan.md) | Complete through 0010 | No later Engine task is Ready or detailed. |
| 12 | [`backends/metal`](backends/metal/master-plan.md) | Complete through 0004 | Reassess exactly one next frontier; no later Metal task is Ready. |
| 13 | [`backends/cuda`](backends/cuda/master-plan.md) | Draft | Create a detailed 0001 brief only when CUDA becomes the authorized frontier. |
| 14 | [`extensions/onnx`](extensions/onnx/master-plan.md) | Draft | Define the first bounded mapping task only at an authorized frontier. |
| 15 | [`extensions/data`](extensions/data/master-plan.md) | Draft; architecture decision required | 0001 must authorize the Data/Text/Vision modules, build edges, decision record, and architecture tests first. |
| 16 | [`extensions/nn`](extensions/nn/master-plan.md) | In progress under recorded interleaves; 0001–0020C, 0021A, and 0025–0025A Complete; 0021B–0024 Draft | 0021B needs a detailed brief and truthful concrete-backend recurrent coverage; no NN task is Ready. |
| 17 | [`extensions/text`](extensions/text/master-plan.md) | Draft; architecture decision required | Wait for Data 0001, then define the first tokenizer task. |
| 18 | [`extensions/vision`](extensions/vision/master-plan.md) | Draft; architecture decision required | Join the coordinated Data 0001 decision before decoder or image APIs. |
| 19 | [`extensions/training`](extensions/training/master-plan.md) | Draft | Begin only after stable NN parameters, published gradients, and Engine training execution justify 0001. |
| 20 | [`extensions/checkpoint`](extensions/checkpoint/master-plan.md) | Draft; architecture decision required | Authorize the model-only and optional Training adapter boundaries before 0001. |
| 21 | [`tools/benchmarks`](tools/benchmarks/master-plan.md) | Draft | Define 0001 only for stable operational paths and workload contracts. |
| 22 | [`tools/tuning`](tools/tuning/master-plan.md) | Complete through 0003 | Reopen only for a concrete producer or public-composition need. |
| 23 | [`tools/cli`](tools/cli/master-plan.md) | Draft | Define commands only after their Engine and diagnostic contracts are stable. |

## Current frontier

[`Metal 0004`](backends/metal/tasks/0004-typed-metal-route-candidate-generators-and-cache-compatibility.md)
is `Complete`. It added the Metal-local typed complete route-candidate and session-compatible
decision-codec foundation without adding outer tuning/cache integration. No new task is `Ready`;
the next authorized frontier requires a planning reassessment. The recorded NN interleave remains
an explicit exception, but it does not promote NN 0021B–0024; each new interleaved task still
requires specific authorization and non-overlap evidence. The completed Config 0006A/0006B staged
exceptions authorize no additional Config work.

## Blocked, review-needed, and deferred work

- [OpenBLAS provider 0004](backends/openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md)
  and dependent CPU 0010D1 remain an optional blocked branch. Pinned evidence proves neither the
  required exported direct BFLOAT16-output ABI nor full-contraction FLOAT32 accumulation followed
  by one final BFLOAT16 narrowing. Existing FLOAT32/FLOAT64 and portable BFLOAT16 work remain
  unaffected.
- CPU 0011 remains `Blocked` in the [CPU master plan](backends/cpu/master-plan.md) until a concrete
  Intel workload and supported oneMKL interface evidence both exist.
- CPU 0007A1D remains `Review needed`, not an active frontier; its stable semantic and structural
  evidence does not satisfy its failed performance gate.
- Model 0026 and Compiler 0006C/0007 are independent Draft side branches in their
  [Model](modules/model/master-plan.md) and [Compiler](modules/compiler/master-plan.md) plans.
  They gate only the capabilities named by those rows and do not block Metal 0004.
- Data, Text, and Vision require the coordinated decision owned by
  [Data 0001](extensions/data/master-plan.md); Checkpoint requires its separate decision in the
  [Checkpoint plan](extensions/checkpoint/master-plan.md). Planning authorization alone does not
  create these Gradle projects or permit implementation.

## Nearest next step

1. Reassess exactly one next frontier; do not promote a Draft task without its required planning
   and dependency audit.

## History policy

Completed task narratives, prior validation evidence, agent context identifiers, retrospectives,
and old reorder stories remain historical evidence in Git and task files. They are not default
executor input. Read historical evidence only when a current brief explicitly identifies an
unresolved question that requires it. Do not reconstruct global history before executing a
verified current brief.

## Update policy

Update this roadmap only when project order, project-area status, the active frontier, a blocker,
or an explicit ordering exception changes. Keep detailed scope and result evidence in task files;
keep master plans and this roadmap to links and one-line status or gate summaries. If a planning
change conflicts with `ARCHITECTURE.md`, stop and use the architecture-decision process instead of
changing this roadmap alone.

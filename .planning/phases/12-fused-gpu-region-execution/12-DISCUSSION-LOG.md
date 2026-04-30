# Phase 12: Fused GPU Region Execution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30T19:01:24Z
**Phase:** 12-fused-gpu-region-execution
**Areas discussed:** GPU fusion unit model, compound representation, existing FUSED handling, first fused patterns, Metal/CUDA parity, fallback and trace contract

---

## GPU Fusion Unit Model

| Option | Description | Selected |
|--------|-------------|----------|
| Compound accelerator DAG | Fused GPU region is a longer Metal/CUDA partition lowered as a multi-node accelerator DAG. | |
| GPU-specific `FUSED` handling | Planner creates a special GPU fused execution unit similar in shape to CPU fusion but with separate backend implementation. | |
| Region compound lowering | Region can first fuse/canonicalize operations and then lower the compound representation to backend primitives. | ✓ |

**User's choice:** Region-level lowering pipeline where a region can both fuse operations and lower them to primitives.
**Notes:** The user clarified that the desired model is not simply sending existing ops as a DAG; the region may identify compound patterns first, then lower to Metal/CUDA primitives.

---

## Compound Representation

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit pattern types | Add pattern identifiers such as `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, or `REDUCTION_ADJACENT`. | |
| Generic accelerator DAG only | Do not add pattern summaries; rely only on raw DAG nodes. | |
| Pattern summary plus DAG | Use pattern summaries for trace/cost/fallback and the DAG for execution lowering. | ✓ |

**User's choice:** Pattern summary plus DAG.
**Notes:** This balances auditability with the existing accelerator DAG execution model.

---

## Existing FUSED Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Explicitly reject for GPU | Keep CPU `FUSED` CPU-only; GPU compound regions arise from normal ops and reject `FUSED` visibly. | ✓ |
| Safely expand selected subset | Decode simple CPU fused elementwise chains back into a GPU compound DAG. | |
| Use `FUSED` as GPU compound source | Share `FUSED` region representation between CPU and GPU lowerers. | |

**User's choice:** Explicitly reject `Operation.OpType.FUSED` for GPU in Phase 12.
**Notes:** CPU fused ASM/vector internals must remain independent from GPU compound region execution.

---

## First Fused Patterns

| Option | Description | Selected |
|--------|-------------|----------|
| `linear + bias + activation` first | Focus on the roadmap's linear hot path first. | |
| Elementwise chains first | Start with simpler pure elementwise chains. | |
| Two minimal patterns in parallel | Implement one `linear + bias + activation` target and one representative elementwise chain. | ✓ |

**User's choice:** Two minimal patterns in parallel.
**Notes:** This covers both `GPUFUSE-01` and `GPUFUSE-02`; reduction-adjacent candidates remain third priority.

---

## Metal/CUDA Parity

| Option | Description | Selected |
|--------|-------------|----------|
| Same minimum pattern for both | Phase 12 passes only when Metal and CUDA support the same minimal patterns. | |
| Metal broader, CUDA minimum | Metal may be wider via MPSGraph; CUDA must provide a stable minimum subset or capability-gated path. | ✓ |
| Backend-specific without parity | Each backend implements what it can; phase pass depends mostly on trace/fallback contracts. | |

**User's choice:** Metal may be broader, CUDA must provide a stable minimum subset.
**Notes:** The phase should not claim broad CUDA coverage unless the provider can actually execute it.

---

## Fallback And Trace Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Only hidden fallback fails | Visible fallback is acceptable. | |
| CPU materialization between fused ops fails | Supported fused patterns must not round-trip through CPU between fused ops. | |
| Region shortening also fails | If a supported target pattern is promised, shortening it to a smaller GPU region plus CPU remainder fails. | ✓ |

**User's choice:** Region shortening is also a failure for explicitly supported target patterns.
**Notes:** Unsupported candidates may still fallback visibly, but supported target workloads must stay whole on GPU.

---

## the agent's Discretion

- Exact class names and package placement for the compound pattern summary are left to planning.
- Planner can decide where pattern recognition sits, as long as it extends the existing lowering pipeline and remains trace-visible.

## Deferred Ideas

- Phase 13 owns benchmark coverage ratios, workload comparisons, and regression gates.
- Full universal accelerator provider coverage remains future scope.
- Sharing CPU fused internals with GPU compound execution is deferred.

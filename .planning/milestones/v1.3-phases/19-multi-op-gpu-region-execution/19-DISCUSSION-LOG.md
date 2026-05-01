# Phase 19: Multi-Op GPU Region Execution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-01
**Phase:** 19-Multi-Op GPU Region Execution
**Mode:** `--auto`
**Areas discussed:** Region execution granularity, Internal device handoff semantics, Primitive coverage priority, Backend contract, Coverage evidence

---

## Region Execution Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| One backend-owned lowered region | Keep one selected GPU partition as one backend-owned native DAG/executable when internal primitives are supported. | ✓ |
| Per-op prepared GPU steps | Decompose the selected region into Java-visible prepared GPU steps with explicit handoffs after each operation. | |
| CPU-interleaved fallback inside region | Let unsupported internal operations execute on CPU inside a selected GPU region. | |

**User's choice:** `[auto] Selected recommended default: one backend-owned lowered region.`
**Notes:** Unsupported internal steps should split, shorten, or reject before execution with manifest-backed reasons.

---

## Internal Device Handoff Semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Keep internals device-owned | Supported internal values stay inside native DAG/device bindings until true CPU consumer, graph output, or gradient publication. | ✓ |
| Materialize after each primitive | Read intermediate values back to CPU after each internal primitive for simpler orchestration. | |
| Add public device tensors | Expose device residency directly through public `Tensor` API. | |

**User's choice:** `[auto] Selected recommended default: keep internals device-owned.`
**Notes:** Public `Tensor` remains logical. `ExecutionState` and device buffer bindings own residency.

---

## Primitive Coverage Priority

| Option | Description | Selected |
|--------|-------------|----------|
| Existing supported primitives plus hot paths | Focus on matmul/linear, layout/view, elementwise, epilogues, and softmax-ish paths first. | ✓ |
| Broad normalization/reduction/conv implementation | Attempt wide native support for normalization, reductions, conv, and loss-adjacent flows in this phase. | |
| Vendor library routing | Introduce cuBLAS/cuDNN/MPSGraph-style routing and backend primitive cost selection now. | |

**User's choice:** `[auto] Selected recommended default: existing supported primitives plus measured hot paths.`
**Notes:** Normalization/reduction/conv/loss-adjacent gaps consume Phase 17 support or stable rejection unless a narrow proven implementation is safe.

---

## Backend Contract

| Option | Description | Selected |
|--------|-------------|----------|
| Shared planning contract, backend-specific execution | Keep planning/manifest/trace backend-neutral while Metal and CUDA execute through their own bridge/executable paths. | ✓ |
| Metal-first contract fork | Let Metal-specific runtime behavior define the shared contract and retrofit CUDA later. | |
| Native ABI broadening by default | Treat Phase 19 as a native ABI expansion phase. | |

**User's choice:** `[auto] Selected recommended default: shared planning contract with backend-specific execution.`
**Notes:** Native ABI changes must be additive, capability-gated, and portable-test-covered if they become necessary.

---

## Coverage Evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 14 targets with explicit residency metrics | Use transformer, MLP, conv, and layer-norm targets with region length, lowered primitive count, fused count, materialization, handoff, and runtime path evidence. | ✓ |
| Timing-only comparison | Use benchmark speedup as the primary proof that Phase 19 worked. | |
| Transformer-only proof | Prove only one workload and leave other v1.3 targets uninspected. | |

**User's choice:** `[auto] Selected recommended default: Phase 14 targets with explicit residency metrics.`
**Notes:** `transformer_block_hot_path` and `mlp_classifier_small` are implementation priorities; conv/norm can remain visible blockers if outside safe Phase 19 execution scope.

---

## the agent's Discretion

- Exact class decomposition and helper naming.
- Whether to add narrow helper records or extend existing trace/report records.
- Exact focused test slicing, provided it proves CPU parity, visible fallback, and source hygiene.

## Deferred Ideas

- Vendor library routing and backend-native primitive cost model (`GPULIB-*`).
- Universal reductions, normalizations, convolutions, dynamic shapes, sparse, high-rank, and advanced indexing.
- Public GPU tensor/device API.
- Treating tensor-array bridge execution as native buffer GPU coverage.
- Native CUDA hardware evidence as canonical proof rather than capability-skipped portable evidence.

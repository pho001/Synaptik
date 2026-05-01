# GPU Lowered Region Manifest

## Purpose

`GpuLoweredRegionManifest` describes one selected GPU region as Java-side lowered DAG metadata. It exists so traces, benchmark reports, and later coverage gates can explain which original graph operations were selected, which backend primitives they lowered into, which dtype/layout/storage assumptions were required, and why candidate pieces were rejected or shortened.

GpuLoweredRegionManifest is Java-side metadata and does not change the Metal or CUDA native ABI.

Public Tensor remains logical; device residency lives in compile/prepare/execute runtime state.

## Manifest Fields

The manifest records a stable `regionId`, selected `backend`, anchor node id, ordered original node ids, external input node ids, output node ids, selected region length, original op metadata, lowered primitive metadata, input/output value assumptions, fused subpattern metadata, rejection evidence, candidate-span evidence, and backend extension metadata.

Backend-specific data belongs in `backendExtensions` unless it becomes part of the shared accelerator contract.

## Original Ops And Lowered Primitives

`originalOps` maps compiled graph nodes to the primitive ids created for the selected region. A high-level op may map to more than one primitive. For example, `LOG_SOFTMAX` can lower to `SOFTMAX` followed by `LOG` without adding a new native ABI op code.

`loweredPrimitives` records deterministic primitive ids such as `p0`, `p1`, and `p2`, primitive type, source original node ids, input references, output reference, dtype, shape, and primitive-level reason codes.

## Value Assumptions

Input and output assumptions capture dtype, rank, shape, layout category, contiguity, storage-offset presence, and storage offset. These assumptions are trace/report evidence, not a public tensor residency API.

## Fused Subpattern Placeholder

The manifest embeds existing `GpuCompoundRegionSummary` data as region-internal fused subpattern metadata. GPU fusion metadata is region-internal lowering/fusion metadata, not CPU Operation.OpType.FUSED.

CPU `Operation.OpType.FUSED` remains CPU-only. GPU regions are selected from normal graph operations and lowered into backend-supported accelerator primitives.

## Rejection And Shortening Reasons

Manifest rejection records attribute stable reason codes to original ops, primitives, fused subpatterns, or region boundaries. Phase 15 reason codes include:

- `DAG_PRIMITIVE_UNSUPPORTED`
- `DAG_REGION_BOUNDARY_MATERIALIZATION`
- `DAG_CANDIDATE_SHORTENED`
- `DAG_FUSED_SUBPATTERN_REJECTED`

`candidateSpan` records which original candidate nodes were considered, which were accepted, and the first rejected original node or primitive when the region was shortened.

## Trace Contract

Prepare/backend-selection trace is the source of truth for the structured manifest.

Benchmark text reports render a compact `GPU Lowered Region` block for selected decisions. Benchmark JSON reports expose the structured `gpuLoweredRegionManifest` object with stable keys for later regression gates.

Run trace references the region id and runtime outcome without duplicating the full manifest.

## Backend Boundaries

The manifest is backend-neutral. Metal and CUDA selected plans expose it through the shared `PartitionPlan.gpuLoweredRegionManifest()` contract, while prepared executables expose it only for trace metadata.

Backend planners and native bridges still own dtype, layout, capability, runtime availability, buffer binding, and ABI checks. A manifest does not bypass fallback, CPU materialization, or required-mode failure behavior.

Native Metal/CUDA ABI unchanged.

Public Tensor API unchanged.

## Phase Handoff

Phase 16 and later dtype/storage work can use value assumptions and backend extensions to decide which residency gaps to close. Phase 17, Phase 18, and Phase 19 can use original-op and primitive mappings for lowering and fusion expansion. Phase 20 can consume the stable trace/report fields to enforce coverage regression gates without parsing ad hoc text.

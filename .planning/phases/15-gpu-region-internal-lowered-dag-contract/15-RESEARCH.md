# Phase 15 Research: GPU Region Internal Lowered DAG Contract

**Phase:** 15 - GPU Region Internal Lowered DAG Contract
**Date:** 2026-05-01
**Status:** Complete

## RESEARCH COMPLETE

## Planning Question

What needs to be known to plan Phase 15 well?

Phase 15 should define the Java-side manifest that describes a selected GPU region as an internal lowered DAG. The phase should make selected region structure, original-operation mapping, lowered primitive mapping, dtype/layout/storage assumptions, fused subpattern placeholders, and rejection/materialization attribution visible in stable trace/report fields. It should not broaden operation coverage, add new dtype residency, change native Metal/CUDA ABI, or expose a public GPU tensor API.

## Current Architecture

### Lowered DAG Inputs Already Exist

- `AcceleratorDagSpec` is the backend-neutral DAG passed to native graph bridges. It contains external inputs, topologically ordered `AcceleratorDagNode` values, output node indexes, and output node ids.
- `AcceleratorSubgraphLowerer` lowers an `AcceleratorSubgraphSpec` to an `AcceleratorSubgraphLoweringResult`. It already supports generic ordered DAG lowering plus specialized expansions such as `LOG_SOFTMAX` to `SOFTMAX` followed by `LOG`.
- `AcceleratorSubgraphLoweringResult` currently carries the compute node id, optional legacy matmul descriptor, `AcceleratorDagSpec`, estimated work, and `GpuCompoundRegionSummary`.
- `GpuCompoundRegionSummary` already describes supported and unsupported compound/fusion-like patterns. Phase 15 should embed it as fused-subpattern summary metadata, not replace it.

### Region And Backend Selection Anchors Already Exist

- `LoweredRegion` and `LoweredExecutionUnit` provide an existing internal region/unit vocabulary that can inspire region ids, lowering families, and ordered node lists.
- `DefaultBackendSelectionPolicy` creates `BackendSelectionDecisionTrace` entries for every selected and rejected candidate. That is the right prepare-time place to attach selected-region manifest metadata.
- `BackendSelectionTrace` is immutable and already consumed by coverage summaries and benchmark report renderers.
- `PreparedExecution` run traces publish accelerator execution attributes from prepared executables. That layer should reference the selected region id and runtime outcome if needed, but should not duplicate the full prepare-time manifest.

### Existing Observability Patterns

- Trace records are Java records with null-normalizing compact constructors.
- Report renderers use stable text headings and JSON field names that tests assert exactly.
- Unsupported GPU behavior uses `GpuLoweringUnsupportedReason` instead of free-form strings where a stable matrix reason is needed.
- Coverage and benchmark reports already summarize backend selection and CPU materialization. Phase 15 should add fields that later Phase 20 gates can consume without parsing ad hoc text.

## Gaps To Close

### Gap 1: No Region Manifest Around The Lowered DAG

`AcceleratorDagSpec` is a native bridge contract, not a complete planning/debug manifest. It does not record region id, backend id, selected region length, original op summaries, bidirectional original-op to lowered-primitive mapping, dtype/layout/storage assumptions, fused subpattern summary, backend extension metadata, or candidate-shortening evidence.

Phase 15 should add a Java-side wrapper, tentatively `GpuLoweredRegionManifest`, that references `AcceleratorDagSpec` and associated metadata without adding debug/report fields to the native DAG ABI.

### Gap 2: Lowered Primitive Attribution Is Not Explicit

Current `AcceleratorDagNode.nodeId()` can point back to an original compiled node, but specialized expansions such as `LOG_SOFTMAX` producing `SOFTMAX` plus `LOG` need first-class metadata:

- one original op can produce multiple lowered primitives,
- one lowered primitive can reference a source op or source span,
- original ops need aggregate lowered primitive ids for human/debug readability,
- failing primitives need their own primary reason when a multi-primitive expansion fails.

### Gap 3: DType/Layout/Storage Assumptions Are Not A Manifest Contract

Legality and lowerers inspect dtype, shape, layout, storage offset, and contiguity, but that evidence is not packaged per region input, primitive, and output. Phase 15 should expose assumptions as manifest values so later dtype/storage and hardening phases can detect why a region stayed on GPU or exited.

### Gap 4: Rejection And Shortening Need Stable Levels

Rejected backend candidates currently have one decision reason string. Phase 15 needs stable attribution levels:

- original operation,
- lowered primitive,
- fused subpattern,
- region boundary.

Candidate shortening should record original candidate span, accepted span, rejected original node or primitive, and reason. That makes shortened hot paths auditable later.

## Recommended Implementation Shape

### Manifest Model

Add backend-neutral manifest records under `backend.accelerator.lowering`:

- `GpuLoweredRegionManifest`
- `GpuLoweredRegionOriginalOp`
- `GpuLoweredPrimitiveManifest`
- `GpuLoweredRegionValueAssumption`
- `GpuLoweredRegionRejection`
- `GpuLoweredRegionCandidateSpan`

The top-level manifest should include:

- `String regionId`
- `ComputeBackend backend`
- `int anchorNodeId`
- `List<Integer> orderedNodeIds`
- `List<Integer> externalInputNodeIds`
- `List<Integer> outputNodeIds`
- `int selectedRegionLength`
- `List<GpuLoweredRegionOriginalOp> originalOps`
- `List<GpuLoweredPrimitiveManifest> loweredPrimitives`
- `List<GpuLoweredRegionValueAssumption> inputAssumptions`
- `List<GpuLoweredRegionValueAssumption> outputAssumptions`
- `GpuCompoundRegionSummary fusedSummary`
- `List<GpuLoweredRegionRejection> rejections`
- `GpuLoweredRegionCandidateSpan candidateSpan`
- `Map<String, String> backendExtensions`

### Reason Vocabulary

Extend `GpuLoweringUnsupportedReason` only where stable DAG-level attribution needs new codes. Recommended exact additions:

- `DAG_PRIMITIVE_UNSUPPORTED`
- `DAG_REGION_BOUNDARY_MATERIALIZATION`
- `DAG_CANDIDATE_SHORTENED`
- `DAG_FUSED_SUBPATTERN_REJECTED`

Do not add free-form reason enums or backend-specific reason forks in Phase 15.

### Manifest Construction

Create manifest construction in the shared accelerator lowering path:

- `AcceleratorSubgraphLoweringResult` should carry `GpuLoweredRegionManifest manifest`.
- `AcceleratorSubgraphLowerer.tryLower(ComputeBackend, AcceleratorSubgraphSpec, PartitionPlanningContext)` should build the manifest after the `AcceleratorDagSpec` and `GpuCompoundRegionSummary` are available.
- The manifest should derive original ops from `subgraph.ops()` and `context.compiledNode(...)`.
- Lowered primitives should derive ids from the ordered `AcceleratorDagSpec.nodes()` index, such as `p0`, `p1`, or another deterministic stable string.
- `LOG_SOFTMAX` should prove one original op maps to two lowered primitives: `SOFTMAX` and `LOG`.
- Input and output assumptions should record node id, dtype, rank, shape, layout category (`CONTIGUOUS`, `STRIDED_VIEW`, `STORAGE_OFFSET_VIEW`, or `UNKNOWN`), storage offset, and contiguous flag.

### Trace And Rendering

Attach the structured manifest to selected `BackendSelectionDecisionTrace` entries. Keep constructor overloads so existing tests and synthetic traces do not need wide churn.

Add a compact renderer, likely `GpuLoweredRegionManifestRenderer`, with stable text headings:

- `GPU Lowered Region`
- `Original Ops`
- `Lowered Primitives`
- `Value Assumptions`
- `Fused Subpatterns`
- `Rejections`

Benchmark JSON/text renderers should include selected manifests under stable fields only for selected GPU decisions. Later Phase 20 gates should consume these fields, not scrape prose.

### Runtime Boundary

Run trace should only reference selected region id and runtime outcome if needed. The full manifest belongs in prepare/backend-selection trace because it is produced during planning/lowering. Public `Tensor` remains logical and CPU fusion remains CPU-owned.

## Validation Architecture

### Automated Sampling

- After model work, run `./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest`.
- After lowerer integration, run `./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest`.
- After trace/report work, run `./gradlew test --tests CompiledGraphTraceTest --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest`.
- Before phase verification, run `./gradlew classes` plus the focused Phase 15 test slice and `git status --short`.

### Nyquist Targets

- Every implementation plan has focused automated tests.
- At least one test proves one original op expanding to multiple lowered primitives.
- At least one test proves selected backend trace contains stable manifest fields.
- At least one guard test proves public `Tensor` API and CPU execution/fusion semantics are unchanged.
- Verification must show local tuning profile files under `profiles/platform/.../tuning/abc/*` remain unstaged.

## Research Risks

- The manifest could leak debug/report concerns into `AcceleratorDagSpec`. Mitigation: keep a wrapper record and reference the DAG without changing native bridge semantics.
- Trace constructors could break many existing tests. Mitigation: add overloads and null-normalizing defaults.
- Reason attribution could drift back to strings. Mitigation: use `GpuLoweringUnsupportedReason` for stable reason codes and reserve detail strings only for auxiliary text.
- Phase 15 could accidentally implement broad execution coverage. Mitigation: plans explicitly exclude native ABI changes, new operation coverage, dtype residency expansion, and fused execution.

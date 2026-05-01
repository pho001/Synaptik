package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import graph.optimizer.partition.PartitionPlan;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import tensor.DataType;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * CUDA partition plan carrying the lowered accelerator DAG selected for execution.
 *
 * @param anchorNodeId node that triggers execution of this partition
 * @param subgraph original accelerator subgraph candidate
 * @param dagSpec lowered DAG consumed by the CUDA bridge
 * @param estimatedWork planner work estimate used by backend selection
 * @param compoundSummary stable compound GPU pattern summary for trace and preparation metadata
 * @param manifest Java-side lowered-region manifest used for trace/report metadata
 */
public record CudaGpuPartitionPlan(
        int anchorNodeId,
        AcceleratorSubgraphSpec subgraph,
        AcceleratorDagSpec dagSpec,
        long estimatedWork,
        GpuCompoundRegionSummary compoundSummary,
        GpuLoweredRegionManifest manifest
) implements PartitionPlan {
    public CudaGpuPartitionPlan {
        Objects.requireNonNull(subgraph, "subgraph cannot be null");
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        compoundSummary = compoundSummary == null
                ? GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, subgraph.orderedNodeIds())
                : compoundSummary;
        manifest = manifest == null
                ? defaultManifest(anchorNodeId, subgraph, dagSpec, compoundSummary)
                : manifest;
        if (!subgraph.orderedNodeIds().contains(anchorNodeId)) {
            throw new IllegalArgumentException("anchorNodeId must be part of nodeIds");
        }
    }

    public CudaGpuPartitionPlan(
            int anchorNodeId,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            long estimatedWork,
            GpuCompoundRegionSummary compoundSummary
    ) {
        this(anchorNodeId, subgraph, dagSpec, estimatedWork, compoundSummary, null);
    }

    public CudaGpuPartitionPlan(
            int anchorNodeId,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            long estimatedWork
    ) {
        this(anchorNodeId, subgraph, dagSpec, estimatedWork, null);
    }

    /**
     * Returns {@link ComputeBackend#GPU_CUDA}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

    /**
     * Returns the node ids covered by this partition in execution order.
     */
    @Override
    public List<Integer> nodeIds() {
        return subgraph.orderedNodeIds();
    }

    /**
     * Returns node ids read from outside the partition.
     */
    @Override
    public List<Integer> externalInputNodeIds() {
        return subgraph.externalInputNodeIds();
    }

    /**
     * Returns node ids produced by this partition.
     */
    @Override
    public List<Integer> producedOutputNodeIds() {
        return subgraph.outputNodeIds();
    }

    @Override
    public GpuLoweredRegionManifest gpuLoweredRegionManifest() {
        return manifest;
    }

    private static GpuLoweredRegionManifest defaultManifest(
            int anchorNodeId,
            AcceleratorSubgraphSpec subgraph,
            AcceleratorDagSpec dagSpec,
            GpuCompoundRegionSummary compoundSummary
    ) {
        return new GpuLoweredRegionManifest(
                "gpu-gpu_cuda-region-" + subgraph.computeNodeId(),
                ComputeBackend.GPU_CUDA,
                anchorNodeId,
                subgraph.orderedNodeIds(),
                subgraph.externalInputNodeIds(),
                subgraph.outputNodeIds(),
                subgraph.orderedNodeIds().size(),
                List.of(),
                loweredPrimitives(dagSpec),
                List.of(),
                List.of(),
                compoundSummary,
                List.of(),
                GpuLoweredRegionCandidateSpan.none(subgraph.orderedNodeIds()),
                Map.of("dagNodeCount", Integer.toString(dagSpec.nodes().size()))
        );
    }

    private static List<GpuLoweredPrimitiveManifest> loweredPrimitives(AcceleratorDagSpec dagSpec) {
        java.util.ArrayList<GpuLoweredPrimitiveManifest> out = new java.util.ArrayList<>(dagSpec.nodes().size());
        for (int i = 0; i < dagSpec.nodes().size(); i++) {
            AcceleratorDagNode node = dagSpec.nodes().get(i);
            out.add(new GpuLoweredPrimitiveManifest(
                    "p" + i,
                    node.type().name(),
                    List.of(node.nodeId()),
                    List.of(),
                    "node:" + i,
                    DataType.FLOAT32,
                    outputShape(node),
                    List.of()
            ));
        }
        return List.copyOf(out);
    }

    private static List<Integer> outputShape(AcceleratorDagNode node) {
        return switch (node.outputRank()) {
            case 1 -> List.of(node.outputDim0());
            case 2 -> List.of(node.outputDim0(), node.outputDim1());
            case 3 -> List.of(node.outputDim0(), node.outputDim1(), node.outputDim2());
            default -> List.of(node.outputDim0(), node.outputDim1(), node.outputDim2(), node.outputDim3());
        };
    }
}

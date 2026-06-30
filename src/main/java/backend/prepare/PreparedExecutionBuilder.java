package backend.prepare;

import backend.select.BackendSelectionResult;
import backend.select.DefaultBackendSelectionPolicy;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringInput;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringPipeline;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweredRegion;
import backend.lowering.region.RegionExecutionPlan;
import backend.partition.BackendPartitionDescriptorRegistry;
import graph.model.CompiledNode;
import graph.compile.CompiledProgram;
import graph.compile.CompileArtifacts;
import graph.compile.publication.PublicationPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import trace.prepare.PrepareTrace;
import planning.partition.PartitionPlan;
import planning.partition.PlannedPartition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class PreparedExecutionBuilder {
    private PreparedExecutionBuilder() {
    }

    public static PreparedExecution prepare(CompileArtifacts artifacts, config.runtime.RuntimeConfig runtimeConfig) {
        Objects.requireNonNull(artifacts, "artifacts cannot be null");
        Objects.requireNonNull(runtimeConfig, "runtimeConfig cannot be null");
        CompiledProgram program = artifacts.program();
        PublicationPlan publication = artifacts.publication();
        publication.graphContract().validateOrThrow(publication.rootTensor());
        long t0 = System.nanoTime();
        List<CompiledNode> compiledNodes = program.compiledNodes();
        Map<Integer, List<CompiledNode>> consumers = buildConsumerMap(compiledNodes);
        BackendPrepareContext context = new BackendPrepareContext(
                runtimeConfig,
                program.supportsBackward(),
                compiledNodes,
                program.descriptorIndex(),
                consumers
        );
        BackendSelectionResult selection = new DefaultBackendSelectionPolicy().select(
                artifacts.plannedPartitions(),
                runtimeConfig
        );
        context.publishBackendPlans(selection.selectedPlans());
        LoweringInput loweringInput = artifacts.loweringInput();
        publishLoweredRegions(artifacts, compiledNodes, context, runtimeConfig, selection, loweringInput);
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        List<PreparedExecutionStep> executionSteps = new ArrayList<>();
        List<PreparedExecutionStep> forwardSteps = new ArrayList<>();
        List<PreparedExecutionStep> backwardSteps = new ArrayList<>();
        Set<Integer> coveredNodeIds = new java.util.HashSet<>();
        for (CompiledNode node : compiledNodes) {
            if (coveredNodeIds.contains(node.id())) {
                continue;
            }
            LoweredExecutionUnit fusedUnit = context.cpuFusedUnitForStart(node.id());
            if (fusedUnit != null) {
                addPreparedRegionStep(
                        prepareCpuFusedStep(fusedUnit, context, dispatcher),
                        context,
                        program.forwardBoundaryNodeId(),
                        executionSteps,
                        forwardSteps,
                        backwardSteps,
                        coveredNodeIds
                );
                continue;
            }
            LoweredExecutionUnit specializedUnit = context.cpuSpecializedUnitForStart(node.id());
            if (specializedUnit != null) {
                addPreparedRegionStep(
                        prepareCpuSpecializedStep(specializedUnit, context, dispatcher),
                        context,
                        program.forwardBoundaryNodeId(),
                        executionSteps,
                        forwardSteps,
                        backwardSteps,
                        coveredNodeIds
                );
                continue;
            }
            LoweredRegion metalRegion = context.metalLoweredRegionForStart(node.id());
            if (metalRegion != null) {
                addPreparedRegionStep(
                        prepareAcceleratorRegionStep(metalRegion, context, dispatcher, AcceleratorRegionBackend.METAL),
                        context,
                        program.forwardBoundaryNodeId(),
                        executionSteps,
                        forwardSteps,
                        backwardSteps,
                        coveredNodeIds
                );
                continue;
            }
            LoweredRegion cudaRegion = context.cudaLoweredRegionForStart(node.id());
            if (cudaRegion != null) {
                addPreparedRegionStep(
                        prepareAcceleratorRegionStep(cudaRegion, context, dispatcher, AcceleratorRegionBackend.CUDA),
                        context,
                        program.forwardBoundaryNodeId(),
                        executionSteps,
                        forwardSteps,
                        backwardSteps,
                        coveredNodeIds
                );
                continue;
            }
            if (node.operation() == null || node.inputIds().isEmpty()) {
                continue;
            }
            CompiledNodeExecutionMetadata metadata = dispatcher.prepare(node, context);
            context.publishPreparedMetadata(node.id(), metadata);
            PreparedExecutionStep step = new PreparedExecutionStep(node, metadata);
            addStep(step, program.forwardBoundaryNodeId(), executionSteps, forwardSteps, backwardSteps);
        }
        return new PreparedExecution(
                runtimeConfig,
                program.supportsBackward(),
                executionSteps,
                forwardSteps,
                backwardSteps,
                compiledNodes,
                program.descriptorIndex(),
                publication,
                program.forwardOutputNode(),
                loweringInput == null ? program.memoryPlan() : loweringInput.memoryPlan(),
                new PrepareTrace(
                        true,
                        System.nanoTime() - t0,
                        forwardSteps.size(),
                        backwardSteps.size(),
                        selection.trace(),
                        BackendPrepareTraceContributors.diagnostics(selection, loweringInput)
                )
        );
    }

    private enum AcceleratorRegionBackend {
        METAL,
        CUDA
    }

    private static PreparedExecutionStep prepareCpuFusedStep(
            LoweredExecutionUnit fusedUnit,
            BackendPrepareContext context,
            BackendPrepareDispatcher dispatcher
    ) {
        var regionPlan = fusedUnit.requireRegionPlan();
        if (regionPlan.boundaryOutputNodeIds().size() != 1) {
            throw new IllegalStateException("CPU fused prepared step requires exactly one boundary output. unit="
                    + fusedUnit.unitId() + ", boundaryOutputs=" + regionPlan.boundaryOutputNodeIds());
        }
        int outputNodeId = regionPlan.boundaryOutputNodeIds().getFirst();
        if (fusedUnit.orderedNodeIds().isEmpty() || fusedUnit.orderedNodeIds().getLast() != outputNodeId) {
            throw new IllegalStateException("CPU fused prepared step output must be the last ordered node. unit="
                    + fusedUnit.unitId() + ", outputNodeId=" + outputNodeId
                    + ", orderedNodeIds=" + fusedUnit.orderedNodeIds());
        }
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        if (outputNode == null) {
            throw new IllegalStateException("Missing CPU fused output node id=" + outputNodeId);
        }
        CompiledNodeExecutionMetadata metadata = dispatcher.prepareCpuFusedStep(outputNode, fusedUnit, context);
        return new PreparedExecutionStep(
                outputNode,
                metadata,
                fusedUnit.orderedNodeIds(),
                regionPlan.boundaryOutputNodeIds()
        );
    }

    private static PreparedExecutionStep prepareCpuSpecializedStep(
            LoweredExecutionUnit specializedUnit,
            BackendPrepareContext context,
            BackendPrepareDispatcher dispatcher
    ) {
        var regionPlan = requireBoundaryStepNode(
                specializedUnit.requireRegionPlan(),
                context,
                "CPU specialized"
        );
        int outputNodeId = representativeBoundaryNodeId(regionPlan);
        if (specializedUnit.orderedNodeIds().isEmpty() || specializedUnit.orderedNodeIds().getLast() != outputNodeId) {
            throw new IllegalStateException("CPU specialized prepared step output must be the last ordered node. unit="
                    + specializedUnit.unitId() + ", outputNodeId=" + outputNodeId
                    + ", orderedNodeIds=" + specializedUnit.orderedNodeIds());
        }
        CompiledNode outputNode = context.compiledNode(outputNodeId);
        CompiledNodeExecutionMetadata metadata = dispatcher.prepareCpuSpecializedStep(outputNode, specializedUnit, context);
        return new PreparedExecutionStep(
                outputNode,
                metadata,
                regionPlan.orderedNodeIds(),
                regionPlan.boundaryOutputNodeIds()
        );
    }

    private static PreparedExecutionStep prepareAcceleratorRegionStep(
            LoweredRegion region,
            BackendPrepareContext context,
            BackendPrepareDispatcher dispatcher,
            AcceleratorRegionBackend backend
    ) {
        RegionExecutionPlan regionPlan = requireBoundaryStepNode(region.units().getFirst().requireRegionPlan(), context, backend.name());
        CompiledNode outputNode = context.compiledNode(representativeBoundaryNodeId(regionPlan));
        CompiledNodeExecutionMetadata metadata = switch (backend) {
            case METAL -> dispatcher.prepareMetalRegionStep(region, context);
            case CUDA -> dispatcher.prepareCudaRegionStep(region, context);
        };
        return new PreparedExecutionStep(
                outputNode,
                metadata,
                regionPlan.orderedNodeIds(),
                regionPlan.boundaryOutputNodeIds()
        );
    }

    private static RegionExecutionPlan requireBoundaryStepNode(
            RegionExecutionPlan regionPlan,
            BackendPrepareContext context,
            String label
    ) {
        if (regionPlan.boundaryOutputNodeIds().isEmpty()) {
            throw new IllegalStateException(label + " prepared region step requires at least one boundary output. region="
                    + regionPlan.regionId() + ", boundaryOutputs=" + regionPlan.boundaryOutputNodeIds());
        }
        int representativeNodeId = representativeBoundaryNodeId(regionPlan);
        if (context.compiledNode(representativeNodeId) == null) {
            throw new IllegalStateException("Missing " + label + " region boundary node id=" + representativeNodeId);
        }
        return regionPlan;
    }

    private static int representativeBoundaryNodeId(RegionExecutionPlan regionPlan) {
        if (regionPlan.boundaryOutputNodeIds().contains(regionPlan.anchorNodeId())) {
            return regionPlan.anchorNodeId();
        }
        return regionPlan.boundaryOutputNodeIds().getFirst();
    }

    private static void addPreparedRegionStep(
            PreparedExecutionStep step,
            BackendPrepareContext context,
            int forwardBoundaryNodeId,
            List<PreparedExecutionStep> executionSteps,
            List<PreparedExecutionStep> forwardSteps,
            List<PreparedExecutionStep> backwardSteps,
            Set<Integer> coveredNodeIds
    ) {
        context.publishPreparedMetadata(step.compiledNode().id(), step.metadata());
        addStep(step, forwardBoundaryNodeId, executionSteps, forwardSteps, backwardSteps);
        coveredNodeIds.addAll(step.orderedNodeIds());
    }

    private static void addStep(
            PreparedExecutionStep step,
            int forwardBoundaryNodeId,
            List<PreparedExecutionStep> executionSteps,
            List<PreparedExecutionStep> forwardSteps,
            List<PreparedExecutionStep> backwardSteps
    ) {
        executionSteps.add(step);
        if (step.compiledNode().id() <= forwardBoundaryNodeId) {
            forwardSteps.add(step);
        } else {
            backwardSteps.add(step);
        }
    }

    private static void publishLoweredRegions(
            CompileArtifacts artifacts,
            List<CompiledNode> compiledNodes,
            BackendPrepareContext context,
            config.runtime.RuntimeConfig runtimeConfig,
            BackendSelectionResult selection,
            LoweringInput loweringInput
    ) {
        if (loweringInput == null || loweringInput.plannedRegions().isEmpty() || loweringInput.memoryPlan() == null) {
            return;
        }
        LoweringPipeline pipeline = new LoweringPipeline(BackendPartitionDescriptorRegistry.defaults().lowerers());
        Map<String, PartitionPlan> selectedPlansByPartitionId =
                selectedPlansByPartitionId(selection);
        Set<backend.contract.ComputeBackend> supportedBackends = new java.util.LinkedHashSet<>();
        supportedBackends.add(backend.contract.ComputeBackend.CPU);
        for (PartitionPlan plan : selectedPlansByPartitionId.values()) {
            if (plan != null) {
                supportedBackends.add(plan.backend());
            }
        }
        var lowered = pipeline.lower(
                loweringInput,
                new BackendCapabilities(supportedBackends),
                new LoweringContext(
                        runtimeConfig,
                        compiledNodes,
                        artifacts.descriptorIndex(),
                        selectedPlansByPartitionId
                )
        );
        context.publishLoweredRegions(lowered.lowered().loweredRegions());
    }

    private static Map<String, PartitionPlan> selectedPlansByPartitionId(BackendSelectionResult selection) {
        if (selection == null || selection.selectedPartitions().isEmpty()) {
            return Map.of();
        }
        HashMap<String, PartitionPlan> out = new HashMap<>();
        for (PlannedPartition selectedPartition : selection.selectedPartitions()) {
            if (selectedPartition == null
                    || selectedPartition.partition() == null
                    || selectedPartition.plan() == null) {
                continue;
            }
            String partitionId = selectedPartition.partition().partitionId();
            if (partitionId != null && !partitionId.isBlank()) {
                out.put(partitionId, selectedPartition.plan());
            }
        }
        return Map.copyOf(out);
    }

    private static Map<Integer, List<CompiledNode>> buildConsumerMap(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

}

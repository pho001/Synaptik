package graph.compile.session;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.CompileConfig;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.BackendPlanningRequest;
import graph.compile.planning.BackendPlanningResult;
import graph.compile.planning.BackendPlanningService;
import graph.compile.planning.partition.PlannedPartition;
import graph.execution.trace.PartitionCompileTrace;
import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Delegates backend ownership partition planning for compiled nodes.
 */
final class BackendOwnershipPlanningStage {
    private BackendOwnershipPlanningStage() {
    }

    record Result(
            List<PlannedPartition> plannedPartitions,
            PartitionCompileTrace trace
    ) {
        public Result {
            plannedPartitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
            trace = trace == null ? PartitionCompileTrace.empty() : trace;
        }
    }

    static Result plan(
            BackendPlanningService backendPlanningService,
            CompileConfig compileConfig,
            boolean supportsBackward,
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode forwardOutput,
            Map<Tensor, CompiledGradientBinding> compiledGradients,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        Objects.requireNonNull(backendPlanningService, "backendPlanningService cannot be null");
        CompileConfig config = Objects.requireNonNull(compileConfig, "compileConfig cannot be null");
        BackendPlanningResult planning = backendPlanningService.plan(new BackendPlanningRequest(
                config.backendPlanning(),
                supportsBackward,
                compiledNodes,
                descriptorIndex,
                forwardOutput,
                compiledGradients,
                backendPartitionDescriptors
        ));
        return new Result(planning.plannedPartitions(), planning.trace());
    }
}

package graph.compile.session;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.CompileConfig;
import graph.CompiledGradientBinding;
import graph.CompiledProgram;
import graph.SemanticForwardCanonicalizer;
import graph.compile.CompileArtifacts;
import graph.compile.GraphStructureContract;
import graph.compile.planning.BackendPlanningService;
import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.state.OptimizerTrace;
import tensor.AutogradCompilationScope;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.Map;
import java.util.Objects;

/**
 * Orchestrates one graph compile through named compile stages.
 */
public final class CompileSession {
    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final CompileConfig compileConfig;
    private final CompileMode compileMode;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final BackendPlanningService backendPlanningService;

    private int forwardGraphSize;
    private OptimizerTrace optimizerTrace = OptimizerTrace.empty();
    private PartitionCompileTrace partitionPlanningTrace = PartitionCompileTrace.empty();

    public CompileSession(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors,
            BackendPlanningService backendPlanningService
    ) {
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.compileConfig = Objects.requireNonNull(compileConfig, "compileConfig cannot be null");
        this.compileMode = Objects.requireNonNull(compileMode, "compileMode cannot be null");
        this.backendPartitionDescriptors = Objects.requireNonNull(
                backendPartitionDescriptors,
                "backendPartitionDescriptors cannot be null"
        );
        this.backendPlanningService = Objects.requireNonNull(
                backendPlanningService,
                "backendPlanningService cannot be null"
        );
    }

    public CompileArtifacts compile() {
        ForwardGraphCapture.Result forward = ForwardGraphCapture.capture(rootTensor, forwardCanonicalizer);
        forwardGraphSize = forward.forwardGraph().size();

        try (AutogradCompilationScope ignored = AutogradCompilationScope.open()) {
            BackwardGraphCompiler.Result backward = BackwardGraphCompiler.compile(
                    rootTensor,
                    forward.forwardGraph(),
                    forward.forwardOutput(),
                    compileMode
            );
            OptimizerSnapshotStage.Result optimized = OptimizerSnapshotStage.optimize(
                    optimizer,
                    backward.workingGraph(),
                    forward.forwardOutput(),
                    forward.publicationTensors(),
                    backward.supportsBackward()
            );
            optimizerTrace = optimized.optimizerState().trace();

            String graphDescription = backward.supportsBackward() ? "finalGraph" : "inference finalGraph";
            CompiledProgramSnapshotStage.Result snapshot = CompiledProgramSnapshotStage.snapshot(
                    optimized.optimizedGraph(),
                    optimized.forwardOutput(),
                    graphDescription
            );
            Map<Tensor, CompiledGradientBinding> compiledGradients = backward.supportsBackward()
                    ? GradientBindingCollector.captureCompiledGradients(
                            snapshot.graph(),
                            optimized.publicationTensors(),
                            snapshot.compiledNodeByTensor()
                    )
                    : Map.of();
            CompiledGradientBinding forwardSeedGradient = GradientBindingCollector.captureForwardSeedGradient(
                    BackwardGraphCompiler.requireForwardRoot(optimized.forwardOutput()),
                    snapshot.compiledNodeByTensor()
            );
            BackendOwnershipPlanningStage.Result backendPlanning = BackendOwnershipPlanningStage.plan(
                    backendPlanningService,
                    compileConfig,
                    backward.supportsBackward(),
                    snapshot.compiledNodes(),
                    snapshot.descriptorIndex(),
                    snapshot.forwardOutput(),
                    compiledGradients,
                    backendPartitionDescriptors
            );
            partitionPlanningTrace = backendPlanning.trace();

            RegionAndMemoryPlanningStage.Result planning = RegionAndMemoryPlanningStage.plan(
                    compileConfig,
                    backendPlanning.plannedPartitions(),
                    snapshot.compiledNodes(),
                    optimized.optimizerState(),
                    snapshot.graph(),
                    optimized.forwardOutput(),
                    backward.supportsBackward(),
                    snapshot.forwardBoundaryNodeId()
            );
            optimizerTrace = planning.optimizerState().trace();

            GraphStructureContract graphContract = GraphStructureContract.capture(rootTensor);
            var publicationPlan = PublicationPlanBuilder.build(
                    rootTensor,
                    graphContract,
                    snapshot.graph(),
                    snapshot.compiledNodeByTensor(),
                    compiledGradients,
                    forwardSeedGradient,
                    snapshot.forwardOutput(),
                    optimized.forwardOutput(),
                    snapshot.forwardBoundaryNodeId(),
                    optimized.publicationTensors()
            );
            return new CompileArtifacts(
                    new CompiledProgram(
                            snapshot.compiledNodes(),
                            snapshot.descriptorIndex(),
                            snapshot.forwardOutput().id(),
                            snapshot.forwardBoundaryNodeId(),
                            backward.supportsBackward(),
                            backendPlanning.plannedPartitions(),
                            planning.optimizedRegions(),
                            planning.memoryPlan()
                    ),
                    publicationPlan
            );
        }
    }

    public int forwardGraphSize() {
        return forwardGraphSize;
    }

    public OptimizerTrace optimizerTrace() {
        return optimizerTrace;
    }

    public PartitionCompileTrace partitionPlanningTrace() {
        return partitionPlanningTrace;
    }
}

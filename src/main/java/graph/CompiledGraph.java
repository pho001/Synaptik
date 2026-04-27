package graph;

import backend.runtime.ExecutionMode;
import backend.prepare.PreparedExecutionBuilder;
import graph.compile.CompileArtifacts;
import graph.compile.GraphCompiler;
import graph.execution.PreparedExecution;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.PartitionCompileTrace;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

public class CompiledGraph {
    private final Tensor rootTensor;
    private final CompileMode compileMode;
    private final GraphCompiler compiler;
    private CompileTrace compileTrace = CompileTrace.skipped();
    private CompileArtifacts artifacts;

    private CompiledGraph(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer forwardOptimizer,
            config.optimizer.PartitionConfig partitionConfig,
            CompileMode compileMode
    ) {
        this.rootTensor = rootTensor;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        this.compiler = new GraphCompiler(
                rootTensor,
                forwardCanonicalizer,
                forwardOptimizer,
                partitionConfig,
                this.compileMode
        );
        compile();
    }

    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig) {
        return compile(rootTensor, optimizerConfig, CompileMode.AUTO);
    }

    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizerConfig == null) {
            throw new IllegalArgumentException("optimizerConfig cannot be null");
        }
        return new CompiledGraph(
                rootTensor,
                graph.optimizer.OptimizerFactory.createSemanticForwardCanonicalizer(optimizerConfig),
                graph.optimizer.OptimizerFactory.create(optimizerConfig),
                optimizerConfig.partition(),
                compileMode
        );
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer) {
        return compile(rootTensor, optimizer, CompileMode.AUTO);
    }

    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        return new CompiledGraph(rootTensor, null, optimizer, config.optimizer.PartitionConfig.defaults(), compileMode);
    }

    public void compile() {
        GraphCompiler.Result result = compiler.compile();
        artifacts = result.artifacts();
        compileTrace = result.trace();
    }

    public boolean supportsBackward() {
        return compileArtifacts().supportsBackward();
    }

    public CompileMode compileMode() {
        return compileMode;
    }

    public PreparedExecution prepare() {
        return prepare((config.runtime.RuntimeConfig) null);
    }

    public PreparedExecution prepare(config.runtime.RuntimeConfig runtimeConfig) {
        config.runtime.RuntimeConfig effectiveConfig = runtimeConfig == null
                ? (supportsBackward() ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults())
                : runtimeConfig;
        return PreparedExecutionBuilder.prepare(compileArtifacts(), effectiveConfig);
    }

    public PreparedExecution prepare(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime());
    }

    public void execute(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        prepare(runtimeConfig).execute(mode);
    }

    public void execute(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        prepare(profile.runtime()).execute(profile.mode());
    }

    public RunTrace executeTraced(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return prepare(runtimeConfig).executeTraced(mode);
    }

    public RunTrace executeTraced(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime()).executeTraced(profile.mode());
    }

    public void executePrepared(PreparedExecution execution, ExecutionMode mode) {
        execution.execute(mode);
    }

    public void zeroGrad() {
        for (CompiledNode node : compileArtifacts().compiledNodes()) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor gradient = node.sourceTensor().getGradient();
            if (gradient == null) {
                continue;
            }
            switch (gradient.getDataType()) {
                case FLOAT64 -> java.util.Arrays.fill(gradient.getFloat64Data(), 0.0d);
                case FLOAT32 -> java.util.Arrays.fill(gradient.getFloat32Data(), 0.0f);
                case BFLOAT16 -> java.util.Arrays.fill(gradient.getBFloat16Data(), (short) 0);
                case INT32 -> java.util.Arrays.fill(gradient.getInt32Data(), 0);
                case BOOL -> java.util.Arrays.fill(gradient.getBoolData(), (byte) 0);
            }
        }
    }

    public Tensor getRootTensor() {
        return rootTensor;
    }

    public List<Tensor> getCompiledGraphAsList() {
        return compileArtifacts().finalGraph();
    }

    public CompileTrace compileTrace() {
        return compileTrace;
    }

    public CompileArtifacts compileArtifacts() {
        if (artifacts == null) {
            throw new IllegalStateException("CompiledGraph has not been compiled.");
        }
        return artifacts;
    }

    Map<Tensor, CompiledGradientBinding> compiledGradients() {
        return compileArtifacts().gradientBindings();
    }

    CompiledGradientBinding forwardSeedGradient() {
        return compileArtifacts().forwardSeedGradient();
    }

    List<CompiledNode> compiledNodesView() {
        return compileArtifacts().compiledNodes();
    }

    int forwardBoundaryNodeId() {
        return compileArtifacts().forwardBoundaryNodeId();
    }

    CompiledNode compiledForwardOutputNode() {
        return compileArtifacts().forwardOutputNode();
    }

    Map<Tensor, CompiledGradientBinding> compiledGradientBindings() {
        return compileArtifacts().gradientBindings();
    }

    MemoryPlan compiledMemoryPlan() {
        return compileArtifacts().memoryPlan();
    }

    OptimizerState compiledOptimizerState() {
        return compileArtifacts().optimizerState();
    }

    List<PartitionPlan> compiledBackendPlansView() {
        return compileArtifacts().backendPlans();
    }

    List<Partition> compiledPartitionsView() {
        return compileArtifacts().partitions();
    }

    List<BackendCandidatePartition> compiledBackendSelectionCandidatesView() {
        return compileArtifacts().backendSelectionCandidates();
    }

    PartitionCompileTrace compiledPartitionPlanningTrace() {
        return compileArtifacts().partitionPlanningTrace();
    }

}

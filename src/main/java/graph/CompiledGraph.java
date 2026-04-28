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

/**
 * Public facade for compiling a tensor expression into an immutable execution plan.
 *
 * <p>The lifecycle is:
 * <ol>
 *     <li>{@link #compile(Tensor, config.optimizer.OptimizerConfig)} captures the root tensor's forward graph,
 *     optionally builds backward nodes, applies optimizer stages, snapshots compiled nodes, plans partitions and
 *     computes a memory plan.</li>
 *     <li>{@link #prepare(config.runtime.RuntimeConfig)} lowers those compile artifacts into runtime execution steps
 *     for a specific runtime configuration.</li>
 *     <li>{@link PreparedExecution#execute(ExecutionMode)} or the convenience {@code execute(...)} methods run the
 *     prepared steps and publish forward data and gradients back to the source tensors.</li>
 * </ol>
 *
 * <p>Compilation and preparation allocate new artifact objects, while execution mutates tensor data, gradient buffers,
 * and backend workspaces. Instances are not intended to be used concurrently for recompilation or execution against the
 * same source tensors.
 */
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
            config.optimizer.OptimizerConfig optimizerConfig,
            CompileMode compileMode
    ) {
        this.rootTensor = rootTensor;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        this.compiler = new GraphCompiler(
                rootTensor,
                forwardCanonicalizer,
                forwardOptimizer,
                optimizerConfig,
                this.compileMode
        );
        compile();
    }

    /**
     * Compiles {@code rootTensor} using an optimizer built from {@code optimizerConfig} and automatic compile mode.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param optimizerConfig optimizer, partition, fusion, rewrite, and memory planning configuration
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if either argument is {@code null}
     */
    public static CompiledGraph compile(Tensor rootTensor, config.optimizer.OptimizerConfig optimizerConfig) {
        return compile(rootTensor, optimizerConfig, CompileMode.AUTO);
    }

    /**
     * Compiles {@code rootTensor} using an optimizer built from {@code optimizerConfig}.
     *
     * <p>{@code compileMode} controls whether backward artifacts are built. When it is {@code null}, the compiler uses
     * {@link CompileMode#AUTO}.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param optimizerConfig optimizer, partition, fusion, rewrite, and memory planning configuration
     * @param compileMode requested forward/backward compilation mode, or {@code null} for automatic mode
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code optimizerConfig} is {@code null}
     */
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
                optimizerConfig,
                compileMode
        );
    }

    /**
     * Compiles {@code rootTensor} with an explicit optimizer and automatic compile mode.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param optimizer ordered optimizer rule pipeline to apply to the graph
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if either argument is {@code null}
     */
    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer) {
        return compile(rootTensor, optimizer, CompileMode.AUTO);
    }

    /**
     * Compiles {@code rootTensor} with an explicit optimizer and compile mode.
     *
     * <p>This overload does not apply semantic forward canonicalization and uses default partition configuration. It is
     * useful for tests and custom optimizer pipelines where the rule order is supplied directly.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param optimizer ordered optimizer rule pipeline to apply to the graph
     * @param compileMode requested forward/backward compilation mode, or {@code null} for automatic mode
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public static CompiledGraph compile(Tensor rootTensor, GraphOptimizer optimizer, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        return new CompiledGraph(rootTensor, null, optimizer, config.optimizer.OptimizerConfig.inferenceDefaults(), compileMode);
    }

    /**
     * Rebuilds compile artifacts from the current tensor graph state.
     *
     * <p>Recompilation refreshes snapshots, optimizer state, partition plans, memory plans, and compile trace metadata.
     * It does not execute kernels or clear existing source tensor gradients.
     */
    public void compile() {
        GraphCompiler.Result result = compiler.compile();
        artifacts = result.artifacts();
        compileTrace = result.trace();
    }

    /**
     * Returns whether the compiled artifacts include backward execution steps.
     *
     * @return {@code true} when {@link ExecutionMode#FORWARD_BACKWARD} can be run
     */
    public boolean supportsBackward() {
        return compileArtifacts().supportsBackward();
    }

    /**
     * Returns the compile mode resolved for this graph.
     *
     * @return compile mode, never {@code null}
     */
    public CompileMode compileMode() {
        return compileMode;
    }

    /**
     * Prepares runtime execution steps using default runtime settings.
     *
     * <p>Training defaults are selected when backward artifacts are available; otherwise inference defaults are used.
     *
     * @return prepared execution plan bound to this graph's compile artifacts
     */
    public PreparedExecution prepare() {
        return prepare((config.runtime.RuntimeConfig) null);
    }

    /**
     * Prepares runtime execution steps for a runtime configuration.
     *
     * <p>Preparation chooses execution operations and kernels, builds per-node runtime metadata, and records
     * {@link graph.execution.trace.PrepareTrace}. The returned object is reusable for repeated executions with the same
     * runtime configuration, but each execution mutates tensor storage and gradients.
     *
     * @param runtimeConfig runtime settings, or {@code null} to choose training or inference defaults
     * @return prepared execution plan bound to this graph's compile artifacts
     */
    public PreparedExecution prepare(config.runtime.RuntimeConfig runtimeConfig) {
        config.runtime.RuntimeConfig effectiveConfig = runtimeConfig == null
                ? (supportsBackward() ? config.runtime.RuntimeConfig.trainingDefaults() : config.runtime.RuntimeConfig.inferenceDefaults())
                : runtimeConfig;
        return PreparedExecutionBuilder.prepare(compileArtifacts(), effectiveConfig);
    }

    /**
     * Prepares runtime execution steps using the runtime portion of an execution profile.
     *
     * @param profile execution profile that supplies runtime settings
     * @return prepared execution plan bound to this graph's compile artifacts
     * @throws IllegalArgumentException if {@code profile} is {@code null}
     */
    public PreparedExecution prepare(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime());
    }

    /**
     * Prepares and immediately executes this graph.
     *
     * @param runtimeConfig runtime settings, or {@code null} for defaults
     * @param mode execution mode to run
     */
    public void execute(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        prepare(runtimeConfig).execute(mode);
    }

    /**
     * Prepares and immediately executes this graph using an execution profile.
     *
     * @param profile profile that supplies runtime settings and execution mode
     * @throws IllegalArgumentException if {@code profile} is {@code null}
     */
    public void execute(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        prepare(profile.runtime()).execute(profile.mode());
    }

    /**
     * Prepares, executes, and returns per-run trace metadata.
     *
     * @param runtimeConfig runtime settings, or {@code null} for defaults
     * @param mode execution mode to run
     * @return run trace with duration and step metadata
     */
    public RunTrace executeTraced(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        return prepare(runtimeConfig).executeTraced(mode);
    }

    /**
     * Prepares, executes, and returns per-run trace metadata using an execution profile.
     *
     * @param profile profile that supplies runtime settings and execution mode
     * @return run trace with duration and step metadata
     * @throws IllegalArgumentException if {@code profile} is {@code null}
     */
    public RunTrace executeTraced(config.profile.ExecutionProfile profile) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        return prepare(profile.runtime()).executeTraced(profile.mode());
    }

    /**
     * Executes a previously prepared plan.
     *
     * <p>The prepared plan must have been created from compatible compile artifacts. This method does not check that the
     * plan came from this facade.
     *
     * @param execution prepared execution plan to run
     * @param mode execution mode to run
     */
    public void executePrepared(PreparedExecution execution, ExecutionMode mode) {
        execution.execute(mode);
    }

    /**
     * Clears gradient buffers for the source tensors represented by forward compiled nodes.
     *
     * <p>This method mutates existing gradient tensors in place and skips backward-only compiled nodes. It does not
     * allocate missing gradient buffers.
     */
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

    /**
     * Returns the source root tensor passed to compilation.
     *
     * @return root tensor
     */
    public Tensor getRootTensor() {
        return rootTensor;
    }

    /**
     * Returns the optimized graph tensors in execution order.
     *
     * @return immutable final graph view from the latest compile artifacts
     */
    public List<Tensor> getCompiledGraphAsList() {
        return compileArtifacts().finalGraph();
    }

    /**
     * Returns metadata from the latest compile call.
     *
     * @return compile trace, or a skipped trace before compilation has run
     */
    public CompileTrace compileTrace() {
        return compileTrace;
    }

    /**
     * Returns the latest compile artifacts.
     *
     * @return immutable compile artifact bundle
     * @throws IllegalStateException if this facade has not compiled successfully
     */
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

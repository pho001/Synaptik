package graph;

import backend.runtime.ExecutionMode;
import backend.prepare.PreparedExecutionBuilder;
import config.compile.CompileConfig;
import graph.compile.CompileArtifacts;
import graph.compile.GraphCompiler;
import graph.execution.PreparedExecution;
import graph.execution.PublicationPolicy;
import graph.execution.trace.CompileTrace;
import graph.execution.trace.RunTrace;
import graph.optimizer.GraphOptimizer;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;

/**
 * Public facade for compiling a tensor expression into an immutable execution plan.
 *
 * <p>The lifecycle is:
 * <ol>
 *     <li>{@link #compile(Tensor, CompileConfig)} captures the root tensor's forward graph,
 *     optionally builds backward nodes, applies optimizer stages, snapshots compiled nodes, plans partitions and
 *     computes a memory plan.</li>
 *     <li>{@link #prepare(config.runtime.RuntimeConfig)} lowers those compile artifacts into runtime execution steps
 *     for a specific runtime configuration.</li>
 *     <li>{@link PreparedExecution#execute(ExecutionMode)} or the convenience {@code execute(...)} methods run the
 *     prepared steps and publish values according to the selected {@link PublicationPolicy}.</li>
 * </ol>
 *
 * <p>Compilation and preparation allocate new artifact objects, while execution mutates tensor data, gradient buffers,
 * and backend workspaces. Instances are not intended to be used concurrently for recompilation or execution against the
 * same user-visible tensors.
 */
public final class CompiledGraph {
    private final Tensor rootTensor;
    private final CompileMode compileMode;
    private final CompileTrace compileTrace;
    private final CompileArtifacts artifacts;

    private CompiledGraph(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer forwardOptimizer,
            CompileConfig compileConfig,
            CompileMode compileMode
    ) {
        this.rootTensor = rootTensor;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        GraphCompiler compiler = new GraphCompiler(
                rootTensor,
                forwardCanonicalizer,
                forwardOptimizer,
                compileConfig,
                this.compileMode
        );
        GraphCompiler.Result result = compiler.compile();
        this.artifacts = result.artifacts();
        this.compileTrace = result.trace();
    }

    /**
     * Compiles {@code rootTensor} using a compile configuration.
     *
     * <p>{@code compileMode} controls whether backward artifacts are built. When it is {@code null}, the compiler uses
     * {@link CompileMode#AUTO}.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param compileConfig compile-time semantic, graph, backend, region, and memory planning configuration
     * @param compileMode requested forward/backward compilation mode, or {@code null} for automatic mode
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code compileConfig} is {@code null}
     */
    public static CompiledGraph compile(Tensor rootTensor, CompileConfig compileConfig, CompileMode compileMode) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (compileConfig == null) {
            throw new IllegalArgumentException("compileConfig cannot be null");
        }
        return new CompiledGraph(
                rootTensor,
                graph.optimizer.OptimizerFactory.createSemanticForwardCanonicalizer(compileConfig.semanticCanonicalization()),
                graph.optimizer.OptimizerFactory.create(compileConfig.graphOptimization()),
                compileConfig,
                compileMode
        );
    }

    /**
     * Compiles {@code rootTensor} using a compile configuration and automatic compile mode.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param compileConfig compile-time semantic, graph, backend, region, and memory planning configuration
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if either argument is {@code null}
     */
    public static CompiledGraph compile(Tensor rootTensor, CompileConfig compileConfig) {
        return compile(rootTensor, compileConfig, CompileMode.AUTO);
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
        return new CompiledGraph(rootTensor, null, optimizer, CompileConfig.inference(), compileMode);
    }

    /**
     * Returns whether the compiled artifacts include backward execution steps.
     *
     * @return {@code true} when {@link ExecutionMode#FORWARD_BACKWARD} can be run
     */
    public boolean supportsBackward() {
        return compileArtifacts().program().supportsBackward();
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
                ? (supportsBackward()
                ? config.runtime.RuntimeConfig.trainingDefaults(rootTensor.getDataType())
                : config.runtime.RuntimeConfig.inferenceDefaults(rootTensor.getDataType()))
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
        try (PreparedExecution execution = prepare(runtimeConfig)) {
            execution.execute(mode);
        }
    }

    /**
     * Prepares and immediately executes this graph with an explicit publication policy.
     *
     * @param runtimeConfig runtime settings, or {@code null} for defaults
     * @param mode execution mode to run
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     */
    public void execute(
            config.runtime.RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            PublicationPolicy publicationPolicy
    ) {
        try (PreparedExecution execution = prepare(runtimeConfig)) {
            execution.execute(mode, publicationPolicy);
        }
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
        try (PreparedExecution execution = prepare(profile.runtime())) {
            execution.execute(profile.mode());
        }
    }

    /**
     * Prepares and immediately executes this graph using an execution profile and publication policy.
     *
     * @param profile profile that supplies runtime settings and execution mode
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @throws IllegalArgumentException if {@code profile} is {@code null}
     */
    public void execute(config.profile.ExecutionProfile profile, PublicationPolicy publicationPolicy) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        try (PreparedExecution execution = prepare(profile.runtime())) {
            execution.execute(profile.mode(), publicationPolicy);
        }
    }

    /**
     * Prepares, executes, and returns per-run trace metadata.
     *
     * @param runtimeConfig runtime settings, or {@code null} for defaults
     * @param mode execution mode to run
     * @return run trace with duration and step metadata
     */
    public RunTrace executeTraced(config.runtime.RuntimeConfig runtimeConfig, ExecutionMode mode) {
        try (PreparedExecution execution = prepare(runtimeConfig)) {
            return execution.executeTraced(mode);
        }
    }

    /**
     * Prepares, executes, and returns per-run trace metadata with an explicit publication policy.
     *
     * @param runtimeConfig runtime settings, or {@code null} for defaults
     * @param mode execution mode to run
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @return run trace with duration and step metadata
     */
    public RunTrace executeTraced(
            config.runtime.RuntimeConfig runtimeConfig,
            ExecutionMode mode,
            PublicationPolicy publicationPolicy
    ) {
        try (PreparedExecution execution = prepare(runtimeConfig)) {
            return execution.executeTraced(mode, publicationPolicy);
        }
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
        try (PreparedExecution execution = prepare(profile.runtime())) {
            return execution.executeTraced(profile.mode());
        }
    }

    /**
     * Prepares, executes, and returns per-run trace metadata using an execution profile and publication policy.
     *
     * @param profile profile that supplies runtime settings and execution mode
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     * @return run trace with duration and step metadata
     * @throws IllegalArgumentException if {@code profile} is {@code null}
     */
    public RunTrace executeTraced(config.profile.ExecutionProfile profile, PublicationPolicy publicationPolicy) {
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        try (PreparedExecution execution = prepare(profile.runtime())) {
            return execution.executeTraced(profile.mode(), publicationPolicy);
        }
    }

    /**
     * Executes a previously prepared plan.
     *
     * <p>The prepared plan must have been created from this compiled graph facade.
     *
     * @param execution prepared execution plan to run
     * @param mode execution mode to run
     */
    public void executePrepared(PreparedExecution execution, ExecutionMode mode) {
        execution.requireCompatibleGraph(
                compileArtifacts().publication().rootTensor(),
                compileArtifacts().publication().graphContract()
        );
        execution.execute(mode);
    }

    /**
     * Executes a previously prepared plan with an explicit publication policy.
     *
     * @param execution prepared execution plan to run
     * @param mode execution mode to run
     * @param publicationPolicy values to publish back to user-visible tensors after execution
     */
    public void executePrepared(PreparedExecution execution, ExecutionMode mode, PublicationPolicy publicationPolicy) {
        execution.requireCompatibleGraph(
                compileArtifacts().publication().rootTensor(),
                compileArtifacts().publication().graphContract()
        );
        execution.execute(mode, publicationPolicy);
    }


    /**
     * Returns publication tensors marked as trainable parameters in the compiled forward graph.
     *
     * @return immutable trainable parameter list
     */
    public List<Tensor> trainableParameters() {
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        return compileArtifacts().publication().trainableParameters().stream()
                .map(graph.compile.publication.PublicationPlan.TrainableParameterBinding::parameterTensor)
                .filter(tensor -> seen.put(tensor, Boolean.TRUE) == null)
                .toList();
    }

    /**
     * Returns the user-visible root tensor passed to compilation.
     *
     * @return root tensor
     */
    public Tensor getRootTensor() {
        return rootTensor;
    }

    /**
     * Returns the compiled program nodes in execution order.
     *
     * @return immutable compiled node snapshots from the latest compile artifacts
     */
    public List<CompiledNode> compiledNodes() {
        return compileArtifacts().program().compiledNodes();
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
        return artifacts;
    }

}

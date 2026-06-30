package graph;

import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.compile.CompileArtifacts;
import graph.compile.CompiledProgram;
import graph.compile.GraphCompiler;
import graph.compile.canonical.SemanticForwardCanonicalizer;
import graph.compile.publication.PublicationPlan;
import graph.optimizer.GraphOptimizer;
import planning.intent.BackendIntentPlan;
import prepare.orchestration.PreparedExecutionBuilder;
import runtime.contract.ExecutionMode;
import runtime.execution.PreparedExecution;
import trace.compile.CompileTrace;
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
 *     <li>{@link #prepare(RuntimeConfig)} lowers those compile artifacts into runtime execution steps
 *     for a specific runtime configuration.</li>
 *     <li>{@link PreparedExecution#execute(ExecutionMode)} runs the prepared steps and publishes values according to
 *     the selected {@link runtime.execution.PublicationPolicy}.</li>
 * </ol>
 *
 * <p>Compilation and preparation allocate new artifact objects, while execution mutates tensor data, gradient buffers,
 * and backend workspaces. Instances are immutable compile results and can be prepared for multiple runtime configs.
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
            CompileMode compileMode,
            BackendIntentPlan backendIntentPlan
    ) {
        this.rootTensor = rootTensor;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        GraphCompiler compiler = new GraphCompiler(
                rootTensor,
                forwardCanonicalizer,
                forwardOptimizer,
                compileConfig,
                this.compileMode,
                backendIntentPlan
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
        return compile(rootTensor, compileConfig, compileMode, BackendIntentPlan.empty());
    }

    /**
     * Compiles {@code rootTensor} using a compile configuration and explicit backend intent plan.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param compileConfig compile-time semantic, graph, backend, region, and memory planning configuration
     * @param backendIntentPlan compile-local backend intent plan, or {@code null} for CPU-default
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code compileConfig} is {@code null}
     */
    public static CompiledGraph compile(
            Tensor rootTensor,
            CompileConfig compileConfig,
            BackendIntentPlan backendIntentPlan
    ) {
        return compile(rootTensor, compileConfig, CompileMode.AUTO, backendIntentPlan);
    }

    /**
     * Compiles {@code rootTensor} using a compile configuration and explicit backend intent plan.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param compileConfig compile-time semantic, graph, backend, region, and memory planning configuration
     * @param compileMode requested forward/backward compilation mode, or {@code null} for automatic mode
     * @param backendIntentPlan compile-local backend intent plan, or {@code null} for CPU-default
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code compileConfig} is {@code null}
     */
    public static CompiledGraph compile(
            Tensor rootTensor,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendIntentPlan backendIntentPlan
    ) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (compileConfig == null) {
            throw new IllegalArgumentException("compileConfig cannot be null");
        }
        return new CompiledGraph(
                rootTensor,
                compileConfig.semanticCanonicalization().enabled()
                        ? new SemanticForwardCanonicalizer(compileConfig.semanticCanonicalization().rewrite())
                        : null,
                graph.optimizer.OptimizerFactory.create(compileConfig.graphOptimization()),
                compileConfig,
                compileMode,
                backendIntentPlan
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
        return compile(rootTensor, optimizer, compileMode, BackendIntentPlan.empty());
    }

    /**
     * Compiles {@code rootTensor} with an explicit optimizer, compile mode, and backend intent plan.
     *
     * @param rootTensor output tensor that anchors the graph to compile
     * @param optimizer ordered optimizer rule pipeline to apply to the graph
     * @param compileMode requested forward/backward compilation mode, or {@code null} for automatic mode
     * @param backendIntentPlan compile-local backend intent plan, or {@code null} for CPU-default
     * @return compiled graph facade ready for preparation or direct execution
     * @throws IllegalArgumentException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public static CompiledGraph compile(
            Tensor rootTensor,
            GraphOptimizer optimizer,
            CompileMode compileMode,
            BackendIntentPlan backendIntentPlan
    ) {
        if (rootTensor == null) {
            throw new IllegalArgumentException("rootTensor cannot be null");
        }
        if (optimizer == null) {
            throw new IllegalArgumentException("optimizer cannot be null");
        }
        return new CompiledGraph(rootTensor, null, optimizer, CompileConfig.inference(), compileMode, backendIntentPlan);
    }

    /**
     * Returns whether the compiled artifacts include backward execution steps.
     *
     * @return {@code true} when {@link ExecutionMode#FORWARD_BACKWARD} can be run
     */
    public boolean supportsBackward() {
        return artifacts.program().supportsBackward();
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
        return prepare((RuntimeConfig) null);
    }

    /**
     * Prepares runtime execution steps for a runtime configuration.
     *
     * <p>Preparation chooses execution operations and kernels, builds per-node runtime metadata, and records
     * {@link trace.prepare.PrepareTrace}. The returned object is reusable for repeated executions with the same
     * runtime configuration, but each execution mutates tensor storage and gradients.
     *
     * @param runtimeConfig runtime settings, or {@code null} to choose training or inference defaults
     * @return prepared execution plan bound to this graph's compile artifacts
     */
    public PreparedExecution prepare(RuntimeConfig runtimeConfig) {
        RuntimeConfig effectiveConfig = runtimeConfig == null
                ? (supportsBackward()
                ? RuntimeConfig.trainingDefaults(rootTensor.getDataType())
                : RuntimeConfig.inferenceDefaults(rootTensor.getDataType()))
                : runtimeConfig;
        return PreparedExecutionBuilder.prepare(artifacts, effectiveConfig);
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
     * Returns publication tensors marked as trainable parameters in the compiled forward graph.
     *
     * @return immutable trainable parameter list
     */
    public List<Tensor> trainableParameters() {
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        return artifacts.publication().trainableParameters().stream()
                .map(graph.compile.publication.PublicationPlan.TrainableParameterBinding::parameterTensor)
                .filter(tensor -> seen.put(tensor, Boolean.TRUE) == null)
                .toList();
    }

    /**
     * Returns the executable compiled program snapshot.
     *
     * @return immutable compiled program
     */
    public CompiledProgram program() {
        return artifacts.program();
    }

    /**
     * Returns the publication plan for user-visible tensors and gradients.
     *
     * @return immutable publication plan
     */
    public PublicationPlan publication() {
        return artifacts.publication();
    }

    /**
     * Returns metadata from the latest compile call.
     *
     * @return compile trace, or a skipped trace before compilation has run
     */
    public CompileTrace compileTrace() {
        return compileTrace;
    }

}

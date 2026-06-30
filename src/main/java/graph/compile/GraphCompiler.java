package graph.compile;

import backend.partition.BackendPartitionDescriptorRegistry;
import config.compile.CompileConfig;
import graph.compile.canonical.SemanticForwardCanonicalizer;
import graph.compile.intent.BackendIntentPlan;
import graph.compile.planning.BackendPlanningJobResolver;
import graph.compile.planning.BackendPlanningService;
import graph.compile.session.CompileSession;
import graph.execution.trace.CompileTrace;
import graph.optimizer.GraphOptimizer;
import tensor.CompileMode;
import tensor.Tensor;

import java.util.Objects;

/**
 * Builds compile artifacts for a tensor graph.
 *
 * <p>A compiler session captures the semantic forward graph, optionally canonicalizes it, builds backward targets when
 * the selected {@link CompileMode} requires gradients, applies optimizer rules, snapshots compiled nodes, plans backend
 * partitions, and produces the memory plan used during preparation. Each call to {@link #compile()} creates a fresh
 * session and fresh artifacts; the compiler object itself only stores construction-time configuration.
 *
 * <p>The compiler is not designed for concurrent calls against mutable user-visible tensors. The returned artifacts are
 * immutable views, but compilation reads and updates graph metadata such as gradient bindings.
 */
public final class GraphCompiler {
    private final Tensor rootTensor;
    private final SemanticForwardCanonicalizer forwardCanonicalizer;
    private final GraphOptimizer optimizer;
    private final CompileConfig compileConfig;
    private final BackendPartitionDescriptorRegistry backendPartitionDescriptors;
    private final BackendPlanningService backendPlanningService;
    private final BackendIntentPlan backendIntentPlan;
    private final CompileMode compileMode;

    /**
     * Creates a compiler using the default graph policy configuration.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param compileConfig compile-time graph policy configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                compileConfig,
                compileMode,
                BackendPartitionDescriptorRegistry.defaults(),
                BackendIntentPlan.empty()
        );
    }

    /**
     * Creates a compiler with an explicit backend partition descriptor registry.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param compileConfig compile-time graph policy configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     * @param backendPartitionDescriptors registry used to resolve backend legality and lowering plans
     * @throws NullPointerException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendIntentPlan backendIntentPlan
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                compileConfig,
                compileMode,
                BackendPartitionDescriptorRegistry.defaults(),
                backendIntentPlan
        );
    }

    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors
    ) {
        this(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                compileConfig,
                compileMode,
                backendPartitionDescriptors,
                BackendIntentPlan.empty()
        );
    }

    /**
     * Creates a compiler with explicit compile-local backend intent.
     *
     * @param rootTensor output tensor that anchors the graph
     * @param forwardCanonicalizer optional semantic forward canonicalizer; {@code null} disables this stage
     * @param optimizer optimizer pipeline applied after graph construction
     * @param compileConfig compile-time graph policy configuration, or {@code null} for defaults
     * @param compileMode requested compile mode, or {@code null} for {@link CompileMode#AUTO}
     * @param backendPartitionDescriptors registry used to resolve backend legality and lowering plans
     * @param backendIntentPlan explicit backend intent plan, or {@code null} for CPU-default
     * @throws NullPointerException if {@code rootTensor} or {@code optimizer} is {@code null}
     */
    public GraphCompiler(
            Tensor rootTensor,
            SemanticForwardCanonicalizer forwardCanonicalizer,
            GraphOptimizer optimizer,
            CompileConfig compileConfig,
            CompileMode compileMode,
            BackendPartitionDescriptorRegistry backendPartitionDescriptors,
            BackendIntentPlan backendIntentPlan
    ) {
        this.rootTensor = Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        this.forwardCanonicalizer = forwardCanonicalizer;
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
        this.compileConfig = compileConfig == null ? CompileConfig.inference() : compileConfig;
        this.compileMode = compileMode == null ? CompileMode.AUTO : compileMode;
        this.backendIntentPlan = backendIntentPlan == null ? BackendIntentPlan.empty() : backendIntentPlan;
        this.backendPartitionDescriptors = backendPartitionDescriptors == null
                ? BackendPartitionDescriptorRegistry.defaults()
                : backendPartitionDescriptors;
        this.backendPlanningService = new BackendPlanningService(
                new BackendPlanningJobResolver(),
                this.backendPartitionDescriptors
        );
    }

    /**
     * Runs a full compile session.
     *
     * <p>The result contains both the immutable artifact bundle and timing/count metadata. Compile tracing stops at the
     * artifact boundary; kernel selection and execution timing are recorded later during preparation and execution.
     *
     * @return compile result for the latest graph state
     */
    public Result compile() {
        long t0 = System.nanoTime();
        CompileSession session = new CompileSession(
                rootTensor,
                forwardCanonicalizer,
                optimizer,
                compileConfig,
                compileMode,
                backendPartitionDescriptors,
                backendPlanningService,
                backendIntentPlan
        );
        CompileArtifacts artifacts = session.compile();
        CompileTrace trace = new CompileTrace(
                true,
                System.nanoTime() - t0,
                artifacts.compiledNodes().size(),
                session.forwardGraphSize(),
                artifacts.supportsBackward(),
                session.partitionPlanningTrace(),
                session.optimizerTrace()
        );
        return new Result(artifacts, trace);
    }

    /**
     * Pair of compile artifacts and the trace that describes the compile session.
     *
     * @param artifacts immutable artifact bundle produced by compilation
     * @param trace compile timing and node-count metadata; {@link CompileTrace#skipped()} is substituted for
     *              {@code null}
     */
    public record Result(CompileArtifacts artifacts, CompileTrace trace) {
        public Result {
            artifacts = Objects.requireNonNull(artifacts, "artifacts cannot be null");
            trace = trace == null ? CompileTrace.skipped() : trace;
        }
    }
}

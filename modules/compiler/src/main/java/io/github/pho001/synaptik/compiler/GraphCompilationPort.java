package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.util.List;
import java.util.Optional;

/**
 * Provides the narrow Compiler integration boundary used by lifecycle-composition modules.
 *
 * <p>This service-provider interface (SPI) exposes the complete immutable Compiler result without
 * exposing the package-private graph compiler or its explicit logical-constant ingress. It exists
 * for future Engine and other lifecycle-composition code; ordinary application code should use
 * the future Engine facade instead. This type does not provide an Engine lifecycle, preparation,
 * or execution.</p>
 *
 * <p>Because this boundary always supplies empty explicit constant ingress, every reachable
 * provenance-free forward leaf remains caller-bindable. Constants that the Compiler creates
 * internally for a functional gradient request retain their existing Compiler-owned treatment.</p>
 */
public final class GraphCompilationPort {
    private GraphCompilationPort() {}

    /**
     * Compiles one ordered Tensor-expression boundary into immutable backend-neutral artifacts.
     *
     * <p>The request delegates to the complete Compiler pipeline with empty explicit forward
     * constant ingress. Arguments, list elements, the mode/request combination, and the graph
     * retain the pipeline's declaration-order validation. Capability providers and availability
     * snapshots are encountered in their supplied order when graph nodes are planned; neither
     * list is retained in the result.</p>
     *
     * @param mode non-null graph-scope mode
     * @param forwardOutputs non-null, non-empty ordered forward boundary; exact Tensor references
     *     and resolved logical values must be unique, and the list is not mutated
     * @param functionalGradientRequest non-null optional functional request, absent exactly for
     *     {@link CompileMode#FORWARD_ONLY} and present for both backward-capable modes
     * @param optimizationConfig non-null permission for the existing optional exact graph
     *     optimizations; mandatory canonicalization and validation still occur
     * @param backendIntent non-null hard-target request, which may be unconstrained
     * @param partitionScoringConfig non-null coarse device-class preference request, which may be
     *     neutral
     * @param capabilityProviders non-null ordered capability-provider list; elements must be
     *     non-null when inspected and are encountered only when a graph node is planned
     * @param availabilitySnapshots non-null ordered availability-snapshot list; elements must be
     *     non-null when inspected and are encountered only when a graph node is planned
     * @return non-null immutable complete compile artifacts containing the final graph,
     *     publication and input roles, backend-neutral plans, diagnostics, and derivative
     *     metadata; the result retains no provider or availability snapshot
     * @throws NullPointerException if a required argument or inspected nested value is
     *     {@code null}
     * @throws IllegalArgumentException if graph compilation, publication, planning composition,
     *     partitioning, memory derivation, or artifact cross-validation rejects the request
     * @throws IllegalStateException if a graph node has no hard-eligible backend; the existing
     *     node-context message and Planning cause are preserved
     * @throws RuntimeException if a capability provider throws another runtime failure; the same
     *     failure instance propagates unchanged
     */
    public static CompileArtifacts compile(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            Optional<FunctionalGradientRequest> functionalGradientRequest,
            GraphOptimizationConfig optimizationConfig,
            BackendIntent backendIntent,
            PartitionScoringConfig partitionScoringConfig,
            List<BackendCapabilityProvider> capabilityProviders,
            List<BackendAvailabilitySnapshot> availabilitySnapshots) {
        return GraphCompiler.compile(
                mode,
                forwardOutputs,
                functionalGradientRequest,
                CompileTimeConstantGraph.Ingress.empty(),
                optimizationConfig,
                backendIntent,
                partitionScoringConfig,
                capabilityProviders,
                availabilitySnapshots);
    }
}

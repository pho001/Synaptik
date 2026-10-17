package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Provides the narrow Compiler integration boundary used by current Engine and other
 * lifecycle-composition code.
 *
 * <p>This service-provider interface (SPI) exposes the complete immutable Compiler result without
 * exposing the package-private graph compiler or its explicit logical-constant ingress. Current
 * Engine and other lifecycle-composition code use this boundary; ordinary application code
 * should use Engine instead. This type does not provide an Engine lifecycle, preparation, or
 * execution.</p>
 *
 * <p>The cold source methods additionally preserve a bounded raw forward graph and reconstruct
 * ordinary symbolic expressions without entering the compile/prepare/run lifecycle. Their
 * borrowed {@link SourceCapture} leaf sidecar is separate from Tensor-free compile artifacts.</p>
 *
 * <p>Compilation and source capture supply empty explicit constant ingress, so every reachable
 * provenance-free forward leaf remains caller-bindable and is associated with its immutable
 * logical Tensor identity. Constants that the Compiler creates internally for a functional
 * gradient request retain their existing Compiler-owned treatment.</p>
 *
 * <p>Compiler-generated derivative formulas consist of ordinary public Tensor operations and
 * inherit Model's operation semantics. Capability questions remain occurrence-specific; neither
 * the request nor its artifacts carry a numerical selector or production tolerance policy.</p>
 */
public final class GraphCompilationPort {
    private GraphCompilationPort() {}

    /**
     * Captures and verifies a bounded raw forward source graph and its exact ordered leaves.
     *
     * <p>One identity-based traversal assigns inputs and collects leaves together. Admission is
     * limited to fully static ADD, MUL, NEG, MATMUL and rank-two PERMUTE with axes {@code [1,0]}.
     * Existing family inference verifies every raw descriptor without rewriting it. This is not
     * the optimized, layout-closed graph returned by {@link CompileArtifacts#graph()}.</p>
     *
     * @param forwardOutputs non-null, non-empty ordered results without nulls, repeated exact
     *     references or duplicate logical outputs; the list is not mutated
     * @return non-null immutable raw graph and borrowed leaf snapshot aligned to graph inputs;
     *     no storage is read, allocated or owned, and no execution or planning occurs
     * @throws NullPointerException if the list or an element is null
     * @throws IllegalArgumentException if the boundary is empty or duplicated, or any source
     *     occurrence, static geometry or descriptor is invalid or unsupported; checked geometry
     *     overflow is retained as a cause
     */
    public static SourceCapture captureSource(List<Tensor> forwardOutputs) {
        return GraphCapture.captureSource(forwardOutputs);
    }

    /**
     * Reconstructs a bounded raw graph through ordinary public Tensor family constructors.
     *
     * <p>Complete cold preflight precedes every producer allocation. Each stored node is built
     * once, preserving occurrence identity and shared uses. A call-local reconstruction table is
     * discarded on return or failure; returned Tensors retain only ordinary provenance. No
     * storage is accessed or allocated, and nothing is compiled, prepared or executed.</p>
     *
     * @param graph non-null raw graph in the same static FORWARD slice as {@link #captureSource};
     *     every stored node and value is checked, including dead nodes and unused inputs
     * @param leaves non-null ordered substitutions, snapshotted without caller mutation; position
     *     {@code i} supplies {@code graph.inputs().get(i)} with full descriptor equality,
     *     including layout and gradient eligibility; fresh identities, ordinary expressions and
     *     repeated references are permitted, and upstream provenance is neither detached nor
     *     traversed by source admission; same-descriptor swaps cannot be detected
     * @return non-null immutable outputs in graph order; pass-through outputs are exact supplied
     *     references, so explicit repeated substitutions may produce repeated output references
     * @throws NullPointerException if an argument or supplied element is null
     * @throws IllegalArgumentException if source admission or positional association fails,
     *     including a layout-closed result descriptor or checked geometry overflow
     * @throws IllegalStateException if Tensor identity allocation is exhausted; consumed IDs are
     *     not rolled back and no partial result is published
     */
    public static List<Tensor> reconstructSource(CompiledGraphModel graph, List<Tensor> leaves) {
        return SourceGraphReconstruction.reconstruct(graph, leaves);
    }

    /**
     * Pairs a verified raw source graph with exact provenance-free input references.
     *
     * <p>This borrowed sidecar has no storage ownership, close lifecycle or cache and is never
     * retained by compile artifacts. Record equality includes both components; Tensor elements
     * retain ordinary identity semantics. The graph remains Tensor-free.</p>
     *
     * @param graph non-null raw graph admitted by the bounded cold source boundary
     * @param leaves non-null immutable snapshot of unique provenance-free exact Tensors in
     *     graph-input order, each with its input's complete descriptor
     */
    public record SourceCapture(CompiledGraphModel graph, List<Tensor> leaves) {
        /**
         * Verifies source admission and snapshots exact borrowed leaves, including on direct use.
         *
         * @param graph non-null raw graph to retain unchanged; all stored nodes and values must
         *     satisfy static FORWARD source admission
         * @param leaves non-null input-aligned list to snapshot without mutation; elements must
         *     be non-null, provenance-free, identity-unique and descriptor-equal to their inputs
         * @throws NullPointerException if an argument or element is null
         * @throws IllegalArgumentException if source admission, checked geometry, count,
         *     descriptor equality, provenance absence or identity uniqueness fails
         */
        public SourceCapture {
            Objects.requireNonNull(graph, "graph");
            leaves = List.copyOf(Objects.requireNonNull(leaves, "leaves"));
            var descriptors = SourceGraphReconstruction.preflight(graph);
            SourceGraphReconstruction.validateLeaves(graph, descriptors, leaves, true);
        }
    }

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
     *     publication and input roles including typed caller identities, backend-neutral plans,
     *     diagnostics, and derivative metadata; the result retains no Tensor, provider, or
     *     availability snapshot
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

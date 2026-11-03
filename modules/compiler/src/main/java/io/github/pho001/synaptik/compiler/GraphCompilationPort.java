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
 *
 * <p>Ordinary compilation captures general scan bodies in independent lexical scopes. Per-value
 * floating and discrete primals use ordinary PURE inference; first-order differentiation proves
 * the exact selected finite-time root/input routes and the actual generated ordinary closure
 * before materialization. Joint selected carry/emission seeds create one compact reverse sibling.
 * Structural cotangent absence is retained separately from numerical zero in component-eight
 * derivative metadata. Residual differentiation and Scan/fixed-recurrent higher-order requests
 * still reject before derivative allocation. Final capability queries
 * retain the complete exact body and its applicable activity plan, without granting backend
 * support. Cold source reconstruction remains independently bounded as documented by
 * {@link #captureSource(List)}; execution and persistence retain their own gates.</p>
 * <p>Length-aware ordinary compilation independently checks bounded row semantics. Its first-order
 * AD requires every original carry and sequence to be ROW(0), even unused/non-gradient positions;
 * generated reverse histories/seeds retain explicit mixed roles and the same lengths actual.
 * Cold capture/reconstruction admits only the fully static floating length intersection below,
 * retaining each role's dtype. Compiler success grants no length-mode backend or codec support.</p>
 */
public final class GraphCompilationPort {
    private GraphCompilationPort() {}

    /**
     * Captures and verifies a bounded raw forward source graph and its exact ordered leaves.
     *
     * <p>One identity-based traversal assigns inputs and collects leaves together. Admission is
     * limited to fully static FORWARD ADD/MUL/NEG/MATMUL, floating SUB/RELU/TANH/SIGMOID, rank-two
     * PERMUTE {@code [1,0]}, positive-target RESHAPE, axis-two EXPAND_DIMS/SQUEEZE,
     * rank-three axis-zero SELECT/CONCAT, rank-two axis-zero GATHER with non-gradient INT64
     * vector indices, one nonempty step-one last-axis rank-two SLICE, biased default
     * CONV2D/CONV3D, nonoverlapping floor MAX_POOL2D with stride equal to kernel, no padding
     * and unit dilation, MAX_POOL3D/AVERAGE_POOL3D with their complete typed Model geometry,
     * and the six biased/unbiased fixed scan signatures in either direction. General SCAN is
     * additionally admitted with independently scoped bodies in the bounded constructor closure
     * described below; body-only families do not widen this flat inventory.
     * Index/length values are never read; time/batch may be zero for scans. This grants no scan
     * execution or BPTT. Conv3d differentiation is independently checked by autograd preflight;
     * this source inventory does not widen its constructor geometry. Public constructors retain ordinary family guards.
     * Existing family inference verifies every raw descriptor without rewriting it. This is not
     * the optimized, layout-closed graph returned by {@link CompileArtifacts#graph()}.</p>
     *
     * <p>General bodies reuse the flat mappings except scans, and additionally admit tensor DIV,
     * integral tensor SUB, exact typed scalar ADD/SUB/MUL/DIV, full/axis/ordered-multi-axis/target-Shape
     * SUM, static EXPAND, scalar/empty RESHAPE, arbitrary normalized EXPAND_DIMS/SQUEEZE,
     * CONTIGUOUS and floating-to-floating or identity CAST. Model family/dtype guards still apply,
     * independently of CPU qualification (including low TANH and FLOAT32 RELU). No other PURE
     * family is implicitly admitted. Constants are explicit outer bindings to invariant formals
     * or scalar attributes. All-valid bodies retain every applicable Model dtype. Length bodies
     * require floating FLOAT32/FLOAT64/BFLOAT16/FLOAT16 declarations, actuals and interiors, with
     * each carry matching its own dtype and explicit CAST preserving mixed working boundaries.
     * Nonfloating length-body values remain unsupported. Separate non-gradient INT64[B] lengths
     * and the existing ROW(0), rank-two ROW(1), SHARED and terminal-SHARED rules are retained.
     * Pool3d retains those same length-mode coordinate rules, including unchanged row axes;
     * no other formerly excluded length-body constructor is admitted by this addition.
     * Bodies require empty raw demand and graph outputs equal to first-encounter-unique
     * next-carry, emission and residual roles in that order. Inference success alone is not cold
     * admission. Consumers retain independent persistence, execution and AD gates.</p>
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
     * Validates complete cold-source metadata before a consumer allocates candidate Tensor leaves.
     * Uses the same static FORWARD constructor, exact descriptor, scoped body and length-role
     * checks as {@link #captureSource(List)} and {@link SourceCapture}; this method constructs
     * no Tensor identities, reads no storage and grants no execution or persistence capability.
     * Leaf identity/substitution validation still belongs to acquisition or reconstruction.
     * @param graph non-null raw source graph, including all stored nodes and body attachments
     * @throws NullPointerException if graph is null
     * @throws IllegalArgumentException if any source metadata is invalid or outside the closed
     *     cold inventory; checked geometry overflow is retained with its existing context
     */
    public static void validateSource(CompiledGraphModel graph) {
        SourceGraphReconstruction.preflight(graph);
    }

    /**
     * Reconstructs a bounded raw graph through ordinary public Tensor family constructors.
     *
     * <p>Complete outer and contained-body cold preflight precedes every formal or producer
     * allocation, including dead scans and zero-step bodies. Each stored node is built
     * once, preserving every canonical output port, occurrence identity and shared uses, including
     * unconsumed scan siblings and latent residuals. Each general occurrence receives distinct
     * fresh canonical storage-free formals for all declarations (including unused positions),
     * one ordinary body and one Scan.apply, never time-unrolled. Reused templates and repeated
     * actual/role positions preserve independent scopes and canonical ports. Length reconstruction
     * preserves ordered row roles and traversal, binding exactly c+s+i formals and the separate
     * final lengths actual. Newly tied substitutions must identify the same outer row axis or both
     * be SHARED within each occurrence; these checks precede all IDs. Separate call-local
     * reconstruction tables are discarded on return or failure; body IDs never resolve outside
     * their containing scope. Returned Tensors retain only ordinary provenance. No
     * storage is accessed or allocated, and nothing is compiled, prepared or executed.</p>
     *
     * @param graph non-null raw graph in the same static FORWARD slice as {@link #captureSource};
     *     every stored node and value is checked, including dead nodes and unused inputs
     * @param leaves non-null ordered substitutions, snapshotted without caller mutation; position
     *     {@code i} supplies {@code graph.inputs().get(i)} with full descriptor equality,
     *     including layout and gradient eligibility; fresh identities, ordinary expressions and
     *     repeated references are permitted, and upstream provenance is neither detached nor
     *     traversed by source admission; length-role compatibility is checked per occurrence,
     *     and same-descriptor swaps cannot otherwise be detected
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
     * retained by compile artifacts. General bodies are Tensor-free independent graph scopes
     * with raw empty demand and ordered unique role outputs, not borrowed source templates.
     * Length bodies retain each role's floating dtype, explicit casts, row roles and separate
     * lengths input; static body eligibility is checked independently of ordinary inference and
     * of codec eligibility.
     * Record equality includes both components; Tensor elements
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
     * <p>General scan source admission and first-order AD do not imply an eligible provider. The
     * provider must accept the entire body and loop binder; no inner mixed-owner planning occurs.</p>
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

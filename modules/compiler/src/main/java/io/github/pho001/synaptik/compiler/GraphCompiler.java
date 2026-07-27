package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Owns package-private mode routing for capture, first-order expansion, and exact optimization.
 *
 * <p>{@link CompileMode#FORWARD_ONLY} captures and compiles only the requested forward boundary.
 * The two backward-capable modes preflight one scalar first-order request, construct gradients
 * with public Tensor operations, and capture forward outputs and gradient roots together once.
 * Every mode then passes its single immutable graph through inference, mandatory
 * canonicalization, bounded exact optional optimization, and final validation.</p>
 *
 * <p>This owner returns internal graph-stage state only. It performs no publication or planning
 * orchestration, storage allocation, backend lowering, preparation, execution, optimizer update,
 * or higher-order differentiation and exposes no public compiler facade.</p>
 */
final class GraphCompiler {
    private GraphCompiler() {}

    /**
     * Compiles one forward-only or combined first-order Tensor expression graph.
     *
     * <p>Top-level arguments are checked in declaration order before graph construction. Known
     * unsupported first-order facts fail during complete preflight before a seed, derivative
     * constant, or formula Tensor is created. Failures after expansion begins may consume
     * temporary opaque Tensor IDs; identifiers are never rolled back or reused.</p>
     *
     * @param mode non-null graph-scope mode
     * @param forwardOutputs non-null, non-empty ordered forward boundary; exact Tensor references
     *     and resolved logical values must be unique, and the list is not mutated
     * @param firstOrderRequest non-null optional scalar-objective request, absent exactly for
     *     {@link CompileMode#FORWARD_ONLY} and present for both backward-capable modes
     * @param forwardConstants non-null ordered explicit logical-splat bindings for reachable
     *     forward leaves
     * @param optimizationConfig non-null permission controlling the optional exact pass sequence;
     *     inference, canonicalization, and validation remain mandatory
     * @return a non-null immutable mode-neutral graph compilation with ordered forward values and
     *     target-specific gradient roles
     * @throws NullPointerException if a top-level argument or required list element is
     *     {@code null}, checked in declaration order
     * @throws IllegalArgumentException if the forward boundary is empty or duplicates a Tensor or
     *     logical value, the mode/request matrix is invalid, preflight rejects the request,
     *     ingress is invalid, or capture, inference, validation, optimization, or final boundary
     *     validation fails
     */
    static GraphCompilation compile(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            Optional<AutogradPreflight.FirstOrderRequest> firstOrderRequest,
            CompileTimeConstantGraph.Ingress forwardConstants,
            GraphOptimizationConfig optimizationConfig) {
        Objects.requireNonNull(mode, "mode");
        validateForwardOutputs(forwardOutputs);
        Objects.requireNonNull(firstOrderRequest, "firstOrderRequest");
        Objects.requireNonNull(forwardConstants, "forwardConstants");
        Objects.requireNonNull(optimizationConfig, "optimizationConfig");

        if (mode == CompileMode.FORWARD_ONLY) {
            if (firstOrderRequest.isPresent()) {
                throw new IllegalArgumentException(
                        "FORWARD_ONLY must not include a first-order request");
            }
            CompileTimeConstantGraph captured =
                    GraphCapture.capture(forwardOutputs, forwardConstants);
            ValidatedGraph inferred = CapturedGraphInference.inferAndValidate(captured);
            ValidatedGraph optimized =
                    ForwardGraphOptimization.optimize(inferred, optimizationConfig);
            List<ValueId> finalForward = List.copyOf(optimized.graph().outputs());
            return new GraphCompilation(mode, optimized, finalForward, List.of());
        }
        if (firstOrderRequest.isEmpty()) {
            throw new IllegalArgumentException(mode + " requires a first-order request");
        }

        AutogradPreflight.Plan plan =
                AutogradPreflight.preflight(
                        mode,
                        forwardOutputs,
                        firstOrderRequest.orElseThrow(),
                        forwardConstants);
        FirstOrderAutograd.Expansion expansion =
                FirstOrderAutograd.expand(plan, forwardConstants);
        GraphCapture.CombinedCapture captured = GraphCapture.captureCombined(
                forwardOutputs,
                expansion.targetGradients(),
                expansion.originalProducers(),
                expansion.ingress());
        ValidatedGraph inferred =
                CapturedGraphInference.inferAndValidate(captured.constantGraph());
        ValidatedGraph optimized =
                ForwardGraphOptimization.optimize(inferred, optimizationConfig);

        List<ValueId> finalForward = List.copyOf(
                optimized.graph().outputs().subList(0, captured.forwardOutputCount()));
        List<GraphCompilation.GradientResultRole> gradientResults =
                new ArrayList<>(expansion.targetGradients().size());
        for (int index = 0; index < expansion.targetGradients().size(); index++) {
            int ordinal = captured.gradientOutputOrdinals().get(index);
            gradientResults.add(new GraphCompilation.GradientResultRole(
                    expansion.targetGradients().get(index).target().id(),
                    optimized.graph().outputs().get(ordinal)));
        }
        return new GraphCompilation(mode, optimized, finalForward, gradientResults);
    }

    private static void validateForwardOutputs(List<Tensor> forwardOutputs) {
        Objects.requireNonNull(forwardOutputs, "forwardOutputs");
        if (forwardOutputs.isEmpty()) {
            throw new IllegalArgumentException("forwardOutputs must not be empty");
        }
        IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
        IdentityHashMap<TensorProducer, int[]> producerOutputPositions = new IdentityHashMap<>();
        for (int index = 0; index < forwardOutputs.size(); index++) {
            Tensor output = Objects.requireNonNull(
                    forwardOutputs.get(index), "forwardOutputs[" + index + "]");
            Integer first = positions.putIfAbsent(output, index);
            if (first != null) {
                throw new IllegalArgumentException(
                        "forwardOutputs[" + index + "] duplicates forwardOutputs[" + first + "]");
            }
            TensorProvenance provenance = output.provenance().orElse(null);
            if (provenance != null) {
                int[] outputPositions = producerOutputPositions.computeIfAbsent(
                        provenance.producer(),
                        producer -> {
                            int[] created = new int[producer.outputCount()];
                            Arrays.fill(created, -1);
                            return created;
                        });
                int firstLogical = outputPositions[provenance.outputIndex()];
                if (firstLogical >= 0) {
                    throw new IllegalArgumentException(
                            "forwardOutputs[" + index + "] duplicates forwardOutputs["
                                    + firstLogical + "] logical producer output");
                }
                outputPositions[provenance.outputIndex()] = index;
            }
        }
    }
}

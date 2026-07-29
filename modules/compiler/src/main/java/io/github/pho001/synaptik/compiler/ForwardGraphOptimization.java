package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import java.util.Objects;

/**
 * Orders mandatory graph canonicalization and the bounded optional whole-graph optimization
 * pipeline.
 *
 * <p>The input is the single graph selected by the compile mode: forward-only or combined
 * forward/backward. Canonicalization always rebuilds graph-local identifiers densely and is then
 * validated. When optional optimization is enabled, the pipeline performs one guarded
 * seven-rule exact arithmetic scan, one exact logical-splat fold scan, whole-graph sidecar-aware
 * dead-code elimination (DCE), exact phase- and derivative-order-local common-subexpression
 * elimination (CSE), and one
 * whole-graph sidecar-aware cleanup DCE, once each in that order. Compiler verification is
 * repeated only after a helper returns a changed immutable candidate and before the next
 * transform consumes it.</p>
 *
 * <p>No pass reads Tensor storage, evaluates floating-point arithmetic, materializes constants,
 * iterates to a fixed point, merges across phases, plans a backend, or executes computation.</p>
 */
final class ForwardGraphOptimization {
    private ForwardGraphOptimization() {}

    /**
     * Canonicalizes and validates a successful graph, then optionally applies one guarded exact
     * arithmetic rewrite scan, constant folding, and one whole-graph
     * {@code DCE -> phase/order-local CSE -> DCE} sequence.
     *
     * @param validatedGraph the non-null successful compiler validation result whose immutable
     *     graph is read; neither the result nor its graph is mutated
     * @param optimizationConfig the non-null permission controlling only the optional pass
     *     sequence; canonicalization and validation remain mandatory
     * @return the non-null successful validation result for the final canonical candidate,
     *     retaining graph-output order, constant/source roles, constraints, phases, and
     *     derivative orders; an unchanged helper result is not revalidated
     * @throws NullPointerException if {@code validatedGraph} or {@code optimizationConfig} is
     *     {@code null}, checked in that order
     * @throws IllegalArgumentException if inference or validation rejects a rebuilt candidate
     */
    static ValidatedGraph optimize(
            ValidatedGraph validatedGraph, GraphOptimizationConfig optimizationConfig) {
        Objects.requireNonNull(validatedGraph, "validatedGraph");
        Objects.requireNonNull(optimizationConfig, "optimizationConfig");

        GraphCanonicalization.Result canonical =
                GraphCanonicalization.canonicalize(validatedGraph.derivatives());
        CompileTimeConstantGraph canonicalConstants =
                validatedGraph.constantGraph().replaceGraphPreservingInputRoles(canonical.graph());
        ValidatedGraph current = CapturedGraphInference.inferAndValidate(
                canonicalConstants, canonical.derivatives());
        if (!optimizationConfig.optionalOptimizationsEnabled()) {
            return current;
        }

        ForwardExactArithmeticRewriting.Result rewritten =
                ForwardExactArithmeticRewriting.rewrite(current.derivatives());
        current = validateWhenChanged(
                current,
                current.constantGraph().replaceGraphPreservingInputRoles(rewritten.graph()),
                rewritten.derivatives());
        ForwardConstantFolding.Result folded =
                ForwardConstantFolding.fold(current.constantGraph(), current.derivatives());
        current = validateWhenChanged(
                current, folded.constantGraph(), folded.derivatives());
        ForwardDeadCodeElimination.Result eliminated =
                ForwardDeadCodeElimination.DerivativeAware.eliminate(
                        current.constantGraph(), current.derivatives());
        current = validateWhenChanged(
                current, eliminated.constantGraph(), eliminated.derivatives());
        ForwardCommonSubexpressionElimination.Result common =
                ForwardCommonSubexpressionElimination.DerivativeAware.eliminate(
                        current.derivatives());
        current = validateWhenChanged(
                current,
                current.constantGraph().replaceGraphPreservingInputRoles(common.graph()),
                common.derivatives());
        ForwardDeadCodeElimination.Result cleanup =
                ForwardDeadCodeElimination.DerivativeAware.eliminate(
                        current.constantGraph(), current.derivatives());
        return validateWhenChanged(
                current, cleanup.constantGraph(), cleanup.derivatives());
    }

    private static ValidatedGraph validateWhenChanged(
            ValidatedGraph current,
            CompileTimeConstantGraph candidate,
            DerivativeGraphMetadata derivatives) {
        if (candidate == current.constantGraph()) {
            return current;
        }
        return CapturedGraphInference.inferAndValidate(candidate, derivatives);
    }
}

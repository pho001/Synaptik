package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import java.util.Objects;

/**
 * Orders mandatory graph canonicalization and the bounded optional forward optimization pipeline.
 *
 * <p>Canonicalization always rebuilds graph-local identifiers densely and is then validated.
 * When optional optimization is enabled, the pipeline performs one guarded seven-rule exact
 * arithmetic scan, one exact logical-splat fold scan, sidecar-aware dead-code elimination (DCE),
 * exact common-subexpression elimination (CSE), and one sidecar-aware cleanup DCE, once each in
 * that order. Compiler verification is repeated only after a helper returns a changed immutable
 * candidate and before the next transform consumes that candidate.</p>
 */
final class ForwardGraphOptimization {
    private ForwardGraphOptimization() {}

    /**
     * Canonicalizes and validates a successful graph, then optionally applies one guarded exact
     * arithmetic rewrite scan, constant folding, and one {@code DCE -> CSE -> DCE} sequence.
     *
     * @param validatedGraph the non-null successful compiler validation result whose immutable
     *     graph is read; neither the result nor its graph is mutated
     * @param optimizationConfig the non-null permission controlling only the optional pass
     *     sequence; canonicalization and validation remain mandatory
     * @return the non-null successful validation result for the final canonical candidate; an
     *     unchanged helper result is not revalidated
     * @throws NullPointerException if {@code validatedGraph} or {@code optimizationConfig} is
     *     {@code null}, checked in that order
     * @throws IllegalArgumentException if inference or validation rejects a rebuilt candidate
     */
    static ValidatedGraph optimize(
            ValidatedGraph validatedGraph, GraphOptimizationConfig optimizationConfig) {
        Objects.requireNonNull(validatedGraph, "validatedGraph");
        Objects.requireNonNull(optimizationConfig, "optimizationConfig");

        CompiledGraphModel canonical =
                GraphCanonicalization.canonicalize(validatedGraph.graph());
        CompileTimeConstantGraph canonicalConstants =
                validatedGraph.constantGraph().replaceGraphPreservingInputRoles(canonical);
        ValidatedGraph current = CapturedGraphInference.inferAndValidate(canonicalConstants);
        if (!optimizationConfig.optionalOptimizationsEnabled()) {
            return current;
        }

        CompiledGraphModel rewritten = ForwardExactArithmeticRewriting.rewrite(current.graph());
        current = validateWhenChanged(
                current, current.constantGraph().replaceGraphPreservingInputRoles(rewritten));
        current = validateWhenChanged(
                current, ForwardConstantFolding.fold(current.constantGraph()));
        current = validateWhenChanged(
                current, ForwardDeadCodeElimination.eliminate(current.constantGraph()));
        CompiledGraphModel common =
                ForwardCommonSubexpressionElimination.eliminate(current.graph());
        current = validateWhenChanged(
                current, current.constantGraph().replaceGraphPreservingInputRoles(common));
        return validateWhenChanged(
                current, ForwardDeadCodeElimination.eliminate(current.constantGraph()));
    }

    private static ValidatedGraph validateWhenChanged(
            ValidatedGraph current, CompileTimeConstantGraph candidate) {
        if (candidate == current.constantGraph()) {
            return current;
        }
        return CapturedGraphInference.inferAndValidate(candidate);
    }
}

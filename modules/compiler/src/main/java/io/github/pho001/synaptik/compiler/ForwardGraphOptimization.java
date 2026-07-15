package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import java.util.Objects;

/**
 * Orders mandatory graph canonicalization and the bounded optional forward optimization pipeline.
 *
 * <p>Canonicalization always rebuilds graph-local identifiers densely. When optional optimization
 * is enabled, the pipeline invokes dead-code elimination (DCE), exact common-subexpression
 * elimination (CSE), and one cleanup DCE, once each in that order. Every changed immutable
 * candidate is inferred and validated before another transform can consume it.</p>
 */
final class ForwardGraphOptimization {
    private ForwardGraphOptimization() {}

    /**
     * Canonicalizes a successful graph and optionally applies one {@code DCE -> CSE -> DCE} pass.
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
        ValidatedGraph current = CapturedGraphInference.inferAndValidate(canonical);
        if (!optimizationConfig.optionalOptimizationsEnabled()) {
            return current;
        }

        current = validateWhenChanged(
                current, ForwardDeadCodeElimination.eliminate(current.graph()));
        current = validateWhenChanged(
                current, ForwardCommonSubexpressionElimination.eliminate(current.graph()));
        return validateWhenChanged(
                current, ForwardDeadCodeElimination.eliminate(current.graph()));
    }

    private static ValidatedGraph validateWhenChanged(
            ValidatedGraph current, CompiledGraphModel candidate) {
        if (candidate == current.graph()) {
            return current;
        }
        return CapturedGraphInference.inferAndValidate(candidate);
    }
}

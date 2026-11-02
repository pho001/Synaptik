package io.github.pho001.synaptik.model.graph;

import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * One independently scoped captured scan body and its ordered outer-role bindings.
 * Numeric IDs are meaningful only inside {@code graph}; they may coincide with outer graph IDs.
 * Graph inputs preserve every declared formal position, including unused ones. Graph outputs are
 * unique, while role bindings may repeat a result without merging distinct outer ports.
 * This record validates structure only, not descriptor inference, AD, demand analysis or execution.
 *
 * <p>All role/demand lists are immutable defensive snapshots. Inputs are declared in carry, step
 * and invariant order; callers preserve that mapping, while the containing {@link CompiledNode}
 * checks cardinalities against its scalar header. Actual outer bindings may repeat, but local
 * formals remain distinct graph inputs. A length-aware outer node has one extra lengths actual,
 * never a corresponding local formal; ordered roles remain in its ScanAttrs, not this record.
 * Structural acceptance grants no inference, AD or execution capability.
 * No source-to-IR capture is performed here.</p>
 *
 * <p>Demand indices address the concatenation of next-carry, emission and residual outer ports,
 * not numeric local ValueIds. Empty demand is valid and reserves no history allocation. The
 * supplied graph and all ordered lists participate in record equality, hashing and diagnostic
 * text; that text is not a serialization or execution format.</p>
 *
 * @param graph non-null existing graph representation, with no nested general scan
 * @param nextCarry non-null ordered non-null local graph-output bindings for next carry, possibly empty
 * @param emissions non-null ordered non-null local graph-output bindings for emissions; together
 *     with nextCarry these contain at least one semantic root
 * @param residuals non-null ordered non-null local graph-output bindings for latent residuals
 * @param demandedOutputIndices non-null strictly increasing non-null zero-based selected indices
 *     into concatenated outer roles, possibly empty
 */
public record CompiledScanBody(CompiledGraphModel graph, List<ValueId> nextCarry,
        List<ValueId> emissions, List<ValueId> residuals, List<Integer> demandedOutputIndices) {
    /**
     * Snapshots and validates local role/demand structure without comparing IDs across scopes.
     * The graph-output set must equal the distinct union of the ordered role bindings.
     *
     * @param graph non-null immutable local body graph
     * @param nextCarry non-null ordered next-carry bindings, possibly empty, repetitions allowed
     * @param emissions non-null ordered emission bindings, repetitions allowed; nextCarry and
     *     emissions cannot both be empty
     * @param residuals non-null ordered residual bindings, repetitions allowed
     * @param demandedOutputIndices non-null increasing unique outer-port indices, possibly empty
     * @throws NullPointerException if a component or list element is null
     * @throws IllegalArgumentException if roles do not cover graph outputs, a demand index is
     *     invalid/out of order, semantic roots are empty, or a node contains a nested general scan
     * @throws ArithmeticException if total role cardinality overflows an int
     */
    public CompiledScanBody {
        Objects.requireNonNull(graph, "graph");
        nextCarry = List.copyOf(nextCarry);
        emissions = List.copyOf(emissions);
        residuals = List.copyOf(residuals);
        demandedOutputIndices = List.copyOf(demandedOutputIndices);
        if (nextCarry.isEmpty() && emissions.isEmpty()) {
            throw new IllegalArgumentException("scan semantic roots must not be empty");
        }
        Set<ValueId> bound = new HashSet<>(nextCarry);
        bound.addAll(emissions);
        bound.addAll(residuals);
        if (!bound.equals(new HashSet<>(graph.outputs()))) {
            throw new IllegalArgumentException("role bindings must cover exactly local graph outputs");
        }
        int count = Math.addExact(Math.addExact(nextCarry.size(), emissions.size()), residuals.size());
        int previous = -1;
        for (int index : demandedOutputIndices) {
            if (index <= previous || index >= count) {
                throw new IllegalArgumentException("demand indices must be increasing valid outer ports");
            }
            previous = index;
        }
        for (CompiledNode node : graph.nodes()) {
            if (node.scanBody().isPresent() || node.operation().kind() == ScanKind.SCAN) {
                throw new IllegalArgumentException("nested compiled scans are not supported");
            }
        }
    }
}

/**
 * Defines immutable backend-neutral compile-time partition recipes.
 *
 * <p>A {@link io.github.pho001.synaptik.planning.partition.PlannedPartition} is the current public
 * owner-plus-node-ID recipe. Package-private generation validates a complete node-to-owner
 * assignment and groups maximal equal-owner runs over the immutable compiled graph's stored
 * topological node order:</p>
 *
 * <pre>{@code
 * CompiledGraphModel.nodes() + complete Map<NodeId, BackendId>
 *     -> internal consecutive same-owner grouping
 *     -> immutable List<PlannedPartition>
 * }</pre>
 *
 * <p>Only consecutive graph-order occurrences are adjacent for this step. Graph connectivity,
 * phase changes, graph inputs and outputs, fan-out, merges, and multi-output values do not replace
 * that rule or split one operation occurrence. The package provides no public orchestration
 * facade, ownership row, partition boundary or transfer model, logical memory plan, cost profile,
 * diagnostics schema, device or route selection, lowering, preparation, runtime state, or
 * execution behavior.</p>
 */
package io.github.pho001.synaptik.planning.partition;

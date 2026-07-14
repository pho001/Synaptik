/**
 * Defines immutable backend-neutral logical materialization and memory requirements.
 *
 * <p>A {@link io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement} retains one
 * graph value's exact identity and logical descriptor together with its optional producing
 * partition, ordered distinct consuming partitions, and graph-output obligation. A
 * {@link io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan} collects those requirements
 * in graph-value order. Package-private derivation validates complete, graph-ordered, maximal
 * partition recipes before constructing the plan:</p>
 *
 * <pre>{@code
 * CompiledGraphModel + ordered complete PlannedPartition recipes
 *     -> validate exact coverage, graph order, and maximal owner runs
 *     -> derive producer, distinct consumers, descriptor, and graph-output facts
 *     -> immutable LogicalMemoryPlan
 * }</pre>
 *
 * <p>The stored facts define overlapping roles relative to a partition. A graph input has no
 * producer. A value is an input of partition {@code P} when {@code P} consumes it and its producer
 * is absent or different from {@code P}. A produced value is a partition output when it is a graph
 * output or another partition consumes it. A cross-owner boundary has unequal producer and
 * external-consumer owners. A partition-internal value has a producer, no graph-output
 * obligation, and no consumer outside that producer. The package stores the underlying
 * relationships instead of another role enum.</p>
 *
 * <p>The exact {@code TensorDescriptor} is retained because dynamic and expression dimensions may
 * not have a numeric element count, and logical element width does not choose backend padding or
 * representation. A logical materialization requirement states only that a value must be
 * available to a consuming partition or preserved at the graph-output boundary. This package
 * does not choose aliasing, copying, transfers, bytes, lifetimes, buffers, slots, allocation,
 * publication targets, devices, layouts, routes, kernels, schedules, or runtime residency.</p>
 */
package io.github.pho001.synaptik.planning.memory;

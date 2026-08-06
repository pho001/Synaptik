package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lowers one complete CPU-owned partition into the current single fused execution unit with
 * fully static right-aligned access normalization.
 * Fusion legality is decided before the four exact boundary buffers are declared.
 * This owner returns the original direct access structure; CPU analysis may subsequently derive
 * one route-independent contiguous-copy plan and an adjusted consumer IR without changing graph
 * values or backend-neutral logical memory.
 */
public final class CpuPartitionLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /**
     * Lowers the exact ADD-to-GELU-to-MUL proving topology and rejects every other partition.
     *
     * @param context complete validated CPU partition projection
     * @return one immutable lowering with exactly four materialized boundary values
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if any semantic, broadcast, layout, write-injectivity,
     *     ownership, publication, fan-out, alias, or cross-partition condition is unsupported
     */
    public LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (!context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        if (context.nodes().size() != 3) throw new IllegalArgumentException(
                "CPU 0005A requires exactly three partition nodes");
        var add = context.nodes().get(0);
        var gelu = context.nodes().get(1);
        var mul = context.nodes().get(2);
        if (add.operation().kind() != BinaryArithmeticKind.ADD
                || gelu.operation().kind() != UnaryElementwiseKind.GELU
                || mul.operation().kind() != BinaryArithmeticKind.MUL
                || add.operation().attrs() != NoOperationAttrs.INSTANCE
                || gelu.operation().attrs() != NoOperationAttrs.INSTANCE
                || mul.operation().attrs() != NoOperationAttrs.INSTANCE
                || add.inputs().size() != 2 || add.outputs().size() != 1
                || gelu.inputs().size() != 1 || gelu.outputs().size() != 1
                || mul.inputs().size() != 2 || mul.outputs().size() != 1
                || !gelu.inputs().getFirst().equals(add.outputs().getFirst())
                || !mul.inputs().getFirst().equals(gelu.outputs().getFirst())) {
            throw new IllegalArgumentException("partition is not the exact ADD -> GELU -> MUL topology");
        }
        ValueId a = add.inputs().get(0);
        ValueId b = add.inputs().get(1);
        ValueId sum = add.outputs().getFirst();
        ValueId activated = gelu.outputs().getFirst();
        ValueId c = mul.inputs().get(1);
        ValueId output = mul.outputs().getFirst();
        if (java.util.Set.of(a, b, sum, activated, c, output).size() != 6) {
            throw new IllegalArgumentException("proving topology values must not alias");
        }
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(requirement -> memory.put(requirement.valueId(), requirement));
        GraphValue first = require(values, a);
        for (ValueId id : List.of(a, b, sum, activated, c, output)) {
            GraphValue value = require(values, id);
            var descriptor = value.descriptor();
            if (descriptor.dataType() != DataType.FLOAT64 || !descriptor.shape().isFullyStatic()
                    || descriptor.layout().isEmpty()) {
                throw new IllegalArgumentException("all proving values must be static resolved FLOAT64");
            }
        }
        Shape addShape = ShapeBroadcast.broadcast(first.descriptor().shape(),
                require(values, b).descriptor().shape());
        if (!require(values, sum).descriptor().shape().equals(addShape)
                || !require(values, activated).descriptor().shape().equals(addShape)) {
            throw new IllegalArgumentException("ADD result and GELU shape must match exactly");
        }
        Shape shape = ShapeBroadcast.broadcast(addShape, require(values, c).descriptor().shape());
        if (!require(values, output).descriptor().shape().equals(shape)) {
            throw new IllegalArgumentException("MUL output must equal its exact broadcast result");
        }
        assertOccurrence(add.operation(), List.of(require(values, a), require(values, b)),
                List.of(require(values, sum)));
        assertOccurrence(gelu.operation(), List.of(require(values, sum)),
                List.of(require(values, activated)));
        assertOccurrence(mul.operation(), List.of(require(values, activated), require(values, c)),
                List.of(require(values, output)));
        requireVirtual(memory.get(sum), context.partition(), "sum");
        requireVirtual(memory.get(activated), context.partition(), "activated");
        long elementCount = shape.knownElementCount().orElseThrow();
        Math.multiplyExact(elementCount, DataType.FLOAT64.byteWidth());
        var aAccess = normalize(first.descriptor().shape(), first.descriptor().layout().orElseThrow(),
                shape, CpuAccessPlan.AccessKind.READ);
        var bValue = require(values, b);
        var bAccess = normalize(bValue.descriptor().shape(), bValue.descriptor().layout().orElseThrow(),
                shape, CpuAccessPlan.AccessKind.READ);
        var cValue = require(values, c);
        var cAccess = normalize(cValue.descriptor().shape(), cValue.descriptor().layout().orElseThrow(),
                shape, CpuAccessPlan.AccessKind.READ);
        var outputValue = require(values, output);
        var outputAccess = normalize(outputValue.descriptor().shape(),
                outputValue.descriptor().layout().orElseThrow(), shape,
                CpuAccessPlan.AccessKind.WRITE);
        var virtualAccess = normalize(addShape, LayoutDescriptor.contiguous(addShape), shape,
                CpuAccessPlan.AccessKind.READ);
        var ir = new CpuKernelIr(
                List.of(new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, aAccess.plan()),
                        new CpuKernelIr.Value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, bAccess.plan()),
                        new CpuKernelIr.Value(2, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, cAccess.plan()),
                        new CpuKernelIr.Value(3, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, virtualAccess.plan()),
                        new CpuKernelIr.Value(4, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, virtualAccess.plan()),
                        new CpuKernelIr.Value(5, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, outputAccess.plan())),
                List.of(new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.ADD,
                                List.of(0, 1), 3),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.GELU_EXACT,
                                List.of(3), 4),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.MUL,
                                List.of(4, 2), 5)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(5, 0)));
        return new LoweredPartition(ir, List.of(a, b, c, output),
                List.of(aAccess.binding(), bAccess.binding(), cAccess.binding(), outputAccess.binding()),
                List.of(first.descriptor().layout().orElseThrow().referencedElementSpan(),
                        bValue.descriptor().layout().orElseThrow().referencedElementSpan(),
                        cValue.descriptor().layout().orElseThrow().referencedElementSpan(),
                        outputValue.descriptor().layout().orElseThrow().referencedElementSpan()), sum, activated,
                shape.toLongArray(), elementCount,
                "legal: exact ordered pointwise chain with single-use private intermediates");
    }

    private static Normalized normalize(Shape source, LayoutDescriptor layout, Shape iteration,
            CpuAccessPlan.AccessKind kind) {
        long[] extents = iteration.toLongArray();
        long[] sourceExtents = source.toLongArray();
        long[] sourceStrides = layout.strides();
        long[] strides = new long[extents.length];
        int offset = extents.length - sourceExtents.length;
        for (int axis = 0; axis < extents.length; axis++) {
            if (axis < offset) strides[axis] = 0;
            else {
                long sourceExtent = sourceExtents[axis - offset];
                long targetExtent = extents[axis];
                if (sourceExtent != targetExtent && sourceExtent != 1) {
                    throw new IllegalArgumentException("shape does not right-broadcast to unit iteration shape");
                }
                strides[axis] = sourceExtent == 1 && targetExtent != 1
                        ? 0 : sourceStrides[axis - offset];
            }
        }
        if (kind == CpuAccessPlan.AccessKind.WRITE) validateDistinctWrites(extents, strides);
        int suffix = 0;
        long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        var roles = new java.util.ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        boolean allZero = java.util.Arrays.stream(strides).allMatch(value -> value == 0);
        boolean dense = suffix == extents.length;
        boolean bias = extents.length > 0 && suffix == 1
                && roles.subList(0, extents.length - 1).stream()
                        .allMatch(role -> role == CpuAccessPlan.AxisRole.BROADCAST);
        CpuAccessPlan.Regime regime = allZero && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : dense ? CpuAccessPlan.Regime.DENSE_LINEAR
                : bias ? CpuAccessPlan.Regime.LAST_AXIS_BIAS
                : suffix > 0 ? CpuAccessPlan.Regime.BLOCK_OUTER
                : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        var plan = new CpuAccessPlan(kind, regime, extents.length, roles, suffix);
        long count = iteration.knownElementCount().orElseThrow();
        return new Normalized(plan, CpuAccessPlan.Binding.create(plan, extents,
                layout.storageOffset(), strides, count, 0, count, layout.referencedElementSpan()));
    }

    private static void validateDistinctWrites(long[] extents, long[] strides) {
        if (java.util.Arrays.stream(extents).anyMatch(extent -> extent == 0)) return;
        var axes = new java.util.ArrayList<WriteAxis>();
        for (int axis = 0; axis < extents.length; axis++) {
            if (extents[axis] > 1 && strides[axis] == 0) throw new IllegalArgumentException(
                    "output geometry repeats a write address");
            if (extents[axis] > 1) axes.add(new WriteAxis(extents[axis] - 1, strides[axis]));
        }
        axes.sort(java.util.Comparator.comparingLong((WriteAxis axis) -> axis.stride()).reversed());
        long[] remaining = new long[axes.size() + 1];
        for (int index = axes.size() - 1; index >= 0; index--) remaining[index] = Math.addExact(
                remaining[index + 1], Math.multiplyExact(axes.get(index).bound(),
                        axes.get(index).stride()));
        if (hasNonZeroCollision(axes, remaining, 0, 0, false)) {
            throw new IllegalArgumentException("output geometry repeats a write address");
        }
    }

    private static boolean hasNonZeroCollision(List<WriteAxis> axes, long[] remaining,
            int index, long sum, boolean nonZero) {
        if (index == axes.size()) return nonZero && sum == 0;
        WriteAxis axis = axes.get(index);
        long tail = remaining[index + 1];
        long minimum = Math.max(-axis.bound(), ceilDiv(Math.subtractExact(-tail, sum),
                axis.stride()));
        long maximum = Math.min(axis.bound(), Math.floorDiv(Math.subtractExact(tail, sum),
                axis.stride()));
        for (long delta = minimum; delta <= maximum; delta++) {
            long next = Math.addExact(sum, Math.multiplyExact(delta, axis.stride()));
            if (hasNonZeroCollision(axes, remaining, index + 1, next,
                    nonZero || delta != 0)) return true;
            if (delta == Long.MAX_VALUE) break;
        }
        return false;
    }

    private static long ceilDiv(long dividend, long divisor) {
        return -Math.floorDiv(-dividend, divisor);
    }

    private record WriteAxis(long bound, long stride) { }

    private record Normalized(CpuAccessPlan plan, CpuAccessPlan.Binding binding) { }

    private void assertOccurrence(io.github.pho001.synaptik.model.operation.Operation operation,
            List<GraphValue> inputs, List<GraphValue> outputs) {
        if (!capabilities.supports(new OperationCapabilityQuery(operation,
                inputs.stream().map(GraphValue::descriptor).toList(),
                outputs.stream().map(GraphValue::descriptor).toList()))) {
            throw new IllegalArgumentException("partition contains an unsupported CPU occurrence");
        }
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static void requireVirtual(LogicalMemoryRequirement requirement,
            io.github.pho001.synaptik.planning.partition.PlannedPartition partition, String name) {
        if (requirement == null || requirement.graphOutput()
                || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(partition)
                || !requirement.consumerPartitions().equals(List.of(partition))) {
            throw new IllegalArgumentException(name
                    + " must be private, unpublished, and single-partition consumed");
        }
    }

    /**
     * Immutable result consumed by route-neutral preparation.
     *
     * @param kernelIr non-null route-independent canonical unit IR
     * @param boundaryValues non-null ordered materialized values {@code a}, {@code b}, {@code c},
     *     and output; copied defensively
     * @param accessBindings non-null ordered normalized cold bindings for the four boundaries;
     *     copied defensively
     * @param referencedElementSpans non-null ordered exact layout spans used for declarations;
     *     copied defensively
     * @param sum non-null virtual ADD result
     * @param activated non-null virtual exact-GELU result
     * @param extents non-null compatible static extents; copied defensively
     * @param elementCount checked product of {@code extents}
     * @param fusionReason non-null cold diagnostic explanation
     */
    public record LoweredPartition(CpuKernelIr kernelIr, List<ValueId> boundaryValues,
            List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
            ValueId sum, ValueId activated, long[] extents, long elementCount,
            String fusionReason) {
        /**
         * Validates and snapshots the completed unit lowering.
         *
         * @throws NullPointerException if a required component is {@code null}
         */
        public LoweredPartition {
            Objects.requireNonNull(kernelIr, "kernelIr");
            boundaryValues = List.copyOf(boundaryValues);
            accessBindings = List.copyOf(accessBindings);
            referencedElementSpans = List.copyOf(referencedElementSpans);
            if (boundaryValues.size() != 4 || accessBindings.size() != 4
                    || referencedElementSpans.size() != 4) throw new IllegalArgumentException(
                            "lowering must retain four ordered boundary bindings");
            Objects.requireNonNull(sum, "sum");
            Objects.requireNonNull(activated, "activated");
            extents = extents.clone();
            Objects.requireNonNull(fusionReason, "fusionReason");
        }
        /** Returns instance geometry.
         * @return a new defensive copy of compatible extents */
        @Override public long[] extents() { return extents.clone(); }
    }
}

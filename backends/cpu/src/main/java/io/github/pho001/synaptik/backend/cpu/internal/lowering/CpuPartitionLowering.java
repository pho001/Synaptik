package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lowers one complete CPU-owned partition into the single legal CPU-0005A fused execution unit.
 * Fusion legality is decided before the four exact boundary buffers are declared.
 */
public final class CpuPartitionLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /**
     * Lowers the exact ADD-to-GELU-to-MUL proving topology and rejects every other partition.
     *
     * @param context complete validated CPU partition projection
     * @return one immutable lowering with exactly four materialized boundary values
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if any semantic, geometry, ownership, publication,
     *     fan-out, alias, or cross-partition condition is outside task 0005A
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
        var shape = first.descriptor().shape();
        for (ValueId id : List.of(a, b, sum, activated, c, output)) {
            GraphValue value = require(values, id);
            var descriptor = value.descriptor();
            if (descriptor.dataType() != DataType.FLOAT64 || !descriptor.shape().equals(shape)
                    || !descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty()
                    || descriptor.layout().orElseThrow().kind()
                            != io.github.pho001.synaptik.model.layout.LayoutKind.DENSE_CONTIGUOUS
                    || descriptor.layout().orElseThrow().storageOffset() != 0
                    || descriptor.layout().orElseThrow().isView()) {
                throw new IllegalArgumentException(
                        "all proving values must be static canonical-dense zero-offset FLOAT64");
            }
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
        var dense = new CpuAccessPlan(CpuAccessPlan.Regime.CANONICAL_DENSE);
        var ir = new CpuKernelIr(
                List.of(new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, dense),
                        new CpuKernelIr.Value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, dense),
                        new CpuKernelIr.Value(2, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, dense),
                        new CpuKernelIr.Value(3, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, dense),
                        new CpuKernelIr.Value(4, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL, dense),
                        new CpuKernelIr.Value(5, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, dense)),
                List.of(new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.ADD,
                                List.of(0, 1), 3),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.GELU_EXACT,
                                List.of(3), 4),
                        new CpuKernelIr.Instruction(CpuKernelIr.Instruction.Semantic.MUL,
                                List.of(4, 2), 5)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(5, 0)));
        return new LoweredPartition(ir, List.of(a, b, c, output), sum, activated,
                shape.toLongArray(), elementCount,
                "legal: exact ordered pointwise chain with single-use private intermediates");
    }

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
     * @param sum non-null virtual ADD result
     * @param activated non-null virtual exact-GELU result
     * @param extents non-null compatible static extents; copied defensively
     * @param elementCount checked product of {@code extents}
     * @param fusionReason non-null cold diagnostic explanation
     */
    public record LoweredPartition(CpuKernelIr kernelIr, List<ValueId> boundaryValues,
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

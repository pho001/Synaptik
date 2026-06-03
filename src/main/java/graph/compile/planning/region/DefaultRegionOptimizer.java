package graph.compile.planning.region;

import graph.CompiledNode;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.region.specialization.DefaultRegionSpecializationCapability;
import graph.compile.planning.region.specialization.RegionSpecializationCapability;
import graph.compile.planning.region.specialization.RegionSpecializationPlanner;
import graph.compile.planning.region.specialization.RegionSpecializationResult;
import graph.compile.planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Default partition-to-region optimizer.
 *
 * <p>The optimizer builds structural execution units, then derives region values and their transport kinds from
 * partition outputs, required materialization, and unit-to-unit value flow. Backend-specific lowering families and
 * capability decisions belong to backend lowerers.
 */
public final class DefaultRegionOptimizer {
    private final RegionSpecializationCapability specializationCapability;

    public DefaultRegionOptimizer() {
        this(new DefaultRegionSpecializationCapability());
    }

    public DefaultRegionOptimizer(RegionSpecializationCapability specializationCapability) {
        this.specializationCapability = Objects.requireNonNull(
                specializationCapability,
                "specializationCapability cannot be null"
        );
    }

    /**
     * Converts a partition to an optimized region.
     *
     * @param partition accepted partition
     * @param context compiled node and fusion context
     * @return optimized region with execution units and value transport metadata
     */
    public OptimizedRegion optimize(Partition partition, RegionOptimizationContext context) {
        if (partition == null) {
            throw new IllegalArgumentException("partition cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }

        UnitBuildResult unitBuild = buildUnits(partition, context);
        List<ExecutionUnit> units = unitBuild.units();
        List<RegionValue> regionValues = partition.values().stream()
                .map(value -> toRegionValue(value, partition, context, units))
                .toList();

        ArrayList<String> traceEvents = new ArrayList<>();
        traceEvents.add("units=" + units.size());
        traceEvents.add("target=" + partition.target().name());
        traceEvents.add("regionKind=" + partition.regionKind().name());
        traceEvents.add("plannerStrategy=" + partition.plannerStrategy().name());
        traceEvents.addAll(unitBuild.traceEvents());
        RegionOptimizationTrace trace = new RegionOptimizationTrace(traceEvents);

        return new OptimizedRegion(
                partition.partitionId(),
                partition,
                partition.target(),
                units,
                regionValues,
                partition.requiredMaterializedValueRefs(),
                trace
        );
    }

    private UnitBuildResult buildUnits(Partition partition, RegionOptimizationContext context) {
        RegionSpecializationResult specialization = RegionSpecializationPlanner.tryBuildUnits(
                partition,
                context,
                specializationCapability
        );
        if (specialization.accepted()) {
            return new UnitBuildResult(specialization.units(), specialization.traceEvents());
        }
        List<ExecutionUnit> units;
        if (partition.target() == PartitionTarget.CPU) {
            units = new CpuRegionOptimizationPolicy().buildUnits(partition, context);
        } else {
            units = StructuralRegionUnitPlanner.buildUnits(partition, context);
        }
        return new UnitBuildResult(units, specialization.traceEvents());
    }

    private RegionValue toRegionValue(
            PartitionValue value,
            Partition partition,
            RegionOptimizationContext context,
            List<ExecutionUnit> units
    ) {
        GraphValueRef ref = value.ref();
        ValueTransportKind transportKind = partition.requiredMaterializedValueRefs().contains(ref)
                ? ValueTransportKind.MATERIALIZED
                : (isExecutionUnitContinuation(ref, units) || partition.outputValueRefs().contains(ref)
                    ? ValueTransportKind.CONTINUATION
                    : ValueTransportKind.VIRTUAL);
        return new RegionValue(
                ref,
                ref,
                value.producerNodeId(),
                contextElementCount(context, value.producerNodeId()),
                transportKind,
                contextTypeContract(context, value.producerNodeId()),
                partition.requiredMaterializedValueRefs().contains(ref)
        );
    }

    private boolean isExecutionUnitContinuation(GraphValueRef ref, List<ExecutionUnit> units) {
        GraphValueRef valueRef = ref;
        int producerUnits = 0;
        int consumerUnits = 0;
        for (ExecutionUnit unit : units) {
            if (unit.outputValueRefs().contains(valueRef)) {
                producerUnits++;
            }
            if (unit.inputValueRefs().contains(valueRef)) {
                consumerUnits++;
            }
        }
        return producerUnits > 0 && consumerUnits > 0;
    }

    private ValueTypeContract contextTypeContract(RegionOptimizationContext context, int producerNodeId) {
        // Initial version keeps all dtypes identical; later phases may widen compute/transport.
        CompiledNode producer = context.compiledNode(producerNodeId);
        if (producer == null) {
            throw new IllegalStateException("Missing compiled node for producerNodeId=" + producerNodeId);
        }
        return ValueTypeContract.same(producer.dataType());
    }

    private int contextElementCount(RegionOptimizationContext context, int producerNodeId) {
        CompiledNode producer = context.compiledNode(producerNodeId);
        if (producer == null) {
            throw new IllegalStateException("Missing compiled node for producerNodeId=" + producerNodeId);
        }
        return Math.max(0, producer.flatDataSize());
    }

    private record UnitBuildResult(
            List<ExecutionUnit> units,
            List<String> traceEvents
    ) {
        private UnitBuildResult {
            units = List.copyOf(units == null ? List.of() : units);
            traceEvents = List.copyOf(traceEvents == null ? List.of() : traceEvents);
        }
    }

}

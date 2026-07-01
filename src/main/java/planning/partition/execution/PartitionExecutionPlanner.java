package planning.partition.execution;

import graph.model.CompiledNode;
import planning.partition.Partition;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.partition.specialization.DefaultPartitionSpecializationCapability;
import planning.partition.specialization.PartitionSpecializationCapability;
import planning.partition.specialization.PartitionSpecializationPlanner;
import planning.partition.specialization.PartitionSpecializationResult;
import planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Default partition execution planner.
 *
 * <p>The planner builds structural execution units, then derives execution values and their transport kinds from
 * partition outputs, required materialization, and unit-to-unit value flow. Backend-specific lowering families and
 * capability decisions belong to backend lowerers.
 */
public final class PartitionExecutionPlanner {
    private final PartitionSpecializationCapability specializationCapability;

    public PartitionExecutionPlanner() {
        this(new DefaultPartitionSpecializationCapability());
    }

    public PartitionExecutionPlanner(PartitionSpecializationCapability specializationCapability) {
        this.specializationCapability = Objects.requireNonNull(
                specializationCapability,
                "specializationCapability cannot be null"
        );
    }

    /**
     * Builds the execution details owned by a partition.
     *
     * @param partition accepted partition
     * @param context compiled node and fusion context
     * @return execution plan with units and value transport metadata
     */
    public PartitionExecutionPlan planPartition(Partition partition, PartitionExecutionPlanningContext context) {
        if (partition == null) {
            throw new IllegalArgumentException("partition cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }

        UnitBuildResult unitBuild = buildUnits(partition, context);
        List<ExecutionUnit> units = unitBuild.units();
        List<PartitionExecutionValue> executionValues = partition.values().stream()
                .map(value -> toPartitionExecutionValue(value, partition, context, units))
                .toList();

        ArrayList<String> traceEvents = new ArrayList<>();
        traceEvents.add("units=" + units.size());
        traceEvents.add("target=" + partition.target().name());
        traceEvents.add("partitionKind=" + partition.partitionKind().name());
        traceEvents.add("plannerStrategy=" + partition.plannerStrategy().name());
        traceEvents.addAll(unitBuild.traceEvents());
        PartitionExecutionTrace trace = new PartitionExecutionTrace(traceEvents);

        return new PartitionExecutionPlan(
                units,
                executionValues,
                partition.requiredMaterializedValueRefs(),
                trace
        );
    }

    private UnitBuildResult buildUnits(Partition partition, PartitionExecutionPlanningContext context) {
        PartitionSpecializationResult specialization = PartitionSpecializationPlanner.tryBuildUnits(
                partition,
                context,
                specializationCapability
        );
        if (specialization.accepted()) {
            return new UnitBuildResult(specialization.units(), specialization.traceEvents());
        }
        List<ExecutionUnit> units;
        if (partition.target() == PartitionTarget.CPU) {
            units = new CpuPartitionExecutionPlanningPolicy().buildUnits(partition, context);
        } else {
            units = StructuralPartitionExecutionUnitPlanner.buildUnits(partition, context);
        }
        return new UnitBuildResult(units, specialization.traceEvents());
    }

    private PartitionExecutionValue toPartitionExecutionValue(
            PartitionValue value,
            Partition partition,
            PartitionExecutionPlanningContext context,
            List<ExecutionUnit> units
    ) {
        GraphValueRef ref = value.ref();
        ValueTransportKind transportKind = partition.requiredMaterializedValueRefs().contains(ref)
                ? ValueTransportKind.MATERIALIZED
                : (isExecutionUnitContinuation(ref, units) || partition.outputValueRefs().contains(ref)
                    ? ValueTransportKind.CONTINUATION
                    : ValueTransportKind.VIRTUAL);
        return new PartitionExecutionValue(
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

    private ValueTypeContract contextTypeContract(PartitionExecutionPlanningContext context, int producerNodeId) {
        // Initial version keeps all dtypes identical; later phases may widen compute/transport.
        CompiledNode producer = context.compiledNode(producerNodeId);
        if (producer == null) {
            throw new IllegalStateException("Missing compiled node for producerNodeId=" + producerNodeId);
        }
        return ValueTypeContract.same(producer.dataType());
    }

    private int contextElementCount(PartitionExecutionPlanningContext context, int producerNodeId) {
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

package graph.optimizer.region;

import graph.CompiledNode;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.GraphValueRef;

import java.util.List;

/**
 * Default partition-to-region optimizer.
 *
 * <p>The optimizer builds structural execution units, then derives region values and their transport kinds from
 * partition outputs, required materialization, and unit-to-unit value flow. Backend-specific lowering families and
 * capability decisions belong to backend lowerers.
 */
public final class DefaultRegionOptimizer {
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

        List<ExecutionUnit> units = buildUnits(partition, context);
        List<RegionValue> regionValues = partition.values().stream()
                .map(value -> toRegionValue(value, partition, context, units))
                .toList();

        RegionOptimizationTrace trace = new RegionOptimizationTrace(
                List.of(
                        "units=" + units.size(),
                        "target=" + partition.target().name(),
                        "regionKind=" + partition.regionKind().name(),
                        "plannerStrategy=" + partition.plannerStrategy().name()
                )
        );

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

    private List<ExecutionUnit> buildUnits(Partition partition, RegionOptimizationContext context) {
        if (partition.target() == PartitionTarget.CPU) {
            return new CpuRegionOptimizationPolicy().buildUnits(partition, context);
        }
        return StructuralRegionUnitPlanner.buildUnits(partition, context);
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
                contextSemanticTensor(context, value.producerNodeId()),
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

    private tensor.Tensor contextSemanticTensor(RegionOptimizationContext context, int producerNodeId) {
        CompiledNode producer = context.compiledNode(producerNodeId);
        if (producer == null) {
            throw new IllegalStateException("Missing compiled node for producerNodeId=" + producerNodeId);
        }
        return producer.semanticTensor();
    }
}

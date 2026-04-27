package graph.optimizer.region;

import graph.CompiledNode;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;

import java.util.List;

public final class DefaultRegionOptimizer implements RegionOptimizer {
    private final RegionOptimizationPolicy cpuPolicy;
    private final RegionOptimizationPolicy genericPolicy;

    public DefaultRegionOptimizer() {
        this(new CpuRegionOptimizationPolicy(), new GenericGpuRegionOptimizationPolicy());
    }

    public DefaultRegionOptimizer(
            RegionOptimizationPolicy cpuPolicy,
            RegionOptimizationPolicy genericPolicy
    ) {
        this.cpuPolicy = cpuPolicy == null ? new CpuRegionOptimizationPolicy() : cpuPolicy;
        this.genericPolicy = genericPolicy == null ? new GenericGpuRegionOptimizationPolicy() : genericPolicy;
    }

    @Override
    public OptimizedRegion optimize(Partition partition, RegionOptimizationContext context) {
        if (partition == null) {
            throw new IllegalArgumentException("partition cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }

        List<ExecutionUnit> units = policyFor(partition.target()).buildUnits(partition, context);
        List<RegionValue> regionValues = partition.values().stream()
                .map(value -> toRegionValue(value, partition, context, units))
                .toList();

        RegionOptimizationTrace trace = new RegionOptimizationTrace(
                List.of("units=" + units.size(), "target=" + partition.target().name())
        );

        return new OptimizedRegion(
                partition.partitionId(),
                partition,
                partition.target(),
                units,
                regionValues,
                partition.requiredMaterializedValueRefs().stream().map(RegionOptimizationUnitSupport::toRegionValueRef).toList(),
                trace
        );
    }

    private RegionOptimizationPolicy policyFor(PartitionTarget target) {
        return target == PartitionTarget.CPU ? cpuPolicy : genericPolicy;
    }

    private RegionValue toRegionValue(
            PartitionValue value,
            Partition partition,
            RegionOptimizationContext context,
            List<ExecutionUnit> units
    ) {
        PartitionValueRef ref = value.ref();
        ValueTransportKind transportKind = partition.requiredMaterializedValueRefs().contains(ref)
                ? ValueTransportKind.MATERIALIZED
                : (isExecutionUnitContinuation(ref, units) || partition.outputValueRefs().contains(ref)
                    ? ValueTransportKind.CONTINUATION
                    : ValueTransportKind.VIRTUAL);
        return new RegionValue(
                RegionOptimizationUnitSupport.toRegionValueRef(ref),
                ref,
                contextSemanticTensor(context, value.producerNodeId()),
                value.producerNodeId(),
                contextElementCount(context, value.producerNodeId()),
                transportKind,
                contextTypeContract(context, value.producerNodeId()),
                partition.requiredMaterializedValueRefs().contains(ref)
        );
    }

    private boolean isExecutionUnitContinuation(PartitionValueRef ref, List<ExecutionUnit> units) {
        RegionValueRef valueRef = RegionOptimizationUnitSupport.toRegionValueRef(ref);
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

package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

public class CpuOrderingLoweringTest {
    @Test void lowersAllFamiliesWithSliceRangesAndExactScratch() {
        var sort = lower(context(new Operation(OrderingKind.SORT, new SortAttrs(1, false)),
                DataType.FLOAT64, Shape.of(2, 4), Shape.of(2, 4), false));
        var top = lower(context(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, false)),
                DataType.FLOAT32, Shape.of(3, 5), Shape.of(3, 2), true));
        assertAll(() -> assertEquals(CpuOrderingIr.Family.SORT,
                        ((CpuOrderingIr) sort.portableKernelIr()).family()),
                () -> assertEquals(2, sort.elementCount()),
                () -> assertEquals(64, sort.orderingGeometry().orElseThrow().scratchSliceBytes()),
                () -> assertEquals(3, top.boundaryValues().size()),
                () -> assertEquals(3, top.elementCount()),
                () -> assertEquals(80, top.orderingGeometry().orElseThrow().scratchSliceBytes()));
    }

    @Test void supportsSixTypesAndRejectsWrongOutputContracts() {
        for (DataType type : DataType.values()) assertDoesNotThrow(() -> lower(context(
                new Operation(OrderingKind.SORT, new SortAttrs(0, false)), type,
                Shape.of(3), Shape.of(3), false)));
        assertThrows(IllegalArgumentException.class, () -> lower(context(
                new Operation(TopKKind.TOP_K, new TopKAttrs(0, 2, true, true)),
                DataType.INT32, Shape.of(3), Shape.of(3), true)));
    }

    @Test void emptyAxisEmptyOuterAndZeroKSelectNoWorkWithExactScratch() {
        var emptyAxis = lower(context(new Operation(OrderingKind.SORT, new SortAttrs(1, false)),
                DataType.INT64, Shape.of(3, 0), Shape.of(3, 0), false));
        var emptyOuter = lower(context(new Operation(OrderingKind.ARGSORT, new SortAttrs(1, false)),
                DataType.BOOL, Shape.of(0, 4), Shape.of(0, 4), false));
        var zeroK = lower(context(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 0, true, false)),
                DataType.FLOAT64, Shape.of(3, 4), Shape.of(3, 0), true));
        assertAll(() -> assertEquals(0, emptyAxis.elementCount()),
                () -> assertEquals(0, emptyAxis.orderingGeometry().orElseThrow().scratchSliceBytes()),
                () -> assertEquals(0, emptyOuter.elementCount()),
                () -> assertEquals(0, zeroK.elementCount()),
                () -> assertEquals(0, zeroK.orderingGeometry().orElseThrow().scratchSliceBytes()));
    }

    private static CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionLowering().lower(context);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Operation operation,
            DataType type, Shape inputShape, Shape outputShape, boolean topK) {
        TensorDescriptor input = descriptor(type, inputShape);
        var outputs = new ArrayList<TensorDescriptor>();
        outputs.add(descriptor(topK ? type : operation.kind() == OrderingKind.ARGSORT
                ? DataType.INT64 : type, outputShape));
        if (topK) outputs.add(descriptor(DataType.INT64, outputShape));
        ValueId inputId = new ValueId(0);
        List<ValueId> outputIds = java.util.stream.IntStream.range(0, outputs.size())
                .mapToObj(i -> new ValueId(i + 1)).toList();
        var node = new CompiledNode(new NodeId(0), operation, List.of(inputId), outputIds);
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var values = new ArrayList<GraphValue>(); values.add(new GraphValue(inputId, input));
        var memory = new ArrayList<LogicalMemoryRequirement>();
        memory.add(new LogicalMemoryRequirement(inputId, input, Optional.empty(), List.of(partition), false));
        for (int i = 0; i < outputs.size(); i++) {
            values.add(new GraphValue(outputIds.get(i), outputs.get(i)));
            memory.add(new LogicalMemoryRequirement(outputIds.get(i), outputs.get(i),
                    Optional.of(partition), List.of(), true));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}

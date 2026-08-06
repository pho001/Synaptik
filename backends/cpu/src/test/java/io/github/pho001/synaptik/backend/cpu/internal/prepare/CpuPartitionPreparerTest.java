package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;

public class CpuPartitionPreparerTest {
    @Test void formsOneFusedUnitAndDeclaresOnlyFourBoundaries() {
        var analysis = analyze(Shape.of(2, 3));
        assertAll(
                () -> assertEquals(1, analysis.plan().units().size()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2),
                                new ValueId(5)),
                        analysis.plan().boundaryValues()),
                () -> assertEquals(4, analysis.requirements().size()),
                () -> assertFalse(analysis.plan().boundaryValues().contains(new ValueId(3))),
                () -> assertFalse(analysis.plan().boundaryValues().contains(new ValueId(4))),
                () -> assertEquals("scalar", analysis.plan().executionStrategy().toString()));
    }

    @Test void failsClosedForPublishedIntermediate() {
        var context = context(Shape.of(2));
        var memory = new ArrayList<>(context.memoryRequirements());
        var old = memory.get(3);
        memory.set(3, new LogicalMemoryRequirement(old.valueId(), old.descriptor(),
                old.producerPartition(), old.consumerPartitions(), true));
        var changed = new PrepareContext<>(context.partition(), context.nodes(), context.values(),
                memory, Map.of(), context.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(changed));
    }

    @Test void analysisInputsSnapshotPatternAndValidateCountBeforeLoweringConsumption() {
        var mutable = new ArrayList<>(List.of(CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT));
        var inputs = new CpuPartitionAnalysisInputs(true, mutable);
        mutable.set(0, CarrierAccess.MEMORY_SEGMENT);
        assertAll(
                () -> assertFalse(CpuPartitionAnalysisInputs.DEFAULT.loweringManifestEnabled()),
                () -> assertEquals(List.of(CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT, CarrierAccess.MEMORY_SEGMENT,
                        CarrierAccess.MEMORY_SEGMENT), CpuPartitionAnalysisInputs.DEFAULT.carrierPattern()),
                () -> assertEquals(CarrierAccess.DOUBLE_ARRAY, inputs.carrierPattern().getFirst()),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuPartitionAnalysisInputs(false, null)),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuPartitionAnalysisInputs(false,
                                java.util.Arrays.asList(CarrierAccess.DOUBLE_ARRAY, null))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionPreparer().analyze(new PrepareContext<>(
                                context(Shape.of(2)).partition(), context(Shape.of(2)).nodes(),
                                context(Shape.of(2)).values(), context(Shape.of(2)).memoryRequirements(),
                                Map.of(), new CpuPartitionAnalysisInputs(false,
                                        List.of(CarrierAccess.MEMORY_SEGMENT))))));
    }

    public static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(Shape shape) {
        return new CpuPartitionPreparer().analyze(context(shape));
    }

    public static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            TensorDescriptor a, TensorDescriptor b, TensorDescriptor c, TensorDescriptor output,
            CpuPartitionAnalysisInputs inputs) {
        return new CpuPartitionPreparer().analyze(context(a, b, c, output, inputs));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Shape shape) {
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return context(descriptor, descriptor, descriptor, descriptor,
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor cDescriptor, TensorDescriptor outputDescriptor,
            CpuPartitionAnalysisInputs inputs) {
        ValueId a = new ValueId(0), b = new ValueId(1), c = new ValueId(2);
        ValueId sum = new ValueId(3), activated = new ValueId(4), output = new ValueId(5);
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(a, b), List.of(sum)),
                new CompiledNode(new NodeId(1),
                        new Operation(UnaryElementwiseKind.GELU, NoOperationAttrs.INSTANCE),
                        List.of(sum), List.of(activated)),
                new CompiledNode(new NodeId(2),
                        new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                        List.of(activated, c), List.of(output)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        Shape addShape = ShapeBroadcast.broadcast(aDescriptor.shape(), bDescriptor.shape());
        var virtualDescriptor = new TensorDescriptor(DataType.FLOAT64, addShape,
                Optional.of(LayoutDescriptor.contiguous(addShape)), false);
        var descriptors = List.of(aDescriptor, bDescriptor, cDescriptor, virtualDescriptor,
                virtualDescriptor, outputDescriptor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 6; i++) {
            ValueId id = new ValueId(i);
            var descriptor = descriptors.get(i);
            values.add(new GraphValue(id, descriptor));
            boolean produced = i >= 3;
            boolean consumed = i != 5;
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), i == 5));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                inputs);
    }
}

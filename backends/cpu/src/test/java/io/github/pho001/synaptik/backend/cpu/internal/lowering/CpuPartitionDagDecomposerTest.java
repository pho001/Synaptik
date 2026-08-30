package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class CpuPartitionDagDecomposerTest {
    @Test void liveValuesAreBornAndDieAtTheirActualInstructionAndStoreEvents() {
        assertAll(
                () -> assertEquals(2, CpuPartitionDagDecomposer.liveMaximum(lateInputs(16))),
                () -> assertEquals(2, CpuPartitionDagDecomposer.liveMaximum(repeatedOperand())),
                () -> assertEquals(16,
                        CpuPartitionDagDecomposer.liveMaximum(pendingStores(15))),
                () -> assertEquals(17,
                        CpuPartitionDagDecomposer.liveMaximum(pendingStores(16))));
    }

    @Test void boundaryIndexingAndGeneratedSizeGatesAcceptCeilingsAndRejectNextUnit() {
        assertAll(
                () -> assertTrue(CpuPartitionDagDecomposer.withinBudgets(boundaryIr(false))),
                () -> assertFalse(CpuPartitionDagDecomposer.withinBudgets(boundaryIr(true))),
                () -> assertTrue(CpuPartitionDagDecomposer.withinBudgets(indexingIr(false))),
                () -> assertFalse(CpuPartitionDagDecomposer.withinBudgets(indexingIr(true))),
                () -> assertTrue(CpuPartitionDagDecomposer.withinBudgets(codeIr(false))),
                () -> assertFalse(CpuPartitionDagDecomposer.withinBudgets(codeIr(true))));
    }

    @Test void oneAndEightNodeAndUnitEdgesAreStableAndTheImmediateExcessFailsClosed() {
        var one = new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(1)).plan();
        var fusedEight = new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(8)).plan();
        var splitEight = new CpuPartitionPreparer().analyze(publishedChain(8)).plan();
        assertAll(
                () -> assertEquals(1, one.units().size()),
                () -> assertEquals(8, fusedEight.units().size()),
                () -> assertEquals(1, fusedEight.fusionDecisions().stream()
                        .filter(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision
                                .Selection.class::isInstance)
                        .map(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision
                                .Selection.class::cast)
                        .findFirst().orElseThrow().compatibilityBaseline().units().size()),
                () -> assertEquals(8, splitEight.units().size()),
                () -> assertEquals(28, CpuPartitionDagDecomposer.MAX_ATTEMPTS),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        new CpuPartitionPreparer().analyze(
                                CpuPointwisePartitionLoweringTest.chain(9))));
    }

    @Test void diamondMaterializesTheFanOutAndFusesOnlyTheConsumerSiblings() {
        var plan = new CpuPartitionPreparer().analyze(diamond()).plan();
        assertAll(
                () -> assertEquals(2, plan.units().size()),
                () -> assertEquals(List.of(0), plan.units().get(1).dependencies()),
                () -> assertEquals(List.of(1, 2),
                        plan.units().get(1).memberNodeOrdinals()),
                () -> assertEquals(2,
                        plan.units().get(1).portablePlan().kernelIr().stores().size()),
                () -> assertEquals(1,
                        plan.boundaryValues().stream().filter(new ValueId(1)::equals).count()));
    }

    @Test void sevenWayFanOutIsMaterializedAndTheEighthConsumerCannotEnterThePartition() {
        var atCeiling = new CpuPartitionPreparer().analyze(fanOut(7)).plan();
        assertAll(
                () -> assertTrue(atCeiling.units().size() >= 2),
                () -> assertTrue(atCeiling.units().size() <= 8),
                () -> assertTrue(atCeiling.boundaryValues().contains(new ValueId(1))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionPreparer().analyze(fanOut(8))));
    }

    @Test void publicationAffineNumericalAndStateEdgesRemainMaterializedBarriers() {
        var publication = new CpuPartitionPreparer().analyze(publishedChain(2)).plan();
        var affine = new CpuPartitionPreparer().analyze(affineThenPointwise()).plan();
        var numerical = new CpuPartitionPreparer().analyze(aggregateThenPointwise()).plan();
        var pool3d = new CpuPartitionPreparer().analyze(pool3dThenPointwise()).plan();
        var state = new CpuPartitionPreparer().analyze(dropoutThenPointwise()).plan();
        assertAll(
                () -> assertEquals(2, publication.units().size()),
                () -> assertEquals(2, affine.units().size()),
                () -> assertEquals(2, numerical.units().size()),
                () -> assertEquals(2, pool3d.units().size()),
                () -> assertEquals(2, state.units().size()),
                () -> assertEquals(List.of(0), affine.units().get(1).dependencies()),
                () -> assertEquals(List.of(0), numerical.units().get(1).dependencies()),
                () -> assertEquals(List.of(0), pool3d.units().get(1).dependencies()),
                () -> assertEquals(List.of(0), state.units().get(1).dependencies()));
    }

    @Test void repeatedAnalysisRetainsExactUnitMemberDependencyAndBoundaryOrder() {
        var context = fanOut(7);
        var expected = snapshot(new CpuPartitionPreparer().analyze(context).plan());
        for (int repetition = 0; repetition < 12; repetition++) {
            assertEquals(expected,
                    snapshot(new CpuPartitionPreparer().analyze(context).plan()));
        }
    }

    @Test void repeatedPortsAndMultiOutputProducerPositionsSurviveUnitProjection() {
        Shape shape = Shape.of(2, 3, 4);
        Shape vector = Shape.of(3);
        var layouts = List.of(LayoutDescriptor.contiguous(shape),
                LayoutDescriptor.contiguous(vector), LayoutDescriptor.contiguous(vector),
                LayoutDescriptor.contiguous(vector), LayoutDescriptor.contiguous(vector),
                LayoutDescriptor.contiguous(shape),
                LayoutDescriptor.contiguous(vector), LayoutDescriptor.contiguous(vector),
                LayoutDescriptor.contiguous(vector), LayoutDescriptor.contiguous(vector));
        var context = CpuBatchNormTrainingLoweringTest.context(
                java.util.Collections.nCopies(5, DataType.FLOAT32), shape, 1,
                List.of(0, 1, 1, 2, 3), layouts);
        var dag = context.partitionDag();
        var plan = new CpuPartitionPreparer().analyze(context).plan();

        assertAll(
                () -> assertEquals(List.of(1, 2), dag.consumers(new ValueId(1)).stream()
                        .map(occurrence -> occurrence.inputPosition()).toList()),
                () -> assertEquals(List.of(0, 1, 2, 3, 4),
                        java.util.stream.IntStream.range(4, 9)
                                .mapToObj(index -> dag.producer(new ValueId(index)).orElseThrow()
                                        .outputPosition()).toList()),
                () -> assertEquals(List.of(0), plan.units().getFirst().memberNodeOrdinals()),
                () -> assertTrue(plan.units().getFirst().dependencies().isEmpty()));
    }

    @Test void nonzeroBatchNormOutputFeedsLaterUnitWithExactProjectionAndDependency() {
        Shape shape = Shape.of(2, 3, 4);
        var context = batchNormTrainingOutputOneConsumer(shape);
        var value = context.nodes().getFirst().outputs().get(1);
        var producer = context.partitionDag().producer(value).orElseThrow();
        var consumer = context.partitionDag().consumers(value).getFirst();
        var requirement = context.memoryRequirements().stream()
                .filter(candidate -> candidate.valueId().equals(value)).findFirst().orElseThrow();
        var decomposer = new CpuPartitionDagDecomposer();
        var producerOnly = decomposer.unitContext(context, List.of(producer.node()),
                CpuPartitionAnalysisInputs.DEFAULT);
        var producerOnlyOccurrence = producerOnly.partitionDag().producer(value).orElseThrow();
        var producerOnlyRequirement = producerOnly.memoryRequirements().stream()
                .filter(candidate -> candidate.valueId().equals(value)).findFirst().orElseThrow();
        var consumerOnly = decomposer.unitContext(context, List.of(consumer.node()),
                CpuPartitionAnalysisInputs.DEFAULT);
        var consumerOnlyInput = consumerOnly.partitionDag().externalInputs().stream()
                .filter(occurrence -> occurrence.valueId().equals(value)).findFirst().orElseThrow();
        var consumerOnlyRequirement = consumerOnly.memoryRequirements().stream()
                .filter(candidate -> candidate.valueId().equals(value)).findFirst().orElseThrow();
        var plan = new CpuPartitionPreparer().analyze(context).plan();

        assertAll(
                () -> assertEquals(1, producer.outputPosition()),
                () -> assertEquals(0, producer.nodePosition()),
                () -> assertEquals(List.of(1), context.partitionDag().consumers(value).stream()
                        .map(occurrence -> occurrence.nodePosition()).toList()),
                () -> assertEquals(List.of(0), context.partitionDag().consumers(value).stream()
                        .map(occurrence -> occurrence.inputPosition()).toList()),
                () -> assertFalse(requirement.graphOutput()),
                () -> assertEquals(List.of(context.partition()),
                        requirement.consumerPartitions()),
                () -> assertSame(producer.node(), producerOnlyOccurrence.node()),
                () -> assertEquals(1, producerOnlyOccurrence.outputPosition()),
                () -> assertTrue(producerOnly.partitionDag().consumers(value).isEmpty()),
                () -> assertEquals(Optional.of(producerOnly.partition()),
                        producerOnlyRequirement.producerPartition()),
                () -> assertTrue(producerOnlyRequirement.consumerPartitions().isEmpty()),
                () -> assertTrue(producerOnlyRequirement.graphOutput()),
                () -> assertTrue(consumerOnly.partitionDag().producer(value).isEmpty()),
                () -> assertEquals(1, consumerOnly.partitionDag().externalInputs().size()),
                () -> assertSame(consumer.node(), consumerOnlyInput.node()),
                () -> assertEquals(value, consumerOnlyInput.valueId()),
                () -> assertEquals(0, consumerOnlyInput.nodePosition()),
                () -> assertEquals(0, consumerOnlyInput.inputPosition()),
                () -> assertTrue(consumerOnlyRequirement.producerPartition().isEmpty()),
                () -> assertEquals(List.of(consumerOnly.partition()),
                        consumerOnlyRequirement.consumerPartitions()),
                () -> assertFalse(consumerOnlyRequirement.graphOutput()),
                () -> assertEquals(List.of(List.of(0), List.of(1)), plan.units().stream()
                        .map(unit -> unit.memberNodeOrdinals()).toList()),
                () -> assertEquals(List.of(), plan.units().get(0).dependencies()),
                () -> assertEquals(List.of(0), plan.units().get(1).dependencies()));
    }

    @Test void malformedLaterNodeFailsBeforeAnyPreparationRequirementExists() {
        var base = publishedChain(2);
        var values = new ArrayList<>(base.values());
        values.removeLast();
        assertThrows(IllegalArgumentException.class, () -> new CpuPartitionPreparer().analyze(
                new PrepareContext<>(base.partition(), base.nodes(), values,
                        base.memoryRequirements(), Map.of(), base.backendInputs())));
    }

    private static String snapshot(
            io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan) {
        return plan.units().stream().map(unit -> unit.memberNodeOrdinals() + ":"
                + unit.dependencies() + ":" + unit.boundaryValues() + ":"
                + unit.portablePlan().kernelIr().structuralKey()).toList().toString()
                + "|" + plan.boundaryValues();
    }

    private static CpuKernelIr lateInputs(int count) {
        var values = new ArrayList<CpuKernelIr.Value>();
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        for (int index = 0; index < count; index++) {
            values.add(value(index, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT,
                    CpuAccessPlan.Regime.DENSE_LINEAR));
        }
        for (int index = 0; index < count; index++) {
            int output = count + index;
            values.add(value(output, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL,
                    CpuAccessPlan.Regime.DENSE_LINEAR));
            instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                    List.of(index), output));
        }
        return ir(values, instructions, List.of());
    }

    private static CpuKernelIr repeatedOperand() {
        return ir(List.of(
                value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT,
                        CpuAccessPlan.Regime.DENSE_LINEAR),
                value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL,
                        CpuAccessPlan.Regime.DENSE_LINEAR)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD,
                        List.of(0, 0), 1)), List.of());
    }

    private static CpuKernelIr pendingStores(int count) {
        var values = new ArrayList<CpuKernelIr.Value>();
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        var stores = new ArrayList<CpuKernelIr.Store>();
        for (int index = 0; index < count; index++) {
            values.add(value(index, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT,
                    CpuAccessPlan.Regime.DENSE_LINEAR));
        }
        for (int index = 0; index < count; index++) {
            int output = count + index;
            values.add(value(output, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT,
                    CpuAccessPlan.Regime.DENSE_LINEAR));
            instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                    List.of(index), output));
            stores.add(new CpuKernelIr.Store(output, index));
        }
        return ir(values, instructions, stores);
    }

    private static CpuKernelIr boundaryIr(boolean exceed) {
        int inputs = exceed ? 12 : 11;
        var values = new ArrayList<CpuKernelIr.Value>();
        Set<Integer> bools = Set.of(0, 3, 6);
        for (int index = 0; index < inputs; index++) values.add(value(index,
                bools.contains(index) ? DataType.BOOL : DataType.FLOAT64,
                CpuKernelIr.Value.Kind.INPUT, CpuAccessPlan.Regime.DENSE_LINEAR));
        for (int index = 0; index < 5; index++) values.add(value(inputs + index,
                DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT,
                CpuAccessPlan.Regime.DENSE_LINEAR));
        var instructions = List.of(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE, List.of(0, 1, 2), inputs),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE, List.of(3, 4, 5), inputs + 1),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE, List.of(6, 7, 8), inputs + 2),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.ADD, List.of(9, 10), inputs + 3),
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                        List.of(exceed ? 11 : 1), inputs + 4));
        var stores = java.util.stream.IntStream.range(0, 5)
                .mapToObj(index -> new CpuKernelIr.Store(inputs + index, index)).toList();
        return ir(values, instructions, stores);
    }

    private static CpuKernelIr indexingIr(boolean exceed) {
        int valueCount = exceed ? 9 : 8;
        var values = new ArrayList<CpuKernelIr.Value>();
        Set<Integer> bools = Set.of(0, 3);
        for (int index = 0; index < valueCount; index++) {
            boolean output = index >= 6;
            CpuAccessPlan.Regime regime = exceed && index >= 6
                    ? CpuAccessPlan.Regime.BLOCK_OUTER
                    : CpuAccessPlan.Regime.GENERAL_ODOMETER;
            values.add(value(index, bools.contains(index) ? DataType.BOOL : DataType.FLOAT64,
                    output ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.INPUT, regime));
        }
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                List.of(0, 1, 2), 6));
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.WHERE,
                List.of(3, 4, 5), 7));
        if (exceed) instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                List.of(1), 8));
        var stores = java.util.stream.IntStream.range(6, valueCount)
                .mapToObj(value -> new CpuKernelIr.Store(value, value - 6)).toList();
        return ir(values, instructions, stores);
    }

    private static CpuKernelIr codeIr(boolean exceed) {
        var values = new ArrayList<CpuKernelIr.Value>();
        values.add(value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT,
                CpuAccessPlan.Regime.GENERAL_ODOMETER));
        for (int ordinal = 1; ordinal <= 5; ordinal++) values.add(value(ordinal,
                DataType.FLOAT64, CpuKernelIr.Value.Kind.VIRTUAL,
                CpuAccessPlan.Regime.GENERAL_ODOMETER));
        values.add(value(6, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT,
                exceed ? CpuAccessPlan.Regime.BLOCK_OUTER : CpuAccessPlan.Regime.LAST_AXIS_BIAS));
        values.add(value(7, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT,
                CpuAccessPlan.Regime.LAST_AXIS_BIAS));
        values.add(value(8, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT,
                CpuAccessPlan.Regime.LAST_AXIS_BIAS));
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        for (int ordinal = 1; ordinal <= 5; ordinal++) instructions.add(
                new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG,
                        List.of(ordinal - 1), ordinal));
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(5), 6));
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(5), 7));
        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(5), 8));
        return ir(values, instructions, List.of(new CpuKernelIr.Store(6, 0),
                new CpuKernelIr.Store(7, 1), new CpuKernelIr.Store(8, 2)));
    }

    private static CpuKernelIr.Value value(int ordinal, DataType type,
            CpuKernelIr.Value.Kind kind, CpuAccessPlan.Regime regime) {
        CpuAccessPlan.AccessKind access = kind == CpuKernelIr.Value.Kind.OUTPUT
                ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ;
        CpuAccessPlan.AxisRole role = regime == CpuAccessPlan.Regime.DENSE_LINEAR
                || regime == CpuAccessPlan.Regime.LAST_AXIS_BIAS
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED;
        int suffix = role == CpuAccessPlan.AxisRole.CONTIGUOUS ? 1 : 0;
        return new CpuKernelIr.Value(ordinal, type, kind,
                new CpuAccessPlan(access, regime, 1, List.of(role), suffix));
    }

    private static CpuKernelIr ir(List<CpuKernelIr.Value> values,
            List<CpuKernelIr.Instruction> instructions, List<CpuKernelIr.Store> stores) {
        return new CpuKernelIr(values, instructions, new CpuKernelIr.Loop("start", "end"), stores);
    }

    static PrepareContext<CpuPartitionAnalysisInputs> publishedChain(int count) {
        var nodes = new ArrayList<CompiledNode>();
        for (int index = 0; index < count; index++) nodes.add(scalar(index,
                new ValueId(index), new ValueId(index + 1L), ScalarElementwiseKind.ADD));
        var descriptors = java.util.stream.IntStream.rangeClosed(0, count)
                .mapToObj(ignored -> descriptor(DataType.FLOAT32, Shape.of(4))).toList();
        return context(nodes, descriptors,
                java.util.stream.LongStream.rangeClosed(1, count).mapToObj(ValueId::new)
                        .collect(java.util.stream.Collectors.toSet()));
    }

    static PrepareContext<CpuPartitionAnalysisInputs> diamond() {
        var nodes = List.of(
                scalar(0, new ValueId(0), new ValueId(1), ScalarElementwiseKind.ADD),
                scalar(1, new ValueId(1), new ValueId(2), ScalarElementwiseKind.MUL),
                scalar(2, new ValueId(1), new ValueId(3), ScalarElementwiseKind.SUB));
        var descriptors = java.util.stream.IntStream.range(0, 4)
                .mapToObj(ignored -> descriptor(DataType.FLOAT32, Shape.of(4))).toList();
        return context(nodes, descriptors, Set.of(new ValueId(2)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> fanOut(int consumers) {
        var nodes = new ArrayList<CompiledNode>();
        nodes.add(scalar(0, new ValueId(0), new ValueId(1), ScalarElementwiseKind.ADD));
        var outputs = new HashSet<ValueId>();
        for (int index = 0; index < consumers; index++) {
            ValueId output = new ValueId(index + 2L);
            nodes.add(scalar(index + 1, new ValueId(1), output, ScalarElementwiseKind.MUL));
            outputs.add(output);
        }
        var descriptors = java.util.stream.IntStream.range(0, consumers + 2)
                .mapToObj(ignored -> descriptor(DataType.FLOAT32, Shape.of(4))).toList();
        return context(nodes, descriptors, outputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> affineThenPointwise() {
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                        List.of(new ValueId(0)), List.of(new ValueId(1))),
                scalar(1, new ValueId(1), new ValueId(2), ScalarElementwiseKind.ADD));
        var descriptors = java.util.stream.IntStream.range(0, 3)
                .mapToObj(ignored -> descriptor(DataType.FLOAT32, Shape.of(4))).toList();
        return context(nodes, descriptors, Set.of(new ValueId(2)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> aggregateThenPointwise() {
        var input = descriptor(DataType.FLOAT32, Shape.of(2, 3));
        var reduced = descriptor(DataType.FLOAT32, Shape.of(2));
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(AggregateReductionKind.SUM,
                        new AxisReductionAttrs(1, false)), List.of(new ValueId(0)),
                        List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1), new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE), List.of(new ValueId(1)),
                        List.of(new ValueId(2))));
        return context(nodes, List.of(input, reduced, reduced), Set.of(new ValueId(2)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> pool3dThenPointwise() {
        var input=descriptor(DataType.FLOAT32,Shape.of(1,1,3,3,3));
        var output=descriptor(DataType.FLOAT32,Shape.of(1,1,2,2,2));
        var nodes=List.of(new CompiledNode(new NodeId(0),new Operation(
                        io.github.pho001.synaptik.model.operation.pooling.Pool3dKind.MAX_POOL3D,
                        new io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs(
                                2,2,2,1,1,1,0,0,0,1,1,1,false)),List.of(new ValueId(0)),
                        List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1),new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE),List.of(new ValueId(1)),List.of(new ValueId(2))));
        return context(nodes,List.of(input,output,output),Set.of(new ValueId(2)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> dropoutThenPointwise() {
        Shape shape = Shape.of(4);
        Shape stateShape = Shape.of(2);
        var value = descriptor(DataType.FLOAT32, shape);
        var state = descriptor(DataType.INT64, stateShape);
        var mask = descriptor(DataType.BOOL, shape);
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(DropoutKind.DROPOUT,
                        new DropoutAttrs(.25)), List.of(new ValueId(0), new ValueId(1)),
                        List.of(new ValueId(2), new ValueId(3), new ValueId(4))),
                new CompiledNode(new NodeId(1), new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE), List.of(new ValueId(2)),
                        List.of(new ValueId(5))));
        return context(nodes, List.of(value, state, value, mask, state, value),
                Set.of(new ValueId(3), new ValueId(4), new ValueId(5)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> batchNormTrainingOutputOneConsumer(
            Shape shape) {
        var base = CpuBatchNormTrainingLoweringTest.context(shape, 1);
        CompiledNode batchNorm = base.nodes().getFirst();
        ValueId consumedOutput = batchNorm.outputs().get(1);
        ValueId result = new ValueId(10);
        CompiledNode consumer = new CompiledNode(new NodeId(1),
                new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                List.of(consumedOutput), List.of(result));
        var nodes = List.of(batchNorm, consumer);
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<>(base.values());
        TensorDescriptor resultDescriptor = values.stream()
                .filter(value -> value.id().equals(consumedOutput)).findFirst().orElseThrow()
                .descriptor();
        values.add(new GraphValue(result, resultDescriptor));
        var memory = values.stream().map(value -> {
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(value.id()));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(value.id()));
            boolean published = produced && !consumed;
            return new LogicalMemoryRequirement(value.id(), value.descriptor(),
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), published);
        }).toList();
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static CompiledNode scalar(int node, ValueId input, ValueId output,
            ScalarElementwiseKind kind) {
        return new CompiledNode(new NodeId(node), new Operation(kind,
                new ScalarValueAttrs(ScalarValue.float32(2))), List.of(input), List.of(output));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(List<CompiledNode> nodes,
            List<TensorDescriptor> descriptors, Set<ValueId> graphOutputs) {
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = descriptors.get(index);
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), graphOutputs.contains(id)));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }
}

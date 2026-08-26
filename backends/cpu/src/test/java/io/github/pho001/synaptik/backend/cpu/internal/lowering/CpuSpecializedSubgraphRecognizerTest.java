package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuSpecializedSubgraphRecognizerTest {
    @Test void retainsFloatingReductionFactWithoutChangingBaselineIr() {
        var context = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM,
                DataType.FLOAT32, Shape.of(2, 3), new AxisReductionAttrs(1, false), Shape.of(2));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        String key = plan.units().getFirst().portablePlan().portableKernelIr().structuralKey();
        List<CpuSpecializedSubgraph> facts = new CpuSpecializedSubgraphRecognizer()
                .recognize(context, plan.units());
        assertAll(() -> assertEquals(1, facts.size()),
                () -> assertInstanceOf(ReductionEpilogue.class, facts.getFirst()),
                () -> assertEquals(ExecutionDisposition.ORDINARY_SPLIT,
                        facts.getFirst().disposition()),
                () -> assertEquals(List.of(0), facts.getFirst().baselineUnitIndices()),
                () -> assertEquals(facts, plan.specializedSubgraphs()),
                () -> assertEquals(key, plan.units().getFirst().portablePlan()
                        .portableKernelIr().structuralKey()));
    }

    @Test void excludesTargetShapeAndRecognizesOnlyFirstClassSemanticKinds() {
        var targetShape = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM,
                DataType.FLOAT32, Shape.of(2, 3), new SumToShapeAttrs(Shape.of(1, 3)),
                Shape.of(1, 3));
        assertTrue(new CpuPartitionPreparer().analyze(targetShape).plan()
                .specializedSubgraphs().isEmpty());

        List<PrepareContext<CpuPartitionAnalysisInputs>> contexts = List.of(
                CpuSoftmaxLoweringTest.context(SoftmaxKind.SOFTMAX, DataType.FLOAT32,
                        Shape.of(2, 3), 1),
                CpuSoftmaxLoweringTest.context(SoftmaxKind.LOG_SOFTMAX, DataType.FLOAT64,
                        Shape.of(2, 3), 1),
                CpuTrailingNormalizationLoweringTest.context(true, false, DataType.FLOAT32,
                        Shape.of(2, 3), Shape.of(3), List.of(0)),
                CpuTrailingNormalizationLoweringTest.context(false, true, DataType.FLOAT32,
                        Shape.of(2, 3), Shape.of(3), List.of(0, 1)),
                CpuBatchNormInferenceLoweringTest.context(Collections.nCopies(5, DataType.FLOAT32),
                        Shape.of(2, 3, 4), 1, List.of(0, 1, 2, 3, 4)),
                CpuBatchNormTrainingLoweringTest.context(Shape.of(2, 3), 1));
        List<Form> expected = List.of(Form.SOFTMAX, Form.LOG_SOFTMAX, Form.LAYER_NORM,
                Form.RMS_NORM, Form.BATCH_NORM_INFERENCE, Form.BATCH_NORM_TRAINING);
        for (int index = 0; index < contexts.size(); index++) {
            var current = contexts.get(index);
            var plan = new CpuPartitionPreparer().analyze(current).plan();
            assertEquals(1, plan.specializedSubgraphs().size());
            var fact = assertInstanceOf(ExplicitSemanticKernel.class,
                    plan.specializedSubgraphs().getFirst());
            assertEquals(expected.get(index), fact.form());
            assertEquals(List.of(0), fact.memberNodeOrdinals());
            assertEquals(List.of(0), fact.baselineUnitIndices());
            if (fact.form() == Form.BATCH_NORM_TRAINING) {
                assertAll(() -> assertEquals(10, fact.accessFacts().size()),
                        () -> assertEquals(5, fact.inputDataTypes().size()),
                        () -> assertEquals(5, fact.resultDataTypes().size()),
                        () -> assertEquals(10, current.values().stream()
                                .map(value -> value.id()).distinct().count()));
            }
        }
    }

    @Test void closedFamilyAndParentInvariantsProveTheImpossibleNextCeilings() {
        List<Integer> closedBoundaryInventory = List.of(
                2 + 1, // MATMUL
                3 + 1, // convolution
                1 + 1, // reduction
                1 + 1, // softmax and log-softmax
                3 + 1, // Layer Norm
                2 + 1, // RMS Norm
                5 + 1, // batch-normalization inference
                5 + 5); // batch-normalization training
        assertAll(() -> assertEquals(24, CpuSpecializedSubgraphRecognizer.MAX_ATTEMPTS),
                () -> assertEquals(8, CpuSpecializedSubgraphRecognizer.MAX_FACTS),
                () -> assertEquals(6, CpuSpecializedSubgraphRecognizer.MAX_MEMBERS),
                () -> assertEquals(10, CpuSpecializedSubgraphRecognizer.MAX_BOUNDARIES),
                () -> assertEquals(2, CpuSpecializedSubgraphRecognizer.MAX_UNITS),
                () -> assertEquals(8, CpuPartitionDagDecomposer.MAX_NODES),
                () -> assertEquals(6, 4 + 2),
                () -> assertEquals(List.of(3, 4, 2, 2, 4, 3, 6, 10),
                        closedBoundaryInventory),
                () -> assertEquals(10, closedBoundaryInventory
                        .stream().mapToInt(Integer::intValue).max().orElseThrow()));
    }

    @Test void eightOneNodeFactsReachTheParentCeilingAndARealTwoUnitSuffixReachesItsCeiling() {
        var eight = new CpuPartitionPreparer().analyze(independentSoftmax(8)).plan();
        var twoUnits = new CpuPartitionPreparer().analyze(reductionAddRelu()).plan();
        var reduction = assertInstanceOf(ReductionEpilogue.class,
                twoUnits.specializedSubgraphs().getFirst());
        assertAll(() -> assertEquals(8, eight.units().size()),
                () -> assertEquals(8, eight.specializedSubgraphs().size()),
                () -> assertEquals(8, eight.specializedSubgraphs().stream()
                        .flatMap(fact -> fact.memberNodeOrdinals().stream()).distinct().count()),
                () -> assertEquals(2, twoUnits.units().size()),
                () -> assertEquals(List.of(0, 1), reduction.baselineUnitIndices()),
                () -> assertEquals(2,
                        reduction.structuralIdentity().baselineUnits().size()),
                () -> assertEquals(twoUnits.units().getFirst().portablePlan().specialization(),
                        reduction.structuralIdentity().baselineUnits().getFirst()
                                .execution().specialization()),
                () -> assertEquals(twoUnits.units().get(1).portablePlan().specialization(),
                        reduction.structuralIdentity().baselineUnits().get(1)
                                .execution().specialization()),
                () -> assertEquals(RuntimeTopology.AGGREGATE,
                        reduction.structuralIdentity().baselineUnits().getFirst()
                                .execution().runtimeTopology()),
                () -> assertEquals(RuntimeTopology.POINTWISE,
                        reduction.structuralIdentity().baselineUnits().get(1)
                                .execution().runtimeTopology()),
                () -> assertEquals(2, reduction.epilogue().operationCount()),
                () -> assertEquals(List.of(0),
                        twoUnits.units().getFirst().memberNodeOrdinals()),
                () -> assertEquals(List.of(1, 2),
                        twoUnits.units().get(1).memberNodeOrdinals()),
                () -> assertEquals(List.of(0), twoUnits.units().get(1).dependencies()));
    }

    @Test void twentyFourthAttemptRetainsPriorFactsAndTheNextAttemptLeavesWorkOrdinary() {
        var plan = new CpuPartitionPreparer().analyze(softmaxThenUnrecognizedPointwise(4)).plan();
        assertAll(() -> assertEquals(5, plan.units().stream()
                        .flatMap(unit -> unit.memberNodeOrdinals().stream()).count()),
                () -> assertEquals(1, plan.specializedSubgraphs().size()),
                () -> assertEquals(Form.SOFTMAX,
                        ((ExplicitSemanticKernel) plan.specializedSubgraphs().getFirst()).form()),
                () -> assertEquals(List.of(0),
                        plan.specializedSubgraphs().getFirst().memberNodeOrdinals()));
    }

    @Test void conv1dAddReluReachesSixMembersAndALiteralThirdSuffixIsNotRecognized() {
        var six = new CpuPartitionPreparer().analyze(
                CpuConv1dCompositionLoweringTest.contextWithSuffix(false)).plan();
        var fact = assertInstanceOf(ConvolutionEpilogue.class,
                six.specializedSubgraphs().getFirst());
        var third = new CpuPartitionPreparer().analyze(
                CpuConv1dCompositionLoweringTest.contextWithSuffix(true)).plan();
        assertAll(() -> assertEquals(Form.CONV1D_COMPOSITION, fact.form()),
                () -> assertEquals(List.of(0, 1, 2, 3, 4, 5),
                        fact.memberNodeOrdinals()),
                () -> assertEquals(6, fact.memberNodeOrdinals().size()),
                () -> assertEquals(2, fact.epilogue().operationCount()),
                () -> assertEquals(List.of(0, 1), fact.baselineUnitIndices()),
                () -> assertTrue(third.specializedSubgraphs().stream()
                        .noneMatch(candidate -> candidate instanceof ConvolutionEpilogue)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> independentSoftmax(int count) {
        var nodes = new ArrayList<CompiledNode>();
        var descriptors = new ArrayList<TensorDescriptor>();
        var outputs = new java.util.HashSet<ValueId>();
        Shape shape = Shape.of(2, 3);
        for (int index = 0; index < count; index++) {
            ValueId input = new ValueId(index * 2L);
            ValueId output = new ValueId(index * 2L + 1);
            nodes.add(new CompiledNode(new NodeId(index), new Operation(SoftmaxKind.SOFTMAX,
                    new SoftmaxAttrs(1)), List.of(input), List.of(output)));
            descriptors.add(descriptor(shape));
            descriptors.add(descriptor(shape));
            outputs.add(output);
        }
        return context(nodes, descriptors, outputs);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> reductionAddRelu() {
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(AggregateReductionKind.SUM,
                        new AxisReductionAttrs(1, false)), List.of(new ValueId(0)),
                        List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1), new Operation(
                        io.github.pho001.synaptik.model.operation.elementwise.binary
                                .BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(new ValueId(1), new ValueId(2)), List.of(new ValueId(3))),
                new CompiledNode(new NodeId(2), new Operation(UnaryElementwiseKind.RELU,
                        NoOperationAttrs.INSTANCE), List.of(new ValueId(3)),
                        List.of(new ValueId(4))));
        return context(nodes, List.of(descriptor(Shape.of(2, 3)), descriptor(Shape.of(2)),
                descriptor(Shape.of(2)), descriptor(Shape.of(2)), descriptor(Shape.of(2))),
                Set.of(new ValueId(4)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> softmaxThenUnrecognizedPointwise(
            int pointwiseCount) {
        var nodes = new ArrayList<CompiledNode>();
        var descriptors = new ArrayList<TensorDescriptor>();
        var outputs = new java.util.HashSet<ValueId>();
        Shape shape = Shape.of(2, 3);
        nodes.add(new CompiledNode(new NodeId(0), new Operation(SoftmaxKind.SOFTMAX,
                new SoftmaxAttrs(1)), List.of(new ValueId(0)), List.of(new ValueId(1))));
        descriptors.add(descriptor(shape)); descriptors.add(descriptor(shape));
        outputs.add(new ValueId(1));
        for (int index = 0; index < pointwiseCount; index++) {
            long input = 2L + index * 2L;
            long output = input + 1;
            nodes.add(new CompiledNode(new NodeId(index + 1),
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(new ValueId(input)), List.of(new ValueId(output))));
            descriptors.add(descriptor(shape)); descriptors.add(descriptor(shape));
            outputs.add(new ValueId(output));
        }
        return context(nodes, descriptors, outputs);
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
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
            boolean output = graphOutputs.contains(id);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed && !output ? List.of(partition) : List.of(), output));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }
}

package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.reflect.Modifier;
import java.util.*;
import org.junit.jupiter.api.Test;

final class CapturedGraphInferenceTest {
    @Test
    void retainsExactGraphAndRejectsNullFirst() {
        Tensor x = tensor(DataType.FLOAT32, Shape.of(2), true);
        CompiledGraphModel graph = GraphCapture.capture(List.of(x.abs()));

        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);

        assertSame(graph, validated.graph());
        assertTrue(validated.constraints().isEmpty());
        assertEquals("graph", assertThrows(NullPointerException.class,
                () -> CapturedGraphInference.inferAndValidate(null)).getMessage());
    }

    @Test
    void acceptsCapturedPassThroughGraph() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        CompiledGraphModel graph = GraphCapture.capture(List.of(input));

        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);

        assertAll(
                () -> assertSame(graph, validated.graph()),
                () -> assertTrue(validated.constraints().isEmpty()),
                () -> assertTrue(graph.nodes().isEmpty()),
                () -> assertEquals(graph.inputs(), graph.outputs()),
                () -> assertEquals(1, graph.values().size()));
    }

    @Test
    void acceptsCapturedFiveOutputBatchNormalizationTraining() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor scale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor runningMean = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor runningVariance = tensor(DataType.FLOAT32, Shape.of(3), true);
        BatchNormTrainingResult result = input.batchNormTraining(
                1, scale, bias, runningMean, runningVariance,
                ScalarValue.float32(0.1f), ScalarValue.float32(1.0e-5f));
        CompiledGraphModel graph = GraphCapture.capture(List.of(
                result.output(), result.nextRunningMean(), result.nextRunningVariance()));

        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);

        assertAll(
                () -> assertSame(graph, validated.graph()),
                () -> assertTrue(validated.constraints().isEmpty()),
                () -> assertEquals(1, graph.nodes().size()),
                () -> assertEquals(5, graph.nodes().getFirst().outputs().size()),
                () -> assertEquals(3, graph.outputs().size()));
    }

    @Test
    void snapshotsDeferredConstraints() {
        DynamicDimension n = new DynamicDimension("N");
        Tensor x = tensor(DataType.FLOAT32, Shape.ofDimensions(n), false);
        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(GraphCapture.capture(
                List.of(x.reshape(Shape.ofDimensions(new DynamicDimension("M"))))));

        assertEquals(1, validated.constraints().size());
        assertThrows(UnsupportedOperationException.class, () -> validated.constraints().clear());
    }

    @Test
    void retainsEqualLookingConstraintsFromDistinctNodesInNodeOrder() {
        DynamicDimension n = new DynamicDimension("N");
        DynamicDimension m = new DynamicDimension("M");
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.ofDimensions(n), false);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.ofDimensions(m), false);
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output.shape()));
        ValueId inputId = new ValueId(0);
        ValueId firstOutput = new ValueId(1);
        ValueId secondOutput = new ValueId(2);
        NodeId firstNode = new NodeId(0);
        NodeId secondNode = new NodeId(1);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(inputId, input),
                        new GraphValue(firstOutput, output),
                        new GraphValue(secondOutput, output)),
                List.of(
                        new CompiledNode(firstNode, reshape, List.of(inputId), List.of(firstOutput)),
                        new CompiledNode(secondNode, reshape, List.of(inputId), List.of(secondOutput))),
                List.of(inputId),
                List.of(firstOutput, secondOutput),
                Map.of(firstNode, GraphPhase.FORWARD, secondNode, GraphPhase.FORWARD));

        List<DeferredGraphConstraint> constraints =
                CapturedGraphInference.inferAndValidate(graph).constraints();

        assertAll(
                () -> assertEquals(2, constraints.size()),
                () -> assertEquals(List.of(firstNode, secondNode),
                        constraints.stream().map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertEquals(List.of("reshape element count", "reshape element count"),
                        constraints.stream().map(DeferredGraphConstraint::subject).toList()),
                () -> assertEquals(constraints.get(0).predicate(), constraints.get(1).predicate()),
                () -> assertNotEquals(constraints.get(0), constraints.get(1)));
    }

    @Test
    void preservesMultipleConstraintRuleOrderAndReportsFirstDisproof() {
        DynamicDimension m = new DynamicDimension("M");
        DynamicDimension n = new DynamicDimension("N");
        Operation deferredOperation = new Operation(
                AggregateReductionKind.SUM,
                new SumToShapeAttrs(Shape.ofDimensions(m, n)));
        CompiledGraphModel deferredGraph = singleNodeGraph(
                deferredOperation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.ofDimensions(m, n), true)));

        List<DeferredGraphConstraint> constraints =
                CapturedGraphInference.inferAndValidate(deferredGraph).constraints();

        assertEquals(List.of("sum-to axis 0", "sum-to axis 1"),
                constraints.stream().map(DeferredGraphConstraint::subject).toList());

        Operation disprovenOperation = new Operation(
                AggregateReductionKind.SUM, new SumToShapeAttrs(Shape.of(4, 5)));
        CompiledGraphModel disprovenGraph = singleNodeGraph(
                disprovenOperation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(4, 5), true)));
        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(disprovenGraph)).getMessage();

        assertAll(
                () -> assertTrue(message.contains("constraint sum-to axis 0 failed:")),
                () -> assertFalse(message.contains("sum-to axis 1")));

        Operation laterOperation = new Operation(
                AggregateReductionKind.SUM, new SumToShapeAttrs(Shape.of(6, 7)));
        ValueId inputId = new ValueId(0);
        ValueId firstOutput = new ValueId(1);
        ValueId secondOutput = new ValueId(2);
        NodeId firstNode = new NodeId(0);
        NodeId secondNode = new NodeId(1);
        CompiledGraphModel twoFailingNodes = new CompiledGraphModel(
                List.of(
                        new GraphValue(inputId,
                                descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                        new GraphValue(firstOutput,
                                descriptor(DataType.FLOAT32, Shape.of(4, 5), true)),
                        new GraphValue(secondOutput,
                                descriptor(DataType.FLOAT32, Shape.of(6, 7), true))),
                List.of(
                        new CompiledNode(firstNode, disprovenOperation,
                                List.of(inputId), List.of(firstOutput)),
                        new CompiledNode(secondNode, laterOperation,
                                List.of(inputId), List.of(secondOutput))),
                List.of(inputId),
                List.of(firstOutput, secondOutput),
                Map.of(firstNode, GraphPhase.FORWARD, secondNode, GraphPhase.FORWARD));
        String nodeMessage = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(twoFailingNodes)).getMessage();
        assertAll(
                () -> assertTrue(nodeMessage.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertFalse(nodeMessage.contains("nodes[1]")));
    }

    @Test
    void comparesDescriptorsBeforeEvaluatingConstraints() {
        Operation operation = new Operation(
                AggregateReductionKind.SUM, new SumToShapeAttrs(Shape.of(4, 5)));
        CompiledGraphModel graph = singleNodeGraph(
                operation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 3), true)),
                List.of(descriptor(DataType.FLOAT64, Shape.of(4, 5), true)));

        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();

        assertAll(
                () -> assertTrue(message.contains("output[0] ValueId[value=1] expected=")),
                () -> assertFalse(message.contains("constraint sum-to axis")));
    }

    @Test
    void reportsEveryDescriptorMismatchComponentWithCompleteContext() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(2), true);
        Operation operation = new Operation(
                UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE);

        assertDescriptorMismatch(operation, input,
                descriptor(DataType.FLOAT64, Shape.of(2), true), "dataType=FLOAT64");
        assertDescriptorMismatch(operation, input,
                descriptor(DataType.FLOAT32, Shape.of(3), true), "shape=Shape[3]");
        assertDescriptorMismatch(operation, input,
                new TensorDescriptor(DataType.FLOAT32, Shape.of(2),
                        Optional.of(LayoutDescriptor.contiguous(Shape.of(2))), true),
                "layout=Optional[LayoutDescriptor");
        assertDescriptorMismatch(operation, input,
                descriptor(DataType.FLOAT32, Shape.of(2), false), "requiresGrad=false");
    }

    @Test
    void reportsDescriptorMismatchAtSecondMultiOutputPosition() {
        Operation operation = new Operation(
                TopKKind.TOP_K, new TopKAttrs(0, 2, true, true));
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor values = descriptor(DataType.FLOAT32, Shape.of(2), true);
        TensorDescriptor badIndices = descriptor(DataType.BOOL, Shape.of(2), false);
        CompiledGraphModel graph = singleNodeGraph(
                operation, List.of(input), List.of(values, badIndices));

        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();

        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains("output[1] ValueId[value=2] expected=")),
                () -> assertTrue(message.contains("dataType=INT64")),
                () -> assertTrue(message.contains("stored=TensorDescriptor[dataType=BOOL")));
    }

    @Test
    void exposesNoPublicCompilerDeclarations() throws Exception {
        assertFalse(Modifier.isPublic(CapturedGraphInference.class.getModifiers()));
        assertFalse(Modifier.isPublic(ValidatedGraph.class.getModifiers()));
        assertFalse(Modifier.isPublic(DeferredGraphConstraint.class.getModifiers()));
        assertFalse(Modifier.isPublic(CapturedGraphInference.class
                .getDeclaredMethod("inferAndValidate", CompiledGraphModel.class).getModifiers()));
    }

    @Test
    void evaluatesAllThreeProofOutcomesConservatively() {
        DynamicDimension n = new DynamicDimension("N");
        assertEquals(ProofStatus.PROVEN,
                GraphPredicateProof.evaluate(new DimensionEqual(n, n)));
        assertEquals(ProofStatus.DISPROVEN, GraphPredicateProof.evaluate(
                new DimensionEqual(new StaticDimension(2), new StaticDimension(3))));
        assertEquals(ProofStatus.DEFERRED, GraphPredicateProof.evaluate(
                new DimensionEqual(n, new DynamicDimension("M"))));
        assertEquals(ProofStatus.PROVEN, GraphPredicateProof.evaluate(
                new ShapeElementCountEqual(
                        Shape.of(Long.MAX_VALUE, 2), Shape.of(2, Long.MAX_VALUE))));
    }

    @Test
    void failsClosedForCustomKindWithRequiredContext() {
        Operation operation = new Operation(CustomKind.CUSTOM, NoOperationAttrs.INSTANCE);
        CompiledGraphModel graph = singleNodeGraph(
                operation,
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), false)),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2), false)));

        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();

        assertTrue(message.contains(CustomKind.class.getName()
                + ".CUSTOM: unsupported operation kind"));
    }

    private static void assertDescriptorMismatch(
            Operation operation,
            TensorDescriptor input,
            TensorDescriptor stored,
            String categoryEvidence) {
        CompiledGraphModel graph = singleNodeGraph(
                operation, List.of(input), List.of(stored));
        String message = assertThrows(IllegalArgumentException.class,
                () -> CapturedGraphInference.inferAndValidate(graph)).getMessage();
        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains("output[0] ValueId[value=1] expected=")),
                () -> assertTrue(message.contains(", stored=")),
                () -> assertTrue(message.contains(categoryEvidence)));
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(descriptor(type, shape, requiresGrad));
    }

    private static TensorDescriptor descriptor(
            DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(type, shape, Optional.empty(), requiresGrad);
    }

    private static CompiledGraphModel singleNodeGraph(
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        List<GraphValue> values = new ArrayList<>();
        List<ValueId> inputIds = new ArrayList<>();
        List<ValueId> outputIds = new ArrayList<>();
        long nextId = 0;
        for (TensorDescriptor input : inputs) {
            ValueId id = new ValueId(nextId++);
            values.add(new GraphValue(id, input));
            inputIds.add(id);
        }
        for (TensorDescriptor output : outputs) {
            ValueId id = new ValueId(nextId++);
            values.add(new GraphValue(id, output));
            outputIds.add(id);
        }
        NodeId nodeId = new NodeId(0);
        CompiledNode node = new CompiledNode(
                nodeId, operation, inputIds, outputIds);
        return new CompiledGraphModel(
                values,
                List.of(node),
                inputIds,
                outputIds,
                Map.of(nodeId, GraphPhase.FORWARD));
    }

    private enum CustomKind implements OperationKind {
        CUSTOM;

        @Override
        public List<OperationSignature> signatures() {
            return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
        }
    }
}

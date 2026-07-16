package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ForwardDeadCodeEliminationTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessDceContract() throws Exception {
        var method = ForwardDeadCodeElimination.class.getDeclaredMethod(
                "eliminate", CompiledGraphModel.class);
        var sidecarMethod = ForwardDeadCodeElimination.class.getDeclaredMethod(
                "eliminate", CompileTimeConstantGraph.class);
        var constructor = ForwardDeadCodeElimination.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        ForwardDeadCodeElimination.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        ForwardDeadCodeElimination.class.getModifiers())),
                () -> assertEquals(0, ForwardDeadCodeElimination.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(CompiledGraphModel.class, method.getReturnType()),
                () -> assertSame(CompileTimeConstantGraph.class, sidecarMethod.getReturnType()),
                () -> assertEquals(2, Arrays.stream(
                                ForwardDeadCodeElimination.class.getDeclaredMethods())
                        .filter(declared -> !declared.isSynthetic())
                        .filter(declared -> !Modifier.isPrivate(declared.getModifiers()))
                        .count()));
    }

    @Test
    void sidecarPrunesOnlyUnusedConstantsAndRebuildsDenseSourceRoles() {
        TensorDescriptor descriptor = descriptor(DataType.INT32, Shape.of(2), false);
        Operation live = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = graph(
                List.of(descriptor, descriptor, descriptor, descriptor),
                List.of(node(0, live, List.of(2L), List.of(3L))),
                List.of(0L, 1L, 2L),
                List.of(3L),
                List.of(GraphPhase.FORWARD));
        var unused = new CompileTimeConstantGraph.Splat(ScalarValue.int32(4));
        var consumed = new CompileTimeConstantGraph.Splat(ScalarValue.int32(9));
        CompileTimeConstantGraph source = new CompileTimeConstantGraph(
                graph, Map.of(new ValueId(1), unused, new ValueId(2), consumed));

        CompileTimeConstantGraph result = ForwardDeadCodeElimination.eliminate(source);

        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)),
                        result.graph().inputs()),
                () -> assertEquals(List.of(new ValueId(0)), result.bindableInputs()),
                () -> assertEquals(Map.of(new ValueId(1), consumed), result.constants()),
                () -> assertEquals(List.of(new ValueId(1)),
                        result.graph().nodes().getFirst().inputs()),
                () -> assertEquals(List.of(new ValueId(2)), result.graph().outputs()),
                () -> assertSame(live, result.graph().nodes().getFirst().operation()),
                () -> assertEquals(GraphPhase.FORWARD,
                        result.graph().nodePhases().get(new NodeId(0))));
    }

    @Test
    void sidecarRetainsConstantGraphOutputsConsumedConstantsAndNonForwardDependencies() {
        TensorDescriptor descriptor = descriptor(DataType.INT32, Shape.of(2), false);
        var splat = new CompileTimeConstantGraph.Splat(ScalarValue.int32(3));
        CompiledGraphModel passThrough = graph(
                List.of(descriptor), List.of(), List.of(0L), List.of(0L), List.of());
        CompileTimeConstantGraph graphOutput =
                new CompileTimeConstantGraph(passThrough, Map.of(new ValueId(0), splat));

        Operation backward = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel backwardGraph = graph(
                List.of(descriptor, descriptor),
                List.of(node(0, backward, List.of(0L), List.of(1L))),
                List.of(0L), List.of(0L), List.of(GraphPhase.BACKWARD));
        CompileTimeConstantGraph backwardSidecar = new CompileTimeConstantGraph(
                backwardGraph, Map.of(new ValueId(0), splat));

        assertAll(
                () -> assertSame(graphOutput,
                        ForwardDeadCodeElimination.eliminate(graphOutput)),
                () -> assertSame(backwardSidecar,
                        ForwardDeadCodeElimination.eliminate(backwardSidecar)),
                () -> assertEquals(1, ForwardDeadCodeElimination
                        .eliminate(backwardSidecar).constants().size()));
    }

    @Test
    void sidecarPrunesConstantUsedOnlyByRemovedForwardWorkAndKeepsUnusedBindableInput() {
        TensorDescriptor descriptor = descriptor(DataType.INT32, Shape.of(2), false);
        CompiledGraphModel graph = graph(
                List.of(descriptor, descriptor, descriptor, descriptor),
                List.of(
                        node(0, operation(UnaryElementwiseKind.NEG),
                                List.of(1L), List.of(2L)),
                        node(1, operation(UnaryElementwiseKind.NEG),
                                List.of(0L), List.of(3L))),
                List.of(0L, 1L), List.of(3L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD));
        CompileTimeConstantGraph source = new CompileTimeConstantGraph(
                graph,
                Map.of(new ValueId(1),
                        new CompileTimeConstantGraph.Splat(ScalarValue.int32(8))));

        CompileTimeConstantGraph result = ForwardDeadCodeElimination.eliminate(source);

        assertAll(
                () -> assertTrue(result.constants().isEmpty()),
                () -> assertEquals(List.of(new ValueId(0)), result.bindableInputs()),
                () -> assertEquals(1, result.graph().nodes().size()),
                () -> assertEquals(List.of(new ValueId(0)),
                        result.graph().nodes().getFirst().inputs()));
    }

    @Test
    void rejectsNullAndReturnsExactGraphWhenEveryNodeIsLive() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2), true);
        CompiledGraphModel graph = graph(
                List.of(descriptor, descriptor),
                List.of(node(0, operation(UnaryElementwiseKind.ABS),
                        List.of(0L), List.of(1L))),
                List.of(0L),
                List.of(1L),
                List.of(GraphPhase.FORWARD));

        assertAll(
                () -> assertEquals("graph", assertThrows(NullPointerException.class,
                        () -> ForwardDeadCodeElimination.eliminate((CompiledGraphModel) null))
                        .getMessage()),
                () -> assertSame(graph, ForwardDeadCodeElimination.eliminate(graph)));
    }

    @Test
    void removesOnlyDeadForwardNodesAndRetainsInputsAndWholeMultiOutputNodes() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor selected = descriptor(DataType.FLOAT32, Shape.of(2), true);
        TensorDescriptor indices = descriptor(DataType.INT64, Shape.of(2), false);
        Operation dead = operation(UnaryElementwiseKind.ABS);
        Operation topK = new Operation(TopKKind.TOP_K, new TopKAttrs(0, 2, true, true));
        CompiledGraphModel graph = graph(
                List.of(input, input, input, input, selected, indices),
                List.of(
                        node(0, dead, List.of(0L), List.of(3L)),
                        node(1, topK, List.of(1L), List.of(4L, 5L))),
                List.of(0L, 1L, 2L),
                List.of(4L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD));

        CompiledGraphModel result = ForwardDeadCodeElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                        result.inputs()),
                () -> assertEquals(1, result.nodes().size()),
                () -> assertSame(topK, result.nodes().getFirst().operation()),
                () -> assertEquals(List.of(new ValueId(3), new ValueId(4)),
                        result.nodes().getFirst().outputs()),
                () -> assertEquals(List.of(new ValueId(3)), result.outputs()),
                () -> assertSame(selected, result.values().get(3).descriptor()),
                () -> assertSame(indices, result.values().get(4).descriptor()),
                () -> assertSame(result,
                        CapturedGraphInference.inferAndValidate(result).graph()),
                () -> assertEquals(2, graph.nodes().size()));
    }

    @Test
    void retainsNonForwardWorkAndItsForwardDependencyClosure() {
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(2), true);
        Operation dependency = operation(UnaryElementwiseKind.ABS);
        Operation backward = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = graph(
                List.of(descriptor, descriptor, descriptor, descriptor),
                List.of(
                        node(0, dependency, List.of(0L), List.of(1L)),
                        node(1, backward, List.of(1L), List.of(2L)),
                        node(2, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(3L))),
                List.of(0L),
                List.of(0L),
                List.of(GraphPhase.FORWARD, GraphPhase.BACKWARD, GraphPhase.FORWARD));

        CompiledGraphModel result = ForwardDeadCodeElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(2, result.nodes().size()),
                () -> assertSame(dependency, result.nodes().get(0).operation()),
                () -> assertSame(backward, result.nodes().get(1).operation()),
                () -> assertEquals(GraphPhase.BACKWARD,
                        result.nodePhases().get(new NodeId(1))),
                () -> assertEquals(List.of(new ValueId(0)), result.outputs()));
    }

    @Test
    void retainsExplicitRngStateMaskAndNextStateSlotsOfLiveDropout() {
        TensorDescriptor input = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        TensorDescriptor output = descriptor(DataType.FLOAT32, Shape.of(4), true);
        TensorDescriptor mask = descriptor(DataType.BOOL, Shape.of(4), false);
        Operation initialState = new Operation(
                GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(5, 9));
        Operation dropout = new Operation(DropoutKind.DROPOUT, new DropoutAttrs(0.5));
        CompiledGraphModel graph = graph(
                List.of(input, input, state, output, mask, state),
                List.of(
                        node(0, operation(UnaryElementwiseKind.ABS),
                                List.of(0L), List.of(1L)),
                        node(1, initialState, List.of(), List.of(2L)),
                        node(2, dropout, List.of(0L, 2L), List.of(3L, 4L, 5L))),
                List.of(0L),
                List.of(3L),
                List.of(GraphPhase.FORWARD, GraphPhase.FORWARD, GraphPhase.FORWARD));

        CompiledGraphModel result = ForwardDeadCodeElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(2, result.nodes().size()),
                () -> assertSame(initialState, result.nodes().get(0).operation()),
                () -> assertSame(dropout, result.nodes().get(1).operation()),
                () -> assertEquals(List.of(new ValueId(2), new ValueId(3), new ValueId(4)),
                        result.nodes().get(1).outputs()),
                () -> assertEquals(5, result.values().size()),
                () -> assertSame(result,
                        CapturedGraphInference.inferAndValidate(result).graph()));
    }

    @Test
    void walksDeepDependencyChainsWithoutJavaRecursion() {
        int depth = 20_000;
        TensorDescriptor descriptor = descriptor(DataType.FLOAT32, Shape.of(1), false);
        List<TensorDescriptor> descriptors = new ArrayList<>(depth + 2);
        List<CompiledNode> nodes = new ArrayList<>(depth + 1);
        List<GraphPhase> phases = new ArrayList<>(depth + 1);
        descriptors.add(descriptor);
        for (int index = 0; index < depth; index++) {
            descriptors.add(descriptor);
            nodes.add(node(index, operation(UnaryElementwiseKind.ABS),
                    List.of((long) index), List.of((long) index + 1)));
            phases.add(GraphPhase.FORWARD);
        }
        descriptors.add(descriptor);
        nodes.add(node(depth, operation(UnaryElementwiseKind.NEG),
                List.of(0L), List.of((long) depth + 1)));
        phases.add(GraphPhase.FORWARD);
        CompiledGraphModel graph = graph(
                descriptors,
                nodes,
                List.of(0L),
                List.of((long) depth),
                phases);

        CompiledGraphModel result = ForwardDeadCodeElimination.eliminate(graph);

        assertAll(
                () -> assertEquals(depth, result.nodes().size()),
                () -> assertEquals(depth + 1, result.values().size()),
                () -> assertEquals(List.of(new ValueId(depth)), result.outputs()));
    }

    private static CompiledGraphModel graph(
            List<TensorDescriptor> descriptors,
            List<CompiledNode> nodes,
            List<Long> inputIds,
            List<Long> outputIds,
            List<GraphPhase> phases) {
        List<GraphValue> values = new ArrayList<>(descriptors.size());
        for (int index = 0; index < descriptors.size(); index++) {
            values.add(new GraphValue(new ValueId(index), descriptors.get(index)));
        }
        Map<NodeId, GraphPhase> phaseMap = new LinkedHashMap<>();
        for (int index = 0; index < nodes.size(); index++) {
            phaseMap.put(nodes.get(index).id(), phases.get(index));
        }
        return new CompiledGraphModel(
                values, nodes, ids(inputIds), ids(outputIds), phaseMap);
    }

    private static CompiledNode node(
            long id, Operation operation, List<Long> inputs, List<Long> outputs) {
        return new CompiledNode(new NodeId(id), operation, ids(inputs), ids(outputs));
    }

    private static List<ValueId> ids(List<Long> values) {
        return values.stream().map(ValueId::new).toList();
    }

    private static Operation operation(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(
            DataType dataType, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad);
    }
}

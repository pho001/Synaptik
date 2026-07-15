package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TopKResult;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GraphCaptureTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessCaptureContract() throws Exception {
        var capture = GraphCapture.class.getDeclaredMethod("capture", List.class);
        assertAll(
                () -> assertTrue(Modifier.isFinal(GraphCapture.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(GraphCapture.class.getModifiers())),
                () -> assertEquals(0, GraphCapture.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isStatic(capture.getModifiers())),
                () -> assertFalse(Modifier.isPublic(capture.getModifiers())),
                () -> assertSame(CompiledGraphModel.class, capture.getReturnType()),
                () -> assertEquals(1, Arrays.stream(GraphCapture.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .filter(method -> !Modifier.isPrivate(method.getModifiers()))
                        .count()));
    }

    @Test
    void capturesSingleOutputChainInDeterministicPostorderWithExactReferences() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor first = input.abs();
        Tensor output = first.neg();

        CompiledGraphModel graph = GraphCapture.capture(List.of(output));

        assertAll(
                () -> assertEquals(ids(0, 1, 2), graph.values().stream()
                        .map(value -> value.id()).toList()),
                () -> assertEquals(List.of(new ValueId(0)), graph.inputs()),
                () -> assertEquals(List.of(new ValueId(2)), graph.outputs()),
                () -> assertEquals(List.of(new NodeId(0), new NodeId(1)), graph.nodes().stream()
                        .map(node -> node.id()).toList()),
                () -> assertEquals(List.of(new ValueId(0)), graph.nodes().get(0).inputs()),
                () -> assertEquals(List.of(new ValueId(1)), graph.nodes().get(1).inputs()),
                () -> assertSame(input.descriptor(), graph.values().get(0).descriptor()),
                () -> assertSame(first.descriptor(), graph.values().get(1).descriptor()),
                () -> assertSame(output.descriptor(), graph.values().get(2).descriptor()),
                () -> assertSame(first.provenance().orElseThrow().operation(),
                        graph.nodes().get(0).operation()),
                () -> assertSame(output.provenance().orElseThrow().operation(),
                        graph.nodes().get(1).operation()),
                () -> assertEquals(Map.of(
                        new NodeId(0), GraphPhase.FORWARD,
                        new NodeId(1), GraphPhase.FORWARD), graph.nodePhases()));
    }

    @Test
    void capturesPassThroughLeafAsBothBoundaryRolesWithoutCopyingCallerList() {
        Tensor leaf = tensor(DataType.INT64, Shape.scalar(), false);
        var requested = new ArrayList<>(List.of(leaf));

        CompiledGraphModel graph = GraphCapture.capture(requested);
        requested.clear();

        assertAll(
                () -> assertEquals(1, graph.values().size()),
                () -> assertSame(leaf.descriptor(), graph.values().get(0).descriptor()),
                () -> assertEquals(List.of(new ValueId(0)), graph.inputs()),
                () -> assertEquals(List.of(new ValueId(0)), graph.outputs()),
                () -> assertTrue(graph.nodes().isEmpty()),
                () -> assertTrue(graph.nodePhases().isEmpty()));
    }

    @Test
    void deduplicatesSharedIdentitiesAndPreservesRepeatedInputPositionsAndEncounterOrder() {
        Tensor left = tensor(DataType.FLOAT64, Shape.of(2), false);
        Tensor right = tensor(DataType.FLOAT64, Shape.of(2), false);
        Tensor shared = left.add(right);
        Tensor repeated = shared.add(shared);
        Tensor fanOut = shared.mul(left);

        CompiledGraphModel graph = GraphCapture.capture(List.of(repeated, fanOut));

        assertAll(
                () -> assertEquals(ids(0, 1, 2, 3, 4), graph.values().stream()
                        .map(value -> value.id()).toList()),
                () -> assertEquals(ids(0, 1), graph.inputs()),
                () -> assertEquals(ids(3, 4), graph.outputs()),
                () -> assertEquals(3, graph.nodes().size()),
                () -> assertEquals(ids(0, 1), graph.nodes().get(0).inputs()),
                () -> assertEquals(ids(2, 2), graph.nodes().get(1).inputs()),
                () -> assertEquals(ids(2, 0), graph.nodes().get(2).inputs()),
                () -> assertEquals(List.of(new ValueId(2)), graph.nodes().get(0).outputs()),
                () -> assertEquals(List.of(new ValueId(3)), graph.nodes().get(1).outputs()),
                () -> assertEquals(List.of(new ValueId(4)), graph.nodes().get(2).outputs()));
    }

    @Test
    void mapsSharedMultiOutputRequestsInCallerOrderWithoutDuplicatingProducer() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(5), true);
        TopKResult result = input.topK(3, 0);

        CompiledGraphModel graph = GraphCapture.capture(List.of(result.indices(), result.values()));

        assertAll(
                () -> assertEquals(3, graph.values().size()),
                () -> assertEquals(1, graph.nodes().size()),
                () -> assertEquals(ids(1, 2), graph.nodes().get(0).outputs()),
                () -> assertEquals(ids(2, 1), graph.outputs()),
                () -> assertSame(result.values().descriptor(), graph.values().get(1).descriptor()),
                () -> assertSame(result.indices().descriptor(), graph.values().get(2).descriptor()),
                () -> assertSame(result.values().provenance().orElseThrow().operation(),
                        graph.nodes().get(0).operation()));
    }

    @Test
    void preservesHiddenDropoutOutputAndOpaqueStateSourceAndEdge() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        GraphRngState initial = GraphRngState.initial(17L, 29L);
        DropoutResult result = input.dropout(0.25d, initial);
        var producer = result.output().provenance().orElseThrow().producer();

        CompiledGraphModel graph = GraphCapture.capture(List.of(result.output()));

        assertAll(
                () -> assertEquals(5, graph.values().size()),
                () -> assertEquals(List.of(new ValueId(0)), graph.inputs()),
                () -> assertEquals(2, graph.nodes().size()),
                () -> assertSame(GraphRngKind.INITIAL_STATE, graph.nodes().get(0).operation().kind()),
                () -> assertTrue(graph.nodes().get(0).inputs().isEmpty()),
                () -> assertEquals(List.of(new ValueId(1)), graph.nodes().get(0).outputs()),
                () -> assertSame(DropoutKind.DROPOUT, graph.nodes().get(1).operation().kind()),
                () -> assertEquals(ids(0, 1), graph.nodes().get(1).inputs()),
                () -> assertEquals(ids(2, 3, 4), graph.nodes().get(1).outputs()),
                () -> assertEquals(List.of(new ValueId(2)), graph.outputs()),
                () -> assertSame(producer.outputDescriptors().get(0),
                        graph.values().get(2).descriptor()),
                () -> assertSame(DataType.BOOL, graph.values().get(3).descriptor().dataType()),
                () -> assertSame(producer.outputDescriptors().get(1),
                        graph.values().get(3).descriptor()),
                () -> assertSame(producer.outputDescriptors().get(2),
                        graph.values().get(4).descriptor()));
    }

    @Test
    void keepsStructurallyEqualButIdentityDistinctProducerOccurrences() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor first = left.add(right);
        Tensor second = left.add(right);

        CompiledGraphModel graph = GraphCapture.capture(List.of(first, second));

        assertAll(
                () -> assertEquals(2, graph.nodes().size()),
                () -> assertEquals(graph.nodes().get(0).operation(), graph.nodes().get(1).operation()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertEquals(ids(2, 3), graph.outputs()));
    }

    @Test
    void restartsGraphLocalIdentifiersAndRepeatsEqualCapture() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor output = input.abs().neg();

        CompiledGraphModel first = GraphCapture.capture(List.of(output));
        tensor(DataType.INT32, Shape.of(99), false).topK(1, 0);
        CompiledGraphModel second = GraphCapture.capture(List.of(output));

        assertAll(
                () -> assertEquals(first, second),
                () -> assertEquals(new ValueId(0), second.values().get(0).id()),
                () -> assertEquals(new NodeId(0), second.nodes().get(0).id()));
    }

    @Test
    void snapshotsImmutableGraphCollections() {
        Tensor output = tensor(DataType.FLOAT32, Shape.of(1), false).abs();
        CompiledGraphModel graph = GraphCapture.capture(List.of(output));

        assertAll(
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.values().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.nodes().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.inputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.outputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.nodePhases().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.nodes().get(0).inputs().clear()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> graph.nodes().get(0).outputs().clear()));
    }

    @Test
    void capturesDeepChainWithoutRecursiveJavaStackUse() {
        Tensor output = tensor(DataType.FLOAT32, Shape.of(1), false);
        for (int index = 0; index < 20_000; index++) {
            output = output.abs();
        }

        CompiledGraphModel graph = GraphCapture.capture(List.of(output));

        assertAll(
                () -> assertEquals(20_001, graph.values().size()),
                () -> assertEquals(20_000, graph.nodes().size()),
                () -> assertEquals(new ValueId(20_000), graph.outputs().get(0)),
                () -> assertEquals(new NodeId(19_999), graph.nodes().get(19_999).id()));
    }

    @Test
    void rejectsInvalidRequestsInRequiredOrderAndByIdentity() {
        Tensor first = tensor(DataType.FLOAT32, Shape.of(1), false);
        Tensor second = tensor(DataType.FLOAT32, Shape.of(1), false);

        assertAll(
                () -> assertEquals("outputs", assertThrows(NullPointerException.class,
                        () -> GraphCapture.capture(null)).getMessage()),
                () -> assertEquals("outputs must not be empty",
                        assertThrows(IllegalArgumentException.class,
                                () -> GraphCapture.capture(List.of())).getMessage()),
                () -> assertEquals("outputs[1]", assertThrows(NullPointerException.class,
                        () -> GraphCapture.capture(Arrays.asList(first, null, null))).getMessage()),
                () -> assertEquals("outputs[2] duplicates outputs[0]",
                        assertThrows(IllegalArgumentException.class,
                                () -> GraphCapture.capture(List.of(first, second, first)))
                                .getMessage()));
    }

    @Test
    void rejectsDistinctWrappersThatResolveToOneLogicalValue() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor output = input.abs();
        Tensor duplicateWrapper = duplicateWrapper(output);

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> GraphCapture.capture(List.of(output, duplicateWrapper)));

        assertAll(
                () -> assertTrue(failure.getMessage().contains("outputs[1]")),
                () -> assertTrue(failure.getMessage().contains("outputs[0]")),
                () -> assertTrue(failure.getMessage().contains("ValueId[value=1]")));
    }

    private static Tensor duplicateWrapper(Tensor source) throws Exception {
        Constructor<Tensor> constructor = Tensor.class.getDeclaredConstructor(
                TensorId.class,
                TensorDescriptor.class,
                Optional.class,
                Optional.class,
                Optional.class);
        constructor.setAccessible(true);
        return constructor.newInstance(
                new TensorId(Long.MAX_VALUE),
                source.descriptor(),
                Optional.empty(),
                Optional.of(source.provenance().orElseThrow()),
                Optional.empty());
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
    }

    private static List<ValueId> ids(long... values) {
        return Arrays.stream(values).mapToObj(ValueId::new).toList();
    }
}

package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class ScanBodyInferenceTest {
    static Tensor tensor(DataType type, boolean grad, long... dims) {
        Shape shape = Shape.of(dims);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    @Test void sourceAdmissionIsIndependentOfAdDomainAndKeepsUnusedFormals() {
        for (DataType type : List.of(DataType.FLOAT16, DataType.FLOAT32)) {
            Tensor h = tensor(type, false, 2), x = tensor(type, false, 2), unused = tensor(type, false, 2);
            Tensor next = type == DataType.FLOAT16 ? h.add(x).tanh() : h.add(x).relu();
            var body = new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(next), List.of(next, h));
            var scan = Scan.apply(List.of(tensor(type, false, 2)), List.of(tensor(type, false, 3, 2)),
                    List.of(tensor(type, false, 2)), body, false);
            var graph = GraphCapture.capture(List.of(scan.outputs().getFirst()));
            assertTrue(CapturedGraphInference.inferAndValidate(graph).constraints().isEmpty());
            var captured = graph.nodes().getFirst().scanBody().orElseThrow();
            assertEquals(3, captured.graph().inputs().size());
            assertEquals(captured.nextCarry().getFirst(), captured.emissions().getFirst());
            assertEquals(body.retainedValues().size(), captured.residuals().size());
        }
    }

    @Test void templateReuseHasIndependentScopesButEscapeAndDifferentTemplateSharingReject() {
        Tensor h = tensor(DataType.FLOAT64, true), x = tensor(DataType.FLOAT64, true);
        Tensor next = h.add(x).tanh();
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
        Tensor initial = tensor(DataType.FLOAT64, true), sequence = tensor(DataType.FLOAT64, true, 3);
        var a = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
        var b = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, true);
        var graph = GraphCapture.capture(List.of(a.finalCarry().getFirst(), b.finalCarry().getFirst()));
        var first = graph.nodes().get(0).scanBody().orElseThrow();
        var second = graph.nodes().get(1).scanBody().orElseThrow();
        assertNotSame(first.graph(), second.graph());
        assertEquals(first.graph().inputs(), second.graph().inputs());
        assertThrows(IllegalArgumentException.class, () -> GraphCapture.capture(List.of(a.finalCarry().getFirst(), next)));
        assertThrows(IllegalArgumentException.class, () -> GraphCapture.capture(List.of(a.finalCarry().getFirst(), h)));
        var other = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of());
        var c = Scan.apply(List.of(initial), List.of(sequence), List.of(), other, false);
        assertThrows(IllegalArgumentException.class, () -> GraphCapture.capture(List.of(a.finalCarry().getFirst(), c.finalCarry().getFirst())));
    }

    @Test void capturesUnusedCanonicalSiblingAndRejectsReorderedResidualRoles() {
        Tensor h = tensor(DataType.FLOAT32, false, 2), x = tensor(DataType.FLOAT32, false, 2);
        var top = x.topK(1, 0);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x)), List.of(top.values()));
        var source = Scan.apply(List.of(tensor(DataType.FLOAT32, false, 2)), List.of(tensor(DataType.FLOAT32, false, 3, 2)), List.of(), body, false);
        var graph = GraphCapture.capture(List.of(source.finalCarry().getFirst()));
        var node = graph.nodes().getFirst();
        var captured = node.scanBody().orElseThrow();
        assertEquals(body.retainedValues().size(), captured.residuals().size());
        assertTrue(captured.graph().nodes().stream().anyMatch(n -> n.outputs().size() == 2));
        assertTrue(CapturedGraphInference.inferAndValidate(graph).constraints().isEmpty());
        var reversed = new ArrayList<>(captured.residuals()); Collections.reverse(reversed);
        var malformed = new io.github.pho001.synaptik.model.graph.CompiledScanBody(captured.graph(), captured.nextCarry(), captured.emissions(), reversed, List.of());
        var badNode = new io.github.pho001.synaptik.model.graph.CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(malformed));
        var badGraph = new io.github.pho001.synaptik.model.graph.CompiledGraphModel(graph.values(), List.of(badNode), graph.inputs(), graph.outputs(), graph.nodePhases());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(badGraph));
    }
}

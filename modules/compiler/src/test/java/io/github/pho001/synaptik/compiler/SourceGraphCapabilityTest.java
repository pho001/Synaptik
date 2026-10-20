package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Raw, storage-free constructor coverage for the joint CNN/static-recurrent/flat-scan slice. */
final class SourceGraphCapabilityTest {
    /**
     * Creates a storage-free leaf with unresolved layout for cold-boundary checks.
     * @param type exact logical dtype
     * @param grad gradient eligibility, valid for the dtype
     * @param shape nonnegative static extents
     * @return fresh provenance-free leaf retaining no host values
     */
    private static Tensor leaf(DataType type, boolean grad, long... shape) {
        return TensorFactory.create(new TensorDescriptor(type, Shape.of(shape), Optional.empty(), grad));
    }

    /**
     * Creates an unresolved gradient-eligible FLOAT32 prototype.
     * @param shape nonnegative static extents
     * @return fresh storage-free leaf
     */
    private static Tensor floating(long... shape) { return leaf(DataType.FLOAT32, true, shape); }

    /**
     * Checks exact descriptor and recaptured-graph equality through Compiler's cold boundary.
     * @param outputs distinct ordered supported roots
     * @return reconstructed ordinary roots in the same order
     */
    private static List<Tensor> roundtrip(List<Tensor> outputs) {
        var source = GraphCompilationPort.captureSource(outputs);
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), source.leaves());
        assertEquals(outputs.stream().map(Tensor::descriptor).toList(), rebuilt.stream().map(Tensor::descriptor).toList());
        assertEquals(source.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
        return rebuilt;
    }

    /** Covers every added non-scan constructor without layout closure or index storage. */
    @Test void narrowConstructorsAndRawUnresolvedChainsRoundtripWithoutStorage() {
        Tensor matrix = floating(2, 3), weight = floating(3, 4);
        Tensor product = matrix.matmul(weight);
        Tensor sliced = product.sliceByLength(new long[]{1}, new long[]{2}, new int[]{1}, new long[]{1});
        // The element count exceeds OptionalLong's usual small-value cache: equality is numeric.
        Tensor large = floating(20, 30).reshape(Shape.of(10, 60));
        Tensor concatenated = Tensor.concat(0, floating(1, 2, 3), floating(1, 2, 3));
        Tensor indices = leaf(DataType.INT64, false, 2);
        Tensor gathered = matrix.gather(indices, 0);
        List<Tensor> rebuilt = roundtrip(List.of(matrix.sub(matrix), sliced, product.reshape(4, 2), large,
                concatenated.select(0, 1), concatenated.reshape(2, 6), gathered,
                matrix.expandDims(2).squeeze(2)));
        for (int i : new int[]{1, 2, 3, 4, 5, 6}) assertTrue(rebuilt.get(i).descriptor().layout().isEmpty());
        assertTrue(indices.hostStorage().isEmpty());
        Tensor conv1 = floating(1, 1, 4).conv1d(floating(2, 1, 2), floating(2), Conv1dAttrs.defaults());
        Tensor conv2 = floating(1, 1, 4, 4).conv2d(floating(2, 1, 2, 2), floating(2), Conv2dAttrs.defaults());
        Tensor conv3 = floating(1, 1, 3, 3, 3).conv3d(floating(2, 1, 2, 2, 2), floating(2), Conv3dAttrs.defaults());
        roundtrip(List.of(conv1, conv2, conv3, conv2.maxPool2d(new MaxPool2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false))));
    }

    /** Checks live and dead out-of-slice occurrences before any reconstructed identity allocation. */
    @Test void rejectsBroaderGeometryAndIntegralSubWithoutConstructingAnyProducer() {
        Tensor matrix = floating(2, 3), cube = floating(2, 2, 3);
        Tensor integer = leaf(DataType.INT64, false, 2, 3);
        for (Tensor outside : List.of(integer.sub(integer), matrix.expandDims(0),
                cube.select(1, 0), Tensor.concat(1, cube, cube),
                matrix.gather(leaf(DataType.INT32, false, 1), 0),
                matrix.sliceByLength(new long[]{0}, new long[]{1}, new int[]{0}, new long[]{1}),
                matrix.sliceByLength(new long[]{0}, new long[]{2}, new int[]{1}, new long[]{2}),
                floating(1, 1, 3, 3).conv2d(floating(1, 1, 1, 1), Conv2dAttrs.defaults()),
                floating(1, 1, 3, 3).conv2d(floating(1, 1, 1, 1), floating(1),
                        new Conv2dAttrs(2, 1, 0, 0, 1, 1, 1)),
                floating(1, 1, 3, 3).maxPool2d(new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false)))) {
            var graph = GraphCapture.capture(List.of(outside));
            var leaves = graph.inputs().stream().map(id -> graph.values().stream()
                    .filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor())
                    .map(TensorFactory::create).toList();
            Tensor marker = floating(1);
            assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(List.of(outside)));
            assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.reconstructSource(graph, leaves));
            var dead = new io.github.pho001.synaptik.model.graph.CompiledGraphModel(graph.values(), graph.nodes(),
                    graph.inputs(), List.of(graph.inputs().getFirst()), graph.nodePhases());
            assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.reconstructSource(dead, leaves));
            assertEquals(marker.id().value() + 1, floating(1).id().value());
            assertNotNull(graph);
        }
    }

    /** Checks all six scan signatures, both directions and empty time/batch with exact canonical ports. */
    @Test void allScanSignaturesDirectionsAndEmptyBoundariesRetainCanonicalPorts() {
        for (var kind : RecurrentScanKind.values()) for (boolean bias : List.of(false, true))
            for (var direction : RecurrentDirection.values()) for (long[] dimensions : List.of(
                    new long[]{2, 2}, new long[]{0, 2}, new long[]{2, 0})) {
                var ports = scan(kind, bias, direction, dimensions[0], dimensions[1]);
                Tensor last = ports.getLast(), hidden = ports.get(1);
                Tensor consumer = hidden.add(hidden);
                var rebuilt = roundtrip(List.of(last, consumer, ports.getFirst()));
                var producer = rebuilt.getFirst().provenance().orElseThrow().producer();
                assertEquals(ports.size(), producer.outputDescriptors().size());
                assertSame(rebuilt.getFirst(), producer.output(ports.size() - 1));
                assertSame(rebuilt.get(2), producer.output(0));
                assertSame(producer.output(1), rebuilt.get(1).provenance().orElseThrow().inputs().get(0));
                assertSame(producer.output(1), rebuilt.get(1).provenance().orElseThrow().inputs().get(1));
                var onlyLast = roundtrip(List.of(last));
                assertEquals(ports.size() - 1, onlyLast.getFirst().provenance().orElseThrow().outputIndex());
            }
    }

    /** Verifies that equal scans do not merge and opaque produced substitutions retain their provenance. */
    @Test void scansStayDistinctAndOpaqueRepeatedSubstitutionsStayAttached() {
        Tensor x = floating(2, 1, 1), lengths = leaf(DataType.INT64, false, 1);
        Tensor h = floating(1, 1), w = floating(1, 1);
        var a = RecurrentScan.rnn(x, lengths, h, w, w, RecurrentDirection.FORWARD);
        var b = RecurrentScan.rnn(x, lengths, h, w, w, RecurrentDirection.FORWARD);
        var source = GraphCompilationPort.captureSource(List.of(a.finalHidden(), b.outputs(), x, h));
        var replacements = source.leaves().stream().map(t -> t == lengths ? t : t.abs()).toList();
        var rebuilt = GraphCompilationPort.reconstructSource(source.graph(), replacements);
        assertNotSame(rebuilt.get(0).provenance().orElseThrow().producer(), rebuilt.get(1).provenance().orElseThrow().producer());
        var producer = rebuilt.get(0).provenance().orElseThrow().producer();
        assertSame(producer.inputs().get(3), producer.inputs().get(4));
        assertSame(rebuilt.get(2), producer.inputs().getFirst());
        assertSame(rebuilt.get(3), producer.inputs().get(2));
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(rebuilt));
    }

    /** Requires the documented boundary failure for unsupported zero-input producers, including dropout. */
    @Test void unsupportedZeroInputRngRejectsWithBoundaryFailureBeforeProducerAllocation() {
        Tensor input = floating(2, 3);
        Tensor output = input.dropout(.2, GraphRngState.initial(7, 11)).output();
        var raw = GraphCapture.capture(List.of(output));
        Tensor marker = floating(1);
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.captureSource(List.of(output)));
        assertThrows(IllegalArgumentException.class, () -> GraphCompilationPort.reconstructSource(raw, List.of(input)));
        assertEquals(marker.id().value() + 1, floating(1).id().value());
    }

    /** Verifies that untrusted target attributes cannot bypass checked-geometry failure translation. */
    @Test void overflowingReshapeTargetAttributeRejectsBeforeProducerAllocation() {
        Tensor input = floating(2, 3);
        var raw = GraphCapture.capture(List.of(input.reshape(3, 2)));
        var old = raw.nodes().getFirst();
        var node = new io.github.pho001.synaptik.model.graph.CompiledNode(old.id(),
                new io.github.pho001.synaptik.model.operation.Operation(
                        io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE,
                        new io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs(Shape.of(Long.MAX_VALUE, 2))),
                old.inputs(), old.outputs());
        var graph = new io.github.pho001.synaptik.model.graph.CompiledGraphModel(raw.values(), List.of(node),
                raw.inputs(), raw.outputs(), raw.nodePhases());
        Tensor marker = floating(1);
        var failure = assertThrows(IllegalArgumentException.class,
                () -> GraphCompilationPort.reconstructSource(graph, List.of(input)));
        assertInstanceOf(ArithmeticException.class, failure.getCause());
        assertEquals(marker.id().value() + 1, floating(1).id().value());
    }

    /**
     * Builds one fixed scan using storage-free prototypes and independent input/hidden weights.
     * @param kind one of the three fixed transition families
     * @param bias whether to use its optional-bias overload
     * @param direction traversal direction, retained as ordinary operation attributes
     * @param time nonnegative static time extent
     * @param batch nonnegative static batch extent
     * @return canonical output ports in the family's declared order
     */
    private static List<Tensor> scan(RecurrentScanKind kind, boolean bias, RecurrentDirection direction, long time, long batch) {
        Tensor x = floating(time, batch, 1), lengths = leaf(DataType.INT64, false, batch), h = floating(batch, 1);
        int gates = kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1;
        Tensor w = floating(gates, 1), u = floating(gates, 1), b = floating(gates);
        if (kind == RecurrentScanKind.LSTM) {
            Tensor c = floating(batch, 1);
            var result = bias ? RecurrentScan.lstm(x, lengths, h, c, w, u, b, direction)
                    : RecurrentScan.lstm(x, lengths, h, c, w, u, direction);
            return List.of(result.outputs(), result.finalHidden(), result.finalCell());
        }
        var result = kind == RecurrentScanKind.RNN_TANH
                ? (bias ? RecurrentScan.rnn(x, lengths, h, w, u, b, direction) : RecurrentScan.rnn(x, lengths, h, w, u, direction))
                : (bias ? RecurrentScan.gru(x, lengths, h, w, u, b, direction) : RecurrentScan.gru(x, lengths, h, w, u, direction));
        return List.of(result.outputs(), result.finalHidden());
    }
}

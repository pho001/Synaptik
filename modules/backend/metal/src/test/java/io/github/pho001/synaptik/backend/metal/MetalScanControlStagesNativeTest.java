package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Device regressions for ordinary stages inside a compiled Scan body. The shared raw harness
 * requires SYNAPTIK_METAL_TEST_LIBRARY; these definitions are not device-execution evidence.
 * Integral expectations compare represented bytes, never floating conversions.
 */
class MetalScanControlStagesNativeTest {
    @Test void replacementScatterRunsUpdateStageForBoolI32AndI64InBothModesAndTraversals() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                try (var run = new MetalGeneralScanNativeTest.RawScan(scatter(type, 2, 2, lengths, reverse), null)) {
                    long low = type == DataType.BOOL ? 0 : type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                    long high = type == DataType.BOOL ? 1 : type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                    byte[] first = represented(type, low, high, high, low);
                    byte[] second = represented(type, high, low, low, high);
                    run.bytes(0, represented(type, 1, 1, 1, 1));
                    run.longs(1, 1, 0, 1, 0, 1, 0, 1, 0);
                    run.bytes(2, concat(first, second));
                    if (lengths) run.longs(3, 2, 2);
                    run.execute();
                    assertArrayEquals(reverse ? second : first, run.outputBytes(0));
                }
            }
    }

    @Test void activeReplacementDuplicatesRejectBeforeEveryPublicationIncludingBool() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                try (var run = new MetalGeneralScanNativeTest.RawScan(scatter(type, 1, 2, lengths, reverse), null)) {
                    run.bytes(0, represented(type, 1, 0, 0, 1));
                    run.longs(1, 0, 0, 1, 0);
                    run.bytes(2, represented(type, 1, 1, 0, 0));
                    if (lengths) run.longs(3, 1, 1);
                    byte[] finalPoison = new byte[represented(type, 0, 0, 0, 0).length];
                    Arrays.fill(finalPoison, (byte) 0x5a);
                    // Seed both publications independently, including byte-sized BOOL results.
                    run.bytes(lengths ? 4 : 3, finalPoison);
                    run.bytes(lengths ? 5 : 4, finalPoison);
                    var failure = assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
                    assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
                    assertArrayEquals(finalPoison, run.outputBytes(0));
                    assertArrayEquals(finalPoison, run.outputBytes(1));
                }
            }
    }

    @Test void inactiveScatterRowsIgnoreExtremeIndicesAndDuplicateUpdates() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (boolean reverse : List.of(false, true)) {
                try (var run = new MetalGeneralScanNativeTest.RawScan(scatter(type, 1, 2, true, reverse), null)) {
                    run.bytes(0, represented(type, 0, 0, 1, 0));
                    run.longs(1, 1, 0, Long.MIN_VALUE, Long.MAX_VALUE);
                    run.bytes(2, represented(type, 1, 0, 1, 1));
                    run.longs(3, 1, 0);
                    run.execute();
                    assertArrayEquals(represented(type, 0, 1, 1, 0), run.outputBytes(0));
                    assertArrayEquals(represented(type, 0, 1, 0, 0), run.outputBytes(1));
                    run.longs(1, 1, 0, 0, 0);
                    run.execute();
                    assertArrayEquals(represented(type, 0, 1, 1, 0), run.outputBytes(0));
                }
            }
    }

    @Test void unreducedMsePreservesBothRowOperandsAndSkipsInactiveNans() {
        for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            try (var run = new MetalGeneralScanNativeTest.RawScan(query(DataType.FLOAT32, 1, 2, lengths, reverse,
                    new Operation(LossKind.MEAN_SQUARED_ERROR, new MeanSquaredErrorAttrs(LossReduction.NONE)), 2, new long[] {2}, false), null)) {
                run.floats(0, 5, 7, lengths ? Float.NaN : 3, lengths ? Float.NaN : 8);
                run.floats(1, 2, 3, lengths ? Float.NaN : 1, lengths ? Float.NaN : 5);
                if (lengths) run.longs(2, 1, 0);
                run.execute();
                assertArrayEquals(floatBits(9, 16, lengths ? Float.NaN : 4, lengths ? Float.NaN : 9), run.output(0));
                assertArrayEquals(floatBits(9, 16, lengths ? 0 : 4, lengths ? 0 : 9), run.output(1));
            }
        }
    }

    @Test void integralNonoverlappingFoldPlacesSignedBitsExactlyWithoutFloatingConversion() {
        for (var type : List.of(DataType.INT32, DataType.INT64))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                try (var run = new MetalGeneralScanNativeTest.RawScan(fold(type, 1, 2, lengths, reverse), null)) {
                    long min = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                    long max = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                    byte[] initial = represented(type, 9, 8, 7, 6, min, max, -1, 0);
                    byte[] source = represented(type, min, max, -1, 0, 0, -1, max, min);
                    run.bytes(0, initial); run.bytes(1, source);
                    if (lengths) run.longs(2, 1, 0);
                    run.execute();
                    assertArrayEquals(lengths ? concat(Arrays.copyOf(source, source.length / 2), Arrays.copyOfRange(initial, initial.length / 2, initial.length)) : source, run.outputBytes(0));
                    assertArrayEquals(lengths ? concat(Arrays.copyOf(source, source.length / 2), new byte[source.length / 2]) : source, run.outputBytes(1));
                }
            }
    }

    @Test void zeroTimeAndBatchRetainIdentitiesAndValidateLengthsAndBodyMetadata() {
        for (var type : List.of(DataType.INT32, DataType.INT64)) for (boolean lengths : List.of(false, true))
            for (boolean reverse : List.of(false, true)) for (int[] geometry : new int[][] {{0, 2}, {2, 0}}) {
                int time = geometry[0], batch = geometry[1];
                var query = fold(type, time, batch, lengths, reverse);
                try (var run = new MetalGeneralScanNativeTest.RawScan(query, null)) {
                    byte[] initial = batch == 0 ? new byte[0] : represented(type, -1, 0, 1, 2, 3, 4, 5, 6);
                    run.bytes(0, initial);
                    if (lengths) run.longs(2, new long[batch]);
                    run.execute();
                    assertArrayEquals(initial, run.outputBytes(0));
                    assertArrayEquals(new byte[0], run.outputBytes(1));
                    if (lengths && batch != 0) {
                        run.longs(2, 0, 1);
                        assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
                        assertArrayEquals(initial, run.outputBytes(0));
                    }
                }
                var wrongOutputs = new ArrayList<>(query.outputs());
                wrongOutputs.set(0, MetalScanPlanTest.descriptor(type, batch, 3));
                var malformed = new OperationCapabilityQuery(query.operation(), query.inputs(), wrongOutputs, query.scanBody(), Optional.empty());
                assertThrows(RuntimeException.class, () -> { try (var ignored = new MetalGeneralScanNativeTest.RawScan(malformed, null)) {} });
            }
    }

    private static OperationCapabilityQuery scatter(DataType type, int time, int batch, boolean lengths, boolean reverse) {
        return query(type, time, batch, lengths, reverse,
                new Operation(AxisScatterKind.SCATTER_ELEMENTS, new ScatterElementsAttrs(1, ScatterReduction.NONE)), 2, new long[] {2}, true);
    }

    private static OperationCapabilityQuery fold(DataType type, int time, int batch, boolean lengths, boolean reverse) {
        return query(type, time, batch, lengths, reverse,
                new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(1, 4, 2)), 4, new long[] {2, 2}, false);
    }

    private static OperationCapabilityQuery query(DataType type, int time, int batch, boolean lengths, boolean reverse,
            Operation operation, int width, long[] sequenceFeatures, boolean scatter) {
        var carry = MetalScanPlanTest.descriptor(type, batch, width);
        long[] stepShape = new long[sequenceFeatures.length + 1]; stepShape[0] = batch;
        System.arraycopy(sequenceFeatures, 0, stepShape, 1, sequenceFeatures.length);
        var step = MetalScanPlanTest.descriptor(type, stepShape);
        var h = new ValueId(0); var x = new ValueId(1); var y = new ValueId(scatter ? 3 : 2);
        var values = new ArrayList<GraphValue>(); values.add(new GraphValue(h, carry));
        values.add(new GraphValue(x, scatter ? MetalScanPlanTest.descriptor(DataType.INT64, stepShape) : step));
        var formals = new ArrayList<>(List.of(h, x));
        List<ValueId> operands;
        if (scatter) { var updates = new ValueId(2); values.add(new GraphValue(updates, step)); formals.add(updates); operands = List.copyOf(formals); }
        else operands = operation.kind() == WindowTransformKind.FOLD_AXIS ? List.of(x) : List.of(h, x);
        values.add(new GraphValue(y, carry));
        var node = new CompiledNode(new NodeId(0), operation, operands, List.of(y));
        var graph = new CompiledGraphModel(values, List.of(node), formals, List.of(y), Map.of(node.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(y), List.of(y), List.of(), List.of(0, 1));
        int sequences = scatter ? 2 : 1;
        var attrs = lengths ? new ScanAttrs(time, 1, sequences, 0, 1, 0, reverse, true, 1,
                Collections.nCopies(sequences, OptionalInt.of(0))) : new ScanAttrs(time, 1, sequences, 0, 1, 0, reverse);
        long[] stacked = new long[stepShape.length + 1]; stacked[0] = time;
        System.arraycopy(stepShape, 0, stacked, 1, stepShape.length);
        var inputs = new ArrayList<TensorDescriptor>(); inputs.add(carry);
        inputs.add(MetalScanPlanTest.descriptor(scatter ? DataType.INT64 : type, stacked));
        if (scatter) inputs.add(MetalScanPlanTest.descriptor(type, stacked));
        if (lengths) inputs.add(MetalScanPlanTest.descriptor(DataType.INT64, batch));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(carry, MetalScanPlanTest.descriptor(type, time, batch, width)), Optional.of(body), Optional.empty());
    }

    private static byte[] represented(DataType type, long... values) {
        int width = type == DataType.BOOL ? 1 : type == DataType.INT32 ? 4 : 8;
        var bytes = ByteBuffer.allocate(values.length * width).order(ByteOrder.nativeOrder());
        for (long value : values) { if (width == 1) bytes.put((byte) value); else if (width == 4) bytes.putInt((int) value); else bytes.putLong(value); }
        return bytes.array();
    }

    private static byte[] concat(byte[] first, byte[] second) {
        byte[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length); return result;
    }

    private static int[] floatBits(float... values) {
        int[] bits = new int[values.length];
        for (int i = 0; i < values.length; i++) bits[i] = Float.floatToRawIntBits(values[i]);
        return bits;
    }
}

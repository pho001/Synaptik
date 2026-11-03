package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
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
    @Test void optionalIntegralAddLoadsOnlyPresentTermsAndPreservesSoleRawIdentities() {
        for (var type : List.of(DataType.INT32, DataType.INT64)) for (boolean swapped : List.of(false, true))
            for (int present = 0; present < 4; present++)
                for (long base : new long[] {100, type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE}) {
                    try (var run = new MetalGeneralScanNativeTest.RawScan(optionalIntegralAdd(type, present, swapped), null)) {
                        run.floats(0, 3); run.bytes(1, represented(type, base));
                        run.poisonOutputs(0x5a5a5a5a); run.execute();
                        long left = swapped ? 3 : base, right = swapped ? base : 3;
                        long expected = present == 0 ? 0 : present == 1 ? left : present == 2 ? right : left + right;
                        if (type == DataType.INT32) expected = (int) expected;
                        assertArrayEquals(represented(type, expected), run.outputBytes(0));
                    }
                }
    }

    private static OperationCapabilityQuery optionalIntegralAdd(DataType type, int present, boolean swapped) {
        var floating = MetalScanPlanTest.descriptor(DataType.FLOAT32);
        var integral = MetalScanPlanTest.descriptor(type);
        var seed = new ValueId(0); var invariant = new ValueId(1);
        var converted = new ValueId(2); var result = new ValueId(3);
        var cast = new CompiledNode(new NodeId(0), new Operation(CastKind.CAST, new CastAttrs(type)),
                List.of(seed), List.of(converted));
        var add = new CompiledNode(new NodeId(1), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                swapped ? List.of(converted, invariant) : List.of(invariant, converted), List.of(result));
        var graph = new CompiledGraphModel(List.of(new GraphValue(seed, floating), new GraphValue(invariant, integral),
                new GraphValue(converted, integral), new GraphValue(result, integral)), List.of(cast, add),
                List.of(seed, invariant), List.of(result), Map.of(cast.id(), GraphPhase.BACKWARD, add.id(), GraphPhase.BACKWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(result), List.of(), List.of(0));
        var activity = new ScanActivityPlan(graph, List.of(), List.of(0), List.of(List.of(), List.of(0)),
                Map.of(cast.id(), 1, add.id(), present == 0 ? 0 : 1),
                Map.of(add.id(), new ScanActivityPlan.Merge(present & 1, (present >>> 1) & 1)), List.of());
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(1, 0, 1, 1, 1, 0, false)),
                List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 1), integral),
                List.of(MetalScanPlanTest.descriptor(type, 1)), Optional.of(body), Optional.of(activity));
    }

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

    @Test void replacementScatterNdCopiesBaseThenPlacesExactBoolAndSignedUpdates() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (var indexType : List.of(DataType.INT32, DataType.INT64))
                for (int tuples : new int[] {1, 2}) for (boolean reverse : List.of(false, true)) {
                    var query = scatterNd(type, indexType, tuples, false, reverse);
                    assertTrue(MetalScanLowering.supports(query));
                    try (var run = new MetalGeneralScanNativeTest.RawScan(query, null)) {
                        long low = type == DataType.BOOL ? 0 : type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                        long high = type == DataType.BOOL ? 1 : type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                        byte[] base = type == DataType.BOOL ? represented(type, 1, 0, 0, 0)
                                : represented(type, 11, -13, 17, -19);
                        run.bytes(0, base);
                        run.bytes(1, represented(indexType, tuples == 1 ? new long[] {1} : new long[] {1, 0}));
                        run.bytes(2, represented(type, tuples == 1 ? new long[] {low, high} : new long[] {low, high, high, low}));
                        run.execute();
                        byte[] expected = tuples == 1
                                ? concat(Arrays.copyOfRange(base, 0, 2 * type.byteWidth()), represented(type, low, high))
                                : represented(type, high, low, low, high);
                        assertArrayEquals(expected, run.outputBytes(0));
                        assertArrayEquals(expected, run.outputBytes(1));
                    }
                }
    }

    @Test void replacementScatterNdPreflightsAllBoundsAndDuplicatesBeforeEitherStage() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (var indexType : List.of(DataType.INT32, DataType.INT64))
                for (boolean reverse : List.of(false, true)) {
                    try (var run = new MetalGeneralScanNativeTest.RawScan(scatterNd(type, indexType, 2, false, reverse), null)) {
                        run.bytes(0, represented(type, 1, 0, 0, 1));
                        run.bytes(2, represented(type, 0, 1, 1, 0));
                        byte[] poison = new byte[4 * type.byteWidth()]; Arrays.fill(poison, (byte) 0x5a);
                        for (long[] indices : new long[][] {{1, 1}, {1, 2}, {Long.MAX_VALUE, 0}}) {
                            run.bytes(1, represented(indexType, indices));
                            run.bytes(3, poison); run.bytes(4, poison);
                            var failure = assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
                            assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
                            assertArrayEquals(poison, run.outputBytes(0)); assertArrayEquals(poison, run.outputBytes(1));
                        }
                        run.bytes(1, represented(indexType, 1, 0)); run.execute();
                        assertArrayEquals(represented(type, 1, 0, 0, 1), run.outputBytes(0));
                        assertArrayEquals(represented(type, 1, 0, 0, 1), run.outputBytes(1));
                    }
                }
    }

    @Test void replacementScatterNdPreservesBatchRowsWithoutInspectingInactiveIndexTuples() {
        for (var type : List.of(DataType.BOOL, DataType.INT32, DataType.INT64))
            for (var indexType : List.of(DataType.INT32, DataType.INT64))
                for (boolean reverse : List.of(false, true)) {
                    try (var run = new MetalGeneralScanNativeTest.RawScan(scatterNd(type, indexType, 2, true, reverse), null)) {
                        long low = type == DataType.BOOL ? 0 : type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                        long high = type == DataType.BOOL ? 1 : type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                        long badMin = indexType == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                        long badMax = indexType == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                        byte[] base = represented(type, 1, 0, 0, 1, 1, 0, 0, 1);
                        run.bytes(0, base); run.bytes(1, represented(indexType, 1, 0, badMin, badMax));
                        run.bytes(2, represented(type, low, high, high, low, high, low, low, high));
                        run.longs(3, 1, 0); run.execute();
                        byte[] active = represented(type, high, low, low, high);
                        byte[] expectedCarry = concat(active, represented(type, 1, 0, 0, 1));
                        byte[] expectedHistory = concat(active, represented(type, 0, 0, 0, 0));
                        assertArrayEquals(expectedCarry, run.outputBytes(0)); assertArrayEquals(expectedHistory, run.outputBytes(1));
                        run.bytes(1, represented(indexType, 1, 0, 0, 0)); run.execute();
                        assertArrayEquals(expectedCarry, run.outputBytes(0)); assertArrayEquals(expectedHistory, run.outputBytes(1));
                        run.longs(3, 0, 0); run.execute();
                        assertArrayEquals(base, run.outputBytes(0));
                        assertArrayEquals(represented(type, 0, 0, 0, 0, 0, 0, 0, 0), run.outputBytes(1));
                    }
                }
    }

    @Test void replacementScatterNdStillRejectsMalformedUpdateGeometryAtZeroTime() {
        for (int time : new int[] {0, 1}) {
            var malformed = singleCarryQuery(new Operation(ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(0, ScatterReduction.NONE)),
                    time, false, false, List.of(MetalScanPlanTest.descriptor(DataType.INT32, 2, 2),
                            MetalScanPlanTest.descriptor(DataType.INT64, 2, 1), MetalScanPlanTest.descriptor(DataType.INT32, 2, 3)),
                    List.of(0, 1, 2));
            assertFalse(MetalScanLowering.supports(malformed));
            assertThrows(IllegalArgumentException.class, () -> MetalScanLowering.plan(malformed));
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

    @Test void integralMatmulPromotesAndWrapsExactlyForEveryOperandPairAndTraversal() {
        for (var left : List.of(DataType.INT32, DataType.INT64))
            for (var right : List.of(DataType.INT32, DataType.INT64))
                for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                    var result = left == DataType.INT64 || right == DataType.INT64 ? DataType.INT64 : DataType.INT32;
                    long min = left == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                    long max = left == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                    for (long[] operands : new long[][] {{min, max}, {-16777217, 2147483647}}) {
                        long value = -operands[0] + 2 * operands[1];
                        byte[] expected = represented(result, value);
                        try (var run = new MetalGeneralScanNativeTest.RawScan(integralMatmul(left, right, lengths, reverse, false), null)) {
                            run.bytes(0, lengths ? represented(result, 13, -17) : represented(result, 13));
                            byte[] step = lengths ? represented(left, operands[0], operands[1], min, max)
                                    : represented(left, operands[0], operands[1]);
                            run.bytes(1, concat(step, step));
                            run.bytes(2, represented(right, -1, 2));
                            if (lengths) run.longs(3, 2, 0);
                            run.execute();
                            assertArrayEquals(lengths ? concat(expected, represented(result, -17)) : expected, run.outputBytes(0));
                            byte[] emission = lengths ? concat(expected, represented(result, 0)) : expected;
                            assertArrayEquals(concat(emission, emission), run.outputBytes(1));
                        }
                    }
                }
    }

    @Test void integralTerminalMatmulContractsOnlyMatchedActiveRowsForEveryOperandPair() {
        for (var left : List.of(DataType.INT32, DataType.INT64))
            for (var right : List.of(DataType.INT32, DataType.INT64))
                for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                    var result = left == DataType.INT64 || right == DataType.INT64 ? DataType.INT64 : DataType.INT32;
                    long min = left == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
                    long max = left == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
                    try (var run = new MetalGeneralScanNativeTest.RawScan(integralMatmul(left, right, lengths, reverse, true), null)) {
                        byte[] l = represented(left, min, max), r = represented(right, -1, 2);
                        run.bytes(0, represented(result, -17, 13));
                        run.bytes(1, concat(l, l)); run.bytes(2, concat(r, r));
                        if (lengths) run.longs(3, 2, 0);
                        run.execute();
                        byte[] expected = represented(result, lengths ? -min : -min + 2 * max);
                        assertArrayEquals(concat(expected, expected), run.outputBytes(0));
                    }
                }
    }

    /**
     * Builds ordinary integral MATMUL bodies, preserving the Model's promoted result type.
     * Feature contraction keeps ROW on the left matrix; terminal contraction matches both
     * vector ROW axes as admitted by Model's matmulRow and publishes a residual, not a
     * forbidden terminal emission. An undemanded ROW carry passes through unchanged so the
     * semantic header retains a carry without adding arithmetic to the contraction body.
     *
     * @param left signed left carrier
     * @param right signed right carrier
     * @param lengths whether to attach original-row activity
     * @param reverse whether to visit time in reverse
     * @param terminal whether both vector operands contract their matched ROW axis
     * @return a two-step query with exact ordinary operand/result descriptors
     */
    private static OperationCapabilityQuery integralMatmul(DataType left, DataType right, boolean lengths,
            boolean reverse, boolean terminal) {
        var result = left == DataType.INT64 || right == DataType.INT64 ? DataType.INT64 : DataType.INT32;
        int batch = terminal || lengths ? 2 : 1;
        var h = new ValueId(0); var l = new ValueId(1); var r = new ValueId(2); var y = new ValueId(3);
        var ld = MetalScanPlanTest.descriptor(left, terminal ? new long[] {2} : new long[] {batch, 2});
        var rd = MetalScanPlanTest.descriptor(right, terminal ? new long[] {2} : new long[] {2, 1});
        var yd = MetalScanPlanTest.descriptor(result, terminal ? new long[0] : new long[] {batch, 1});
        var carry = MetalScanPlanTest.descriptor(result, batch, 1);
        var values = new ArrayList<GraphValue>();
        values.add(new GraphValue(h, carry));
        values.add(new GraphValue(l, ld)); values.add(new GraphValue(r, rd)); values.add(new GraphValue(y, yd));
        var op = new Operation(io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
        var node = new CompiledNode(new NodeId(0), op, List.of(l, r), List.of(y));
        var graph = new CompiledGraphModel(values, List.of(node), List.of(h, l, r),
                terminal ? List.of(h, y) : List.of(y), Map.of(node.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, terminal ? List.of(h) : List.of(y),
                terminal ? List.of() : List.of(y), terminal ? List.of(y) : List.of(),
                terminal ? List.of(1) : List.of(0, 1));
        int carries = 1, sequences = terminal ? 2 : 1, invariants = terminal ? 0 : 1;
        var attrs = lengths ? new ScanAttrs(2, carries, sequences, invariants, terminal ? 0 : 1, terminal ? 1 : 0, reverse, true,
                1, Collections.nCopies(sequences, OptionalInt.of(0)))
                : new ScanAttrs(2, carries, sequences, invariants, terminal ? 0 : 1, terminal ? 1 : 0, reverse);
        var inputs = new ArrayList<TensorDescriptor>();
        inputs.add(carry);
        inputs.add(MetalScanPlanTest.descriptor(left, terminal ? new long[] {2, 2} : new long[] {2, batch, 2}));
        inputs.add(terminal ? MetalScanPlanTest.descriptor(right, 2, 2) : rd);
        if (lengths) inputs.add(MetalScanPlanTest.descriptor(DataType.INT64, terminal ? 2 : batch));
        var outputs = terminal ? List.of(carry, MetalScanPlanTest.descriptor(result, 2))
                : List.of(yd, MetalScanPlanTest.descriptor(result, 2, batch, 1));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs, outputs, Optional.of(body), Optional.empty());
    }

    private static OperationCapabilityQuery scatter(DataType type, int time, int batch, boolean lengths, boolean reverse) {
        return query(type, time, batch, lengths, reverse,
                new Operation(AxisScatterKind.SCATTER_ELEMENTS, new ScatterElementsAttrs(1, ScatterReduction.NONE)), 2, new long[] {2}, true);
    }

    private static OperationCapabilityQuery scatterNd(
            DataType type, DataType indexType, int tuples, boolean lengths, boolean reverse) {
        return singleCarryQuery(new Operation(ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(lengths ? 1 : 0, ScatterReduction.NONE)),
                1, lengths, reverse, List.of(
                        MetalScanPlanTest.descriptor(type, lengths ? new long[] {2, 2, 2} : new long[] {2, 2}),
                        MetalScanPlanTest.descriptor(indexType, lengths ? new long[] {2, tuples, 1} : new long[] {tuples, 1}),
                        MetalScanPlanTest.descriptor(type, lengths ? new long[] {2, tuples, 2} : new long[] {tuples, 2})),
                List.of(0, 1, 2));
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
        var local = new ArrayList<TensorDescriptor>(); local.add(carry);
        local.add(scatter ? MetalScanPlanTest.descriptor(DataType.INT64, stepShape) : step);
        if (scatter) local.add(step);
        return singleCarryQuery(operation, time, lengths, reverse, local,
                scatter ? List.of(0, 1, 2) : operation.kind() == WindowTransformKind.FOLD_AXIS ? List.of(1) : List.of(0, 1));
    }

    /** One ordinary node and two canonical ports, shared by the stage/geometry fixtures. */
    private static OperationCapabilityQuery singleCarryQuery(Operation operation, int time, boolean lengths,
            boolean reverse, List<TensorDescriptor> local, List<Integer> operandPositions) {
        var carry = local.getFirst();
        var values = new ArrayList<GraphValue>();
        var formals = new ArrayList<ValueId>();
        var inputs = new ArrayList<TensorDescriptor>();
        for (int f = 0; f < local.size(); f++) {
            var id = new ValueId(f); formals.add(id); values.add(new GraphValue(id, local.get(f)));
            inputs.add(f == 0 ? local.get(f) : stacked(local.get(f), time));
        }
        var y = new ValueId(local.size()); values.add(new GraphValue(y, carry));
        var node = new CompiledNode(new NodeId(0), operation,
                operandPositions.stream().map(formals::get).toList(), List.of(y));
        var graph = new CompiledGraphModel(values, List.of(node), formals, List.of(y), Map.of(node.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(y), List.of(y), List.of(), List.of(0, 1));
        int sequences = local.size() - 1;
        var attrs = lengths ? new ScanAttrs(time, 1, sequences, 0, 1, 0, reverse, true, 1,
                Collections.nCopies(sequences, OptionalInt.of(0))) : new ScanAttrs(time, 1, sequences, 0, 1, 0, reverse);
        if (lengths) inputs.add(MetalScanPlanTest.descriptor(DataType.INT64, carry.shape().dimension(0).staticSize().orElseThrow()));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), inputs,
                List.of(carry, stacked(carry, time)), Optional.of(body), Optional.empty());
    }

    private static TensorDescriptor stacked(TensorDescriptor step, int time) {
        long[] shape = step.shape().toLongArray();
        long[] stacked = new long[shape.length + 1]; stacked[0] = time;
        System.arraycopy(shape, 0, stacked, 1, shape.length);
        return MetalScanPlanTest.descriptor(step.dataType(), stacked);
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

package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

/** Source-matched device consumers; no child-side numerical claims are made by defining them. */
class MetalGeneralScanNativeTest {
    @Test void nativeTraversalSavesOriginalTimeAndPreviousCarryWithoutReadingPadding() {
        for (boolean reverse : List.of(false, true)) try (var run = new RawScan(MetalScanPlanTest.additiveQuery(3, 2, true, reverse), null)) {
            run.floats(0, 1, -0.0f);
            run.floats(1, 1, 10, 2, Float.NaN, 3, Float.NaN);
            run.longs(2, 3, 1);
            run.execute();
            assertRaw(new float[] {7, 10}, run.output(0));
            assertRaw(reverse ? new float[] {7, 10, 6, 0, 4, 0} : new float[] {2, 10, 4, 0, 7, 0}, run.output(1));
            assertRaw(reverse ? new float[] {6, -0.0f, 4, 0, 1, 0} : new float[] {1, -0.0f, 2, 0, 4, 0}, run.output(2));
        }
    }

    @Test void allNumericCarriersUseOriginalStorageAndRoundAtEveryCarryTransition() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                float large = type == DataType.FLOAT16 ? 2048 : 256;
                float[] original = {large, -0.0f};
                float[] sequence = {1, 3, -large, Float.NaN, 2, Float.NaN};
                if (!lengths) { sequence[3] = 4; sequence[5] = 5; }
                float[] carry = original.clone(), history = new float[6], previous = new float[6];
                for (int iteration = 0; iteration < 3; iteration++) {
                    int time = reverse ? 2 - iteration : iteration;
                    for (int row = 0; row < 2; row++) if (!lengths || row == 0 || time == 0) {
                        previous[time * 2 + row] = carry[row];
                        carry[row] = rounded(type, carry[row] + sequence[time * 2 + row]);
                        history[time * 2 + row] = carry[row];
                    }
                }
                try (var run = new RawScan(MetalScanPlanTest.additiveQuery(type, 3, 2, lengths, reverse), null)) {
                    run.bytes(0, numeric(type, original)); run.bytes(1, numeric(type, sequence));
                    if (lengths) run.longs(2, 3, 1);
                    run.poisonOutputs(0x5a5a5a5a); run.execute();
                    assertArrayEquals(numeric(type, carry), run.outputBytes(0));
                    assertArrayEquals(numeric(type, history), run.outputBytes(1));
                    assertArrayEquals(numeric(type, previous), run.outputBytes(2));
                    if (lengths) {
                        run.longs(2, 0, 0); run.execute();
                        assertArrayEquals(numeric(type, original), run.outputBytes(0));
                        assertArrayEquals(new byte[6 * type.byteWidth()], run.outputBytes(1));
                        assertArrayEquals(new byte[6 * type.byteWidth()], run.outputBytes(2));
                    }
                }
            }
    }

    @Test void stridedOffsetExternalNumericIngressMapsTimeAndRowsBeforeDenseScanBinding() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                var query = MetalScanPlanTest.stridedAdditiveQuery(type, lengths, reverse);
                try (var run = new RawScan(query, null)) {
                    run.bytes(0, numeric(type, Float.NaN, 0, Float.NaN, -0.0f));
                    run.bytes(1, numeric(type, Float.NaN, 1, Float.NaN, 10, Float.NaN,
                            Float.NaN, 2, Float.NaN, lengths ? Float.NaN : 20, Float.NaN,
                            Float.NaN, 3, Float.NaN, lengths ? Float.NaN : 30));
                    run.poisonOutputs(0x5a5a5a5a);
                    if (lengths) {
                        run.longs(2, 3, 4);
                        assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
                        for (int output = 0; output < run.outputs.length; output++) {
                            byte[] sentinel = new byte[run.outputSizes[output]];
                            Arrays.fill(sentinel, (byte) 0x5a);
                            assertArrayEquals(sentinel, run.outputBytes(output),
                                    "invalid lengths must precede mapped ingress and every publication");
                        }
                        run.longs(2, 3, 1);
                    }
                    float[] carry = {0, -0.0f}, history = new float[6], previous = new float[6];
                    for (int iteration = 0; iteration < 3; iteration++) {
                        int time = reverse ? 2 - iteration : iteration;
                        for (int row = 0; row < 2; row++) if (!lengths || row == 0 || time == 0) {
                            previous[time * 2 + row] = carry[row];
                            carry[row] += (time + 1) * (row == 0 ? 1 : 10);
                            history[time * 2 + row] = carry[row];
                        }
                    }
                    run.execute();
                    assertEquals(6.0f, carry[0], "audited sequence 1,2,3 ends at six");
                    assertArrayEquals(numeric(type, carry), run.outputBytes(0));
                    assertArrayEquals(numeric(type, history), run.outputBytes(1));
                    assertArrayEquals(numeric(type, previous), run.outputBytes(2));
                }
            }
    }

    @Test void mappedSourceOnlyIngressPreservesRawCarrySequenceAndOffsetInvariantPayloads() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var original = MetalScanPlanTest.sourceOnlyQuery(type, 2, 2, false, false);
            var inputs = new ArrayList<>(original.inputs());
            inputs.set(0, MetalScanPlanTest.mapped(inputs.get(0), new long[] {2, 1}, 1));
            inputs.set(1, MetalScanPlanTest.mapped(inputs.get(1), new long[] {5, 2, 1}, 1));
            inputs.set(2, MetalScanPlanTest.mapped(inputs.get(2), new long[0], 1));
            var query = new OperationCapabilityQuery(original.operation(), inputs, original.outputs(),
                    original.scanBody(), original.scanActivityPlan());
            int negativeZero = type == DataType.FLOAT32 ? 0x80000000 : 0x8000;
            int nan = type == DataType.FLOAT32 ? 0x7fc12345 : type == DataType.BFLOAT16 ? 0x7fc5 : 0x7e45;
            try (var run = new RawScan(query, null)) {
                run.bytes(0, rawNumeric(type, 0, negativeZero, 0, nan));
                run.bytes(1, rawNumeric(type, 0, nan, 0, negativeZero, 0, 0, negativeZero, 0, nan));
                run.bytes(2, rawNumeric(type, 0, nan));
                run.execute();
                assertArrayEquals(rawNumeric(type, negativeZero, nan), run.outputBytes(0));
                assertArrayEquals(rawNumeric(type, nan, negativeZero, negativeZero, nan), run.outputBytes(1));
                assertArrayEquals(rawNumeric(type, nan, nan), run.outputBytes(2));
            }
        }
    }

    @Test void mixedComparisonBodiesBindOriginalCarrierWidthsAndPublishBoolRows() {
        var types = List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16);
        for (var left : types) for (var right : types)
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true))
                try (var run = new RawScan(MetalScanPlanTest.mixedComparisonQuery(left, right, lengths, reverse), null)) {
                    run.bytes(0, numeric(left, 3, lengths ? Float.NaN : 4, 1, lengths ? Float.NaN : 0));
                    run.bytes(1, numeric(right, 2));
                    if (lengths) run.longs(2, 2, 0);
                    run.poisonOutputs(0x5a5a5a5a); run.execute();
                    assertArrayEquals(new byte[] {1, (byte) (lengths ? 0 : 1), 0, 0}, run.outputBytes(0));
                }
    }

    @Test void variadicCompositionInsideScanRetainsAllInputsAndFullOutputRowCoordinates() {
        int arity = 33;
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean stack : List.of(false, true)) for (boolean lengths : List.of(false, true))
                for (boolean reverse : List.of(false, true)) {
                    var query = MetalScanPlanTest.variadicQuery(type, arity, stack, lengths, reverse);
                    try (var run = new RawScan(query, null)) {
                        run.bytes(0, numeric(type, 1, lengths ? Float.NaN : 3, 2, lengths ? Float.NaN : 4));
                        if (lengths) run.longs(1, 2, 0);
                        run.poisonOutputs(0x5a5a5a5a); run.execute();
                        float[] expected = new float[4 * arity];
                        for (int time = 0; time < 2; time++) for (int row = 0; row < 2; row++)
                            Arrays.fill(expected, (time * 2 + row) * arity, (time * 2 + row + 1) * arity,
                                    row == 0 ? time + 1 : lengths ? 0 : time + 3);
                        assertArrayEquals(numeric(type, expected), run.outputBytes(0));
                    }
                }
    }

    @Test void sourceOnlyFloat64ScanPreservesRawCarrySequenceAndInvariantWithoutArithmetic() {
        long negativeZero = 0x8000000000000000L, nan = 0xfff0000000004321L;
        for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
            var query = MetalScanPlanTest.sourceOnlyQuery(DataType.FLOAT64, 2, 2, lengths, reverse);
            try (var run = new RawScan(query, null)) {
                run.longs(0, negativeZero, nan);
                run.longs(1, nan, negativeZero, negativeZero, nan);
                run.longs(2, 1L);
                if (lengths) run.longs(3, 2, 0);
                run.poisonOutputs(0x5a5a5a5a); run.execute();
                assertArrayEquals(longs(negativeZero, nan), run.outputBytes(0));
                assertArrayEquals(longs(nan, lengths ? 0L : negativeZero,
                        negativeZero, lengths ? 0L : nan), run.outputBytes(1));
                assertArrayEquals(longs(1L, 1L), run.outputBytes(2));
                if (lengths) {
                    run.longs(3, 0, 0); run.execute();
                    assertArrayEquals(longs(negativeZero, nan), run.outputBytes(0));
                    assertArrayEquals(new byte[4 * Long.BYTES], run.outputBytes(1));
                    assertArrayEquals(new byte[2 * Long.BYTES], run.outputBytes(2));
                }
            }
        }
    }

    @Test void sourceOnlyLowBindingsPreserveRawPayloadsAcrossCarryHistoryAndInvariant() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                int negativeZero = type == DataType.FLOAT32 ? 0x80000000 : 0x8000;
                int nan = type == DataType.FLOAT32 ? 0x7fc12345 : type == DataType.BFLOAT16 ? 0x7fc5 : 0x7e45;
                var query = MetalScanPlanTest.sourceOnlyQuery(type, 2, 2, lengths, reverse);
                try (var run = new RawScan(query, null)) {
                    byte[] carry = rawNumeric(type, negativeZero, nan);
                    run.bytes(0, carry);
                    run.bytes(1, rawNumeric(type, nan, negativeZero, negativeZero, nan));
                    run.bytes(2, rawNumeric(type, nan));
                    if (lengths) run.longs(3, 2, 0);
                    run.poisonOutputs(0x5a5a5a5a); run.execute();
                    assertArrayEquals(carry, run.outputBytes(0));
                    assertArrayEquals(rawNumeric(type, nan, lengths ? 0 : negativeZero, negativeZero, lengths ? 0 : nan), run.outputBytes(1));
                    assertArrayEquals(rawNumeric(type, nan, nan), run.outputBytes(2));
                    if (lengths) {
                        run.longs(3, 0, 0); run.execute();
                        assertArrayEquals(carry, run.outputBytes(0));
                        assertArrayEquals(new byte[4 * type.byteWidth()], run.outputBytes(1));
                        assertArrayEquals(new byte[2 * type.byteWidth()], run.outputBytes(2));
                    }
                }
            }
    }

    @Test void emptyTimeAndBatchAreRealNativeTypedGeometries() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (long time : new long[] {0, 2}) for (long batch : new long[] {0, 2}) {
                if (time != 0 && batch != 0) continue;
                for (boolean lengths : List.of(false, true)) for (boolean reverse : List.of(false, true))
                    try (var run = new RawScan(MetalScanPlanTest.additiveQuery(type, time, batch, lengths, reverse), null)) {
                        float[] initial = batch == 0 ? new float[0] : new float[] {-0.0f, 5};
                        run.bytes(0, numeric(type, initial));
                        if (lengths) run.longs(2, batch == 0 ? new long[0] : new long[] {0, 0});
                        run.poisonOutputs(0x5a5a5a5a); run.execute();
                        assertArrayEquals(numeric(type, initial), run.outputBytes(0));
                        assertArrayEquals(new byte[0], run.outputBytes(1));
                        assertArrayEquals(new byte[0], run.outputBytes(2));
                    }
            }
    }

    @Test void integralCarryAndHistoryUseControlledOrdinaryModularArithmetic() {
        for (var type : List.of(DataType.INT32, DataType.INT64))
                for (boolean reverse : List.of(false, true)) for (boolean lengths : List.of(false, true)) {
            var query = MetalScanPlanTest.additiveQuery(type, 3, 2, lengths, reverse);
            long high = type == DataType.INT32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
            long low = type == DataType.INT32 ? Integer.MIN_VALUE : Long.MIN_VALUE;
            long[] original = {high, low}, sequence = {1, 2, 3, 4, 5, 6};
            long[] valid = {3, lengths ? 1 : 3};
            long[] carry = original.clone(), history = new long[6], previous = new long[6];
            for (int iteration = 0; iteration < 3; iteration++) {
                int time = reverse ? 2 - iteration : iteration;
                for (int row = 0; row < 2; row++) if (time < valid[row]) {
                    previous[time * 2 + row] = carry[row];
                    long next = carry[row] + sequence[time * 2 + row];
                    carry[row] = type == DataType.INT32 ? (int) next : next;
                    history[time * 2 + row] = carry[row];
                }
            }
            try (var run = new RawScan(query, null)) {
                run.bytes(0, integers(type, original)); run.bytes(1, integers(type, sequence));
                if (lengths) run.longs(2, valid);
                run.execute();
                assertArrayEquals(integers(type, carry), run.outputBytes(0));
                assertArrayEquals(integers(type, history), run.outputBytes(1));
                assertArrayEquals(integers(type, previous), run.outputBytes(2));
                if (lengths) {
                    run.longs(2, 0, 0); run.execute();
                    assertArrayEquals(integers(type, original), run.outputBytes(0));
                    assertArrayEquals(integers(type, new long[6]), run.outputBytes(1));
                    assertArrayEquals(integers(type, new long[6]), run.outputBytes(2));
                }
            }
        }
    }

    @Test void invalidLastLengthPrecedesEveryOutputWriteIncludingZeroTimeAndAllowsRetry() {
        for (long time : new long[] {0, 3}) try (var run = new RawScan(MetalScanPlanTest.additiveQuery(time, 2, true, false), null)) {
            run.floats(0, -0.0f, 5);
            run.longs(2, 0, time + 1);
            run.poisonOutputs(0x7fc12345);
            assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
            for (int output = 0; output < run.outputs.length; output++) {
                int[] sentinel = new int[run.output(output).length]; Arrays.fill(sentinel, 0x7fc12345);
                assertArrayEquals(sentinel, run.output(output), "invalid length changed a Scan-local publication");
            }
            run.longs(2, 0, 0);
            run.execute();
            assertRaw(new float[] {-0.0f, 5}, run.output(0));
            assertArrayEquals(new int[(int) time * 2], run.output(1));
            assertArrayEquals(new int[(int) time * 2], run.output(2));
        }
    }

    @Test void overlappingAxisFoldUsesTheOrdinaryKernelInWholeTensorAndControlledRows() {
        for (boolean lengths : List.of(false, true)) {
            var operation = new Operation(WindowTransformKind.FOLD_AXIS,
                    new FoldAxisAttrs(lengths ? 1 : 0, 4, 1));
            var query = singleEmissionQuery(operation,
                    lengths ? new long[] {2, 2, 3} : new long[] {2, 3},
                    lengths ? new long[] {2, 4} : new long[] {4});
            if (lengths) {
                var inputs = new ArrayList<>(query.inputs());
                inputs.add(MetalScanPlanTest.descriptor(DataType.INT64, 2));
                query = new OperationCapabilityQuery(new Operation(ScanKind.SCAN,
                        new ScanAttrs(2, 0, 1, 0, 1, 0, false, true, 0,
                                List.of(OptionalInt.of(0)))),
                        inputs, query.outputs(), query.scanBody(), Optional.empty());
            }
            try (var run = new RawScan(query, null)) {
                if (lengths) {
                    run.floats(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            13, 14, 15, 16, 17, 18,
                            Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
                    run.longs(1, 2, 1);
                } else run.floats(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
                run.execute();
                assertRaw(lengths
                        ? new float[] {1, 6, 8, 6, 7, 18, 20, 12, 13, 30, 32, 18, 0, 0, 0, 0}
                        : new float[] {1, 6, 8, 6, 7, 18, 20, 12}, run.output(0));
            }
        }
    }

    @Test void emptyAllValidRowsStillPublishNonemptyOrdinarySumIdentities() {
        for (int form = 0; form < 3; form++) try (var run = new RawScan(emptySumQuery(form), null)) {
            run.poisonOutputs(0x7fc12345);
            run.execute();
            assertArrayEquals(new int[form == 1 ? 2 : 6], run.output(0));
        }
        var query = emptySumQuery(0);
        var valid = MetalScanLowering.lower(query, new int[] {0}, new int[] {1});
        long[] words = valid.attributeWords();
        int unit = Math.toIntExact(words[24]);
        int attributes = unit + 14 + Math.toIntExact(words[unit + 3] + words[unit + 4]);
        words[attributes + 3] = 2; // A real out-of-rank reduction axis, despite zero contributors.
        assertThrows(RuntimeException.class, () -> {
            var malformed = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.GENERAL_SCAN,
                    valid.inputs(), valid.outputs(), MetalMpsGraphProgram.AttributeKind.GENERAL_SCAN, words);
            try (var ignored = new RawScan(query, malformed)) {}
        });
    }

    @Test void emptyAdvancedReducersKeepActualAxesAndPublishOwnedEmptyIdentities() {
        for (var kind : List.of(AggregateReductionKind.PROD, AggregateReductionKind.LOG_SUM_EXP,
                AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM,
                AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
            float identity = switch (kind) {
                case PROD -> 1f;
                case LOG_SUM_EXP, MAX -> Float.NEGATIVE_INFINITY;
                case MIN -> Float.POSITIVE_INFINITY;
                default -> 0f;
            };
            for (int axis : new int[] {0, 1}) {
                var query = emptyReductionQuery(kind, new MultiAxisReductionAttrs(List.of(axis), false),
                        axis == 0 ? new long[] {3} : new long[] {0});
                assertTrue(MetalScanLowering.supports(query));
                try (var run = new RawScan(query, null)) {
                    run.poisonOutputs(0x7fc12345); run.execute();
                    float[] expected = new float[axis == 0 ? 6 : 0]; Arrays.fill(expected, identity);
                    assertRaw(expected, run.output(0));
                }
            }
        }
    }

    @Test void emptyStatisticsStillValidateSelectedCountAndCorrectionBeforeSkippingWork() {
        for (var kind : List.of(AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION)) {
            var emptyResults = emptyReductionQuery(kind, new StatisticalReductionAttrs(List.of(1), false, 2), 0);
            assertTrue(MetalScanLowering.supports(emptyResults));
            try (var run = new RawScan(emptyResults, null)) { run.execute(); assertArrayEquals(new int[0], run.output(0)); }
            assertFalse(MetalScanLowering.supports(emptyReductionQuery(
                    kind, new StatisticalReductionAttrs(List.of(1), false, 3), 0)));
            assertFalse(MetalScanLowering.supports(emptyReductionQuery(
                    kind, new StatisticalReductionAttrs(List.of(0), false, 0), 3)));
        }
    }

    @Test void activeIndexFailurePrecedesHistoryWritesAndInactiveIndicesAreNeverInspected() {
        try (var run = new RawScan(indexQuery(false), null)) {
            run.floats(0, 1, 2, 4, 5);
            run.longs(1, 2);
            run.poisonOutputs(0x7fc12345);
            assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
            assertArrayEquals(new int[] {0x7fc12345, 0x7fc12345}, run.output(0));
            run.longs(1, 1); run.execute(); assertRaw(new float[] {2, 5}, run.output(0));
        }
        try (var run = new RawScan(indexQuery(true), null)) {
            run.floats(0, 10, 20, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
            run.longs(1, 1, Long.MIN_VALUE, Long.MAX_VALUE, -9);
            run.longs(2, 1, 0); run.execute(); assertRaw(new float[] {20, 0, 0, 0}, run.output(0));
            run.longs(1, 2, Long.MIN_VALUE, Long.MAX_VALUE, -9); run.poisonOutputs(0x7fc12345);
            assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
            assertArrayEquals(new int[] {0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345}, run.output(0));
            run.longs(2, 0, 0); run.execute(); assertRaw(new float[] {0, 0, 0, 0}, run.output(0));
        }
    }

    @Test void transposedRowsUseProvedCoordinatesForPaddingAndEveryHistory() {
        for (boolean reverse : List.of(false, true)) {
            try (var run = new RawScan(MetalScanPlanTest.transposedQuery(2, 2, reverse), null)) {
                run.floats(0, 1, 2, 3, 4);
                run.floats(1, 10, 20, Float.NaN, Float.NaN, 30, 40, Float.NaN, Float.NaN);
                run.longs(2, 2, 0); run.execute();
                assertRaw(reverse ? new float[] {10, 20, 3, 4} : new float[] {30, 40, 3, 4}, run.output(0));
                assertRaw(new float[] {10, 20, 0, 0, 30, 40, 0, 0}, run.output(1));
                assertRaw(new float[] {10, 0, 20, 0, 30, 0, 40, 0}, run.output(2));
            }
        }
    }

    @Test void nativeRawConstructorIndependentlyRejectsConsistentForgedRowMappings() throws Throwable {
        Path path = library();
        var query = MetalScanPlanTest.transposedQuery(2, 2, false);
        var descriptors = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        query.inputs().forEach(d -> descriptors.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        query.outputs().forEach(d -> descriptors.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
        int[] inputs = {0, 1, 2}, outputs = {3, 4, 5};
        var node = MetalScanLowering.lower(query, inputs, outputs);
        byte[] valid = new MetalMpsGraphProgram(List.of(node)).encodedProgramImage(
                descriptors, inputs, outputs, MetalPreparedRoute.CUSTOM_PROGRAM);
        long[] forgedWords = MetalScanPlanTest.forgedTransposedCoordinates(MetalScanLowering.plan(query));
        byte[] forged = valid.clone();
        int dimensions = descriptors.stream().mapToInt(MetalMpsGraphProgram.ValueDescriptor::rank).sum();
        int references = inputs.length + outputs.length + node.inputCount() + node.outputCount();
        int payload = MetalMpsGraphProgram.HEADER_BYTES
                + descriptors.size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES + dimensions * 2 * Long.BYTES
                + references * Integer.BYTES;
        var image = ByteBuffer.wrap(forged).order(ByteOrder.LITTLE_ENDIAN);
        for (int word = 0; word < forgedWords.length; word++) image.putLong(payload + word * Long.BYTES, forgedWords[word]);
        try (var api = MetalNativeApi.open(path); Arena arena = Arena.ofConfined()) {
            var context = api.createContext();
            try {
                var symbol = SymbolLookup.libraryLookup(path, arena)
                        .find("synaptik_metal_mpsgraph_executable_create").orElseThrow();
                var create = Linker.nativeLinker().downcallHandle(symbol,
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
                for (byte[] bytes : List.of(valid, forged)) {
                    var program = arena.allocate(bytes.length, Long.BYTES);
                    program.copyFrom(MemorySegment.ofArray(bytes));
                    var target = arena.allocate(ADDRESS); target.set(ADDRESS, 0, MemorySegment.NULL);
                    int status = (int) create.invokeExact(context.carrier(), program, bytes.length, target);
                    var carrier = target.get(ADDRESS, 0);
                    try {
                        assertEquals(bytes == valid ? 0 : 1, status);
                        if (bytes != valid) assertEquals(0L, carrier.address(), "raw rejection cannot publish an executable");
                    } finally {
                        if (carrier.address() != 0L) api.releaseExecutable(new MetalNativeApi.Handle(carrier));
                    }
                }
            } finally { api.releaseContext(context); }
        }
    }

    @Test void rowPreservingFeatureReductionSkipsInactiveNanRows() {
        try (var run = new RawScan(rowProductQuery(), null)) {
            run.floats(0, 2, 3, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
            run.longs(1, 1, 0); run.poisonOutputs(0x7fc12345); run.execute();
            assertRaw(new float[] {6, 0, 0, 0}, run.output(0));
            run.longs(1, 0, 0); run.execute(); assertRaw(new float[] {0, 0, 0, 0}, run.output(0));
        }
    }

    @Test void controlledMatmulSingletonUsesOneActiveProductAndSkipsBothInactiveOperands() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            try (var run = new RawScan(terminalMatmulQuery(type, 1), null)) {
                run.bytes(0, numeric(type, -0.0f, Float.NaN));
                run.bytes(1, numeric(type, 1, Float.NaN));
                run.longs(2, 1); run.poisonOutputs(0x5a5a5a5a); run.execute();
                assertArrayEquals(numeric(type, -0.0f, 0.0f), run.outputBytes(0), type.toString());
                run.bytes(0, numeric(type, 0.0f, -0.0f));
                run.bytes(1, numeric(type, -1, 1));
                run.longs(2, 2); run.execute();
                assertArrayEquals(numeric(type, -0.0f, -0.0f), run.outputBytes(0), type.toString());
                run.bytes(0, numeric(type, Float.NaN, Float.NaN));
                run.bytes(1, numeric(type, Float.NaN, Float.NaN));
                run.longs(2, 0); run.execute();
                assertArrayEquals(numeric(type, 0.0f, 0.0f), run.outputBytes(0), type.toString());
            }
            try (var run = new RawScan(terminalMatmulQuery(type, 2), null)) {
                run.bytes(0, numeric(type, Float.NaN, 3, Float.NaN, Float.NaN));
                run.bytes(1, numeric(type, Float.NaN, 2, Float.NaN, Float.NaN));
                run.longs(2, 0, 1); run.poisonOutputs(0x5a5a5a5a); run.execute();
                assertArrayEquals(numeric(type, 6, 0.0f), run.outputBytes(0), type.toString());
            }
            try (var run = new RawScan(terminalMatmulQuery(type, 0), null)) {
                run.poisonOutputs(0x5a5a5a5a); run.execute();
                assertArrayEquals(numeric(type, 0.0f, 0.0f), run.outputBytes(0), type.toString());
            }
        }
    }

    private static OperationCapabilityQuery terminalMatmulQuery(DataType type, long contraction) {
        var operand = MetalScanPlanTest.descriptor(type, contraction);
        var result = MetalScanPlanTest.descriptor(type);
        var left = new ValueId(0); var right = new ValueId(1); var output = new ValueId(2);
        var product = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(left, right), List.of(output));
        var graph = new CompiledGraphModel(List.of(
                new GraphValue(left, operand), new GraphValue(right, operand),
                new GraphValue(output, result)), List.of(product),
                List.of(left, right), List.of(left, output), Map.of(product.id(), GraphPhase.FORWARD));
        // Retain a semantic row emission; only the terminal contraction residual is demanded.
        var body = new CompiledScanBody(graph, List.of(), List.of(left), List.of(output), List.of(1));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,
                new ScanAttrs(2, 0, 2, 0, 1, 1, false, true, 0,
                        List.of(OptionalInt.of(0), OptionalInt.of(0)))),
                List.of(MetalScanPlanTest.descriptor(type, 2, contraction),
                        MetalScanPlanTest.descriptor(type, 2, contraction),
                        MetalScanPlanTest.descriptor(DataType.INT64, contraction)),
                List.of(MetalScanPlanTest.descriptor(type, 2, contraction),
                        MetalScanPlanTest.descriptor(type, 2)), Optional.of(body), Optional.empty());
    }

    @Test void canonicalRnnReverseMaterializesExpandedAccumulatorsAndRowLiterals() {
        library();
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (boolean reverse : List.of(false, true)) {
        Tensor h = tensor(type, false, 2, 3), x = tensor(type, false, 2, 2);
        Tensor w = tensor(type, true, 3, 2), u = tensor(type, true, 3, 3), b = tensor(type, true, 3);
        Tensor initial = tensor(type, false, 2, 3), sequence = tensor(type, false, 2, 2, 2);
        Tensor weight = tensor(type, true, 3, 2), recurrent = tensor(type, true, 3, 3), bias = tensor(type, true, 3);
        Tensor lengths = TensorFactory.create(MetalScanPlanTest.descriptor(DataType.INT64, 2));
        Tensor next = x.matmul(w.permute(1, 0)).add(b).add(h.matmul(u.permute(1, 0))).tanh();
        var body = new ScanBody(List.of(h), List.of(x), List.of(w, u, b), List.of(next), List.of(next));
        Tensor root = Scan.apply(List.of(initial), List.of(sequence), List.of(weight, recurrent, bias),
                lengths, 1, List.of(OptionalInt.of(0)), body, reverse).finalCarry().getFirst();
        Tensor seed = tensor(type, false, 2, 3);
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(root)), List.of(Optional.of(seed)),
                List.of(weight, recurrent, bias), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
        var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"), DeviceClass.ACCELERATOR));
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(root), Optional.of(request),
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                List.of(new MetalCapabilityProvider()), List.of(availability));
        assertTrue(artifacts.graph().nodes().stream().anyMatch(n -> n.operation().kind() == ScanKind.SCAN
                && artifacts.derivatives().derivativeOrderByNode().get(n.id()) == 1));
        var first = new HashMap<TensorId, byte[]>();
        first.put(initial.id(), numeric(type, 0, 0, 0, 0, 0, 0));
        first.put(sequence.id(), numeric(type, 1, 2, 3, 4, 5, 6, Float.NaN, Float.NaN));
        first.put(weight.id(), numeric(type, 0, 0, 0, 0, 0, 0));
        first.put(recurrent.id(), numeric(type, 1, 0, 0, 0, 1, 0, 0, 0, 1));
        first.put(bias.id(), numeric(type, 0, 0, 0));
        first.put(seed.id(), numeric(type, 1, 1, 1, 1, 1, 1));
        first.put(lengths.id(), longs(2, 1));
        var zero = new HashMap<>(first);
        zero.put(sequence.id(), numeric(type, Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                Float.NaN, Float.NaN, Float.NaN, Float.NaN));
        zero.put(lengths.id(), longs(0, 0));
        assertPrepared(type, artifacts, List.of(first, zero, first), List.of(
                List.of(new float[6], new float[] {9, 12, 9, 12, 9, 12}, new float[9], new float[] {3, 3, 3}),
                List.of(new float[6], new float[6], new float[9], new float[3]),
                List.of(new float[6], new float[] {9, 12, 9, 12, 9, 12}, new float[9], new float[] {3, 3, 3})));
        }
    }

    private static OperationCapabilityQuery rowProductQuery() {
        var input = MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 2);
        var output = MetalScanPlanTest.descriptor(DataType.FLOAT32, 2);
        var x = new ValueId(0); var y = new ValueId(1);
        var product = new CompiledNode(new NodeId(0), new Operation(AggregateReductionKind.PROD,
                new AxisReductionAttrs(1, false)), List.of(x), List.of(y));
        var graph = new CompiledGraphModel(List.of(new GraphValue(x, input), new GraphValue(y, output)), List.of(product),
                List.of(x), List.of(y), Map.of(product.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(y), List.of(), List.of(0));
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,
                new ScanAttrs(2, 0, 1, 0, 1, 0, false, true, 0, List.of(OptionalInt.of(0)))),
                List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 2, 2), MetalScanPlanTest.descriptor(DataType.INT64, 2)),
                List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 2)), Optional.of(body), Optional.empty());
    }

    @Test void typedGatherPreflightsActiveIndicesWithoutInspectingInactiveRowsAndAllowsRetry() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            try (var run = new RawScan(indexQuery(type, true), null)) {
                run.bytes(0, numeric(type, 1, 2, Float.NaN, Float.NaN, 3, 4, Float.NaN, Float.NaN));
                run.longs(1, 2, Long.MAX_VALUE, 0, Long.MIN_VALUE);
                run.longs(2, 2, 0);
                run.poisonOutputs(0x5a5a5a5a);
                var failure = assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
                assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
                byte[] poison = new byte[4 * type.byteWidth()]; Arrays.fill(poison, (byte) 0x5a);
                assertArrayEquals(poison, run.outputBytes(0));
                run.longs(1, 1, Long.MAX_VALUE, 0, Long.MIN_VALUE); run.execute();
                assertArrayEquals(numeric(type, 2, 0, 3, 0), run.outputBytes(0));
            }
    }

    private static OperationCapabilityQuery indexQuery(boolean lengths) {
        return indexQuery(DataType.FLOAT32, lengths);
    }

    private static OperationCapabilityQuery indexQuery(DataType type, boolean lengths) {
        var data = MetalScanPlanTest.descriptor(type, lengths ? new long[] {2, 2} : new long[] {2});
        var indices = MetalScanPlanTest.descriptor(DataType.INT64, lengths ? new long[] {2, 1} : new long[] {1});
        var result = MetalScanPlanTest.descriptor(type, lengths ? new long[] {2, 1} : new long[] {1});
        var d = new ValueId(0); var i = new ValueId(1); var y = new ValueId(2);
        var gather = new CompiledNode(new NodeId(0), new Operation(lengths ? AxisGatherKind.GATHER_ELEMENTS : AxisGatherKind.GATHER,
                new IndexAxisAttrs(lengths ? 1 : 0)), List.of(d, i), List.of(y));
        var graph = new CompiledGraphModel(List.of(new GraphValue(d, data), new GraphValue(i, indices), new GraphValue(y, result)),
                List.of(gather), List.of(d, i), List.of(y), Map.of(gather.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(y), List.of(), List.of(0));
        var attrs = lengths ? new ScanAttrs(2, 0, 2, 0, 1, 0, false, true, 0, List.of(OptionalInt.of(0), OptionalInt.of(0)))
                : new ScanAttrs(2, 0, 1, 1, 1, 0, false);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), lengths
                ? List.of(MetalScanPlanTest.descriptor(type, 2, 2, 2), MetalScanPlanTest.descriptor(DataType.INT64, 2, 2, 1),
                        MetalScanPlanTest.descriptor(DataType.INT64, 2))
                : List.of(MetalScanPlanTest.descriptor(type, 2, 2), indices),
                List.of(MetalScanPlanTest.descriptor(type, lengths ? new long[] {2, 2, 1} : new long[] {2, 1})),
                Optional.of(body), Optional.empty());
    }

    private static OperationCapabilityQuery emptySumQuery(int form) {
        long[] outputShape = form == 1 ? new long[0] : form == 2 ? new long[] {1, 3} : new long[] {3};
        OperationAttrs attrs = form == 0 ? new AxisReductionAttrs(0, false) : form == 1 ? NoOperationAttrs.INSTANCE : new SumToShapeAttrs(Shape.of(1, 3));
        return emptyReductionQuery(AggregateReductionKind.SUM, attrs, outputShape);
    }

    private static OperationCapabilityQuery emptyReductionQuery(
            AggregateReductionKind kind, OperationAttrs attrs, long... outputShape) {
        return singleEmissionQuery(new Operation(kind, attrs), new long[] {0, 3}, outputShape);
    }

    @Test void typedEmptyContributorsPublishTheSharedOrdinarySumIdentity() {
        for (var type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            var query = singleEmissionQuery(type, new Operation(AggregateReductionKind.SUM,
                    new AxisReductionAttrs(0, false)), new long[] {0, 3}, 3);
            try (var run = new RawScan(query, null)) {
                run.poisonOutputs(0x5a5a5a5a); run.execute();
                assertArrayEquals(new byte[6 * type.byteWidth()], run.outputBytes(0));
            }
        }
    }

    private static OperationCapabilityQuery singleEmissionQuery(
            Operation operation, long[] inputShape, long... outputShape) {
        return singleEmissionQuery(DataType.FLOAT32, operation, inputShape, outputShape);
    }

    private static OperationCapabilityQuery singleEmissionQuery(
            DataType type, Operation operation, long[] inputShape, long... outputShape) {
        var input = MetalScanPlanTest.descriptor(type, inputShape);
        var output = MetalScanPlanTest.descriptor(type, outputShape);
        var id = new ValueId(0); var result = new ValueId(1);
        var ordinary = new CompiledNode(new NodeId(0), operation, List.of(id), List.of(result));
        var graph = new CompiledGraphModel(List.of(new GraphValue(id, input), new GraphValue(result, output)),
                List.of(ordinary), List.of(id), List.of(result), Map.of(ordinary.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(result), List.of(), List.of(0));
        long[] stackedInput = new long[inputShape.length + 1];
        stackedInput[0] = 2; System.arraycopy(inputShape, 0, stackedInput, 1, inputShape.length);
        long[] stackedOutput = new long[outputShape.length + 1];
        stackedOutput[0] = 2; System.arraycopy(outputShape, 0, stackedOutput, 1, outputShape.length);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,
                new ScanAttrs(2, 0, 1, 0, 1, 0, false)),
                List.of(MetalScanPlanTest.descriptor(type, stackedInput)),
                List.of(MetalScanPlanTest.descriptor(type, stackedOutput)),
                Optional.of(body), Optional.empty());
    }

    private static void assertPrepared(DataType type, CompileArtifacts artifacts, List<Map<TensorId, byte[]>> runs, List<List<float[]>> expected) {
        List<ValueId> publications = new ArrayList<>();
        artifacts.publication().forwardBindings().forEach(b -> publications.add(b.valueId()));
        artifacts.publication().gradientBindings().forEach(b -> publications.add(b.valueId()));
        try (MetalDeviceContext context = MetalDeviceContext.open(library()); Arena arena = Arena.ofConfined()) {
            var analyzed = new AtomicReference<MetalNegPreparationPlan>();
            var preparer = new MetalNegPartitionPreparer();
            var preparation = new PartitionPreparation<>(new MetalNegAnalysisInputs(context), c -> {
                var analysis = preparer.analyze(c); analyzed.set(analysis.plan()); return analysis;
            }, new MetalNegPartitionFinalizer(context));
            var buffers = new ArrayList<BufferRepresentation>();
            try (var execution = GraphPreparation.prepare(artifacts, List.of(preparation), c ->
                    new MetalNegPreparedScheduleAssembler(context, analyzed.get(), publications).assemble(c))) {
                for (var binding : artifacts.constants().bindableInputBindings()) {
                    byte[] bytes = Objects.requireNonNull(runs.getFirst().get(binding.tensorId()));
                    buffers.add(context.createBuffer(bytes.length));
                }
                for (int invocation = 0; invocation < runs.size(); invocation++) {
                    int input = 0;
                    for (var binding : artifacts.constants().bindableInputBindings()) {
                        byte[] bytes = Objects.requireNonNull(runs.get(invocation).get(binding.tensorId()));
                        var source = arena.allocate(bytes.length, 8); MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0, bytes.length);
                        if (bytes.length != 0) ((MetalBufferRepresentation) buffers.get(input)).upload(0, source, 0, bytes.length);
                        input++;
                    }
                    try (var result = new PreparedExecutionRunner().run(execution, buffers)) {
                        for (int output = 0; output < expected.get(invocation).size(); output++) {
                            float[] values = expected.get(invocation).get(output);
                            byte[] represented = numeric(type, values);
                            var target = arena.allocate(represented.length, 8);
                            var representation = (MetalBufferRepresentation) result.publicationRepresentation(output);
                            assertEquals(represented.length, representation.byteSize());
                            if (represented.length != 0) representation.download(0, target, 0, target.byteSize());
                            assertArrayEquals(represented, target.toArray(JAVA_BYTE));
                        }
                    }
                }
            } finally { for (var buffer : buffers) buffer.close(); }
        }
    }

    private static Tensor tensor(DataType type, boolean grad, long... shape) {
        var descriptor = MetalScanPlanTest.descriptor(type, shape);
        return TensorFactory.create(new TensorDescriptor(descriptor.dataType(), descriptor.shape(), descriptor.layout(), grad));
    }
    private static byte[] numeric(DataType type, float... values) {
        var bytes = ByteBuffer.allocate(values.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (float value : values) switch (type) {
            case FLOAT32 -> bytes.putInt(Float.floatToRawIntBits(value));
            case BFLOAT16 -> bytes.putShort(ScalarValue.bfloat16(value).bfloat16Bits());
            case FLOAT16 -> bytes.putShort(Float.floatToFloat16(value));
            default -> throw new IllegalArgumentException("not a numeric Scan carrier");
        }
        return bytes.array();
    }

    private static float rounded(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> value;
            case BFLOAT16 -> Float.intBitsToFloat(Short.toUnsignedInt(ScalarValue.bfloat16(value).bfloat16Bits()) << 16);
            case FLOAT16 -> Float.float16ToFloat(Float.floatToFloat16(value));
            default -> throw new IllegalArgumentException("not a numeric Scan carrier");
        };
    }

    private static byte[] rawNumeric(DataType type, int... words) {
        var bytes = ByteBuffer.allocate(words.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (int word : words) {
            if (type == DataType.FLOAT32) bytes.putInt(word); else bytes.putShort((short) word);
        }
        return bytes.array();
    }

    private static byte[] integers(DataType type, long... values) {
        var bytes = ByteBuffer.allocate(values.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (long value : values) {
            if (type == DataType.INT32) bytes.putInt((int) value); else bytes.putLong(value);
        }
        return bytes.array();
    }

    private static byte[] floats(float... values) { var bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.nativeOrder()); for (float value : values) bytes.putInt(Float.floatToRawIntBits(value)); return bytes.array(); }
    private static byte[] longs(long... values) { var bytes = ByteBuffer.allocate(values.length * 8).order(ByteOrder.nativeOrder()); for (long value : values) bytes.putLong(value); return bytes.array(); }
    private static void assertRaw(float[] expected, int[] actual) { int[] bits = new int[expected.length]; for (int i = 0; i < bits.length; i++) bits[i] = Float.floatToRawIntBits(expected[i]); assertArrayEquals(bits, actual); }
    private static Path library() { String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY"); assumeTrue(configured != null && !configured.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set"); return Path.of(configured).toAbsolutePath(); }

    static final class RawScan implements AutoCloseable {
        private final MetalNativeApi api;
        private MetalNativeApi.Handle context, executable;
        private final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        private final List<MetalMpsGraphProgram.ValueDescriptor> descriptors;
        private final int[] outputs;
        private final int[] outputSizes;
        RawScan(OperationCapabilityQuery query, MetalMpsGraphProgram.Node replacement) {
            api = MetalNativeApi.open(library());
            var list = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
            query.inputs().forEach(d -> list.add(MetalMpsGraphProgram.ValueDescriptor.from(d)));
            var demands = query.scanBody().orElseThrow().demandedOutputIndices();
            demands.forEach(p -> list.add(MetalMpsGraphProgram.ValueDescriptor.from(query.outputs().get(p)))); descriptors = List.copyOf(list);
            int[] inputs = java.util.stream.IntStream.range(0, query.inputs().size()).toArray();
            outputs = java.util.stream.IntStream.range(query.inputs().size(), descriptors.size()).toArray();
            outputSizes = new int[outputs.length];
            for (int i = 0; i < outputs.length; i++) outputSizes[i] = Math.toIntExact(descriptors.get(outputs[i]).byteCount());
            var node = replacement == null ? MetalScanLowering.lower(query, inputs, outputs) : replacement;
            var program = new MetalMpsGraphProgram(List.of(node));
            try {
                context = api.createContext(); executable = api.createProgramExecutable(context, descriptors, program, inputs, outputs, MetalPreparedRoute.CUSTOM_PROGRAM);
                for (var descriptor : descriptors) buffers.add(api.createBuffer(context, Math.max(8, descriptor.byteCount())));
                buffers.add(api.createBuffer(context, MetalProgramWorkspace.scratchBytes(node, descriptors)));
            } catch (RuntimeException failure) { close(); throw failure; }
        }
        void floats(int ordinal, float... values) { upload(ordinal, MetalGeneralScanNativeTest.floats(values)); }
        void longs(int ordinal, long... values) { upload(ordinal, MetalGeneralScanNativeTest.longs(values)); }
        void bytes(int ordinal, byte[] values) { upload(ordinal, values); }
        byte[] outputBytes(int index) { if (outputSizes[index] == 0) return new byte[0]; try (Arena arena = Arena.ofConfined()) { var target = arena.allocate(outputSizes[index], 8); api.download(buffers.get(outputs[index]), 0, target, target.byteSize()); return target.toArray(JAVA_BYTE); } }
        private void upload(int ordinal, byte[] bytes) { if (bytes.length == 0) return; try (Arena arena = Arena.ofConfined()) { var source = arena.allocate(bytes.length, 8); MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0, bytes.length); api.upload(buffers.get(ordinal), 0, source, bytes.length); } }
        void poisonOutputs(int word) { for (int i = 0; i < outputs.length; i++) { var bytes = new byte[outputSizes[i]]; for (int b = 0; b < bytes.length; b++) bytes[b] = (byte) (word >>> ((b & 3) * 8)); upload(outputs[i], bytes); } }
        int[] output(int index) { if (outputSizes[index] == 0) return new int[0]; try (Arena arena = Arena.ofConfined()) { var target = arena.allocate(outputSizes[index], 4); api.download(buffers.get(outputs[index]), 0, target, target.byteSize()); return target.toArray(JAVA_INT); } }
        void execute() { try (Arena arena = Arena.ofConfined()) { var table = arena.allocate(ADDRESS, buffers.size()); for (int i = 0; i < buffers.size(); i++) table.setAtIndex(ADDRESS, i, buffers.get(i).carrier()); var out = arena.allocate(ADDRESS, outputs.length); for (int i = 0; i < outputs.length; i++) out.setAtIndex(ADDRESS, i, buffers.get(outputs[i]).carrier()); api.runExecutable(executable, buffers.size(), table, outputs.length, out); } }
        @Override public void close() { for (int i = buffers.size() - 1; i >= 0; i--) api.releaseBuffer(buffers.get(i)); buffers.clear(); if (executable != null) api.releaseExecutable(executable); executable = null; if (context != null) api.releaseContext(context); context = null; api.close(); }
    }
}

package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.scan.*;
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

    @Test void canonicalRnnReverseMaterializesExpandedAccumulatorsAndRowLiterals() {
        library();
        Tensor h = tensor(false, 2, 3), x = tensor(false, 2, 2);
        Tensor w = tensor(true, 3, 2), u = tensor(true, 3, 3), b = tensor(true, 3);
        Tensor initial = tensor(false, 2, 3), sequence = tensor(false, 2, 2, 2);
        Tensor weight = tensor(true, 3, 2), recurrent = tensor(true, 3, 3), bias = tensor(true, 3);
        Tensor lengths = TensorFactory.create(MetalScanPlanTest.descriptor(DataType.INT64, 2));
        Tensor next = x.matmul(w.permute(1, 0)).add(b).add(h.matmul(u.permute(1, 0))).tanh();
        var body = new ScanBody(List.of(h), List.of(x), List.of(w, u, b), List.of(next), List.of(next));
        Tensor root = Scan.apply(List.of(initial), List.of(sequence), List.of(weight, recurrent, bias),
                lengths, 1, List.of(OptionalInt.of(0)), body, false).finalCarry().getFirst();
        Tensor seed = tensor(false, 2, 3);
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
        first.put(initial.id(), floats(0, 0, 0, 0, 0, 0));
        first.put(sequence.id(), floats(1, 2, 3, 4, 5, 6, Float.NaN, Float.NaN));
        first.put(weight.id(), floats(0, 0, 0, 0, 0, 0));
        first.put(recurrent.id(), floats(1, 0, 0, 0, 1, 0, 0, 0, 1));
        first.put(bias.id(), floats(0, 0, 0));
        first.put(seed.id(), floats(1, 1, 1, 1, 1, 1));
        first.put(lengths.id(), longs(2, 1));
        var zero = new HashMap<>(first);
        zero.put(sequence.id(), floats(Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                Float.NaN, Float.NaN, Float.NaN, Float.NaN));
        zero.put(lengths.id(), longs(0, 0));
        assertPrepared(artifacts, List.of(first, zero, first), List.of(
                List.of(new float[6], new float[] {9, 12, 9, 12, 9, 12}, new float[9], new float[] {3, 3, 3}),
                List.of(new float[6], new float[6], new float[9], new float[3]),
                List.of(new float[6], new float[] {9, 12, 9, 12, 9, 12}, new float[9], new float[] {3, 3, 3})));
    }

    @Test void compilerActualReverseUsesControlledSumAndMatmulAndCanBeReused() {
        library();
        for (boolean matrix : List.of(false, true)) for (boolean reverse : List.of(false, true))
                for (boolean lengthMode : List.of(false, true)) {
            for (int[] geometry : new int[][] {{3, 2}, {0, 2}, {3, 0}}) {
                int time = geometry[0], batch = geometry[1];
                Tensor h = tensor(true, batch, 1), x = tensor(true, batch, 1);
                Tensor w = tensor(true, matrix ? new long[] {1, 1} : new long[] {1});
                Tensor initial = tensor(true, batch, 1), sequence = tensor(true, time, batch, 1);
                Tensor weight = tensor(true, matrix ? new long[] {1, 1} : new long[] {1});
                Tensor lengths = TensorFactory.create(MetalScanPlanTest.descriptor(DataType.INT64, batch));
                Tensor next = h.add(matrix ? x.matmul(w) : x.mul(w));
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of());
                Tensor root = (lengthMode ? Scan.apply(List.of(initial), List.of(sequence), List.of(weight), lengths, 1,
                        List.of(OptionalInt.of(0)), body, reverse) : Scan.apply(List.of(initial), List.of(sequence),
                        List.of(weight), body, reverse)).finalCarry().getFirst();
                Tensor seed = tensor(false, batch, 1);
                var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                        List.of(new FunctionalGradientRequest.ForwardTensorReference(root)), List.of(Optional.of(seed)),
                        List.of(initial, sequence, weight), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
                var availability = new BackendAvailabilitySnapshot(MetalCapabilityProvider.METAL_BACKEND_ID,
                        Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"), DeviceClass.ACCELERATOR));
                var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(root), Optional.of(request),
                        GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(),
                        List.of(new MetalCapabilityProvider()), List.of(availability));
                if (time > 0 && batch > 0) assertTrue(artifacts.graph().nodes().stream().anyMatch(n -> n.operation().kind() == ScanKind.SCAN
                        && artifacts.derivatives().derivativeOrderByNode().get(n.id()) == 1), "expected actual Compiler-generated reverse body");
                float poison = Float.intBitsToFloat(0x7fc12345);
                float[] initialValues = batch == 0 ? new float[0] : new float[] {1, -0.0f};
                float[] seedValues = batch == 0 ? new float[0] : new float[] {1, lengthMode ? poison : 1};
                float[] sequenceValues = new float[time * batch]; Arrays.fill(sequenceValues, poison);
                for (int t = 0; t < time && batch > 0; t++) {
                    sequenceValues[t * batch] = t + 1;
                    if (!lengthMode) sequenceValues[t * batch + 1] = (t + 1) * 10;
                }
                long[] active = batch == 0 ? new long[0] : new long[] {time, 0};
                var data = new HashMap<TensorId, byte[]>();
                data.put(initial.id(), floats(initialValues)); data.put(sequence.id(), floats(sequenceValues));
                data.put(weight.id(), floats(2)); data.put(lengths.id(), longs(active)); data.put(seed.id(), floats(seedValues));
                float[] forward = initialValues.clone();
                if (batch > 0) { forward[0] += time * (time + 1); if (!lengthMode && time > 0) forward[1] = time * (time + 1) * 10; }
                float[] dSequence = new float[time * batch];
                for (int t = 0; t < time && batch > 0; t++) { dSequence[t * batch] = 2; if (!lengthMode) dSequence[t * batch + 1] = 2; }
                float dWeight = batch == 0 ? 0 : time * (time + 1) / 2f * (lengthMode ? 1 : 11);
                List<float[]> expected = List.of(forward, seedValues, dSequence, new float[] {dWeight});
                var changed = new HashMap<>(data);
                List<float[]> changedExpected;
                if (lengthMode) {
                    changed.put(weight.id(), floats(poison)); changed.put(lengths.id(), longs(new long[batch]));
                    changedExpected = List.of(initialValues, seedValues, new float[time * batch], new float[] {0});
                } else {
                    changed.put(weight.id(), floats(3));
                    float[] changedForward = initialValues.clone(), changedSequence = new float[time * batch];
                    if (batch > 0) {
                        changedForward[0] += time * (time + 1) * 1.5f;
                        if (time > 0) changedForward[1] = time * (time + 1) * 15;
                    }
                    Arrays.fill(changedSequence, 3);
                    changedExpected = List.of(changedForward, seedValues, changedSequence, new float[] {dWeight});
                }
                assertPrepared(artifacts, List.of(data, changed), List.of(expected, changedExpected));
            }
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

    private static OperationCapabilityQuery indexQuery(boolean lengths) {
        var data = MetalScanPlanTest.descriptor(DataType.FLOAT32, lengths ? new long[] {2, 2} : new long[] {2});
        var indices = MetalScanPlanTest.descriptor(DataType.INT64, lengths ? new long[] {2, 1} : new long[] {1});
        var result = MetalScanPlanTest.descriptor(DataType.FLOAT32, lengths ? new long[] {2, 1} : new long[] {1});
        var d = new ValueId(0); var i = new ValueId(1); var y = new ValueId(2);
        var gather = new CompiledNode(new NodeId(0), new Operation(lengths ? AxisGatherKind.GATHER_ELEMENTS : AxisGatherKind.GATHER,
                new IndexAxisAttrs(lengths ? 1 : 0)), List.of(d, i), List.of(y));
        var graph = new CompiledGraphModel(List.of(new GraphValue(d, data), new GraphValue(i, indices), new GraphValue(y, result)),
                List.of(gather), List.of(d, i), List.of(y), Map.of(gather.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(y), List.of(), List.of(0));
        var attrs = lengths ? new ScanAttrs(2, 0, 2, 0, 1, 0, false, true, 0, List.of(OptionalInt.of(0), OptionalInt.of(0)))
                : new ScanAttrs(2, 0, 1, 1, 1, 0, false);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, attrs), lengths
                ? List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 2, 2), MetalScanPlanTest.descriptor(DataType.INT64, 2, 2, 1),
                        MetalScanPlanTest.descriptor(DataType.INT64, 2))
                : List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 2), indices),
                List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, lengths ? new long[] {2, 2, 1} : new long[] {2, 1})),
                Optional.of(body), Optional.empty());
    }

    private static OperationCapabilityQuery emptySumQuery(int form) {
        long[] outputShape = form == 1 ? new long[0] : form == 2 ? new long[] {1, 3} : new long[] {3};
        OperationAttrs attrs = form == 0 ? new AxisReductionAttrs(0, false) : form == 1 ? NoOperationAttrs.INSTANCE : new SumToShapeAttrs(Shape.of(1, 3));
        return emptyReductionQuery(AggregateReductionKind.SUM, attrs, outputShape);
    }

    private static OperationCapabilityQuery emptyReductionQuery(
            AggregateReductionKind kind, OperationAttrs attrs, long... outputShape) {
        var input = MetalScanPlanTest.descriptor(DataType.FLOAT32, 0, 3);
        var output = MetalScanPlanTest.descriptor(DataType.FLOAT32, outputShape);
        var id = new ValueId(0); var result = new ValueId(1);
        var reduce = new CompiledNode(new NodeId(0), new Operation(kind, attrs), List.of(id), List.of(result));
        var graph = new CompiledGraphModel(List.of(new GraphValue(id, input), new GraphValue(result, output)), List.of(reduce),
                List.of(id), List.of(result), Map.of(reduce.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(result), List.of(), List.of(0));
        long[] stacked = new long[outputShape.length + 1]; stacked[0] = 2; System.arraycopy(outputShape, 0, stacked, 1, outputShape.length);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(2, 0, 1, 0, 1, 0, false)),
                List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, 2, 0, 3)), List.of(MetalScanPlanTest.descriptor(DataType.FLOAT32, stacked)), Optional.of(body), Optional.empty());
    }

    private static void assertPrepared(CompileArtifacts artifacts, List<Map<TensorId, byte[]>> runs, List<List<float[]>> expected) {
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
                            var target = arena.allocate((long) values.length * 4, 4);
                            var representation = (MetalBufferRepresentation) result.publicationRepresentation(output);
                            assertEquals((long) values.length * Float.BYTES, representation.byteSize());
                            if (values.length != 0) representation.download(0, target, 0, target.byteSize());
                            int[] actual = new int[values.length]; for (int i = 0; i < actual.length; i++) actual[i] = target.getAtIndex(JAVA_INT, i);
                            assertRaw(values, actual);
                        }
                    }
                }
            } finally { for (var buffer : buffers) buffer.close(); }
        }
    }

    private static Tensor tensor(boolean grad, long... shape) {
        var descriptor = MetalScanPlanTest.descriptor(DataType.FLOAT32, shape);
        return TensorFactory.create(new TensorDescriptor(descriptor.dataType(), descriptor.shape(), descriptor.layout(), grad));
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
                buffers.add(api.createBuffer(context, MetalScanLowering.plan(query).workspaceBytes()));
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

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
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
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
        var malformed = MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.GENERAL_SCAN,
                valid.inputs(), valid.outputs(), MetalMpsGraphProgram.AttributeKind.GENERAL_SCAN, words);
        assertThrows(RuntimeException.class, () -> { try (var ignored = new RawScan(query, malformed)) {} });
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
            run.floats(0, 10, 20, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN);
            run.longs(1, 1, Long.MIN_VALUE, Long.MAX_VALUE, -9);
            run.longs(2, 1, 0); run.execute(); assertRaw(new float[] {20, 0, 0, 0}, run.output(0));
            run.longs(1, 2, Long.MIN_VALUE, Long.MAX_VALUE, -9); run.poisonOutputs(0x7fc12345);
            assertThrows(MetalNativeApi.NativeFailure.class, run::execute);
            assertArrayEquals(new int[] {0x7fc12345, 0x7fc12345, 0x7fc12345, 0x7fc12345}, run.output(0));
            run.longs(2, 0, 0); run.execute(); assertRaw(new float[] {0, 0, 0, 0}, run.output(0));
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
        var input = MetalScanPlanTest.descriptor(DataType.FLOAT32, 0, 3);
        long[] outputShape = form == 1 ? new long[0] : form == 2 ? new long[] {1, 3} : new long[] {3};
        var output = MetalScanPlanTest.descriptor(DataType.FLOAT32, outputShape);
        OperationAttrs attrs = form == 0 ? new AxisReductionAttrs(0, false) : form == 1 ? NoOperationAttrs.INSTANCE : new SumToShapeAttrs(Shape.of(1, 3));
        var id = new ValueId(0); var result = new ValueId(1);
        var sum = new CompiledNode(new NodeId(0), new Operation(AggregateReductionKind.SUM, attrs), List.of(id), List.of(result));
        var graph = new CompiledGraphModel(List.of(new GraphValue(id, input), new GraphValue(result, output)), List.of(sum),
                List.of(id), List.of(result), Map.of(sum.id(), GraphPhase.FORWARD));
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
            var buffers = new ArrayList<MetalBufferRepresentation>();
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
                        if (bytes.length != 0) buffers.get(input).upload(0, source, 0, bytes.length);
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

    private static final class RawScan implements AutoCloseable {
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
        private void upload(int ordinal, byte[] bytes) { if (bytes.length == 0) return; try (Arena arena = Arena.ofConfined()) { var source = arena.allocate(bytes.length, 8); MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0, bytes.length); api.upload(buffers.get(ordinal), 0, source, bytes.length); } }
        void poisonOutputs(int word) { for (int i = 0; i < outputs.length; i++) { var bytes = ByteBuffer.allocate(outputSizes[i]).order(ByteOrder.nativeOrder()); while (bytes.hasRemaining()) bytes.putInt(word); upload(outputs[i], bytes.array()); } }
        int[] output(int index) { if (outputSizes[index] == 0) return new int[0]; try (Arena arena = Arena.ofConfined()) { var target = arena.allocate(outputSizes[index], 4); api.download(buffers.get(outputs[index]), 0, target, target.byteSize()); return target.toArray(JAVA_INT); } }
        void execute() { try (Arena arena = Arena.ofConfined()) { var table = arena.allocate(ADDRESS, buffers.size()); for (int i = 0; i < buffers.size(); i++) table.setAtIndex(ADDRESS, i, buffers.get(i).carrier()); var out = arena.allocate(ADDRESS, outputs.length); for (int i = 0; i < outputs.length; i++) out.setAtIndex(ADDRESS, i, buffers.get(outputs[i]).carrier()); api.runExecutable(executable, buffers.size(), table, outputs.length, out); } }
        @Override public void close() { for (int i = buffers.size() - 1; i >= 0; i--) api.releaseBuffer(buffers.get(i)); buffers.clear(); if (executable != null) api.releaseExecutable(executable); executable = null; if (context != null) api.releaseContext(context); context = null; api.close(); }
    }
}

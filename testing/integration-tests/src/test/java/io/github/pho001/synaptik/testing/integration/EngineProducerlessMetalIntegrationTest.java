package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** Real Metal-only public source publications, without a registered CPU composition. */
final class EngineProducerlessMetalIntegrationTest {
    @Test
    void reverseScanPublishesCallerAndConstantGradientsForEmptyAndInactiveRows() {
        try (Engine engine = metalOnly(); Arena caller = Arena.ofConfined()) {
            for (boolean matrix : List.of(false, true)) for (boolean reverse : List.of(false, true))
                    for (boolean lengthMode : List.of(false, true))
                        for (int[] geometry : new int[][] {{3, 2}, {0, 2}, {3, 0}}) {
                int time = geometry[0], batch = geometry[1];
                Tensor h = tensor(true, batch, 1), x = tensor(true, batch, 1);
                Tensor w = tensor(true, matrix ? new long[] {1, 1} : new long[] {1});
                Tensor initial = tensor(true, batch, 1), sequence = tensor(true, time, batch, 1);
                Tensor weight = tensor(true, matrix ? new long[] {1, 1} : new long[] {1});
                Tensor lengths = TensorFactory.create(descriptor(DataType.INT64, Shape.of(batch), false));
                Tensor next = h.add(matrix ? x.matmul(w) : x.mul(w));
                var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of());
                Tensor root = (lengthMode
                        ? Scan.apply(List.of(initial), List.of(sequence), List.of(weight), lengths, 1,
                                List.of(OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(initial), List.of(sequence), List.of(weight), body, reverse))
                        .finalCarry().getFirst();
                Tensor seed = tensor(false, batch, 1);
                CompiledGraph compiled = EngineCpuGeneralScanIntegrationTest.compileScanGradients(
                        engine, List.of(root), List.of(seed), List.of(initial, sequence, weight),
                        time == 0 || batch == 0);
                float poison = Float.intBitsToFloat(0x7fc12345);
                float[] initialValues = batch == 0 ? new float[0] : new float[] {1, -0.0f};
                float[] seedValues = batch == 0 ? new float[0] : new float[] {1, lengthMode ? poison : 1};
                float[] sequenceValues = new float[time * batch]; Arrays.fill(sequenceValues, poison);
                for (int t = 0; t < time && batch > 0; t++) {
                    sequenceValues[t * batch] = t + 1;
                    if (!lengthMode) sequenceValues[t * batch + 1] = (t + 1) * 10;
                }
                long[] active = batch == 0 ? new long[0] : new long[] {time, 0};
                var data = new HashMap<TensorId, Tensor>();
                data.put(initial.id(), stored(caller, initial.descriptor(), floatWords(initialValues), false));
                data.put(sequence.id(), stored(caller, sequence.descriptor(), floatWords(sequenceValues), false));
                data.put(weight.id(), stored(caller, weight.descriptor(), floatWords(2), false));
                data.put(lengths.id(), stored(caller, lengths.descriptor(), active, false));
                data.put(seed.id(), stored(caller, seed.descriptor(), floatWords(seedValues), false));
                float[] forward = initialValues.clone();
                if (batch > 0) {
                    forward[0] += time * (time + 1);
                    if (!lengthMode && time > 0) forward[1] = time * (time + 1) * 10;
                }
                float[] dSequence = new float[time * batch];
                for (int t = 0; t < time && batch > 0; t++) {
                    dSequence[t * batch] = 2;
                    if (!lengthMode) dSequence[t * batch + 1] = 2;
                }
                float dWeight = batch == 0 ? 0 : time * (time + 1) / 2f * (lengthMode ? 1 : 11);
                List<float[]> expected = List.of(forward, seedValues, dSequence, new float[] {dWeight});
                var changed = new HashMap<>(data);
                List<float[]> changedExpected;
                if (lengthMode) {
                    changed.put(weight.id(), stored(caller, weight.descriptor(), floatWords(poison), false));
                    changed.put(lengths.id(), stored(caller, lengths.descriptor(), new long[batch], false));
                    changedExpected = List.of(initialValues, seedValues, new float[time * batch], new float[] {0});
                } else {
                    changed.put(weight.id(), stored(caller, weight.descriptor(), floatWords(3), false));
                    float[] changedForward = initialValues.clone(), changedSequence = new float[time * batch];
                    if (batch > 0) {
                        changedForward[0] += time * (time + 1) * 1.5f;
                        if (time > 0) changedForward[1] = time * (time + 1) * 15;
                    }
                    Arrays.fill(changedSequence, 3);
                    changedExpected = List.of(changedForward, seedValues, changedSequence, new float[] {dWeight});
                }
                try (var prepared = engine.prepare(compiled)) {
                    var runs = List.of(data, changed);
                    var outputs = List.of(expected, changedExpected);
                    for (int invocation = 0; invocation < runs.size(); invocation++) {
                        try (var result = engine.runValues(prepared, ordered(compiled, runs.get(invocation)))) {
                            for (int output = 0; output < outputs.get(invocation).size(); output++) {
                                float[] values = outputs.get(invocation).get(output);
                                assertArrayEquals(floatWords(values), raw(result.materialize(
                                        result.publications().get(output), (long) values.length * Float.BYTES)));
                            }
                        }
                    }
                }
            }
        }
    }

    private static Tensor tensor(boolean grad, long... shape) {
        return TensorFactory.create(descriptor(DataType.FLOAT32, Shape.of(shape), grad));
    }

    private static long[] floatWords(float... values) {
        long[] words = new long[values.length];
        for (int index = 0; index < values.length; index++) words[index] = bits(DataType.FLOAT32, values[index]);
        return words;
    }

    @Test
    void scalarObjectiveCastContiguousAndScalarAddSubtractPublishConstantGradients() {
        try (Engine engine = metalOnly(); Arena caller = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                Tensor target = stored(caller, descriptor(type, Shape.scalar(), true),
                        new long[] {bits(type, 2.0f)}, false);
                DataType castType = type == DataType.FLOAT32 ? DataType.FLOAT16 : DataType.FLOAT32;
                var objectives = new ArrayList<>(List.of(target.cast(castType), target.contiguous()));
                if (type == DataType.FLOAT32) {
                    objectives.add(target.add(scalar(type, 3.0f)));
                    objectives.add(target.sub(scalar(type, 3.0f)));
                }
                for (int index = 0; index < objectives.size(); index++) {
                    for (int pass = 0; pass < 2; pass++) {
                        var result = engine.backward(objectives.get(index), List.of(target), 32);
                        assertArrayEquals(new long[] {bits(type, 1.0f)}, raw(result.gradients().getFirst()),
                                type + " objective=" + index + " pass=" + pass);
                        float expected = index == 2 ? 5.0f : index == 3 ? -1.0f : 2.0f;
                        assertArrayEquals(new long[] {bits(result.objective().dataType(), expected)},
                                raw(result.objective()));
                    }
                }
                assertTrue(target.hostStorage().orElseThrow().isAlive());
                assertEquals(bits(type, 2.0f), read(target.hostStorage().orElseThrow().segment(), type, 0));
            }
        }
    }

    @Test
    void preparedScalarIdentityVjpsPublishCallerSeedsWithExactBitsAndIndependentResults() {
        try (Engine engine = metalOnly(); Arena caller = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                Tensor target = TensorFactory.create(descriptor(type, Shape.scalar(), true));
                Tensor seed = TensorFactory.create(descriptor(type, Shape.scalar(), false));
                var objectives = new ArrayList<>(List.of(target.contiguous()));
                if (type == DataType.FLOAT32) {
                    objectives.add(target.add(scalar(type, 3.0f)));
                    objectives.add(target.sub(scalar(type, 3.0f)));
                }
                for (Tensor objective : objectives) {
                    CompiledGraph compiled = engine.compile(List.of(objective), List.of(seed), List.of(target));
                    try (var prepared = engine.prepare(compiled)) {
                        Tensor actualTarget = stored(caller, target.descriptor(), new long[] {bits(type, 2.0f)}, false);
                        long firstBits = patterns(type, 1, 0)[0];
                        long secondBits = patterns(type, 1, 1)[0];
                        Tensor firstSeed = stored(caller, seed.descriptor(), new long[] {firstBits}, true);
                        Tensor secondSeed = stored(caller, seed.descriptor(), new long[] {secondBits}, false);
                        try (var first = engine.runValues(prepared, ordered(compiled,
                                     Map.of(target.id(), actualTarget, seed.id(), firstSeed)));
                                var second = engine.runValues(prepared, ordered(compiled,
                                     Map.of(target.id(), actualTarget, seed.id(), secondSeed)))) {
                            for (var result : List.of(first, second)) {
                                RunResult.Publication gradient = result.publications().get(1);
                                assertEquals(target.id(), gradient.tensorId());
                                assertEquals(RunResult.Role.GRADIENT, gradient.role());
                                assertEquals(OptionalInt.of(1), gradient.derivativeOrder());
                                assertEquals(OptionalInt.of(0), gradient.targetIndex());
                                assertFalse(gradient.descriptor().requiresGrad());
                            }
                            assertArrayEquals(new long[] {firstBits}, raw(first.materialize(first.publications().get(1), 8)));
                            assertArrayEquals(new long[] {secondBits}, raw(second.materialize(second.publications().get(1), 8)));
                            assertArrayEquals(new long[] {firstBits}, raw(first.materialize(first.publications().get(1), 8)));
                        }
                        assertTrue(firstSeed.hostStorage().orElseThrow().isAlive());
                        assertTrue(secondSeed.hostStorage().orElseThrow().isAlive());
                    }
                }
            }
        }
    }

    @Test
    void sourceOnlyViewsAndEmptyBoundsPreserveAllSevenRawCarriersAndCallerOwnership() {
        List<View> views = List.of(
                new View(Shape.of(2, 3), new long[] {5, 2}, 1, new int[] {1, 3, 5, 6, 8, 10}),
                new View(Shape.of(2, 3), new long[] {0, 1}, 2, new int[] {2, 3, 4, 2, 3, 4}),
                new View(Shape.scalar(), new long[0], 3, new int[] {3}),
                new View(Shape.of(0, 3), new long[] {5, 1}, 7, new int[0]),
                new View(Shape.of(0, 3), new long[] {Long.MAX_VALUE, 1}, Long.MAX_VALUE, new int[0]),
                new View(Shape.of(2, 0, 3), new long[] {3, 3, 1}, 0, new int[0]));
        try (Engine engine = metalOnly()) {
            for (DataType type : DataType.values()) for (View view : views) {
                LayoutDescriptor layout = LayoutDescriptor.of(view.shape(), view.strides(), view.offset(), true);
                TensorDescriptor descriptor = new TensorDescriptor(type, view.shape(), Optional.of(layout), false);
                Tensor source = TensorFactory.create(descriptor);
                Tensor work = TensorFactory.create(descriptor(DataType.FLOAT32, Shape.of(2), false));
                CompiledGraph compiled = engine.compile(List.of(source, work.neg()));
                assertEquals(List.of(source.id(), work.id()),
                        compiled.inputs().stream().map(CompiledGraph.Input::tensorId).toList());
                try (var prepared = engine.prepare(compiled); Arena caller = Arena.ofConfined()) {
                    for (int pass = 0; pass < 3; pass++) {
                        // Exercise exact capacity (including zero), then unused caller capacity.
                        long[] physical = patterns(type, Math.toIntExact(layout.referencedElementSpan())
                                + (pass == 0 ? 0 : 2), pass);
                        if (type == DataType.BOOL) {
                            boolean[] referenced = new boolean[physical.length];
                            for (int address : view.addresses()) referenced[address] = true;
                            for (int index = 0; index < physical.length; index++) {
                                if (!referenced[index]) physical[index] = 0x7f;
                            }
                        }
                        Tensor actualSource = stored(caller, descriptor, physical, pass == 1);
                        Tensor actualWork = stored(caller, work.descriptor(),
                                new long[] {bits(DataType.FLOAT32, 2.0f), bits(DataType.FLOAT32, -3.0f)}, false);
                        var supplied = Map.of(source.id(), actualSource, work.id(), actualWork);
                        long[] expected = Arrays.stream(view.addresses()).mapToLong(address -> physical[address]).toArray();
                        try (var result = engine.runValues(prepared, ordered(compiled, supplied))) {
                            var publication = result.publications().getFirst();
                            assertEquals(descriptor, publication.descriptor());
                            assertArrayEquals(expected, raw(result.materialize(publication,
                                    (long) expected.length * type.byteWidth())), type + " " + view.shape());
                            assertArrayEquals(new long[] {bits(DataType.FLOAT32, -2.0f), bits(DataType.FLOAT32, 3.0f)},
                                    raw(result.materialize(result.publications().get(1), 8)));
                            if (expected.length > 0) assertThrows(IllegalArgumentException.class,
                                    () -> result.materialize(publication, (long) expected.length * type.byteWidth() - 1));
                        }
                        assertTrue(actualSource.hostStorage().orElseThrow().isAlive());
                        for (int index = 0; index < physical.length; index++) {
                            assertEquals(physical[index], read(actualSource.hostStorage().orElseThrow().segment(), type, index));
                        }
                    }
                }
            }
        }
    }

    @Test
    void sourceStorageFailureAndDescriptorMismatchLeavePreparedRecipeReusable() {
        try (Engine engine = metalOnly(); Arena caller = Arena.ofConfined()) {
            TensorDescriptor descriptor = new TensorDescriptor(DataType.INT64, Shape.of(2),
                    Optional.of(LayoutDescriptor.of(Shape.of(2), new long[] {2}, 1, true)), false);
            Tensor source = TensorFactory.create(descriptor);
            Tensor work = TensorFactory.create(descriptor(DataType.FLOAT32, Shape.of(2), false));
            CompiledGraph compiled = engine.compile(List.of(source, work.neg()));
            try (var prepared = engine.prepare(compiled)) {
                Tensor actualWork = stored(caller, work.descriptor(), new long[] {bits(DataType.FLOAT32, 2), bits(DataType.FLOAT32, -3)}, false);
                Tensor wrong = stored(caller, descriptor(DataType.INT64, Shape.of(2), false), new long[] {1, 2}, false);
                assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared,
                        ordered(compiled, Map.of(source.id(), wrong, work.id(), actualWork))));
                Arena dead = Arena.ofConfined();
                Tensor deadSource = stored(dead, descriptor, new long[] {1, 2, 3, 4}, false);
                dead.close();
                assertThrows(IllegalStateException.class, () -> engine.runValues(prepared,
                        ordered(compiled, Map.of(source.id(), deadSource, work.id(), actualWork))));
                Tensor live = stored(caller, descriptor, new long[] {91, Long.MIN_VALUE, 17, Long.MAX_VALUE}, false);
                try (var result = engine.runValues(prepared,
                        ordered(compiled, Map.of(source.id(), live, work.id(), actualWork)))) {
                    assertArrayEquals(new long[] {Long.MIN_VALUE, Long.MAX_VALUE},
                            raw(result.materialize(result.publications().getFirst(), 16)));
                }
                assertTrue(live.hostStorage().orElseThrow().isAlive());
                assertArrayEquals(new long[] {bits(DataType.FLOAT32, 2), bits(DataType.FLOAT32, -3)},
                        new long[] {read(actualWork.hostStorage().orElseThrow().segment(), DataType.FLOAT32, 0),
                                read(actualWork.hostStorage().orElseThrow().segment(), DataType.FLOAT32, 1)});
            }
        }
    }

    private record View(Shape shape, long[] strides, long offset, int[] addresses) {}

    private static Engine metalOnly() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        Assumptions.assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is required for real Metal integration");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        assertTrue(Files.isRegularFile(library), "configured Metal library must exist: " + library);
        try (Engine.Builder builder = Engine.builder()) {
            return builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library))).build();
        }
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean gradient) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    private static List<Tensor> ordered(CompiledGraph graph, Map<TensorId, Tensor> values) {
        return graph.inputs().stream().map(input -> values.get(input.tensorId())).toList();
    }

    private static Tensor stored(Arena arena, TensorDescriptor descriptor, long[] physical, boolean readOnly) {
        int width = descriptor.dataType().byteWidth();
        MemorySegment segment = arena.allocate((long) physical.length * width, width);
        for (int index = 0; index < physical.length; index++) {
            long offset = (long) index * width;
            switch (width) {
                case 1 -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) physical[index]);
                case 2 -> segment.set(ValueLayout.JAVA_SHORT, offset, (short) physical[index]);
                case 4 -> segment.set(ValueLayout.JAVA_INT, offset, (int) physical[index]);
                case 8 -> segment.set(ValueLayout.JAVA_LONG, offset, physical[index]);
                default -> throw new AssertionError(width);
            }
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(new MemorySegmentStorage(
                descriptor.dataType(), physical.length, readOnly ? segment.asReadOnly() : segment)));
    }

    private static ScalarValue scalar(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> ScalarValue.float32(value);
            case BFLOAT16 -> ScalarValue.bfloat16(value);
            case FLOAT16 -> ScalarValue.float16(value);
            default -> throw new AssertionError(type);
        };
    }

    private static long bits(DataType type, float value) {
        return switch (type) {
            case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(value));
            case BFLOAT16 -> Short.toUnsignedLong(BFloat16Bits.fromFloat(value));
            case FLOAT16 -> Short.toUnsignedLong(Float16Bits.fromFloat(value));
            default -> throw new AssertionError(type);
        };
    }

    private static long[] patterns(DataType type, int count, int shift) {
        long[] result = new long[count];
        for (int index = 0; index < count; index++) {
            int variant = (index + shift) % 3;
            result[index] = switch (type) {
                case BOOL -> (index + shift) & 1;
                case FLOAT16 -> variant == 0 ? 0x8000 : variant == 1 ? 0x7e15 : 0x3555;
                case BFLOAT16 -> variant == 0 ? 0x8000 : variant == 1 ? 0x7fc1 : 0x3f80;
                case FLOAT8_E4M3, FLOAT8_E5M2 -> throw new AssertionError("Metal does not support FP8");
                case FLOAT32, INT32 -> variant == 0 ? 0x80000000L : variant == 1 ? 0x7fc12345L : 0x3f400000L;
                case FLOAT64, INT64 -> variant == 0 ? Long.MIN_VALUE : variant == 1 ? 0x7ff8123456789abcL : 0x3fe8000000000000L;
            };
        }
        return result;
    }

    private static long read(MemorySegment segment, DataType type, int index) {
        long offset = (long) index * type.byteWidth();
        return switch (type.byteWidth()) {
            case 1 -> Byte.toUnsignedLong(segment.get(ValueLayout.JAVA_BYTE, offset));
            case 2 -> Short.toUnsignedLong(segment.get(ValueLayout.JAVA_SHORT, offset));
            case 4 -> Integer.toUnsignedLong(segment.get(ValueLayout.JAVA_INT, offset));
            case 8 -> segment.get(ValueLayout.JAVA_LONG, offset);
            default -> throw new AssertionError(type);
        };
    }

    private static long[] raw(HostTensorValue value) {
        ByteBuffer bytes = value.bytes();
        long[] result = new long[Math.toIntExact(value.byteSize() / value.dataType().byteWidth())];
        for (int index = 0; index < result.length; index++) {
            result[index] = switch (value.dataType().byteWidth()) {
                case 1 -> Byte.toUnsignedLong(bytes.get());
                case 2 -> Short.toUnsignedLong(bytes.getShort());
                case 4 -> Integer.toUnsignedLong(bytes.getInt());
                case 8 -> bytes.getLong();
                default -> throw new AssertionError(value.dataType());
            };
        }
        return result;
    }
}

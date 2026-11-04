package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public consumers of source publications alongside real nonempty CPU work. */
class EngineProducerlessPublicationTest {
    @Test void zeroTimeSelectedInitialVjpPublishesCallerSeedBitsWithoutTakingCallerOwnership() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
                for (boolean named : List.of(false, true)) for (boolean lengthMode : List.of(false, true))
                    for (boolean reverse : List.of(false, true)) {
                        Dimension rows = named ? new DynamicDimension("sourceRows") : new StaticDimension(2);
                        Shape row = Shape.ofDimensions(rows, new StaticDimension(2));
                        Shape stack = Shape.ofDimensions(new StaticDimension(0), rows, new StaticDimension(2));
                        Tensor h = leaf(type, row, true), x = leaf(type, row, false);
                        Tensor initial = leaf(type, row, true), sequence = leaf(type, stack, false);
                        Tensor finalSeed = leaf(type, row, false), emissionSeed = leaf(type, stack, false);
                        Tensor lengths = leaf(DataType.INT64, Shape.ofDimensions(rows), false);
                        Tensor next = h.add(x).relu();
                        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next));
                        var scan = lengthMode
                                ? Scan.apply(List.of(initial), List.of(sequence), List.of(), lengths, 1,
                                        List.of(OptionalInt.of(0)), body, reverse)
                                : Scan.apply(List.of(initial), List.of(sequence), List.of(), body, reverse);
                        var graph = engine.compile(List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()),
                                List.of(finalSeed, emissionSeed), List.of(initial));
                        try (var prepared = engine.prepare(graph)) {
                            int pass = 0;
                            for (int batch : named ? new int[] {2, 0, 3, 2} : new int[] {2, 2})
                                try (Arena caller = Arena.ofConfined()) {
                                    Shape actualRow = Shape.of(batch, 2), actualStack = Shape.of(0, batch, 2);
                                    long[] initialBits = patterns(type, batch * 2, pass);
                                    long[] seedBits = patterns(type, batch * 2, pass + 1);
                                    Tensor actualInitial = stored(caller, descriptor(type, actualRow, true), initialBits);
                                    Tensor actualSeed = stored(caller, descriptor(type, actualRow, false), seedBits);
                                    var supplied = Map.of(initial.id(), actualInitial, finalSeed.id(), actualSeed,
                                            sequence.id(), stored(caller, descriptor(type, actualStack, false), new long[0]),
                                            emissionSeed.id(), stored(caller, descriptor(type, actualStack, false), new long[0]),
                                            lengths.id(), stored(caller, descriptor(DataType.INT64, Shape.of(batch), false), new long[batch]));
                                    try (var result = engine.runValues(prepared, ordered(graph, supplied))) {
                                        assertArrayEquals(initialBits, raw(result.materialize(result.publications().get(0), 4096)));
                                        assertEquals(actualStack, result.publications().get(1).descriptor().shape());
                                        assertEquals(0, result.materialize(result.publications().get(1), 0).byteSize());
                                        var gradient = result.publications().get(2);
                                        assertEquals(initial.id(), gradient.tensorId());
                                        assertEquals(RunResult.Role.GRADIENT, gradient.role());
                                        assertEquals(OptionalInt.of(1), gradient.derivativeOrder());
                                        assertEquals(OptionalInt.of(0), gradient.targetIndex());
                                        assertEquals(type, gradient.descriptor().dataType());
                                        assertEquals(actualRow, gradient.descriptor().shape());
                                        assertFalse(gradient.descriptor().requiresGrad());
                                        assertArrayEquals(seedBits, raw(result.materialize(gradient, 4096)));
                                    }
                                    assertTrue(actualInitial.hostStorage().orElseThrow().isAlive());
                                    assertTrue(actualSeed.hostStorage().orElseThrow().isAlive());
                                    if (batch > 0) assertEquals(seedBits[0], read(actualSeed.hostStorage().orElseThrow().segment(), type, 0));
                                    pass++;
                                }
                        }
                    }
        }
    }

    @Test void sourceOnlyBatchKeepsLeadingCallerOrderAndUnrelatedFixedWorkAcrossEmptyReentry() {
        try (Engine engine = Engine.standard()) {
            for (boolean empty : List.of(false, true)) {
                var batch = new DynamicDimension("publicationRows");
                Shape template = empty ? Shape.ofDimensions(batch, new StaticDimension(0), new StaticDimension(2))
                        : Shape.ofDimensions(batch, new StaticDimension(2));
                Tensor before = leaf(DataType.FLOAT32, template, false);
                Tensor left = leaf(DataType.FLOAT32, Shape.of(2), false);
                Tensor between = leaf(DataType.FLOAT32, template, false);
                Tensor right = leaf(DataType.FLOAT32, Shape.of(2), false);
                Tensor after = leaf(DataType.FLOAT32, template, false);
                var graph = engine.compile(List.of(before, left.neg(), between, right.neg(), after));
                assertEquals(List.of(before.id(), left.id(), between.id(), right.id(), after.id()),
                        graph.inputs().stream().map(CompiledGraph.Input::tensorId).toList());
                try (var prepared = engine.prepare(graph)) {
                    for (int rows : new int[] {2, 0, 3, 2}) try (Arena caller = Arena.ofConfined()) {
                        Shape shape = empty ? Shape.of(rows, 0, 2) : Shape.of(rows, 2);
                        int count = empty ? 0 : rows * 2;
                        long[] first = patterns(DataType.FLOAT32, count, 0);
                        long[] middle = patterns(DataType.FLOAT32, count, 1);
                        long[] last = patterns(DataType.FLOAT32, count, 2);
                        var supplied = new HashMap<TensorId, Tensor>();
                        supplied.put(before.id(), stored(caller, descriptor(DataType.FLOAT32, shape, false), first));
                        supplied.put(between.id(), stored(caller, descriptor(DataType.FLOAT32, shape, false), middle));
                        supplied.put(after.id(), stored(caller, descriptor(DataType.FLOAT32, shape, false), last));
                        supplied.put(left.id(), TensorFactory.fromFlatArray(left.descriptor(), Optional.empty(), new float[] {1.25f, -3.5f}));
                        supplied.put(right.id(), TensorFactory.fromFlatArray(right.descriptor(), Optional.empty(), new float[] {4, -5}));
                        try (var result = engine.runValues(prepared, ordered(graph, supplied))) {
                            assertArrayEquals(first, raw(result.materialize(result.publications().get(0), 4096)));
                            assertArrayEquals(new long[] {Integer.toUnsignedLong(Float.floatToRawIntBits(-1.25f)),
                                    Integer.toUnsignedLong(Float.floatToRawIntBits(3.5f))}, raw(result.materialize(result.publications().get(1), 8)));
                            assertArrayEquals(middle, raw(result.materialize(result.publications().get(2), 4096)));
                            assertArrayEquals(new long[] {Integer.toUnsignedLong(Float.floatToRawIntBits(-4)),
                                    Integer.toUnsignedLong(Float.floatToRawIntBits(5))}, raw(result.materialize(result.publications().get(3), 8)));
                            assertArrayEquals(last, raw(result.materialize(result.publications().get(4), 4096)));
                        }
                        Shape wrongShape = empty ? Shape.of(rows + 1, 0, 2) : Shape.of(rows + 1, 2);
                        Tensor validFirst = supplied.put(before.id(), stored(caller,
                                descriptor(DataType.FLOAT32, wrongShape, false), patterns(DataType.FLOAT32, empty ? 0 : (rows + 1) * 2, 0)));
                        assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared, ordered(graph, supplied)));
                        supplied.put(before.id(), validFirst);
                        try (var retry = engine.runValues(prepared, ordered(graph, supplied))) {
                            assertArrayEquals(last, raw(retry.materialize(retry.publications().get(4), 4096)));
                        }
                    }
                }
            }
        }
    }

    @Test void sourceOnlyBatchDoesNotBecomeTheRowCountOfAnUnrelatedFixedScan() {
        try (Engine engine = Engine.standard()) {
            for (boolean reverse : List.of(false, true)) {
                Tensor source = leaf(DataType.FLOAT64, Shape.ofDimensions(new DynamicDimension("externalRows"), new StaticDimension(2)), false);
                Tensor h = leaf(DataType.FLOAT64, Shape.of(2, 2), false), x = leaf(DataType.FLOAT64, Shape.of(2, 2), false);
                Tensor initial = leaf(DataType.FLOAT64, Shape.of(2, 2), false), sequence = leaf(DataType.FLOAT64, Shape.of(2, 2, 2), false);
                Tensor next = h.add(x).relu();
                var scan = Scan.apply(List.of(initial), List.of(sequence), List.of(),
                        new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(next)), reverse);
                var graph = engine.compile(List.of(source, scan.finalCarry().getFirst(), scan.outputs().getFirst()));
                try (var prepared = engine.prepare(graph)) {
                    for (int rows : new int[] {2, 0, 3, 2}) try (Arena caller = Arena.ofConfined()) {
                        long[] bits = patterns(DataType.FLOAT64, rows * 2, 0);
                        var supplied = Map.of(source.id(), stored(caller, descriptor(DataType.FLOAT64, Shape.of(rows, 2), false), bits),
                                initial.id(), TensorFactory.fromFlatArray(initial.descriptor(), Optional.empty(), new double[] {1, 2, 3, 4}),
                                sequence.id(), TensorFactory.fromFlatArray(sequence.descriptor(), Optional.empty(), new double[] {1, 1, 1, 1, 2, 2, 2, 2}));
                        try (var result = engine.runValues(prepared, ordered(graph, supplied))) {
                            assertArrayEquals(bits, raw(result.materialize(result.publications().get(0), 4096)));
                            assertArrayEquals(new double[] {4, 5, 6, 7}, doubles(result.materialize(result.publications().get(1), 32)));
                            assertArrayEquals(reverse ? new double[] {4, 5, 6, 7, 3, 4, 5, 6}
                                    : new double[] {2, 3, 4, 5, 4, 5, 6, 7}, doubles(result.materialize(result.publications().get(2), 64)));
                        }
                    }
                }
            }
        }
    }

    @Test void sourceOnlyNamedActualDescriptorsAndStorageRejectBeforeExecutionAndPermitRetry() {
        Tensor source = leaf(DataType.FLOAT32, Shape.ofDimensions(
                new DynamicDimension("checkedSourceRows"), new StaticDimension(2)), false);
        Tensor work = leaf(DataType.FLOAT32, Shape.of(2), false);
        Shape actual = Shape.of(2, 2);
        try (Engine engine = Engine.standard()) {
            var graph = engine.compile(List.of(source, work.neg()));
            try (var prepared = engine.prepare(graph); Arena caller = Arena.ofConfined()) {
                Tensor workValue = TensorFactory.fromFlatArray(work.descriptor(), Optional.empty(), new float[] {2, -3});
                var badDescriptors = List.of(descriptor(DataType.FLOAT64, actual, false),
                        descriptor(DataType.FLOAT32, actual, true),
                        new TensorDescriptor(DataType.FLOAT32, actual, Optional.empty(), false),
                        new TensorDescriptor(DataType.FLOAT32, actual,
                                Optional.of(LayoutDescriptor.of(actual, new long[] {2, 1}, 1, true)), false));
                for (var bad : badDescriptors) {
                    var values = Map.of(source.id(), TensorFactory.create(bad), work.id(), workValue);
                    assertThrows(IllegalArgumentException.class, () -> engine.runValues(prepared, ordered(graph, values)));
                }
                var missing = Map.of(source.id(), leaf(DataType.FLOAT32, actual, false), work.id(), workValue);
                assertThrows(IllegalStateException.class, () -> engine.runValues(prepared, ordered(graph, missing)));
                var denseView = new TensorDescriptor(DataType.FLOAT32, actual,
                        Optional.of(LayoutDescriptor.of(actual, new long[] {2, 1}, 0, true)), false);
                long[] expected = patterns(DataType.FLOAT32, 4, 0);
                Tensor value = stored(caller, denseView, expected);
                try (var result = engine.runValues(prepared, ordered(graph, Map.of(source.id(), value, work.id(), workValue)))) {
                    assertArrayEquals(expected, raw(result.materialize(result.publications().get(0), 16)));
                    assertArrayEquals(new long[] {Integer.toUnsignedLong(Float.floatToRawIntBits(-2)),
                            Integer.toUnsignedLong(Float.floatToRawIntBits(3))}, raw(result.materialize(result.publications().get(1), 8)));
                }
                assertTrue(value.hostStorage().orElseThrow().isAlive());
            }
        }
    }

    @Test void affineAndEmptyCallerPublicationsPreserveEveryStorageTypeAndBorrowedLifetime() {
        var views = List.of(new SourceView(Shape.of(2, 3), new long[] {5, 2}, 1, new int[] {1, 3, 5, 6, 8, 10}),
                new SourceView(Shape.of(2, 3), new long[] {0, 1}, 2, new int[] {2, 3, 4, 2, 3, 4}),
                new SourceView(Shape.scalar(), new long[0], 3, new int[] {3}),
                new SourceView(Shape.of(0, 3), new long[] {5, 1}, 7, new int[0]));
        try (Engine engine = Engine.standard()) {
            for (DataType type : DataType.values()) for (var view : views) {
                var layout = LayoutDescriptor.of(view.shape(), view.strides(), view.offset(), true);
                var descriptor = new TensorDescriptor(type, view.shape(), Optional.of(layout), false);
                Tensor source = TensorFactory.create(descriptor), work = leaf(DataType.FLOAT32, Shape.of(2), false);
                var graph = engine.compile(List.of(source, work.neg()));
                try (var prepared = engine.prepare(graph); Arena caller = Arena.ofConfined()) {
                    long[] physical = patterns(type, Math.toIntExact(layout.referencedElementSpan()), 0);
                    Tensor payload = stored(caller, descriptor, physical);
                    var supplied = Map.of(source.id(), payload, work.id(),
                            TensorFactory.fromFlatArray(work.descriptor(), Optional.empty(), new float[] {2, -3}));
                    long[] expected = Arrays.stream(view.addresses()).mapToLong(address -> physical[address]).toArray();
                    try (var result = engine.runValues(prepared, ordered(graph, supplied))) {
                        assertArrayEquals(expected, raw(result.materialize(result.publications().get(0), 4096)));
                        assertArrayEquals(new long[] {Integer.toUnsignedLong(Float.floatToRawIntBits(-2)),
                                Integer.toUnsignedLong(Float.floatToRawIntBits(3))}, raw(result.materialize(result.publications().get(1), 8)));
                        assertEquals(view.shape(), result.publications().get(0).descriptor().shape());
                    }
                    assertTrue(payload.hostStorage().orElseThrow().isAlive());
                    if (physical.length > 0) assertEquals(physical[0], read(payload.hostStorage().orElseThrow().segment(), type, 0));
                }
            }
        }
    }

    private record SourceView(Shape shape, long[] strides, long offset, int[] addresses) {}

    private static Tensor leaf(DataType type, Shape shape, boolean gradient) {
        return TensorFactory.create(descriptor(type, shape, gradient));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean gradient) {
        return new TensorDescriptor(type, shape, shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), gradient);
    }

    private static List<Tensor> ordered(CompiledGraph graph, Map<TensorId, Tensor> supplied) {
        return graph.inputs().stream().map(input -> Objects.requireNonNull(supplied.get(input.tensorId()))).toList();
    }

    private static Tensor stored(Arena arena, TensorDescriptor descriptor, long[] physical) {
        int width = descriptor.dataType().byteWidth();
        MemorySegment segment = arena.allocate(Math.multiplyExact((long) physical.length, width), width);
        for (int i = 0; i < physical.length; i++) {
            long offset = (long) i * width, bits = physical[i];
            switch (width) {
                case 1 -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) bits);
                case 2 -> segment.set(ValueLayout.JAVA_SHORT, offset, (short) bits);
                case 4 -> segment.set(ValueLayout.JAVA_INT, offset, (int) bits);
                case 8 -> segment.set(ValueLayout.JAVA_LONG, offset, bits);
                default -> throw new AssertionError("unsupported represented width");
            }
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(descriptor.dataType(), physical.length, segment)));
    }

    private static long[] patterns(DataType type, int count, int shift) {
        long[] bits = new long[count];
        for (int i = 0; i < count; i++) {
            int variant = (i + shift) % 3;
            bits[i] = switch (type) {
                case BOOL -> (i + shift) & 1;
                case FLOAT16 -> variant == 0 ? 0x8000 : variant == 1 ? 0x7e15 : 0x3555;
                case BFLOAT16 -> variant == 0 ? 0x8000 : variant == 1 ? 0x7fc1 : 0x3f80;
                case FLOAT8_E4M3 -> variant == 0 ? 0x80 : variant == 1 ? 0xff : 0x38;
                case FLOAT8_E5M2 -> variant == 0 ? 0x80 : variant == 1 ? 0x7d : 0x3c;
                case FLOAT32, INT32 -> variant == 0 ? 0x80000000L : variant == 1 ? 0x7fc12345L : 0x3f400000L;
                case FLOAT64, INT64 -> variant == 0 ? Long.MIN_VALUE : variant == 1 ? 0x7ff8123456789abcL : 0x3fe8000000000000L;
            };
        }
        return bits;
    }

    private static long read(MemorySegment segment, DataType type, int index) {
        long offset = (long) index * type.byteWidth();
        return switch (type.byteWidth()) {
            case 1 -> segment.get(ValueLayout.JAVA_BYTE, offset) & 0xffL;
            case 2 -> segment.get(ValueLayout.JAVA_SHORT, offset) & 0xffffL;
            case 4 -> Integer.toUnsignedLong(segment.get(ValueLayout.JAVA_INT, offset));
            case 8 -> segment.get(ValueLayout.JAVA_LONG, offset);
            default -> throw new AssertionError("unsupported represented width");
        };
    }

    private static long[] raw(HostTensorValue value) {
        var bytes = value.bytes();
        int width = value.dataType().byteWidth();
        long[] result = new long[Math.toIntExact(value.elementCount())];
        for (int i = 0; i < result.length; i++) result[i] = switch (width) {
            case 1 -> bytes.get(i) & 0xffL;
            case 2 -> bytes.getShort(i * width) & 0xffffL;
            case 4 -> Integer.toUnsignedLong(bytes.getInt(i * width));
            case 8 -> bytes.getLong(i * width);
            default -> throw new AssertionError("unsupported represented width");
        };
        return result;
    }

    private static double[] doubles(HostTensorValue value) {
        var bytes = value.bytes();
        double[] result = new double[Math.toIntExact(value.elementCount())];
        for (int i = 0; i < result.length; i++) result[i] = bytes.getDouble(i * Double.BYTES);
        return result;
    }
}

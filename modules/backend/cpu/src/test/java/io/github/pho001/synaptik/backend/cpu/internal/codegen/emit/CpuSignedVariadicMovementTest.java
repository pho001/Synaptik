package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.CpuGeneralScanCapabilityTest;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

/** Behavioral represented-bit and actual-Compiler-adjoint closure of signed/variadic movement. */
class CpuSignedVariadicMovementTest {
    @Test void signedExtractionMaterializesEveryRawTypeFromStridedSources() throws Throwable {
        for (var type : DataType.values()) try (var arena = Arena.ofConfined()) {
            Shape source = Shape.of(2, 5), target = Shape.of(2, 3);
            var input = CpuNonAffineMovementLoweringTest.descriptor(type, source,
                    LayoutDescriptor.of(source, new long[]{13, 2}, 1, true));
            var output = CpuNonAffineMovementLoweringTest.descriptor(type, target);
            var context = CpuNonAffineMovementLoweringTest.context(new Operation(SliceKind.SLICE,
                    new SliceAttrs(List.of(1L, 4L), List.of(2L, 3L), List.of(0, 1),
                            List.of(-1L, -2L))), List.of(0), List.of(input), output);
            var in = arena.allocate(23L * type.byteWidth(), type.byteWidth());
            var out = arena.allocate(6L * type.byteWidth(), type.byteWidth());
            for (int i = 0; i < 23; i++) put(type, in, i, pattern(type, i));
            byte[] before = in.toArray(ValueLayout.JAVA_BYTE);
            out.fill((byte) 0x55);
            invoke(context, List.of(in, out), 0, 2);
            invoke(context, List.of(in, out), 2, 6);
            for (int row = 0; row < 2; row++) for (int col = 0; col < 3; col++)
                assertEquals(get(type, in, 1 + (1-row)*13 + (4-2*col)*2),
                        get(type, out, row*3+col), type.toString());
            assertArrayEquals(before, in.toArray(ValueLayout.JAVA_BYTE));
            assertEquals(LayoutDescriptor.contiguous(target), output.layout().orElseThrow());
        }
    }

    @Test void fp8AffineAndPaddingReplayEveryByteAcrossCarrierDirections() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            try (Arena arena = Arena.ofConfined()) {
                Shape shape = Shape.of(256);
                var input = CpuNonAffineMovementLoweringTest.descriptor(type, shape,
                        LayoutDescriptor.of(shape, new long[] {2}, 1, true));
                var dense = CpuNonAffineMovementLoweringTest.descriptor(type, shape);
                var affine = CpuNonAffineMovementLoweringTest.context(new Operation(
                        ContiguousKind.CONTIGUOUS,
                        io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE),
                        List.of(0), List.of(input), dense);
                var source = arena.allocate(514, 1);
                source.fill((byte) 0x55);
                byte[] expected = new byte[256];
                for (int i = 0; i < 256; i++) {
                    expected[i] = (byte) i;
                    source.setAtIndex(ValueLayout.JAVA_BYTE, 1+2L*i, (byte) i);
                }
                byte[] before = source.toArray(ValueLayout.JAVA_BYTE);
                var copied = arena.allocate(256, 1);
                invoke(affine, List.of(source, copied), 0, 256);
                assertArrayEquals(expected, copied.toArray(ValueLayout.JAVA_BYTE), type.toString());
                byte[] heapResult = new byte[256];
                var segmentToHeap = new PrepareContext<>(affine.partition(), affine.nodes(), affine.values(),
                        affine.memoryRequirements(), affine.constants(), new CpuPartitionAnalysisInputs(false,
                                List.of(CarrierAccess.MEMORY_SEGMENT, CarrierAccess.BYTE_ARRAY)));
                invokeGenerated(segmentToHeap, List.of(source, heapResult), 0, 256);
                assertArrayEquals(expected, heapResult);
                copied.fill((byte) 0x5a);
                var heapToSegment = new PrepareContext<>(affine.partition(), affine.nodes(), affine.values(),
                        affine.memoryRequirements(), affine.constants(), new CpuPartitionAnalysisInputs(false,
                                List.of(CarrierAccess.BYTE_ARRAY, CarrierAccess.MEMORY_SEGMENT)));
                invokeGenerated(heapToSegment, List.of(before.clone(), copied), 0, 256);
                assertArrayEquals(expected, copied.toArray(ValueLayout.JAVA_BYTE));

                Shape paddedShape = Shape.of(260);
                var paddedDescriptor = CpuNonAffineMovementLoweringTest.descriptor(type, paddedShape,
                        LayoutDescriptor.of(paddedShape, new long[] {2}, 2, true));
                var literal = type == DataType.FLOAT8_E4M3
                        ? io.github.pho001.synaptik.model.datatype.ScalarValue.float8E4m3Bits((byte) 255)
                        : io.github.pho001.synaptik.model.datatype.ScalarValue.float8E5m2Bits((byte) 255);
                var padding = CpuNonAffineMovementLoweringTest.context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(2L), List.of(2L), literal)),
                        List.of(0), List.of(input), paddedDescriptor);
                var padded = arena.allocate(525, 1);
                padded.fill((byte) 0x5a);
                invoke(padding, List.of(source, padded), 0, 129);
                invoke(padding, List.of(source, padded), 129, 260);
                for (int i = 0; i < 525; i++) {
                    int logical = (i - 2) / 2;
                    int raw = i >= 2 && i <= 520 && (i & 1) == 0
                            ? logical < 2 || logical >= 258 ? 255 : logical - 2 : 0x5a;
                    assertEquals(raw, Byte.toUnsignedInt(padded.getAtIndex(ValueLayout.JAVA_BYTE, i)),
                            type + " physical=" + i);
                }
                assertArrayEquals(before, source.toArray(ValueLayout.JAVA_BYTE), "borrowed source");
            }
    }

    @Test void signedEmptyAndSingletonDoNotInventNegativeStorageOrOverflow() throws Throwable {
        for (long length : new long[]{0, 1}) {
            var context = CpuNonAffineMovementLoweringTest.context(new Operation(SliceKind.SLICE,
                    new SliceAttrs(List.of(0L), List.of(length), List.of(0), List.of(Long.MIN_VALUE))),
                    List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(1))),
                    CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(length)));
            long[] in = {Long.MIN_VALUE}, out = new long[(int) length];
            invoke(context, List.of(MemorySegment.ofArray(in), MemorySegment.ofArray(out)), 0, length);
            invoke(context, List.of(MemorySegment.ofArray(in), MemorySegment.ofArray(out)), length, length);
            if (length != 0) assertEquals(Long.MIN_VALUE, out[0]);
        }
    }

    @Test void denseAffineEmptyRangesDoNotReadPacksOrCarriersOrTouchOutputCanaries() throws Throwable {
        for (var type : DataType.values()) try (var arena = Arena.ofConfined()) {
            var context = CpuNonAffineMovementLoweringTest.context(new Operation(SliceKind.SLICE,
                    new SliceAttrs(List.of(0L), List.of(1L), List.of(0), List.of(-1L))),
                    List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(type, Shape.of(1))),
                    CpuNonAffineMovementLoweringTest.descriptor(type, Shape.of(1)));
            var output = arena.allocate(3L * type.byteWidth(), type.byteWidth());
            for (int i = 0; i < 3; i++) put(type, output, i, pattern(type, i));
            byte[] before = output.toArray(ValueLayout.JAVA_BYTE);
            var absent = MemorySegment.ofArray(new byte[0]);
            invoke(context, List.of(absent, output), 0, 0);
            invoke(context, List.of(absent, output), 1, 1);
            assertArrayEquals(before, output.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @Test void variadicConcatAndStackWriteEveryOccurrenceAcrossAxesAndRawCarriers() throws Throwable {
        for (var type : DataType.values()) for (var kind : TensorCompositionKind.values())
            for (int axis = 0; axis < (kind == TensorCompositionKind.STACK ? 3 : 2); axis++)
                for (boolean repeated : List.of(false, true)) try (var arena = Arena.ofConfined()) {
                    int count = 257;
                    var descriptors = new ArrayList<TensorDescriptor>();
                    var carriers = new ArrayList<Object>();
                    int unique = repeated ? 2 : count;
                    Shape shape = Shape.of(2, 1);
                    for (int i = 0; i < unique; i++) {
                        descriptors.add(CpuNonAffineMovementLoweringTest.descriptor(type, shape,
                                LayoutDescriptor.of(shape, new long[]{repeated && i == 1 ? 0 : 3, 1}, 1, true)));
                        var storage = arena.allocate(5L * type.byteWidth(), type.byteWidth());
                        put(type, storage, 1, pattern(type, i));
                        put(type, storage, 4, pattern(type, i+37));
                        carriers.add(storage);
                    }
                    var map = IntStream.range(0, count).map(i -> repeated ? i % 2 : i).boxed().toList();
                    long[] extents = kind == TensorCompositionKind.STACK
                            ? axis == 0 ? new long[]{count, 2, 1}
                              : axis == 1 ? new long[]{2, count, 1} : new long[]{2, 1, count}
                            : axis == 0 ? new long[]{2L*count, 1} : new long[]{2, count};
                    var context = CpuNonAffineMovementLoweringTest.context(new Operation(kind,
                            new CompositionAxisAttrs(axis)), map, descriptors,
                            CpuNonAffineMovementLoweringTest.descriptor(type, Shape.of(extents)));
                    var out = arena.allocate(2L * count * type.byteWidth(), type.byteWidth());
                    carriers.add(out);
                    invoke(context, carriers, 0, 11);
                    invoke(context, carriers, 11, 2L*count);
                    for (int occurrence = 0; occurrence < count; occurrence++) for (int row = 0; row < 2; row++) {
                        int logical = axis == 0 ? occurrence*2+row : row*count+occurrence;
                        long stride = descriptors.get(map.get(occurrence)).layout().orElseThrow().strides()[0];
                        assertEquals(get(type, (MemorySegment) carriers.get(map.get(occurrence)), 1+row*stride),
                                get(type, out, logical), type + " " + kind + " axis=" + axis);
                    }
                }
    }

    @Test void variadicEmptySourcesAndLargerCheckedEntryExecuteWithoutRecursiveCopies() throws Throwable {
        for (var kind : TensorCompositionKind.values()) for (int count : new int[]{102, 257, 1025}) {
            var inputs = Collections.nCopies(count,
                    CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(1)));
            var context = CpuNonAffineMovementLoweringTest.context(new Operation(kind, new CompositionAxisAttrs(0)),
                    IntStream.range(0, count).boxed().toList(), inputs,
                    CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32,
                            kind == TensorCompositionKind.STACK ? Shape.of(count, 1) : Shape.of(count)));
            var carriers = new ArrayList<Object>();
            for (int i = 0; i < count; i++) carriers.add(MemorySegment.ofArray(new int[]{i*7}));
            int[] output = new int[count]; carriers.add(MemorySegment.ofArray(output));
            invoke(context, carriers, 0, count);
            assertArrayEquals(IntStream.range(0, count).map(i -> i*7).toArray(), output);
        }
        var empty = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 0));
        var context = CpuNonAffineMovementLoweringTest.context(new Operation(TensorCompositionKind.CONCAT,
                new CompositionAxisAttrs(1)), Collections.nCopies(33, 0), List.of(empty), empty);
        invoke(context, List.of(MemorySegment.ofArray(new int[0]), MemorySegment.ofArray(new int[0])), 0, 0);
        var mixed = CpuNonAffineMovementLoweringTest.context(new Operation(TensorCompositionKind.CONCAT,
                new CompositionAxisAttrs(1)), IntStream.range(0, 17).map(i -> i % 2).boxed().toList(),
                List.of(empty, CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 1))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 8)));
        int[] mixedOutput = new int[16];
        invoke(mixed, List.of(MemorySegment.ofArray(new int[0]), MemorySegment.ofArray(new int[]{3, 7}),
                MemorySegment.ofArray(mixedOutput)), 0, 16);
        assertArrayEquals(new int[]{3,3,3,3,3,3,3,3,7,7,7,7,7,7,7,7}, mixedOutput);
    }
    @Test void publicWideCompositionPreservesEveryStoredTypeAndLateOperand() {
        for (var type : DataType.values()) for (var kind : TensorCompositionKind.values())
            for (int count : new int[]{102, 257, 1025}) {
                Tensor[] inputs = IntStream.range(0, count).mapToObj(i -> tensor(type, false, 1))
                        .toArray(Tensor[]::new);
                Tensor output = kind == TensorCompositionKind.CONCAT
                        ? Tensor.concat(0, inputs) : Tensor.stack(0, inputs);
                var artifact = CpuGeneralScanCapabilityTest.compile(List.of(output), Optional.empty(),
                        false, new CpuCapabilityProvider());
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    long[] expected = new long[count];
                    for (int i = 0; i < count; i++) {
                        expected[i] = pattern(type, i);
                        consumer.raw(inputs[i], new long[]{expected[i]});
                    }
                    consumer.bind(); consumer.execute();
                    assertArrayEquals(expected, consumer.forwardRaw(0), type + " " + kind + " " + count);
                    consumer.execute();
                    assertArrayEquals(expected, consumer.forwardRaw(0));
                }
            }
    }

    @Test void wideCompositionStillRejectsFiniteExtentAndByteOverflow() {
        assertThrows(RuntimeException.class, () -> {
            Tensor huge = tensor(DataType.INT64, false, Long.MAX_VALUE / 257 + 1);
            Tensor.concat(0, Collections.nCopies(257, huge).toArray(Tensor[]::new));
        });
        assertThrows(RuntimeException.class, () -> {
            long extent = Long.MAX_VALUE / 257 / Long.BYTES + 1;
            var input = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(extent));
            var output = CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(257, extent));
            var context = CpuNonAffineMovementLoweringTest.context(
                    new Operation(TensorCompositionKind.STACK, new CompositionAxisAttrs(0)),
                    Collections.nCopies(257, 0), List.of(input), output);
            // Model element geometry fits; checked physical byte capacity belongs to preparation.
            new CpuPartitionPreparer().analyze(context);
        });
    }

    @Test void variadicMovementFeedsOrdinaryConsumersBeyondCandidateResourceCeilings() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : TensorCompositionKind.values()) for (int count : new int[]{17, 102, 257}) {
                Tensor[] inputs = IntStream.range(0, count).mapToObj(i -> tensor(type, false, 2, 1))
                        .toArray(Tensor[]::new);
                Tensor composed = kind == TensorCompositionKind.CONCAT
                        ? Tensor.concat(1, inputs) : Tensor.stack(1, inputs);
                Tensor output = composed.neg();
                var artifact = CpuGeneralScanCapabilityTest.compile(List.of(output), Optional.empty(),
                        false, new CpuCapabilityProvider());
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    for (int i = 0; i < count; i++) consumer.numbers(inputs[i], new double[]{i + 1, i + 101});
                    consumer.bind(); consumer.execute();
                    double[] expected = new double[count * 2];
                    for (int i = 0; i < count; i++) {
                        expected[i] = -i - 1; expected[count + i] = -i - 101;
                    }
                    assertArrayEquals(expected, consumer.forward(0), 0);
                }
            }
    }

    @Test void boundedUnitsWithMoreThanSixtyFourCombinedBoundaryPositionsExecuteDirectly() {
        Tensor[] inputs = IntStream.range(0, 60).mapToObj(i -> tensor(DataType.FLOAT32, false, 1))
                .toArray(Tensor[]::new);
        Tensor[] groups = IntStream.range(0, 5).mapToObj(i ->
                Tensor.concat(0, Arrays.copyOfRange(inputs, i * 12, (i + 1) * 12))).toArray(Tensor[]::new);
        Tensor output = Tensor.concat(0, groups).neg();
        var artifact = CpuGeneralScanCapabilityTest.compile(List.of(output), Optional.empty(),
                false, new CpuCapabilityProvider());
        try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
            for (int i = 0; i < inputs.length; i++) consumer.numbers(inputs[i], new double[]{i + 1});
            consumer.bind(); consumer.execute();
            assertArrayEquals(IntStream.range(0, 60).mapToDouble(i -> -i - 1).toArray(),
                    consumer.forward(0), 0);
        }
    }
    @Test void publicSignedAndEmptySlicesExecuteWithoutDiscardedStrideOverflow() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean empty : List.of(false, true)) {
            Tensor input = tensor(type, false, 1, 2);
            Tensor output = empty
                    ? input.sliceByLength(new long[]{0}, new long[]{0}, new int[]{0},
                            new long[]{Long.MAX_VALUE})
                    : input.sliceByLength(new long[]{0, 1}, new long[]{1, 2}, new int[]{0, 1},
                            new long[]{Long.MAX_VALUE, -1});
            var artifact = CpuGeneralScanCapabilityTest.compile(List.of(output), Optional.empty(),
                    false, new CpuCapabilityProvider());
            try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                consumer.numbers(input, new double[]{3, 7});
                consumer.bind(); consumer.execute();
                assertArrayEquals(empty ? new double[0] : new double[]{7, 3}, consumer.forward(0), 0);
            }
        }
    }


    @Test void actualSliceAndSliceUpdateAdjointsScatterAndExtractNormalizedCoordinates() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean update : List.of(false, true)) {
            Tensor base = tensor(type, true, 2, 5), replacement = tensor(type, true, 2, 3);
            Tensor output = update ? base.sliceUpdate(replacement, new long[]{4}, new int[]{1}, new long[]{-2})
                    : base.sliceByLength(new long[]{4}, new long[]{3}, new int[]{1}, new long[]{-2});
            Tensor seed = tensor(type, false, output.descriptor().shape().toLongArray());
            var targets = update ? List.of(base, replacement) : List.of(base);
            var artifact = compile(output, seed, targets);
            try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                double[] x = IntStream.range(0, 10).mapToDouble(i -> i+1).toArray();
                double[] u = IntStream.range(0, 6).mapToDouble(i -> i+21).toArray();
                double[] s = IntStream.range(0, update ? 10 : 6).mapToDouble(i -> i+2).toArray();
                consumer.numbers(base, x); if (update) consumer.numbers(replacement, u); consumer.numbers(seed, s);
                consumer.bind(); consumer.execute();
                double[] expected = update ? x.clone() : new double[6], dx = new double[10], du = new double[6];
                if (update) System.arraycopy(s, 0, dx, 0, dx.length);
                for (int row = 0; row < 2; row++) for (int col = 0; col < 3; col++) {
                    int selected = row*5+4-col*2, ordinal = row*3+col;
                    if (update) { expected[selected] = u[ordinal]; dx[selected] = 0; du[ordinal] = s[selected]; }
                    else { expected[ordinal] = x[selected]; dx[selected] = s[ordinal]; }
                }
                assertArrayEquals(expected, consumer.forward(0), 0);
                assertArrayEquals(dx, consumer.gradient(0), 0);
                if (update) assertArrayEquals(du, consumer.gradient(1), 0);
            }
        }
    }

    @Test void actualVariadicCompositionAdjointsAccumulateRepeatedInputIdentities() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : TensorCompositionKind.values()) {
                Tensor a = tensor(type, true, 2, 1), b = tensor(type, true, 2, 1);
                int count = 257;
                Tensor[] occurrences = IntStream.range(0, count).mapToObj(i -> i % 2 == 0 ? a : b)
                        .toArray(Tensor[]::new);
                Tensor output = kind == TensorCompositionKind.CONCAT
                        ? Tensor.concat(1, occurrences) : Tensor.stack(1, occurrences);
                Tensor seed = tensor(type, false, output.descriptor().shape().toLongArray());
                var artifact = compile(output, seed, List.of(a, b));
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    consumer.numbers(a, new double[]{3, 7}); consumer.numbers(b, new double[]{5, 11});
                    double[] s = IntStream.range(0, 2 * count).mapToDouble(i -> i+1).toArray();
                    consumer.numbers(seed, s); consumer.bind(); consumer.execute();
                    double[] expected = new double[2 * count], da = new double[2], db = new double[2];
                    for (int row = 0; row < 2; row++) for (int i = 0; i < count; i++) {
                        expected[row*count+i] = i % 2 == 0 ? row == 0 ? 3 : 7 : row == 0 ? 5 : 11;
                        if (i % 2 == 0) da[row] += s[row*count+i]; else db[row] += s[row*count+i];
                    }
                    assertArrayEquals(expected, consumer.forward(0), 0);
                    assertArrayEquals(da, consumer.gradient(0), 0);
                    assertArrayEquals(db, consumer.gradient(1), 0);
                }
            }
    }

    @Test void publicWideCompositionAdjointReachesTheLastDistinctInput() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : TensorCompositionKind.values()) {
                int count = 257;
                Tensor[] inputs = IntStream.range(0, count).mapToObj(i -> tensor(type, true, 1))
                        .toArray(Tensor[]::new);
                Tensor output = kind == TensorCompositionKind.CONCAT
                        ? Tensor.concat(0, inputs) : Tensor.stack(0, inputs);
                Tensor seed = tensor(type, false, output.descriptor().shape().toLongArray());
                var artifact = compile(output, seed, List.of(inputs[count - 1]));
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    for (int i = 0; i < count; i++) consumer.numbers(inputs[i], new double[]{i + 1});
                    double[] seeds = IntStream.range(0, count).mapToDouble(i -> i + 7).toArray();
                    consumer.numbers(seed, seeds); consumer.bind(); consumer.execute();
                    assertArrayEquals(new double[]{seeds[count - 1]}, consumer.gradient(0), 0);
                }
            }
    }

    @Test void wideCompositionInsideControlledScanUsesOriginalRowsAndEncounterOrder() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (var kind : TensorCompositionKind.values()) for (boolean reverse : List.of(false, true)) {
                int count = 257;
                Tensor x = tensor(type, false, 2, 1), y = tensor(type, false, 2, 1);
                Tensor[] occurrences = IntStream.range(0, count).mapToObj(i -> i % 2 == 0 ? x : y)
                        .toArray(Tensor[]::new);
                Tensor output = kind == TensorCompositionKind.CONCAT
                        ? Tensor.concat(1, occurrences) : Tensor.stack(1, occurrences);
                var body = new ScanBody(List.of(), List.of(x, y), List.of(), List.of(), List.of(output));
                Tensor xs = tensor(type, false, 2, 2, 1), ys = tensor(type, false, 2, 2, 1);
                Tensor lengths = tensor(DataType.INT64, false, 2);
                var scan = Scan.apply(List.of(), List.of(xs, ys), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                var artifact = CpuGeneralScanCapabilityTest.compile(scan.outputs(), Optional.empty(),
                        false, new CpuCapabilityProvider());
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    consumer.numbers(xs, new double[]{3, 5, 7, Double.NaN});
                    consumer.numbers(ys, new double[]{11, 13, 17, Double.NaN});
                    consumer.raw(lengths, new long[]{2, 1});
                    consumer.bind(); consumer.execute();
                    double[] expected = new double[4 * count];
                    for (int i = 0; i < count; i++) {
                        expected[i] = i % 2 == 0 ? 3 : 11;
                        expected[count + i] = i % 2 == 0 ? 5 : 13;
                        expected[2 * count + i] = i % 2 == 0 ? 7 : 17;
                    }
                    assertArrayEquals(expected, consumer.forward(0), 0);
                }
            }
    }

    @Test void qualifiedScanUsesTheSameSignedUpdateAndActualAdjointsInBothTraversals() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (boolean reverse : List.of(false, true)) for (boolean lengthMode : List.of(false, true)) {
                Tensor base = tensor(type, true, 2, 5), update = tensor(type, true, 2, 3);
                Tensor result = base.sliceUpdate(update, new long[]{4}, new int[]{1}, new long[]{-2});
                var body = new ScanBody(List.of(), List.of(base, update), List.of(), List.of(), List.of(result));
                Tensor xs = tensor(type, true, 3, 2, 5), us = tensor(type, true, 3, 2, 3);
                Tensor lengths = tensor(DataType.INT64, false, 2), seed = tensor(type, false, 3, 2, 5);
                var scan = lengthMode ? Scan.apply(List.of(), List.of(xs, us), List.of(), lengths, 0,
                        List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse)
                        : Scan.apply(List.of(), List.of(xs, us), List.of(), body, reverse);
                var artifact = compile(scan.outputs().getFirst(), seed, List.of(xs, us));
                try (var consumer = new Consumer(artifact, OptionalLong.empty())) {
                    double[] x = IntStream.range(0, 30).mapToDouble(i -> i+1).toArray();
                    double[] u = IntStream.range(0, 18).mapToDouble(i -> i+51).toArray();
                    double[] s = IntStream.range(0, 30).mapToDouble(i -> i+2).toArray();
                    consumer.numbers(xs, x); consumer.numbers(us, u); consumer.numbers(seed, s);
                    if (lengthMode) consumer.raw(lengths, new long[]{3, 1});
                    consumer.bind(); consumer.execute();
                    double[] expected = new double[30], dx = new double[30], du = new double[18];
                    for (int t = 0; t < 3; t++) for (int row = 0; row < 2; row++)
                        if (!lengthMode || row == 0 || t == 0) {
                            for (int col = 0; col < 5; col++) { int i = (t*2+row)*5+col; expected[i] = x[i]; dx[i] = s[i]; }
                            for (int col = 0; col < 3; col++) {
                                int i = (t*2+row)*5+4-col*2, j = (t*2+row)*3+col;
                                expected[i] = u[j]; dx[i] = 0; du[j] = s[i];
                            }
                        }
                    assertArrayEquals(expected, consumer.forward(0), 0);
                    assertArrayEquals(dx, consumer.gradient(0), 0);
                    assertArrayEquals(du, consumer.gradient(1), 0);
                }
            }
    }

    private static io.github.pho001.synaptik.compiler.CompileArtifacts compile(Tensor output, Tensor seed, List<Tensor> targets) {
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(output)), List.of(Optional.of(seed)),
                targets, false, FunctionalGradientRequest.DisconnectedPolicy.ERROR)));
        return CpuGeneralScanCapabilityTest.compile(List.of(output), Optional.of(request), false, new CpuCapabilityProvider());
    }

    private static Tensor tensor(DataType type, boolean grad, long... shape) {
        Shape dimensions = Shape.of(shape);
        return TensorFactory.create(new TensorDescriptor(type, dimensions,
                Optional.of(LayoutDescriptor.contiguous(dimensions)), grad));
    }

    private static void invoke(PrepareContext<CpuPartitionAnalysisInputs> context, List<Object> carriers,
            long start, long end) throws Throwable {
        var heapCarriers = new ArrayList<Object>();
        var pattern = new ArrayList<CarrierAccess>();
        for (int i = 0; i < carriers.size(); i++) {
            var type = context.values().get(i).descriptor().dataType();
            var segment = (MemorySegment) carriers.get(i);
            heapCarriers.add(switch (type) {
                case FLOAT64 -> segment.toArray(ValueLayout.JAVA_DOUBLE_UNALIGNED);
                case FLOAT32 -> segment.toArray(ValueLayout.JAVA_FLOAT_UNALIGNED);
                case BFLOAT16, FLOAT16 -> segment.toArray(ValueLayout.JAVA_SHORT_UNALIGNED);
                case INT32 -> segment.toArray(ValueLayout.JAVA_INT_UNALIGNED);
                case INT64 -> segment.toArray(ValueLayout.JAVA_LONG_UNALIGNED);
                case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> segment.toArray(ValueLayout.JAVA_BYTE);
            });
            pattern.add(switch (type) {
                case FLOAT64 -> CarrierAccess.DOUBLE_ARRAY;
                case FLOAT32 -> CarrierAccess.FLOAT_ARRAY;
                case BFLOAT16, FLOAT16 -> CarrierAccess.SHORT_ARRAY;
                case INT32 -> CarrierAccess.INT_ARRAY;
                case INT64 -> CarrierAccess.LONG_ARRAY;
                case BOOL, FLOAT8_E4M3, FLOAT8_E5M2 -> CarrierAccess.BYTE_ARRAY;
            });
        }
        invokeGenerated(context, carriers, start, end);
        var heapContext = new PrepareContext<>(context.partition(), context.nodes(), context.values(),
                context.memoryRequirements(), context.constants(), new CpuPartitionAnalysisInputs(false, pattern));
        invokeGenerated(heapContext, heapCarriers, start, end);
        MemorySegment heapOutput = switch (heapCarriers.getLast()) {
            case double[] a -> MemorySegment.ofArray(a); case float[] a -> MemorySegment.ofArray(a);
            case short[] a -> MemorySegment.ofArray(a); case int[] a -> MemorySegment.ofArray(a);
            case long[] a -> MemorySegment.ofArray(a); case byte[] a -> MemorySegment.ofArray(a);
            default -> throw new AssertionError("unexpected heap carrier");
        };
        assertArrayEquals(((MemorySegment) carriers.getLast()).toArray(ValueLayout.JAVA_BYTE),
                heapOutput.toArray(ValueLayout.JAVA_BYTE));
        var mixedCarriers = new ArrayList<Object>(carriers);
        var mixedPattern = new ArrayList<CarrierAccess>(Collections.nCopies(carriers.size(),
                CarrierAccess.MEMORY_SEGMENT));
        for (int i = 1; i + 1 < carriers.size(); i += 2) {
            mixedCarriers.set(i, heapCarriers.get(i)); mixedPattern.set(i, pattern.get(i));
        }
        var mixedContext = new PrepareContext<>(context.partition(), context.nodes(), context.values(),
                context.memoryRequirements(), context.constants(), new CpuPartitionAnalysisInputs(false, mixedPattern));
        invokeGenerated(mixedContext, mixedCarriers, start, end);
        assertArrayEquals(heapOutput.toArray(ValueLayout.JAVA_BYTE),
                ((MemorySegment) carriers.getLast()).toArray(ValueLayout.JAVA_BYTE));
    }

    private static void invokeGenerated(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<Object> carriers, long start, long end) throws Throwable {
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                generator.generateClassBytes(route.specialization(), route.kernelIr()));
        var arguments = new ArrayList<Object>();
        if (route.specialization().movementOperandTable()) {
            MemorySegment[] table = carriers.stream().map(carrier -> switch (carrier) {
                case MemorySegment value -> value;
                case double[] value -> MemorySegment.ofArray(value);
                case float[] value -> MemorySegment.ofArray(value);
                case short[] value -> MemorySegment.ofArray(value);
                case int[] value -> MemorySegment.ofArray(value);
                case long[] value -> MemorySegment.ofArray(value);
                case byte[] value -> MemorySegment.ofArray(value);
                default -> throw new AssertionError("unexpected carrier");
            }).toArray(MemorySegment[]::new);
            arguments.add(table);
        } else arguments.addAll(carriers);
        arguments.add(plan.movementGeometry().isPresent()
                ? plan.movementGeometry().orElseThrow().pack(new long[carriers.size()], start, end)
                : plan.affineAddressPairs());
        arguments.add(start); arguments.add(end);
        artifact.entryPoint().invokeWithArguments(arguments);
    }

    private static long pattern(DataType type, int i) {
        return switch (type) {
            case FLOAT64 -> switch (i % 4) { case 0 -> Long.MIN_VALUE; case 1 -> 0x7ff8_0000_0000_0042L+i; case 2 -> 1; default -> 0x3ff0_0000_0000_0000L+i; };
            case FLOAT32 -> switch (i % 4) { case 0 -> 0x8000_0000L; case 1 -> 0x7fc0_0042L+i; case 2 -> 1; default -> 0x3f80_0000L+i; };
            case BFLOAT16 -> (i % 2 == 0 ? 0x8000L : 0x7fc0L+i) & 0xffffL;
            case FLOAT16 -> (i % 2 == 0 ? 0x8000L : 0x7e00L+i) & 0xffffL;
            case FLOAT8_E4M3, FLOAT8_E5M2 -> i & 255;
            case INT64 -> Long.MIN_VALUE+i;
            case INT32 -> Integer.toUnsignedLong(Integer.MIN_VALUE+i);
            case BOOL -> i & 1;
        };
    }
    private static void put(DataType type, MemorySegment segment, long index, long bits) {
        switch (type.byteWidth()) {
            case 1 -> segment.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) bits);
            case 2 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, (short) bits);
            case 4 -> segment.setAtIndex(ValueLayout.JAVA_INT, index, (int) bits);
            case 8 -> segment.setAtIndex(ValueLayout.JAVA_LONG, index, bits);
            default -> throw new AssertionError(type);
        }
    }
    private static long get(DataType type, MemorySegment segment, long index) {
        return switch (type.byteWidth()) {
            case 1 -> Byte.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_BYTE, index));
            case 2 -> Short.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case 4 -> Integer.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_INT, index));
            case 8 -> segment.getAtIndex(ValueLayout.JAVA_LONG, index);
            default -> throw new AssertionError(type);
        };
    }
}

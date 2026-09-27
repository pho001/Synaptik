package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

class MetalMpsGraphAffineNativeTest {
    private static final int[] ADVERSARIAL_BITS = {
            0x00000000, 0x80000000,
            0x00000001, 0x80000001, 0x00012345, 0x80054321,
            0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
            0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
            0x7fc00001, 0x7fc12345, 0xffc54321,
            0x3f800001, 0xbf400002, 0x41200003, 0xc0a00004,
            0x3e000005, 0x40000006, 0xc0000007
    };

    @Test
    void realSelectorsWriteSuppliedDenseTargetsWithExactShapesBitsAndRepeatability() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            for (AffineCase test : cases()) {
                runCase(api, context, test);
            }
        } finally {
            if (context != null) {
                api.releaseContext(context);
            }
            api.close();
        }
    }

    @Test
    void realGraphComposesViewContiguousNegAndReshapeWithPublishedIntermediate() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            int[] ranks = {3, 3, 3, 3, 2};
            long[] dimensions = dimensions(new long[][] {
                {2, 3, 4}, {4, 2, 3}, {4, 2, 3}, {4, 2, 3}, {4, 6}
            });
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks, dimensions, new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.permutation(
                            0, 1, List.of(2, 0, 1)),
                    MetalMpsGraphProgram.Node.contiguous(1, 2),
                    MetalMpsGraphProgram.Node.neg(2, 3),
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            3,
                            4,
                            new long[] {4, 6})))), new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.permutation(
            0, 1, List.of(2, 0, 1)),
                                MetalMpsGraphProgram.Node.contiguous(1, 2),
                                MetalMpsGraphProgram.Node.neg(2, 3),
                                MetalMpsGraphProgram.Node.targetShape(
            MetalMpsGraphProgram.NodeKind.RESHAPE,
            3,
            4,
            new long[] {4, 6}))), new int[] {0}, new int[] {1, 2, 3, 4}, MetalPreparedRoute.CUSTOM_PROGRAM);
            input = api.createBuffer(context, 96L);
            for (int target = 0; target < 4; target++) {
                outputs.add(api.createBuffer(context, 96L));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment source = arena.allocate(96L, Integer.BYTES);
                for (int index = 0; index < ADVERSARIAL_BITS.length; index++) {
                    source.setAtIndex(JAVA_INT, index, ADVERSARIAL_BITS[index]);
                }
                api.upload(input, 0L, source, 96L);
                MemorySegment inputs = arena.allocate(ADDRESS, 5);
                MemorySegment targetAddresses = arena.allocate(ADDRESS, outputs.size());
                inputs.setAtIndex(ADDRESS, 0L, input.carrier());
                for (int target = 0; target < outputs.size(); target++) {
          inputs.setAtIndex(ADDRESS, target + 1L, outputs.get(target).carrier());
                    targetAddresses.setAtIndex(
                            ADDRESS, target, outputs.get(target).carrier());
                }
                var downloaded = new ArrayList<MemorySegment>();
                for (int target = 0; target < outputs.size(); target++) {
                    downloaded.add(arena.allocate(96L, Integer.BYTES));
                }
                MemorySegment sentinel = arena.allocate(96L, Integer.BYTES);
                for (int index = 0; index < ADVERSARIAL_BITS.length; index++) {
                    sentinel.setAtIndex(JAVA_INT, index, 0x5a5a0000 | index);
                }
                for (MetalNativeApi.Handle output : outputs) {
                    api.upload(output, 0L, sentinel, 96L);
                }
                MetalNativeApi.Handle runExecutable = executable;

                inputs.setAtIndex(ADDRESS, 1, input.carrier());
                assertIncompatibleResource(() -> api.runExecutable(
                        runExecutable, 5, inputs, 4, targetAddresses));
                inputs.setAtIndex(ADDRESS, 1, outputs.getFirst().carrier());

                targetAddresses.setAtIndex(
                        ADDRESS, 0, outputs.get(1).carrier());
                assertIncompatibleResource(() -> api.runExecutable(
                        runExecutable, 5, inputs, 4, targetAddresses));
                targetAddresses.setAtIndex(
                        ADDRESS, 0, outputs.getFirst().carrier());

                targetAddresses.setAtIndex(
                        ADDRESS, 0, outputs.get(1).carrier());
                targetAddresses.setAtIndex(
                        ADDRESS, 1, outputs.getFirst().carrier());
                assertIncompatibleResource(() -> api.runExecutable(
                        runExecutable, 5, inputs, 4, targetAddresses));
                targetAddresses.setAtIndex(
                        ADDRESS, 0, outputs.getFirst().carrier());
                targetAddresses.setAtIndex(
                        ADDRESS, 1, outputs.get(1).carrier());

                targetAddresses.setAtIndex(
                        ADDRESS, 1, outputs.getFirst().carrier());
                assertIncompatibleResource(() -> api.runExecutable(
                        runExecutable, 5, inputs, 4, targetAddresses));
                targetAddresses.setAtIndex(
                        ADDRESS, 1, outputs.get(1).carrier());

                MemorySegment unchangedInput = arena.allocate(96L, Integer.BYTES);
                api.download(input, 0L, unchangedInput, 96L);
                for (int target = 0; target < outputs.size(); target++) {
                    api.download(
                            outputs.get(target),
                            0L,
                            downloaded.get(target),
                            96L);
                }
                for (int index = 0; index < ADVERSARIAL_BITS.length; index++) {
                    assertEquals(
                            Integer.toUnsignedLong(ADVERSARIAL_BITS[index]),
                            Integer.toUnsignedLong(
                                    unchangedInput.getAtIndex(JAVA_INT, index)),
                            "custom alias rejection must preserve input " + index);
                    for (int target = 0; target < outputs.size(); target++) {
                        assertEquals(
                                Integer.toUnsignedLong(0x5a5a0000 | index),
                                Integer.toUnsignedLong(
                                        downloaded.get(target)
                                                .getAtIndex(JAVA_INT, index)),
                                "custom alias rejection must preserve target "
                                        + target + '[' + index + ']');
                    }
                }
                for (int iteration = 0; iteration < 2; iteration++) {
                    api.runExecutable(executable, 5, inputs, 4, targetAddresses);
                    for (int target = 0; target < outputs.size(); target++) {
                        api.download(outputs.get(target), 0L, downloaded.get(target), 96L);
                    }
                    for (int outputIndex = 0; outputIndex < ADVERSARIAL_BITS.length;
                            outputIndex++) {
                        int k = outputIndex / 6;
                        int remainder = outputIndex % 6;
                        int i = remainder / 3;
                        int j = remainder % 3;
                        int sourceIndex = i * 12 + j * 4 + k;
                        int expected = ADVERSARIAL_BITS[sourceIndex];
                        assertEquals(Integer.toUnsignedLong(expected),
                                Integer.toUnsignedLong(
                                        downloaded.get(0).getAtIndex(JAVA_INT, outputIndex)));
                        assertEquals(Integer.toUnsignedLong(expected),
                                Integer.toUnsignedLong(
                                        downloaded.get(1).getAtIndex(JAVA_INT, outputIndex)));
                        assertEquals(Integer.toUnsignedLong(expected ^ 0x80000000),
                                Integer.toUnsignedLong(
                                        downloaded.get(2).getAtIndex(JAVA_INT, outputIndex)));
                        assertEquals(Integer.toUnsignedLong(expected ^ 0x80000000),
                                Integer.toUnsignedLong(
                                        downloaded.get(3).getAtIndex(JAVA_INT, outputIndex)));
                    }
                }
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) {
                api.releaseBuffer(outputs.get(index));
            }
            if (input != null) {
                api.releaseBuffer(input);
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
            if (context != null) {
                api.releaseContext(context);
            }
            api.close();
        }
    }

    @Test
    void publicSelectAndSliceBoundaryTransfersPreservePhysicalPrefixGapsAndLogicalPlacement() {
        Path library = configuredLibrary();
        try (Arena arena = Arena.ofConfined();
                MetalBackendIntegration integration = MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library))) {
            Shape selectShape = Shape.of(4);
            TensorDescriptor selectDescriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    selectShape,
                    java.util.Optional.of(
                            LayoutDescriptor.of(selectShape, new long[] {1}, 4L, true)),
                    false);
            MemorySegment selectStorage = arena.allocate(8L * Integer.BYTES, Integer.BYTES);
            for (int index = 0; index < 8; index++) {
                selectStorage.setAtIndex(
                        JAVA_INT, index, index < 4 ? 0x5a5a5a5a : ADVERSARIAL_BITS[index]);
            }
            try (BufferRepresentation select = integration.borrow(
                    new MemorySegmentStorage(DataType.FLOAT32, 8L, selectStorage))) {
                assertTrue(integration.acceptsStorageLayoutTransfer(
                        select, selectDescriptor));
                MemorySegment downloaded =
                        arena.allocate(8L * Integer.BYTES, Integer.BYTES);
                integration.bindStorageLayoutDownload(
                        select, selectDescriptor).accept(downloaded);
                for (int index = 0; index < 8; index++) {
                    assertEquals(
                            Integer.toUnsignedLong(selectStorage.getAtIndex(JAVA_INT, index)),
                            Integer.toUnsignedLong(downloaded.getAtIndex(JAVA_INT, index)));
                }
            }

            Shape sliceShape = Shape.of(2, 2);
            TensorDescriptor sliceDescriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    sliceShape,
                    java.util.Optional.of(
                            LayoutDescriptor.of(sliceShape, new long[] {8, 2}, 5L, true)),
                    false);
            MemorySegment sliceStorage = arena.allocate(16L * Integer.BYTES, Integer.BYTES);
            MemorySegment replacement = arena.allocate(16L * Integer.BYTES, Integer.BYTES);
            for (int index = 0; index < 16; index++) {
                sliceStorage.setAtIndex(JAVA_INT, index, 0x6b6b6b6b);
                replacement.setAtIndex(JAVA_INT, index, 0x7c7c7c7c + index);
            }
            int[] logicalIndices = {5, 7, 13, 15};
            for (int index = 0; index < logicalIndices.length; index++) {
                sliceStorage.setAtIndex(
                        JAVA_INT, logicalIndices[index], ADVERSARIAL_BITS[index]);
            }
            try (BufferRepresentation slice = integration.borrow(
                    new MemorySegmentStorage(DataType.FLOAT32, 16L, sliceStorage))) {
                assertTrue(integration.acceptsStorageLayoutTransfer(
                        slice, sliceDescriptor));
                integration.bindStorageLayoutUpload(
                        slice, sliceDescriptor).accept(replacement);
                MemorySegment downloaded =
                        arena.allocate(16L * Integer.BYTES, Integer.BYTES);
                integration.bindStorageLayoutDownload(
                        slice, sliceDescriptor).accept(downloaded);
                for (int index = 0; index < 16; index++) {
                    assertEquals(
                            Integer.toUnsignedLong(replacement.getAtIndex(JAVA_INT, index)),
                            Integer.toUnsignedLong(downloaded.getAtIndex(JAVA_INT, index)));
                }
            }
        }
    }

    @Test
    void publicStorageTransferBoundaryRejectsUnresolvedZeroStrideAndOverlappingLayouts() {
        Path library = configuredLibrary();
        try (Arena arena = Arena.ofConfined();
                MetalBackendIntegration integration = MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library))) {
            Shape shape = Shape.of(2, 2);
            TensorDescriptor unresolved = new TensorDescriptor(
                    DataType.FLOAT32, shape, java.util.Optional.empty(), false);
            TensorDescriptor broadcast = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    java.util.Optional.of(
                            LayoutDescriptor.of(shape, new long[] {0, 1}, 0L, true)),
                    false);
            TensorDescriptor overlapping = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    java.util.Optional.of(
                            LayoutDescriptor.of(shape, new long[] {1, 1}, 0L, true)),
                    false);
            try (BufferRepresentation twoElements = integration.borrow(
                            new MemorySegmentStorage(
                                    DataType.FLOAT32,
                                    2L,
                                    arena.allocate(2L * Integer.BYTES, Integer.BYTES)));
                    BufferRepresentation threeElements = integration.borrow(
                            new MemorySegmentStorage(
                                    DataType.FLOAT32,
                                    3L,
                                    arena.allocate(3L * Integer.BYTES, Integer.BYTES)))) {
                assertFalse(integration.acceptsStorageLayoutTransfer(
                        twoElements, unresolved));
                assertFalse(integration.acceptsStorageLayoutTransfer(
                        twoElements, broadcast));
                assertFalse(integration.acceptsStorageLayoutTransfer(
                        threeElements, overlapping));
            }
            Shape overflowShape = Shape.of(1);
            TensorDescriptor overflow = new TensorDescriptor(
                    DataType.FLOAT64,
                    overflowShape,
                    java.util.Optional.of(LayoutDescriptor.of(
                            overflowShape,
                            new long[] {1L},
                            Long.MAX_VALUE - 1L,
                            true)),
                    false);
            try (BufferRepresentation oneElement = integration.borrow(
                    new MemorySegmentStorage(
                            DataType.FLOAT64,
                            1L,
                            arena.allocate(Double.BYTES, Double.BYTES)))) {
                assertFalse(integration.acceptsStorageLayoutTransfer(oneElement, overflow));
                assertThrows(IllegalArgumentException.class,
                        () -> integration.bindStorageLayoutDownload(oneElement, overflow));
                assertThrows(IllegalArgumentException.class,
                        () -> integration.bindStorageLayoutUpload(oneElement, overflow));
            }
            assertThrows(
                    IllegalArgumentException.class,
                    () -> LayoutDescriptor.of(shape, new long[] {-1, 1}, 0L, true));
        }
    }

    @Test
    void realSelectSliceChainUsesEncodedStorageForAllCarriersAndPreservesHoles() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            Shape inputShape = Shape.of(3, 4);
            Shape selectShape = Shape.of(4);
            Shape sliceShape = Shape.of(2);
            LayoutDescriptor inputLayout =
                    LayoutDescriptor.of(inputShape, new long[] {6, 1}, 2L, true);
            LayoutDescriptor selectLayout =
                    LayoutDescriptor.of(selectShape, new long[] {1}, 8L, true);
            LayoutDescriptor sliceLayout =
                    LayoutDescriptor.of(sliceShape, new long[] {2}, 8L, true);
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.SELECT,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.SELECT,
                            new long[] {0, 1}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.SLICE,
                            new int[] {1},
                            new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.SLICE,
                            new long[] {1, 0, 2, 0, 2})));
            for (DataType type : DataType.values()) {
                runSelectSliceCarrier(
                        api,
                        context,
                        type,
                        inputShape,
                        inputLayout,
                        selectShape,
                        selectLayout,
                        sliceShape,
                        sliceLayout,
                        program);
            }
        } finally {
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void runSelectSliceCarrier(
            MetalNativeApi api,
            MetalNativeApi.Handle context,
            DataType type,
            Shape inputShape,
            LayoutDescriptor inputLayout,
            Shape selectShape,
            LayoutDescriptor selectLayout,
            Shape sliceShape,
            LayoutDescriptor sliceLayout,
            MetalMpsGraphProgram program) {
        int width = type.byteWidth();
        long inputBytes = Math.multiplyExact(inputLayout.referencedElementSpan(), width);
        long selectBytes = Math.multiplyExact(selectLayout.referencedElementSpan(), width);
        long sliceBytes = Math.multiplyExact(sliceLayout.referencedElementSpan(), width);
        var values = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        type,
                        inputShape.toLongArray(),
                        java.util.Optional.of(inputLayout),
                        false,
                        false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        type,
                        selectShape.toLongArray(),
                        java.util.Optional.of(selectLayout),
                        false,
                        false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        type,
                        sliceShape.toLongArray(),
                        java.util.Optional.of(sliceLayout),
                        false,
                        false));
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        MetalNativeApi.Handle selected = null;
        MetalNativeApi.Handle sliced = null;
        try {
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    values,
                    program,
                    new int[] {0},
                    new int[] {1, 2},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            input = api.createBuffer(context, inputBytes);
            selected = api.createBuffer(context, selectBytes);
            sliced = api.createBuffer(context, sliceBytes);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment source = arena.allocate(inputBytes, width);
                MemorySegment selectedSentinel = arena.allocate(selectBytes, width);
                MemorySegment slicedSentinel = arena.allocate(sliceBytes, width);
                source.fill((byte) 0x5a);
                selectedSentinel.fill((byte) 0x6b);
                slicedSentinel.fill((byte) 0x7c);
                for (int row = 0; row < 3; row++) {
                    for (int column = 0; column < 4; column++) {
                        long element = 2L + row * 6L + column;
                        for (int byteIndex = 0; byteIndex < width; byteIndex++) {
                            byte value = type == DataType.BOOL
                                    ? (byte) ((row + column) & 1)
                                    : (byte) (0x10 + row * 4 * width
                                            + column * width + byteIndex);
                            source.set(
                                    JAVA_BYTE,
                                    Math.multiplyExact(element, width) + byteIndex,
                                    value);
                        }
                    }
                }
                api.upload(input, 0L, source, inputBytes);
                MemorySegment inputs = arena.allocate(ADDRESS, 3);
                MemorySegment outputs = arena.allocate(ADDRESS, 2);
                inputs.setAtIndex(ADDRESS, 0L, input.carrier());
                inputs.setAtIndex(ADDRESS, 1L, selected.carrier());
                inputs.setAtIndex(ADDRESS, 2L, sliced.carrier());
                outputs.setAtIndex(ADDRESS, 0L, selected.carrier());
                outputs.setAtIndex(ADDRESS, 1L, sliced.carrier());
                MemorySegment actualInput = arena.allocate(inputBytes, width);
                MemorySegment actualSelected = arena.allocate(selectBytes, width);
                MemorySegment actualSliced = arena.allocate(sliceBytes, width);
                for (int iteration = 0; iteration < 2; iteration++) {
                    api.upload(selected, 0L, selectedSentinel, selectBytes);
                    api.upload(sliced, 0L, slicedSentinel, sliceBytes);
                    api.runExecutable(executable, 3, inputs, 2, outputs);
                    api.download(input, 0L, actualInput, inputBytes);
                    api.download(selected, 0L, actualSelected, selectBytes);
                    api.download(sliced, 0L, actualSliced, sliceBytes);
                    assertRawBytesEqual(source, actualInput, inputBytes);
                    for (long element = 0L;
                            element < selectLayout.referencedElementSpan();
                            element++) {
                        int column = Math.toIntExact(element - 8L);
                        boolean logical = column >= 0 && column < 4;
                        for (int byteIndex = 0; byteIndex < width; byteIndex++) {
                            byte expected = logical
                                    ? source.get(
                                            JAVA_BYTE,
                                            Math.multiplyExact(8L + column, width)
                                                    + byteIndex)
                                    : (byte) 0x6b;
                            assertEquals(
                                    expected,
                                    actualSelected.get(
                                            JAVA_BYTE,
                                            Math.multiplyExact(element, width) + byteIndex),
                                    type + " SELECT element=" + element);
                        }
                    }
                    for (long element = 0L;
                            element < sliceLayout.referencedElementSpan();
                            element++) {
                        boolean logical = element == 8L || element == 10L;
                        int column = element == 10L ? 2 : 0;
                        for (int byteIndex = 0; byteIndex < width; byteIndex++) {
                            byte expected = logical
                                    ? source.get(
                                            JAVA_BYTE,
                                            Math.multiplyExact(8L + column, width)
                                                    + byteIndex)
                                    : (byte) 0x7c;
                            assertEquals(
                                    expected,
                                    actualSliced.get(
                                            JAVA_BYTE,
                                            Math.multiplyExact(element, width) + byteIndex),
                                    type + " SLICE element=" + element);
                        }
                    }
                }
            }
        } finally {
            if (sliced != null) api.releaseBuffer(sliced);
            if (selected != null) api.releaseBuffer(selected);
            if (input != null) api.releaseBuffer(input);
            if (executable != null) api.releaseExecutable(executable);
        }
    }

    private static void assertRawBytesEqual(
            MemorySegment expected, MemorySegment actual, long byteCount) {
        for (long offset = 0L; offset < byteCount; offset++) {
            assertEquals(
                    expected.get(JAVA_BYTE, offset),
                    actual.get(JAVA_BYTE, offset),
                    "raw byte offset=" + offset);
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private static void runCase(
            MetalNativeApi api, MetalNativeApi.Handle context, AffineCase test) {
        int inputCount = elementCount(test.inputShape());
        int outputCount = elementCount(test.outputShape());
        long inputBytes = Math.multiplyExact((long) inputCount, Float.BYTES);
        long outputBytes = Math.multiplyExact((long) outputCount, Float.BYTES);
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        MetalNativeApi.Handle output = null;
        try {
            int[] ranks = {test.inputShape().length, test.outputShape().length};
            long[] dimensions = new long[32];
            System.arraycopy(test.inputShape(), 0, dimensions, 0, test.inputShape().length);
            System.arraycopy(test.outputShape(), 0, dimensions, 16, test.outputShape().length);
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks, dimensions, new MetalMpsGraphProgram(List.of(test.node()))), new MetalMpsGraphProgram(List.of(test.node())), new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
            input = api.createBuffer(context, inputBytes);
            output = api.createBuffer(context, outputBytes);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment source = arena.allocate(inputBytes, Integer.BYTES);
                int[] sourceBits = new int[inputCount];
                for (int index = 0; index < inputCount; index++) {
                    sourceBits[index] = ADVERSARIAL_BITS[index % ADVERSARIAL_BITS.length];
                    source.setAtIndex(JAVA_INT, index, sourceBits[index]);
                }
                api.upload(input, 0L, source, inputBytes);
                MemorySegment allValues = arena.allocate(ADDRESS, 2);
                MemorySegment outputHandle = arena.allocate(ADDRESS);
        allValues.setAtIndex(ADDRESS, 0L, input.carrier());
        allValues.setAtIndex(ADDRESS, 1L, output.carrier());
                outputHandle.setAtIndex(ADDRESS, 0L, output.carrier());
                MemorySegment sentinel = arena.allocate(outputBytes, Integer.BYTES);
                MemorySegment actual = arena.allocate(outputBytes, Integer.BYTES);
                for (int iteration = 0; iteration < 2; iteration++) {
                    for (int index = 0; index < outputCount; index++) {
                        sentinel.setAtIndex(JAVA_INT, index, 0xdeadbeef);
                    }
                    api.upload(output, 0L, sentinel, outputBytes);
                    api.runExecutable(executable, 2, allValues, 1, outputHandle);
                    api.download(output, 0L, actual, outputBytes);
                    for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
                        int inputIndex = test.inputIndex(outputIndex);
                        assertEquals(
                                Integer.toUnsignedLong(sourceBits[inputIndex]),
                                Integer.toUnsignedLong(actual.getAtIndex(JAVA_INT, outputIndex)),
                                test.name() + " iteration=" + iteration
                                        + " outputIndex=" + outputIndex);
                    }
                }
            }
        } finally {
            if (output != null) {
                api.releaseBuffer(output);
            }
            if (input != null) {
                api.releaseBuffer(input);
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static List<AffineCase> cases() {
        return List.of(
                target("reshape-scalar-identity", Kind.RESHAPE, new long[] {}, new long[] {}),
                target("reshape-scalar-to-singleton", Kind.RESHAPE, new long[] {}, new long[] {1}),
                target("reshape-singleton-to-scalar", Kind.RESHAPE, new long[] {1}, new long[] {}),
                permutation("permute-scalar", new long[] {}, new long[] {}),
                axis("expand-dims-scalar", Kind.EXPAND_DIMS, new long[] {}, new long[] {1}, 0),
                axis("squeeze-to-scalar", Kind.SQUEEZE, new long[] {1}, new long[] {}, 0),
                target("reshape-identity", Kind.RESHAPE, new long[] {24}, new long[] {24}),
                target(
                        "reshape-shape-change",
                        Kind.RESHAPE,
                        new long[] {2, 3, 4},
                        new long[] {4, 6}),
                target("expand-leading", Kind.EXPAND, new long[] {3}, new long[] {2, 3}),
                target("expand-singleton", Kind.EXPAND, new long[] {2, 1, 3}, new long[] {2, 4, 3}),
                target(
                        "expand-multi-axis",
                        Kind.EXPAND,
                        new long[] {1, 2, 1},
                        new long[] {3, 2, 4}),
                target("expand-same-shape", Kind.EXPAND, new long[] {2, 3}, new long[] {2, 3}),
                permutation("permute-identity", new long[] {2, 3}, new long[] {2, 3}, 0, 1),
                permutation(
                        "permute-rank-two-transpose", new long[] {2, 3}, new long[] {3, 2}, 1, 0),
                permutation(
                        "permute-rank-three", new long[] {2, 3, 4}, new long[] {4, 2, 3}, 2, 0, 1),
                permutation(
                        "permute-rank-sixteen",
                        new long[] {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3},
                        new long[] {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2},
                        15,
                        14,
                        13,
                        12,
                        11,
                        10,
                        9,
                        8,
                        7,
                        6,
                        5,
                        4,
                        3,
                        2,
                        1,
                        0),
                axis(
                        "expand-dims-front",
                        Kind.EXPAND_DIMS,
                        new long[] {2, 3},
                        new long[] {1, 2, 3},
                        0),
                axis(
                        "expand-dims-middle",
                        Kind.EXPAND_DIMS,
                        new long[] {2, 3},
                        new long[] {2, 1, 3},
                        1),
                axis(
                        "expand-dims-end",
                        Kind.EXPAND_DIMS,
                        new long[] {2, 3},
                        new long[] {2, 3, 1},
                        2),
                axis("squeeze-front", Kind.SQUEEZE, new long[] {1, 2, 3}, new long[] {2, 3}, 0),
                axis("squeeze-middle", Kind.SQUEEZE, new long[] {2, 1, 3}, new long[] {2, 3}, 1),
                axis("squeeze-end", Kind.SQUEEZE, new long[] {2, 3, 1}, new long[] {2, 3}, 2));
    }

    private static AffineCase target(
            String name, Kind kind, long[] input, long[] output) {
        return new AffineCase(name, kind, input, output, new int[0]);
    }

    private static AffineCase permutation(
            String name, long[] input, long[] output, int... permutation) {
        return new AffineCase(name, Kind.PERMUTE, input, output, permutation);
    }

    private static AffineCase axis(
            String name, Kind kind, long[] input, long[] output, int axis) {
        return new AffineCase(name, kind, input, output, new int[] {axis});
    }

    private static void assertIncompatibleResource(
            org.junit.jupiter.api.function.Executable invocation) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class, invocation);
        assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
    }

    private enum Kind { RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE }

    private record AffineCase(
            String name,
            Kind kind,
            long[] inputShape,
            long[] outputShape,
            int[] attributes) {
        private AffineCase {
            inputShape = inputShape.clone();
            outputShape = outputShape.clone();
            attributes = attributes.clone();
        }

        @Override
        public long[] inputShape() {
            return inputShape.clone();
        }

        @Override
        public long[] outputShape() {
            return outputShape.clone();
        }

        MetalMpsGraphProgram.Node node() {
            return switch (kind) {
                case RESHAPE -> MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.RESHAPE,
                        0, 1, outputShape);
                case EXPAND -> MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND,
                        0, 1, outputShape);
                case PERMUTE -> {
                    var axes = new ArrayList<Integer>(attributes.length);
                    for (int axis : attributes) {
                        axes.add(axis);
                    }
                    yield MetalMpsGraphProgram.Node.permutation(0, 1, axes);
                }
                case EXPAND_DIMS -> MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.EXPAND_DIMS,
                        0, 1, attributes[0]);
                case SQUEEZE -> MetalMpsGraphProgram.Node.axis(
                        MetalMpsGraphProgram.NodeKind.SQUEEZE,
                        0, 1, attributes[0]);
            };
        }

        int inputIndex(int outputIndex) {
            if (kind == Kind.RESHAPE) {
                return outputIndex;
            }
            long[] outputCoordinates = coordinates(outputIndex, outputShape);
            long[] inputCoordinates = new long[inputShape.length];
            switch (kind) {
                case EXPAND -> {
                    int padding = outputShape.length - inputShape.length;
                    for (int axis = 0; axis < inputShape.length; axis++) {
                        inputCoordinates[axis] = inputShape[axis] == 1L
                                ? 0L : outputCoordinates[axis + padding];
                    }
                }
                case PERMUTE -> {
                    for (int axis = 0; axis < outputShape.length; axis++) {
                        inputCoordinates[attributes[axis]] = outputCoordinates[axis];
                    }
                }
                case EXPAND_DIMS -> {
                    int inserted = attributes[0];
                    for (int axis = 0; axis < inputShape.length; axis++) {
                        inputCoordinates[axis] = outputCoordinates[
                                axis < inserted ? axis : axis + 1];
                    }
                }
                case SQUEEZE -> {
                    int removed = attributes[0];
                    for (int axis = 0; axis < inputShape.length; axis++) {
                        inputCoordinates[axis] = axis == removed
                                ? 0L
                                : outputCoordinates[axis < removed ? axis : axis - 1];
                    }
                }
                case RESHAPE -> throw new AssertionError();
            }
            return Math.toIntExact(linear(inputCoordinates, inputShape));
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] result = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, result, value * 16, shapes[value].length);
        }
        return result;
    }

    private static int elementCount(long[] shape) {
        long count = 1L;
        for (long dimension : shape) {
            count = Math.multiplyExact(count, dimension);
        }
        return Math.toIntExact(count);
    }

    private static long[] coordinates(long index, long[] shape) {
        long[] result = new long[shape.length];
        for (int axis = shape.length - 1; axis >= 0; axis--) {
            result[axis] = index % shape[axis];
            index /= shape[axis];
        }
        return result;
    }

    private static long linear(long[] coordinates, long[] shape) {
        long result = 0L;
        for (int axis = 0; axis < shape.length; axis++) {
            result = Math.addExact(
                    Math.multiplyExact(result, shape[axis]), coordinates[axis]);
        }
        return result;
    }
}

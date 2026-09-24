package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

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
            executable = api.createMpsGraphExecutable(
                    context,
                    ranks,
                    dimensions,
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.permutation(
                                    0, 1, List.of(2, 0, 1)),
                            MetalMpsGraphProgram.Node.contiguous(1, 2),
                            MetalMpsGraphProgram.Node.neg(2, 3),
                            MetalMpsGraphProgram.Node.targetShape(
                                    MetalMpsGraphProgram.NodeKind.RESHAPE,
                                    3,
                                    4,
                                    new long[] {4, 6}))),
                    new int[] {0},
                    new int[] {1, 2, 3, 4});
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
                MemorySegment inputs = arena.allocate(ADDRESS);
                MemorySegment targetAddresses = arena.allocate(ADDRESS, outputs.size());
                inputs.setAtIndex(ADDRESS, 0L, input.carrier());
                for (int target = 0; target < outputs.size(); target++) {
                    targetAddresses.setAtIndex(
                            ADDRESS, target, outputs.get(target).carrier());
                }
                var downloaded = new ArrayList<MemorySegment>();
                for (int target = 0; target < outputs.size(); target++) {
                    downloaded.add(arena.allocate(96L, Integer.BYTES));
                }
                for (int iteration = 0; iteration < 2; iteration++) {
                    api.runExecutable(executable, 1, inputs, 4, targetAddresses);
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
    void affineViewStillRejectsCrossOwnerTransferWhileCanonicalTransferRemainsAccepted() {
        Path library = configuredLibrary();
        try (Arena arena = Arena.ofConfined();
                MetalBackendIntegration integration = MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library));
                BufferRepresentation representation = integration.borrow(
                        new MemorySegmentStorage(
                                DataType.FLOAT32, 6L, arena.allocate(24L, Float.BYTES)))) {
            Shape shape = Shape.of(2, 3);
            TensorDescriptor canonical = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    java.util.Optional.of(LayoutDescriptor.contiguous(shape)),
                    false);
            TensorDescriptor affineView = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    java.util.Optional.of(
                            LayoutDescriptor.of(shape, new long[] {3, 1}, 0L, true)),
                    false);

            assertTrue(integration.acceptsContiguousFloat32Transfer(
                    representation, canonical));
            assertFalse(integration.acceptsContiguousFloat32Transfer(
                    representation, affineView));
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
            executable = api.createMpsGraphExecutable(
                    context,
                    ranks,
                    dimensions,
                    new MetalMpsGraphProgram(List.of(test.node())),
                    new int[] {0},
                    new int[] {1});
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
                MemorySegment inputHandle = arena.allocate(ADDRESS);
                MemorySegment outputHandle = arena.allocate(ADDRESS);
                inputHandle.setAtIndex(ADDRESS, 0L, input.carrier());
                outputHandle.setAtIndex(ADDRESS, 0L, output.carrier());
                MemorySegment sentinel = arena.allocate(outputBytes, Integer.BYTES);
                MemorySegment actual = arena.allocate(outputBytes, Integer.BYTES);
                for (int iteration = 0; iteration < 2; iteration++) {
                    for (int index = 0; index < outputCount; index++) {
                        sentinel.setAtIndex(JAVA_INT, index, 0xdeadbeef);
                    }
                    api.upload(output, 0L, sentinel, outputBytes);
                    api.runExecutable(executable, 1, inputHandle, 1, outputHandle);
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
                target("reshape-identity", Kind.RESHAPE,
                        new long[] {24}, new long[] {24}),
                target("reshape-shape-change", Kind.RESHAPE,
                        new long[] {2, 3, 4}, new long[] {4, 6}),
                target("expand-leading", Kind.EXPAND,
                        new long[] {3}, new long[] {2, 3}),
                target("expand-singleton", Kind.EXPAND,
                        new long[] {2, 1, 3}, new long[] {2, 4, 3}),
                target("expand-multi-axis", Kind.EXPAND,
                        new long[] {1, 2, 1}, new long[] {3, 2, 4}),
                target("expand-same-shape", Kind.EXPAND,
                        new long[] {2, 3}, new long[] {2, 3}),
                permutation("permute-identity",
                        new long[] {2, 3}, new long[] {2, 3}, 0, 1),
                permutation("permute-rank-two-transpose",
                        new long[] {2, 3}, new long[] {3, 2}, 1, 0),
                permutation("permute-rank-three",
                        new long[] {2, 3, 4}, new long[] {4, 2, 3}, 2, 0, 1),
                permutation("permute-rank-sixteen",
                        new long[] {2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3},
                        new long[] {3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2},
                        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
                axis("expand-dims-front", Kind.EXPAND_DIMS,
                        new long[] {2, 3}, new long[] {1, 2, 3}, 0),
                axis("expand-dims-middle", Kind.EXPAND_DIMS,
                        new long[] {2, 3}, new long[] {2, 1, 3}, 1),
                axis("expand-dims-end", Kind.EXPAND_DIMS,
                        new long[] {2, 3}, new long[] {2, 3, 1}, 2),
                axis("squeeze-front", Kind.SQUEEZE,
                        new long[] {1, 2, 3}, new long[] {2, 3}, 0),
                axis("squeeze-middle", Kind.SQUEEZE,
                        new long[] {2, 1, 3}, new long[] {2, 3}, 1),
                axis("squeeze-end", Kind.SQUEEZE,
                        new long[] {2, 3, 1}, new long[] {2, 3}, 2));
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

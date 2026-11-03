package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Optional real-native consumer checks. The caller supplies SYNAPTIK_ONEDNN_LIBRARY; these tests
 * neither discover libraries nor change process configuration. Integer operands keep every product
 * and every possible contraction tree exact, independently of the vendor's reduction strategy.
 */
final class CpuOneDnnNativeContractTest {
    private static final byte CANARY = (byte) 0x5a;
    private static final long GUARD_BYTES = 64;

    @ParameterizedTest
    @EnumSource(value = CpuOneDnnRoutePlan.Api.class, names = {"PRIMITIVE", "GRAPH"})
    void bareMatmulUsesCurrentWeightsAtTheSameAddressesWithFreshWorkspaces(
            CpuOneDnnRoutePlan.Api api) {
        var spec = matmul(dense(2, 3), dense(3, 2), dense(2, 2));
        try (CpuOneDnnLibrary library = openLibrary();
                CpuOneDnnKernel kernel = CpuOneDnnCompiler.compile(library, spec, api);
                Arena arena = Arena.ofShared()) {
            var left = buffer(arena, spec.inputs().get(0));
            var right = buffer(arena, spec.inputs().get(1));
            var output = buffer(arena, spec.output());
            int[] a = {1, 2, 3, 4, 5, 6};
            int[] firstWeights = {7, 8, 9, 10, 11, 12};
            int[] secondWeights = {-2, 3, 1, -4, 5, 2};
            put(left, a);
            put(right, firstWeights);
            long leftAddress = left.data.address();
            long rightAddress = right.data.address();
            long outputAddress = output.data.address();
            try (CpuOneDnnWorkspace first = kernel.createWorkspace()) {
                // Read-only native inputs are valid: only the output is written.
                first.bind(new MemorySegment[] {left.data.asReadOnly(), right.data.asReadOnly()}, output.data);
                first.execute();
                assertOutput(output, new int[] {58, 64, 139, 154});
            }
            put(right, secondWeights);
            output.reset();
            try (CpuOneDnnWorkspace second = kernel.createWorkspace()) {
                second.bind(new MemorySegment[] {left.data, right.data}, output.data);
                second.execute();
                assertOutput(output, new int[] {15, 1, 27, 4});
            }
            assertEquals(leftAddress, left.data.address());
            assertEquals(rightAddress, right.data.address());
            assertEquals(outputAddress, output.data.address());
            assertInput(left, a);
            assertInput(right, secondWeights);
        }
    }

    @ParameterizedTest
    @EnumSource(value = CpuOneDnnRoutePlan.Api.class, names = {"PRIMITIVE", "GRAPH"})
    void cnnGroupedBroadcastAndTransposedOffsetStridedViewsMatchIntegerOracle(
            CpuOneDnnRoutePlan.Api api) {
        List<CpuOneDnnKernelSpec> cases = List.of(
                // Grouped convolution's unfolded forward: batch-broadcast group weights.
                matmul(dense(2, 3, 2, 4), dense(1, 3, 4, 3), dense(2, 3, 2, 3)),
                // Input cotangent: dense last-two-axis-transposed weights.
                matmul(dense(2, 3, 2, 3), tensor(new long[] {1, 3, 3, 4}, 36, 12, 1, 3),
                        dense(2, 3, 2, 4)),
                // Weight cotangent contraction, before the ordinary batch reduction.
                matmul(tensor(new long[] {2, 3, 4, 2}, 24, 8, 1, 4), dense(2, 3, 2, 3),
                        dense(2, 3, 4, 3)),
                // Both broadcast axes are active in different operands.
                matmul(dense(2, 1, 2, 3), dense(1, 3, 3, 2), dense(2, 3, 2, 2)));
        try (CpuOneDnnLibrary library = openLibrary()) {
            for (CpuOneDnnKernelSpec spec : cases) {
                try (CpuOneDnnKernel kernel = CpuOneDnnCompiler.compile(library, spec, api);
                        Arena arena = Arena.ofShared()) {
                    var left = buffer(arena, spec.inputs().get(0));
                    var right = buffer(arena, spec.inputs().get(1));
                    var output = buffer(arena, spec.output());
                    int[] a = integers(elements(left.tensor), 1);
                    int[] b = integers(elements(right.tensor), 4);
                    put(left, a);
                    put(right, b);
                    try (CpuOneDnnWorkspace workspace = kernel.createWorkspace()) {
                        workspace.bind(new MemorySegment[] {left.data, right.data}, output.data);
                        workspace.execute();
                        assertOutput(output, oracle(spec, a, b));
                    }
                    assertInput(left, a);
                    assertInput(right, b);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(value = CpuOneDnnRoutePlan.Api.class, names = {"PRIMITIVE", "GRAPH"})
    void issuedWorkspaceSurvivesKernelAndLibraryOwnerClose(CpuOneDnnRoutePlan.Api api) {
        var spec = matmul(dense(2, 3), dense(3, 2), dense(2, 2));
        try (CpuOneDnnLibrary library = openLibrary();
                CpuOneDnnKernel kernel = CpuOneDnnCompiler.compile(library, spec, api);
                Arena arena = Arena.ofShared();
                CpuOneDnnWorkspace workspace = kernel.createWorkspace()) {
            var left = buffer(arena, spec.inputs().get(0));
            var right = buffer(arena, spec.inputs().get(1));
            var output = buffer(arena, spec.output());
            int[] a = integers(6, 1);
            int[] b = integers(6, 2);
            put(left, a);
            put(right, b);
            kernel.close();
            assertThrows(IllegalStateException.class, () -> {
                try (CpuOneDnnWorkspace unexpected = kernel.createWorkspace()) {
                    fail("closed kernel issued a new workspace");
                }
            });
            library.close();
            // Binding after both owners close must still use the already-issued native lease.
            workspace.bind(new MemorySegment[] {left.data, right.data}, output.data);
            workspace.execute();
            assertOutput(output, oracle(spec, a, b));
            b = integers(6, 5);
            put(right, b);
            output.reset();
            workspace.execute();
            assertOutput(output, oracle(spec, a, b));
            workspace.close();
            assertThrows(IllegalStateException.class, workspace::execute);
            assertThrows(IllegalStateException.class,
                    () -> workspace.bind(new MemorySegment[] {left.data, right.data}, output.data));
            assertThrows(IllegalStateException.class, () -> {
                try (CpuOneDnnWorkspace unexpected = kernel.createWorkspace()) {
                    fail("closed kernel issued a new workspace");
                }
            });
        }
    }

    @ParameterizedTest
    @EnumSource(value = CpuOneDnnRoutePlan.Api.class, names = {"PRIMITIVE", "GRAPH"})
    void independentlyIssuedWorkspacesKeepTheirOwnMutableBindings(CpuOneDnnRoutePlan.Api api) {
        var spec = matmul(dense(2, 3), dense(3, 2), dense(2, 2));
        try (CpuOneDnnLibrary library = openLibrary();
                CpuOneDnnKernel kernel = CpuOneDnnCompiler.compile(library, spec, api);
                Arena arena = Arena.ofShared();
                CpuOneDnnWorkspace first = kernel.createWorkspace();
                CpuOneDnnWorkspace second = kernel.createWorkspace()) {
            var a1 = buffer(arena, spec.inputs().get(0));
            var b1 = buffer(arena, spec.inputs().get(1));
            var out1 = buffer(arena, spec.output());
            var a2 = buffer(arena, spec.inputs().get(0));
            var b2 = buffer(arena, spec.inputs().get(1));
            var out2 = buffer(arena, spec.output());
            int[] av1 = integers(6, 1), bv1 = integers(6, 3);
            int[] av2 = integers(6, 5), bv2 = integers(6, 6);
            put(a1, av1);
            put(b1, bv1);
            put(a2, av2);
            put(b2, bv2);
            byte[] untouchedSecond = out2.snapshot();
            first.bind(new MemorySegment[] {a1.data, b1.data}, out1.data);
            second.bind(new MemorySegment[] {a2.data, b2.data}, out2.data);
            first.execute();
            assertOutput(out1, oracle(spec, av1, bv1));
            assertArrayEquals(untouchedSecond, out2.snapshot(), "first execution must not use second's handles");
            second.execute();
            assertOutput(out2, oracle(spec, av2, bv2));
            // Rebinding one workspace cannot change an already-bound sibling's output or inputs.
            first.bind(new MemorySegment[] {a2.data, b1.data}, out1.data);
            out2.reset();
            second.execute();
            assertOutput(out2, oracle(spec, av2, bv2));
            first.execute();
            assertOutput(out1, oracle(spec, av2, bv1));
            first.close();
            out2.reset();
            second.bind(new MemorySegment[] {a1.data, b2.data}, out2.data);
            second.execute();
            assertOutput(out2, oracle(spec, av1, bv2));
            assertInput(a1, av1);
            assertInput(b1, bv1);
            assertInput(a2, av2);
            assertInput(b2, bv2);
        }
    }

    @ParameterizedTest
    @EnumSource(value = CpuOneDnnRoutePlan.Api.class, names = {"PRIMITIVE", "GRAPH"})
    void invalidBuffersAreRejectedBeforeAnyWrite(CpuOneDnnRoutePlan.Api api) {
        var spec = matmul(dense(2, 3), dense(3, 2), dense(2, 2));
        try (CpuOneDnnLibrary library = openLibrary();
                CpuOneDnnKernel kernel = CpuOneDnnCompiler.compile(library, spec, api);
                Arena arena = Arena.ofShared()) {
            for (InvalidBinding invalid : InvalidBinding.values()) {
                var left = buffer(arena, spec.inputs().get(0));
                var right = buffer(arena, spec.inputs().get(1));
                var output = buffer(arena, spec.output());
                put(left, integers(6, 1));
                put(right, integers(6, 2));
                byte[] leftBefore = left.snapshot(), rightBefore = right.snapshot();
                byte[] outputBefore = output.snapshot();
                float[] heap = {31, 32, 33, 34, 35, 36};
                float[] heapBefore = heap.clone();
                MemorySegment a = left.data, b = right.data, out = output.data;
                switch (invalid) {
                    case READ_ONLY_OUTPUT -> out = out.asReadOnly();
                    case HEAP_OUTPUT -> out = MemorySegment.ofArray(heap);
                    case SMALL_OUTPUT -> out = out.asSlice(0, spec.output().byteSize() - Float.BYTES);
                    case HEAP_LEFT -> a = MemorySegment.ofArray(heap);
                    case HEAP_RIGHT -> b = MemorySegment.ofArray(heap);
                    case SMALL_LEFT -> a = a.asSlice(0, spec.inputs().get(0).byteSize() - Float.BYTES);
                    case SMALL_RIGHT -> b = b.asSlice(0, spec.inputs().get(1).byteSize() - Float.BYTES);
                    case OVERLAP_LEFT -> out = a.asSlice(0, spec.output().byteSize());
                    case OVERLAP_RIGHT -> out = b.asSlice(0, spec.output().byteSize());
                    case PARTIAL_OVERLAP_LEFT -> out = a.asSlice(Float.BYTES, spec.output().byteSize());
                    case PARTIAL_OVERLAP_RIGHT -> out = b.asSlice(Float.BYTES, spec.output().byteSize());
                }
                MemorySegment[] inputs = {a, b};
                MemorySegment destination = out;
                try (CpuOneDnnWorkspace workspace = kernel.createWorkspace()) {
                    assertThrows(IllegalArgumentException.class,
                            () -> workspace.bind(inputs, destination), invalid.name());
                    assertThrows(IllegalStateException.class, workspace::execute, invalid.name());
                }
                assertArrayEquals(leftBefore, left.snapshot(), invalid.name());
                assertArrayEquals(rightBefore, right.snapshot(), invalid.name());
                assertArrayEquals(outputBefore, output.snapshot(), invalid.name());
                assertArrayEquals(heapBefore, heap, invalid.name());
            }
        }
    }

    private enum InvalidBinding {
        READ_ONLY_OUTPUT, HEAP_OUTPUT, SMALL_OUTPUT, HEAP_LEFT, HEAP_RIGHT, SMALL_LEFT, SMALL_RIGHT,
        OVERLAP_LEFT, OVERLAP_RIGHT, PARTIAL_OVERLAP_LEFT, PARTIAL_OVERLAP_RIGHT
    }

    private static CpuOneDnnLibrary openLibrary() {
        String requested = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(requested != null && !requested.isBlank(), "SYNAPTIK_ONEDNN_LIBRARY is not supplied");
        Path path = Path.of(requested);
        assertTrue(path.isAbsolute(), "SYNAPTIK_ONEDNN_LIBRARY must be an absolute path");
        return CpuOneDnnLibrary.open(path);
    }

    private static CpuOneDnnKernelSpec matmul(CpuOneDnnKernelSpec.Tensor left,
            CpuOneDnnKernelSpec.Tensor right, CpuOneDnnKernelSpec.Tensor output) {
        return new CpuOneDnnKernelSpec(CpuOneDnnKernelSpec.Operation.MATMUL, List.of(left, right),
                output, 1, List.of(), List.of(), List.of(), List.of(),
                CpuOneDnnKernelSpec.Bias.NONE, CpuOneDnnKernelSpec.Activation.NONE);
    }

    private static CpuOneDnnKernelSpec.Tensor dense(long... dimensions) {
        long[] strides = new long[dimensions.length];
        long next = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            strides[axis] = next;
            next *= dimensions[axis];
        }
        return tensor(dimensions, strides);
    }

    private static CpuOneDnnKernelSpec.Tensor tensor(long[] dimensions, long... strides) {
        return new CpuOneDnnKernelSpec.Tensor(
                java.util.Arrays.stream(dimensions).boxed().toList(),
                java.util.Arrays.stream(strides).boxed().toList());
    }

    private static int elements(CpuOneDnnKernelSpec.Tensor tensor) {
        int count = 1;
        for (long dimension : tensor.dimensions()) count *= Math.toIntExact(dimension);
        return count;
    }

    private static int[] integers(int count, int salt) {
        int[] values = new int[count];
        for (int index = 0; index < count; index++) values[index] = (index * 3 + salt) % 11 - 5;
        return values;
    }

    // Oracle indexes only logical integer arrays, never native descriptors' strides or output data.
    private static int[] oracle(CpuOneDnnKernelSpec spec, int[] left, int[] right) {
        int[] a = fourDimensions(spec.inputs().get(0));
        int[] b = fourDimensions(spec.inputs().get(1));
        int[] c = fourDimensions(spec.output());
        int[] result = new int[elements(spec.output())];
        int destination = 0;
        for (int batch = 0; batch < c[0]; batch++) {
            for (int group = 0; group < c[1]; group++) {
                for (int row = 0; row < c[2]; row++) {
                    for (int column = 0; column < c[3]; column++) {
                        int sum = 0;
                        for (int k = 0; k < a[3]; k++) {
                            int ai = (((batch % a[0]) * a[1] + group % a[1]) * a[2] + row) * a[3] + k;
                            int bi = (((batch % b[0]) * b[1] + group % b[1]) * b[2] + k) * b[3] + column;
                            sum += left[ai] * right[bi];
                        }
                        result[destination++] = sum;
                    }
                }
            }
        }
        return result;
    }

    private static int[] fourDimensions(CpuOneDnnKernelSpec.Tensor tensor) {
        int[] dimensions = {1, 1, 1, 1};
        int first = 4 - tensor.dimensions().size();
        for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
            dimensions[first + axis] = Math.toIntExact(tensor.dimensions().get(axis));
        }
        return dimensions;
    }

    private static long byteOffset(CpuOneDnnKernelSpec.Tensor tensor, int logicalIndex) {
        long offset = 0;
        for (int axis = tensor.dimensions().size() - 1; axis >= 0; axis--) {
            long dimension = tensor.dimensions().get(axis);
            offset += (logicalIndex % dimension) * tensor.strides().get(axis);
            logicalIndex /= Math.toIntExact(dimension);
        }
        return offset * Float.BYTES;
    }

    private static GuardedBuffer buffer(Arena arena, CpuOneDnnKernelSpec.Tensor tensor) {
        // Include the complete outer-stride extent as well as the referenced span. Native memory
        // descriptors may require that extent even though only the logical cells may be changed.
        long capacity = tensor.byteSize();
        for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
            capacity = Math.max(capacity,
                    tensor.dimensions().get(axis) * tensor.strides().get(axis) * Float.BYTES);
        }
        MemorySegment allocation = arena.allocate(GUARD_BYTES + capacity + GUARD_BYTES, 64);
        var result = new GuardedBuffer(tensor, allocation, allocation.asSlice(GUARD_BYTES, capacity));
        result.reset();
        return result;
    }

    private static void put(GuardedBuffer buffer, int[] values) {
        assertEquals(elements(buffer.tensor), values.length);
        for (int index = 0; index < values.length; index++) {
            buffer.data.set(JAVA_FLOAT, byteOffset(buffer.tensor, index), values[index]);
        }
    }

    private static void assertInput(GuardedBuffer buffer, int[] expected) {
        assertOutput(buffer, expected);
    }

    private static void assertOutput(GuardedBuffer buffer, int[] expected) {
        assertEquals(elements(buffer.tensor), expected.length);
        boolean[] occupied = new boolean[Math.toIntExact(buffer.allocation.byteSize())];
        for (int index = 0; index < expected.length; index++) {
            long offset = byteOffset(buffer.tensor, index);
            assertEquals((float) expected[index], buffer.data.get(JAVA_FLOAT, offset),
                    "logical element " + index + " of " + buffer.tensor);
            for (int valueByte = 0; valueByte < Float.BYTES; valueByte++) {
                occupied[Math.toIntExact(GUARD_BYTES + offset + valueByte)] = true;
            }
        }
        for (int index = 0; index < occupied.length; index++) {
            if (!occupied[index]) {
                assertEquals(CANARY, buffer.allocation.get(JAVA_BYTE, index),
                        "guard or non-dense gap byte " + index + " of " + buffer.tensor);
            }
        }
    }

    private record GuardedBuffer(CpuOneDnnKernelSpec.Tensor tensor,
            MemorySegment allocation, MemorySegment data) {
        void reset() { allocation.fill(CANARY); }
        byte[] snapshot() { return allocation.toArray(JAVA_BYTE); }
    }
}

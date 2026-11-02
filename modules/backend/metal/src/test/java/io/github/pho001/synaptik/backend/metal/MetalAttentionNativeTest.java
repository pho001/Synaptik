package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Actual native SDPA checks, including behavior that cannot be delegated to ordinary softmax. */
class MetalAttentionNativeTest {
    @Test
    void canonicalSiblingAndDeclaredScratchHaveTheSameDefaultAndExplicitScaleResults() {
        for (boolean explicit : List.of(false, true)) {
            float scale = explicit ? 0.5f : (float) (1.0 / Math.sqrt(2));
            float first = (float) Math.exp(scale);
            float second = (float) Math.exp(2 * scale);
            float sum = first + second;
            float[] weights = {first / sum, second / sum, 0.5f, 0.5f};
            float[] expected = {
                    weights[0] * 2 + weights[1] * 6,
                    weights[0] * 4 + weights[1] * 8,
                    4, 6};
            for (boolean sibling : List.of(false, true)) {
                List<byte[]> result = execute(
                        attention(false, sibling, explicit, false),
                        descriptors(sibling, false,
                                new long[] {2, 2}, new long[] {2, 2}, new long[] {2, 2},
                                new long[] {2, 2}, new long[] {2, 2}, null),
                        new int[] {0, 1, 2}, sibling ? new int[] {3, 4} : new int[] {3},
                        List.of(floats(1, 0, 0, 0), floats(1, 0, 2, 0), floats(2, 4, 6, 8)));
                assertArrayEquals(expected, decode(result.getFirst()), 3e-6f);
                if (sibling) assertArrayEquals(weights, decode(result.get(1)), 3e-6f);
            }
        }
    }

    @Test
    void maskAndTopLeftCausalEligibilitySkipExceptionalScoresAndValues() {
        List<byte[]> result = execute(attention(true, true, true, true),
                descriptors(true, true,
                        new long[] {2, 1}, new long[] {3, 1}, new long[] {3, 1},
                        new long[] {2, 1}, new long[] {2, 3}, new long[] {1, 3}),
                new int[] {0, 1, 2, 3}, new int[] {4, 5},
                List.of(floats(1, 1), floats(0, 0, Float.NaN), floats(2, 6, Float.NaN),
                        new byte[] {1, 1, 0}));
        assertArrayEquals(new float[] {2, 4}, decode(result.getFirst()), 0);
        assertArrayEquals(floats(1, 0, 0, 0.5f, 0.5f, 0), result.get(1));

        // Mask excludes all loads, including a NaN query shared by every key.
        result = execute(attention(true, true, true, false),
                descriptors(true, true,
                        new long[] {1, 1}, new long[] {2, 1}, new long[] {2, 1},
                        new long[] {1, 1}, new long[] {1, 2}, new long[] {2}),
                new int[] {0, 1, 2, 3}, new int[] {4, 5},
                List.of(floats(Float.NaN), floats(Float.NaN, Float.POSITIVE_INFINITY),
                        floats(Float.NaN, Float.NEGATIVE_INFINITY), new byte[] {0, 0}));
        assertArrayEquals(floats(0), result.getFirst());
        assertArrayEquals(floats(0, 0), result.get(1));
    }

    @Test
    void guardedRowsPreserveNoEligibleNegativeInfinityPositiveTiesAndNaNPriority() {
        assertRow(new float[] {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY},
                new float[] {Float.NaN, Float.POSITIVE_INFINITY},
                new float[] {0, 0}, 0);
        assertRow(new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, 0},
                new float[] {2, 6, 100}, new float[] {0.5f, 0.5f, 0}, 4);
        assertRow(new float[] {Float.POSITIVE_INFINITY, Float.NaN},
                new float[] {2, 6}, new float[] {Float.NaN, Float.NaN}, Float.NaN);
        assertRow(new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, 0},
                new float[] {2, 6, Float.NaN}, new float[] {0.5f, 0.5f, 0}, Float.NaN);
    }

    @Test
    void finiteAttentionRetainsLiteralExpSitesRatherThanOrdinarySoftmaxShift() {
        assertRow(new float[] {200, 202}, new float[] {2, 6},
                new float[] {Float.NaN, Float.NaN}, Float.NaN);
        assertRow(new float[] {-400, -402}, new float[] {2, 6},
                new float[] {Float.NaN, Float.NaN}, Float.NaN);
        // A finite overflowed EXP has zero-weight finite peers, unlike +infinity score ties.
        assertRow(new float[] {200, 0}, new float[] {2, 6},
                new float[] {Float.NaN, 0}, Float.NaN);
    }

    @Test
    void broadcastsIndependentBatchAndHeadRolesIntoCanonicalSiblingGeometry() {
        List<byte[]> result = execute(attention(true, true, true, false),
                descriptors(true, true,
                        new long[] {2, 1, 1, 1}, new long[] {1, 3, 2, 1},
                        new long[] {2, 1, 2, 2}, new long[] {2, 3, 1, 2},
                        new long[] {2, 3, 1, 2}, new long[] {3, 1, 2}),
                new int[] {0, 1, 2, 3}, new int[] {4, 5},
                List.of(floats(0, 0), floats(0, 0, 0, 0, 0, 0),
                        floats(2, 4, 6, 8, 10, 12, 14, 16), new byte[] {1, 1, 1, 0, 0, 1}));
        assertArrayEquals(floats(4, 6, 2, 4, 6, 8, 12, 14, 10, 12, 14, 16), result.getFirst());
        assertArrayEquals(floats(0.5f, 0.5f, 1, 0, 0, 1, 0.5f, 0.5f, 1, 0, 0, 1), result.get(1));
    }

    @Test
    void emptyKeyDomainPublishesPositiveZerosWithOrWithoutSiblingStorage() {
        for (boolean sibling : List.of(false, true)) {
            List<byte[]> result = execute(attention(false, sibling, false, false),
                    descriptors(sibling, false,
                            new long[] {2, 1}, new long[] {0, 1}, new long[] {0, 3},
                            new long[] {2, 3}, new long[] {2, 0}, null),
                    new int[] {0, 1, 2}, sibling ? new int[] {3, 4} : new int[] {3},
                    List.of(floats(Float.NaN, Float.POSITIVE_INFINITY), floats(), floats()));
            assertArrayEquals(new byte[6 * Float.BYTES], result.getFirst());
            if (sibling) assertEquals(0, result.get(1).length);
        }
    }

    @Test
    void affineRoleOffsetsStridesAndExactMaskBytesAreNotErased() {
        var values = new ArrayList<>(descriptors(true, true,
                new long[] {1, 2}, new long[] {2, 2}, new long[] {2, 2},
                new long[] {1, 2}, new long[] {1, 2}, new long[] {2}));
        values.set(0, affine(DataType.FLOAT32, new long[] {1, 2}, new long[] {4, 2}, 1));
        values.set(1, affine(DataType.FLOAT32, new long[] {2, 2}, new long[] {1, 2}, 1));
        values.set(2, affine(DataType.FLOAT32, new long[] {2, 2}, new long[] {3, 1}, 1));
        values.set(3, affine(DataType.BOOL, new long[] {2}, new long[] {2}, 1));
        List<byte[]> result = execute(attention(true, true, true, false), values,
                new int[] {0, 1, 2, 3}, new int[] {4, 5},
                List.of(floats(Float.NaN, 0, Float.NaN, 0), floats(Float.NaN, 1, 2, 3, 4),
                        floats(Float.NaN, 2, 4, Float.NaN, 6, 8), new byte[] {0, 1, 0, 1}));
        assertArrayEquals(floats(4, 6), result.getFirst());
        assertArrayEquals(floats(0.5f, 0.5f), result.get(1));
    }

    private static MetalMpsGraphProgram.ValueDescriptor affine(
            DataType type, long[] dimensions, long[] strides, long offset) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions,
                Optional.of(LayoutDescriptor.of(Shape.of(dimensions), strides, offset, true)), false, false);
    }

    private static void assertRow(float[] keys, float[] values, float[] expectedWeights, float expected) {
        List<byte[]> result = execute(attention(false, true, true, false),
                descriptors(true, false,
                        new long[] {1, 1}, new long[] {keys.length, 1}, new long[] {keys.length, 1},
                        new long[] {1, 1}, new long[] {1, keys.length}, null),
                new int[] {0, 1, 2}, new int[] {3, 4}, List.of(floats(1), floats(keys), floats(values)));
        assertClassesAndValues(new float[] {expected}, decode(result.getFirst()));
        assertClassesAndValues(expectedWeights, decode(result.get(1)));
    }

    private static void assertClassesAndValues(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (Float.isNaN(expected[i])) assertTrue(Float.isNaN(actual[i]), "NaN at " + i);
            else assertEquals(Float.floatToRawIntBits(expected[i]), Float.floatToRawIntBits(actual[i]),
                    "exact value/zero sign at " + i);
        }
    }

    private static MetalMpsGraphProgram attention(boolean mask, boolean sibling, boolean explicit, boolean causal) {
        int output = mask ? 4 : 3;
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION,
                mask ? new int[] {0, 1, 2, 3} : new int[] {0, 1, 2},
                sibling ? new int[] {output, output + 1} : new int[] {output},
                MetalMpsGraphProgram.AttributeKind.ATTENTION,
                explicit ? new long[] {1, 1, Float.floatToRawIntBits(0.5f), causal ? 1 : 0}
                        : new long[] {0, causal ? 1 : 0})));
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> descriptors(
            boolean sibling, boolean mask, long[] query, long[] key, long[] value,
            long[] output, long[] weights, long[] maskShape) {
        var result = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (long[] shape : List.of(query, key, value))
            result.add(new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, shape, false));
        if (mask) result.add(new MetalMpsGraphProgram.ValueDescriptor(DataType.BOOL, maskShape, false));
        result.add(new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, output, false));
        if (sibling) result.add(new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, weights, false));
        return List.copyOf(result);
    }

    private static List<byte[]> execute(MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values, int[] feeds, int[] targets,
            List<byte[]> inputs) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createProgramExecutable(context, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            for (long bytes : MetalProgramWorkspace.scratchBytes(program, values)) {
                buffers.add(api.createBuffer(context, bytes));
            }
            try (Arena arena = Arena.ofConfined()) {
                for (int i = 0; i < feeds.length; i++) {
                    byte[] bytes = inputs.get(i);
                    if (bytes.length == 0) continue;
                    var source = arena.allocateFrom(java.lang.foreign.ValueLayout.JAVA_BYTE, bytes);
                    api.upload(buffers.get(feeds[i]), 0, source, bytes.length);
                }
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int i = 0; i < buffers.size(); i++)
                    addresses.setAtIndex(ADDRESS, i, buffers.get(i).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int i = 0; i < targets.length; i++)
                    outputs.setAtIndex(ADDRESS, i, buffers.get(targets[i]).carrier());
                // Reuse catches scratch/sibling lifetime bugs without another prepared artifact.
                api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                var result = new ArrayList<byte[]>();
                for (int target : targets) {
                    long bytes = values.get(target).byteCount();
                    if (bytes == 0) { result.add(new byte[0]); continue; }
                    var destination = arena.allocate(bytes, 1);
                    api.download(buffers.get(target), 0, destination, bytes);
                    result.add(destination.toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                }
                return List.copyOf(result);
            }
        } finally {
            for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static byte[] floats(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putFloat(value);
        return bytes.array();
    }

    private static float[] decode(byte[] bytes) {
        var buffer = ByteBuffer.wrap(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        float[] result = new float[buffer.remaining()];
        buffer.get(result);
        return result;
    }
}

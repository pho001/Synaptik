package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Direct native regressions for rank bounds, exact low words, and creating-context identity. */
class MetalNativeBoundaryRepairTest {
    @Test
    void rankSixteenFalseWhereBranchRetainsLastAxisAndBroadcastsExactly() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
                for (boolean broadcast : List.of(false, true)) {
                    long[] output = new long[16];
                    Arrays.fill(output, 1);
                    output[14] = 2;
                    output[15] = 3;
                    long[] falseShape = output.clone();
                    if (broadcast) falseShape[15] = 1;
                    var values = List.of(value(DataType.BOOL, 3), value(type, 2, broadcast ? 3 : 1),
                            value(type, falseShape), value(type, output));
                    var node = MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.WHERE, new int[] {0, 1, 2},
                            new int[] {3}, MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
                    int nan = type == DataType.FLOAT32 ? 0xff812345
                            : type == DataType.BFLOAT16 ? 0xff81 : 0xfc01;
                    int negativeZero = type == DataType.FLOAT32 ? 0x80000000 : 0x8000;
                    byte[] whenFalse = words(type, broadcast
                            ? new int[] {nan, 1} : new int[] {nan, 1, negativeZero, 2, 3, 4});
                    byte[] expected = words(type, broadcast
                            ? new int[] {nan, negativeZero, nan, 1, 0, 1}
                            : new int[] {nan, negativeZero, negativeZero, 2, 0, 4});
                    assertArrayEquals(expected, scope.execute(values, node,
                            List.of(new byte[] {0, 1, 0}, words(type, broadcast
                                    ? new int[] {negativeZero, negativeZero, negativeZero, 0, 0, 0}
                                    : new int[] {negativeZero, 0}), whenFalse)),
                            type + " broadcast=" + broadcast);
                }
            }
        }
    }

    @Test
    void lowScatterCopiesEveryUntouchedEncodingForBothIndexWidths() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                int one = type == DataType.BFLOAT16 ? 0x3f80 : 0x3c00;
                int three = type == DataType.BFLOAT16 ? 0x4040 : 0x4200;
                int[] base = type == DataType.BFLOAT16
                        ? new int[] {one, 0x7f81, 0xff81, 0x7fc5, 0xffc7, 0, 0x8000, 1, 0x8001}
                        : new int[] {one, 0x7c01, 0xfc01, 0x7e05, 0xfe07, 0, 0x8000, 1, 0x8001};
                int[] expected = base.clone();
                expected[0] = three;
                for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                    var values = List.of(value(type, base.length), value(indexType, 2),
                            value(type, 2), value(type, base.length));
                    byte[] indices = new byte[Math.toIntExact(2 * indexType.byteWidth())];
                    assertArrayEquals(words(type, expected), scope.execute(values,
                            MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 0),
                            List.of(words(type, base), indices, words(type, one, one))),
                            type + "/" + indexType);
                }
            }
        }
    }

    @Test
    void lowScansSeedFirstContributorAndKeepExclusiveIdentitiesInBothDirections() {
        try (NativeScope scope = new NativeScope()) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                int one = type == DataType.BFLOAT16 ? 0x3f80 : 0x3c00;
                int two = 0x4000;
                int three = type == DataType.BFLOAT16 ? 0x4040 : 0x4200;
                for (boolean reverse : List.of(false, true)) {
                    for (boolean exclusive : List.of(false, true)) {
                        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.CUM_SUM,
                                MetalMpsGraphProgram.NodeKind.CUM_PROD)) {
                            boolean product = kind == MetalMpsGraphProgram.NodeKind.CUM_PROD;
                            // Two independent lines: all-negative zero and ordinary finite terms.
                            int[] input = {0x8000, 0x8000, 0x8000, 0x8000, one, two};
                            int[] expected = product
                                    ? new int[] {0x8000, 0, 0x8000, 0x8000, 0x8000, 0x8000}
                                    : new int[] {0x8000, 0x8000, 0x8000, 0x8000, one, three};
                            if (exclusive) {
                                expected = product
                                        ? new int[] {one, 0x8000, 0, one, 0x8000, 0x8000}
                                        : new int[] {0, 0x8000, 0x8000, 0, 0x8000, one};
                            }
                            if (reverse) {
                                reverseLines(input);
                                reverseLines(expected);
                            }
                            assertArrayEquals(words(type, expected), scope.execute(
                                    List.of(value(type, 2, 3), value(type, 2, 3)),
                                    MetalMpsGraphProgram.Node.scan(kind, 0, 1, 1, exclusive, reverse),
                                    List.of(words(type, input))),
                                    type + "/" + kind + "/exclusive=" + exclusive + "/reverse=" + reverse);
                        }
                    }
                }
            }
        }
    }

    @Test
    void allThreeNativeRoutesRejectForeignBuffersBeforeMutationAndSurviveOwnerRelease() {
        try (NativeScope scope = new NativeScope()) {
            // Both contexts use MTLCreateSystemDefaultDevice; device equality is not ownership.
            var foreignContext = scope.context();
            var localInput = scope.buffer(scope.owner, words(DataType.FLOAT32, 0x3f800000));
            var localOutput = scope.buffer(scope.owner, words(DataType.FLOAT32, 0x5a5a5a5a));
            var foreignInput = scope.buffer(foreignContext, words(DataType.FLOAT32, 0x40000000));
            var foreignOutput = scope.buffer(foreignContext, words(DataType.FLOAT32, 0x6a6a6a6a));
            for (MetalPreparedRoute route : MetalPreparedRoute.values()) {
                var executable = scope.executable(route, List.of(value(DataType.FLOAT32, 1),
                        value(DataType.FLOAT32, 1)), route == MetalPreparedRoute.CUSTOM_PROGRAM
                                ? MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.FLOOR,
                                        new int[] {0}, new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0])
                                : MetalMpsGraphProgram.Node.neg(0, 1));
                for (int foreignRole = 1; foreignRole <= 3; foreignRole++) {
                    var input = (foreignRole & 1) != 0 ? foreignInput : localInput;
                    var output = (foreignRole & 2) != 0 ? foreignOutput : localOutput;
                    var failure = assertThrows(MetalNativeApi.NativeFailure.class,
                            () -> scope.run(route, executable, List.of(input, output)));
                    assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
                    assertArrayEquals(words(DataType.FLOAT32, 0x5a5a5a5a), scope.download(localOutput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x6a6a6a6a), scope.download(foreignOutput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x3f800000), scope.download(localInput, 4));
                    assertArrayEquals(words(DataType.FLOAT32, 0x40000000), scope.download(foreignInput, 4));
                }
                scope.run(route, executable, List.of(localInput, localOutput));
                assertArrayEquals(words(DataType.FLOAT32,
                        route == MetalPreparedRoute.CUSTOM_PROGRAM ? 0x3f800000 : 0xbf800000),
                        scope.download(localOutput, 4));
                scope.upload(localOutput, words(DataType.FLOAT32, 0x5a5a5a5a));
            }
            // Buffers keep their identity alive after their context handle is consumed, without
            // owning its queue. New contexts must never accept those surviving foreign buffers.
            scope.releaseContext(foreignContext);
            for (int iteration = 0; iteration < 16; iteration++) {
                var replacement = scope.context();
                var output = scope.buffer(replacement, words(DataType.FLOAT32, 0x5a5a5a5a));
                var pipeline = scope.api.createNegKernelPipeline(replacement, 1);
                try {
                    var failure = assertThrows(MetalNativeApi.NativeFailure.class,
                            () -> scope.api.runNegKernelPipeline(pipeline, foreignInput, output));
                    assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
                    assertArrayEquals(words(DataType.FLOAT32, 0x5a5a5a5a), scope.download(output, 4));
                } finally {
                    scope.api.releaseNegKernelPipeline(pipeline);
                    scope.releaseContext(replacement);
                }
            }
            scope.releaseContext(scope.owner);
            // Existing executables already retain their context. Consuming its external handle
            // must not invalidate those resources, and release order remains caller-owned.
            for (int index = 0; index < scope.executables.size(); index++) {
                scope.run(scope.routes.get(index), scope.executables.get(index), List.of(localInput, localOutput));
                assertArrayEquals(words(DataType.FLOAT32,
                        scope.routes.get(index) == MetalPreparedRoute.CUSTOM_PROGRAM ? 0x3f800000 : 0xbf800000),
                        scope.download(localOutput, 4));
            }
        }
    }

    private static void reverseLines(int[] words) {
        for (int start = 0; start < words.length; start += 3) {
            int first = words[start];
            words[start] = words[start + 2];
            words[start + 2] = first;
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, false);
    }

    private static byte[] words(DataType type, int... words) {
        ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(words.length * type.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (int word : words) {
            if (type == DataType.FLOAT32) bytes.putInt(word);
            else bytes.putShort((short) word);
        }
        return bytes.array();
    }

    /** Owns raw handles only; intentionally bypasses Java context-bound representation checks. */
    private static final class NativeScope implements AutoCloseable {
        private final MetalNativeApi api;
        private final List<MetalNativeApi.Handle> contexts = new ArrayList<>();
        private final List<MetalNativeApi.Handle> buffers = new ArrayList<>();
        private final List<MetalNativeApi.Handle> executables = new ArrayList<>();
        private final List<MetalPreparedRoute> routes = new ArrayList<>();
        private final MetalNativeApi.Handle owner;

        private NativeScope() {
            String library = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(library != null && !library.isBlank(), "native device library not configured");
            api = MetalNativeApi.open(Path.of(library));
            try {
                owner = context();
            } catch (RuntimeException | Error failure) {
                api.close();
                throw failure;
            }
        }

        private MetalNativeApi.Handle context() {
            var context = api.createContext();
            contexts.add(context);
            return context;
        }

        private void releaseContext(MetalNativeApi.Handle context) {
            api.releaseContext(context);
            contexts.remove(context);
        }

        private MetalNativeApi.Handle buffer(MetalNativeApi.Handle context, byte[] bytes) {
            var buffer = api.createBuffer(context, bytes.length);
            buffers.add(buffer);
            upload(buffer, bytes);
            return buffer;
        }

        private void upload(MetalNativeApi.Handle buffer, byte[] bytes) {
            try (Arena arena = Arena.ofConfined()) {
                var source = arena.allocate(bytes.length, 1);
                source.copyFrom(MemorySegment.ofArray(bytes));
                api.upload(buffer, 0, source, bytes.length);
            }
        }

        private byte[] download(MetalNativeApi.Handle buffer, int count) {
            try (Arena arena = Arena.ofConfined()) {
                var target = arena.allocate(count, 1);
                api.download(buffer, 0, target, count);
                return target.toArray(JAVA_BYTE);
            }
        }

        private MetalNativeApi.Handle executable(MetalPreparedRoute route,
                List<MetalMpsGraphProgram.ValueDescriptor> values, MetalMpsGraphProgram.Node node) {
            int[] feeds = new int[values.size() - 1];
            for (int index = 0; index < feeds.length; index++) feeds[index] = index;
            var executable = route == MetalPreparedRoute.CUSTOM_SINGLE_NEG
                    ? api.createNegKernelPipeline(owner, 1)
                    : api.createProgramExecutable(owner, values, new MetalMpsGraphProgram(List.of(node)),
                            feeds, new int[] {values.size() - 1}, route);
            executables.add(executable);
            routes.add(route);
            return executable;
        }

        private byte[] execute(List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalMpsGraphProgram.Node node, List<byte[]> inputs) {
            var route = MetalPreparedRoute.CUSTOM_PROGRAM;
            var executable = executable(route, values, node);
            var table = new ArrayList<MetalNativeApi.Handle>();
            for (int index = 0; index < values.size(); index++) {
                table.add(buffer(owner, index < inputs.size() ? inputs.get(index)
                        : new byte[Math.toIntExact(values.get(index).byteCount())]));
            }
            run(route, executable, table);
            for (int index = 0; index < inputs.size(); index++) {
                assertArrayEquals(inputs.get(index), download(table.get(index), inputs.get(index).length));
            }
            return download(table.getLast(), Math.toIntExact(values.getLast().byteCount()));
        }

        private void run(MetalPreparedRoute route, MetalNativeApi.Handle executable,
                List<MetalNativeApi.Handle> table) {
            if (route == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
                api.runNegKernelPipeline(executable, table.getFirst(), table.getLast());
                return;
            }
            try (Arena arena = Arena.ofConfined()) {
                int count = route == MetalPreparedRoute.CUSTOM_PROGRAM ? table.size() : table.size() - 1;
                var inputs = arena.allocate(ADDRESS, count);
                for (int index = 0; index < count; index++) {
                    inputs.setAtIndex(ADDRESS, index, table.get(index).carrier());
                }
                var outputs = arena.allocate(ADDRESS);
                outputs.set(ADDRESS, 0, table.getLast().carrier());
                api.runExecutable(executable, count, inputs, 1, outputs);
            }
        }

        @Override
        public void close() {
            for (int index = executables.size(); index-- > 0;) {
                if (routes.get(index) == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
                    api.releaseNegKernelPipeline(executables.get(index));
                } else api.releaseExecutable(executables.get(index));
            }
            for (var buffer : buffers.reversed()) api.releaseBuffer(buffer);
            for (var context : contexts.reversed()) api.releaseContext(context);
            api.close();
        }
    }
}

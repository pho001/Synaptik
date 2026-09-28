package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalPointwiseDispatchObserverNativeTest {
    @Test
    void generatedUnitsDispatchExactlyOnceWithAuthenticatedPointMeta() throws Throwable {
        String libraryValue = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        String observerValue = System.getenv("SYNAPTIK_METAL_TEST_OBSERVER");
        assumeTrue(libraryValue != null && !libraryValue.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assumeTrue(observerValue != null && !observerValue.isBlank(),
                "SYNAPTIK_METAL_TEST_OBSERVER is not set");
        Path library = Path.of(libraryValue).toAbsolutePath().normalize();
        Path observerLibrary = Path.of(observerValue).toAbsolutePath().normalize();

        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        MetalMpsGraphProgram.NodeKind[] kinds = {
            MetalMpsGraphProgram.NodeKind.FLOOR,
            MetalMpsGraphProgram.NodeKind.CEIL,
            MetalMpsGraphProgram.NodeKind.SIGN,
            MetalMpsGraphProgram.NodeKind.RELU
        };
        for (int index = 0; index < 9; index++) {
            nodes.add(MetalMpsGraphProgram.Node.generic(
                    kinds[index % kinds.length],
                    new int[] {index},
                    new int[] {index + 1},
                    MetalMpsGraphProgram.AttributeKind.NONE,
                    new long[0]));
        }
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (int index = 0; index < 10; index++)
            values.add(new MetalMpsGraphProgram.ValueDescriptor(
                    DataType.FLOAT32, new long[] {4}, false));
        var program = new MetalMpsGraphProgram(nodes);
        MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0},
                new int[] {9},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(2, fusion.generatedUnitCount());
        assertArrayEquals(new int[] {0, 7, 9}, fusion.materializedProgramValueIndices());

        try (Observer observer = new Observer(observerLibrary)) {
            observer.reset();
            execute(library, program, values, fusion);
            assertEquals(2, observer.count());
            for (int record = 0; record < 2; record++) {
                assertEquals(record, observer.field(record, 0));
                assertEquals(MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE.wire(),
                        observer.field(record, 1));
                assertEquals(4L, observer.field(record, 2));
                assertEquals(1L, observer.field(record, 3));
                assertEquals(1L, observer.field(record, 4));
                assertTrue(observer.field(record, 5) > 0L);
                assertEquals(32L, observer.field(record, 6));
                assertEquals(4L, observer.field(record, 7));
                assertEquals(4L, observer.field(record, 8));
                assertEquals(1L, observer.field(record, 9));
                assertEquals(0L, observer.field(record, 10));
                assertEquals(0L, observer.field(record, 11));
                assertArrayEquals(fusion.canonicalManifestDigest(), observer.digest(record));
            }
        }
    }

    @Test
    void oneGeneratedUnitProducesOneObservedNativeDispatch() throws Throwable {
        String libraryValue = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        String observerValue = System.getenv("SYNAPTIK_METAL_TEST_OBSERVER");
        assumeTrue(libraryValue != null && !libraryValue.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assumeTrue(observerValue != null && !observerValue.isBlank(),
                "SYNAPTIK_METAL_TEST_OBSERVER is not set");
        Path library = Path.of(libraryValue).toAbsolutePath().normalize();
        Path observerLibrary = Path.of(observerValue).toAbsolutePath().normalize();

        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FLOOR,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0]),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.CEIL,
                        new int[] {1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0]),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SIGN,
                        new int[] {2}, new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0])));
        var values = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, new long[] {4}, false));
        MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0},
                new int[] {3},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(1, fusion.generatedUnitCount());
        assertArrayEquals(new int[] {0, 3}, fusion.materializedProgramValueIndices());

        try (Observer observer = new Observer(observerLibrary)) {
            observer.reset();
            execute(library, program, values, fusion);
            assertEquals(1, observer.count());
            assertEquals(0L, observer.field(0, 0));
            assertEquals(MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE.wire(),
                    observer.field(0, 1));
            assertEquals(32L, observer.field(0, 6));
            assertEquals(4L, observer.field(0, 7));
            assertEquals(4L, observer.field(0, 8));
            assertEquals(1L, observer.field(0, 9));
            assertEquals(0L, observer.field(0, 10));
            assertEquals(0L, observer.field(0, 11));
            assertArrayEquals(fusion.canonicalManifestDigest(), observer.digest(0));
        }
    }

    private static void execute(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalPointwiseFusionPlan fusion) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    values,
                    program,
                    new int[] {0},
                    new int[] {values.size() - 1},
                    MetalPreparedRoute.CUSTOM_PROGRAM,
                    fusion);
            for (int value : fusion.materializedProgramValueIndices())
                buffers.add(api.createBuffer(context, values.get(value).byteCount()));
            try (Arena arena = Arena.ofConfined()) {
                int[] words = {0xc020_0000, 0x8000_0000, 0x3fa0_0000, 0x7fc1_2345};
                MemorySegment input = arena.allocate(4L * Integer.BYTES, Integer.BYTES);
                for (int index = 0; index < words.length; index++)
                    input.setAtIndex(JAVA_INT, index, words[index]);
                api.upload(buffers.getFirst(), 0L, input, input.byteSize());
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++)
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS);
                outputs.set(ADDRESS, 0L, buffers.getLast().carrier());
                api.runExecutable(executable, buffers.size(), addresses, 1, outputs);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static final class Observer implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MethodHandle reset;
        private final MethodHandle count;
        private final MethodHandle field;
        private final MethodHandle digestByte;

        private Observer(Path library) {
            Linker linker = Linker.nativeLinker();
            SymbolLookup lookup = SymbolLookup.libraryLookup(library, arena);
            reset = linker.downcallHandle(
                    lookup.findOrThrow("synaptik_metal_test_observer_reset"),
                    FunctionDescriptor.ofVoid());
            count = linker.downcallHandle(
                    lookup.findOrThrow("synaptik_metal_test_observer_count"),
                    FunctionDescriptor.of(JAVA_INT));
            field = linker.downcallHandle(
                    lookup.findOrThrow("synaptik_metal_test_observer_field"),
                    FunctionDescriptor.of(JAVA_LONG, JAVA_INT, JAVA_INT));
            digestByte = linker.downcallHandle(
                    lookup.findOrThrow("synaptik_metal_test_observer_digest_byte"),
                    FunctionDescriptor.of(JAVA_INT, JAVA_INT, JAVA_INT));
        }

        private void reset() throws Throwable {
            reset.invokeExact();
        }

        private int count() throws Throwable {
            return (int) count.invokeExact();
        }

        private long field(int record, int index) throws Throwable {
            return (long) field.invokeExact(record, index);
        }

        private byte[] digest(int record) throws Throwable {
            byte[] result = new byte[32];
            for (int index = 0; index < result.length; index++)
                result[index] = (byte) (int) digestByte.invokeExact(record, index);
            return result;
        }

        @Override
        public void close() {
            arena.close();
        }
    }
}

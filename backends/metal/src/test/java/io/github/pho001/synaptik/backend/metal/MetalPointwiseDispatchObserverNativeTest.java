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

    @Test
    void mixedPlanExecutesGeneratedMpsGraphAndEveryRequiredFixedKind() throws Throwable {
        String libraryValue = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        String observerValue = System.getenv("SYNAPTIK_METAL_TEST_OBSERVER");
        assumeTrue(libraryValue != null && !libraryValue.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        assumeTrue(observerValue != null && !observerValue.isBlank(),
                "SYNAPTIK_METAL_TEST_OBSERVER is not set");
        Path library = Path.of(libraryValue).toAbsolutePath().normalize();
        Path observerLibrary = Path.of(observerValue).toAbsolutePath().normalize();

        var program = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.ABS, 2, 3),
                unary(MetalMpsGraphProgram.NodeKind.NEG, 3, 4),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.L1_NORM,
                        5,
                        6,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        List.of(0),
                        false),
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.VARIANCE,
                        7,
                        8,
                        List.of(0),
                        false,
                        0L),
                MetalMpsGraphProgram.Node.scatterAdd(9, 10, 11, 12, 0)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                value(DataType.FLOAT32, 4), value(DataType.FLOAT32, 4),
                value(DataType.FLOAT32, 4), value(DataType.FLOAT32, 4),
                value(DataType.FLOAT32, 4),
                value(DataType.FLOAT32, 1), value(DataType.FLOAT32),
                value(DataType.FLOAT32, 1), value(DataType.FLOAT32),
                value(DataType.FLOAT32, 3), value(DataType.INT32, 2),
                value(DataType.FLOAT32, 2), value(DataType.FLOAT32, 3));
        int[] feeds = {0, 5, 7, 9, 10, 11};
        int[] targets = {4, 6, 8, 12};
        MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                NumericalProfile.ACCELERATOR,
                program,
                values,
                feeds,
                targets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(
                List.of(
                        MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE,
                        MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY,
                        MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY,
                        MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                        MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                        MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM),
                fusion.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());
        assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5, 6}, fusion.memberNodePositions());
        assertEquals(1, fusion.generatedUnitCount());

        try (Observer observer = new Observer(observerLibrary)) {
            observer.reset();
            execute(
                    library,
                    NumericalProfile.ACCELERATOR,
                    program,
                    values,
                    feeds,
                    targets,
                    fusion,
                    List.of(
                            new int[] {0xc020_0000, 0x8000_0000, 0x3fa0_0000, 0x7fc1_2345},
                            new int[] {0xc040_0000},
                            new int[] {0x40a0_0000},
                            new int[] {0x3f80_0000, 0x4000_0000, 0x4040_0000},
                            new int[] {0, 2},
                            new int[] {0x4080_0000, 0x40a0_0000}));
            assertEquals(4, observer.count());
            long[] ordinals = {0L, 3L, 4L, 5L};
            for (int record = 0; record < ordinals.length; record++) {
                assertEquals(ordinals[record], observer.field(record, 0));
                assertEquals(
                        record == 0
                                ? MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE.wire()
                                : MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM.wire(),
                        observer.field(record, 1));
                assertTrue(observer.field(record, 5) > 0L);
                assertArrayEquals(fusion.canonicalManifestDigest(), observer.digest(record));
            }
            assertEquals(32L, observer.field(0, 6));
            assertEquals(4L, observer.field(0, 7));
            assertEquals(4L, observer.field(0, 8));
            assertEquals(1L, observer.field(0, 9));
            assertEquals(0L, observer.field(0, 10));
            assertEquals(0L, observer.field(0, 11));
            for (int record = 1; record < 4; record++) {
                assertEquals(0L, observer.field(record, 7));
                assertEquals(0L, observer.field(record, 8));
                assertEquals(0L, observer.field(record, 9));
                assertEquals(0L, observer.field(record, 10));
                assertEquals(0L, observer.field(record, 11));
            }
        }
    }

    private static MetalMpsGraphProgram.Node unary(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind,
                new int[] {input},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static void execute(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalPointwiseFusionPlan fusion) {
        execute(
                library,
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0},
                new int[] {values.size() - 1},
                fusion,
                List.of(new int[] {
                    0xc020_0000, 0x8000_0000, 0x3fa0_0000, 0x7fc1_2345
                }));
    }

    private static void execute(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            MetalPointwiseFusionPlan fusion,
            List<int[]> feedWords) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context,
                    profile,
                    values,
                    program,
                    feeds,
                    targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM,
                    fusion);
            for (int value : fusion.materializedProgramValueIndices())
                buffers.add(api.createBuffer(context, values.get(value).byteCount()));
            int[] programToSlot = fusion.programToMaterializedSlot();
            int[] targetSlots = fusion.targetMaterializedSlots();
            try (Arena arena = Arena.ofConfined()) {
                for (int feed = 0; feed < feeds.length; feed++) {
                    int[] words = feedWords.get(feed);
                    MemorySegment input =
                            arena.allocate((long) words.length * Integer.BYTES, Integer.BYTES);
                    for (int index = 0; index < words.length; index++)
                        input.setAtIndex(JAVA_INT, index, words[index]);
                    api.upload(
                            buffers.get(programToSlot[feeds[feed]]),
                            0L,
                            input,
                            input.byteSize());
                }
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++)
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, targetSlots.length);
                for (int target = 0; target < targetSlots.length; target++)
                    outputs.setAtIndex(
                            ADDRESS, target, buffers.get(targetSlots[target]).carrier());
                api.runExecutable(
                        executable, buffers.size(), addresses, targetSlots.length, outputs);
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

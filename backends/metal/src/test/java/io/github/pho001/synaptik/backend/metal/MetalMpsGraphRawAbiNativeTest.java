package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class MetalMpsGraphRawAbiNativeTest {
    private static final int INVALID_ARGUMENT = 1;
    private static final int UNSUPPORTED_SHAPE = 8;
    private static final Consumer<MemorySegment> UNCHANGED = ignored -> { };

    @Test
    void rawVersionFiveRecordRejectsEveryMalformedHeaderAndUnusedField() {
        try (RawAbi abi = RawAbi.open()) {
            MetalMpsGraphProgram reshape = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {3, 2})));
            int[] ranks = {2, 2};
            long[] dimensions = dimensions(new long[][] {{2, 3}, {3, 2}});

            abi.assertRejected("wrong attribute discriminator", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 4L, 0));
            abi.assertRejected("unknown attribute discriminator", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 4L, 99));
            abi.assertRejected("unary second input sentinel", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 12L, 0));
            abi.assertRejected("target-shape axis sentinel", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("reserved scalar", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("unused target-shape payload", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_LONG, 48L, 1L));
            abi.assertRejected("unknown operation", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 0L, 99));
            abi.assertRejected("zero target-shape count", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 20L, 0));
            abi.assertRejected("oversized target-shape count", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_INT, 20L, 17));
            abi.assertRejected("zero target dimension", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_LONG, 32L, 0L));
            abi.assertRejected("mismatched target dimension", ranks, dimensions, reshape,
                    new int[] {0}, new int[] {1}, record -> record.set(JAVA_LONG, 32L, 2L));

            abi.assertRejected("schema version one", INVALID_ARGUMENT, 1,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("withdrawn schema version two", INVALID_ARGUMENT, 2,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("withdrawn schema version three", INVALID_ARGUMENT, 3,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version four", INVALID_ARGUMENT, 4,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("unknown schema version six", INVALID_ARGUMENT, 6,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    @Test
    void rawBinaryRecordsRejectMalformedOrderedTopologyAndState() {
        try (RawAbi abi = RawAbi.open()) {
            int[] ranks = {2, 1, 2};
            long[] dimensions = dimensions(new long[][] {{2, 3}, {3}, {2, 3}});
            MetalMpsGraphProgram binary = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.SUB, 0, 1, 2)));
            abi.assertRejected("binary second input sentinel", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 12L, -1));
            abi.assertRejected("binary second input out of range", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 12L, 3));
            abi.assertRejected("binary attribute discriminator", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 4L, 1));
            abi.assertRejected("binary attribute count", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("binary axis sentinel", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("binary reserved cell", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("binary payload", ranks, dimensions, binary,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_LONG, 32L, 1L));
            abi.assertRejected("binary output is not exact broadcast",
                    new int[] {2, 1, 2},
                    dimensions(new long[][] {{2, 3}, {3}, {3, 2}}),
                    binary,
                    new int[] {0, 1},
                    new int[] {2},
                    UNCHANGED);

            int[] viewRanks = {1, 2, 2, 2};
            long[] viewDimensions = dimensions(
                    new long[][] {{6}, {2, 3}, {2, 3}, {2, 3}});
            MetalMpsGraphProgram affineToBinary = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {2, 3}),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.ADD, 1, 2, 3)));
            abi.assertRejected("binary first input must be canonical",
                    viewRanks,
                    viewDimensions,
                    affineToBinary,
                    new int[] {0, 2},
                    new int[] {3},
                    UNCHANGED);
        }
    }

    @Test
    void rawTypedAttributeBoundsRejectBeforeGraphCompilation() {
        try (RawAbi abi = RawAbi.open()) {
            int[] rankTwo = {2, 2};
            long[] transposeDimensions = dimensions(new long[][] {{2, 3}, {3, 2}});
            MetalMpsGraphProgram permutation = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0))));
            abi.assertRejected("permutation count does not equal rank", rankTwo,
                    transposeDimensions, permutation, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("duplicate permutation axis", rankTwo,
                    transposeDimensions, permutation, new int[] {0}, new int[] {1}, record -> {
                        record.set(JAVA_LONG, 32L, 1L);
                        record.set(JAVA_LONG, 40L, 1L);
                    });
            abi.assertRejected("out-of-range permutation axis", rankTwo,
                    transposeDimensions, permutation, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 2L));
            abi.assertRejected("permutation axis sentinel", rankTwo,
                    transposeDimensions, permutation, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("unused permutation payload", rankTwo,
                    transposeDimensions, permutation, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 48L, 1L));

            int[] expandRanks = {2, 3};
            long[] expandDimensions = dimensions(new long[][] {{2, 3}, {2, 1, 3}});
            MetalMpsGraphProgram expandDims = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.axis(
                            MetalMpsGraphProgram.NodeKind.EXPAND_DIMS, 0, 1, 1)));
            abi.assertRejected("axis attribute count zero", expandRanks, expandDimensions,
                    expandDims, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 0));
            abi.assertRejected("axis attribute count two", expandRanks, expandDimensions,
                    expandDims, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 2));
            abi.assertRejected("axis schema bound", expandRanks, expandDimensions,
                    expandDims, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 16));
            abi.assertRejected("axis payload must be unused", expandRanks, expandDimensions,
                    expandDims, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 1L));

            int[] squeezeRanks = {3, 2};
            long[] squeezeDimensions = dimensions(new long[][] {{2, 1, 3}, {2, 3}});
            MetalMpsGraphProgram squeeze = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.axis(
                            MetalMpsGraphProgram.NodeKind.SQUEEZE, 0, 1, 1)));
            abi.assertRejected("squeeze axis must select a singleton", squeezeRanks,
                    squeezeDimensions, squeeze, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("axis discriminator pairing", squeezeRanks,
                    squeezeDimensions, squeeze, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 1));

            MetalMpsGraphProgram contiguous = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.contiguous(0, 1)));
            long[] contiguousDimensions = dimensions(new long[][] {{2, 3}, {2, 3}});
            abi.assertRejected("contiguous attribute discriminator", rankTwo,
                    contiguousDimensions, contiguous, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 1));
            abi.assertRejected("contiguous attribute count", rankTwo,
                    contiguousDimensions, contiguous, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("contiguous axis sentinel", rankTwo,
                    contiguousDimensions, contiguous, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("contiguous payload", rankTwo,
                    contiguousDimensions, contiguous, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 1L));
        }
    }

    @Test
    void rawTopologyClassesRejectWithNullOutputHandles() {
        try (RawAbi abi = RawAbi.open()) {
            int[] ranks = {2, 2};
            long[] dimensions = dimensions(new long[][] {{2, 3}, {3, 2}});
            MetalMpsGraphProgram reshape = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {3, 2})));

            abi.assertRejected("input is not topologically available", ranks, dimensions,
                    reshape, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 8L, 1));
            abi.assertRejected("output collides with a feed", ranks, dimensions,
                    reshape, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 16L, 0));
            abi.assertRejected("output index is out of range", ranks, dimensions,
                    reshape, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 16L, 2));
            abi.assertRejected("target is an unproduced feed", ranks, dimensions,
                    reshape, new int[] {0}, new int[] {0}, UNCHANGED);
            abi.assertRejected("duplicate feed", ranks, dimensions,
                    reshape, new int[] {0, 0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("duplicate target", ranks, dimensions,
                    reshape, new int[] {0}, new int[] {1, 1}, UNCHANGED);

            int[] extraRanks = {2, 2, 1};
            long[] extraDimensions = dimensions(new long[][] {{2, 3}, {3, 2}, {1}});
            abi.assertRejected("declared value is unused", extraRanks, extraDimensions,
                    reshape, new int[] {0}, new int[] {1}, UNCHANGED);

            MetalMpsGraphProgram affineChain = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {3, 2}),
                    MetalMpsGraphProgram.Node.neg(1, 2)));
            int[] chainRanks = {2, 2, 2};
            long[] chainDimensions = dimensions(
                    new long[][] {{2, 3}, {3, 2}, {3, 2}});
            abi.assertRejected("affine result cannot feed NEG without CONTIGUOUS", chainRanks,
                    chainDimensions, affineChain, new int[] {0}, new int[] {2}, UNCHANGED);

            MetalMpsGraphProgram duplicateOutput = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {3, 2}),
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {3, 2})));
            abi.assertRejected("output is produced twice", ranks, dimensions,
                    duplicateOutput, new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    @Test
    void rawNegNoneAttributesAndWithdrawnBinaryOperationsRejectEveryMalformedField() {
        try (RawAbi abi = RawAbi.open()) {
            int[] unaryRanks = {1, 1};
            long[] unaryDimensions = dimensions(new long[][] {{2}, {2}});
            MetalMpsGraphProgram neg = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.neg(0, 1)));
            abi.assertRejected("NEG attribute discriminator", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 1));
            abi.assertRejected("NEG attribute count", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("NEG axis sentinel", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("NEG reserved scalar", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("NEG unused payload", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 1L));
            abi.assertRejected("NEG second-input sentinel", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 12L, 0));

            for (int operation = 2; operation <= 5; operation++) {
                int withdrawn = operation;
                abi.assertRejected(
                        "withdrawn binary operation wire " + withdrawn,
                        unaryRanks,
                        unaryDimensions,
                        neg,
                        new int[] {0},
                        new int[] {1},
                        record -> record.set(JAVA_INT, 0L, withdrawn));
            }
        }
    }

    @Test
    void rawExpandAndBoundaryIndicesRejectTheirExactNativeBranches() {
        try (RawAbi abi = RawAbi.open()) {

            int[] expandRanks = {2, 2};
            long[] expandDimensions = dimensions(new long[][] {{2, 3}, {2, 4}});
            MetalMpsGraphProgram expand = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.EXPAND,
                            0,
                            1,
                            new long[] {2, 4})));
            abi.assertRejected("EXPAND source cannot broadcast to target",
                    expandRanks, expandDimensions, expand,
                    new int[] {0}, new int[] {1}, UNCHANGED);

            int[] unaryRanks = {1, 1};
            long[] unaryDimensions = dimensions(new long[][] {{2}, {2}});
            MetalMpsGraphProgram neg = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.neg(0, 1)));
            abi.assertRejected("feed index is out of range",
                    unaryRanks, unaryDimensions, neg,
                    new int[] {2}, new int[] {1}, UNCHANGED);
            abi.assertRejected("target index is out of range",
                    unaryRanks, unaryDimensions, neg,
                    new int[] {0}, new int[] {2}, UNCHANGED);
        }
    }

    @Test
    void rawValueGeometryTablesReturnExactUnsupportedOrInvalidStatuses() {
        try (RawAbi abi = RawAbi.open()) {
            MetalMpsGraphProgram neg = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.neg(0, 1)));

            abi.assertRejected("rank zero", UNSUPPORTED_SHAPE,
                    new int[] {0, 1},
                    dimensions(new long[][] {{}, {1}}),
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("rank exceeds schema bound", UNSUPPORTED_SHAPE,
                    new int[] {17, 1},
                    dimensions(new long[][] {{1}, {1}}),
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("zero dimension", UNSUPPORTED_SHAPE,
                    new int[] {1, 1},
                    dimensions(new long[][] {{0}, {1}}),
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);

            long[] nonzeroPadding = dimensions(new long[][] {{2}, {2}});
            nonzeroPadding[1] = 1L;
            abi.assertRejected("unused dimension padding", INVALID_ARGUMENT,
                    new int[] {1, 1},
                    nonzeroPadding,
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);

            abi.assertRejected("element-count multiplication overflow", UNSUPPORTED_SHAPE,
                    new int[] {2, 2},
                    dimensions(new long[][] {
                            {Long.MAX_VALUE, 3},
                            {Long.MAX_VALUE, 3}
                    }),
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("FLOAT32 byte geometry overflow", UNSUPPORTED_SHAPE,
                    new int[] {1, 1},
                    dimensions(new long[][] {
                            {Long.MAX_VALUE},
                            {Long.MAX_VALUE}
                    }),
                    neg, new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] result = new long[Math.multiplyExact(shapes.length, MetalMpsGraphProgram.MAX_RANK)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, result,
                    value * MetalMpsGraphProgram.MAX_RANK, shapes[value].length);
        }
        return result;
    }

    private record CreateOutcome(int status, long outputAddress) { }

    private static final class RawAbi implements AutoCloseable {
        private final MetalNativeApi api;
        private final MetalNativeApi.Handle context;
        private final Arena lookupArena;
        private final MethodHandle create;

        private RawAbi(
                MetalNativeApi api,
                MetalNativeApi.Handle context,
                Arena lookupArena,
                MethodHandle create) {
            this.api = api;
            this.context = context;
            this.lookupArena = lookupArena;
            this.create = create;
        }

        static RawAbi open() {
            String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
            assumeTrue(configured != null && !configured.isBlank(),
                    "SYNAPTIK_METAL_TEST_LIBRARY is not set");
            Path library = Path.of(configured).toAbsolutePath().normalize();
            MetalNativeApi api = MetalNativeApi.open(library);
            MetalNativeApi.Handle context = null;
            Arena lookupArena = null;
            try {
                context = api.createContext();
                lookupArena = Arena.ofShared();
                SymbolLookup lookup = SymbolLookup.libraryLookup(library, lookupArena);
                MemorySegment symbol = lookup.find(MetalNativeApi.EXECUTABLE_CREATE_OPERATION)
                        .orElseThrow();
                MethodHandle create = Linker.nativeLinker().downcallHandle(
                        symbol,
                        FunctionDescriptor.of(
                                JAVA_INT,
                                ADDRESS,
                                JAVA_INT,
                                JAVA_INT,
                                ADDRESS,
                                ADDRESS,
                                JAVA_INT,
                                ADDRESS,
                                JAVA_INT,
                                ADDRESS,
                                JAVA_INT,
                                ADDRESS,
                                ADDRESS));
                return new RawAbi(api, context, lookupArena, create);
            } catch (RuntimeException | Error failure) {
                if (context != null) api.releaseContext(context);
                api.close();
                if (lookupArena != null) lookupArena.close();
                throw failure;
            }
        }

        void assertRejected(
                String name,
                int[] ranks,
                long[] dimensions,
                MetalMpsGraphProgram program,
                int[] feeds,
                int[] targets,
                Consumer<MemorySegment> mutation) {
            assertRejected(name, INVALID_ARGUMENT, MetalMpsGraphProgram.SCHEMA_VERSION,
                    ranks, dimensions, program, feeds, targets, mutation);
        }

        void assertRejected(
                String name,
                int expectedStatus,
                int[] ranks,
                long[] dimensions,
                MetalMpsGraphProgram program,
                int[] feeds,
                int[] targets,
                Consumer<MemorySegment> mutation) {
            assertRejected(name, expectedStatus, MetalMpsGraphProgram.SCHEMA_VERSION,
                    ranks, dimensions, program, feeds, targets, mutation);
        }

        void assertRejected(
                String name,
                int expectedStatus,
                int schemaVersion,
                int[] ranks,
                long[] dimensions,
                MetalMpsGraphProgram program,
                int[] feeds,
                int[] targets,
                Consumer<MemorySegment> mutation) {
            CreateOutcome outcome = invoke(
                    schemaVersion, ranks, dimensions, program, feeds, targets, mutation);
            assertEquals(expectedStatus, outcome.status(), name);
            assertEquals(0L, outcome.outputAddress(), name + " must null the output cell");
        }

        private CreateOutcome invoke(
                int schemaVersion,
                int[] ranks,
                long[] dimensions,
                MetalMpsGraphProgram program,
                int[] feeds,
                int[] targets,
                Consumer<MemorySegment> mutation) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment rankTable = copyInts(arena, ranks);
                MemorySegment dimensionTable = copyLongs(arena, dimensions);
                MemorySegment nodes = program.encodeNative(arena);
                assertEquals(
                        Math.multiplyExact(
                                (long) program.nodes().size(),
                                MetalMpsGraphProgram.NODE_RECORD_BYTES),
                        nodes.byteSize());
                assertTrue(nodes.byteSize() >= MetalMpsGraphProgram.NODE_RECORD_BYTES);
                mutation.accept(nodes);
                MemorySegment feedTable = copyInts(arena, feeds);
                MemorySegment targetTable = copyInts(arena, targets);
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.ofAddress(0x1234L));
                int status;
                try {
                    status = (int) create.invokeExact(
                            context.carrier(),
                            schemaVersion,
                            ranks.length,
                            rankTable,
                            dimensionTable,
                            program.nodes().size(),
                            nodes,
                            feeds.length,
                            feedTable,
                            targets.length,
                            targetTable,
                            output);
                } catch (RuntimeException | Error failure) {
                    throw failure;
                } catch (Throwable failure) {
                    throw new IllegalStateException("raw executable-create invocation failed", failure);
                }
                return new CreateOutcome(status, output.get(ADDRESS, 0L).address());
            }
        }

        @Override
        public void close() {
            try {
                api.releaseContext(context);
            } finally {
                try {
                    api.close();
                } finally {
                    lookupArena.close();
                }
            }
        }

        private static MemorySegment copyInts(Arena arena, int[] values) {
            MemorySegment result = arena.allocate(JAVA_INT, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_INT, index, values[index]);
            }
            return result;
        }

        private static MemorySegment copyLongs(Arena arena, long[] values) {
            MemorySegment result = arena.allocate(JAVA_LONG, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_LONG, index, values[index]);
            }
            return result;
        }
    }
}

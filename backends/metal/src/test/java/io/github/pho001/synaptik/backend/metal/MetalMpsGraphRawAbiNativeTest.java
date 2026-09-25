package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;

class MetalMpsGraphRawAbiNativeTest {
    private static final int INVALID_ARGUMENT = 1;
    private static final int UNSUPPORTED_SHAPE = 8;
    private static final Consumer<MemorySegment> UNCHANGED = ignored -> { };

    @Test
    void rawVersionTwelveRecordRejectsEveryMalformedHeaderAndUnusedField() {
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
            abi.assertRejected("unused auxiliary scalar", ranks, dimensions, reshape,
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
            abi.assertRejected("stale schema version five", INVALID_ARGUMENT, 5,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version six", INVALID_ARGUMENT, 6,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version seven", INVALID_ARGUMENT, 7,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version eight", INVALID_ARGUMENT, 8,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version nine", INVALID_ARGUMENT, 9,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version ten", INVALID_ARGUMENT, 10,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema version eleven", INVALID_ARGUMENT, 11,
                    ranks, dimensions, reshape, new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    @Test
    void rawVersionTwelveScatterUsesWireEighteenAndTypedAuxiliaryInput() {
        MetalMpsGraphProgram.Node scatter =
                MetalMpsGraphProgram.Node.scatterElements(0, 1, 2, 3, 1);
        assertEquals(12, MetalMpsGraphProgram.SCHEMA_VERSION);
        assertEquals(18, scatter.kind().wireIdentity());
        assertEquals(3, scatter.kind().inputCount());
        assertEquals(0, scatter.firstInputIndex());
        assertEquals(1, scatter.secondInputIndex());
        assertEquals(2, scatter.auxiliary());
        assertEquals(3, scatter.outputIndex());
        assertEquals(1, scatter.axis());

        try (RawAbi abi = RawAbi.open()) {
            int[] ranks = {2, 2, 2, 2};
            long[] dimensions = dimensions(new long[][] {
                    {2, 3}, {2, 2}, {2, 2}, {2, 3}
            });
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(scatter));
            abi.assertAccepted(
                    "canonical replacement scatter",
                    ranks,
                    dimensions,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3});
            abi.assertRejected(
                    "scatter auxiliary sentinel",
                    ranks,
                    dimensions,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    record -> record.set(JAVA_INT, 28L, -1));
            abi.assertRejected(
                    "scatter auxiliary out of range",
                    ranks,
                    dimensions,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    record -> record.set(JAVA_INT, 28L, 4));
            abi.assertRejected(
                    "scatter axis out of range",
                    ranks,
                    dimensions,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    record -> record.set(JAVA_INT, 24L, 2));
            abi.assertRejected(
                    "scatter indices and updates shape mismatch",
                    ranks,
                    dimensions(new long[][] {{2, 3}, {2, 2}, {2, 1}, {2, 3}}),
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    UNCHANGED);
            abi.assertRejected(
                    "scatter output and data shape mismatch",
                    ranks,
                    dimensions(new long[][] {{2, 3}, {2, 2}, {2, 2}, {2, 2}}),
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    UNCHANGED);
            abi.assertRejected(
                    "stale schema ten cannot reinterpret scatter",
                    INVALID_ARGUMENT,
                    10,
                    ranks,
                    dimensions,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    UNCHANGED);
        }
    }

    @Test
    void rawVersionTwelveUnfoldAxisUsesWireNineteenAndRejectsMalformedWindowState() {
        assertEquals(16, MetalMpsGraphProgram.MAX_SELECTOR_EXPANSION);
        assertThrows(IllegalArgumentException.class,
                () -> MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 17, 1),
                "Java schema rejects selector expansion above the named cap");
        MetalMpsGraphProgram.Node unfold =
                MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 3, 2);
        assertEquals(19, unfold.kind().wireIdentity());
        assertEquals(6, unfold.kind().attributeKind().wireIdentity());
        assertEquals(3, unfold.attributeCount());
        assertEquals(1, unfold.axis());
        assertEquals(0, unfold.auxiliary());
        assertEquals(List.of(3L, 2L),
                java.util.Arrays.stream(unfold.attributeValues()).boxed().toList());

        ByteBuffer encodedRecord = ByteBuffer.wrap(
                new MetalMpsGraphProgram(List.of(unfold)).encodedNodeRecords())
                .order(ByteOrder.BIG_ENDIAN);
        assertEquals(MetalMpsGraphProgram.NODE_RECORD_BYTES, encodedRecord.remaining());
        assertEquals(19, encodedRecord.getInt());
        assertEquals(6, encodedRecord.getInt());
        assertEquals(0, encodedRecord.getInt());
        assertEquals(-1, encodedRecord.getInt());
        assertEquals(1, encodedRecord.getInt());
        assertEquals(3, encodedRecord.getInt());
        assertEquals(1, encodedRecord.getInt());
        assertEquals(0, encodedRecord.getInt());
        assertEquals(3L, encodedRecord.getLong());
        assertEquals(2L, encodedRecord.getLong());
        while (encodedRecord.hasRemaining()) assertEquals(0L, encodedRecord.getLong());
        int[] ranks = {2, 3};
        long[] dimensions = dimensions(new long[][] {{2, 6}, {2, 2, 3}});
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(unfold));
        int[] capRanks = {2, 3};
        long[] capDimensions = dimensions(new long[][] {{2, 17}, {2, 1, 17}});
        MetalMpsGraphProgram capProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 16, 1)));
        try (RawAbi abi = RawAbi.open()) {
            abi.assertAccepted("canonical unfold axis", ranks, dimensions, program,
                    new int[] {0}, new int[] {1});
            abi.assertRejected("unfold attribute discriminator", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 3));
            abi.assertRejected("unfold semantic attribute count", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 2));
            abi.assertRejected("unfold axis out of range", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 2));
            abi.assertRejected("unfold zero size", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 0L));
            abi.assertRejected("unfold size seventeen exceeds only selector cap",
                    capRanks, capDimensions, capProgram,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 17L));
            abi.assertRejected("unfold zero step", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 40L, 0L));
            abi.assertRejected("unfold unused payload", ranks, dimensions, program,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 48L, 1L));
            abi.assertRejected("unfold wrong position extent", ranks,
                    dimensions(new long[][] {{2, 6}, {2, 3, 3}}), program,
                    new int[] {0}, new int[] {1}, UNCHANGED);
            abi.assertRejected("stale schema ten cannot reinterpret unfold",
                    INVALID_ARGUMENT, 10, ranks, dimensions, program,
                    new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    @Test
    void rawTask0052ScanRejectsMalformedTypedAttributesAndStaleSchema() {
        int[] ranks = {2, 2};
        long[] dimensions = dimensions(new long[][] {{2, 3}, {2, 3}});
        MetalMpsGraphProgram scan = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scan(
                        MetalMpsGraphProgram.NodeKind.CUM_SUM,
                        0,
                        1,
                        1,
                        true,
                        true)));
        try (RawAbi abi = RawAbi.open()) {
            abi.assertAccepted("canonical cumulative scan", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1});
            abi.assertRejected("scan attribute discriminator", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 0));
            abi.assertRejected("scan attribute count", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("scan axis", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 2));
            abi.assertRejected("scan exclusive flag", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 2L));
            abi.assertRejected("scan reverse flag", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 40L, 2L));
            abi.assertRejected("scan unused payload", ranks, dimensions, scan,
                    new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 48L, 1L));
            abi.assertRejected("stale schema eleven cannot reinterpret scan",
                    INVALID_ARGUMENT, 11, ranks, dimensions, scan,
                    new int[] {0}, new int[] {1}, UNCHANGED);
        }
    }

    @Test
    void rawTask0052RunRejectsDistinctLiveValuesSharingOnePhysicalBuffer() {
        try (RawAbi abi = RawAbi.open();
                Arena arena = Arena.ofConfined()) {
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.TENSOR_MIN, 0, 1, 2)));
            MetalNativeApi.Handle executable = abi.api.createMpsGraphExecutable(
                    abi.context,
                    io.github.pho001.synaptik.config.compile.NumericalProfile.ACCELERATOR,
                    new int[] {1, 1, 1},
                    dimensions(new long[][] {{2}, {2}, {2}}),
                    program,
                    new int[] {0, 1},
                    new int[] {2});
            MetalNativeApi.Handle left = abi.api.createBuffer(abi.context, 8);
            MetalNativeApi.Handle right = abi.api.createBuffer(abi.context, 8);
            try {
                MemorySegment values = arena.allocate(ADDRESS, 3);
                values.setAtIndex(ADDRESS, 0, left.carrier());
                values.setAtIndex(ADDRESS, 1, right.carrier());
                values.setAtIndex(ADDRESS, 2, left.carrier());
                MemorySegment target = arena.allocate(ADDRESS);
                target.set(ADDRESS, 0L, left.carrier());

                MetalNativeApi.NativeFailure failure = assertThrows(
                        MetalNativeApi.NativeFailure.class,
                        () -> abi.api.runExecutable(executable, 3, values, 1, target));

                assertEquals(MetalNativeApi.Status.INCOMPATIBLE_RESOURCE, failure.status());
            } finally {
                abi.api.releaseBuffer(right);
                abi.api.releaseBuffer(left);
                abi.api.releaseExecutable(executable);
            }
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
            abi.assertRejected("binary unused auxiliary cell", ranks, dimensions, binary,
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
    void rawReductionRecordsRejectMalformedFormsGeometryAndScalarTopology() {
        try (RawAbi abi = RawAbi.open()) {
            int[] singleRanks = {3, 2};
            long[] singleDimensions =
                    dimensions(new long[][] {{2, 3, 4}, {2, 4}});
            MetalMpsGraphProgram single = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                            List.of(1),
                            false)));
            abi.assertRejected("reduction attribute discriminator", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 4L, 0));
            abi.assertRejected("unknown reduction form", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 24L, 99));
            abi.assertRejected("reduction keep flag exceeds boolean", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 28L, 2));
            abi.assertRejected("single-axis count is not one", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 0));
            abi.assertRejected("single-axis value is out of range", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 3L));
            abi.assertRejected("single-axis keep shape disagrees", singleRanks,
                    singleDimensions, single, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 28L, 1));

            int[] multiRanks = {3, 1};
            long[] multiDimensions =
                    dimensions(new long[][] {{2, 3, 4}, {3}});
            MetalMpsGraphProgram multi = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.MEAN,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                            List.of(2, 0),
                            false)));
            abi.assertRejected("duplicate multi-axis value", multiRanks,
                    multiDimensions, multi, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 40L, 2L));
            abi.assertRejected("multi-axis value is out of range", multiRanks,
                    multiDimensions, multi, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 40L, 3L));
            abi.assertRejected("unused multi-axis payload", multiRanks,
                    multiDimensions, multi, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 48L, 1L));
            abi.assertRejected("multi-axis output shape disagrees",
                    new int[] {3, 1},
                    dimensions(new long[][] {{2, 3, 4}, {2}}),
                    multi, new int[] {0}, new int[] {1}, UNCHANGED);

            MetalMpsGraphProgram full = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(),
                            false)));
            int[] fullRanks = {3, 0};
            long[] fullDimensions =
                    dimensions(new long[][] {{2, 3, 4}, {}});
            abi.assertRejected("full reduction keep flag", fullRanks,
                    fullDimensions, full, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("full reduction count", fullRanks,
                    fullDimensions, full, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("full reduction unused payload", fullRanks,
                    fullDimensions, full, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 1L));
            abi.assertRejected("full reduction output is not scalar",
                    new int[] {3, 1},
                    dimensions(new long[][] {{2, 3, 4}, {1}}),
                    full, new int[] {0}, new int[] {1}, UNCHANGED);

            MetalMpsGraphProgram sumTo = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.sumToShape(
                            0, 1, new long[] {1, 4})));
            int[] sumToRanks = {3, 2};
            long[] sumToDimensions =
                    dimensions(new long[][] {{2, 3, 4}, {1, 4}});
            abi.assertRejected("sum-to-Shape is SUM only", sumToRanks,
                    sumToDimensions, sumTo, new int[] {0}, new int[] {1},
                    record -> record.set(
                            JAVA_INT, 0L,
                            MetalMpsGraphProgram.NodeKind.MEAN.wireIdentity()));
            abi.assertRejected("sum-to-Shape target payload is positive and exact", sumToRanks,
                    sumToDimensions, sumTo, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 0L));
            abi.assertRejected("sum-to-Shape target must right-align to source", sumToRanks,
                    sumToDimensions, sumTo, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 2L));

            abi.assertRejected("rank-zero feed is forbidden", INVALID_ARGUMENT,
                    new int[] {0, 0},
                    dimensions(new long[][] {{}, {}}),
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.reduction(
                                    MetalMpsGraphProgram.NodeKind.SUM,
                                    0,
                                    1,
                                    MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                                    List.of(),
                                    false))),
                    new int[] {0}, new int[] {1}, UNCHANGED);

            MetalMpsGraphProgram scalarConsumer = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            0,
                            1,
                            MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(),
                            false),
                    MetalMpsGraphProgram.Node.abs(1, 2)));
            abi.assertRejected("rank-zero reduction must be a direct target",
                    new int[] {3, 0, 0},
                    dimensions(new long[][] {{2, 3, 4}, {}, {}}),
                    scalarConsumer, new int[] {0}, new int[] {2}, UNCHANGED);
        }
    }

    @Test
    void rawMatmulRecordsRejectMalformedGeometryAndAuthenticateOnlyConsumedAffineEdges() {
        try (RawAbi abi = RawAbi.open()) {
            int[] ranks = {2, 2, 2};
            long[] dimensions = dimensions(new long[][] {{2, 3}, {3, 4}, {2, 4}});
            MetalMpsGraphProgram direct = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.matmul(0, 1, 2)));
            abi.assertRejected("MATMUL second-input sentinel", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 12L, -1));
            abi.assertRejected("MATMUL attribute discriminator", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 4L, 1));
            abi.assertRejected("MATMUL attribute count", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 20L, 1));
            abi.assertRejected("MATMUL axis sentinel", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 24L, 0));
            abi.assertRejected("MATMUL unused auxiliary cell", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("MATMUL unused payload", ranks, dimensions, direct,
                    new int[] {0, 1}, new int[] {2},
                    record -> record.set(JAVA_LONG, 32L, 1L));
            abi.assertRejected(
                    "MATMUL contraction dimensions disagree",
                    ranks,
                    dimensions(new long[][] {{2, 3}, {2, 4}, {2, 4}}),
                    direct,
                    new int[] {0, 1},
                    new int[] {2},
                    UNCHANGED);
            abi.assertRejected(
                    "MATMUL output dimensions disagree",
                    ranks,
                    dimensions(new long[][] {{2, 3}, {3, 4}, {2, 5}}),
                    direct,
                    new int[] {0, 1},
                    new int[] {2},
                    UNCHANGED);
            abi.assertRejected(
                    "MATMUL rank must be two",
                    new int[] {1, 2, 2},
                    dimensions(new long[][] {{3}, {3, 4}, {1, 4}}),
                    direct,
                    new int[] {0, 1},
                    new int[] {2},
                    UNCHANGED);

            MetalMpsGraphProgram hiddenMaterialization = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            2,
                            new long[] {2, 3}),
                    MetalMpsGraphProgram.Node.matmul(2, 1, 3)));
            abi.assertRejected(
                    "MATMUL affine input must be exact local transpose",
                    new int[] {1, 2, 2, 2},
                    dimensions(new long[][] {{6}, {3, 4}, {2, 3}, {2, 4}}),
                    hiddenMaterialization,
                    new int[] {0, 1},
                    new int[] {3},
                    UNCHANGED);

            MetalMpsGraphProgram localTranspose = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.permutation(0, 2, List.of(1, 0)),
                    MetalMpsGraphProgram.Node.matmul(2, 1, 3)));
            abi.assertAccepted(
                    "MATMUL local transpose may also be a target",
                    new int[] {2, 2, 2, 2},
                    dimensions(new long[][] {{3, 2}, {3, 4}, {2, 3}, {2, 4}}),
                    localTranspose,
                    new int[] {0, 1},
                    new int[] {2, 3});
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
    void rawNegAbsAndMalformedUnaryEncodingsOfBinaryOperationsRejectEveryField() {
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
            abi.assertRejected("NEG unused auxiliary scalar", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 28L, 1));
            abi.assertRejected("NEG unused payload", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_LONG, 32L, 1L));
            abi.assertRejected("NEG second-input sentinel", unaryRanks, unaryDimensions,
                    neg, new int[] {0}, new int[] {1},
                    record -> record.set(JAVA_INT, 12L, 0));
            MetalMpsGraphProgram abs = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.abs(0, 1)));
            for (int operation : new int[] {13, 14, 99}) {
                abi.assertRejected(
                        "unknown post-ABS wire " + operation,
                        unaryRanks,
                        unaryDimensions,
                        abs,
                        new int[] {0},
                        new int[] {1},
                        record -> record.set(JAVA_INT, 0L, operation));
            }

            for (int operation = 2; operation <= 5; operation++) {
                int binaryOperation = operation;
                abi.assertRejected(
                        "current binary operation encoded with unary topology "
                                + binaryOperation,
                        unaryRanks,
                        unaryDimensions,
                        neg,
                        new int[] {0},
                        new int[] {1},
                        record -> record.set(JAVA_INT, 0L, binaryOperation));
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

            abi.assertRejected("rank-zero feed", INVALID_ARGUMENT,
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

        void assertAccepted(
                String name,
                int[] ranks,
                long[] dimensions,
                MetalMpsGraphProgram program,
                int[] feeds,
                int[] targets) {
            CreateOutcome outcome = invoke(
                    MetalMpsGraphProgram.SCHEMA_VERSION,
                    ranks,
                    dimensions,
                    program,
                    feeds,
                    targets,
                    UNCHANGED);
            assertEquals(0, outcome.status(), name);
            assertTrue(outcome.outputAddress() != 0L, name + " must return an executable");
            api.releaseExecutable(
                    new MetalNativeApi.Handle(MemorySegment.ofAddress(outcome.outputAddress())));
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

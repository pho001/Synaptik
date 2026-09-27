package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.pho001.synaptik.config.compile.NumericalProfile;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

class MetalMpsGraphIndexingNativeTest {
    private static final int[] DATA_BITS = {
        0x00000000, 0x80000000, 0x00000001, 0x7fc12345,
        0xffc54321, 0x7f800000, 0x80000001, 0x3f800000
    };
    private static final byte SENTINEL = (byte) 0x5a;

    @Test
    void realGatherMovesRawFloat32BitsAndOneHotWritesExactBoolBytesBeforeBoundsFailures() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.gather(0, 1, 2, 1),
                    MetalMpsGraphProgram.Node.oneHot(3, 4, 4)));
            long[][] shapes = {{2, 4}, {3}, {2, 3}, {3}, {3, 4}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), program), program, new int[] {0, 1, 3}, new int[] {2, 4}, MetalPreparedRoute.CUSTOM_PROGRAM);

            MetalNativeApi.Handle data = api.createBuffer(context, 8L * Integer.BYTES);
            MetalNativeApi.Handle gatherIndices = api.createBuffer(context, 3L * Integer.BYTES);
            MetalNativeApi.Handle oneHotIndices = api.createBuffer(context, 3L * Integer.BYTES);
            MetalNativeApi.Handle gathered = api.createBuffer(context, 6L * Integer.BYTES);
            MetalNativeApi.Handle oneHot = api.createBuffer(context, 12L);
            inputs.add(data);
            inputs.add(gatherIndices);
            inputs.add(gathered);
      inputs.add(oneHotIndices);
      inputs.add(oneHot);
            outputs.add(gathered);
            outputs.add(oneHot);

            uploadInts(api, data, DATA_BITS);
            uploadInts(api, gatherIndices, new int[] {3, 0, 2});
            uploadInts(api, oneHotIndices, new int[] {2, 0, 3});
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }

                api.runExecutable(
                        executable, inputs.size(), inputAddresses,
                        outputs.size(), outputAddresses);
                assertArrayEquals(new int[] {
                    0x7fc12345, 0x00000000, 0x00000001,
                    0x3f800000, 0xffc54321, 0x80000001
                }, downloadInts(api, gathered, 6));
                assertArrayEquals(new byte[] {
                    0, 0, 1, 0,
                    1, 0, 0, 0,
                    0, 0, 0, 1
                }, downloadBytes(api, oneHot, 12));
                assertArrayEquals(DATA_BITS, downloadInts(api, data, DATA_BITS.length));

                uploadInts(api, gatherIndices, new int[] {3, 4, 0});
                uploadInts(api, oneHotIndices, new int[] {2, 0, 3});
                fill(api, gathered, 6L * Integer.BYTES, SENTINEL);
                fill(api, oneHot, 12L, SENTINEL);
                assertBoundsFailure(api, executable, inputAddresses, outputAddresses);
                assertFilled(downloadBytes(api, gathered, 6 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, oneHot, 12), SENTINEL);

                uploadInts(api, gatherIndices, new int[] {3, 0, 2});
                uploadInts(api, oneHotIndices, new int[] {2, -1, 3});
                assertBoundsFailure(api, executable, inputAddresses, outputAddresses);
                assertFilled(downloadBytes(api, gathered, 6 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, oneHot, 12), SENTINEL);
            }
        } finally {
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void task0066ScalarGatherAndOneHotExecuteThroughCustomProgram() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program =
                    new MetalMpsGraphProgram(
                            List.of(
                                    MetalMpsGraphProgram.Node.gather(0, 1, 2, 0),
                                    MetalMpsGraphProgram.Node.oneHot(3, 4, 5)));
            long[][] shapes = {{4}, {}, {}, {}, {5}};
            executable =
                    api.createMpsGraphExecutable(
                            context,
                            NumericalProfile.STRICT_IEEE,
                            MetalTestProgram.descriptors(
                                    ranks(shapes), dimensions(shapes), program),
                            program,
                            new int[] {0, 1, 3},
                            new int[] {2, 4},
                            MetalPreparedRoute.CUSTOM_PROGRAM);
            buffers.add(api.createBuffer(context, 4L * Integer.BYTES));
            buffers.add(api.createBuffer(context, Integer.BYTES));
            buffers.add(api.createBuffer(context, Integer.BYTES));
            buffers.add(api.createBuffer(context, Integer.BYTES));
            buffers.add(api.createBuffer(context, 5L));
            uploadInts(
                    api,
                    buffers.get(0),
                    new int[] {DATA_BITS[0], DATA_BITS[1], DATA_BITS[2], DATA_BITS[3]});
            uploadInts(api, buffers.get(1), new int[] {2});
            uploadInts(api, buffers.get(3), new int[] {3});
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment allValues = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    allValues.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment targets = arena.allocate(ADDRESS, 2);
                targets.setAtIndex(ADDRESS, 0, buffers.get(2).carrier());
                targets.setAtIndex(ADDRESS, 1, buffers.get(4).carrier());
                api.runExecutable(executable, buffers.size(), allValues, 2, targets);
            }
            assertArrayEquals(new int[] {DATA_BITS[2]}, downloadInts(api, buffers.get(2), 1));
            assertArrayEquals(new byte[] {0, 0, 0, 1, 0}, downloadBytes(api, buffers.get(4), 5));
        } finally {
            for (int index = buffers.size(); index-- > 0; ) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void task0059GatherElementsAndGatherNdRejectBothIndexWidthsBeforeAnyWrite() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                            new int[] {0, 1}, new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {1}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.GATHER_ND,
                            new int[] {0, 3}, new int[] {4},
                            MetalMpsGraphProgram.AttributeKind.GATHER_ND, new long[] {0})));
            var values = List.of(
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 3),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.INT32, 2, 2),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 2),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.INT64, 2, 1),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 3));
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    values,
                    program,
                    new int[] {0, 1, 3},
                    new int[] {2, 4},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) {
                buffers.add(api.createBuffer(context, value.byteCount()));
            }
            uploadInts(api, buffers.get(0), new int[] {
                DATA_BITS[0], DATA_BITS[1], DATA_BITS[2],
                DATA_BITS[3], DATA_BITS[4], DATA_BITS[5]
            });
            uploadInts(api, buffers.get(1), new int[] {2, 0, 1, 1});
            uploadLongs(api, buffers.get(3), new long[] {1, 0});
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment allValues = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    allValues.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment targets = arena.allocate(ADDRESS, 2);
                targets.setAtIndex(ADDRESS, 0, buffers.get(2).carrier());
                targets.setAtIndex(ADDRESS, 1, buffers.get(4).carrier());
                api.runExecutable(
                        executable, buffers.size(), allValues, 2, targets);
                assertArrayEquals(new int[] {
                    DATA_BITS[2], DATA_BITS[0], DATA_BITS[4], DATA_BITS[4]
                }, downloadInts(api, buffers.get(2), 4));
                assertArrayEquals(new int[] {
                    DATA_BITS[3], DATA_BITS[4], DATA_BITS[5],
                    DATA_BITS[0], DATA_BITS[1], DATA_BITS[2]
                }, downloadInts(api, buffers.get(4), 6));

                fill(api, buffers.get(2), 4L * Integer.BYTES, SENTINEL);
                fill(api, buffers.get(4), 6L * Integer.BYTES, SENTINEL);
                uploadInts(api, buffers.get(1), new int[] {2, -1, 1, 1});
                assertCustomBoundsFailure(api, executable, buffers.size(), allValues, targets);
                assertFilled(downloadBytes(api, buffers.get(2), 4 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, buffers.get(4), 6 * Integer.BYTES), SENTINEL);

                uploadInts(api, buffers.get(1), new int[] {2, 0, 1, 1});
                uploadLongs(api, buffers.get(3), new long[] {1, 2});
                assertCustomBoundsFailure(api, executable, buffers.size(), allValues, targets);
                assertFilled(downloadBytes(api, buffers.get(2), 4 * Integer.BYTES), SENTINEL);
                assertFilled(downloadBytes(api, buffers.get(4), 6 * Integer.BYTES), SENTINEL);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void task0059StagesValidationAfterAnInternalIndexCast() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.CAST,
                            new int[] {1}, new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                            new long[] {2}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                            new int[] {0, 2}, new int[] {3},
                            MetalMpsGraphProgram.AttributeKind.AXIS,
                            new long[] {1})));
            var values = List.of(
                    descriptor(
                            io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                            2, 3),
                    descriptor(
                            io.github.pho001.synaptik.model.datatype.DataType.INT64,
                            2, 2),
                    descriptor(
                            io.github.pho001.synaptik.model.datatype.DataType.INT32,
                            2, 2),
                    descriptor(
                            io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                            2, 2));
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    values,
                    program,
                    new int[] {0, 1},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) {
                buffers.add(api.createBuffer(context, value.byteCount()));
            }
            uploadInts(api, buffers.get(0), new int[] {
                DATA_BITS[0], DATA_BITS[1], DATA_BITS[2],
                DATA_BITS[3], DATA_BITS[4], DATA_BITS[5]
            });
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment allValues = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    allValues.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment target = arena.allocate(ADDRESS);
                target.set(ADDRESS, 0L, buffers.get(3).carrier());

                uploadLongs(api, buffers.get(1), new long[] {2, 0, 1, 1});
                api.runExecutable(executable, buffers.size(), allValues, 1, target);
                assertArrayEquals(
                        new int[] {
                            DATA_BITS[2], DATA_BITS[0], DATA_BITS[4], DATA_BITS[4]
                        },
                        downloadInts(api, buffers.get(3), 4));

                fill(api, buffers.get(3), 4L * Integer.BYTES, SENTINEL);
                uploadLongs(api, buffers.get(1), new long[] {2, -1, 1, 1});
                assertRangeFailure(api, executable, buffers.size(), allValues, 1, target);
                assertFilled(downloadBytes(api, buffers.get(3), 4 * Integer.BYTES), SENTINEL);

                uploadLongs(api, buffers.get(1), new long[] {2, 0, 1, 1});
                api.runExecutable(executable, buffers.size(), allValues, 1, target);
                assertArrayEquals(
                        new int[] {
                            DATA_BITS[2], DATA_BITS[0], DATA_BITS[4], DATA_BITS[4]
                        },
                        downloadInts(api, buffers.get(3), 4));
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void realReplacementScatterMovesRawBitsAndRejectsBoundsAndDuplicatesBeforeWrites() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        int[] dataBits = {
            0x00000000, 0x80000000, 0x00000001,
            0x7fa12345, 0xffa54321, 0x3f800000
        };
        int[] updateBits = {0x7fa22222, 0xffa33333, 0x80000001, 0x7f800000};
        try {
            context = api.createContext();
            MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.scatterElements(0, 1, 2, 3, 1)));
            long[][] shapes = {{2, 3}, {2, 2}, {2, 2}, {2, 3}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), program), program, new int[] {0, 1, 2}, new int[] {3}, MetalPreparedRoute.CUSTOM_PROGRAM);

            MetalNativeApi.Handle data = api.createBuffer(context, 6L * Integer.BYTES);
            MetalNativeApi.Handle indices = api.createBuffer(context, 4L * Integer.BYTES);
            MetalNativeApi.Handle updates = api.createBuffer(context, 4L * Integer.BYTES);
            MetalNativeApi.Handle output = api.createBuffer(context, 6L * Integer.BYTES);
            inputs.add(data);
            inputs.add(indices);
            inputs.add(updates);
      inputs.add(output);
            outputs.add(output);
            uploadInts(api, data, dataBits);
            uploadInts(api, indices, new int[] {2, 0, 1, 2});
            uploadInts(api, updates, updateBits);

            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                outputAddresses.setAtIndex(ADDRESS, 0, output.carrier());

                api.runExecutable(executable, inputs.size(), inputAddresses, 1, outputAddresses);
                assertArrayEquals(new int[] {
                    updateBits[1], dataBits[1], updateBits[0],
                    dataBits[3], updateBits[2], updateBits[3]
                }, downloadInts(api, output, 6));
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {2, 0, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));

                uploadInts(api, indices, new int[] {2, 3, 1, 2});
                fill(api, output, 6L * Integer.BYTES, SENTINEL);
                assertRangeFailure(api, executable, inputs.size(), inputAddresses, 1, outputAddresses);
                assertFilled(downloadBytes(api, output, 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {2, 3, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));

                uploadInts(api, indices, new int[] {1, 1, 1, 2});
                assertRangeFailure(api, executable, inputs.size(), inputAddresses, 1, outputAddresses);
                assertFilled(downloadBytes(api, output, 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, data, 6));
                assertArrayEquals(new int[] {1, 1, 1, 2}, downloadInts(api, indices, 4));
                assertArrayEquals(updateBits, downloadInts(api, updates, 4));
            }
        } finally {
            for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void task0060ScatterNdPreflightsAllTuplesBeforeCopyOrReplacement() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        int[] dataBits = {
            0x8000_0000, 0x7fc1_2345, 0x0000_0001,
            0xffc5_4321, 0x8000_0001, 0x3f80_0000
        };
        int[] updateBits = {0x7fa2_2222, 0xffa3_3333, 0x7f80_0000, 0xff80_0000};
        try {
            context = api.createContext();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.SCATTER_ND,
                            new int[] {0, 1, 2},
                            new int[] {3},
                            MetalMpsGraphProgram.AttributeKind.SCATTER_ND,
                            new long[] {1, 1})));
            var values = List.of(
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 3),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.INT64, 2, 2, 1),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 2),
                    descriptor(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32, 2, 3));
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.STRICT_IEEE,
                    values,
                    program,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, value.byteCount()));
            uploadInts(api, buffers.get(0), dataBits);
            uploadLongs(api, buffers.get(1), new long[] {0, 2, 1, 0});
            uploadInts(api, buffers.get(2), updateBits);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment allValues = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    allValues.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment target = arena.allocate(ADDRESS);
                target.set(ADDRESS, 0L, buffers.get(3).carrier());
                api.runExecutable(executable, buffers.size(), allValues, 1, target);
                assertArrayEquals(new int[] {
                    updateBits[0], dataBits[1], updateBits[1],
                    updateBits[3], updateBits[2], dataBits[5]
                }, downloadInts(api, buffers.get(3), 6));

                fill(api, buffers.get(3), 6L * Integer.BYTES, SENTINEL);
                uploadLongs(api, buffers.get(1), new long[] {1, 1, 1, 0});
                assertRangeFailure(
                        api, executable, buffers.size(), allValues, 1, target);
                assertFilled(downloadBytes(api, buffers.get(3), 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, buffers.get(0), 6));
                assertArrayEquals(updateBits, downloadInts(api, buffers.get(2), 4));

                uploadLongs(api, buffers.get(1), new long[] {0, 3, 1, 0});
                assertRangeFailure(
                        api, executable, buffers.size(), allValues, 1, target);
                assertFilled(downloadBytes(api, buffers.get(3), 6 * Integer.BYTES), SENTINEL);
                assertArrayEquals(dataBits, downloadInts(api, buffers.get(0), 6));
                assertArrayEquals(updateBits, downloadInts(api, buffers.get(2), 4));
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void realUnfoldAxisMapsOverlapAndTailWithExactFloat32BitsWithoutMutatingInput() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        MetalNativeApi.Handle output = null;
        int[] inputBits = {
            0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000,
            0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321, 0x3f800000, 0xbf800000
        };
        int[] expected = {
            inputBits[0], inputBits[1], inputBits[2],
            inputBits[2], inputBits[3], inputBits[4],
            inputBits[6], inputBits[7], inputBits[8],
            inputBits[8], inputBits[9], inputBits[10]
        };
        try {
            context = api.createContext();
            long[][] shapes = {{2, 6}, {2, 2, 3}};
            executable = api.createMpsGraphExecutable(context, NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks(shapes), dimensions(shapes), new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 3, 2)))), new MetalMpsGraphProgram(List.of(
            MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 1, 3, 2))), new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
            input = api.createBuffer(context, (long) inputBits.length * Integer.BYTES);
            output = api.createBuffer(context, (long) expected.length * Integer.BYTES);
            uploadInts(api, input, inputBits);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddress = arena.allocate(ADDRESS);
                MemorySegment outputAddress = arena.allocate(ADDRESS);
                inputAddress.setAtIndex(ADDRESS, 0, input.carrier());
        MemorySegment allValues = arena.allocate(ADDRESS, 2);
        allValues.setAtIndex(ADDRESS, 0, input.carrier());
        allValues.setAtIndex(ADDRESS, 1, output.carrier());
                outputAddress.setAtIndex(ADDRESS, 0, output.carrier());
                api.runExecutable(executable, 2, allValues, 1, outputAddress);
            }
            assertArrayEquals(expected, downloadInts(api, output, expected.length));
            assertArrayEquals(inputBits, downloadInts(api, input, inputBits.length));
        } finally {
            if (output != null) api.releaseBuffer(output);
            if (input != null) api.releaseBuffer(input);
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    @Test
    void task0066UnfoldAxisWindowMayExceedHistoricalSelectorLimit() {
        Path library = configuredLibrary();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        MetalNativeApi.Handle input = null;
        MetalNativeApi.Handle output = null;
        int[] inputBits = new int[20];
        for (int index = 0; index < inputBits.length; index++) {
            inputBits[index] = 0x3f000000 + index;
        }
        int[] expected = new int[17];
        System.arraycopy(inputBits, 0, expected, 0, expected.length);
        try {
            context = api.createContext();
            long[][] shapes = {{20}, {1, 17}};
            var program =
                    new MetalMpsGraphProgram(
                            List.of(MetalMpsGraphProgram.Node.unfoldAxis(0, 1, 0, 17, 4)));
            executable =
                    api.createMpsGraphExecutable(
                            context,
                            NumericalProfile.STRICT_IEEE,
                            MetalTestProgram.descriptors(
                                    ranks(shapes), dimensions(shapes), program),
                            program,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM);
            input = api.createBuffer(context, (long) inputBits.length * Integer.BYTES);
            output = api.createBuffer(context, (long) expected.length * Integer.BYTES);
            uploadInts(api, input, inputBits);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment allValues = arena.allocate(ADDRESS, 2);
                allValues.setAtIndex(ADDRESS, 0, input.carrier());
                allValues.setAtIndex(ADDRESS, 1, output.carrier());
                MemorySegment target = arena.allocate(ADDRESS);
                target.set(ADDRESS, 0L, output.carrier());
                api.runExecutable(executable, 2, allValues, 1, target);
            }
            assertArrayEquals(expected, downloadInts(api, output, expected.length));
            assertArrayEquals(inputBits, downloadInts(api, input, inputBits.length));
        } finally {
            if (output != null) api.releaseBuffer(output);
            if (input != null) api.releaseBuffer(input);
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(
            io.github.pho001.synaptik.model.datatype.DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static void assertCustomBoundsFailure(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            int valueCount,
            MemorySegment values,
            MemorySegment targets) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class,
                () -> api.runExecutable(executable, valueCount, values, 2, targets));
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
    }

    private static void assertRangeFailure(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            int inputCount,
            MemorySegment inputs,
            int outputCount,
            MemorySegment outputs) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class,
                () -> api.runExecutable(
                        executable, inputCount, inputs, outputCount, outputs));
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
    }

    private static void assertBoundsFailure(
            MetalNativeApi api,
            MetalNativeApi.Handle executable,
            MemorySegment inputs,
            MemorySegment outputs) {
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class,
                () -> api.runExecutable(executable, 5, inputs, 2, outputs));
        assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, failure.status());
    }

    private static void uploadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) values.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < values.length; index++) {
                source.setAtIndex(JAVA_INT, index, values[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static void uploadLongs(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long[] values) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) values.length, Long.BYTES), Long.BYTES);
            for (int index = 0; index < values.length; index++) {
                source.setAtIndex(JAVA_LONG, index, values[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static void fill(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long bytes, byte value) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes, 1L);
            source.fill(value);
            api.upload(buffer, 0L, source, bytes);
        }
    }

    private static int[] downloadInts(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate((long) count * Integer.BYTES, Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] values = new int[count];
            for (int index = 0; index < count; index++) {
                values[index] = target.getAtIndex(JAVA_INT, index);
            }
            return values;
        }
    }

    private static byte[] downloadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(count, 1L);
            api.download(buffer, 0L, target, count);
            return target.toArray(JAVA_BYTE);
        }
    }

    private static void assertFilled(byte[] values, byte expected) {
        for (byte value : values) assertEquals(expected, value);
    }

    private static int[] ranks(long[][] shapes) {
        int[] result = new int[shapes.length];
        for (int index = 0; index < shapes.length; index++) result[index] = shapes[index].length;
        return result;
    }

    private static long[] dimensions(long[][] shapes) {
        long[] result = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, result, value * 16, shapes[value].length);
        }
        return result;
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}

package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MetalRemainingElementwiseNativeTest {
    private static final int[] INPUT = {
        0x00000000, 0x80000000,
        0x00000001, 0x80000001,
        0x007fffff, 0x807fffff,
        0x00800000, 0x80800000,
        0x3f7fffff, 0xbf7fffff,
        0x3f800000, 0xbf800000,
        0x3f800001, 0xbf800001,
        0x3fc00000, 0xbfc00000,
        0x40000000, 0xc0000000,
        0x4affffff, 0xcaffffff,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };
    private static final int[] FLOOR = {
        0x00000000, 0x80000000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x00000000, 0xbf800000,
        0x3f800000, 0xbf800000,
        0x3f800000, 0xc0000000,
        0x3f800000, 0xc0000000,
        0x40000000, 0xc0000000,
        0x4afffffe, 0xcb000000,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };
    private static final int[] CEIL = {
        0x00000000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0x80000000,
        0x3f800000, 0xbf800000,
        0x40000000, 0xbf800000,
        0x40000000, 0xbf800000,
        0x40000000, 0xc0000000,
        0x4b000000, 0xcafffffe,
        0x4b000000, 0xcb000000,
        0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000,
        0x7fc12345, 0xffc54321,
        0x7f812345, 0xff812346
    };

    @Test
    void productionCustomProgramExecutesAllFourExactRawOperationsUnderBothProfiles() {
        Path library = configuredLibrary();
        for (NumericalProfile profile : NumericalProfile.values()) {
            var program = new MetalMpsGraphProgram(List.of(
                    unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                    unary(MetalMpsGraphProgram.NodeKind.CEIL, 0, 2),
                    unary(MetalMpsGraphProgram.NodeKind.SIGN, 0, 3),
                    unary(MetalMpsGraphProgram.NodeKind.RELU, 0, 4)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    value(INPUT.length), value(INPUT.length), value(INPUT.length),
                    value(INPUT.length), value(INPUT.length));
            List<int[]> actual = executeCustom(
                    library, profile, program, values, new int[] {0},
                    new int[] {1, 2, 3, 4}, List.of(INPUT));
            assertModelWords(FLOOR, actual.get(0));
            assertModelWords(CEIL, actual.get(1));
            int[] sign = new int[INPUT.length];
            int[] relu = new int[INPUT.length];
            for (int index = 0; index < INPUT.length; index++) {
                int word = INPUT[index];
                if (isNaN(word) || (word & 0x7fff_ffff) == 0) sign[index] = word;
                else sign[index] = word < 0 ? 0xbf80_0000 : 0x3f80_0000;
                relu[index] = isNaN(word) ? word : word < 0 ? 0 : word;
            }
            assertModelWords(sign, actual.get(2));
            assertModelWords(relu, actual.get(3));
        }
    }

    @Test
    void generatedThreeInstructionChainUsesCompactBoundarySlotsAndPreservesRawBits() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.SIGN, 2, 3)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                value(INPUT.length), value(INPUT.length), value(INPUT.length), value(INPUT.length));
        for (NumericalProfile profile : NumericalProfile.values()) {
            MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                    profile,
                    program,
                    values,
                    new int[] {0},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(1, fusion.generatedUnitCount());
            assertEquals(3, fusion.instructions().size());
            assertArrayEquals(new int[] {0, 3}, fusion.materializedProgramValueIndices());
            int[] expected = new int[INPUT.length];
            for (int index = 0; index < INPUT.length; index++) {
                int word = FLOOR[index];
                if (isNaN(word) || (word & 0x7fff_ffff) == 0) expected[index] = word;
                else expected[index] = word < 0 ? 0xbf80_0000 : 0x3f80_0000;
            }
            List<int[]> actual = executeCustom(
                    library,
                    profile,
                    program,
                    values,
                    new int[] {0},
                    new int[] {3},
                    List.of(INPUT));
            assertModelWords(expected, actual.getFirst());
        }
    }

    @Test
    void task0059CustomProgramExecutesEveryExactRawMovementRecipe() {
        Path library = configuredLibrary();
        int[] data = {
            0x8000_0000, 0x7fc1_2345, 0x0000_0001,
            0x3f80_0000, 0xffc5_4321, 0x8000_0001
        };
        int fill = 0x7fa2_2222;
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {1}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.GATHER_ND,
                        new int[] {0, 3}, new int[] {4},
                        MetalMpsGraphProgram.AttributeKind.GATHER_ND, new long[] {0}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SELECT,
                        new int[] {0}, new int[] {5},
                        MetalMpsGraphProgram.AttributeKind.SELECT, new long[] {0, 1}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.PAD,
                        new int[] {0}, new int[] {6},
                        MetalMpsGraphProgram.AttributeKind.PAD,
                        new long[] {2, 0, 1, 0, 1, 1, Integer.toUnsignedLong(fill)}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SLICE,
                        new int[] {0}, new int[] {7},
                        MetalMpsGraphProgram.AttributeKind.SLICE,
                        new long[] {1, 1, 2, 1, 1}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.CONCAT,
                        new int[] {0, 0}, new int[] {8},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.STACK,
                        new int[] {0, 0}, new int[] {9},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.TILE,
                        new int[] {0}, new int[] {10},
                        MetalMpsGraphProgram.AttributeKind.TILE, new long[] {2, 1, 2}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                        new int[] {11}, new int[] {12},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                        new long[] {2, 2, 1, 1, 0, 0, 1, 1, 0}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.UNFOLD3D,
                        new int[] {13}, new int[] {14},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                        new long[] {2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.CAST,
                        new int[] {15}, new int[] {16},
                        MetalMpsGraphProgram.AttributeKind.CAST_TARGET, new long[] {6})));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.FLOAT32, 2, 3),
                typed(DataType.INT32, 2, 2),
                typed(DataType.FLOAT32, 2, 2),
                typed(DataType.INT64, 2, 1),
                typed(DataType.FLOAT32, 2, 3),
                viewTyped(DataType.FLOAT32, new long[] {3}, new long[] {1}, 3L),
                typed(DataType.FLOAT32, 2, 5),
                viewTyped(DataType.FLOAT32, new long[] {2, 2}, new long[] {3, 1}, 1L),
                typed(DataType.FLOAT32, 4, 3),
                typed(DataType.FLOAT32, 2, 2, 3),
                typed(DataType.FLOAT32, 2, 6),
                typed(DataType.FLOAT32, 1, 1, 2, 2),
                typed(DataType.FLOAT32, 1, 4, 1),
                typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                typed(DataType.FLOAT32, 1, 8, 1),
                typed(DataType.INT32, 3),
                typed(DataType.INT64, 3));
        int[] targets = {2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16};
        List<int[]> actual = executeCustom(
                library,
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0, 1, 3, 11, 13, 15},
                targets,
                List.of(
                        data,
                        new int[] {2, 0, 1, 1},
                        new int[] {1, 0, 0, 0},
                        new int[] {0x8000_0000, 0x7fc1_2345, 0x0000_0001, 0x3f80_0000},
                        new int[] {
                            0x8000_0000, 0x7fc1_2345, 0x0000_0001, 0x3f80_0000,
                            0xffc5_4321, 0x8000_0001, 0x7f80_0000, 0xff80_0000
                        },
                        new int[] {-1, Integer.MIN_VALUE, 7}));
        assertArrayEquals(new int[] {
            data[2], data[0], data[4], data[4]
        }, actual.get(0));
        assertArrayEquals(new int[] {
            data[3], data[4], data[5], data[0], data[1], data[2]
        }, actual.get(1));
        assertArrayEquals(
                new int[] {0, 0, 0, data[3], data[4], data[5]},
                actual.get(2));
        assertArrayEquals(new int[] {
            fill, data[0], data[1], data[2], fill,
            fill, data[3], data[4], data[5], fill
        }, actual.get(3));
        assertArrayEquals(
                new int[] {0, data[1], data[2], 0, data[4], data[5]},
                actual.get(4));
        assertArrayEquals(concat(data, data), actual.get(5));
        assertArrayEquals(concat(data, data), actual.get(6));
        assertArrayEquals(new int[] {
            data[0], data[1], data[2], data[0], data[1], data[2],
            data[3], data[4], data[5], data[3], data[4], data[5]
        }, actual.get(7));
        assertArrayEquals(
                new int[] {0x8000_0000, 0x7fc1_2345, 0x0000_0001, 0x3f80_0000},
                actual.get(8));
        assertArrayEquals(new int[] {
            0x8000_0000, 0x7fc1_2345, 0x0000_0001, 0x3f80_0000,
            0xffc5_4321, 0x8000_0001, 0x7f80_0000, 0xff80_0000
        }, actual.get(9));
        assertArrayEquals(new int[] {
            -1, -1, Integer.MIN_VALUE, -1, 7, 0
        }, actual.get(10));
    }

    @Test
    void task0060CustomIntegerProductCompilesAndExecutesUnderBothProfiles() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.FULL,
                        List.of(),
                        false)));
        List<MetalMpsGraphProgram.ValueDescriptor> values =
                List.of(typed(DataType.INT32, 2, 3), typed(DataType.INT32));
        for (NumericalProfile profile : NumericalProfile.values()) {
            List<byte[]> actual = executeCustomBytes(
                    library,
                    profile,
                    program,
                    values,
                    new int[] {0},
                    new int[] {1},
                    List.of(bytes32(Integer.MAX_VALUE, 2, -1, 3, 5, 7)));
            assertArrayEquals(bytes32(210), actual.getFirst());
        }
    }


    @Test
    void task0060ReplacementPreservesEveryCarrierWord() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SCATTER_ND,
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.SCATTER_ND,
                        new long[] {0, 1}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SLICE_UPDATE,
                        new int[] {4, 5},
                        new int[] {6},
                        MetalMpsGraphProgram.AttributeKind.SLICE,
                        new long[] {1, 3, 2, 0, -2}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SLICE_UPDATE,
                        new int[] {4, 5},
                        new int[] {7},
                        MetalMpsGraphProgram.AttributeKind.CROP_TO_SHAPE,
                        new long[] {1, 2, 1, 1})));
        for (DataType carrier : DataType.values()) {
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(carrier, 4),
                    typed(DataType.INT32, 2, 1),
                    typed(carrier, 2),
                    typed(carrier, 4),
                    typed(carrier, 4),
                    typed(carrier, 2),
                    typed(carrier, 4),
                    typed(carrier, 4));
            byte[] data = sequentialCarrierWords(carrier, 4, 1);
            byte[] updates = sequentialCarrierWords(carrier, 2, 41);
            byte[] scatterExpected = data.clone();
            copyCarrierElement(updates, 0, scatterExpected, 3, carrier.byteWidth());
            copyCarrierElement(updates, 1, scatterExpected, 1, carrier.byteWidth());
            byte[] sliceExpected = scatterExpected.clone();
            byte[] cropExpected = data.clone();
            copyCarrierElement(updates, 0, cropExpected, 1, carrier.byteWidth());
            copyCarrierElement(updates, 1, cropExpected, 2, carrier.byteWidth());
            for (NumericalProfile profile : NumericalProfile.values()) {
                List<byte[]> actual = executeCustomBytes(
                        library,
                        profile,
                        program,
                        values,
                        new int[] {0, 1, 2, 4, 5},
                        new int[] {3, 6, 7},
                        List.of(data, bytes32(3, 1), updates, data, updates));
                assertArrayEquals(scatterExpected, actual.get(0), carrier + " scatter");
                assertArrayEquals(sliceExpected, actual.get(1), carrier + " slice update");
                assertArrayEquals(cropExpected, actual.get(2), carrier + " crop update");
            }
        }
    }

    @Test
    void task0060NonoverlapFoldsPreserveOnlyFloatingCarrierWords() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD_AXIS,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_AXIS,
                        new long[] {1, 2, 3}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD2D,
                        new int[] {2},
                        new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                        new long[] {4, 1, 1, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1}),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD3D,
                        new int[] {4},
                        new int[] {5},
                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                        new long[] {
                            5, 1, 1, 3, 3, 3,
                            2, 2, 2, 2, 2, 2,
                            1, 1, 1, 1, 1, 1, 1
                        })));
        for (DataType carrier :
                List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(carrier, 1, 2, 2),
                    typed(carrier, 1, 5),
                    typed(carrier, 1, 4, 9),
                    typed(carrier, 1, 1, 3, 3),
                    typed(carrier, 1, 8, 27),
                    typed(carrier, 1, 1, 3, 3, 3));
            byte[] foldAxisInput = sequentialCarrierWords(carrier, 4, 81);
            byte[] fold2dInput = sequentialCarrierWords(carrier, 36, 101);
            byte[] fold3dInput = sequentialCarrierWords(carrier, 216, 151);
            byte[] foldAxisExpected = new byte[5 * carrier.byteWidth()];
            copyCarrierElement(foldAxisInput, 0, foldAxisExpected, 0, carrier.byteWidth());
            copyCarrierElement(foldAxisInput, 1, foldAxisExpected, 1, carrier.byteWidth());
            copyCarrierElement(foldAxisInput, 2, foldAxisExpected, 3, carrier.byteWidth());
            copyCarrierElement(foldAxisInput, 3, foldAxisExpected, 4, carrier.byteWidth());
            int[] fold2dSources = {27, 19, 28, 12, 4, 13, 30, 22, 31};
            int[] fold3dSources = new int[27];
            int output = 0;
            for (int depth = 0; depth < 3; depth++) {
                for (int row = 0; row < 3; row++) {
                    for (int column = 0; column < 3; column++) {
                        int shiftedDepth = depth + 1;
                        int shiftedRow = row + 1;
                        int shiftedColumn = column + 1;
                        int position = ((shiftedDepth / 2) * 3 + shiftedRow / 2) * 3
                                + shiftedColumn / 2;
                        int element = ((shiftedDepth % 2) * 2 + shiftedRow % 2) * 2
                                + shiftedColumn % 2;
                        fold3dSources[output++] = element * 27 + position;
                    }
                }
            }
            for (NumericalProfile profile : NumericalProfile.values()) {
                List<byte[]> actual = executeCustomBytes(
                        library,
                        profile,
                        program,
                        values,
                        new int[] {0, 2, 4},
                        new int[] {1, 3, 5},
                        List.of(foldAxisInput, fold2dInput, fold3dInput));
                assertArrayEquals(foldAxisExpected, actual.get(0), carrier + " fold axis");
                assertArrayEquals(
                        selectCarrierElements(
                                fold2dInput, carrier.byteWidth(), fold2dSources),
                        actual.get(1),
                        carrier + " fold2d");
                assertArrayEquals(
                        selectCarrierElements(
                                fold3dInput, carrier.byteWidth(), fold3dSources),
                        actual.get(2),
                        carrier + " fold3d");
            }
        }
    }

    @Test
    void task0066CustomFoldImagesPreserveIntegralAndRejectBoolCarrierWords() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD_AXIS,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_AXIS,
                        new long[] {1, 2, 3})));
        for (DataType carrier : List.of(DataType.INT32, DataType.INT64)) {
            List<MetalMpsGraphProgram.ValueDescriptor> values =
                    List.of(typed(carrier, 1, 2, 2), typed(carrier, 1, 5));
      byte[] input = sequentialCarrierWords(carrier, 4, 1);
      byte[] expected = new byte[5 * carrier.byteWidth
                    ()];
      copyCarrierElement(input, 0, expected, 0, carrier.byteWidth());
      copyCarrierElement(input, 1, expected, 1, carrier.byteWidth());
      copyCarrierElement(input, 2, expected, 3, carrier.byteWidth());
      copyCarrierElement(input, 3, expected, 4, carrier.byteWidth());
      List<byte[]> actual = executeCustomBytes(
                            library,
                            NumericalProfile.STRICT_IEEE,
                            program,
                            values,
                            new int[] {0},
                            new int[] {1},
                            List.of(input));
      assertArrayEquals(expected, actual.getFirst(),
                    carrier.toString());
    }
    List<MetalMpsGraphProgram.ValueDescriptor> boolValues =
        List.of(typed(DataType.BOOL, 1, 2, 2), typed(DataType.BOOL, 1, 5));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            executeCustomBytes(
                library,
                NumericalProfile.STRICT_IEEE,
                program,
                boolValues,
                new int[] {0
        },
                new int[] {1},
                List.of(sequentialCarrierWords(DataType.BOOL, 4, 1))));
    }

    @Test
    void task0060AggregateKernelsUseModularAndLogicalExactSemantics() {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.FULL,
                        List.of(),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        2,
                        3,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ALL,
                        4,
                        5,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ANY,
                        4,
                        6,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(0),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ANY,
                        7,
                        8,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        List.of(),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        9,
                        10,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        List.of(),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        0,
                        11,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        List.of(0, 1),
                        true),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ALL,
                        4,
                        12,
                        MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                        List.of(0, 1),
                        true)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                typed(DataType.INT32, 2, 3),
                typed(DataType.INT32),
                typed(DataType.INT64, 2, 2),
                typed(DataType.INT64, 2),
                typed(DataType.BOOL, 2, 3),
                typed(DataType.BOOL, 2),
                typed(DataType.BOOL, 3),
                typed(DataType.BOOL),
                typed(DataType.BOOL),
                typed(DataType.INT32),
                typed(DataType.INT32),
                typed(DataType.INT32, 1, 1),
                typed(DataType.BOOL, 1, 1));
        List<byte[]> feeds = List.of(
                bytes32(Integer.MAX_VALUE, 2, -1, 3, 5, 7),
                bytes64(Long.MAX_VALUE, 2L, -3L, 5L),
                new byte[] {1, 1, 0, 0, 0, 1},
                new byte[] {1},
                bytes32(0x8000_0001));
        List<byte[]> expected = List.of(
                bytes32(210),
                bytes64(-2L, -15L),
                new byte[] {0, 0},
                new byte[] {1, 1, 1},
                new byte[] {1},
                bytes32(0x8000_0001),
                bytes32(210),
                new byte[] {0});
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (int repetition = 0; repetition < 3; repetition++) {
                List<byte[]> actual = executeCustomBytes(
                        library,
                        profile,
                        program,
                        values,
                        new int[] {0, 2, 4, 7, 9},
                        new int[] {1, 3, 5, 6, 8, 10, 11, 12},
                        feeds);
                for (int target = 0; target < expected.size(); target++) {
                    assertArrayEquals(
                            expected.get(target),
                            actual.get(target),
                            profile + " repetition " + repetition + " target " + target);
                }
            }
        }
    }

    @Test
    void task0059CopyRecipesPreserveEveryCarrierWidthUnderBothProfiles() {
        Path library = configuredLibrary();
        for (DataType carrier : DataType.values()) {
            byte[] source = task0059CarrierWords(carrier);
            long fill = task0059FillWord(carrier);
            int width = carrier.byteWidth();
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                            new int[] {0, 1}, new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {1}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.GATHER_ND,
                            new int[] {0, 3}, new int[] {4},
                            MetalMpsGraphProgram.AttributeKind.GATHER_ND, new long[] {0}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.PAD,
                            new int[] {0}, new int[] {5},
                            MetalMpsGraphProgram.AttributeKind.PAD,
                            new long[] {
                                2, 0, 1, 0, 1,
                                MetalMpsGraphProgram.dataTypeWire(carrier), fill
                            }),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.CONCAT,
                            new int[] {0, 0}, new int[] {6},
                            MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.STACK,
                            new int[] {0, 0}, new int[] {7},
                            MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.TILE,
                            new int[] {0}, new int[] {8},
                            MetalMpsGraphProgram.AttributeKind.TILE, new long[] {2, 1, 2})));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(carrier, 2, 3),
                    typed(DataType.INT32, 2, 2),
                    typed(carrier, 2, 2),
                    typed(DataType.INT64, 2, 1),
                    typed(carrier, 2, 3),
                    typed(carrier, 2, 5),
                    typed(carrier, 4, 3),
                    typed(carrier, 2, 2, 3),
                    typed(carrier, 2, 6));
            byte[] sourceAndFill = concatBytes(
                    source, task0059ScalarBytes(fill, width));
            List<byte[]> expected = List.of(
                    selectElements(source, width, 2, 0, 4, 4),
                    selectElements(source, width, 3, 4, 5, 0, 1, 2),
                    selectElements(sourceAndFill, width, 6, 0, 1, 2, 6, 6, 3, 4, 5, 6),
                    concatBytes(source, source),
                    concatBytes(source, source),
                    selectElements(source, width, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5));
            for (NumericalProfile profile : NumericalProfile.values()) {
                List<byte[]> actual = executeCustomBytes(
                        library,
                        profile,
                        program,
                        values,
                        new int[] {0, 1, 3},
                        new int[] {2, 4, 5, 6, 7, 8},
                        List.of(
                                source,
                                bytes32(2, 0, 1, 1),
                                bytes64(1, 0)));
                assertEquals(expected.size(), actual.size());
                for (int index = 0; index < expected.size(); index++) {
                    assertArrayEquals(
                            expected.get(index),
                            actual.get(index),
                            carrier + " " + profile + " target " + index);
                }
            }
        }
    }

    @Test
    void task0066CastsAllThirtySixCarrierPairsAgainstIndependentExactValueOracle() {
        Path library = configuredLibrary();
        DataType[] carriers = {
            DataType.FLOAT64,
            DataType.FLOAT32,
            DataType.BFLOAT16,
            DataType.INT64,
            DataType.INT32,
            DataType.BOOL
        };
        List<byte[]> feeds = List.of(
                bytes64(
                Double.doubleToRawLongBits(0.0),
                Double.doubleToRawLongBits(-1.0),
                        Double.doubleToRawLongBits(1.0),
                Double.doubleToRawLongBits(2.0)),
                bytes32(
                Float.floatToRawIntBits(0.0f),
                        Float.floatToRawIntBits(-1.0f),
                Float.floatToRawIntBits(1.0f),
                Float.floatToRawIntBits(2.0f)),
                bytes16(0x0000, 0xbf80, 0x3f80, 0x4000),
                bytes64(0L, -1L, 1L, 2L),
                bytes32(0, -1, 1, 2),
                new byte[] {0, 1, 1, 0});
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (DataType carrier : carriers) values.add(typed(carrier, 4));
        var targets = new ArrayList<Integer>();
        var expected = new ArrayList<byte[]>();

        for (int source = 0; source < carriers.length; source++) {
      for(DataType target : carriers) {
        int output =values.size();
        targets.add(output);
            values.add(typed(target, 4));
            nodes.add(castNode(source, output, target));
            expected.add(simpleCastExpected(carriers[source], target));
        }
            }

        List<byte[]> actual = executeCustomBytes(
                library,
                NumericalProfile.STRICT_IEEE,
                new MetalMpsGraphProgram(nodes),
                values,
                new int[] {0, 1, 2, 3, 4, 5},
                targets.stream().mapToInt(Integer::intValue).toArray(),
                feeds);
        assertEquals(36, actual.size());
        for (int index = 0; index < expected.size(); index++) {
            assertArrayEquals(expected.get(index), actual.get(index), "cast pair " + index);
        }
    }

    @Test
    void task0066CastSpecialValuesUseExactIndependentBitSemantics() {
    Path library = configuredLibrary();
    long[] doubles = {
      0x0000_0000_0000_0000L,
      0x8000_0000_0000_0000L,
      0x0000_0000_0000_0001L,
      Double.doubleToRawLongBits(1.0 + Math.scalb(1.0, -24)),
      Double.doubleToRawLongBits(1.0 + 3.0 * Math.scalb(1.0, -24)),
      0x7ff0_0000_0000_0000L,
      0xfff0_0000_0000_0000L,
      0x7ff8_0000_0000_1234L,
      0xfff8_0000_0000_5678L,
      Double.doubleToRawLongBits(1.75),
      Double.doubleToRawLongBits(-1.75),
      Double.doubleToRawLongBits(1.0e300)
    };
    assertArrayEquals(
        bytes64(doubles),
        executeCast(library, DataType.FLOAT64, DataType.FLOAT64, bytes64(doubles)));
    assertArrayEquals(
        bytes32(
            0x0000_0000,
            0x8000_0000,
            0x0000_0000,
            0x3f80_0000,
            0x3f80_0002,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7fc0_0000,
            0x3fe0_0000,
            0xbfe0_0000,
            0x7f80_0000),
        executeCast(library, DataType.FLOAT64, DataType.FLOAT32, bytes64(doubles)));
    assertArrayEquals(
        bytes64(
            0L, 0L, 0L, 1L, 1L, Long.MAX_VALUE, Long.MIN_VALUE, 0L, 0L, 1L, -1L, Long.MAX_VALUE),
        executeCast(library, DataType.FLOAT64, DataType.INT64, bytes64(doubles)));
    assertArrayEquals(
        bytes32(
            0, 0, 0, 1, 1, Integer.MAX_VALUE, Integer.MIN_VALUE, 0, 0, 1, -1, Integer.MAX_VALUE),
        executeCast(library, DataType.FLOAT64, DataType.INT32, bytes64(doubles)));
    assertArrayEquals(
        new byte[] {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        executeCast(library, DataType.FLOAT64, DataType.BOOL, bytes64(doubles)));

    byte[] directBfloatDoubles =
        bytes64(
            Double.doubleToRawLongBits(0.0),
            Double.doubleToRawLongBits(-0.0),
            Double.doubleToRawLongBits(Double.MIN_VALUE),
            Double.doubleToRawLongBits(1.0 + Math.scalb(1.0, -8)),
            Double.doubleToRawLongBits(1.0 + 3.0 * Math.scalb(1.0, -8)),
            0x7ff0_0000_0000_0000L,
            0xfff0_0000_0000_0000L,
            0x7ff8_0000_0000_1234L,
            0xfff8_0000_0000_5678L);
    assertArrayEquals(
        bytes16(0x0000, 0x8000, 0x0000, 0x3f80, 0x3f82, 0x7f80, 0xff80, 0x7fc0, 0x7fc0),
        executeCast(library, DataType.FLOAT64, DataType.BFLOAT16, directBfloatDoubles));
    byte[] floatSubnormalBoundaries =
        bytes64(
            Double.doubleToRawLongBits(Double.MIN_VALUE),
            Double.doubleToRawLongBits(Math.scalb(1.0, -149)),
            Double.doubleToRawLongBits(Math.scalb(1.0, -150)),
            Double.doubleToRawLongBits(3.0 * Math.scalb(1.0, -150)));
    assertArrayEquals(
        bytes32(0x0000_0000, 0x0000_0001, 0x0000_0000, 0x0000_0002),
        executeCast(library, DataType.FLOAT64, DataType.FLOAT32, floatSubnormalBoundaries));
    byte[] floats =
        bytes32(
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x3f80_8000,
            0x3f81_8000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc5_4321,
            0x3fe0_0000,
            0xbfe0_0000);
    assertArrayEquals(
        bytes16(
            0x0000, 0x8000, 0x0000, 0x3f80, 0x3f82, 0x7f80, 0xff80, 0x7fc0, 0x7fc0, 0x3fe0, 0xbfe0),
        executeCast(library, DataType.FLOAT32, DataType.BFLOAT16, floats));
    assertArrayEquals(
        new byte[] {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        executeCast(library, DataType.FLOAT32, DataType.BOOL, floats));
    assertArrayEquals(
        bytes64(
            0x0000_0000_0000_0000L,
            0x8000_0000_0000_0000L,
            0x36a0_0000_0000_0000L,
            0x3ff0_1000_0000_0000L,
            0x3ff0_3000_0000_0000L,
            0x7ff0_0000_0000_0000L,
            0xfff0_0000_0000_0000L,
            0x7ff8_2468_a000_0000L,
            0xfff8_a864_2000_0000L,
            0x3ffc_0000_0000_0000L,
            0xbffc_0000_0000_0000L),
        executeCast(library, DataType.FLOAT32, DataType.FLOAT64, floats));

    byte[] integers =
        bytes64(
            Long.MIN_VALUE,
            Long.MAX_VALUE,
            0L,
            -1L,
            0x0123_4567_89ab_cdefL,
            0xffff_ffff_0000_0001L);
    assertArrayEquals(
        bytes32(0, -1, 0, -1, 0x89ab_cdef, 1),
        executeCast(library, DataType.INT64, DataType.INT32, integers));
    assertArrayEquals(
        new byte[] {1, 1, 0, 1, 1, 1},
        executeCast(library, DataType.INT64, DataType.BOOL, integers));
    assertArrayEquals(
        bytes64(
            0xc3e0_0000_0000_0000L,
            0x43e0_0000_0000_0000L,
            0x0000_0000_0000_0000L,
            0xbff0_0000_0000_0000L,
            0x4372_3456_789a_bcdfL,
            0xc1ef_ffff_ffe0_0000L),
        executeCast(library, DataType.INT64, DataType.FLOAT64, integers));
    assertArrayEquals(
        bytes32(0xdf00_0000, 0x5f00_0000, 0x0000_0000, 0xbf80_0000, 0x5b91_a2b4, 0xcf80_0000),
        executeCast(library, DataType.INT64, DataType.FLOAT32, integers));
    assertArrayEquals(
        bytes16(0xdf00, 0x5f00, 0x0000, 0xbf80, 0x5b92, 0xcf80),
        executeCast(library, DataType.INT64, DataType.BFLOAT16, integers));

    byte[] ints = bytes32(Integer.MIN_VALUE, Integer.MAX_VALUE, 0, -1, 0x0123_4567);
    assertArrayEquals(
        bytes64(Integer.MIN_VALUE, Integer.MAX_VALUE, 0L, -1L, 0x0123_4567L),
        executeCast(library, DataType.INT32, DataType.INT64, ints));
    assertArrayEquals(
        new byte[] {1, 1, 0, 1, 1}, executeCast(library, DataType.INT32, DataType.BOOL, ints));
    assertArrayEquals(
        bytes32(0xcf00_0000, 0x4f00_0000, 0x0000_0000, 0xbf80_0000, 0x4b91_a2b4),
        executeCast(library, DataType.INT32, DataType.FLOAT32, ints));
    assertArrayEquals(
        bytes16(0xcf00, 0x4f00, 0x0000, 0xbf80, 0x4b92),
        executeCast(library, DataType.INT32, DataType.BFLOAT16, ints));

    byte[] bools = {0, 1, 1, 0};
    assertArrayEquals(
        bytes64(
            0x0000_0000_0000_0000L,
            0x3ff0_0000_0000_0000L,
            0x3ff0_0000_0000_0000L,
            0x0000_0000_0000_0000L),
        executeCast(library, DataType.BOOL, DataType.FLOAT64, bools));
  }

  @Test
  void task0066ExhaustsEveryBfloat16WordAcrossAllDistinctTargetBehaviors() {
        Path library = configuredLibrary();
        int count = 1 << 16;
        byte[] source = new byte[count * Short.BYTES];
        byte[] expectedFloat64 = new byte[count * Long.BYTES];
    byte[] expectedFloat32 = new byte[count * Integer.BYTES];
        byte[] expectedInt64 = new byte[count * Long.BYTES];
    byte[] expectedInt32 = new byte[count * Integer.BYTES];
    byte[] expectedBool = new byte[count];
        ByteBuffer sourceWords = ByteBuffer.wrap(source).order(ByteOrder.nativeOrder());
        ByteBuffer float64Words = ByteBuffer.wrap(expectedFloat64).order(ByteOrder.nativeOrder());
    ByteBuffer float32Words = ByteBuffer.wrap(expectedFloat32).order(ByteOrder.nativeOrder());
    ByteBuffer int64Words = ByteBuffer.wrap(expectedInt64).order(ByteOrder.nativeOrder());
    ByteBuffer int32Words = ByteBuffer.wrap(expectedInt32).order(ByteOrder.nativeOrder());
        for (int word = 0; word < count; word++) {
            sourceWords.putShort((short) word);
      float64Words.putLong(bfloat16ToFloat64Bits(word));
      float32Words.putInt(word << 16);
      int64Words.putLong(bfloat16ToInteger(word, 64));
      int32Words.putInt((int) bfloat16ToInteger(word, 32));
      expectedBool[word] = (byte) ((word & 0x7fff) == 0 ? 0 : 1);
        }
        var program = new MetalMpsGraphProgram(List.of(
                castNode(0, 1, DataType.BFLOAT16),
                castNode(0, 2, DataType.FLOAT64),
                castNode(0, 3, DataType.FLOAT32),
                castNode(0, 4, DataType.INT64),
                castNode(0, 5, DataType.INT32),
                castNode(0, 6, DataType.BOOL)));
        List<byte[]> actual = executeCustomBytes(
                library,
                NumericalProfile.STRICT_IEEE,
                program,
                List.of(
                        typed(DataType.BFLOAT16, count),
                        typed(DataType.BFLOAT16, count),
                typed(DataType.FLOAT64, count),
                typed(DataType.FLOAT32, count),
                typed(DataType.INT64, count),
                typed(DataType.INT32, count),
                        typed(DataType.BOOL, count)),
                new int[] {0},
                new int[] {1, 2, 3, 4, 5, 6},
                List.of(source));
        assertArrayEquals(source, actual.get(0));
        assertArrayEquals(expectedFloat64, actual.get(1));
    assertArrayEquals(expectedFloat32, actual.get(2));
    assertArrayEquals(expectedInt64, actual.get(3));
    assertArrayEquals(expectedInt32, actual.get(4));
    assertArrayEquals(expectedBool, actual.get(5));
    }

    @Test
    void task0059ChecksIntegerCastBoundariesAndDeterministicFullWordCorpus() {
        Path library = configuredLibrary();
        int count = 4096;
        byte[] ints = new byte[count * Integer.BYTES];
        byte[] longs = new byte[count * Long.BYTES];
        byte[] widened = new byte[count * Long.BYTES];
        byte[] narrowed = new byte[count * Integer.BYTES];
        byte[] intBool = new byte[count];
        byte[] longBool = new byte[count];
        ByteBuffer intInput = ByteBuffer.wrap(ints).order(ByteOrder.nativeOrder());
        ByteBuffer longInput = ByteBuffer.wrap(longs).order(ByteOrder.nativeOrder());
        ByteBuffer expectedWidened = ByteBuffer.wrap(widened).order(ByteOrder.nativeOrder());
        ByteBuffer expectedNarrowed = ByteBuffer.wrap(narrowed).order(ByteOrder.nativeOrder());
        int intWord = 0x3141_5926;
        long longWord = 0x2718_2818_2845_9045L;
        for (int index = 0; index < count; index++) {
            intWord = intWord * 1_664_525 + 1_013_904_223;
            longWord = longWord * 6_364_136_223_846_793_005L
                    + 1_442_695_040_888_963_407L;
            int currentInt = switch (index) {
                case 0 -> 0;
                case 1 -> Integer.MIN_VALUE;
                case 2 -> Integer.MAX_VALUE;
                case 3 -> -1;
                default -> intWord;
            };
            long currentLong = switch (index) {
                case 0 -> 0L;
                case 1 -> Long.MIN_VALUE;
                case 2 -> Long.MAX_VALUE;
                case 3 -> -1L;
                default -> longWord;
            };
            intInput.putInt(currentInt);
            longInput.putLong(currentLong);
            expectedWidened.putLong(currentInt);
            expectedNarrowed.putInt((int) currentLong);
            intBool[index] = (byte) (currentInt == 0 ? 0 : 1);
            longBool[index] = (byte) (currentLong == 0L ? 0 : 1);
        }
        List<byte[]> actual = executeCustomBytes(
                library,
                NumericalProfile.STRICT_IEEE,
                new MetalMpsGraphProgram(List.of(
                        castNode(0, 2, DataType.INT64),
                        castNode(1, 3, DataType.INT32),
                        castNode(0, 4, DataType.BOOL),
                        castNode(1, 5, DataType.BOOL))),
                List.of(
                        typed(DataType.INT32, count),
                        typed(DataType.INT64, count),
                        typed(DataType.INT64, count),
                        typed(DataType.INT32, count),
                        typed(DataType.BOOL, count),
                        typed(DataType.BOOL, count)),
                new int[] {0, 1},
                new int[] {2, 3, 4, 5},
                List.of(ints, longs));
        assertArrayEquals(widened, actual.get(0));
        assertArrayEquals(narrowed, actual.get(1));
        assertArrayEquals(intBool, actual.get(2));
        assertArrayEquals(longBool, actual.get(3));
    }

    @Test
    void task0066SelectedStructuralRecipesRunCustomAndRemainingRecipesRunDirect() {
        Path library = configuredLibrary();
        int[] data = bits(1, 2, 3, 4, 5, 6);
        int[] image2d = bits(1, 2, 3, 4);
        int[] image3d = bits(1, 2, 3, 4, 5, 6, 7, 8);

        assertArrayEquals(
                new int[] {-1, -1, 7, 0},
                NonProductionStructuralFixture.executeCurrent(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        MetalMpsGraphProgram.Node.generic(
                                MetalMpsGraphProgram.NodeKind.CAST,
                                new int[] {0}, new int[] {1},
                                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                                new long[] {6}),
                        List.of(typed(DataType.INT32, 2), typed(DataType.INT64, 2)),
                        List.of(new int[] {-1, 7})));
        assertEquals(4, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.GATHER_ELEMENTS,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {1}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.INT32, 2, 2),
                        typed(DataType.FLOAT32, 2, 2)),
                List.of(data, new int[] {2, 0, 1, 1})).length);
        assertEquals(6, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.GATHER_ND,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.GATHER_ND, new long[] {0}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.INT32, 2, 1),
                        typed(DataType.FLOAT32, 2, 3)),
                List.of(data, new int[] {1, 0})).length);
        assertEquals(6, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SCATTER_ND,
                        new int[] {0, 1, 2}, new int[] {3},
                        MetalMpsGraphProgram.AttributeKind.SCATTER_ND,
                        new long[] {0, 1}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.INT32, 2, 1),
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 2, 3)),
                List.of(data, new int[] {1, 0}, bits(10, 20, 30, 40, 50, 60))).length);
        assertEquals(10, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.PAD,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.PAD,
                        new long[] {2, 0, 1, 0, 1, 1, 0}),
                List.of(typed(DataType.FLOAT32, 2, 3), typed(DataType.FLOAT32, 2, 5)),
                List.of(data)).length);
        assertEquals(6, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.SLICE_UPDATE,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.SLICE,
                        new long[] {1, 1, 2, 1, 1}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 2, 2),
                        typed(DataType.FLOAT32, 2, 3)),
                List.of(data, bits(10, 20, 30, 40))).length);
        assertEquals(12, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.CONCAT,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 4, 3)),
                List.of(data, data)).length);
        assertEquals(12, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.STACK,
                        new int[] {0, 1}, new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {0}),
                List.of(
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 2, 3),
                        typed(DataType.FLOAT32, 2, 2, 3)),
                List.of(data, data)).length);
        assertEquals(12, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.TILE,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.TILE, new long[] {2, 1, 2}),
                List.of(typed(DataType.FLOAT32, 2, 3), typed(DataType.FLOAT32, 2, 6)),
                List.of(data)).length);
        assertEquals(
        4, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD_AXIS,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_AXIS,
                        new long[] {1, 2, 2}),
                List.of(typed(DataType.FLOAT32, 1, 2, 2), typed(DataType.FLOAT32, 1, 4)),
                List.of(bits(1, 2, 3, 4))).length);
        long[] window2d = {2, 2, 1, 1, 0, 0, 1, 1, 0};
        assertEquals(4, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D, window2d),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 2),
                        typed(DataType.FLOAT32, 1, 4, 1)),
                List.of(image2d)).length);
        assertEquals(4, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD2D,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                        new long[] {4, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 0}),
                List.of(
                        typed(DataType.FLOAT32, 1, 4, 1),
                        typed(DataType.FLOAT32, 1, 1, 2, 2)),
                List.of(image2d)).length);
        long[] window3d = {2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};
        assertEquals(8, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.UNFOLD3D,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.WINDOW_3D, window3d),
                List.of(
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2),
                        typed(DataType.FLOAT32, 1, 8, 1)),
                List.of(image3d)).length);
        assertEquals(8, NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.STRICT_IEEE,
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FOLD3D,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                        new long[] {
                            5, 1, 1, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 0
                        }),
                List.of(
                        typed(DataType.FLOAT32, 1, 8, 1),
                        typed(DataType.FLOAT32, 1, 1, 2, 2, 2)),
                List.of(image3d)).length);
    }

    @Test
    void task0060EveryAggregateStructuralRecipeCreatesRunsAndReturnsExpectedValues() {
        Path library = configuredLibrary();
        var exactProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.PROD,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ALL,
                        2,
                        3,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.ANY,
                        2,
                        4,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(0),
                        false)));
        List<MetalMpsGraphProgram.ValueDescriptor> exactValues = List.of(
                typed(DataType.INT32, 2, 2),
                typed(DataType.INT32, 2),
                typed(DataType.BOOL, 2, 2),
                typed(DataType.BOOL, 2),
                typed(DataType.BOOL, 2));
        for (NumericalProfile profile : NumericalProfile.values()) {
            List<byte[]> exact = NonProductionStructuralFixture.executeDirectBytes(
                    library,
                    profile,
                    exactProgram,
                    exactValues,
                    new int[] {0, 2},
                    new int[] {1, 3, 4},
                    List.of(bytes32(2, 3, -1, 5), new byte[] {1, 1, 0, 1}));
            assertArrayEquals(bytes32(6, -5), exact.get(0));
            assertArrayEquals(new byte[] {1, 0}, exact.get(1));
            assertArrayEquals(new byte[] {1, 1}, exact.get(2));
        }

        int[] input = bits(1.0f, 2.0f, 3.0f, 4.0f);
        List<MetalMpsGraphProgram.ValueDescriptor> values =
                List.of(typed(DataType.FLOAT32, 2, 2), typed(DataType.FLOAT32, 2));
        int[] logSumExp = NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.ACCELERATOR,
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                values,
                List.of(input));
        assertFloatWord(
                (float) Math.log(Math.exp(1.0) + Math.exp(2.0)), logSumExp[0], 0.00001f);
        assertFloatWord(
                (float) Math.log(Math.exp(3.0) + Math.exp(4.0)), logSumExp[1], 0.00001f);

        int[] variance = NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.ACCELERATOR,
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.VARIANCE,
                        0,
                        1,
                        List.of(1),
                        false,
                        1L),
                values,
                List.of(input));
        assertFloatWord(0.5f, variance[0], 0.00001f);
        assertFloatWord(0.5f, variance[1], 0.00001f);

        int[] deviation = NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.ACCELERATOR,
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.STANDARD_DEVIATION,
                        0,
                        1,
                        List.of(1),
                        false,
                        0L),
                values,
                List.of(input));
        assertFloatWord(0.5f, deviation[0], 0.00001f);
        assertFloatWord(0.5f, deviation[1], 0.00001f);

        int[] l2 = NonProductionStructuralFixture.executeCurrent(
                library,
                NumericalProfile.ACCELERATOR,
                MetalMpsGraphProgram.Node.reduction(
                        MetalMpsGraphProgram.NodeKind.L2_NORM,
                        0,
                        1,
                        MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                        List.of(1),
                        false),
                values,
                List.of(input));
        assertFloatWord((float) Math.sqrt(5.0), l2[0], 0.00001f);
        assertFloatWord(5.0f, l2[1], 0.00001f);

        assertArrayEquals(
                input,
                NonProductionStructuralFixture.executeCurrent(
                        library,
                        NumericalProfile.ACCELERATOR,
                        MetalMpsGraphProgram.Node.reduction(
                                MetalMpsGraphProgram.NodeKind.LOG_SUM_EXP,
                                0,
                                1,
                                MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                                List.of(),
                                false),
                        List.of(
                                typed(DataType.FLOAT32, 2, 2),
                                typed(DataType.FLOAT32, 2, 2)),
                        List.of(input)));
    }

    @Test
    void task0059RankSixteenAndPositiveExtremeSliceRemainExecutable() {
        Path library = configuredLibrary();

        long[] singletonShape = new long[16];
        Arrays.fill(singletonShape, 1L);
        long[] rankSixteenSlice = new long[65];
        rankSixteenSlice[0] = 16L;
        for (int axis = 0; axis < 16; axis++) {
            rankSixteenSlice[1 + axis] = 0L;
            rankSixteenSlice[17 + axis] = 1L;
            rankSixteenSlice[33 + axis] = axis;
            rankSixteenSlice[49 + axis] = 1L;
        }
        var fullRankSlice = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SLICE,
                rankSixteenSlice);
        long[] singletonStrides = new long[16];
        Arrays.fill(singletonStrides, 1L);
        List<MetalMpsGraphProgram.ValueDescriptor> singletonValues = List.of(
                typed(DataType.FLOAT32, singletonShape),
                viewTyped(DataType.FLOAT32, singletonShape, singletonStrides, 0L));
        assertArrayEquals(
                bits(7),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(fullRankSlice)),
                        singletonValues,
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(7))).getFirst());

        long[] positiveExtreme = {1, 1, 1, 0, Long.MAX_VALUE};
        var positiveExtremeSlice = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SLICE,
                positiveExtreme);
        List<MetalMpsGraphProgram.ValueDescriptor> positiveExtremeValues = List.of(
                typed(DataType.FLOAT32, 2),
                viewTyped(
                        DataType.FLOAT32,
                        new long[] {1},
                        new long[] {Long.MAX_VALUE},
                        1L));
        assertArrayEquals(
                bits(0, 13),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(positiveExtremeSlice)),
                        positiveExtremeValues,
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(11, 13))).getFirst());
        long[] negativeExtreme = {1, 0, 1, 0, Long.MIN_VALUE + 1L};
        var negativeExtremeSlice = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SLICE,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SLICE,
                negativeExtreme);
        assertThrows(
                IllegalArgumentException.class,
                () -> executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(negativeExtremeSlice)),
                        List.of(
                                typed(DataType.FLOAT32, 2),
                                viewTyped(
                                        DataType.FLOAT32,
                                        new long[] {1},
                                        new long[] {1},
                                        0L)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(11, 13))));
    }

    @Test
    void task0060WindowKernelsHonorCeilDepthMaskingAndWideGeometry() {
        Path library = configuredLibrary();
        long wide = 1L << 32;
        assertArrayEquals(
                bits(1, 0, 0, 0),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                                        new long[] {1, 1, 3, 3, 0, 0, 1, 1, 1}))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 1, 2, 2),
                                typed(DataType.FLOAT32, 1, 1, 4)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(1, 2, 3, 4))).getFirst());
        assertArrayEquals(
                bits(0, 7, 0, 0),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                                        new long[] {
                                            1, 2, 1, wide, 0, wide, 1, 1, 0
                                        }))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 1, 1, 1),
                                typed(DataType.FLOAT32, 1, 2, 2)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(7))).getFirst());
        assertArrayEquals(
                bits(7),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.FOLD2D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                                        new long[] {
                                            4, 1, 1, 1, 1,
                                            1, 2, 1, wide, 0, wide, 1, 1, 0
                                        }))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 2, 2),
                                typed(DataType.FLOAT32, 1, 1, 1, 1)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(0, 7, 0, 0))).getFirst());
        assertArrayEquals(
                bits(1, 0, 0, 0),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.FOLD2D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                                        new long[] {
                                            4, 1, 1, 2, 2,
                                            1, 1, 3, 3, 0, 0, 1, 1, 1
                                        }))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 1, 4),
                                typed(DataType.FLOAT32, 1, 1, 2, 2)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(1, 2, 3, 4))).getFirst());
        assertArrayEquals(
                bits(7),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.FOLD3D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                                        new long[] {
                                            5, 1, 1, 1, 1, 1,
                                            1, 1, 1, 1, 1, 1,
                                            1, 0, 0, 1, 1, 1, 0
                                        }))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 1, 3),
                                typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(5, 7, 11))).getFirst());
        assertArrayEquals(
                bits(2),
                executeCustom(
                        library,
                        NumericalProfile.STRICT_IEEE,
                        new MetalMpsGraphProgram(List.of(
                                MetalMpsGraphProgram.Node.generic(
                                        MetalMpsGraphProgram.NodeKind.FOLD3D,
                                        new int[] {0},
                                        new int[] {1},
                                        MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                                        new long[] {
                                            5, 1, 1, 1, 1, 1,
                                            2, 1, 1, wide, 1, 1,
                                            wide, 0, 0, 1, 1, 1, 0
                                        }))),
                        List.of(
                                typed(DataType.FLOAT32, 1, 2, 2),
                                typed(DataType.FLOAT32, 1, 1, 1, 1, 1)),
                        new int[] {0},
                        new int[] {1},
                        List.of(bits(1, 2, 3, 4))).getFirst());
    }

    @Test
    void directMpsGraphCandidatesRemainForceableForAllFourExactRawOperations() {
        Path library = configuredLibrary();
        int[] ordinary = bits(-2.5f, -0.0f, 0.0f, 1.25f, 4.0f);
        for (NumericalProfile profile : NumericalProfile.values()) {
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.FLOOR,
                    MetalMpsGraphProgram.NodeKind.CEIL,
                    MetalMpsGraphProgram.NodeKind.SIGN,
                    MetalMpsGraphProgram.NodeKind.RELU)) {
                int[] output = NonProductionStructuralFixture.executeCurrent(
                        library, profile, unary(kind, 0, 1),
                        List.of(value(ordinary.length), value(ordinary.length)),
                        List.of(ordinary));
                assertEquals(ordinary.length, output.length, kind.name());
            }
        }
    }

    @Test
    void everyRemainingBlockedCatalogRecipeCreatesRunsAndClosesOnlyThroughTheRawFixture() {
        Path library = configuredLibrary();
        int[] input = bits(0.25f, 0.5f, 1.0f, 2.0f);
        int[] exponent = bits(2.0f, 2.0f, 2.0f, 2.0f);
        int outputWords = input.length;

        int[] tensorPow = NonProductionStructuralFixture.executeCurrent(
                library, NumericalProfile.ACCELERATOR,
                binary(MetalMpsGraphProgram.NodeKind.TENSOR_POW, 0, 1, 2),
                List.of(value(outputWords), value(outputWords), value(outputWords)),
                List.of(input, exponent));
        assertEquals(outputWords, tensorPow.length);

        int[] scalarPow = NonProductionStructuralFixture.executeCurrent(
                library, NumericalProfile.ACCELERATOR,
                scalar(
                        MetalMpsGraphProgram.NodeKind.SCALAR_POW,
                        0,
                        1,
                        Float.floatToRawIntBits(2.0f)),
                List.of(value(outputWords), value(outputWords)), List.of(input));
        assertEquals(outputWords, scalarPow.length);

        for (MetalMpsGraphProgram.NodeKind kind : List.of(
                MetalMpsGraphProgram.NodeKind.LOG,
                MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalMpsGraphProgram.NodeKind.EXPM1,
                MetalMpsGraphProgram.NodeKind.ERF,
                MetalMpsGraphProgram.NodeKind.SQRT,
                MetalMpsGraphProgram.NodeKind.RSQRT,
                MetalMpsGraphProgram.NodeKind.TANH,
                MetalMpsGraphProgram.NodeKind.GELU,
                MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION,
                MetalMpsGraphProgram.NodeKind.SILU)) {
            int[] output = NonProductionStructuralFixture.executeCurrent(
                    library, NumericalProfile.ACCELERATOR, unary(kind, 0, 1),
                    List.of(value(outputWords), value(outputWords)), List.of(input));
            assertEquals(outputWords, output.length, kind.name());
        }
    }

    @Test
    void mixedCustomProgramExecutesRawAndNestedMpsGraphNodesWithDirectPublication() {
        Path library = configuredLibrary();
        int[] input = bits(-1.5f, -0.25f, 0.0f, 1.5f);
        var program = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.ABS, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.SIGN, 2, 3)));
        List<int[]> actual = executeCustom(
                library, NumericalProfile.STRICT_IEEE, program,
                List.of(value(4), value(4), value(4), value(4)),
                new int[] {0}, new int[] {1, 2, 3}, List.of(input));
        assertArrayEquals(bits(-2.0f, -1.0f, 0.0f, 1.0f), actual.get(0));
        assertArrayEquals(bits(2.0f, 1.0f, 0.0f, 1.0f), actual.get(1));
        assertArrayEquals(bits(1.0f, 1.0f, 0.0f, 1.0f), actual.get(2));
    }

    @Test
    void scalarWireRawBitsShapeAndTensorPrimitiveAllDriftInCanonicalImageDigests()
            throws NoSuchAlgorithmException {
        Set<String> digests = new HashSet<>();
        for (MetalMpsGraphProgram.NodeKind kind : List.of(
                MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                MetalMpsGraphProgram.NodeKind.SCALAR_SUB,
                MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                MetalMpsGraphProgram.NodeKind.SCALAR_DIV)) {
            digests.add(digestImage(
                    new MetalMpsGraphProgram(List.of(scalar(kind, 0, 1, 0x3f80_0000))),
                    List.of(value(4), value(4)), new int[] {0}, new int[] {1}));
        }
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(scalar(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 0, 1, 0x3f80_0001))),
                List.of(value(4), value(4)), new int[] {0}, new int[] {1}));
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(scalar(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD, 0, 1, 0x3f80_0000))),
                List.of(value(2, 2), value(2, 2)), new int[] {0}, new int[] {1}));
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(unary(
                        MetalMpsGraphProgram.NodeKind.RECIPROCAL, 0, 1))),
                List.of(value(4), value(4)), new int[] {0}, new int[] {1}));
        digests.add(digestImage(
                new MetalMpsGraphProgram(List.of(binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 0, 1, 2))),
                List.of(value(4), value(4), value(4)), new int[] {0, 1}, new int[] {2}));
        assertEquals(8, digests.size());
    }

    private static String digestImage(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) throws NoSuchAlgorithmException {
        try (Arena arena = Arena.ofConfined()) {
            byte[] image = program.encodeNative(
                    arena, NumericalProfile.STRICT_IEEE,
                    values, feeds, targets, MetalPreparedRoute.MPSGRAPH).toArray(JAVA_BYTE);
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(image));
        }
    }

    private static List<int[]> executeCustom(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<int[]> feedWords) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                    profile, program, values, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM, fusion);
            int[] materialized = fusion.materializedProgramValueIndices();
            int[] programToSlot = fusion.programToMaterializedSlot();
            for (int value : materialized) {
                buffers.add(api.createBuffer(context, values.get(value).byteCount()));
            }
            for (int index = 0; index < feeds.length; index++) {
                upload(api, buffers.get(programToSlot[feeds[index]]), feedWords.get(index));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++)
                    inputAddresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++)
                    outputAddresses.setAtIndex(
                            ADDRESS, index,
                            buffers.get(programToSlot[targets[index]]).carrier());
                api.runExecutable(
                        executable, buffers.size(), inputAddresses,
                        targets.length, outputAddresses);
            }
            List<int[]> output = new ArrayList<>(targets.length);
            for (int target : targets)
                output.add(download(
                        api,
                        buffers.get(programToSlot[target]),
                        values.get(target).byteCount()));
            return List.copyOf(output);
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static List<byte[]> executeCustomBytes(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes) {
        return executeBytes(
                library,
                profile,
                program,
                values,
                feeds,
                targets,
                feedBytes,
                MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    private static List<byte[]> executeBytes(
            Path library,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> feedBytes,
            MetalPreparedRoute route) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            MetalPointwiseFusionPlan fusion = route == MetalPreparedRoute.CUSTOM_PROGRAM
                    ? MetalPointwiseFusionPlanner.plan(profile, program, values, feeds, targets, route)
                    : null;
            executable = api.createMpsGraphExecutable(
                    context, profile, values, program, feeds, targets, route, fusion);
            int[] materialized;
            int[] programToSlot;
            if (fusion != null) {
                materialized = fusion.materializedProgramValueIndices();
                programToSlot = fusion.programToMaterializedSlot();
            } else {
                materialized = new int[values.size()];
                programToSlot = new int[values.size()];
                for (int index = 0; index < values.size(); index++) {
                    materialized[index] = index;
                    programToSlot[index] = index;
                }
            }
            for (int value : materialized)
                buffers.add(api.createBuffer(context, values.get(value).byteCount()));
            for (int index = 0; index < feeds.length; index++) {
                assertEquals(values.get(feeds[index]).byteCount(), feedBytes.get(index).length);
                uploadBytes(api, buffers.get(programToSlot[feeds[index]]), feedBytes.get(index));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++)
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++)
                    outputs.setAtIndex(
                            ADDRESS, index,
                            buffers.get(programToSlot[targets[index]]).carrier());
                api.runExecutable(
                        executable, buffers.size(), addresses, targets.length, outputs);
            }
            var result = new ArrayList<byte[]>(targets.length);
            for (int target : targets)
                result.add(downloadBytes(
                        api,
                        buffers.get(programToSlot[target]),
                        values.get(target).byteCount()));
            return List.copyOf(result);
        } finally {
            for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static final class NonProductionStructuralFixture {
        static List<byte[]> executeDirectBytes(
                Path library,
                NumericalProfile profile,
                MetalMpsGraphProgram program,
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                int[] feeds,
                int[] targets,
                List<byte[]> feedBytes) {
            MetalNativeApi api = MetalNativeApi.open(library);
            MetalNativeApi.Handle context = null;
            MetalNativeApi.Handle executable = null;
            var inputs = new ArrayList<MetalNativeApi.Handle>();
            var outputs = new ArrayList<MetalNativeApi.Handle>();
            try {
                context = api.createContext();
                executable = api.createMpsGraphExecutable(
                        context,
                        profile,
                        values,
                        program,
                        feeds,
                        targets,
                        MetalPreparedRoute.MPSGRAPH);
                for (int index = 0; index < feeds.length; index++) {
                    assertEquals(values.get(feeds[index]).byteCount(), feedBytes.get(index).length);
                    MetalNativeApi.Handle input =
                            api.createBuffer(context, feedBytes.get(index).length);
                    inputs.add(input);
                    uploadBytes(api, input, feedBytes.get(index));
                }
                for (int target : targets)
                    outputs.add(api.createBuffer(context, values.get(target).byteCount()));
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                    for (int index = 0; index < inputs.size(); index++)
                        inputAddresses.setAtIndex(
                                ADDRESS, index, inputs.get(index).carrier());
                    MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                    for (int index = 0; index < outputs.size(); index++)
                        outputAddresses.setAtIndex(
                                ADDRESS, index, outputs.get(index).carrier());
                    api.runExecutable(
                            executable,
                            inputs.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                }
                var result = new ArrayList<byte[]>(outputs.size());
                for (int index = 0; index < outputs.size(); index++)
                    result.add(downloadBytes(
                            api, outputs.get(index), values.get(targets[index]).byteCount()));
                return List.copyOf(result);
            } finally {
                for (int index = outputs.size(); index-- > 0;)
                    api.releaseBuffer(outputs.get(index));
                for (int index = inputs.size(); index-- > 0;)
                    api.releaseBuffer(inputs.get(index));
                if (executable != null) api.releaseExecutable(executable);
                if (context != null) api.releaseContext(context);
                api.close();
            }
        }

        private NonProductionStructuralFixture() {}

        static int[] executeCurrent(
                Path library,
                NumericalProfile profile,
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                List<int[]> feedWords) {
            MetalNativeApi api = MetalNativeApi.open(library);
            MetalNativeApi.Handle context = null;
            MetalNativeApi.Handle executable = null;
            var inputs = new ArrayList<MetalNativeApi.Handle>();
            MetalNativeApi.Handle output = null;
            try {
                context = api.createContext();
                var program = new MetalMpsGraphProgram(List.of(node));
                int[] feeds = java.util.stream.IntStream.range(0, feedWords.size()).toArray();
                int target = values.size() - 1;
        MetalPreparedRoute route =
            node.kind().isTask0066Selected()
                ? MetalPreparedRoute.CUSTOM_PROGRAM
                : MetalPreparedRoute.MPSGRAPH;
                executable = api.createMpsGraphExecutable(
                        context, profile, values, program, feeds, new int[] {target}, route);
                for (int[] words : feedWords) {
                    MetalNativeApi.Handle input = api.createBuffer(
                            context, Math.multiplyExact((long) words.length, Integer.BYTES));
                    inputs.add(input);
                    upload(api, input, words);
                }
                output = api.createBuffer(context, values.get(target).byteCount());
                try (Arena arena = Arena.ofConfined()) {
          int inputCount = inputs.size() + (route == MetalPreparedRoute.CUSTOM_PROGRAM ? 1 : 0);
                    MemorySegment inputAddresses = arena.allocate(ADDRESS, inputCount);
                    for (int index = 0; index < inputs.size(); index++)
                        inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
          if (route == MetalPreparedRoute.CUSTOM_PROGRAM) {
            inputAddresses.setAtIndex(ADDRESS, inputs.size(), output.carrier());
          }
                    MemorySegment outputAddresses = arena.allocate(ADDRESS);
                    outputAddresses.set(ADDRESS, 0L, output.carrier());
                    api.runExecutable(
                            executable, inputCount, inputAddresses, 1, outputAddresses);
                }
                return download(api, output, values.get(target).byteCount());
            } finally {
                if (output != null) api.releaseBuffer(output);
                for (int index = inputs.size(); index-- > 0;) api.releaseBuffer(inputs.get(index));
                if (executable != null) api.releaseExecutable(executable);
                if (context != null) api.releaseContext(context);
                api.close();
            }
        }
    }

    private static MetalMpsGraphProgram.Node unary(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {input}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.Node binary(
            MetalMpsGraphProgram.NodeKind kind, int left, int right, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {left, right}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.Node scalar(
            MetalMpsGraphProgram.NodeKind kind, int input, int output, int rawBits) {
        return MetalMpsGraphProgram.Node.generic(
                kind, new int[] {input}, new int[] {output},
                MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE,
                new long[] {1L, Integer.toUnsignedLong(rawBits)});
  }

  private static byte[] executeCast(Path library, DataType source, DataType target, byte[] input) {
    int count = input.length / source.byteWidth();
    var program = new MetalMpsGraphProgram(List.of(castNode(0, 1, target)));
    return executeCustomBytes(
            library,
            NumericalProfile.STRICT_IEEE,
            program,
            List.of(typed(source, count), typed(target, count)),
            new int[] {0},
            new int[] {1},
            List.of(input))
        .getFirst();
    }

    private static MetalMpsGraphProgram.Node castNode(
            int input, int output, DataType target) {
        return MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {input},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {switch (target) {
                    case FLOAT32 -> 1L;
                    case INT32 -> 2L;
                    case BOOL -> 3L;
                    case FLOAT64 -> 4L;
                    case BFLOAT16 -> 5L;
                    case INT64 -> 6L;
                }});
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dimensions, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor viewTyped(
            DataType type, long[] dimensions, long[] strides, long offset) {
        var shape = io.github.pho001.synaptik.model.shape.Shape.of(dimensions);
        return new MetalMpsGraphProgram.ValueDescriptor(
                type,
                dimensions,
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                shape, strides, offset, true)),
                false,
                false);
    }

    private static byte[] task0059CarrierWords(DataType carrier) {
        return switch (carrier) {
            case FLOAT64 -> bytes64(
                    0x8000_0000_0000_0000L,
                    0x7ff8_0000_0000_0042L,
                    0x0000_0000_0000_0001L,
                    0x3ff0_0000_0000_0000L,
                    0xfff8_0000_0000_1234L,
                    0x8000_0000_0000_0001L);
            case FLOAT32 -> bytes32(
                    0x8000_0000, 0x7fc1_2345, 0x0000_0001,
                    0x3f80_0000, 0xffc5_4321, 0x8000_0001);
            case BFLOAT16 -> bytes16(0x8000, 0x7fc1, 0x0001, 0x3f80, 0xffc5, 0x8001);
            case INT64 -> bytes64(
                    Long.MIN_VALUE, Long.MAX_VALUE, -1L, 0L, 1L, 0x0123_4567_89ab_cdefL);
            case INT32 -> bytes32(
                    Integer.MIN_VALUE, Integer.MAX_VALUE, -1, 0, 1, 0x1234_5678);
            case BOOL -> new byte[] {0, 1, 0, 1, 1, 0};
        };
    }

    private static long task0059FillWord(DataType carrier) {
        return switch (carrier) {
            case FLOAT64 -> 0xfff8_0000_0000_5678L;
            case FLOAT32 -> Integer.toUnsignedLong(0xffa2_2222);
            case BFLOAT16 -> 0xffa2L;
            case INT64 -> 0x7654_3210_fedc_ba98L;
            case INT32 -> Integer.toUnsignedLong(0x8765_4321);
            case BOOL -> 1L;
        };
    }

    private static byte[] task0059ScalarBytes(long word, int width) {
        ByteBuffer bytes = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.nativeOrder());
        bytes.putLong(word);
        return Arrays.copyOf(bytes.array(), width);
    }

    private static byte[] selectElements(byte[] source, int width, int... ordinals) {
        byte[] result = new byte[Math.multiplyExact(width, ordinals.length)];
        for (int index = 0; index < ordinals.length; index++) {
            System.arraycopy(
                    source,
                    Math.multiplyExact(ordinals[index], width),
                    result,
                    Math.multiplyExact(index, width),
                    width);
        }
        return result;
    }

    private static byte[] sequentialCarrierWords(
            DataType carrier, int count, int seed) {
        ByteBuffer words = ByteBuffer.allocate(Math.multiplyExact(count, carrier.byteWidth()))
                .order(ByteOrder.nativeOrder());
        for (int index = 0; index < count; index++) {
            int value = seed + index;
            switch (carrier) {
                case FLOAT64, INT64 ->
                    words.putLong(0x0102_0304_0000_0000L | Integer.toUnsignedLong(value));
                case FLOAT32, INT32 -> words.putInt(0x0100_0000 | value);
                case BFLOAT16 -> words.putShort((short) (0x0100 | value));
                case BOOL -> words.put((byte) (value & 1));
            }
        }
        return words.array();
    }

    private static void copyCarrierElement(
            byte[] source, int sourceIndex, byte[] target, int targetIndex, int width) {
        System.arraycopy(source, sourceIndex * width, target, targetIndex * width, width);
    }

    private static byte[] selectCarrierElements(
            byte[] source, int width, int[] indices) {
        byte[] selected = new byte[Math.multiplyExact(width, indices.length)];
        for (int index = 0; index < indices.length; index++) {
            copyCarrierElement(source, indices[index], selected, index, width);
        }
        return selected;
    }

    private static byte[] concatBytes(byte[]... parts) {
        int length = 0;
        for (byte[] part : parts) length = Math.addExact(length, part.length);
        byte[] result = new byte[length];
        int offset = 0;
        for (byte[] part : parts) {
            System.arraycopy(part, 0, result, offset, part.length);
            offset += part.length;
        }
        return result;
    }

    private static long bfloat16ToFloat64Bits(int word) {
    long sign = ((long) word & 0x8000L) << 48;
    int exponent = (word >>> 7) & 0xff;
    int fraction = word & 0x7f;
    if (exponent == 0xff) {
      return sign | 0x7ff0_0000_0000_0000L | ((long) fraction << 45);
    }
    if (exponent == 0) {
      if (fraction == 0) return sign;
      int highest = 31 - Integer.numberOfLeadingZeros(fraction);
      int unbiased = -133 + highest;
      long remainder = fraction - (1L << highest);
      return sign | ((long) (unbiased + 1023) << 52) | (remainder << (52 - highest));
    }
    return sign | ((long) (exponent - 127 + 1023) << 52) | ((long) fraction << 45);
  }

  private static long bfloat16ToInteger(int word, int bits) {
    boolean negative = (word & 0x8000) != 0;
    int exponent = (word >>> 7) & 0xff;
    int fraction = word & 0x7f;
    if (exponent == 0xff) {
      if (fraction != 0) return 0L;
      return negative
          ? bits == 32 ? Integer.MIN_VALUE : Long.MIN_VALUE
          : bits == 32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
    }
    if (exponent == 0) return 0L;
    int unbiased = exponent - 127;
    if (unbiased < 0) return 0L;
    if (unbiased >= bits - 1) {
      return negative
          ? bits == 32 ? Integer.MIN_VALUE : Long.MIN_VALUE
          : bits == 32 ? Integer.MAX_VALUE : Long.MAX_VALUE;
    }
    long mantissa = 128L + fraction;
    long magnitude = unbiased >= 7 ? mantissa << (unbiased - 7) : mantissa >>> (7 - unbiased);
    return negative ? -magnitude : magnitude;
  }

  private static byte[] simpleCastExpected(DataType source, DataType target) {
    boolean boolSource = source == DataType.BOOL;
    return switch (target) {
      case FLOAT64 ->
          boolSource
              ? bytes64(
                  Double.doubleToRawLongBits(0.0),
                  Double.doubleToRawLongBits(1.0),
                  Double.doubleToRawLongBits(1.0),
                  Double.doubleToRawLongBits(0.0))
              : bytes64(
                  Double.doubleToRawLongBits(0.0),
                  Double.doubleToRawLongBits(-1.0),
                  Double.doubleToRawLongBits(1.0),
                  Double.doubleToRawLongBits(2.0));
      case FLOAT32 ->
          boolSource
              ? bytes32(
                  Float.floatToRawIntBits(0.0f),
                  Float.floatToRawIntBits(1.0f),
                  Float.floatToRawIntBits(1.0f),
                  Float.floatToRawIntBits(0.0f))
              : bytes32(
                  Float.floatToRawIntBits(0.0f),
                  Float.floatToRawIntBits(-1.0f),
                  Float.floatToRawIntBits(1.0f),
                  Float.floatToRawIntBits(2.0f));
      case BFLOAT16 ->
          boolSource
              ? bytes16(0x0000, 0x3f80, 0x3f80, 0x0000)
              : bytes16(0x0000, 0xbf80, 0x3f80, 0x4000);
      case INT64 -> boolSource ? bytes64(0L, 1L, 1L, 0L) : bytes64(0L, -1L, 1L, 2L);
      case INT32 -> boolSource ? bytes32(0, 1, 1, 0) : bytes32(0, -1, 1, 2);
      case BOOL -> boolSource ? new byte[] {0, 1, 1, 0} : new byte[] {0, 1, 1, 1};
    };
  }

  private static byte[] bytes64(long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Long.BYTES)
                .order(ByteOrder.nativeOrder());
        for (long value : values) bytes.putLong(value);
        return bytes.array();
    }

    private static byte[] bytes32(int... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putInt(value);
        return bytes.array();
    }

    private static byte[] bytes16(int... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (int value : values) bytes.putShort((short) value);
        return bytes.array();
    }

    private static int[] concat(int[] left, int[] right) {
        int[] result = Arrays.copyOf(left, left.length + right.length);
        System.arraycopy(right, 0, result, left.length, right.length);
        return result;
    }

    private static int[] bits(float... values) {
        int[] bits = new int[values.length];
        for (int index = 0; index < values.length; index++)
            bits[index] = Float.floatToRawIntBits(values[index]);
        return bits;
    }

    private static void assertFloatWord(float expected, int actual, float tolerance) {
        assertEquals(expected, Float.intBitsToFloat(actual), tolerance);
    }

    private static void assertModelWords(int[] expected, int[] actual) {
        assertEquals(expected.length, actual.length);
        for (int index = 0; index < expected.length; index++) {
            if (isNaN(expected[index])) assertTrue(isNaN(actual[index]), "NaN lane " + index);
            else assertEquals(expected[index], actual[index], "lane " + index);
        }
    }

    private static boolean isNaN(int word) {
        return (word & 0x7f80_0000) == 0x7f80_0000 && (word & 0x007f_ffff) != 0;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] words) {
        ByteBuffer encoded = ByteBuffer.allocate(Math.multiplyExact(words.length, Integer.BYTES))
                .order(ByteOrder.nativeOrder());
        for (int word : words) encoded.putInt(word);
        byte[] bytes = encoded.array();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1L);
            MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0L, bytes.length);
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        int bytes = Math.toIntExact(byteCount);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(bytes, 1L);
            api.download(buffer, 0L, target, bytes);
            ByteBuffer bufferBytes = ByteBuffer.wrap(target.toArray(JAVA_BYTE))
                    .order(ByteOrder.nativeOrder());
            int[] words = new int[bytes / Integer.BYTES];
            for (int index = 0; index < words.length; index++) words[index] = bufferBytes.getInt();
            return words;
        }
    }

    private static void uploadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, byte[] bytes) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(bytes.length, 1L);
            MemorySegment.copy(bytes, 0, source, JAVA_BYTE, 0L, bytes.length);
            api.upload(buffer, 0L, source, bytes.length);
        }
    }

    private static byte[] downloadBytes(
            MetalNativeApi api, MetalNativeApi.Handle buffer, long byteCount) {
        int bytes = Math.toIntExact(byteCount);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(bytes, 1L);
            api.download(buffer, 0L, target, bytes);
            return target.toArray(JAVA_BYTE);
        }
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}

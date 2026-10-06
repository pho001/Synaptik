package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MetalMpsGraphBinaryNativeTest {
    private static final int MIN_SUBNORMAL = 0x00000001;
    private static final int NEGATIVE_MIN_SUBNORMAL = 0x80000001;
    private static final int MIN_NORMAL = 0x00800000;
    private static final int NEGATIVE_MIN_NORMAL = 0x80800000;
    private static final int MAX_FINITE = 0x7f7fffff;
    private static final int POSITIVE_INFINITY = 0x7f800000;

    @Test
    void JavaPreflightClosesTheBinaryProfileMatrix() {

        int[] ranks = {2, 1, 2};
        long[] dimensions = dimensions(new long[][] {{2, 3}, {3}, {2, 3}});
        MetalMpsGraphProgram binary = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.SUB, 0, 1, 2)));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(ranks, dimensions, binary), binary, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(NumericalProfile.STRICT_IEEE, MetalTestProgram.descriptors(ranks, dimensions, binary), binary, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));


        long[] wrongBroadcast = dimensions(new long[][] {{2, 3}, {2}, {2, 3}});
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(ranks, wrongBroadcast, binary), binary, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH));
        assertThrows(IllegalArgumentException.class, () ->
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.NEG, 0, 1, 2));
    }

    @Test
    void realBinaryGraphsStayInsideIndependentDazFtzSetsAcrossTopologyAndReuse() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        Path library = Path.of(configured).toAbsolutePath().normalize();
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        try {
            context = api.createContext();
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.ADD,
                    MetalMpsGraphProgram.NodeKind.SUB,
                    MetalMpsGraphProgram.NodeKind.MUL,
                    MetalMpsGraphProgram.NodeKind.DIV)) {
                runTopology(api, context, kind);
            }
        } finally {
            if (context != null) {
                api.releaseContext(context);
            }
            api.close();
        }
    }

    private static void runTopology(
            MetalNativeApi api,
            MetalNativeApi.Handle context,
            MetalMpsGraphProgram.NodeKind kind) {
        int[][] inputBits = {
            {MIN_NORMAL, NEGATIVE_MIN_NORMAL, MIN_SUBNORMAL,
                    NEGATIVE_MIN_SUBNORMAL, MAX_FINITE, POSITIVE_INFINITY,
                    0x00000000, 0x80000000, 0x7fc12345},
            {Float.floatToRawIntBits(0.5f), Float.floatToRawIntBits(2.0f), MAX_FINITE},
            {MIN_SUBNORMAL, Float.floatToRawIntBits(-0.5f), 0x80000000},
            {MAX_FINITE}
        };
        int[] ranks = {2, 1, 2, 1, 2, 2, 2, 2, 2};
        long[] dimensions = dimensions(new long[][] {
            {3, 3}, {3}, {3, 1}, {1},
            {3, 3}, {3, 3}, {3, 3}, {3, 3}, {3, 3}
        });
        MetalMpsGraphProgram program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.binary(kind, 0, 1, 4),
                MetalMpsGraphProgram.Node.binary(kind, 4, 2, 5),
                MetalMpsGraphProgram.Node.binary(kind, 0, 0, 6),
                MetalMpsGraphProgram.Node.binary(kind, 0, 3, 7),
                MetalMpsGraphProgram.Node.binary(kind, 1, 0, 8)));
        MetalNativeApi.Handle executable = null;
        var inputs = new ArrayList<MetalNativeApi.Handle>();
        var outputs = new ArrayList<MetalNativeApi.Handle>();
        try {
            executable = api.createProgramExecutable(context, NumericalProfile.ACCELERATOR, MetalTestProgram.descriptors(ranks, dimensions, program), program, new int[] {0, 1, 2, 3}, new int[] {4, 5, 6, 7, 8}, MetalPreparedRoute.MPSGRAPH);
            for (int[] bits : inputBits) {
                MetalNativeApi.Handle buffer = api.createBuffer(
                        context, Math.multiplyExact((long) bits.length, Integer.BYTES));
                inputs.add(buffer);
                upload(api, buffer, bits);
            }
            for (int target = 0; target < 5; target++) {
                outputs.add(api.createBuffer(context, 9L * Integer.BYTES));
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment inputAddresses = arena.allocate(ADDRESS, inputs.size());
                MemorySegment outputAddresses = arena.allocate(ADDRESS, outputs.size());
                for (int index = 0; index < inputs.size(); index++) {
                    inputAddresses.setAtIndex(ADDRESS, index, inputs.get(index).carrier());
                }
                for (int index = 0; index < outputs.size(); index++) {
                    outputAddresses.setAtIndex(ADDRESS, index, outputs.get(index).carrier());
                }
                for (int repetition = 0; repetition < 2; repetition++) {
                    for (MetalNativeApi.Handle output : outputs) {
                        upload(api, output, new int[] {
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
                            0xdeadbeef, 0xdeadbeef, 0xdeadbeef
                        });
                    }
                    api.runExecutable(
                            executable,
                            inputs.size(),
                            inputAddresses,
                            outputs.size(),
                            outputAddresses);
                    int[][] actual = new int[outputs.size()][];
                    for (int target = 0; target < outputs.size(); target++) {
                        actual[target] = download(api, outputs.get(target), 9);
                    }
                    validateOutputs(kind, inputBits, actual, repetition);
                    for (int input = 0; input < inputs.size(); input++) {
                        assertArrayEquals(inputBits[input], download(
                                api, inputs.get(input), inputBits[input].length),
                                kind + " input preservation repetition=" + repetition);
                    }
                }
            }
        } finally {
            for (int index = outputs.size(); index-- > 0;) {
                api.releaseBuffer(outputs.get(index));
            }
            for (int index = inputs.size(); index-- > 0;) {
                api.releaseBuffer(inputs.get(index));
            }
            if (executable != null) {
                api.releaseExecutable(executable);
            }
        }
    }

    private static void validateOutputs(
            MetalMpsGraphProgram.NodeKind kind,
            int[][] inputs,
            int[][] actual,
            int repetition) {
        for (int lane = 0; lane < 9; lane++) {
            int row = lane / 3;
            int column = lane % 3;
            assertAllowed(kind, inputs[0][lane], inputs[1][column], actual[0][lane],
                    "row broadcast", repetition, lane);
            assertAllowed(kind, actual[0][lane], inputs[2][row], actual[1][lane],
                    "published intermediate/column broadcast", repetition, lane);
            assertAllowed(kind, inputs[0][lane], inputs[0][lane], actual[2][lane],
                    "repeated operand", repetition, lane);
            assertAllowed(kind, inputs[0][lane], inputs[3][0], actual[3][lane],
                    "scalar-shaped tensor broadcast", repetition, lane);
            assertAllowed(kind, inputs[1][column], inputs[0][lane], actual[4][lane],
                    "ordered reverse operands/fan-out", repetition, lane);
        }
    }

    private static void assertAllowed(
            MetalMpsGraphProgram.NodeKind kind,
            int left,
            int right,
            int observed,
            String topology,
            int repetition,
            int lane) {
        Allowed allowed = allowed(kind, left, right);
        boolean accepted = isNaN(observed) ? allowed.nan() : allowed.bits().contains(observed);
        assertTrue(
                accepted,
                kind + " " + topology + " repetition=" + repetition + " lane=" + lane
                        + " left=0x" + Integer.toHexString(left)
                        + " right=0x" + Integer.toHexString(right)
                        + " observed=0x" + Integer.toHexString(observed));
    }

    private static Allowed allowed(
            MetalMpsGraphProgram.NodeKind kind, int declaredLeft, int declaredRight) {
        Set<Integer> bits = new HashSet<>();
        boolean[] nan = {false};
        int[] left = choices(declaredLeft);
        int[] right = choices(declaredRight);
        for (int selectedLeft : left) {
            for (int selectedRight : right) {
                int result = evaluate(kind, selectedLeft, selectedRight);
                if (isNaN(result)) {
                    nan[0] = true;
                } else {
                    bits.add(result);
                }
                if (isFiniteSubnormal(result)) {
                    bits.add(0x00000000);
                    bits.add(0x80000000);
                }
                if ((kind == MetalMpsGraphProgram.NodeKind.ADD
                        || kind == MetalMpsGraphProgram.NodeKind.SUB)
                        && (result & 0x7fffffff) == 0) {
                    bits.add(0x00000000);
                    bits.add(0x80000000);
                }
            }
        }
        return new Allowed(Set.copyOf(bits), nan[0]);
    }

    private static int[] choices(int bits) {
        return isSubnormal(bits)
                ? new int[] {bits, bits & 0x80000000}
                : new int[] {bits};
    }

    private static int evaluate(
            MetalMpsGraphProgram.NodeKind kind, int leftBits, int rightBits) {
        float left = Float.intBitsToFloat(leftBits);
        float right = Float.intBitsToFloat(rightBits);
        float result = switch (kind) {
            case ADD -> left + right;
            case SUB -> left - right;
            case MUL -> left * right;
            case DIV -> left / right;
            default -> throw new IllegalArgumentException("not a binary kind: " + kind);
        };
        return Float.floatToRawIntBits(result);
    }

    private static boolean isSubnormal(int bits) {
        int magnitude = bits & 0x7fffffff;
        return magnitude != 0 && magnitude < 0x00800000;
    }

    private static boolean isFiniteSubnormal(int bits) {
        return isSubnormal(bits) && (bits & 0x7f800000) != 0x7f800000;
    }

    private static boolean isNaN(int bits) {
        return (bits & 0x7f800000) == 0x7f800000 && (bits & 0x007fffff) != 0;
    }

    private static void upload(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int[] bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(
                    Math.multiplyExact((long) bits.length, Integer.BYTES), Integer.BYTES);
            for (int index = 0; index < bits.length; index++) {
                source.setAtIndex(JAVA_INT, index, bits[index]);
            }
            api.upload(buffer, 0L, source, source.byteSize());
        }
    }

    private static int[] download(
            MetalNativeApi api, MetalNativeApi.Handle buffer, int count) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment target = arena.allocate(
                    Math.multiplyExact((long) count, Integer.BYTES), Integer.BYTES);
            api.download(buffer, 0L, target, target.byteSize());
            int[] result = new int[count];
            for (int index = 0; index < count; index++) {
                result[index] = target.getAtIndex(JAVA_INT, index);
            }
            return result;
        }
    }

    private static long[] dimensions(long[][] shapes) {
        long[] dimensions = new long[Math.multiplyExact(shapes.length, 16)];
        for (int value = 0; value < shapes.length; value++) {
            System.arraycopy(shapes[value], 0, dimensions, value * 16, shapes[value].length);
        }
        return dimensions;
    }

    private record Allowed(Set<Integer> bits, boolean nan) { }
}

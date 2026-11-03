package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

/**
 * Real-device F32 elementary qualification. StrictMath supplies elementary references;
 * ERF uses independent numerical integration, not the production rational algorithm.
 * Composite references expose the Model's FLOAT32 working boundaries explicitly.
 */
final class MetalFloat32MathNativeTest {
    // EXP and SIGMOID retain their dedicated production-route and underflow qualification suites.
    private static final List<MetalMpsGraphProgram.NodeKind> CUSTOM_UNARY = List.of(
            MetalMpsGraphProgram.NodeKind.LOG, MetalMpsGraphProgram.NodeKind.LOG1P,
            MetalMpsGraphProgram.NodeKind.EXPM1,
            MetalMpsGraphProgram.NodeKind.ERF, MetalMpsGraphProgram.NodeKind.SQRT,
            MetalMpsGraphProgram.NodeKind.RSQRT,
            MetalMpsGraphProgram.NodeKind.TANH, MetalMpsGraphProgram.NodeKind.GELU,
            MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION,
            MetalMpsGraphProgram.NodeKind.SILU);

    @Test
    void unaryClassesSignsAndOrdinaryValuesFollowNamedFloat32Sites() {
        float[] input = {-Float.MAX_VALUE, -100f, -88f, -32f, -8f, -4f, -2f, -1f,
                Math.nextUp(-1f), -0.5f, -0.125f, -0.0f, 0.0f, 0.125f, 0.5f,
                1f, 2f, 4f, 8f, 32f, 88f, 100f, Float.MAX_VALUE,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
        for (var kind : CUSTOM_UNARY) {
            float[] actual = execute(unary(kind),
                    List.of(value(input.length), value(input.length)),
                    new int[] {0}, 1, List.of(input));
            for (int index = 0; index < input.length; index++) {
                assertUnaryValue(kind, input[index], actual[index], kind + "(" + input[index] + ")");
            }
        }
    }

    @Test
    void cancellationSensitiveElementarySitesRetainNearZeroResults() {
        float[] input = {-0x1p-80f, -0x1p-40f, -0x1p-25f, -0x1p-24f,
                -0x1p-20f, -0.0001f, -0.0f, 0.0f,
                0x1p-80f, 0x1p-40f, 0x1p-25f, 0x1p-24f, 0x1p-20f, 0.0001f};
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.LOG1P,
                MetalMpsGraphProgram.NodeKind.EXPM1, MetalMpsGraphProgram.NodeKind.TANH)) {
            float[] actual = execute(unary(kind),
                    List.of(value(input.length), value(input.length)),
                    new int[] {0}, 1, List.of(input));
            for (int index = 0; index < input.length; index++) {
                assertValue(reference(kind, input[index]), actual[index],
                        kind + " near zero " + input[index], false);
            }
        }
    }

    @Test
    void ordinaryFiniteSweepQualifiesElementaryAndCompositeAccuracy() {
        float[] input = new float[257];
        for (int index = 0; index < input.length; index++) input[index] = (index - 128) / 32f;
        for (var kind : CUSTOM_UNARY) {
            float[] actual = execute(unary(kind),
                    List.of(value(input.length), value(input.length)),
                    new int[] {0}, 1, List.of(input));
            for (int index = 0; index < input.length; index++) {
                assertUnaryValue(kind, input[index], actual[index], kind + " sweep " + input[index]);
            }
        }
    }

    @Test
    void binaryPowerAndExtremaConsumeTheDeclaredStridedRightLayout() {
        Shape shape = Shape.of(2);
        var right = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                new long[] {2}, Optional.of(LayoutDescriptor.of(
                        shape, new long[] {2}, 0, true)), false, false);
        var kinds = List.of(MetalMpsGraphProgram.NodeKind.TENSOR_POW,
                MetalMpsGraphProgram.NodeKind.TENSOR_MIN, MetalMpsGraphProgram.NodeKind.TENSOR_MAX);
        float[][] expected = {{8, 9}, {2, 2}, {3, 3}};
        for (int index = 0; index < kinds.size(); index++) {
            var node = MetalMpsGraphProgram.Node.generic(kinds.get(index),
                    new int[] {0, 1}, new int[] {2},
                    MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
            float[] actual = execute(node, List.of(value(2), right, value(2)),
                    new int[] {0, 1}, 2, List.of(
                            new float[] {2, 3}, new float[] {3, Float.NaN, 2}));
            assertArrayEquals(expected[index], actual, 0.00001f);
        }
    }

    @Test
    void scalarDivisionReadsTheMaterializedExpandedValueRatherThanItsLogicalSpan() {
        Shape shape = Shape.of(3);
        var expanded = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                new long[] {3}, Optional.of(LayoutDescriptor.of(
                        shape, new long[] {0}, 0, true)), false, true);
        var nodes = List.of(MetalMpsGraphProgram.Node.targetShape(
                        MetalMpsGraphProgram.NodeKind.EXPAND, 0, 1, new long[] {3}),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_DIV, 1, 2, DataType.FLOAT32, 0x40000000L));
        assertArrayEquals(new float[] {4, 4, 4}, execute(nodes,
                List.of(value(), expanded, value(3)), new int[] {0}, 2,
                List.of(new float[] {8})), 0f);
    }

    @Test
    void linearPointKernelsRejectStridedFeedsThatHaveNoDenseProducerRepresentation() {
        Shape shape = Shape.of(3);
        var input = new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32,
                new long[] {3}, Optional.of(LayoutDescriptor.of(
                        shape, new long[] {2}, 0, true)), false, false);
        var node = MetalMpsGraphProgram.Node.scalarValue(
                MetalMpsGraphProgram.NodeKind.SCALAR_MUL, 0, 1, DataType.FLOAT32, 0x40000000L);
        var values = List.of(input, value(3));
        var route = MetalPointwiseFusionPlanner.fixedCustom(node, values)
                ? MetalPreparedRoute.CUSTOM_PROGRAM : MetalPreparedRoute.MPSGRAPH;
        assertThrows(IllegalArgumentException.class, () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values, new MetalMpsGraphProgram(List.of(node)), new int[] {0}, new int[] {1}, route));
    }

    @Test
    void tensorPowerBroadcastsBasesAndExponentsWithExactSpecialClasses() {
        float[] base = {-2f, -1f, -0f, 0f, 1f, 2f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
        float[] exponent = {-3f, -2f, -0.5f, -0f, 0f, 0.5f, 2f, 3f,
                16777215f, 16777216f, Float.NEGATIVE_INFINITY,
                Float.POSITIVE_INFINITY, Float.NaN};
        var node = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.TENSOR_POW, new int[] {0, 1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        float[] actual = execute(node,
                List.of(value(base.length, 1), value(1, exponent.length),
                        value(base.length, exponent.length)),
                new int[] {0, 1}, 2, List.of(base, exponent));
        for (int row = 0; row < base.length; row++) {
            for (int column = 0; column < exponent.length; column++) {
                float expected = (float) StrictMath.pow(base[row], exponent[column]);
                assertValue(expected, actual[row * exponent.length + column],
                        base[row] + " ^ " + exponent[column], false);
            }
        }
    }

    @Test
    void scalarPowerUsesRawTypedExponentAndPreservesZeroSigns() {
        float[] base = {-4f, -2f, -1f, -0f, 0f, 0.25f, 1f, 2f, 4f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN};
        for (float exponent : new float[] {-3f, -0.5f, -0f, 0f, 0.5f, 2f, 3f,
                Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN}) {
            var node = MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.SCALAR_POW, new int[] {0}, new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE,
                    new long[] {1L, Integer.toUnsignedLong(Float.floatToRawIntBits(exponent))});
            float[] actual = execute(node,
                    List.of(value(base.length), value(base.length)),
                    new int[] {0}, 1, List.of(base));
            for (int index = 0; index < base.length; index++) {
                assertValue((float) StrictMath.pow(base[index], exponent), actual[index],
                        "scalar " + base[index] + " ^ " + exponent, false);
            }
        }
    }

    /** Evaluates the named working sites; elementary ERF is independently integrated. */
    private static float reference(MetalMpsGraphProgram.NodeKind kind, float x) {
        return switch (kind) {
            case LOG -> (float) StrictMath.log(x);
            case LOG1P -> (float) StrictMath.log1p(x);
            case EXPM1 -> (float) StrictMath.expm1(x);
            case ERF -> (float) integratedErf(x);
            case SQRT -> (float) StrictMath.sqrt(x);
            case RSQRT -> 1f / (float) StrictMath.sqrt(x);
            case TANH -> (float) StrictMath.tanh(x);
            case GELU -> {
                if (x == Float.NEGATIVE_INFINITY) yield -0f;
                float argument = x / (float) StrictMath.sqrt(2f);
                float elementary = (float) integratedErf(argument);
                yield (0.5f * x) * (1f + elementary);
            }
            case GELU_TANH_APPROXIMATION -> {
                if (x == Float.NEGATIVE_INFINITY) yield -0f;
                float square = x * x;
                float cube = square * x;
                float ratio = 2f / (float) StrictMath.PI;
                float scale = (float) StrictMath.sqrt(ratio);
                float inner = x + 0.044715f * cube;
                float elementary = (float) StrictMath.tanh(scale * inner);
                yield (0.5f * x) * (1f + elementary);
            }
            case SILU -> {
                if (x == Float.NEGATIVE_INFINITY) yield -0f;
                if (x >= 0f) yield x / (1f + (float) StrictMath.exp(-x));
                float e = (float) StrictMath.exp(x);
                yield (x * e) / (1f + e);
            }
            default -> throw new AssertionError(kind);
        };
    }

    /** Composite Simpson integration of ERF's defining integral with 2048 panels. */
    private static double integratedErf(double x) {
        if (Double.isNaN(x) || x == 0d) return x;
        double a = StrictMath.abs(x);
        if (a >= 6d) return StrictMath.copySign(1d, x);
        int panels = 2048;
        double step = a / panels;
        double sum = 1d + StrictMath.exp(-a * a);
        for (int index = 1; index < panels; index++) {
            double t = index * step;
            sum += (index % 2 == 0 ? 2d : 4d) * StrictMath.exp(-t * t);
        }
        return StrictMath.copySign((2d / StrictMath.sqrt(StrictMath.PI)) * step * sum / 3d, x);
    }

    /** SiLU's negative-branch MUL may consume its subnormal EXP operand as positive zero. */
    private static void assertUnaryValue(
            MetalMpsGraphProgram.NodeKind kind, float input, float actual, String label) {
        if (kind == MetalMpsGraphProgram.NodeKind.SILU && Float.isFinite(input) && input < 0f) {
            float exponential = (float) StrictMath.exp(input);
            if (exponential > 0f && exponential < Float.MIN_NORMAL
                    && Float.floatToRawIntBits(actual) == 0x80000000) return;
        }
        assertValue(reference(kind, input), actual, label, activation(kind));
    }

    /** Checks classes/signs separately; finite error uses a route-local test tolerance. */
    private static void assertValue(float expected, float actual, String label, boolean activation) {
        if (Float.isNaN(expected)) {
            assertTrue(Float.isNaN(actual), label + " must be NaN, got " + actual);
        } else if (Float.isInfinite(expected) || expected == 0f) {
            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(actual), label);
        } else {
            assertTrue(Float.isFinite(actual), label + " must be finite, got " + actual);
            double tolerance = Math.max(3e-6 * Math.abs((double) expected),
                    activation ? 3e-7 : 0d);
            assertEquals((double) expected, actual, tolerance, label);
        }
    }

    private static boolean activation(MetalMpsGraphProgram.NodeKind kind) {
        return kind == MetalMpsGraphProgram.NodeKind.GELU
                || kind == MetalMpsGraphProgram.NodeKind.GELU_TANH_APPROXIMATION;
    }

    private static MetalMpsGraphProgram.Node unary(MetalMpsGraphProgram.NodeKind kind) {
        return MetalMpsGraphProgram.Node.generic(kind, new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dimensions, false);
    }

    /** Executes the production fixed route without host numeric repair. */
    private static float[] execute(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds, int target, List<float[]> inputs) {
        return execute(List.of(node), values, feeds, target, inputs);
    }

    private static float[] execute(List<MetalMpsGraphProgram.Node> nodes,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds, int target, List<float[]> inputs) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        try (MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize())) {
            MetalNativeApi.Handle context = api.createContext();
            MetalNativeApi.Handle executable = null;
            var buffers = new ArrayList<MetalNativeApi.Handle>();
            try {
                var program = new MetalMpsGraphProgram(nodes);
                int[] targets = {target};
                var route = nodes.stream().anyMatch(node -> MetalPointwiseFusionPlanner.fixedCustom(node, values))
                        ? MetalPreparedRoute.CUSTOM_PROGRAM : MetalPreparedRoute.MPSGRAPH;
                var fusion = route == MetalPreparedRoute.CUSTOM_PROGRAM
                        ? MetalPointwiseFusionPlanner.plan(program, values, feeds, targets, route) : null;
                executable = api.createProgramExecutable(context, values, program, feeds, targets, route, fusion);
                int[] slots = fusion == null
                        ? IntStream.range(0, values.size()).toArray() : fusion.programToMaterializedSlot();
                int[] materialized = fusion == null ? slots : fusion.materializedProgramValueIndices();
                for (int value : materialized) {
                    buffers.add(api.createBuffer(context, values.get(value).byteCount()));
                }
                try (Arena arena = Arena.ofConfined()) {
                    for (int index = 0; index < feeds.length; index++) {
                        float[] input = inputs.get(index);
                        MemorySegment source = arena.allocate(JAVA_FLOAT, input.length);
                        MemorySegment.copy(input, 0, source, JAVA_FLOAT, 0, input.length);
                        api.upload(buffers.get(slots[feeds[index]]), 0, source, source.byteSize());
                    }
                    int inputCount = route == MetalPreparedRoute.CUSTOM_PROGRAM ? buffers.size() : feeds.length;
                    MemorySegment addresses = arena.allocate(ADDRESS, inputCount);
                    for (int index = 0; index < inputCount; index++) {
                        int slot = route == MetalPreparedRoute.CUSTOM_PROGRAM ? index : slots[feeds[index]];
                        addresses.setAtIndex(ADDRESS, index, buffers.get(slot).carrier());
                    }
                    MemorySegment outputs = arena.allocate(ADDRESS, 1);
                    outputs.setAtIndex(ADDRESS, 0, buffers.get(slots[target]).carrier());
                    api.runExecutable(executable, inputCount, addresses, 1, outputs);
                    MemorySegment result = arena.allocate(values.get(target).byteCount(), 4);
                    api.download(buffers.get(slots[target]), 0, result, result.byteSize());
                    return result.toArray(JAVA_FLOAT);
                }
            } finally {
                for (int index = buffers.size(); index-- > 0;) api.releaseBuffer(buffers.get(index));
                if (executable != null) api.releaseExecutable(executable);
                api.releaseContext(context);
            }
        }
    }
}

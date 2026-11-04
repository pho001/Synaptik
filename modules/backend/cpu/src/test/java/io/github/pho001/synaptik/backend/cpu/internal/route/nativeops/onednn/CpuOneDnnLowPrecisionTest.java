package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGenerator;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.Float8E4M3Bits;
import io.github.pho001.synaptik.model.datatype.Float8E5M2Bits;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

/** Consumer-visible destination rounding and optional real typed-input native execution. */
final class CpuOneDnnLowPrecisionTest {
    private static final byte CANARY = 0x5a;

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"BFLOAT16", "FLOAT16", "FLOAT8_E4M3", "FLOAT8_E5M2"})
    void stagedResultUsesExactFinalCastAndPreservesStridedGaps(DataType type) {
        var output = new CpuOneDnnKernelSpec.Tensor(DataType.FLOAT32, List.of(2L, 6L), List.of(23L, 3L));
        var spec = new CpuOneDnnKernelSpec.Unary(CpuOneDnnKernelSpec.UnaryAlgorithm.SQRT, output, output);
        var projection = new CpuOneDnnProjection(spec, List.of(0L, 0L), List.of(0), 1, type);
        var recipe = CpuOneDnnOutputNarrowing.from(projection);
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(recipe.specialization(), recipe.ir(),
                generator.generateClassBytes(recipe.specialization(), recipe.ir()));
        float tie = switch (type) {
            case BFLOAT16 -> 1.00390625f;
            case FLOAT16 -> 1.00048828125f;
            case FLOAT8_E4M3 -> 1.0625f;
            case FLOAT8_E5M2 -> 1.125f;
            default -> throw new AssertionError(type);
        };
        float[] values = {-0.0f, 0.0f, tie, Math.nextUp(tie), -tie, Math.nextDown(-tie),
                0x1.0p-25f, 464.0f, Math.nextUp(464.0f), Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.intBitsToFloat(0xffc12345)};
        try (Arena arena = Arena.ofConfined()) {
            var source = arena.allocate(output.byteSize(), 64);
            var allocation = arena.allocate(projection.logicalOutputByteSize() + 128, 64);
            allocation.fill(CANARY);
            var destination = allocation.asSlice(64, projection.logicalOutputByteSize());
            boolean[] occupied = new boolean[Math.toIntExact(allocation.byteSize())];
            for (int index = 0; index < values.length; index++) {
                long element = (index / 6) * 23L + (index % 6) * 3L;
                source.set(JAVA_FLOAT, element * Float.BYTES, values[index]);
                for (int b = 0; b < type.byteWidth(); b++) {
                    occupied[Math.toIntExact(64 + element * type.byteWidth() + b)] = true;
                }
            }
            Runnable narrow = recipe.bind(artifact, source, destination);
            narrow.run();
            for (int index = 0; index < values.length; index++) {
                long element = (index / 6) * 23L + (index % 6) * 3L;
                assertEquals(bits(type, values[index]), raw(destination, type, element));
            }
            for (int b = 0; b < occupied.length; b++) {
                if (!occupied[b]) assertEquals(CANARY, allocation.get(JAVA_BYTE, b), "gap/guard " + b);
            }
            // Repeated execution refreshes the same logical store, without an across-run snapshot.
            source.set(JAVA_FLOAT, 0, 2.0f);
            narrow.run();
            assertEquals(bits(type, 2.0f), raw(destination, type, 0));
        }
    }

    @Test
    void graphFp8AndIdentityReductionRejectBeforeTouchingLibrary() {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            var input = dense(type, 2, 2);
            var spec = matmul(input, input, dense(DataType.FLOAT32, 2, 2));
            assertThrows(CpuOneDnnCompiler.UnsupportedRealizationException.class,
                    () -> CpuOneDnnCompiler.describe(null, spec, CpuOneDnnRoutePlan.Api.GRAPH));
            var identity = new CpuOneDnnKernelSpec.Reduction(
                    CpuOneDnnKernelSpec.ReductionAlgorithm.SUM, input, dense(DataType.FLOAT32, 2, 2));
            assertFalse(CpuOneDnnCompiler.qualified(identity));
            assertThrows(IllegalArgumentException.class,
                    () -> CpuOneDnnCompiler.describe(null, identity, CpuOneDnnRoutePlan.Api.PRIMITIVE));
        }
    }

    @ParameterizedTest
    @MethodSource("nativeCases")
    void typedMatmulStagesF32ThenNarrowsAndReadsFreshWeights(DataType type, CpuOneDnnRoutePlan.Api api) {
        String configured = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "explicit native library required");
        var spec = matmul(dense(type, 2, 3), dense(type, 3, 2), dense(DataType.FLOAT32, 2, 2));
        var projection = new CpuOneDnnProjection(spec, List.of(0L, 0L, 0L), List.of(0, 1), 2, type);
        var recipe = CpuOneDnnOutputNarrowing.from(projection);
        var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(recipe.specialization(), recipe.ir(),
                generator.generateClassBytes(recipe.specialization(), recipe.ir()));
        try (CpuOneDnnLibrary library = CpuOneDnnLibrary.open(Path.of(configured));
                Arena arena = Arena.ofConfined();
                CpuOneDnnKernel kernel = available(library, spec, api);
                CpuOneDnnWorkspace workspace = kernel.createWorkspace()) {
            var left = arena.allocate(spec.inputs().getFirst().byteSize(), 64);
            var right = arena.allocate(spec.inputs().get(1).byteSize(), 64);
            var stage = arena.allocate(spec.output().byteSize(), 64);
            var output = arena.allocate(projection.logicalOutputByteSize(), 64);
            float[] a = {-2, -1, 0, 1, 2, 3};
            float[] b = {1, 2, 3, 4, 5, 6};
            for (int i = 0; i < a.length; i++) put(left, type, i, a[i]);
            for (int i = 0; i < b.length; i++) put(right, type, i, b[i]);
            byte[] before = left.toArray(JAVA_BYTE);
            workspace.bind(new MemorySegment[] {left.asReadOnly(), right.asReadOnly()}, stage);
            Runnable narrow = recipe.bind(artifact, stage, output);
            float[] expected = {-5, -8, 22, 28};
            for (int run = 0; run < 2; run++) {
                workspace.execute();
                narrow.run();
                for (int i = 0; i < expected.length; i++) {
                    float value = run == 0 ? expected[i] : -expected[i];
                    assertEquals(value, stage.get(JAVA_FLOAT, i * 4L));
                    assertEquals(bits(type, value), raw(output, type, i));
                }
                for (int i = 0; i < b.length; i++) put(right, type, i, -b[i]);
            }
            assertArrayEquals(before, left.toArray(JAVA_BYTE));
        }
    }

    @ParameterizedTest
    @MethodSource("nativeCases")
    void strictTypedAccumulationDoesNotSpillPartialSumsToLow(DataType type, CpuOneDnnRoutePlan.Api api) {
        String configured = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "explicit native library required");
        var spec = matmul(dense(type, 2, 4097), dense(type, 4097, 2), dense(DataType.FLOAT32, 2, 2));
        try (CpuOneDnnLibrary library = CpuOneDnnLibrary.open(Path.of(configured));
                Arena arena = Arena.ofConfined();
                CpuOneDnnKernel kernel = available(library, spec, api);
                CpuOneDnnWorkspace workspace = kernel.createWorkspace()) {
            var left = arena.allocate(spec.inputs().getFirst().byteSize(), 64);
            var right = arena.allocate(spec.inputs().get(1).byteSize(), 64);
            var output = arena.allocate(spec.output().byteSize(), 64);
            for (int i = 0; i < 8194; i++) {
                put(left, type, i, 1.0f);
                put(right, type, i, 1.0f);
            }
            workspace.bind(new MemorySegment[] {left, right}, output);
            workspace.execute();
            for (int i = 0; i < 4; i++) assertEquals(4097.0f, output.get(JAVA_FLOAT, i * 4L));
        }
    }

    static Stream<Arguments> nativeCases() {
        return Stream.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)
                .flatMap(type -> (type == DataType.BFLOAT16 || type == DataType.FLOAT16
                        ? Stream.of(CpuOneDnnRoutePlan.Api.PRIMITIVE, CpuOneDnnRoutePlan.Api.GRAPH)
                        : Stream.of(CpuOneDnnRoutePlan.Api.PRIMITIVE)).map(api -> Arguments.of(type, api)));
    }

    private static CpuOneDnnKernel available(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec,
            CpuOneDnnRoutePlan.Api api) {
        try { return CpuOneDnnCompiler.compile(library, spec, api); }
        catch (CpuOneDnnCompiler.UnsupportedRealizationException unqualified) {
            assumeTrue(false, "selected typed implementation is not qualified: " + unqualified.getMessage());
            throw new AssertionError("unreachable");
        }
        catch (CpuOneDnnLibrary.NativeStatusException unavailable) {
            if (unavailable.status() != 2 && unavailable.status() != 3) throw unavailable;
            assumeTrue(false, "typed native implementation unavailable: " + unavailable.getMessage());
            throw new AssertionError("unreachable");
        }
    }

    private static CpuOneDnnKernelSpec.Tensor dense(DataType type, long rows, long columns) {
        return new CpuOneDnnKernelSpec.Tensor(type, List.of(rows, columns), List.of(columns, 1L));
    }

    private static CpuOneDnnKernelSpec.Contraction matmul(CpuOneDnnKernelSpec.Tensor left,
            CpuOneDnnKernelSpec.Tensor right, CpuOneDnnKernelSpec.Tensor output) {
        return new CpuOneDnnKernelSpec.Contraction(CpuOneDnnKernelSpec.Operation.MATMUL,
                List.of(left, right), output, 1, List.of(), List.of(), List.of(), List.of(),
                CpuOneDnnKernelSpec.Bias.NONE, CpuOneDnnKernelSpec.Activation.NONE);
    }

    private static int bits(DataType type, float value) {
        return switch (type) {
            case BFLOAT16 -> BFloat16Bits.fromFloat(value) & 0xffff;
            case FLOAT16 -> Float16Bits.fromFloat(value) & 0xffff;
            case FLOAT8_E4M3 -> Float8E4M3Bits.fromFloat(value) & 0xff;
            case FLOAT8_E5M2 -> Float8E5M2Bits.fromFloat(value) & 0xff;
            default -> throw new AssertionError(type);
        };
    }

    private static void put(MemorySegment segment, DataType type, long element, float value) {
        int bits = bits(type, value);
        if (type.byteWidth() == 1) segment.set(JAVA_BYTE, element, (byte) bits);
        else segment.set(JAVA_SHORT, element * 2, (short) bits);
    }

    private static int raw(MemorySegment segment, DataType type, long element) {
        return type.byteWidth() == 1 ? segment.get(JAVA_BYTE, element) & 0xff
                : segment.get(JAVA_SHORT, element * 2) & 0xffff;
    }
}

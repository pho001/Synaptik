package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.StreamSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/** Exact/class/named-site regressions, independent of the production differential reference. */
final class CpuActivationNumericsTest {
    static final List<CpuPointwiseOpcode> COMPOSITES = List.of(CpuPointwiseOpcode.RSQRT,
            CpuPointwiseOpcode.SIGMOID, CpuPointwiseOpcode.GELU_EXACT,
            CpuPointwiseOpcode.GELU_TANH_APPROXIMATION, CpuPointwiseOpcode.SILU);

    @Test void saturationBoundAndIndependentIntegralQualifyElementaryOracle() {
        double bound = StrictMath.exp(-36) / (6 * StrictMath.sqrt(Math.PI));
        assertTrue(bound < 0x1p-54, "strictly below the F64 half-gap immediately below one");
        assertTrue(bound < 0x1p-25, "also below the F32 half-gap");
        for (double x : new double[] {0.125, 0.5, 1, 1.5, 2, 3, 4, 5, 5.9}) {
            // Composite Simpson integration is independent of the production rational data.
            int n = 16_384;
            double h = x / n;
            double sum = 1 + StrictMath.exp(-x * x);
            for (int i = 1; i < n; i++) {
                double t = h * i;
                sum += ((i & 1) == 0 ? 2 : 4) * StrictMath.exp(-t * t);
            }
            double integral = 2 / StrictMath.sqrt(Math.PI) * h * sum / 3;
            assertEquals(integral, CpuActivationDirectOracle.erf(x), 2e-14, "integral " + x);
        }
        assertEquals(0.8427007929497149, CpuActivationDirectOracle.erf(1), 2e-16);
        for (double x : new double[] {Math.nextDown(6d), 6d, Math.nextUp(6d)}) {
            assertEquals(1d, CpuActivationDirectOracle.erf(x));
            assertEquals(-1d, CpuActivationDirectOracle.erf(-x));
        }
    }

    @Test void extremesBoundarySpecialsAndVectorTailsMatchIndependentResults() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            double boundaryBelow = type == DataType.FLOAT32 ? Math.nextDown(6f) : Math.nextDown(6d);
            double boundaryAbove = type == DataType.FLOAT32 ? Math.nextUp(6f) : Math.nextUp(6d);
            double max = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
            double tiny = type == DataType.FLOAT32 ? Float.MIN_VALUE : Double.MIN_VALUE;
            for (CpuPointwiseOpcode kind : List.of(CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.GELU_EXACT))
                for (ExecutionStrategy strategy : List.of(ExecutionStrategy.SCALAR, ExecutionStrategy.VECTOR))
                    for (double x : new double[] {Double.NaN, Double.NEGATIVE_INFINITY,
                            Double.POSITIVE_INFINITY, -0d, 0d, -tiny, tiny, -max, max,
                            -1e10, 1e10, -1e100, 1e100, -boundaryBelow, boundaryBelow,
                            -6, 6, -boundaryAbove, boundaryAbove, -0.5, 0.5, -1, 1, -2, 2}) {
                        int count = 2 * lanes(type) + 1;
                        Object input = type == DataType.FLOAT64 ? new double[count] : new float[count];
                        Object output = type == DataType.FLOAT64 ? new double[count] : new float[count];
                        for (int i = 0; i < count; i++) put(input, i, x);
                        Fixture fixture = fixture(type, kind, strategy, false, false, false);
                        fixture.kernel().entryPoint().invokeWithArguments(input, output,
                                geometry(count, 0, 0, 0, 1, count), 0L, (long) count);
                        double represented = get(input, 0);
                        double expected = kind == CpuPointwiseOpcode.ERF
                                ? CpuActivationDirectOracle.erf(represented)
                                : represented == Double.NEGATIVE_INFINITY ? -0d
                                : (0.5 * represented) * (1 + CpuActivationDirectOracle.erf(
                                        represented / Math.sqrt(2)));
                        if (type == DataType.FLOAT32) expected = (float) expected;
                        for (int i = 0; i < count; i++) {
                            // Existing F32 vector bodies have typed sites; their scalar tail widens.
                            double laneExpected = kind == CpuPointwiseOpcode.GELU_EXACT
                                    && type == DataType.FLOAT32 && strategy == ExecutionStrategy.VECTOR
                                    && i < count - 1 ? CpuActivationDirectOracle.gelu((float) represented) : expected;
                            String label = type + " " + kind + " " + strategy + " x=" + x + " i=" + i;
                            double actual = get(output, i);
                            if (Double.isNaN(laneExpected)) assertTrue(Double.isNaN(actual), label);
                            else if (laneExpected == 0 || Math.abs(represented) >= (kind == CpuPointwiseOpcode.ERF ? boundaryBelow : 9)
                                    || Double.isInfinite(laneExpected)) assertBits(laneExpected, actual, label);
                            else assertEquals(laneExpected, actual, type == DataType.FLOAT32 ? 2e-7 : 3e-16, label);
                        }
                    }
        }
    }

    @Test void allLowRepresentationsUseNamedFloatSitesIncludingCancellation() throws Throwable {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (CpuPointwiseOpcode kind : COMPOSITES) {
                int count = 65_536;
                short[] input = new short[count], actual = new short[count], reference = new short[count];
                for (int i = 0; i < count; i++) input[i] = (short) i;
                Fixture fixture = fixture(type, kind, ExecutionStrategy.SCALAR, false, false, false);
                fixture.kernel().entryPoint().invokeExact(input, actual,
                        geometry(count, 0, 0, 0, 1, count), 0L, (long) count);
                var ir = fixture.ir();
                CpuScalarReferenceKernel.execute(ir, List.of(
                        new CpuBufferArgument.Shorts(input, 0, count * 2L, true),
                        new CpuBufferArgument.Shorts(reference, 0, count * 2L, false)),
                        ir.values().stream().map(value -> CpuAccessPlan.Binding.create(value.accessPlan(),
                                new long[] {count}, 0, new long[] {1}, count, 0, count, count)).toList(),
                        0, count);
                for (int i = 0; i < count; i++) {
                    float x = CpuActivationDirectOracle.decode(type, (short) i);
                    float expected = CpuActivationDirectOracle.composite(kind, x);
                    String label = type + " " + kind + " bits=" + Integer.toHexString(i);
                    if (Float.isNaN(expected)) {
                        assertTrue(Float.isNaN(CpuActivationDirectOracle.decode(type, actual[i])), label);
                        assertTrue(Float.isNaN(CpuActivationDirectOracle.decode(type, reference[i])), label);
                    } else {
                        short bits = CpuActivationDirectOracle.encode(type, expected);
                        assertEquals(bits, actual[i], label);
                        assertEquals(bits, reference[i], "reference " + label);
                    }
                    assertEquals((short) i, input[i], "borrowed input");
                }
                if (kind == CpuPointwiseOpcode.GELU_EXACT) {
                    int index = Short.toUnsignedInt(CpuActivationDirectOracle.encode(type, -6f));
                    assertEquals((short) 0x8000, actual[index], "GELU(-6) is negative zero");
                }
            }
    }

    @Test void carrierPairsOffsetsStridesAndPartialRangesKeepCanaries() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16))
            for (CpuPointwiseOpcode kind : List.of(CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.GELU_EXACT))
                for (boolean strided : List.of(false, true))
                    for (boolean nativeIn : List.of(false, true))
                        for (boolean nativeOut : List.of(false, true)) {
                            int count = 17, stride = strided ? 2 : 1, span = 48, start = 2, end = 15;
                            Object input = array(type, span), output = array(type, span);
                            for (int i = 0; i < span; i++) { putTyped(type, input, i, -71); putTyped(type, output, i, -71); }
                            double[] values = {-6, 6, -1e10, 1e10, -0d, 0, Double.NaN,
                                    Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, -1, 1};
                            for (int i = 0; i < count; i++) putTyped(type, input, 3 + i * stride, values[i % values.length]);
                            try (Arena arena = Arena.ofConfined()) {
                                MemorySegment in = segment(input), out = segment(output);
                                MemorySegment nativeInput = arena.allocate(in.byteSize(), type.byteWidth());
                                MemorySegment nativeOutput = arena.allocate(out.byteSize(), type.byteWidth());
                                nativeInput.copyFrom(in); nativeOutput.copyFrom(out);
                                Fixture fixture = fixture(type, kind, ExecutionStrategy.SCALAR, nativeIn, nativeOut, strided);
                                long[] geometry = geometry(count, start, 3, 5, stride, span);
                                fixture.kernel().entryPoint().invokeWithArguments(
                                        nativeIn ? nativeInput : input, nativeOut ? nativeOutput : output,
                                        geometry, (long) start, (long) end);
                                if (nativeOut) out.copyFrom(nativeOutput);
                                assertEquals(-1L, in.mismatch(nativeInput), "input stays untouched");
                                for (int p = 0; p < span; p++) {
                                    int i = (p - 5) / stride;
                                    if (p < 5 || (p - 5) % stride != 0 || i < start || i >= end) {
                                        assertEquals(-71, getTyped(type, output, p), "output canary");
                                        continue;
                                    }
                                    double x = getTyped(type, input, 3 + i * stride);
                                    double expected = kind == CpuPointwiseOpcode.ERF ? CpuActivationDirectOracle.erf(x)
                                            : type == DataType.BFLOAT16 || type == DataType.FLOAT16
                                            ? CpuActivationDirectOracle.gelu((float) x)
                                            : x == Double.NEGATIVE_INFINITY ? -0d
                                            : 0.5 * x * (1 + CpuActivationDirectOracle.erf(x / Math.sqrt(2)));
                                    Object expectedArray = array(type, 1); putTyped(type, expectedArray, 0, expected);
                                    expected = getTyped(type, expectedArray, 0);
                                    double actual = getTyped(type, output, p);
                                    if (Double.isNaN(expected)) assertTrue(Double.isNaN(actual));
                                    else assertBits(expected, actual, type + " " + kind + " physical=" + p);
                                }
                            }
                        }
    }

    @Test void classFilesExposeFloatSitesAndNoScalarHelperAllocationOrDynamicDispatch() throws Exception {
        Path root = Path.of("build/cpu-0022-classes");
        Files.createDirectories(root);
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (CpuPointwiseOpcode kind : COMPOSITES) {
                Fixture fixture = fixture(type, kind, ExecutionStrategy.SCALAR, false, false, false);
                byte[] bytes = fixture.kernel().classBytes();
                Files.write(root.resolve(type + "-" + kind + ".class"), bytes);
                var model = ClassFile.of().parse(bytes);
                assertTrue(ClassFile.of().verify(bytes).isEmpty());
                assertEquals(1, model.methods().size()); assertTrue(model.fields().isEmpty());
                assertEquals(0, model.constantPool().bootstrapMethodCount());
                var code = model.methods().getFirst().code().orElseThrow();
                List<Opcode> opcodes = code.elementStream().filter(Instruction.class::isInstance)
                        .map(Instruction.class::cast).map(Instruction::opcode).toList();
                for (Opcode forbidden : List.of(Opcode.NEW, Opcode.NEWARRAY, Opcode.ANEWARRAY,
                        Opcode.MULTIANEWARRAY, Opcode.INVOKEDYNAMIC, Opcode.MONITORENTER))
                    assertFalse(opcodes.contains(forbidden), kind + " " + forbidden);
                assertTrue(opcodes.contains(Opcode.D2F), kind + " elementary result rounding");
                if (kind != CpuPointwiseOpcode.GELU_TANH_APPROXIMATION)
                    assertTrue(opcodes.contains(Opcode.FDIV), kind + " named FLOAT32 division");
                if (kind != CpuPointwiseOpcode.RSQRT) assertTrue(opcodes.contains(Opcode.FADD));
                if (kind == CpuPointwiseOpcode.GELU_EXACT || kind == CpuPointwiseOpcode.GELU_TANH_APPROXIMATION
                        || kind == CpuPointwiseOpcode.SILU) assertTrue(opcodes.contains(Opcode.FMUL));
                if (kind != CpuPointwiseOpcode.GELU_EXACT)
                    for (Opcode doubleArithmetic : List.of(Opcode.DADD, Opcode.DMUL, Opcode.DDIV))
                        assertFalse(opcodes.contains(doubleArithmetic), kind + " no double composite work");
                var members = StreamSupport.stream(model.constantPool().spliterator(), false)
                        .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
                assertTrue(members.stream().allMatch(member -> List.of("java/lang/Float", "java/lang/Double",
                        "java/lang/Math", "java/lang/StrictMath").contains(member.owner().asInternalName())));
                assertTrue(members.stream().noneMatch(member -> member.name().stringValue().equals("valueOf")));
            }
    }

    @Test void erfVectorHelpersBoundInputsBeforeMultiplicationAndKeepChunkOnlyCalls() throws Exception {
        Path root = Path.of("build/cpu-0022-classes");
        Files.createDirectories(root);
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (CpuPointwiseOpcode kind : List.of(CpuPointwiseOpcode.ERF, CpuPointwiseOpcode.GELU_EXACT))
                for (ExecutionStrategy strategy : List.of(ExecutionStrategy.SCALAR, ExecutionStrategy.VECTOR)) {
                    byte[] bytes = fixture(type, kind, strategy, false, false, false).kernel().classBytes();
                    Files.write(root.resolve(type + "-" + kind + "-" + strategy + ".class"), bytes);
                    var model = ClassFile.of().parse(bytes);
                    assertTrue(ClassFile.of().verify(bytes).isEmpty());
                    assertEquals(1, model.methods().size());
                    assertTrue(model.fields().isEmpty());
                    assertEquals(0, model.constantPool().bootstrapMethodCount());
                    for (var element : model.methods().getFirst().code().orElseThrow()) {
                        if (element instanceof Instruction instruction) assertFalse(List.of(Opcode.NEW,
                                Opcode.NEWARRAY, Opcode.ANEWARRAY, Opcode.MULTIANEWARRAY,
                                Opcode.INVOKEDYNAMIC, Opcode.MONITORENTER).contains(instruction.opcode()));
                        if (element instanceof InvokeInstruction call
                                && call.owner().asInternalName().startsWith("io/github/pho001/synaptik")) {
                            assertEquals(ExecutionStrategy.VECTOR, strategy);
                            assertEquals("io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath",
                                    call.owner().asInternalName());
                            assertTrue(call.type().stringValue().contains("Vector;"));
                        }
                    }
                }
        try (var stream = CpuVectorMath.class.getResourceAsStream("CpuVectorMath.class")) {
            var model = ClassFile.of().parse(stream.readAllBytes());
            for (var method : model.methods()) if (method.methodName().stringValue().equals("erf")) {
                var code = method.code().orElseThrow();
                var calls = code.elementStream().filter(InvokeInstruction.class::isInstance)
                        .map(InvokeInstruction.class::cast).map(call -> call.name().stringValue()).toList();
                assertTrue(calls.indexOf("min") >= 0 && calls.indexOf("min") < calls.indexOf("mul"));
                assertEquals(2, calls.stream().filter("polevl"::equals).count());
                assertEquals(2, calls.stream().filter("p1evl"::equals).count());
                for (var element : code) if (element instanceof Instruction instruction)
                    assertFalse(List.of(Opcode.NEW, Opcode.NEWARRAY, Opcode.ANEWARRAY,
                            Opcode.MULTIANEWARRAY, Opcode.INVOKEDYNAMIC).contains(instruction.opcode()));
            }
        }
    }

    static Fixture fixture(DataType type, CpuPointwiseOpcode kind, ExecutionStrategy strategy,
            boolean nativeIn, boolean nativeOut, boolean strided) {
        CpuKernelIr ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, access(false, strided)),
                new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.OUTPUT, access(true, strided))),
                List.of(new CpuKernelIr.Instruction(kind, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
        var heap = type == DataType.FLOAT64 ? CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
        var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                strategy, List.of(type, type),
                List.of(nativeIn ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : heap,
                        nativeOut ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : heap),
                strategy == ExecutionStrategy.VECTOR ? lanes(type) * type.byteWidth() * 8 : 0,
                -1, List.of(), false, type == DataType.FLOAT16 ? 68 : type == DataType.BFLOAT16 ? 59 : 52);
        var generator = new CpuClassFileKernelGenerator();
        return new Fixture(ir, generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir)));
    }

    static long[] geometry(int count, int start, int inBase, int outBase, int stride, int span) {
        return new long[] {count, start, inBase + (long) start * stride,
                outBase + (long) start * stride, stride, stride, start, start, span, span};
    }

    private static CpuAccessPlan access(boolean output, boolean strided) {
        return new CpuAccessPlan(output ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ,
                strided ? CpuAccessPlan.Regime.GENERAL_ODOMETER : CpuAccessPlan.Regime.DENSE_LINEAR,
                1, List.of(strided ? CpuAccessPlan.AxisRole.STRIDED : CpuAccessPlan.AxisRole.CONTIGUOUS),
                strided ? 0 : 1);
    }

    private static int lanes(DataType type) {
        return type == DataType.FLOAT64 ? DoubleVector.SPECIES_PREFERRED.length() : FloatVector.SPECIES_PREFERRED.length();
    }
    private static Object array(DataType type, int count) {
        return type == DataType.FLOAT64 ? new double[count] : type == DataType.FLOAT32 ? new float[count] : new short[count];
    }
    private static void put(Object array, int i, double x) {
        if (array instanceof double[] doubles) doubles[i] = x; else ((float[]) array)[i] = (float) x;
    }
    private static double get(Object array, int i) {
        return array instanceof double[] doubles ? doubles[i] : ((float[]) array)[i];
    }
    private static void putTyped(DataType type, Object array, int i, double x) {
        if (array instanceof short[] shorts) shorts[i] = CpuActivationDirectOracle.encode(type, (float) x);
        else put(array, i, x);
    }
    private static double getTyped(DataType type, Object array, int i) {
        return array instanceof short[] shorts ? CpuActivationDirectOracle.decode(type, shorts[i]) : get(array, i);
    }
    private static MemorySegment segment(Object array) {
        return array instanceof double[] doubles ? MemorySegment.ofArray(doubles)
                : array instanceof float[] floats ? MemorySegment.ofArray(floats) : MemorySegment.ofArray((short[]) array);
    }
    private static void assertBits(double expected, double actual, String label) {
        assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual), label);
    }
    record Fixture(CpuKernelIr ir, CpuGeneratedKernel kernel) {}
}

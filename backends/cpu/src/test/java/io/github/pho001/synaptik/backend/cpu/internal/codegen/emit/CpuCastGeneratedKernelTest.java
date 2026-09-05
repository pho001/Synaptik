package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuWorkerGroup;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.classfile.ClassFile;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.ByteOrder;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Direct typed-entry coverage for every generated CPU CAST carrier and access form. */
class CpuCastGeneratedKernelTest {
    private static final ValueLayout.OfDouble DOUBLE = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder());
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder());

    @Test void crossTypeCastRejectsAVectorSpecialization() {
        CpuKernelIr ir = ir(DataType.BOOL, DataType.FLOAT32, Form.DENSE);
        assertThrows(IllegalArgumentException.class, () -> new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()), CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, List.of(DataType.BOOL, DataType.FLOAT32),
                List.of(carrier(DataType.BOOL, false), carrier(DataType.FLOAT32, false)), 0, 0, List.of(), false, 60));
    }

    @Test void directF64ToBfloat16KeepsRawDoubleRoundingAndTieBoundaries() throws Throwable {
        // The first two values lie just beyond a BF16 halfway point but collapse to that tie when
        // first rounded to F32.  They therefore prove this entry does not double-round via F32.
        double[] input = {1.0039062501d, -1.0039062501d, 1.00390625d,
                Double.longBitsToDouble(0xfff0000000000042L)};
        short[] output = new short[4];
        entry(DataType.FLOAT64, DataType.BFLOAT16, CarrierForm.ARRAY_ARRAY, Form.DENSE)
                .invokeWithArguments(input, output, geometry(Form.DENSE), 0L, 4L);
        assertEquals(0x3f81, Short.toUnsignedInt(output[0]), "positive direct double-rounding counterexample");
        assertEquals(0xbf81, Short.toUnsignedInt(output[1]), "negative direct double-rounding counterexample");
        assertEquals(0x3f80, Short.toUnsignedInt(output[2]), "ties round to even");
        assertEquals(0x7fc0, Short.toUnsignedInt(output[3]), "NaNs are canonical target NaNs");
    }

    @Test void denseArrayF64ToBfloat16KeepsTheCompactSingleEntryControlFlow() {
        CpuKernelIr ir = ir(DataType.FLOAT64, DataType.BFLOAT16, Form.DENSE);
        var specification = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT64, DataType.BFLOAT16),
                List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY),
                0, -1, List.of(), false, 60);
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(specification, ir);
        var generated = ClassFile.of().parse(bytes);
        assertEquals(1, generated.methods().size(), "generated class retains one typed static entry");
        var code = generated.methods().getFirst().attributes().stream()
                .filter(java.lang.classfile.attribute.CodeAttribute.class::isInstance)
                .map(java.lang.classfile.attribute.CodeAttribute.class::cast).findFirst().orElseThrow();
        assertEquals(287, code.codeLength(), "compact javac-equivalent post-inline control flow");
        assertEquals(30, code.maxLocals(), "compact conversion local footprint");
    }

    @Test void directCastKeepsSaturationModuloAndTruthConstants() throws Throwable {
        long[] integers = {0x1_0000_0001L, -1L, Long.MIN_VALUE, Long.MAX_VALUE};
        int[] narrowed = new int[4];
        entry(DataType.INT64, DataType.INT32, CarrierForm.ARRAY_ARRAY, Form.DENSE)
                .invokeWithArguments(integers, narrowed, geometry(Form.DENSE), 0L, 4L);
        assertEquals(1, narrowed[0], "low-word modulo narrowing");
        assertEquals(-1, narrowed[1], "negative modulo narrowing");
        double[] floating = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 3.9d};
        int[] saturated = new int[4];
        entry(DataType.FLOAT64, DataType.INT32, CarrierForm.ARRAY_ARRAY, Form.DENSE)
                .invokeWithArguments(floating, saturated, geometry(Form.DENSE), 0L, 4L);
        assertEquals(0, saturated[0], "NaN integral constant");
        assertEquals(Integer.MAX_VALUE, saturated[1], "positive saturation");
        assertEquals(Integer.MIN_VALUE, saturated[2], "negative saturation");
        assertEquals(3, saturated[3], "truncation toward zero");
        double[] truthInput = {-0.0d, 0.0d, Double.NaN, -2.0d}; byte[] truth = new byte[4];
        entry(DataType.FLOAT64, DataType.BOOL, CarrierForm.ARRAY_ARRAY, Form.DENSE)
                .invokeWithArguments(truthInput, truth, geometry(Form.DENSE), 0L, 4L);
        assertEquals(0, truth[0], "negative zero is false");
        assertEquals(0, truth[1], "positive zero is false");
        assertEquals(1, truth[2], "NaN is true");
        assertEquals(1, truth[3], "nonzero is true");
    }

    @Test void boundedTypedCastChainAndFanoutKeepEachIntermediateBoundary() throws Throwable {
        CpuAccessPlan denseRead = plan(CpuAccessPlan.AccessKind.READ, Form.DENSE);
        CpuAccessPlan denseWrite = plan(CpuAccessPlan.AccessKind.WRITE, Form.DENSE);
        CpuKernelIr chain = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(1, DataType.FLOAT32, CpuKernelIr.Value.Kind.VIRTUAL, denseWrite),
                new CpuKernelIr.Value(2, DataType.BFLOAT16, CpuKernelIr.Value.Kind.OUTPUT, denseWrite),
                new CpuKernelIr.Value(3, DataType.INT32, CpuKernelIr.Value.Kind.OUTPUT, denseWrite)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST, List.of(0), 1),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST, List.of(1), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST, List.of(1), 3)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(2, 0),
                        new CpuKernelIr.Store(3, 1)));
        var specification = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(chain.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT64, DataType.BFLOAT16, DataType.INT32),
                List.of(carrier(DataType.FLOAT64, false), carrier(DataType.BFLOAT16, false),
                        carrier(DataType.INT32, false)), 0, -1, List.of(), false, 60);
        var generator = new CpuClassFileKernelGenerator();
        MethodHandle entry = generator.defineClassBytes(specification,
                generator.generateClassBytes(specification, chain)).entryPoint();
        double[] input = {1.0039062501d, -1.0039062501d, 3.9d, Double.NaN};
        short[] bf16 = new short[4]; int[] integers = new int[4];
        entry.invokeWithArguments(input, bf16, integers, new long[] {4, 0, 0, 0, 0, 1, 1, 1, 4, 4, 4},
                0L, 4L);
        assertEquals(0x3f80, Short.toUnsignedInt(bf16[0]), "F64 -> F32 -> BF16 boundary");
        assertEquals(0xbf80, Short.toUnsignedInt(bf16[1]), "negative chain boundary");
        assertEquals(3, integers[2], "fan-out consumes the F32 intermediate");
        assertEquals(0, integers[3], "fan-out keeps the typed NaN-to-integral rule");
    }

    @Test void nonzeroSubrangesPreserveSentinelsAndDoNotLeakPriorRunGeometry() throws Throwable {
        MethodHandle entry = entry(DataType.FLOAT64, DataType.INT32, CarrierForm.ARRAY_ARRAY, Form.BLOCK);
        double[] first = new double[16]; int[] firstOutput = {77, 77, 77, 77, 77, 77, 77, 77,
                77, 77, 77, 77, 77, 77, 77, 77};
        long[] whole = geometry(Form.BLOCK);
        for (int ordinal = 0; ordinal < 4; ordinal++) first[(int) address(whole, Form.BLOCK, 0, ordinal)] =
                10 * (ordinal + 1);
        entry.invokeWithArguments(first, firstOutput, geometryAtRangeStart(whole, Form.BLOCK, 1L), 1L, 3L);
        assertEquals(77, firstOutput[(int) address(whole, Form.BLOCK, 1, 0)], "lower sentinel");
        assertEquals(20, firstOutput[(int) address(whole, Form.BLOCK, 1, 1)], "first subrange value");
        assertEquals(30, firstOutput[(int) address(whole, Form.BLOCK, 1, 2)], "second subrange value");
        assertEquals(77, firstOutput[(int) address(whole, Form.BLOCK, 1, 3)], "upper sentinel");
        double[] second = new double[16]; int[] secondOutput = new int[16];
        for (int ordinal = 0; ordinal < 4; ordinal++) second[(int) address(whole, Form.BLOCK, 0, ordinal)] =
                90 - 10 * ordinal;
        java.util.Arrays.fill(secondOutput, 55);
        entry.invokeWithArguments(second, secondOutput, geometryAtRangeStart(whole, Form.BLOCK, 2L), 2L, 4L);
        assertEquals(55, secondOutput[(int) address(whole, Form.BLOCK, 1, 1)], "second-run lower sentinel");
        assertEquals(70, secondOutput[(int) address(whole, Form.BLOCK, 1, 2)], "second-run start");
        assertEquals(60, secondOutput[(int) address(whole, Form.BLOCK, 1, 3)], "second-run end");
    }

    @Test void generatedTypedCastMatrixCoversAll1296LegalCells() throws Throwable {
        int cells = 0;
        for (DataType source : DataType.values()) for (DataType target : DataType.values())
            for (CarrierForm carriers : CarrierForm.values()) {
                for (Form form : Form.MULTI) {
                    cell(source, target, carriers, form, false);
                    cell(source, target, carriers, form, true);
                    cells += 2;
                }
                cell(source, target, carriers, Form.SCALAR, false);
                cells++;
            }
        assertEquals(1_296, cells);
    }

    private static void cell(DataType source, DataType target, CarrierForm forms, Form form,
            boolean parallel) throws Throwable {
        // Array-backed segments classify as arrays. Native allocation makes the segment cases
        // exercise the MEMORY_SEGMENT carrier. Worker calls need a shared, not confined, arena.
        try (Arena arena = forms.segment() && parallel ? Arena.ofShared() : Arena.ofConfined()) {
            long[] geometry = geometry(form);
            int capacity = capacity(form);
            Object input = storage(source, forms.inputSegment, capacity, arena);
            Object output = storage(target, forms.outputSegment, capacity, arena);
            ScalarValue[] vector = values(source);
            fill(output, target, capacity, sentinel(target));
            for (int i = 0; i < form.count; i++) write(input, source, address(geometry, form, 0, i), vector[i % vector.length]);
            MethodHandle entry = entry(source, target, forms, form);
            if (parallel) invokeWorkerRanges(entry, input, output, geometry, form, form.count);
            else entry.invokeWithArguments(input, output, geometry, 0L, (long) form.count);
            for (int i = 0; i < form.count; i++) {
                ScalarValue expected = CastValueConversions.convert(vector[i % vector.length], target);
                assertEquals(bits(expected), bits(read(output, target, address(geometry, form, 1, i))),
                        (parallel ? "parallel" : "scalar") + " " + source + " -> " + target
                                + " " + forms + " " + form + " at " + i);
            }
            for (int i = 0; i < capacity; i++) if (!outputAddress(geometry, form, i))
                assertEquals(bits(sentinel(target)), bits(read(output, target, i)), "sentinel " + i);
        }
    }

    /*
     * CpuPreparedExecutable is the production owner of this package-private worker entry.
     * Direct generated-entry evidence cannot otherwise submit disjoint ranges without widening
     * that production API. This invokes the same CpuWorkerGroup and waits for its normal join.
     */
    private static void invokeWorkerRanges(MethodHandle entry, Object input, Object output, long[] geometry,
            Form form, int count) throws Throwable {
        try (CpuWorkerGroup group = new CpuWorkerGroup(2)) {
            Method execute = CpuWorkerGroup.class.getDeclaredMethod("execute",
                    Class.forName(CpuWorkerGroup.class.getName() + "$RangeCall").arrayType());
            execute.setAccessible(true);
            Class<?> call = execute.getParameterTypes()[0].componentType();
            Object ranges = Array.newInstance(call, 2);
            for (int part = 0; part < 2; part++) {
                long start = part == 0 ? 0 : count / 2, end = part == 0 ? count / 2 : count;
                Array.set(ranges, part, Proxy.newProxyInstance(call.getClassLoader(), new Class<?>[] {call},
                        (proxy, method, arguments) -> {
                            if (!method.getName().equals("invoke")) throw new AssertionError(method);
                            entry.invokeWithArguments(input, output,
                                    geometryAtRangeStart(geometry, form, start), start, end);
                            return null;
                        }));
            }
            try { execute.invoke(group, ranges); }
            catch (InvocationTargetException failure) { throw failure.getCause(); }
        }
    }

    /**
     * Gives a direct generated worker the same already-positioned cold state that production
     * binding gives it.  The generated entry deliberately treats {@code start}/{@code end} as
     * absolute loop bounds; it does not seek a whole-range geometry snapshot to {@code start}.
     */
    private static long[] geometryAtRangeStart(long[] wholeRange, Form form, long start) {
        long[] snapshot = wholeRange.clone();
        if (form == Form.SCALAR) return snapshot;
        int rank = form == Form.DENSE || form == Form.OFFSET ? 1 : 2;
        int boundaryCount = 2;
        for (int axis = 0; axis < rank; axis++) {
            long coordinate = rank == 1 ? start : axis == 0 ? start / 4 : start % 4;
            snapshot[rank + axis] = coordinate;
        }
        for (int boundary = 0; boundary < boundaryCount; boundary++)
            snapshot[2 * rank + boundary] = address(wholeRange, form, boundary,
                    Math.toIntExact(start));
        if (form == Form.BLOCK) {
            int innerBase = 2 * rank + boundaryCount + boundaryCount * rank;
            for (int boundary = 0; boundary < boundaryCount; boundary++)
                snapshot[innerBase + boundary] = start % 4;
        }
        return snapshot;
    }

    private static MethodHandle entry(DataType source, DataType target, CarrierForm forms, Form form) {
        CpuKernelIr ir = ir(source, target, form);
        var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT, CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(source, target), List.of(carrier(source, forms.inputSegment), carrier(target, forms.outputSegment)),
                0, -1, List.of(), false, source == target ? source == DataType.BFLOAT16 ? 59 : 52 : 60);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(spec, generator.generateClassBytes(spec, ir)).entryPoint();
    }

    private static CpuKernelIr ir(DataType source, DataType target, Form form) {
        return new CpuKernelIr(List.of(new CpuKernelIr.Value(0, source, CpuKernelIr.Value.Kind.INPUT, plan(CpuAccessPlan.AccessKind.READ, form)),
                new CpuKernelIr.Value(1, target, CpuKernelIr.Value.Kind.OUTPUT, plan(CpuAccessPlan.AccessKind.WRITE, form))),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
    }
    private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind, Form form) {
        return switch (form) {
            case DENSE, OFFSET -> new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case BLOCK -> new CpuAccessPlan(kind, CpuAccessPlan.Regime.BLOCK_OUTER, 2, List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
            case GENERAL -> new CpuAccessPlan(kind, CpuAccessPlan.Regime.GENERAL_ODOMETER, 2, List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.STRIDED), 0);
            case SCALAR -> new CpuAccessPlan(kind, kind == CpuAccessPlan.AccessKind.READ ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO : CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0);
        };
    }
    private static long[] geometry(Form form) { return switch (form) {
        case DENSE -> new long[] {4, 0, 0, 0, 1, 1, 4, 4};
        case OFFSET -> new long[] {4, 0, 3, 5, 1, 1, 4, 4};
        case BLOCK -> new long[] {1, 4, 0, 0, 1, 2, 8, 1, 9, 1, 0, 0, 4, 4};
        case GENERAL -> new long[] {1, 4, 0, 0, 1, 2, 9, 2, 11, 2};
        case SCALAR -> new long[] {1, 2};
    }; }
    private static long address(long[] g, Form f, int boundary, int ordinal) {
        if (f == Form.SCALAR) return g[boundary];
        if (f == Form.DENSE || f == Form.OFFSET) return g[2 + boundary] + ordinal;
        int row = ordinal / 4, column = ordinal % 4;
        return g[4 + boundary] + row * g[6 + 2 * boundary] + column * g[7 + 2 * boundary];
    }
    private static boolean outputAddress(long[] g, Form f, int address) {
        for (int i = 0; i < f.count; i++) if (address(g, f, 1, i) == address) return true;
        return false;
    }
    private static int capacity(Form form) { return switch (form) { case DENSE -> 10; case OFFSET -> 14; case BLOCK -> 16; case GENERAL -> 20; case SCALAR -> 4; }; }
    private static CpuKernelSpecialization.CarrierAccess carrier(DataType t, boolean segment) {
        if (segment) return CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        return switch (t) { case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY; case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY; case BFLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY; case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY; case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY; case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY; };
    }
    private static Object storage(DataType t, boolean segment, int n, Arena arena) {
        if (segment) return arena.allocate((long) n * t.byteWidth(), t.byteWidth());
        return switch (t) { case FLOAT64 -> new double[n]; case FLOAT32 -> new float[n]; case BFLOAT16 -> new short[n]; case INT64 -> new long[n]; case INT32 -> new int[n]; case BOOL -> new byte[n]; };
    }
    private static void fill(Object a, DataType t, int n, ScalarValue v) { for (int i = 0; i < n; i++) write(a, t, i, v); }
    private static void write(Object a, DataType t, long i, ScalarValue v) { switch (t) {
        case FLOAT64 -> { if (a instanceof MemorySegment s) s.setAtIndex(DOUBLE, i, v.float64Value()); else ((double[]) a)[(int) i] = v.float64Value(); }
        case FLOAT32 -> { if (a instanceof MemorySegment s) s.setAtIndex(FLOAT, i, v.float32Value()); else ((float[]) a)[(int) i] = v.float32Value(); }
        case BFLOAT16 -> { if (a instanceof MemorySegment s) s.setAtIndex(SHORT, i, v.bfloat16Bits()); else ((short[]) a)[(int) i] = v.bfloat16Bits(); }
        case INT64 -> { if (a instanceof MemorySegment s) s.setAtIndex(LONG, i, v.int64Value()); else ((long[]) a)[(int) i] = v.int64Value(); }
        case INT32 -> { if (a instanceof MemorySegment s) s.setAtIndex(INT, i, v.int32Value()); else ((int[]) a)[(int) i] = v.int32Value(); }
        case BOOL -> { byte b = (byte) (v.booleanValue() ? 1 : 0); if (a instanceof MemorySegment s) s.setAtIndex(ValueLayout.JAVA_BYTE, i, b); else ((byte[]) a)[(int) i] = b; }
    }; }
    private static ScalarValue read(Object a, DataType t, long i) { return switch (t) {
        case FLOAT64 -> ScalarValue.float64(a instanceof MemorySegment s ? s.getAtIndex(DOUBLE, i) : ((double[]) a)[(int) i]);
        case FLOAT32 -> ScalarValue.float32(a instanceof MemorySegment s ? s.getAtIndex(FLOAT, i) : ((float[]) a)[(int) i]);
        case BFLOAT16 -> ScalarValue.bfloat16Bits(a instanceof MemorySegment s ? s.getAtIndex(SHORT, i) : ((short[]) a)[(int) i]);
        case INT64 -> ScalarValue.int64(a instanceof MemorySegment s ? s.getAtIndex(LONG, i) : ((long[]) a)[(int) i]);
        case INT32 -> ScalarValue.int32(a instanceof MemorySegment s ? s.getAtIndex(INT, i) : ((int[]) a)[(int) i]);
        case BOOL -> ScalarValue.bool((a instanceof MemorySegment s ? s.getAtIndex(ValueLayout.JAVA_BYTE, i) : ((byte[]) a)[(int) i]) != 0);
    }; }
    private static long bits(ScalarValue v) { return switch (v.dataType()) {
        case FLOAT64 -> Double.doubleToRawLongBits(v.float64Value()); case FLOAT32 -> Integer.toUnsignedLong(Float.floatToRawIntBits(v.float32Value())); case BFLOAT16 -> Short.toUnsignedLong(v.bfloat16Bits()); case INT64 -> v.int64Value(); case INT32 -> Integer.toUnsignedLong(v.int32Value()); case BOOL -> v.booleanValue() ? 1 : 0;
    }; }
    private static ScalarValue sentinel(DataType t) { return switch (t) {
        case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(0x7ff80000000000a5L)); case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0x7fc000a5)); case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x7fc5); case INT64 -> ScalarValue.int64(0x5a5a5a5a5a5a5a5aL); case INT32 -> ScalarValue.int32(0x5a5a5a5a); case BOOL -> ScalarValue.bool(true);
    }; }
    private static ScalarValue[] values(DataType t) { return switch (t) {
        case FLOAT64 -> new ScalarValue[] {ScalarValue.float64(-0d), ScalarValue.float64(Double.MIN_VALUE), ScalarValue.float64(Double.longBitsToDouble(0x7ff0000000000042L)), ScalarValue.float64(Double.NEGATIVE_INFINITY), ScalarValue.float64(Double.MAX_VALUE), ScalarValue.float64(2_147_483_647.75d), ScalarValue.float64(-2_147_483_648.75d), ScalarValue.float64(1.0d + 0x1.0p-8)};
        case FLOAT32 -> new ScalarValue[] {ScalarValue.float32(-0f), ScalarValue.float32(Float.MIN_VALUE), ScalarValue.float32(Float.intBitsToFloat(0x7fa12345)), ScalarValue.float32(Float.NEGATIVE_INFINITY), ScalarValue.float32(Float.MAX_VALUE), ScalarValue.float32(2_147_483_647f), ScalarValue.float32(-2_147_483_648f), ScalarValue.float32(1.0f + 0x1.0p-8f)};
        case BFLOAT16 -> new ScalarValue[] {ScalarValue.bfloat16Bits((short) 0x8000), ScalarValue.bfloat16Bits((short) 1), ScalarValue.bfloat16Bits((short) 0x7f81), ScalarValue.bfloat16Bits((short) 0xff80), ScalarValue.bfloat16Bits((short) 0x7f7f), ScalarValue.bfloat16Bits((short) 0x4f00), ScalarValue.bfloat16Bits((short) 0xcf00), ScalarValue.bfloat16Bits((short) 0x3f81)};
        case INT64 -> new ScalarValue[] {ScalarValue.int64(0), ScalarValue.int64(1), ScalarValue.int64(-1), ScalarValue.int64(Long.MIN_VALUE), ScalarValue.int64(Long.MAX_VALUE), ScalarValue.int64(2_155_872_257L), ScalarValue.int64(-2_155_872_257L), ScalarValue.int64(16_777_217L)};
        case INT32 -> new ScalarValue[] {ScalarValue.int32(0), ScalarValue.int32(1), ScalarValue.int32(-1), ScalarValue.int32(Integer.MIN_VALUE), ScalarValue.int32(Integer.MAX_VALUE), ScalarValue.int32(0x80000001), ScalarValue.int32(0x7fffffff), ScalarValue.int32(16_777_217)};
        case BOOL -> new ScalarValue[] {ScalarValue.bool(false), ScalarValue.bool(true), ScalarValue.bool(false), ScalarValue.bool(true), ScalarValue.bool(false), ScalarValue.bool(true), ScalarValue.bool(false), ScalarValue.bool(true)};
    }; }
    private enum CarrierForm { ARRAY_ARRAY(false, false), ARRAY_SEGMENT(false, true), SEGMENT_ARRAY(true, false), SEGMENT_SEGMENT(true, true); final boolean inputSegment, outputSegment; CarrierForm(boolean inputSegment, boolean outputSegment) { this.inputSegment = inputSegment; this.outputSegment = outputSegment; } boolean segment() { return inputSegment || outputSegment; } }
    private enum Form { DENSE(4), OFFSET(4), BLOCK(4), GENERAL(4), SCALAR(1); static final Form[] MULTI = {DENSE, OFFSET, BLOCK, GENERAL}; final int count; Form(int count) { this.count = count; } }
}

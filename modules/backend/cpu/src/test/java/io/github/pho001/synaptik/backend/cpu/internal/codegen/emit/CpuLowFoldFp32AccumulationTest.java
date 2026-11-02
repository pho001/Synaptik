package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.*;
import java.lang.classfile.instruction.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Exact low-fold numeric, layout, range and generated-dataflow regressions; no timing. */
class CpuLowFoldFp32AccumulationTest {
    @Test void roundingCancellationOverflowAndSpecialClassesUseOneFinalNarrowing() throws Throwable {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            float half = Math.scalb(1.0f, type == DataType.BFLOAT16 ? -8 : -11);
            float maximum = decode(type, (short) (type == DataType.BFLOAT16 ? 0x7f7f : 0x7bff));
            float min = decode(type, (short) 1);
            float minNormal = decode(type, (short) (type == DataType.BFLOAT16 ? 0x0080 : 0x0400));
            float huge = type == DataType.BFLOAT16 ? Math.scalb(1.0f, 100) : 32768f;
            float overflowTie = type == DataType.BFLOAT16 ? Math.scalb(1.0f, 119) : 16f;
            float[][] cases = {
                {1, half, half}, {1, half, 0}, {1 + 2 * half, half, 0},
                {huge, -huge, 1}, {huge, min, -huge}, {maximum, overflowTie, -overflowTie},
                {maximum, maximum, -maximum}, {maximum, maximum, 0},
                {Float.NaN, 1, 2}, {Float.POSITIVE_INFINITY, 1, 2},
                {Float.NEGATIVE_INFINITY, -1, -2},
                {Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1},
                {+0.0f, -0.0f, -0.0f}, {-0.0f, -0.0f, -0.0f},
                {min, min, min}, {minNormal, -minNormal, min}, {min, -min, -0.0f}
            };
            for (boolean image : List.of(false, true)) for (int carriers = 0; carriers < 4; carriers++) {
                var context = small(type, image, carriers);
                if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(context)) continue;
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                inspect(plan, type);
                for (int i = 0; i < cases.length; i++) {
                    short[] input = new short[9];
                    input[2] = encode(type, cases[i][0]); input[4] = encode(type, cases[i][1]);
                    input[6] = encode(type, cases[i][2]);
                    short[] expected = new short[5], actual = new short[5];
                    oracle(plan, type, input, expected, 0, 5);
                    execute(plan, input, actual, 0, 5, false);
                    same(expected, actual, type);
                    cold(plan, type, input, actual, 0, 5);
                    same(expected, actual, type);
                    if (i == 0) assertEquals((short) (type == DataType.BFLOAT16 ? 0x3f81 : 0x3c01), actual[2]);
                    if (i == 1) assertEquals(encode(type, 1), actual[2], "even final tie");
                    if (i == 2) assertEquals(encode(type, 1 + 4 * half), actual[2], "odd final tie");
                    if (i == 3) assertEquals(encode(type, 1), actual[2], "canonical cancellation order");
                    if (i == 4) assertEquals((short) 0, actual[2], "FP32, not double accumulation");
                    if (i == 5) assertEquals(encode(type, maximum), actual[2], "no premature low overflow");
                    if (i == 6) assertEquals(encode(type, type == DataType.BFLOAT16 ? Float.POSITIVE_INFINITY : maximum), actual[2]);
                    if (i == 12 || i == 13 || i == 16) assertEquals((short) 0, actual[2], "+0 initialization");
                    if (i == 14) assertEquals((short) 3, actual[2], "stored subnormal decode is exact");
                }
            }
        }
    }

    @Test void primitiveOraclesAlsoQualifyFinalUnderflowAndSignedTies() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            float unit = decode(type, (short) 1);
            assertEquals((short) 0, encode(type, unit / 2));
            assertEquals((short) 0x8000, encode(type, -unit / 2));
            assertEquals((short) 2, encode(type, 1.5f * unit));
            // A fold only adds exact low inputs: final half-subnormal ties are not reachable
            // from such multiples. Test the unchanged conversion boundary independently.
        }
    }

    @Test void bothFamiliesCoverNativeHeapMixedOffsetsStridesRangesAndCanonicalOccurrences() throws Throwable {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (int carriers = 0; carriers < 4; carriers++) for (boolean nativeMemory : List.of(false, true)) {
                for (boolean zeroStride : List.of(false, true)) {
                    Shape in = Shape.of(2, 3, 2, 3), out = Shape.of(2, 5, 2);
                    var context = layoutContext(type, new Operation(WindowTransformKind.FOLD_AXIS,
                            new FoldAxisAttrs(1, 5, 1)), in, out,
                            zeroStride ? new long[]{6, 0, 3, 1} : new long[]{36, 12, 6, 2}, carriers);
                    qualify(context, type, nativeMemory);
                }
                for (boolean ceil : List.of(false, true)) {
                    Shape out = Shape.of(2, 2, 4, 5);
                    var w = new Window2dAttrs(2, 3, 2, 2, 1, 1, 2, 1, ceil);
                    long oh = windows(4, 2, 2, 1, 2, ceil), ow = windows(5, 3, 2, 1, 1, ceil);
                    Shape in = Shape.of(2, 12, oh * ow);
                    for (boolean repeated : List.of(false, true)) qualify(layoutContext(type,
                            new Operation(WindowTransformKind.FOLD2D, new Fold2dAttrs(out, w)), in, out,
                            new long[]{12 * oh * ow * 2, repeated ? 0 : oh * ow * 2, 2}, carriers), type, nativeMemory);
                }
                // Step larger than the window leaves uncovered positive-zero gaps.
                qualify(layoutContext(type, new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 6, 3)), Shape.of(2, 2), Shape.of(6),
                        new long[]{4, 2}, carriers), type, nativeMemory);
            }
        }
    }

    @Test void denseIntBodiesUseNonzeroBasesAndHigherRankAxisWithoutChangingTheAlgorithm() throws Throwable {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            for (boolean higherRank : List.of(false, true)) {
                Shape input = higherRank ? Shape.of(2, 3, 2, 3) : Shape.of(3, 3);
                Shape output = higherRank ? Shape.of(2, 5, 2) : Shape.of(5);
                var in = new TensorDescriptor(type, input, Optional.of(LayoutDescriptor.of(input, LayoutDescriptor.contiguous(input).strides(), 2, true)), false);
                var out = new TensorDescriptor(type, output, Optional.of(LayoutDescriptor.of(output, LayoutDescriptor.contiguous(output).strides(), 3, true)), false);
                var context = carriers(CpuScatterLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(higherRank ? 1 : 0, 5, 1)), List.of(0), List.of(in), out), 0);
                qualify(context, type, false);
            }
        }
    }

    @Test void admittedEmptyInputAndOutputDomainsNeverReadNonexistentContributors() throws Throwable {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (int carriers = 0; carriers < 4; carriers++) {
            var unsupported = CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                    new FoldAxisAttrs(0, 2, 1)), type, Shape.of(0, 3), Shape.of(2));
            var unsupportedCarriers = carriers(unsupported, carriers);
            assertThrows(IllegalArgumentException.class, () -> new CpuPartitionPreparer().analyze(unsupportedCarriers));
            for (var base : List.of(
                    CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                            new FoldAxisAttrs(0, 0, 1)), type, Shape.of(0, 1), Shape.of(0)),
                    CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD2D,
                            new Fold2dAttrs(Shape.of(0, 2, 3, 3), new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false))),
                            type, Shape.of(0, 8, 4), Shape.of(0, 2, 3, 3)))) {
                if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(carriers(base, carriers))) continue;
                var plan = new CpuPartitionPreparer().analyze(carriers(base, carriers)).plan();
                short[] output = {(short) 9, (short) 9};
                execute(plan, new short[0], output, 0, 0, false);
                assertArrayEquals(new short[]{9, 9}, output);
            }
        }
    }

    private static void qualify(PrepareContext<CpuPartitionAnalysisInputs> context, DataType type,
            boolean nativeMemory) throws Throwable {
        if (io.github.pho001.synaptik.backend.cpu.CpuGeneratedCoverageUnsupportedBoundaryTest.assertLowUnsupported(context)) return;
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var g = plan.foldGeometry().orElseThrow();
        short[] input = new short[span(g.input())], expected = new short[span(g.output())];
        for (int i = 0; i < input.length; i++) input[i] = encode(type, (i % 11 - 5) / 8f
                + (i % 3 == 0 ? 1 : Math.scalb(1f, type == DataType.BFLOAT16 ? -8 : -11)));
        Arrays.fill(expected, (short) 0x55aa); short[] actual = expected.clone();
        long count = 1; for (long e : g.output().extents()) count *= e;
        long start = 1, end = count - 1;
        oracle(plan, type, input, expected, start, end);
        execute(plan, input, actual, start, end, nativeMemory);
        same(expected, actual, type);
        short[] cold = new short[actual.length]; Arrays.fill(cold, (short) 0x55aa);
        cold(plan, type, input, cold, start, end); same(expected, cold, type);
        short[] partitioned = new short[actual.length]; Arrays.fill(partitioned, (short) 0x55aa);
        execute(plan, input, partitioned, start, start, nativeMemory);
        short[] untouched = new short[partitioned.length]; Arrays.fill(untouched, (short) 0x55aa);
        assertArrayEquals(untouched, partitioned);
        long middle = (start + end) / 2;
        execute(plan, input, partitioned, start, middle, nativeMemory);
        execute(plan, input, partitioned, middle, end, nativeMemory);
        same(expected, partitioned, type);
        inspect(plan, type);
    }

    private static int span(CpuFoldLowering.Layout l) {
        long span = l.offset() + 1;
        for (int a = 0; a < l.extents().length; a++) span += (l.extents()[a] - 1) * l.strides()[a];
        return Math.toIntExact(span);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> small(DataType type, boolean image, int carriers) {
        Shape output = image ? Shape.of(1, 1, 1, 5) : Shape.of(5);
        Operation op = image ? new Operation(WindowTransformKind.FOLD2D, new Fold2dAttrs(output,
                new Window2dAttrs(1, 3, 1, 1, 0, 0, 1, 1, false)))
                : new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 5, 1));
        return carriers(CpuFoldLoweringTest.context(op, type,
                image ? Shape.of(1, 3, 3) : Shape.of(3, 3), output), carriers);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> layoutContext(DataType type, Operation op,
            Shape input, Shape output, long[] strides, int carriers) {
        var in = new TensorDescriptor(type, input, Optional.of(LayoutDescriptor.of(input, strides, 2, true)), false);
        long[] outStrides = LayoutDescriptor.contiguous(output).strides();
        for (int a = 0; a < outStrides.length; a++) outStrides[a] *= 2;
        var out = new TensorDescriptor(type, output, Optional.of(LayoutDescriptor.of(output, outStrides, 3, true)), false);
        return carriers(CpuScatterLoweringTest.context(op, List.of(0), List.of(in), out), carriers);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> carriers(PrepareContext<CpuPartitionAnalysisInputs> b, int c) {
        return new PrepareContext<>(b.partition(), b.nodes(), b.values(), b.memoryRequirements(), b.constants(),
                new CpuPartitionAnalysisInputs(false, List.of((c & 1) == 0 ? CarrierAccess.SHORT_ARRAY : CarrierAccess.MEMORY_SEGMENT,
                        (c & 2) == 0 ? CarrierAccess.SHORT_ARRAY : CarrierAccess.MEMORY_SEGMENT)));
    }

    private static void execute(CpuPartitionPreparationPlan plan, short[] input, short[] output,
            long start, long end, boolean nativeMemory) throws Throwable {
        var route = plan.units().getFirst().portablePlan(); var generator = new CpuClassFileKernelGenerator();
        var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(), generator.generateClassBytes(route.specialization(), route.kernelIr()));
        try (var arena = Arena.ofConfined()) {
            MemorySegment in = MemorySegment.ofArray(input), out = MemorySegment.ofArray(output);
            if (nativeMemory) {
                in = arena.allocate(input.length * 2L, 2); in.copyFrom(MemorySegment.ofArray(input));
                out = arena.allocate(output.length * 2L, 2); out.copyFrom(MemorySegment.ofArray(output));
            }
            Object i = route.specialization().carrierPattern().getFirst() == CarrierAccess.SHORT_ARRAY ? input : in;
            Object o = route.specialization().carrierPattern().getLast() == CarrierAccess.SHORT_ARRAY ? output : out;
            artifact.entryPoint().invokeWithArguments(i, o, plan.foldGeometry().orElseThrow().pack(new long[2], start, end), start, end);
            if (o instanceof MemorySegment) MemorySegment.ofArray(output).copyFrom(out);
        }
    }

    private static void cold(CpuPartitionPreparationPlan plan, DataType type, short[] input, short[] output, long start, long end) {
        CpuScalarReferenceKernel.execute((CpuFoldIr) plan.units().getFirst().portablePlan().portableKernelIr(),
                plan.foldGeometry().orElseThrow(), List.of(new CpuBufferArgument.Shorts(input, 0, input.length * 2L, true),
                        new CpuBufferArgument.Shorts(output, 0, output.length * 2L, false)), start, end);
    }

    private static void oracle(CpuPartitionPreparationPlan plan, DataType type, short[] input, short[] output, long start, long end) {
        var g = plan.foldGeometry().orElseThrow(); var i = g.input(); var o = g.output();
        if (g.mapping() instanceof CpuFoldLowering.AxisGeometry a) {
            if (type == DataType.BFLOAT16) CpuLowFoldCleanJavaOracle.axisBfloatArrays(input, output, i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), a.axis(), a.step(), start, end);
            else CpuLowFoldCleanJavaOracle.axisHalfArrays(input, output, i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), a.axis(), a.step(), start, end);
            short[] segmentResult = output.clone();
            if (type == DataType.BFLOAT16) CpuLowFoldCleanJavaOracle.axisBfloatSegments(MemorySegment.ofArray(input), MemorySegment.ofArray(segmentResult), i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), a.axis(), a.step(), start, end);
            else CpuLowFoldCleanJavaOracle.axisHalfSegments(MemorySegment.ofArray(input), MemorySegment.ofArray(segmentResult), i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), a.axis(), a.step(), start, end);
            same(output, segmentResult, type);
            if (i.extents().length == 2 && o.extents().length == 1 && i.strides()[1] == 1 && i.strides()[0] == a.windowSize() && o.strides()[0] == 1) {
                short[] dense = output.clone();
                if (type == DataType.BFLOAT16) CpuLowFoldCleanJavaOracle.axisBfloatInt(input, dense, (int) i.offset(), (int) o.offset(), (int) (i.extents()[0] * i.extents()[1]), (int) a.windowSize(), (int) a.step(), (int) start, (int) end);
                else CpuLowFoldCleanJavaOracle.axisHalfInt(input, dense, (int) i.offset(), (int) o.offset(), (int) (i.extents()[0] * i.extents()[1]), (int) a.windowSize(), (int) a.step(), (int) start, (int) end);
                same(output, dense, type);
            }
        } else {
            var m = (CpuFoldLowering.TwoDimensionalGeometry) g.mapping();
            if (type == DataType.BFLOAT16) CpuLowFoldCleanJavaOracle.imageBfloatArrays(input, output, i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), m.kernelHeight(), m.kernelWidth(), m.strideHeight(), m.strideWidth(), m.paddingHeight(), m.paddingWidth(), m.dilationHeight(), m.dilationWidth(), m.outputColumnsHeight(), m.outputColumnsWidth(), start, end);
            else CpuLowFoldCleanJavaOracle.imageHalfArrays(input, output, i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), m.kernelHeight(), m.kernelWidth(), m.strideHeight(), m.strideWidth(), m.paddingHeight(), m.paddingWidth(), m.dilationHeight(), m.dilationWidth(), m.outputColumnsHeight(), m.outputColumnsWidth(), start, end);
            short[] segmentResult = output.clone();
            if (type == DataType.BFLOAT16) CpuLowFoldCleanJavaOracle.imageBfloatSegments(MemorySegment.ofArray(input), MemorySegment.ofArray(segmentResult), i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), m.kernelHeight(), m.kernelWidth(), m.strideHeight(), m.strideWidth(), m.paddingHeight(), m.paddingWidth(), m.dilationHeight(), m.dilationWidth(), m.outputColumnsHeight(), m.outputColumnsWidth(), start, end);
            else CpuLowFoldCleanJavaOracle.imageHalfSegments(MemorySegment.ofArray(input), MemorySegment.ofArray(segmentResult), i.offset(), o.offset(), i.extents(), i.strides(), o.extents(), o.strides(), m.kernelHeight(), m.kernelWidth(), m.strideHeight(), m.strideWidth(), m.paddingHeight(), m.paddingWidth(), m.dilationHeight(), m.dilationWidth(), m.outputColumnsHeight(), m.outputColumnsWidth(), start, end);
            same(output, segmentResult, type);
        }
    }

    private static void inspect(CpuPartitionPreparationPlan plan, DataType type) {
        var r = plan.units().getFirst().portablePlan();
        var code = ClassFile.of().parse(new CpuClassFileKernelGenerator().generateClassBytes(r.specialization(), r.kernelIr())).methods().getFirst().code().orElseThrow();
        var instructions = code.elementStream().filter(Instruction.class::isInstance).map(Instruction.class::cast).toList();
        assertTrue(instructions.stream().anyMatch(i -> i.opcode() == Opcode.FADD));
        assertFalse(instructions.stream().anyMatch(i -> i.opcode() == Opcode.DADD || i.opcode() == Opcode.F2D || i.opcode() == Opcode.D2F
                || i instanceof NewObjectInstruction || i instanceof NewPrimitiveArrayInstruction || i instanceof NewReferenceArrayInstruction || i instanceof NewMultiArrayInstruction || i instanceof TypeCheckInstruction));
        String narrow = type == DataType.BFLOAT16 ? "floatToRawIntBits" : "floatToFloat16";
        String decode = type == DataType.BFLOAT16 ? "intBitsToFloat" : "float16ToFloat";
        var calls = instructions.stream().filter(InvokeInstruction.class::isInstance).map(InvokeInstruction.class::cast).toList();
        assertTrue(calls.stream().allMatch(c -> c.owner().asInternalName().equals("java/lang/Float")
                && Set.of(narrow, decode, "isNaN").contains(c.name().stringValue())
                || c.owner().asInternalName().equals("java/lang/foreign/MemorySegment") && Set.of("get", "set").contains(c.name().stringValue())));
        long adds = instructions.stream().filter(i -> i.opcode() == Opcode.FADD).count();
        assertEquals(adds, calls.stream().filter(c -> c.name().stringValue().equals(narrow)).count());
        assertEquals(adds, calls.stream().filter(c -> c.name().stringValue().equals(decode)).count());
        assertEquals(adds, instructions.stream().filter(i -> i.opcode() == Opcode.SASTORE || i instanceof InvokeInstruction c && c.name().stringValue().equals("set")).count());
        // A backwards contributor-loop edge must follow each add and precede final narrowing.
        List<CodeElement> elements = code.elementStream().toList();
        for (int at = 0; at < elements.size(); at++) if (elements.get(at) instanceof Instruction ins && ins.opcode() == Opcode.FADD) {
            int narrowAt = -1;
            for (int next = at + 1; next < elements.size(); next++) if (elements.get(next) instanceof InvokeInstruction call && call.name().stringValue().equals(narrow)) { narrowAt = next; break; }
            assertTrue(narrowAt > at);
            boolean innerEdge = false;
            for (int edge = at + 1; edge < narrowAt; edge++) if (elements.get(edge) instanceof BranchInstruction branch) {
                int target = -1;
                for (int k = 0; k < at; k++) if (elements.get(k) instanceof LabelTarget label && label.label().equals(branch.target())) target = k;
                if (target >= 0) innerEdge = true;
            }
            assertTrue(innerEdge, "no narrowing in contributor loop");
        }
    }

    private static short encode(DataType t, float v) { return t == DataType.BFLOAT16 ? BFloat16Bits.fromFloat(v) : Float.floatToFloat16(v); }
    private static float decode(DataType t, short v) { return t == DataType.BFLOAT16 ? Float.intBitsToFloat(v << 16) : Float.float16ToFloat(v); }
    private static void same(short[] expected, short[] actual, DataType t) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) if (Float.isNaN(decode(t, expected[i]))) assertTrue(Float.isNaN(decode(t, actual[i])), "NaN cell " + i);
        else assertEquals(expected[i], actual[i], "raw cell " + i);
    }
    private static long windows(long s, long k, long stride, long pad, long dilation, boolean ceil) { return Math.floorDiv(s + 2 * pad - dilation * (k - 1) - 1 + (ceil ? stride - 1 : 0), stride) + 1; }
}

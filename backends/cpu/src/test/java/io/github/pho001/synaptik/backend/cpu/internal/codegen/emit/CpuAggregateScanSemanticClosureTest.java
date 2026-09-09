package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Direct generated-entry closure for every ordinary aggregate and cumulative-scan matrix row. */
class CpuAggregateScanSemanticClosureTest {
    @Test void everyAggregateAndScanMatrixCandidateDefinesAndExecutesItsOwnArtifact()
            throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.aggregateOrScanCandidates();
        assertEquals(460, candidates.size());
        assertEquals(300, candidates.stream().filter(c -> c.context().nodes().getFirst().operation().kind()
                instanceof AggregateReductionKind).count());
        assertEquals(160, candidates.size() - 300);
        for (var candidate : candidates) execute(candidate);
    }

    private static void execute(CpuOrdinaryNonPointwiseGeneratedMatrixTest.AggregateOrScanCandidate candidate)
            throws Throwable {
        PrepareContext<?> context = candidate.context();
        var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(bytes, generator.generateClassBytes(route.specialization(), route.kernelIr()),
                candidate.ownerId());
        var entry = generator.defineClassBytes(route.specialization(), bytes).entryPoint();
        DataType type = context.values().getFirst().descriptor().dataType();
        Object input = storage(type, capacity(context, 0));
        Object inputSnapshot;
        Object actual = storage(type, capacity(context, 1));
        Object expected = storage(type, capacity(context, 1));
        fillInput(input, type);
        inputSnapshot = cloneArray(input);
        fill(actual, type, -17);
        fill(expected, type, -17);
        boolean aggregate = context.nodes().getFirst().operation().kind() instanceof AggregateReductionKind;
        long count = aggregate ? plan.elementCount() : plan.scanGeometry().orElseThrow().sliceCount();
        long start = count > 1 ? 1 : 0;
        long end = count;
        if (aggregate) aggregateOracle(context, input, expected, start, end);
        else scanOracle(context, input, expected, start, end);
        var arguments = new ArrayList<Object>();
        List<Object> raw = List.of(input, actual);
        for (int i = 0; i < raw.size(); i++) arguments.add(route.specialization().carrierPattern().get(i)
                == CarrierAccess.MEMORY_SEGMENT ? segment(raw.get(i)) : raw.get(i));
        try (Arena arena = Arena.ofConfined()) {
            if (plan.workspaceDeclaration().isPresent())
                arguments.add(arena.allocate(plan.workspaceDeclaration().orElseThrow().byteSize(), 8));
            arguments.add(aggregate ? plan.aggregateGeometry().orElseThrow().pack(new long[2])
                    : plan.scanGeometry().orElseThrow().pack(new long[2]));
            arguments.add(start);
            arguments.add(end);
            entry.invokeWithArguments(arguments);
        }
        assertRaw(expected, actual, candidate.ownerId());
        assertRaw(inputSnapshot, input, candidate.ownerId() + " immutable input");
    }

    private static void aggregateOracle(PrepareContext<?> context, Object input, Object output,
            long start, long end) {
        var kind = (AggregateReductionKind) context.nodes().getFirst().operation().kind();
        DataType type = context.values().getFirst().descriptor().dataType();
        long[] inputShape = dimensions(context, 0), outputShape = dimensions(context, 1);
        boolean[] reduced = new boolean[inputShape.length];
        var attrs = context.nodes().getFirst().operation().attrs();
        if (attrs instanceof NoOperationAttrs) java.util.Arrays.fill(reduced, true);
        else if (attrs instanceof AxisReductionAttrs axis) reduced[axis.axis()] = true;
        else for (long axis : ((MultiAxisReductionAttrs) attrs).axes()) reduced[(int) axis] = true;
        for (long outOrdinal = start; outOrdinal < end; outOrdinal++) {
            long[] out = coordinates(outOrdinal, outputShape);
            Object accumulator = identity(kind, type);
            long n = 0;
            for (long ordinal = 0, elements = elements(inputShape); ordinal < elements; ordinal++) {
                long[] coordinate = coordinates(ordinal, inputShape);
                if (!matchesOutput(coordinate, out, reduced)) continue;
                Object value = get(input, type, address(context, 0, coordinate));
                accumulator = n++ == 0 && (kind == AggregateReductionKind.MIN
                        || kind == AggregateReductionKind.MAX) ? value : reduce(kind, type, accumulator, value);
            }
            if (kind == AggregateReductionKind.MEAN) accumulator = represented(type, number(type, accumulator) / n);
            set(output, type, address(context, 1, out), accumulator);
        }
    }

    private static void scanOracle(PrepareContext<?> context, Object input, Object output,
            long start, long end) {
        var attrs = (CumulativeScanAttrs) context.nodes().getFirst().operation().attrs();
        var kind = (CumulativeScanKind) context.nodes().getFirst().operation().kind();
        DataType type = context.values().getFirst().descriptor().dataType();
        long[] shape = dimensions(context, 0); int axis = attrs.axis();
        long[] sliceShape = new long[shape.length - 1];
        for (int source = 0, target = 0; source < shape.length; source++) if (source != axis)
            sliceShape[target++] = shape[source];
        for (long slice = start; slice < end; slice++) {
            long[] sliceCoordinate = coordinates(slice, sliceShape); long[] coordinate = new long[shape.length];
            for (int source = 0, target = 0; source < shape.length; source++) if (source != axis)
                coordinate[source] = sliceCoordinate[target++];
            Object accumulator = kind == CumulativeScanKind.CUM_SUM ? represented(type, 0) : represented(type, 1);
            for (long step = 0; step < shape[axis]; step++) {
                long valueAxis = attrs.reverse() ? shape[axis] - 1 - step : step;
                coordinate[axis] = valueAxis;
                Object value = get(input, type, address(context, 0, coordinate));
                Object written = attrs.exclusive() ? accumulator : reduce(kind == CumulativeScanKind.CUM_SUM
                        ? AggregateReductionKind.SUM : AggregateReductionKind.PROD, type, accumulator, value);
                if (!attrs.exclusive()) accumulator = written;
                else accumulator = reduce(kind == CumulativeScanKind.CUM_SUM ? AggregateReductionKind.SUM
                        : AggregateReductionKind.PROD, type, accumulator, value);
                set(output, type, address(context, 1, coordinate), written);
            }
        }
    }

    private static boolean matchesOutput(long[] input, long[] output, boolean[] reduced) {
        for (int i = 0, j = 0; i < input.length; i++) if (!reduced[i] && input[i] != output[j++]) return false;
        return true;
    }
    private static Object identity(AggregateReductionKind kind, DataType type) {
        return switch (kind) { case SUM, MEAN -> represented(type, 0); case PROD -> represented(type, 1);
            case ALL -> (byte) 1; case ANY -> (byte) 0; default -> null; };
    }
    private static Object reduce(AggregateReductionKind kind, DataType type, Object a, Object b) {
        if (kind == AggregateReductionKind.ALL) return (byte) (((Byte) a != 0 && (Byte) b != 0) ? 1 : 0);
        if (kind == AggregateReductionKind.ANY) return (byte) (((Byte) a != 0 || (Byte) b != 0) ? 1 : 0);
        double x = number(type, a), y = number(type, b);
        double value = switch (kind) { case SUM, MEAN -> x + y; case PROD -> x * y;
            case MIN -> Math.min(x, y); case MAX -> Math.max(x, y); default -> throw new AssertionError(kind); };
        return represented(type, value);
    }
    private static double number(DataType type, Object value) { return type == DataType.BFLOAT16
            ? Float.intBitsToFloat((Short.toUnsignedInt((Short) value)) << 16) : ((Number) value).doubleValue(); }
    private static Object represented(DataType type, double value) { return switch (type) {
        case FLOAT64 -> value; case FLOAT32 -> (float) value; case BFLOAT16 -> bfloat((float) value);
        case INT64 -> (long) value; case INT32 -> (int) value; case BOOL -> (byte) ((int) value); }; }
    private static short bfloat(float value) { int bits = Float.floatToRawIntBits(value), upper = bits >>> 16, lower = bits & 0xffff;
        if ((bits & 0x7f800000) == 0x7f800000 && (bits & 0x7fffff) != 0) upper |= 0x40;
        else if (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0) upper++; return (short) upper; }
    private static long[] dimensions(PrepareContext<?> c, int value) { return c.values().get(value).descriptor().shape().dimensions().stream()
            .mapToLong(d -> ((io.github.pho001.synaptik.model.shape.StaticDimension) d).size()).toArray(); }
    private static long capacity(PrepareContext<?> c, int value) { var d = c.values().get(value).descriptor(); long max = d.layout().orElseThrow().storageOffset(); long[] strides = d.layout().orElseThrow().strides(), shape = dimensions(c, value); for (int i = 0; i < strides.length; i++) max += strides[i] * (shape[i] - 1); return max + 3; }
    private static long address(PrepareContext<?> c, int value, long[] coordinate) { var l = c.values().get(value).descriptor().layout().orElseThrow(); long address = l.storageOffset(); long[] strides = l.strides(); for (int i = 0; i < coordinate.length; i++) address += strides[i] * coordinate[i]; return address; }
    private static long elements(long[] shape) { long value = 1; for (long dimension : shape) value *= dimension; return value; }
    private static long[] coordinates(long ordinal, long[] shape) { long[] result = new long[shape.length]; for (int i = shape.length - 1; i >= 0; i--) { result[i] = ordinal % shape[i]; ordinal /= shape[i]; } return result; }
    private static Object storage(DataType type, long size) { return switch (type) { case FLOAT64 -> new double[(int) size]; case FLOAT32 -> new float[(int) size]; case BFLOAT16 -> new short[(int) size]; case INT64 -> new long[(int) size]; case INT32 -> new int[(int) size]; case BOOL -> new byte[(int) size]; }; }
    private static void fillInput(Object storage, DataType type) { for (int i = 0; i < Array.getLength(storage); i++) set(storage, type, i, represented(type, (i % 5) - 2)); }
    private static void fill(Object storage, DataType type, int value) { for (int i = 0; i < Array.getLength(storage); i++) set(storage, type, i, represented(type, value)); }
    private static Object cloneArray(Object value) { if (value instanceof double[] a) return a.clone(); if (value instanceof float[] a) return a.clone(); if (value instanceof short[] a) return a.clone(); if (value instanceof long[] a) return a.clone(); if (value instanceof int[] a) return a.clone(); return ((byte[]) value).clone(); }
    private static Object get(Object storage, DataType type, long index) { return switch (type) { case FLOAT64 -> ((double[]) storage)[(int) index]; case FLOAT32 -> ((float[]) storage)[(int) index]; case BFLOAT16 -> ((short[]) storage)[(int) index]; case INT64 -> ((long[]) storage)[(int) index]; case INT32 -> ((int[]) storage)[(int) index]; case BOOL -> ((byte[]) storage)[(int) index]; }; }
    private static void set(Object storage, DataType type, long index, Object value) { switch (type) { case FLOAT64 -> ((double[]) storage)[(int) index] = (Double) value; case FLOAT32 -> ((float[]) storage)[(int) index] = (Float) value; case BFLOAT16 -> ((short[]) storage)[(int) index] = (Short) value; case INT64 -> ((long[]) storage)[(int) index] = (Long) value; case INT32 -> ((int[]) storage)[(int) index] = (Integer) value; case BOOL -> ((byte[]) storage)[(int) index] = (Byte) value; } }
    private static MemorySegment segment(Object value) { if (value instanceof double[] a) return MemorySegment.ofArray(a); if (value instanceof float[] a) return MemorySegment.ofArray(a); if (value instanceof short[] a) return MemorySegment.ofArray(a); if (value instanceof long[] a) return MemorySegment.ofArray(a); if (value instanceof int[] a) return MemorySegment.ofArray(a); return MemorySegment.ofArray((byte[]) value); }
    private static void assertRaw(Object expected, Object actual, String message) { if (expected instanceof double[] a) assertArrayEquals(a, (double[]) actual, message); else if (expected instanceof float[] a) assertArrayEquals(a, (float[]) actual, message); else if (expected instanceof short[] a) assertArrayEquals(a, (short[]) actual, message); else if (expected instanceof long[] a) assertArrayEquals(a, (long[]) actual, message); else if (expected instanceof int[] a) assertArrayEquals(a, (int[]) actual, message); else assertArrayEquals((byte[]) expected, (byte[]) actual, message); }
}

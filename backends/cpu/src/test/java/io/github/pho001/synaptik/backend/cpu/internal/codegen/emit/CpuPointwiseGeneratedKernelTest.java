package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;
import org.junit.jupiter.api.Test;
import jdk.incubator.vector.DoubleVector;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

class CpuPointwiseGeneratedKernelTest {
    @Test void generatedAndReferenceAgreeForEveryAdmittedOpcodeAndType() throws Throwable {
        var cases = new ArrayList<Case>();
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.ADD, CpuPointwiseOpcode.SUB,
                CpuPointwiseOpcode.MUL)) for (DataType type : numericTypes()) cases.add(new Case(opcode, type));
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SCALAR_ADD,
                CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL))
            for (DataType type : numericTypes()) cases.add(new Case(opcode, type));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32)) {
            cases.add(new Case(CpuPointwiseOpcode.NEG, type));
            for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.IS_FINITE,
                    CpuPointwiseOpcode.IS_NAN, CpuPointwiseOpcode.IS_INF)) cases.add(new Case(opcode, type));
        }
        cases.add(new Case(CpuPointwiseOpcode.GELU_EXACT, DataType.FLOAT64));
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.GREATER_THAN,
                CpuPointwiseOpcode.GREATER_OR_EQUAL, CpuPointwiseOpcode.LESS_THAN,
                CpuPointwiseOpcode.LESS_OR_EQUAL, CpuPointwiseOpcode.EQUAL,
                CpuPointwiseOpcode.NOT_EQUAL)) for (DataType type : numericTypes())
            cases.add(new Case(opcode, type));
        cases.add(new Case(CpuPointwiseOpcode.WHERE, DataType.FLOAT64));
        cases.add(new Case(CpuPointwiseOpcode.WHERE, DataType.FLOAT32));
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                DataType.INT64, DataType.BOOL)) cases.add(new Case(CpuPointwiseOpcode.CAST, type));

        for (Case one : cases) assertCase(one);
        assertEquals(64, cases.size());
    }

    @Test void float64NumericSubsetUsesVectorBodiesWithScalarTails() throws Throwable {
        for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.SUB,
                CpuPointwiseOpcode.SCALAR_ADD, CpuPointwiseOpcode.SCALAR_SUB,
                CpuPointwiseOpcode.SCALAR_MUL, CpuPointwiseOpcode.NEG)) {
            Case one = new Case(opcode, DataType.FLOAT64);
            CpuKernelIr ir = ir(one);
            List<DataType> types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
            List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                    .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
            var specialization = new CpuKernelSpecialization(
                    CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                    CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                    CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR, types, carriers,
                    DoubleVector.SPECIES_PREFERRED.vectorBitSize(), -1);
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(specialization,
                    generator.generateClassBytes(specialization, ir));
            int count = DoubleVector.SPECIES_PREFERRED.length() + 1;
            var generated = new ArrayList<Object>();
            for (Object input : inputs(one)) generated.add(Arrays.copyOf((double[]) input, count));
            double[] output = new double[count]; generated.add(output);
            var call = new ArrayList<>(generated); call.add(geometry(generated.size(), count));
            call.add(0L); call.add((long) count);
            artifact.entryPoint().invokeWithArguments(call);
            double[] expected = new double[count];
            var reference = new ArrayList<Object>(generated.subList(0, generated.size() - 1));
            reference.add(expected);
            CpuScalarReferenceKernel.execute(ir, reference.stream().map(value -> argument(value,
                    DataType.FLOAT64)).toList(), bindings(generated.size(), count), 0, count);
            assertArrayEquals(expected, output, opcode.name());
        }
    }

    @Test void exactSegmentsLoadAndStoreEveryAdmittedCarrierWidth() throws Throwable {
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32,
                    DataType.INT64, DataType.BOOL)) {
                Case one = new Case(CpuPointwiseOpcode.CAST, type);
                CpuKernelIr ir = ir(one);
                List<DataType> types = List.of(type, type);
                List<CpuKernelSpecialization.CarrierAccess> carriers = List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
                var specialization = new CpuKernelSpecialization(
                        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
                var generator = new CpuClassFileKernelGenerator();
                var artifact = generator.defineClassBytes(specialization,
                        generator.generateClassBytes(specialization, ir));
                var input = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                var output = arena.allocate(4L * type.byteWidth(), type.byteWidth());
                writeSegment(type, input);
                artifact.entryPoint().invokeWithArguments(input, output, geometry(2, 4), 0L, 4L);
                assertEquals(input.mismatch(output), -1L, type.name());

                Object heapInput = arrayValues(type, false);
                var mixedToSegment = artifact(ir, types, List.of(heapCarrier(type),
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT));
                output.fill((byte) 0);
                mixedToSegment.entryPoint().invokeWithArguments(heapInput, output,
                        geometry(2, 4), 0L, 4L);
                assertEquals(-1L, segmentOf(heapInput).mismatch(output), type + " heap-to-segment");

                Object heapOutput = array(type, 4, false);
                var mixedToHeap = artifact(ir, types, List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT, heapCarrier(type)));
                mixedToHeap.entryPoint().invokeWithArguments(input, heapOutput,
                        geometry(2, 4), 0L, 4L);
                assertEquals(-1L, input.mismatch(segmentOf(heapOutput)), type + " segment-to-heap");
            }
        }
    }

    private static CpuGeneratedKernel artifact(CpuKernelIr ir, List<DataType> types,
            List<CpuKernelSpecialization.CarrierAccess> carriers) {
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(specialization,
                generator.generateClassBytes(specialization, ir));
    }

    private static java.lang.foreign.MemorySegment segmentOf(Object array) {
        if (array instanceof double[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof float[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof int[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        if (array instanceof long[] value) return java.lang.foreign.MemorySegment.ofArray(value);
        return java.lang.foreign.MemorySegment.ofArray((byte[]) array);
    }

    private static void writeSegment(DataType type, java.lang.foreign.MemorySegment segment) {
        for (int i = 0; i < 4; i++) {
            long offset = (long) i * type.byteWidth();
            switch (type) {
                case FLOAT64 -> segment.set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i - 1.5);
                case FLOAT32 -> segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i - 1.5f);
                case INT32 -> segment.set(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, i * 17);
                case INT64 -> segment.set(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), offset, (long) i * Long.MAX_VALUE);
                case BOOL -> segment.set(ValueLayout.JAVA_BYTE, offset, (byte) (i & 1));
                default -> throw new IllegalArgumentException("unsupported segment type");
            }
        }
    }

    private static void assertCase(Case one) throws Throwable {
        CpuKernelIr ir = ir(one);
        List<CpuKernelIr.Value> boundaries = ir.values();
        List<DataType> types = boundaries.stream().map(CpuKernelIr.Value::dataType).toList();
        List<CpuKernelSpecialization.CarrierAccess> carriers = types.stream()
                .map(CpuPointwiseGeneratedKernelTest::heapCarrier).toList();
        var specialization = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1);
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(specialization,
                new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir));
        List<Object> inputArrays = inputs(one);
        Object generatedOutput = array(one.outputType(), 4, false);
        Object referenceOutput = array(one.outputType(), 4, false);
        List<Object> generated = new ArrayList<>(inputArrays); generated.add(generatedOutput);
        List<Object> reference = new ArrayList<>(inputArrays); reference.add(referenceOutput);
        long[] geometry = geometry(boundaries.size(), 4);
        var arguments = new ArrayList<Object>(generated);
        arguments.add(geometry); arguments.add(0L); arguments.add(4L);
        artifact.entryPoint().invokeWithArguments(arguments);
        CpuScalarReferenceKernel.execute(ir, reference.stream().map((Object value) -> argument(value,
                types.get(reference.indexOf(value)))).toList(), bindings(boundaries.size()), 0, 4);
        assertPrimitiveArrayEquals(referenceOutput, generatedOutput,
                one.opcode() + " " + one.type());
    }

    private static CpuKernelIr ir(Case one) {
        List<DataType> inputs = one.inputTypes();
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputs.size(); i++) values.add(new CpuKernelIr.Value(i, inputs.get(i),
                CpuKernelIr.Value.Kind.INPUT, dense(CpuAccessPlan.AccessKind.READ)));
        int output = inputs.size();
        values.add(new CpuKernelIr.Value(output, one.outputType(), CpuKernelIr.Value.Kind.OUTPUT,
                dense(CpuAccessPlan.AccessKind.WRITE)));
        CpuKernelIr.ScalarImmediate immediate = one.opcode().carriesScalarImmediate()
                ? immediate(one.type()) : null;
        return new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(one.opcode(),
                java.util.stream.IntStream.range(0, inputs.size()).boxed().toList(), output, immediate)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(output, 0)));
    }

    private static List<Object> inputs(Case one) {
        if (one.opcode() == CpuPointwiseOpcode.WHERE) return List.of(
                new byte[] {1, 0, 1, 0}, arrayValues(one.type(), false), arrayValues(one.type(), true));
        var result = new ArrayList<Object>();
        for (int i = 0; i < one.opcode().arity(); i++) result.add(arrayValues(one.type(), i == 1));
        return result;
    }

    private static Object arrayValues(DataType type, boolean second) {
        return switch (type) {
            case FLOAT64 -> second ? new double[] {-0.0, Double.NaN, 3, -4}
                    : new double[] {0.0, Double.POSITIVE_INFINITY, -2, Double.MIN_VALUE};
            case FLOAT32 -> second ? new float[] {-0.0f, Float.NaN, 3, -4}
                    : new float[] {0.0f, Float.POSITIVE_INFINITY, -2, Float.MIN_VALUE};
            case INT32 -> second ? new int[] {1, -1, Integer.MAX_VALUE, 3}
                    : new int[] {Integer.MAX_VALUE, Integer.MIN_VALUE, -2, 7};
            case INT64 -> second ? new long[] {1, -1, Long.MAX_VALUE, 3}
                    : new long[] {Long.MAX_VALUE, Long.MIN_VALUE, -2, 7};
            case BOOL -> new byte[] {0, 1, 1, 0};
            default -> throw new IllegalArgumentException("unsupported test type");
        };
    }

    private static Object array(DataType type, int size, boolean ignored) {
        return switch (type) {
            case FLOAT64 -> new double[size]; case FLOAT32 -> new float[size];
            case INT32 -> new int[size]; case INT64 -> new long[size]; case BOOL -> new byte[size];
            default -> throw new IllegalArgumentException("unsupported test type");
        };
    }

    private static CpuKernelIr.ScalarImmediate immediate(DataType type) {
        return new CpuKernelIr.ScalarImmediate(type, switch (type) {
            case FLOAT64 -> Double.doubleToRawLongBits(-0.0d);
            case FLOAT32 -> Float.floatToRawIntBits(-0.0f) & 0xffff_ffffL;
            case INT32 -> 0xffff_ffffL; case INT64 -> -1L;
            default -> throw new IllegalArgumentException("unsupported immediate");
        });
    }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static List<CpuAccessPlan.Binding> bindings(int count) {
        return bindings(count, 4);
    }

    private static List<CpuAccessPlan.Binding> bindings(int count, long extent) {
        var result = new ArrayList<CpuAccessPlan.Binding>();
        for (int i = 0; i < count; i++) result.add(CpuAccessPlan.Binding.create(
                dense(i == count - 1 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ),
                new long[] {extent}, 0, new long[] {1}, extent, 0, extent, extent));
        return result;
    }

    private static long[] geometry(int count, long extent) {
        long[] result = new long[2 + count + count + 2 * count];
        result[0] = extent;
        for (int i = 0; i < count; i++) {
            result[2 + count + i] = 1;
            result[2 + count + count + count + i] = extent;
        }
        return result;
    }

    private static CpuBufferArgument argument(Object value, DataType type) {
        long bytes = java.lang.reflect.Array.getLength(value) * (long) type.byteWidth();
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) value, 0, bytes, false);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) value, 0, bytes, false);
            case INT32 -> new CpuBufferArgument.Ints((int[]) value, 0, bytes, false);
            case INT64 -> new CpuBufferArgument.Longs((long[]) value, 0, bytes, false);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) value, 0, bytes, false);
            default -> throw new IllegalArgumentException("unsupported argument type");
        };
    }

    private static CpuKernelSpecialization.CarrierAccess heapCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
            default -> throw new IllegalArgumentException("unsupported carrier");
        };
    }

    private static List<DataType> numericTypes() {
        return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.INT32, DataType.INT64);
    }

    private static void assertPrimitiveArrayEquals(Object expected, Object actual, String message) {
        if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else if (expected instanceof long[] value) assertArrayEquals(value, (long[]) actual, message);
        else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }

    private record Case(CpuPointwiseOpcode opcode, DataType type) {
        List<DataType> inputTypes() {
            if (opcode == CpuPointwiseOpcode.WHERE) return List.of(DataType.BOOL, type, type);
            return java.util.Collections.nCopies(opcode.arity(), type);
        }
        DataType outputType() {
            return opcode.resultCategory() == CpuPointwiseOpcode.ResultCategory.BOOL
                    ? DataType.BOOL : type;
        }
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.reference.CpuScalarReferenceKernel;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Exercises represented low pointwise boundaries through the real generated primitive entry. */
class CpuLowPrecisionPointwiseGeneratedTest {
    private static final List<DataType> LOW = List.of(DataType.BFLOAT16, DataType.FLOAT16,
            DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);

    @Test void fp8UnaryPredicatesAndScalarOperationsCoverEveryStoredEncoding() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            byte[] input = new byte[256];
            for (int i = 0; i < input.length; i++) input[i] = (byte) i;
            for (CpuPointwiseOpcode opcode : CpuPointwiseOpcode.values()) {
                if (opcode.family() != CpuPointwiseOpcode.Family.UNARY
                        && opcode.family() != CpuPointwiseOpcode.Family.CLASSIFICATION
                        && opcode.family() != CpuPointwiseOpcode.Family.SCALAR_ARITHMETIC
                        && opcode.family() != CpuPointwiseOpcode.Family.SCALAR_RANGE) continue;
                DataType output = opcode.family() == CpuPointwiseOpcode.Family.CLASSIFICATION
                        ? DataType.BOOL : type;
                compare(opcode, List.of(type), output, List.of(input), 256);
            }
        }
    }

    @Test void lowBinaryArithmeticAndExactComparisonsUseRepresentedValues() throws Throwable {
        for (DataType type : LOW) {
            float[] values = {-0f, 0f, 1f, -1f, .5f, 2f, Float.NaN,
                    Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 448f, 57344f};
            Object left = array(type, values.length), right = array(type, values.length);
            for (int i = 0; i < values.length; i++) {
                put(type, left, i, values[i]); put(type, right, i, values[values.length - 1 - i]);
            }
            for (CpuPointwiseOpcode opcode : CpuPointwiseOpcode.values()) {
                if (opcode.family() != CpuPointwiseOpcode.Family.BINARY_ARITHMETIC
                        && opcode.family() != CpuPointwiseOpcode.Family.COMPARISON) continue;
                compare(opcode, List.of(type, type), opcode.family() == CpuPointwiseOpcode.Family.COMPARISON
                        ? DataType.BOOL : type, List.of(left, right), values.length);
            }
        }
    }

    @Test void unlikeLowOperandsPromoteAndF64DominatesWithoutPrenarrowing() throws Throwable {
        for (DataType leftType : LOW) for (DataType rightType : LOW) if (leftType != rightType) {
            Object left = array(leftType, 4), right = array(rightType, 4);
            for (int i = 0; i < 4; i++) { put(leftType, left, i, i - 1.5f); put(rightType, right, i, .5f); }
            compare(CpuPointwiseOpcode.ADD, List.of(leftType, rightType), DataType.FLOAT32,
                    List.of(left, right), 4);
            compare(CpuPointwiseOpcode.MUL, List.of(leftType, DataType.FLOAT64), DataType.FLOAT64,
                    List.of(left, new double[] {1.0000000001, -2, Double.MIN_VALUE, Double.MAX_VALUE}), 4);
        }
    }

    @Test void rawWhereAndSignOperationsNeverNormalizeFp8Bytes() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            byte[] left = new byte[256], right = new byte[256], mask = new byte[256];
            for (int i = 0; i < 256; i++) { left[i] = (byte) i; right[i] = (byte) (255 - i); mask[i] = (byte) (i & 1); }
            compare(CpuPointwiseOpcode.WHERE, List.of(DataType.BOOL, type, type), type,
                    List.of(mask, left, right), 256);
            for (CpuPointwiseOpcode opcode : List.of(CpuPointwiseOpcode.ABS, CpuPointwiseOpcode.NEG)) {
                byte[] actual = new byte[256];
                kernel(ir(opcode, List.of(type), type)).invokeWithArguments(left, actual,
                        geometry(2, 256), 0L, 256L);
                for (int i = 0; i < 256; i++) assertEquals(opcode == CpuPointwiseOpcode.ABS
                        ? i & 127 : i ^ 128, Byte.toUnsignedInt(actual[i]), type + " " + opcode + " " + i);
            }
        }
    }

    private static void compare(CpuPointwiseOpcode opcode, List<DataType> inputs, DataType output,
            List<Object> values, int count) throws Throwable {
        CpuKernelIr ir = ir(opcode, inputs, output);
        Object actual = array(output, count), expected = array(output, count);
        var args = new ArrayList<Object>(values); args.add(actual); args.add(geometry(inputs.size()+1, count));
        args.add(0L); args.add((long) count);
        kernel(ir).invokeWithArguments(args);
        var reference = new ArrayList<CpuBufferArgument>();
        for (int i = 0; i < values.size(); i++) reference.add(argument(inputs.get(i), values.get(i)));
        reference.add(argument(output, expected));
        CpuScalarReferenceKernel.execute(ir, reference, ir.values().stream().map(v ->
                CpuAccessPlan.Binding.create(v.accessPlan(), new long[] {count}, 0, new long[] {1},
                        count, 0, count, count)).toList(), 0, count);
        String label = opcode + " " + inputs + " -> " + output;
        if (actual instanceof byte[] a) assertArrayEquals((byte[]) expected, a, label);
        else if (actual instanceof short[] a) assertArrayEquals((short[]) expected, a, label);
        else if (actual instanceof float[] a) assertArrayEquals((float[]) expected, a, label);
        else assertArrayEquals((double[]) expected, (double[]) actual, label);
    }

    private static CpuKernelIr ir(CpuPointwiseOpcode opcode, List<DataType> inputs, DataType output) {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputs.size(); i++) values.add(new CpuKernelIr.Value(i, inputs.get(i),
                CpuKernelIr.Value.Kind.INPUT, access(CpuAccessPlan.AccessKind.READ)));
        values.add(new CpuKernelIr.Value(inputs.size(), output, CpuKernelIr.Value.Kind.OUTPUT,
                access(CpuAccessPlan.AccessKind.WRITE)));
        var immediate = opcode.carriesScalarImmediate() ? immediate(inputs.getFirst(), .5f) : null;
        var clamp = opcode == CpuPointwiseOpcode.SCALAR_CLAMP ? new CpuKernelIr.ClampImmediate(
                immediate(output, -1), immediate(output, 1)) : null;
        return new CpuKernelIr(values, List.of(new CpuKernelIr.Instruction(opcode,
                java.util.stream.IntStream.range(0, inputs.size()).boxed().toList(), inputs.size(),
                immediate, opcode == CpuPointwiseOpcode.SCALAR_POW ? CpuKernelIr.PowerRealization.DIRECT : null,
                clamp)), new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(inputs.size(), 0)));
    }

    private static MethodHandle kernel(CpuKernelIr ir) {
        var types = ir.values().stream().map(CpuKernelIr.Value::dataType).toList();
        var carriers = types.stream().map(t -> t.byteWidth() == 1 ? CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY
                : t.byteWidth() == 2 ? CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY
                : t == DataType.FLOAT32 ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY).toList();
        var spec = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers, 0, -1,
                ir.instructions().stream().filter(i -> i.opcode() == CpuPointwiseOpcode.SCALAR_POW)
                        .map(CpuKernelIr.Instruction::powerRealization).toList(), false,
                CpuKernelSpecialization.requiresLowPrecisionArithmeticSchema(ir)
                        ? CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA
                        : types.contains(DataType.FLOAT16) ? 68 : types.contains(DataType.BFLOAT16) ? 59 : 52);
        var generator = new CpuClassFileKernelGenerator();
        return generator.defineClassBytes(spec, ir, generator.generateClassBytes(spec, ir)).entryPoint();
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }
    private static long[] geometry(int boundaries, int count) {
        long[] result = new long[2 + 3 * boundaries]; result[0] = count;
        for (int i = 0; i < boundaries; i++) { result[2+boundaries+i] = 1; result[2+2*boundaries+i] = count; }
        return result;
    }
    private static Object array(DataType type, int count) {
        if (type.byteWidth() == 1) return new byte[count];
        if (type.byteWidth() == 2) return new short[count];
        return type == DataType.FLOAT32 ? new float[count] : new double[count];
    }
    private static CpuBufferArgument argument(DataType type, Object array) {
        return new CpuBufferArgument.Segment(type, segment(array), segment(array).byteSize(), false);
    }
    private static MemorySegment segment(Object array) {
        return switch (array) { case byte[] a -> MemorySegment.ofArray(a); case short[] a -> MemorySegment.ofArray(a);
            case float[] a -> MemorySegment.ofArray(a); case double[] a -> MemorySegment.ofArray(a);
            default -> throw new AssertionError(array); };
    }
    private static void put(DataType type, Object array, int at, float value) {
        var scalar = CastValueConversions.convert(ScalarValue.float32(value), type);
        if (array instanceof byte[] a) a[at] = type == DataType.FLOAT8_E4M3 ? scalar.float8E4m3Bits() : scalar.float8E5m2Bits();
        else ((short[]) array)[at] = type == DataType.FLOAT16 ? scalar.float16Bits() : scalar.bfloat16Bits();
    }
    private static CpuKernelIr.ScalarImmediate immediate(DataType type, float value) {
        var scalar = CastValueConversions.convert(ScalarValue.float32(value), type);
        long bits = switch (type) { case BFLOAT16 -> Short.toUnsignedLong(scalar.bfloat16Bits());
            case FLOAT16 -> Short.toUnsignedLong(scalar.float16Bits());
            case FLOAT8_E4M3 -> Byte.toUnsignedLong(scalar.float8E4m3Bits());
            case FLOAT8_E5M2 -> Byte.toUnsignedLong(scalar.float8E5m2Bits());
            default -> throw new AssertionError(type); };
        return new CpuKernelIr.ScalarImmediate(type, bits);
    }
}

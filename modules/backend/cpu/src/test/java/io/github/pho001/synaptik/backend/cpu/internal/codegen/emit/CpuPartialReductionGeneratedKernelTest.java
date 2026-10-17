package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.classfile.Instruction;
import java.lang.classfile.Opcode;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.invoke.MethodType;
import java.lang.foreign.MemorySegment;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;

/** Validates the direct, typed generated partial-body and ordered-combine artifact ABI. */
class CpuPartialReductionGeneratedKernelTest {
    private static final CpuClassFileKernelGenerator GENERATOR = new CpuClassFileKernelGenerator();

    @Test void emitsAllIntegralSumProductArtifactsWithFixedPartialCountsAndOffsets()
            throws Throwable {
        for (DataType type : new DataType[]{DataType.INT32, DataType.INT64}) {
            for (CpuPartialReductionIr.Kind kind : CpuPartialReductionIr.Kind.values()) {
                for (int partials : new int[]{2, 4}) {
                    CpuAggregateIr.Form form = type == DataType.INT32
                            && kind == CpuPartialReductionIr.Kind.SUM
                            ? CpuAggregateIr.Form.FULL : partials == 2
                                    ? CpuAggregateIr.Form.SINGLE_AXIS
                                    : CpuAggregateIr.Form.MULTI_AXIS;
                    var ir = new CpuPartialReductionIr(kind, type, form, 2, 7, partials);
                    var artifact = GENERATOR.generatePartialReduction(ir);
                    assertArrayEquals(artifact.classBytes(),
                            GENERATOR.generatePartialReduction(ir).classBytes());
                    assertEquals(type == DataType.INT32
                                    ? MethodType.methodType(void.class, int[].class, int.class,
                                            int.class, MemorySegment.class, long.class)
                                    : MethodType.methodType(void.class, long[].class, int.class,
                                            int.class, MemorySegment.class, long.class),
                            artifact.partialBody().type());
                    assertEquals(type == DataType.INT32
                                    ? MethodType.methodType(void.class, MemorySegment.class, int.class,
                                            int.class, int[].class, int.class)
                                    : MethodType.methodType(void.class, MemorySegment.class, int.class,
                                            int.class, long[].class, int.class),
                            artifact.orderedCombine().type());
                    assertDirectPrimitiveStructure(artifact.classBytes(), type, kind);
                    if (type == DataType.INT32) {
                        int[] input = {91, Integer.MAX_VALUE, 1, -2, 3, 4, 5,
                                Integer.MIN_VALUE, -1, 2, 3, -4, 5, 92, 93, 94};
                        int[] output = {-1, -1, -1, -1};
                        invokeInt(artifact, input, 1, output, 1);
                        assertArrayEquals(new int[]{-1, foldInt(kind, input, 1),
                                foldInt(kind, input, 8), -1}, output);
                    } else {
                        long[] input = {91L, Long.MAX_VALUE, 2L, -1L, 3L, 4L, 5L,
                                Long.MIN_VALUE, -1L, 2L, 3L, -4L, 5L, 92L, 93L, 94L};
                        long[] output = {-1L, -1L, -1L, -1L};
                        invokeLong(artifact, input, 1, output, 1);
                        assertArrayEquals(new long[]{-1L, foldLong(kind, input, 1),
                                foldLong(kind, input, 8), -1L}, output);
                    }
                }
            }
        }
    }

    @Test void generatedArtifactContainsOnlyTwoDirectPrimitiveLoops() {
        var ir = new CpuPartialReductionIr(CpuPartialReductionIr.Kind.PROD, DataType.INT64,
                CpuAggregateIr.Form.FULL, 1, 7, 4);
        var model = ClassFile.of().parse(GENERATOR.generatePartialReduction(ir).classBytes());
        var partial = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals("partialBody")).findFirst().orElseThrow();
        var combine = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals("orderedCombine")).findFirst().orElseThrow();
        var members = StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertEquals(2, model.methods().size());
        assertEquals("([JIILjava/lang/foreign/MemorySegment;J)V", partial.methodTypeSymbol().descriptorString());
        assertEquals("(Ljava/lang/foreign/MemorySegment;II[JI)V", combine.methodTypeSymbol().descriptorString());
        assertTrue(opcodes(partial, Opcode.LALOAD) > 0);
        assertTrue(opcodes(partial, Opcode.LMUL) > 0);
        assertTrue(opcodes(combine, Opcode.INVOKEINTERFACE) > 0);
        assertTrue(opcodes(combine, Opcode.LMUL) > 0);
        assertFalse(members.isEmpty());
        assertFalse(model.methods().stream().flatMap(method -> method.code().orElseThrow()
                .elementStream()).anyMatch(element -> element instanceof java.lang.classfile.instruction.NewObjectInstruction));
    }

    private static void assertDirectPrimitiveStructure(byte[] bytes, DataType type,
            CpuPartialReductionIr.Kind kind) {
        var model = ClassFile.of().parse(bytes);
        var partial = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals("partialBody")).findFirst().orElseThrow();
        var combine = model.methods().stream().filter(method -> method.methodName().stringValue()
                .equals("orderedCombine")).findFirst().orElseThrow();
        var members = StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertEquals(2, model.methods().size());
        assertTrue(opcodes(partial, type == DataType.INT32 ? Opcode.IALOAD : Opcode.LALOAD) > 0);
        assertTrue(opcodes(partial, kind == CpuPartialReductionIr.Kind.SUM
                ? type == DataType.INT32 ? Opcode.IADD : Opcode.LADD
                : type == DataType.INT32 ? Opcode.IMUL : Opcode.LMUL) > 0);
        assertTrue(opcodes(combine, Opcode.INVOKEINTERFACE) > 0);
        assertTrue(opcodes(combine, kind == CpuPartialReductionIr.Kind.SUM
                ? type == DataType.INT32 ? Opcode.IADD : Opcode.LADD
                : type == DataType.INT32 ? Opcode.IMUL : Opcode.LMUL) > 0);
        assertFalse(members.isEmpty());
        assertFalse(model.methods().stream().flatMap(method -> method.code().orElseThrow()
                .elementStream()).anyMatch(element -> element
                        instanceof java.lang.classfile.instruction.NewObjectInstruction));
    }

    private static long opcodes(java.lang.classfile.MethodModel method, Opcode opcode) {
        return method.code().orElseThrow().elementStream().filter(Instruction.class::isInstance)
                .map(Instruction.class::cast).filter(instruction -> instruction.opcode() == opcode)
                .count();
    }

    private static void invokeInt(CpuGeneratedKernel.PartialReductionArtifact artifact, int[] input,
            int inputBase, int[] output, int outputBase) throws Throwable {
        MemorySegment state = MemorySegment.ofArray(new long[Math.toIntExact(artifact.ir().outputCount()
                * artifact.ir().partialCount())]);
        for (int cell = 0; cell < artifact.ir().outputCount(); cell++) for (int partial = 0;
                partial < artifact.ir().partialCount(); partial++) {
            int begin = Math.toIntExact(inputBase + cell * artifact.ir().domainCount()
                    + artifact.ir().begin(cell, partial));
            int end = Math.toIntExact(inputBase + cell * artifact.ir().domainCount()
                    + artifact.ir().end(cell, partial));
            artifact.partialBody().invokeExact(input, begin, end, state,
                    artifact.ir().stateOffset(cell, partial));
        }
        artifact.orderedCombine().invokeExact(state, 0, Math.toIntExact(artifact.ir().outputCount()),
                output, outputBase);
    }

    private static void invokeLong(CpuGeneratedKernel.PartialReductionArtifact artifact, long[] input,
            int inputBase, long[] output, int outputBase) throws Throwable {
        MemorySegment state = MemorySegment.ofArray(new long[Math.toIntExact(artifact.ir().outputCount()
                * artifact.ir().partialCount())]);
        for (int cell = 0; cell < artifact.ir().outputCount(); cell++) for (int partial = 0;
                partial < artifact.ir().partialCount(); partial++) {
            int begin = Math.toIntExact(inputBase + cell * artifact.ir().domainCount()
                    + artifact.ir().begin(cell, partial));
            int end = Math.toIntExact(inputBase + cell * artifact.ir().domainCount()
                    + artifact.ir().end(cell, partial));
            artifact.partialBody().invokeExact(input, begin, end, state,
                    artifact.ir().stateOffset(cell, partial));
        }
        artifact.orderedCombine().invokeExact(state, 0, Math.toIntExact(artifact.ir().outputCount()),
                output, outputBase);
    }

    private static int foldInt(CpuPartialReductionIr.Kind kind, int[] values, int offset) {
        int result = kind == CpuPartialReductionIr.Kind.SUM ? 0 : 1;
        for (int index = offset; index < offset + 7; index++) result = kind
                == CpuPartialReductionIr.Kind.SUM ? result + values[index] : result * values[index];
        return result;
    }

    private static long foldLong(CpuPartialReductionIr.Kind kind, long[] values, int offset) {
        long result = kind == CpuPartialReductionIr.Kind.SUM ? 0L : 1L;
        for (int index = offset; index < offset + 7; index++) result = kind
                == CpuPartialReductionIr.Kind.SUM ? result + values[index] : result * values[index];
        return result;
    }
}

package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.lang.classfile.Opcode;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits one allocation-free scalar represented-bit affine boundary copy.
 *
 * <p>The general generated loop reads cold-composed source/result address pairs for the requested
 * half-open range. When specialization proves two dense heap arrays with unit-stride affine
 * progressions, the generated entry narrows its bounds and first address pair once and advances
 * direct integer array indexes. Both forms transfer primitive payloads without conversion and do
 * not inspect a Shape, layout, operation, route, or carrier kind at runtime. One additional
 * completely guarded raw-BFLOAT16 segment-to-short-array body covers the frozen
 * {@code [256,32,32]} PERMUTE/SLICE mapping. It accepts every legal half-open logical range,
 * advances one integer ordinal, derives the three coordinates with shifts and masks, and performs
 * direct native-order unaligned short loads and exact short-array stores. Entry requires the
 * complete packed-geometry length, ordered range bounds, and sentinel source/result pairs at all
 * relevant row and plane transitions; a failed guard enters the unchanged typed general-long
 * address-pair body. Neither body converts or numerically interprets the BFLOAT16 payload.</p>
 */
final class CpuAffineCopyEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc SHORT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
    private static final String GUARDED_BFLOAT16_PERMUTE_SLICE =
            "affine:BFLOAT16:[MappingStep[kind=PERMUTE, inputRank=3, outputRank=3, "
            + "axes=[1, 0, 2]], MappingStep[kind=SLICE, inputRank=3, outputRank=3, "
            + "axes=[0]]]:LOGICAL_ELEMENTS";
    /** Creates one stateless affine copy emitter. */
    CpuAffineCopyEmitter() {
    }

    /**
     * Emits the already-validated two-boundary copy body selected by structural specialization.
     * The emitted method preserves the universal {@code long start, long end} entry contract;
     * any guarded fixed geometry is an internal body of that one artifact, not another route or
     * planner-visible specialization.
     *
     * @param code non-null Class-File method body builder
     * @param specialization non-null scalar specialization with exactly two compatible carriers
     * @param ir non-null instruction-free encoded affine copy form matching the specialization
     */
    void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.loopAddressing(ir)
                == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) {
            emitDenseArrayInt(code, specialization, ir);
            return;
        }
        if (isGuardedBfloat16PermuteSlice(specialization, ir)) {
            emitGuardedBfloat16PermuteSlice(code, specialization, ir);
            return;
        }
        emitGeneralLong(code, specialization, ir);
    }

    private static boolean isGuardedBfloat16PermuteSlice(
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        return ir.values().getFirst().dataType() == DataType.BFLOAT16
                && ir.familyIdentity().equals(GUARDED_BFLOAT16_PERMUTE_SLICE)
                && specialization.carrierPattern().equals(java.util.List.of(
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY));
    }

    private static void emitGuardedBfloat16PermuteSlice(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        var fallback = code.newLabel();
        var done = code.newLabel();
        code.aload(2).arraylength().loadConstant(524_288)
                .branch(Opcode.IF_ICMPNE, fallback);
        code.lload(3).loadConstant(0L).lcmp().branch(Opcode.IFLT, fallback);
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGT, fallback);
        code.lload(5).loadConstant(262_144L).lcmp().branch(Opcode.IFGT, fallback);
        guardAddressPair(code, fallback, 0, 5);
        guardAddressPair(code, fallback, 31, 67);
        guardAddressPair(code, fallback, 32, 2_053);
        guardAddressPair(code, fallback, 8_191, 522_307);
        guardAddressPair(code, fallback, 8_192, 69);
        guardAddressPair(code, fallback, 262_143, 524_291);
        emitBfloat16PermuteSliceCursors(code, done);
        code.branch(Opcode.GOTO, done);
        code.labelBinding(fallback);
        emitGeneralLong(code, specialization, ir);
        code.labelBinding(done);
    }

    private static void guardAddressPair(CodeBuilder code, java.lang.classfile.Label fallback,
            int logical, long expected) {
        int packed = Math.multiplyExact(logical, 2);
        code.aload(2).loadConstant(packed).laload().loadConstant(expected).lcmp()
                .branch(Opcode.IFNE, fallback);
        code.aload(2).loadConstant(packed + 1).laload().loadConstant(expected).lcmp()
                .branch(Opcode.IFNE, fallback);
    }

    private static void emitBfloat16PermuteSliceCursors(CodeBuilder code,
            java.lang.classfile.Label done) {
        int logical = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        code.lload(3).l2i().istore(logical);
        code.lload(5).l2i().istore(end);
        int x = code.allocateLocal(TypeKind.INT);
        int y = code.allocateLocal(TypeKind.INT);
        int z = code.allocateLocal(TypeKind.INT);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        int resultAddress = code.allocateLocal(TypeKind.INT);
        int value = code.allocateLocal(TypeKind.INT);
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(logical).iload(end).branch(Opcode.IF_ICMPGE, done);
        code.iload(logical).loadConstant(13).ishr().istore(x);
        code.iload(logical).loadConstant(5).ishr().loadConstant(255).iand().istore(y);
        code.iload(logical).loadConstant(31).iand().istore(z);
        code.loadConstant(5).iload(x).loadConstant(64).imul().iadd()
                .iload(y).loadConstant(2_048).imul().iadd()
                .iload(z).loadConstant(2).imul().iadd().dup().istore(resultAddress)
                .i2l().lstore(sourceAddress);
        code.aload(0).getstatic(VALUE_LAYOUT, "JAVA_SHORT_UNALIGNED", SHORT_LAYOUT)
                .lload(sourceAddress).loadConstant(2L).lmul()
                .invokeinterface(SEGMENT, "get",
                        MethodTypeDesc.of(TypeKind.SHORT.upperBound(), SHORT_LAYOUT,
                                TypeKind.LONG.upperBound()));
        code.istore(value);
        code.aload(1).iload(resultAddress).iload(value).sastore();
        code.iinc(logical, 1);
        code.branch(Opcode.GOTO, loop);
    }

    private static void emitGeneralLong(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        DataType type = ir.values().getFirst().dataType();
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        var carriers = new CpuCarrierEmitter(code);
        int index = code.allocateLocal(TypeKind.LONG);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        int resultAddress = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(index);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.lload(index).lload(5).lcmp().branch(Opcode.IFGE, done);
        code.aload(2).lload(index).loadConstant(2L).lmul().l2i().laload().lstore(sourceAddress);
        code.aload(2).lload(index).loadConstant(2L).lmul().loadConstant(1L).ladd()
                .l2i().laload().lstore(resultAddress);
            carriers.load(type, specialization.carrierPattern().get(0), 0, sourceAddress);
            switch (type) {
                case FLOAT64 -> code.dstore(value);
                case FLOAT32 -> code.fstore(value);
                case BFLOAT16, INT32, BOOL -> code.istore(value);
                case INT64 -> code.lstore(value);
            }
            carriers.store(type, specialization.carrierPattern().get(1), 1,
                    resultAddress, value);
        code.lload(index).loadConstant(1L).ladd().lstore(index);
        code.branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }

    private static void emitDenseArrayInt(CodeBuilder code,
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        DataType type = ir.values().getFirst().dataType();
        int value = code.allocateLocal(switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32 -> TypeKind.FLOAT;
            case BFLOAT16, INT32, BOOL -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
        });
        int index = code.allocateLocal(TypeKind.INT);
        int end = code.allocateLocal(TypeKind.INT);
        int sourceAddress = code.allocateLocal(TypeKind.INT);
        int resultAddress = code.allocateLocal(TypeKind.INT);
        code.lload(3).l2i().istore(index);
        code.lload(5).l2i().istore(end);
        code.aload(2).iload(index).loadConstant(2).imul().laload().l2i().istore(sourceAddress);
        code.aload(2).iload(index).loadConstant(2).imul().loadConstant(1).iadd()
                .laload().l2i().istore(resultAddress);
        var carriers = new CpuCarrierEmitter(code);
        var done = code.newLabel();
        var loop = code.newLabel();
        code.labelBinding(loop);
        code.iload(index).iload(end).branch(Opcode.IF_ICMPGE, done);
        carriers.load(type, specialization.carrierPattern().get(0), 0, sourceAddress, true);
        switch (type) {
            case FLOAT64 -> code.dstore(value);
            case FLOAT32 -> code.fstore(value);
            case BFLOAT16, INT32, BOOL -> code.istore(value);
            case INT64 -> code.lstore(value);
        }
        carriers.store(type, specialization.carrierPattern().get(1), 1, resultAddress, value, true);
        code.iinc(index, 1).iinc(sourceAddress, 1).iinc(resultAddress, 1)
                .branch(Opcode.GOTO, loop);
        code.labelBinding(done);
    }
}

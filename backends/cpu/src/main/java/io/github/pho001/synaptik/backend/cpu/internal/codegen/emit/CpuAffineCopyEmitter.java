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
 * <p>The general generated loop reads cold-composed source addresses for the requested half-open
 * range. A proved dense result loads its initial address once and advances a unit-stride result
 * cursor; other results retain per-element cold-composed result-address lookup. When
 * specialization proves two dense heap arrays with unit-stride affine
 * progressions, the generated entry narrows its bounds and first address pair once and advances
 * direct integer array indexes. Both forms transfer primitive payloads without conversion and do
 * not inspect a Shape, layout, operation, route, or carrier kind at runtime. One additional
 * completely guarded raw-BFLOAT16 segment-to-short-array body covers the frozen
 * {@code [256,32,32]} PERMUTE/SLICE mapping. It accepts every legal half-open logical range,
 * advances one integer ordinal, derives the three coordinates with shifts and masks, and performs
 * direct native-order unaligned short loads and exact short-array stores. Entry requires the
 * complete packed-geometry length, ordered range bounds, and sentinel source/result pairs at all
 * relevant row and plane transitions; a failed guard enters the unchanged typed general-long
 * address-pair body. Neither body converts or numerically interprets the BFLOAT16 payload. The
 * emitted hot loops follow the matching optimal clean-Java algorithm and dataflow without helper
 * dispatch, allocation, boxing, reflection, or planner/Runtime selection.</p>
 */
final class CpuAffineCopyEmitter {
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");
    private static final ClassDesc SHORT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfShort");
    private static final ClassDesc DOUBLE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfDouble");
    private static final ClassDesc FLOAT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfFloat");
    private static final ClassDesc INT_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfInt");
    private static final ClassDesc LONG_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfLong");
    private static final ClassDesc BYTE_LAYOUT =
            ClassDesc.of("java.lang.foreign.ValueLayout$OfByte");
    private static final String GUARDED_BFLOAT16_PERMUTE_SLICE =
            "affine:BFLOAT16:[MappingStep[kind=PERMUTE, inputRank=3, outputRank=3, "
            + "axes=[1, 0, 2]], MappingStep[kind=SLICE, inputRank=3, outputRank=3, "
            + "axes=[0]]]:LOGICAL_ELEMENTS";
    /** Creates one stateless affine copy emitter. */
    CpuAffineCopyEmitter() {
    }

    /**
     * Proves that the existing general-long dense-result affine body directly emits every typed
     * carrier access and therefore requires no shared invocation-local segment layouts.
     *
     * <p>This predicate is deliberately fail-closed. Dense-array specialization, the guarded
     * BFLOAT16 body, non-dense results, malformed boundary structure, and any future affine form
     * retain the shared carrier-layout preparation path.</p>
     *
     * @param specialization exact non-null generated carrier specialization
     * @param ir exact non-null instruction-free affine kernel IR
     * @return {@code true} only for the existing general-long dense-result body whose direct
     *     primitive array and constant-layout segment accesses are wholly emitted by this type
     */
    static boolean ownsGeneralLongDenseResultCarrierAccess(
            CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization == null || ir == null || !ir.instructions().isEmpty()
                || ir.values().size() != 2 || ir.stores().size() != 1
                || !ir.familyIdentity().startsWith("affine:")
                || !ir.familyIdentity().endsWith(":LOGICAL_ELEMENTS")
                || specialization.boundaryDataTypes().size() != 2
                || specialization.carrierPattern().size() != 2
                || isGuardedBfloat16PermuteSlice(specialization, ir)) return false;
        CpuKernelSpecialization.LoopAddressing addressing;
        try {
            addressing = specialization.loopAddressing(ir);
        } catch (IllegalArgumentException | ArithmeticException exception) {
            return false;
        }
        if (addressing == CpuKernelSpecialization.LoopAddressing.DENSE_HEAP_ARRAY_INT) return false;
        CpuKernelIr.Value source = ir.values().get(0);
        CpuKernelIr.Value result = ir.values().get(1);
        if (source.ordinal() != 0 || source.kind() != CpuKernelIr.Value.Kind.INPUT
                || source.accessPlan().accessKind()
                        != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.READ
                || result.ordinal() != 1 || result.kind() != CpuKernelIr.Value.Kind.OUTPUT
                || result.accessPlan().accessKind()
                        != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AccessKind.WRITE
                || result.accessPlan().regime()
                        != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime.DENSE_LINEAR
                || source.dataType() != result.dataType()
                || specialization.boundaryDataTypes().get(0) != source.dataType()
                || specialization.boundaryDataTypes().get(1) != result.dataType()
                || ir.stores().getFirst().value() != result.ordinal()
                || ir.stores().getFirst().outputOrdinal() != 0) return false;
        return directCarrier(source.dataType(), specialization.carrierPattern().get(0))
                && directCarrier(result.dataType(), specialization.carrierPattern().get(1));
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
        int index = code.allocateLocal(TypeKind.LONG);
        int sourceAddress = code.allocateLocal(TypeKind.LONG);
        int resultAddress = code.allocateLocal(TypeKind.LONG);
        code.lload(3).lstore(index);
        var done = code.newLabel();
        var loop = code.newLabel();
        boolean denseResult = ir.values().stream()
                .filter(candidate -> candidate.kind() == CpuKernelIr.Value.Kind.OUTPUT)
                .allMatch(candidate -> candidate.accessPlan().regime()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Regime
                                .DENSE_LINEAR);
        if (denseResult) {
            code.lload(index).lload(5).lcmp().branch(Opcode.IFGE, done);
            code.aload(2).lload(index).loadConstant(2L).lmul().loadConstant(1L).ladd()
                    .l2i().laload().lstore(resultAddress);
            code.labelBinding(loop);
            code.aload(2).lload(index).loadConstant(2L).lmul().l2i().laload()
                    .lstore(sourceAddress);
            emitDirectLoad(code, type, specialization.carrierPattern().get(0), 0,
                    sourceAddress);
            switch (type) {
                case FLOAT64 -> code.dstore(value);
                case FLOAT32 -> code.fstore(value);
                case BFLOAT16, INT32, BOOL -> code.istore(value);
                case INT64 -> code.lstore(value);
            }
            emitDirectStore(code, type, specialization.carrierPattern().get(1), 1,
                    resultAddress, value);
            code.lload(index).loadConstant(1L).ladd().lstore(index);
            code.lload(resultAddress).loadConstant(1L).ladd().lstore(resultAddress);
            code.lload(index).lload(5).lcmp().branch(Opcode.IFLT, loop);
            code.labelBinding(done);
            return;
        }
        var carriers = new CpuCarrierEmitter(code);
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

    private static void emitDirectStore(CodeBuilder code, DataType type,
            CpuKernelSpecialization.CarrierAccess access, int parameterSlot,
            int resultAddress, int value) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot).lload(resultAddress).l2i();
            switch (type) {
                case FLOAT64 -> code.dload(value).dastore();
                case FLOAT32 -> code.fload(value).fastore();
                case BFLOAT16 -> code.iload(value).sastore();
                case INT32 -> code.iload(value).iastore();
                case INT64 -> code.lload(value).lastore();
                case BOOL -> code.iload(value).bastore();
            }
            return;
        }
        ClassDesc layout = segmentLayout(type);
        String field = segmentLayoutField(type);
        ClassDesc primitive = primitive(type);
        code.aload(parameterSlot).getstatic(VALUE_LAYOUT, field, layout)
                .lload(resultAddress).loadConstant((long) type.byteWidth()).lmul();
        switch (type) {
            case FLOAT64 -> code.dload(value);
            case FLOAT32 -> code.fload(value);
            case BFLOAT16, INT32, BOOL -> code.iload(value);
            case INT64 -> code.lload(value);
        }
        code.invokeinterface(SEGMENT, "set", MethodTypeDesc.of(TypeKind.VOID.upperBound(),
                layout, TypeKind.LONG.upperBound(), primitive));
    }

    private static void emitDirectLoad(CodeBuilder code, DataType type,
            CpuKernelSpecialization.CarrierAccess access, int parameterSlot,
            int sourceAddress) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            code.aload(parameterSlot).lload(sourceAddress).l2i();
            switch (type) {
                case FLOAT64 -> code.daload();
                case FLOAT32 -> code.faload();
                case BFLOAT16 -> code.saload();
                case INT32 -> code.iaload();
                case INT64 -> code.laload();
                case BOOL -> code.baload();
            }
            return;
        }
        ClassDesc layout = segmentLayout(type);
        code.aload(parameterSlot).getstatic(VALUE_LAYOUT, segmentLayoutField(type), layout)
                .lload(sourceAddress).loadConstant((long) type.byteWidth()).lmul()
                .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layout,
                        TypeKind.LONG.upperBound()));
    }

    private static boolean directCarrier(DataType type,
            CpuKernelSpecialization.CarrierAccess access) {
        return access == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                || access == switch (type) {
                    case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
                    case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
                    case BFLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
                    case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
                    case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
                    case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
                };
    }

    private static ClassDesc segmentLayout(DataType type) {
        return switch (type) {
            case FLOAT64 -> DOUBLE_LAYOUT;
            case FLOAT32 -> FLOAT_LAYOUT;
            case BFLOAT16 -> SHORT_LAYOUT;
            case INT32 -> INT_LAYOUT;
            case INT64 -> LONG_LAYOUT;
            case BOOL -> BYTE_LAYOUT;
        };
    }

    private static String segmentLayoutField(DataType type) {
        return switch (type) {
            case FLOAT64 -> "JAVA_DOUBLE_UNALIGNED";
            case FLOAT32 -> "JAVA_FLOAT_UNALIGNED";
            case BFLOAT16 -> "JAVA_SHORT_UNALIGNED";
            case INT32 -> "JAVA_INT_UNALIGNED";
            case INT64 -> "JAVA_LONG_UNALIGNED";
            case BOOL -> "JAVA_BYTE";
        };
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE.upperBound();
            case FLOAT32 -> TypeKind.FLOAT.upperBound();
            case BFLOAT16 -> TypeKind.SHORT.upperBound();
            case INT32 -> TypeKind.INT.upperBound();
            case INT64 -> TypeKind.LONG.upperBound();
            case BOOL -> TypeKind.BYTE.upperBound();
        };
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

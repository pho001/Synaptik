package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;

/**
 * Emits a direct typed output-cell arg-extrema loop into one generated entry.
 *
 * <p>Each output ordinal is decoded once into non-selected logical coordinates. The generated
 * selected-axis loop then advances one affine input address, compares represented primitive
 * values, retains or replaces the logical coordinate according to the fixed tie policy, and
 * performs exactly one INT64 result store. No selected domain is split and no helper owned by
 * Synaptik is called from generated code.</p>
 */
public final class CpuArgExtremaEmitter {
    private static final ClassDesc FLOAT = ClassDesc.of(Float.class.getName());
    private static final ClassDesc DOUBLE = ClassDesc.of(Double.class.getName());
    private static final ClassDesc SEGMENT = ClassDesc.of("java.lang.foreign.MemorySegment");
    private static final ClassDesc VALUE_LAYOUT = ClassDesc.of("java.lang.foreign.ValueLayout");

    /** Creates a stateless generation-time emitter. */
    public CpuArgExtremaEmitter() { }

    /**
     * Emits one direct typed two-boundary, workspace-free body.
     *
     * @param code non-null Class-File method builder mutated during generation only
     * @param specialization non-null exact scalar carrier specialization
     * @param ir non-null canonical arg-extrema identity
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if boundary, result-type, or scratch facts disagree
     */
    public void emit(CodeBuilder code, CpuKernelSpecialization specialization, CpuKernelIr ir) {
        if (specialization.carrierPattern().size() != 2 || specialization.scratchParameter()
                || specialization.boundaryDataTypes().getLast() != DataType.INT64) {
            throw new IllegalArgumentException("arg-extrema requires numeric input and INT64 output");
        }
        DataType type = specialization.boundaryDataTypes().getFirst();
        if (type == DataType.BOOL) throw new IllegalArgumentException(
                "arg-extrema requires a numeric input");
        String identity = ir.familyIdentity();
        boolean maximum = identity.startsWith("arg-extrema:ARG_MAX:");
        boolean keep = identity.contains(":keep=true:");
        boolean last = identity.endsWith(":tie=LAST_INDEX");
        boolean narrowIndex = identity.contains(":narrow-index=true:");
        boolean narrowOutput = identity.contains(":narrow-output=true:");
        int axis = integerAfter(identity, ":axis=");
        int inputRank = ir.values().getFirst().accessPlan().iterationRank();
        int outputRank = ir.values().getLast().accessPlan().iterationRank();
        boolean unitAxisStride = axis == inputRank - 1
                && ir.values().getFirst().accessPlan().axisRoles().get(axis)
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.AxisRole
                                .CONTIGUOUS;
        emitGeneral(code, specialization, type, maximum, keep, last, narrowIndex, narrowOutput, axis,
                inputRank, outputRank, unitAxisStride);
    }

    private static void emitGeneral(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, boolean maximum, boolean keep, boolean last, boolean narrowIndex,
            boolean narrowOutput, int axis, int inputRank, int outputRank,
            boolean unitAxisStride) {
        int inputLayout = 8;
        int inputBaseIndex = inputLayout + 1;
        int inputStrides = inputLayout + 2 + inputRank;
        int outputLayout = inputLayout + 2 + 2 * inputRank;
        int outputBaseIndex = outputLayout + 1;
        int outputExtents = outputLayout + 2;
        int outputStrides = outputExtents + outputRank;
        boolean heapInput = specialization.carrierPattern().getFirst()
                != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        int axisExtent = code.allocateLocal(narrowIndex ? TypeKind.INT : TypeKind.LONG);
        int outputEnd = code.allocateLocal(narrowOutput ? TypeKind.INT : TypeKind.LONG);
        int axisStride = code.allocateLocal(heapInput ? TypeKind.INT : TypeKind.LONG);
        var carriers = new CpuCarrierEmitter(code);
        var done = code.newLabel();
        code.aload(2).loadConstant(7).laload();
        if (narrowIndex) code.l2i().istore(axisExtent); else code.lstore(axisExtent);
        code.lload(5);
        if (narrowOutput) code.l2i().istore(outputEnd); else code.lstore(outputEnd);
        code.aload(2).loadConstant(inputStrides + axis).laload();
        if (heapInput) code.l2i().istore(axisStride); else code.lstore(axisStride);
        code.lload(3).lload(5).lcmp().branch(Opcode.IFGE, done);
        if (!unitAxisStride && (type == DataType.BFLOAT16 || type == DataType.INT32)) {
            var generalStride = code.newLabel();
            if (heapInput) {
                code.iload(axisStride).loadConstant(2).branch(Opcode.IF_ICMPNE, generalStride);
            } else {
                code.lload(axisStride).loadConstant(2L).lcmp().branch(Opcode.IFNE, generalStride);
            }
            emitCells(code, specialization, type, maximum, keep, last, axis, inputRank,
                    outputRank, inputBaseIndex, inputStrides, outputBaseIndex, outputExtents,
                    outputStrides, heapInput, narrowIndex, narrowOutput, axisExtent, outputEnd,
                    axisStride, 2, carriers, done);
            code.labelBinding(generalStride);
            emitCells(code, specialization, type, maximum, keep, last, axis, inputRank,
                    outputRank, inputBaseIndex, inputStrides, outputBaseIndex, outputExtents,
                    outputStrides, heapInput, narrowIndex, narrowOutput, axisExtent, outputEnd,
                    axisStride, 0, carriers, done);
        } else {
            emitCells(code, specialization, type, maximum, keep, last, axis, inputRank,
                    outputRank, inputBaseIndex, inputStrides, outputBaseIndex, outputExtents,
                    outputStrides, heapInput, narrowIndex, narrowOutput, axisExtent, outputEnd,
                    axisStride, unitAxisStride ? 1 : 0, carriers, done);
        }
        code.labelBinding(done);
    }

    private static void emitCells(CodeBuilder code, CpuKernelSpecialization specialization,
            DataType type, boolean maximum, boolean keep, boolean last, int axis, int inputRank,
            int outputRank, int inputBaseIndex, int inputStrides, int outputBaseIndex,
            int outputExtents, int outputStrides, boolean heapInput, boolean narrowIndex,
            boolean narrowOutput, int axisExtent, int outputEnd, int axisStride,
            int fixedAxisStride, CpuCarrierEmitter carriers, Label done) {
        int cell = code.allocateLocal(narrowOutput ? TypeKind.INT : TypeKind.LONG);
        int remaining = code.allocateLocal(TypeKind.LONG);
        int inputBase = code.allocateLocal(TypeKind.LONG);
        int outputAddress = code.allocateLocal(TypeKind.LONG);
        int coordinate = code.allocateLocal(TypeKind.LONG);
        int candidate = code.allocateLocal(narrowIndex ? TypeKind.INT : TypeKind.LONG);
        int inputAddress = code.allocateLocal(heapInput ? TypeKind.INT : TypeKind.LONG);
        int bestIndex = code.allocateLocal(narrowIndex ? TypeKind.INT : TypeKind.LONG);
        int resultIndex = code.allocateLocal(TypeKind.LONG);
        int bestValue = code.allocateLocal(localKind(type));
        int candidateValue = code.allocateLocal(localKind(type));
        var cells = code.newLabel();
        code.lload(3);
        if (narrowOutput) code.l2i().istore(cell); else code.lstore(cell);
        code.labelBinding(cells);
        code.aload(2).loadConstant(inputBaseIndex).laload().lstore(inputBase);
        code.aload(2).loadConstant(outputBaseIndex).laload().lstore(outputAddress);
        if (inputRank == 2 && axis == 1 && (outputRank == 1 || keep && outputRank == 2)) {
            code.lload(inputBase);
            loadLongIndex(code, cell, narrowOutput);
            code.aload(2).loadConstant(inputStrides).laload()
                    .lmul().ladd().lstore(inputBase);
            code.lload(outputAddress);
            loadLongIndex(code, cell, narrowOutput);
            code.aload(2).loadConstant(outputStrides).laload()
                    .lmul().ladd().lstore(outputAddress);
        } else {
            loadLongIndex(code, cell, narrowOutput);
            code.lstore(remaining);
            for (int outAxis = outputRank - 1; outAxis >= 0; outAxis--) {
                code.lload(remaining).aload(2).loadConstant(outputExtents + outAxis).laload()
                        .lrem().lstore(coordinate);
                code.lload(remaining).aload(2).loadConstant(outputExtents + outAxis).laload()
                        .ldiv().lstore(remaining);
                int inputAxis = keep ? outAxis : outAxis < axis ? outAxis : outAxis + 1;
                if (inputAxis != axis) code.lload(inputBase).lload(coordinate).aload(2)
                        .loadConstant(inputStrides + inputAxis).laload().lmul().ladd()
                        .lstore(inputBase);
                code.lload(outputAddress).lload(coordinate).aload(2)
                        .loadConstant(outputStrides + outAxis).laload().lmul().ladd()
                        .lstore(outputAddress);
            }
        }
        if (narrowIndex) code.loadConstant(0).istore(bestIndex);
        else code.loadConstant(0L).lstore(bestIndex);
        code.lload(inputBase);
        if (heapInput) code.l2i().istore(inputAddress); else code.lstore(inputAddress);
        loadCarrier(code, carriers, type, specialization.carrierPattern().getFirst(), 0,
                inputAddress, heapInput);
        normalizeLoaded(code, type);
        store(code, type, bestValue);
        if (narrowIndex) code.loadConstant(1).istore(candidate);
        else code.loadConstant(1L).lstore(candidate);
        var finishDomain = code.newLabel();
        if (narrowIndex) code.iload(candidate).iload(axisExtent)
                .branch(Opcode.IF_ICMPGE, finishDomain);
        else code.lload(candidate).lload(axisExtent).lcmp().branch(Opcode.IFGE, finishDomain);
        var domain = code.newLabel();
        code.labelBinding(domain);
        if (heapInput) {
            code.iload(inputAddress);
            if (fixedAxisStride != 0) code.loadConstant(fixedAxisStride);
            else code.iload(axisStride);
            code.iadd().istore(inputAddress);
        } else {
            code.lload(inputAddress);
            if (fixedAxisStride != 0) code.loadConstant((long) fixedAxisStride);
            else code.lload(axisStride);
            code.ladd().lstore(inputAddress);
        }
        loadCarrier(code, carriers, type, specialization.carrierPattern().getFirst(), 0,
                inputAddress, heapInput);
        normalizeLoaded(code, type);
        store(code, type, candidateValue);
        Label update = code.newLabel();
        Label noUpdate = code.newLabel();
        emitSelection(code, type, maximum, last, candidateValue, bestValue, update, noUpdate);
        code.labelBinding(update);
        load(code, type, candidateValue);
        store(code, type, bestValue);
        if (narrowIndex) code.iload(candidate).istore(bestIndex);
        else code.lload(candidate).lstore(bestIndex);
        code.labelBinding(noUpdate);
        if (narrowIndex) {
            code.iinc(candidate, 1).iload(candidate).iload(axisExtent)
                    .branch(Opcode.IF_ICMPLT, domain);
        } else {
            code.lload(candidate).loadConstant(1L).ladd().lstore(candidate);
            code.lload(candidate).lload(axisExtent).lcmp().branch(Opcode.IFLT, domain);
        }
        code.labelBinding(finishDomain);
        if (narrowIndex) code.iload(bestIndex).i2l().lstore(resultIndex);
        else code.lload(bestIndex).lstore(resultIndex);
        storeCarrier(code, carriers, specialization.carrierPattern().getLast(), 1,
                outputAddress, resultIndex);
        if (narrowOutput) {
            code.iinc(cell, 1).iload(cell).iload(outputEnd).branch(Opcode.IF_ICMPLT, cells);
        } else {
            code.lload(cell).loadConstant(1L).ladd().lstore(cell);
            code.lload(cell).lload(outputEnd).lcmp().branch(Opcode.IFLT, cells);
        }
        code.branch(Opcode.GOTO, done);
    }

    private static void loadLongIndex(CodeBuilder code, int local, boolean narrow) {
        if (narrow) code.iload(local).i2l(); else code.lload(local);
    }

    private static void emitSelection(CodeBuilder code, DataType type, boolean maximum,
            boolean last, int candidate, int best, Label update, Label noUpdate) {
        if (type == DataType.INT32) {
            code.iload(candidate).iload(best).branch(maximum
                    ? last ? Opcode.IF_ICMPLT : Opcode.IF_ICMPLE
                    : last ? Opcode.IF_ICMPGT : Opcode.IF_ICMPGE, noUpdate);
            return;
        }
        if (type == DataType.INT64) {
            code.lload(candidate).lload(best).lcmp().branch(maximum
                    ? last ? Opcode.IFLT : Opcode.IFLE
                    : last ? Opcode.IFGT : Opcode.IFGE, noUpdate);
            return;
        }
        if (type == DataType.FLOAT64) {
            emitDoubleSelection(code, maximum, last, candidate, best, update, noUpdate);
        } else {
            emitFloatSelection(code, maximum, last, candidate, best, update, noUpdate);
        }
    }

    private static void emitFloatSelection(CodeBuilder code, boolean maximum, boolean last,
            int candidate, int best, Label update, Label noUpdate) {
        int candidateBits = code.allocateLocal(TypeKind.INT);
        int bestBits = code.allocateLocal(TypeKind.INT);
        int comparison = code.allocateLocal(TypeKind.INT);
        load(code, DataType.FLOAT32, candidate);
        code.invokestatic(FLOAT, "floatToRawIntBits",
                MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float))
                .istore(candidateBits);
        load(code, DataType.FLOAT32, best);
        code.invokestatic(FLOAT, "floatToRawIntBits",
                MethodTypeDesc.of(ConstantDescs.CD_int, ConstantDescs.CD_float)).istore(bestBits);
        Label candidateNumeric = code.newLabel();
        Label compareNumeric = code.newLabel();
        code.iload(candidateBits).loadConstant(0x7f800000).iand()
                .loadConstant(0x7f800000).branch(Opcode.IF_ICMPNE, candidateNumeric);
        code.iload(candidateBits).loadConstant(0x007fffff).iand()
                .branch(Opcode.IFEQ, candidateNumeric);
        if (last) {
            code.branch(Opcode.GOTO, update);
            code.labelBinding(candidateNumeric);
            code.iload(bestBits).loadConstant(0x7f800000).iand()
                    .loadConstant(0x7f800000).branch(Opcode.IF_ICMPNE, compareNumeric);
            code.iload(bestBits).loadConstant(0x007fffff).iand()
                    .branch(Opcode.IFNE, noUpdate);
            code.labelBinding(compareNumeric);
            emitFloatNumericSelection(code, maximum, true, candidate, best, candidateBits,
                    bestBits, comparison, update, noUpdate);
            return;
        }
        code.iload(bestBits).loadConstant(0x7f800000).iand()
                .loadConstant(0x7f800000).branch(Opcode.IF_ICMPNE, update);
        code.iload(bestBits).loadConstant(0x007fffff).iand().branch(Opcode.IFEQ, update);
        code.branch(Opcode.GOTO, last ? update : noUpdate);
        code.labelBinding(candidateNumeric);
        code.iload(bestBits).loadConstant(0x7f800000).iand()
                .loadConstant(0x7f800000).branch(Opcode.IF_ICMPNE, compareNumeric);
        code.iload(bestBits).loadConstant(0x007fffff).iand().branch(Opcode.IFEQ, compareNumeric);
        code.branch(Opcode.GOTO, noUpdate);
        code.labelBinding(compareNumeric);
        emitFloatNumericSelection(code, maximum, false, candidate, best, candidateBits, bestBits,
                comparison, update, noUpdate);
    }

    private static void emitFloatNumericSelection(CodeBuilder code, boolean maximum, boolean last,
            int candidate, int best, int candidateBits, int bestBits, int comparison,
            Label update, Label noUpdate) {
        code.fload(candidate).fload(best).fcmpl().istore(comparison);
        code.iload(comparison).branch(maximum ? Opcode.IFGT : Opcode.IFLT, update);
        code.iload(comparison).branch(Opcode.IFNE, noUpdate);
        code.iload(candidateBits).iload(bestBits).branch(Opcode.IF_ICMPEQ,
                last ? update : noUpdate);
        code.iload(candidateBits).branch(maximum ? Opcode.IFLT : Opcode.IFGE, noUpdate);
        code.branch(Opcode.GOTO, update);
    }

    private static void emitDoubleSelection(CodeBuilder code, boolean maximum, boolean last,
            int candidate, int best, Label update, Label noUpdate) {
        int candidateBits = code.allocateLocal(TypeKind.LONG);
        int bestBits = code.allocateLocal(TypeKind.LONG);
        int comparison = code.allocateLocal(TypeKind.INT);
        code.dload(candidate).invokestatic(DOUBLE, "doubleToRawLongBits",
                MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double))
                .lstore(candidateBits);
        code.dload(best).invokestatic(DOUBLE, "doubleToRawLongBits",
                MethodTypeDesc.of(ConstantDescs.CD_long, ConstantDescs.CD_double))
                .lstore(bestBits);
        Label candidateNumeric = code.newLabel();
        Label compareNumeric = code.newLabel();
        code.lload(candidateBits).loadConstant(0x7ff0000000000000L).land()
                .loadConstant(0x7ff0000000000000L).lcmp().branch(Opcode.IFNE, candidateNumeric);
        code.lload(candidateBits).loadConstant(0x000fffffffffffffL).land()
                .loadConstant(0L).lcmp().branch(Opcode.IFEQ, candidateNumeric);
        if (last) {
            code.branch(Opcode.GOTO, update);
            code.labelBinding(candidateNumeric);
            code.lload(bestBits).loadConstant(0x7ff0000000000000L).land()
                    .loadConstant(0x7ff0000000000000L).lcmp()
                    .branch(Opcode.IFNE, compareNumeric);
            code.lload(bestBits).loadConstant(0x000fffffffffffffL).land()
                    .loadConstant(0L).lcmp().branch(Opcode.IFNE, noUpdate);
            code.labelBinding(compareNumeric);
            emitDoubleNumericSelection(code, maximum, true, candidate, best, candidateBits,
                    bestBits, comparison, update, noUpdate);
            return;
        }
        code.lload(bestBits).loadConstant(0x7ff0000000000000L).land()
                .loadConstant(0x7ff0000000000000L).lcmp().branch(Opcode.IFNE, update);
        code.lload(bestBits).loadConstant(0x000fffffffffffffL).land()
                .loadConstant(0L).lcmp().branch(Opcode.IFEQ, update);
        code.branch(Opcode.GOTO, last ? update : noUpdate);
        code.labelBinding(candidateNumeric);
        code.lload(bestBits).loadConstant(0x7ff0000000000000L).land()
                .loadConstant(0x7ff0000000000000L).lcmp().branch(Opcode.IFNE, compareNumeric);
        code.lload(bestBits).loadConstant(0x000fffffffffffffL).land()
                .loadConstant(0L).lcmp().branch(Opcode.IFEQ, compareNumeric);
        code.branch(Opcode.GOTO, noUpdate);
        code.labelBinding(compareNumeric);
        emitDoubleNumericSelection(code, maximum, false, candidate, best, candidateBits, bestBits,
                comparison, update, noUpdate);
    }

    private static void emitDoubleNumericSelection(CodeBuilder code, boolean maximum,
            boolean last, int candidate, int best, int candidateBits, int bestBits,
            int comparison, Label update, Label noUpdate) {
        code.dload(candidate).dload(best).dcmpl().istore(comparison);
        code.iload(comparison).branch(maximum ? Opcode.IFGT : Opcode.IFLT, update);
        code.iload(comparison).branch(Opcode.IFNE, noUpdate);
        code.lload(candidateBits).lload(bestBits).lcmp().branch(Opcode.IFEQ,
                last ? update : noUpdate);
        code.lload(candidateBits).loadConstant(0L).lcmp()
                .branch(maximum ? Opcode.IFLT : Opcode.IFGE, noUpdate);
        code.branch(Opcode.GOTO, update);
    }

    private static void normalizeLoaded(CodeBuilder code, DataType type) {
        if (type != DataType.BFLOAT16) return;
        code.loadConstant(0xffff).iand().loadConstant(16).ishl()
                .invokestatic(FLOAT, "intBitsToFloat",
                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int));
    }

    private static void loadCarrier(CodeBuilder code, CpuCarrierEmitter carriers, DataType type,
            CpuKernelSpecialization.CarrierAccess access, int parameter, int address,
            boolean intAddress) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            carriers.load(type, access, parameter, address, intAddress);
            return;
        }
        ClassDesc layout = layoutClass(type);
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(type), layout)
                .lload(address).loadConstant((long) type.byteWidth()).lmul()
                .invokeinterface(SEGMENT, "get", MethodTypeDesc.of(primitive(type), layout,
                        ConstantDescs.CD_long));
    }

    private static void storeCarrier(CodeBuilder code, CpuCarrierEmitter carriers,
            CpuKernelSpecialization.CarrierAccess access, int parameter, int address, int value) {
        if (access != CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT) {
            carriers.store(DataType.INT64, access, parameter, address, value);
            return;
        }
        ClassDesc layout = layoutClass(DataType.INT64);
        code.aload(parameter).getstatic(VALUE_LAYOUT, layoutField(DataType.INT64), layout)
                .lload(address).loadConstant(8L).lmul().lload(value)
                .invokeinterface(SEGMENT, "set", MethodTypeDesc.of(ConstantDescs.CD_void,
                        layout, ConstantDescs.CD_long, ConstantDescs.CD_long));
    }

    private static int integerAfter(String value, String marker) {
        int start = value.indexOf(marker) + marker.length();
        int end = value.indexOf(':', start);
        return Integer.parseInt(value.substring(start, end));
    }

    private static ClassDesc primitive(DataType type) {
        return switch (type) {
            case FLOAT64 -> ConstantDescs.CD_double;
            case FLOAT32 -> ConstantDescs.CD_float;
            case BFLOAT16 -> ConstantDescs.CD_short;
            case INT32 -> ConstantDescs.CD_int;
            case INT64 -> ConstantDescs.CD_long;
            case BOOL -> throw new AssertionError();
        };
    }

    private static ClassDesc layoutClass(DataType type) {
        return ClassDesc.of("java.lang.foreign.ValueLayout$Of" + switch (type) {
            case FLOAT64 -> "Double";
            case FLOAT32 -> "Float";
            case BFLOAT16 -> "Short";
            case INT32 -> "Int";
            case INT64 -> "Long";
            case BOOL -> throw new AssertionError();
        });
    }

    private static String layoutField(DataType type) {
        return "JAVA_" + switch (type) {
            case FLOAT64 -> "DOUBLE";
            case FLOAT32 -> "FLOAT";
            case BFLOAT16 -> "SHORT";
            case INT32 -> "INT";
            case INT64 -> "LONG";
            case BOOL -> throw new AssertionError();
        } + "_UNALIGNED";
    }

    private static TypeKind localKind(DataType type) {
        return switch (type) {
            case FLOAT64 -> TypeKind.DOUBLE;
            case FLOAT32, BFLOAT16 -> TypeKind.FLOAT;
            case INT32 -> TypeKind.INT;
            case INT64 -> TypeKind.LONG;
            case BOOL -> throw new AssertionError();
        };
    }

    private static void store(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dstore(local);
            case FLOAT32, BFLOAT16 -> code.fstore(local);
            case INT32 -> code.istore(local);
            case INT64 -> code.lstore(local);
            case BOOL -> throw new AssertionError();
        }
    }

    private static void load(CodeBuilder code, DataType type, int local) {
        switch (type) {
            case FLOAT64 -> code.dload(local);
            case FLOAT32, BFLOAT16 -> code.fload(local);
            case INT32 -> code.iload(local);
            case INT64 -> code.lload(local);
            case BOOL -> throw new AssertionError();
        }
    }
}
